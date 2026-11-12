"""Task service - orchestrates async analysis task execution."""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.api.core.discussion_log import DiscussionLog, DiscussionStore
from tradingagents.api.core.exceptions import AnalysisCancelled, AnalysisError
from tradingagents.api.core.safe_errors import error_text, traceback_is_safe
from tradingagents.api.core.task_manager import (
    CANCELLED_WHILE_RUNNING,
    CancelOutcome,
    TaskManager,
)
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.config_overrides import resolve_overrides
from tradingagents.api.infrastructure.transcript_files import TranscriptFiles
from tradingagents.api.schemas.task import TaskCreateRequest, TaskStatus
from tradingagents.api.schemas.token_usage import token_usage_model

logger = logging.getLogger(__name__)


class TaskService:
    """Service for managing and executing analysis tasks.

    Wraps the AnalysisService with task lifecycle management. Concurrency is
    bounded by the TaskWorker's thread pool, not here.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        analysis_service: AnalysisService,
        discussions: DiscussionStore | None = None,
        transcripts: TranscriptFiles | None = None,
    ):
        """Initialize the task service.

        Args:
            task_manager: Task manager instance.
            analysis_service: Analysis service instance.
            discussions: The tasks' discussion logs; a task's run records its
                discussion in its log.
            transcripts: Where a finished run's transcript is written.
        """
        self._task_manager = task_manager
        self._analysis_service = analysis_service
        self._discussions = discussions
        self._transcripts = transcripts

    def create_task(
        self, request: TaskCreateRequest
    ) -> tuple[Any, bool]:
        """Create a new task or return existing one if duplicate.

        Args:
            request: Task creation request.

        Returns:
            Tuple of (TaskResponse, is_new).
        """
        return self._task_manager.create_task(request)

    def execute_analysis(self, task_id: str, request: TaskCreateRequest) -> None:
        """Execute analysis and update task status.

        Runs inside a TaskWorker thread. Tasks beyond the worker's concurrency
        limit wait in the executor's queue, so they occupy no thread while
        queued and never block request handling.

        Args:
            task_id: Task UUID.
            request: Original task creation request.
        """
        self._run_analysis_with_status_updates(task_id, request)

    def _run_analysis_with_status_updates(
        self, task_id: str, request: TaskCreateRequest
    ) -> None:
        """Run analysis and update the task record with the outcome.

        The run's token usage goes on the task however the run ends: from the
        result, or from the ``AnalysisCancelled``/``AnalysisError`` it raised.
        """
        # Backstop for a queued task whose Future could not be cancelled: a
        # task cancelled or deleted while it waited never starts.
        cancel_event = self._task_manager.start_task(task_id)
        if cancel_event is None:
            logger.info("Task %s was cancelled or deleted before it started; skipping", task_id)
            return
        log = self._discussions.get(task_id) if self._discussions is not None else None
        try:
            logger.info("Task %s started processing", task_id)

            # Run the analysis
            result = self._analysis_service.run_analysis(
                ticker=request.ticker,
                trade_date=request.trade_date,
                asset_type=request.asset_type,
                selected_analysts=tuple(request.selected_analysts),
                overrides=resolve_overrides(request),
                cancel_event=cancel_event,
                **({"events": log} if log is not None else {}),
            )

            # Update task with result
            self._task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                result=result.to_dict(),
                token_usage=token_usage_model(result.token_usage),
            )
            logger.info("Task %s completed successfully", task_id)

        except AnalysisCancelled as e:
            # The one INFO line of a task cancelled while it ran.
            logger.info("Task %s cancelled while running; stopped at an LLM call", task_id)
            self._task_manager.update_task(
                task_id,
                status=TaskStatus.CANCELLED,
                message=CANCELLED_WHILE_RUNNING,
                token_usage=token_usage_model(e.token_usage),
            )

        except AnalysisError as e:
            logger.error("Task %s analysis failed: %s", task_id, e)
            self._task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e.detail) if e.detail else str(e),
                token_usage=token_usage_model(e.token_usage),
            )

        except Exception as e:
            logger.error(
                "Task %s failed with unexpected error: %s", task_id, error_text(e),
                exc_info=traceback_is_safe(e),
            )
            self._task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=error_text(e),
            )

        self._save_transcript(task_id, request.ticker, log)

    def _save_transcript(self, task_id: str, ticker: str, log: DiscussionLog | None) -> None:
        """Write a finished run's transcript next to its results.

        Skipped for a task deleted meanwhile (its log is gone from the store)
        and for one whose run never started. Never raises: the task's own
        outcome is already recorded, and a transcript failure is only logged.
        """
        if log is None or self._transcripts is None or self._discussions is None:
            return
        try:
            if self._discussions.get(task_id) is not log or not log.finished:
                return
            transcript = log.transcript()
            if transcript["run_started"] is None:
                return
            self._transcripts.write(ticker, transcript)
        except Exception:  # noqa: BLE001 - the worker must never see this
            # The task id only: the error could quote the transcript's content.
            logger.warning("Could not save the discussion transcript of task %s", task_id)

    def get_task(self, task_id: str) -> Any:
        """Get task by ID.

        Args:
            task_id: Task UUID.

        Returns:
            TaskResponse if found, None otherwise.
        """
        return self._task_manager.get_task(task_id)

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        ticker: str | None = None,
    ) -> list[Any]:
        """List tasks with optional filters.

        Args:
            status: Filter by task status.
            ticker: Filter by ticker symbol.

        Returns:
            List of matching TaskResponse objects.
        """
        return self._task_manager.list_tasks(status=status, ticker=ticker)

    def cancel_task(self, task_id: str) -> CancelOutcome:
        """Cancel a task (see ``TaskManager.cancel_task``)."""
        return self._task_manager.cancel_task(task_id)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task, cancelling it first if it is still active.

        Args:
            task_id: Task UUID.

        Returns:
            True if task was deleted, False if not found.
        """
        return self._task_manager.delete_task(task_id)

