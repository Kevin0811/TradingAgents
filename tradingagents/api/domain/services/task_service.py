"""Task service - orchestrates async analysis task execution."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.core.exceptions import AnalysisError
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.schemas.task import TaskCreateRequest, TaskStatus

logger = logging.getLogger(__name__)


class TaskService:
    """Service for managing and executing analysis tasks.

    Wraps the AnalysisService with task lifecycle management and
    concurrency control via a shared threading.Semaphore.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        analysis_service: AnalysisService,
        semaphore: Optional[threading.Semaphore] = None,
    ):
        """Initialize the task service.

        Args:
            task_manager: Task manager instance.
            analysis_service: Analysis service instance.
            semaphore: Shared threading.Semaphore for concurrency control.
                       If None, no concurrency limit is applied.
        """
        self._task_manager = task_manager
        self._analysis_service = analysis_service
        self._semaphore = semaphore

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
        """Execute analysis in background and update task status.

        This method is designed to be run as a FastAPI BackgroundTask.
        Uses a semaphore to limit concurrent task executions. Tasks wait
        for an available slot before starting execution.

        Args:
            task_id: Task UUID.
            request: Original task creation request.
        """
        if self._semaphore is not None:
            logger.info(
                "Task %s waiting for execution slot (concurrent limit active)",
                task_id,
            )
            self._semaphore.acquire()
            logger.info("Task %s acquired execution slot", task_id)

        try:
            self._run_analysis_with_status_updates(task_id, request)
        finally:
            if self._semaphore is not None:
                self._semaphore.release()
                logger.info("Task %s released execution slot", task_id)

    def _run_analysis_with_status_updates(
        self, task_id: str, request: TaskCreateRequest
    ) -> None:
        """Run analysis and update status. Called after acquiring semaphore."""
        try:
            # Update status to processing
            self._task_manager.update_task(task_id, status=TaskStatus.PROCESSING)
            logger.info("Task %s started processing", task_id)

            # Run the analysis
            result = self._analysis_service.run_analysis(
                ticker=request.ticker,
                trade_date=request.trade_date,
                asset_type=request.asset_type,
                selected_analysts=tuple(request.selected_analysts),
            )

            # Update task with result
            self._task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                result=result.to_dict(),
            )
            logger.info("Task %s completed successfully", task_id)

        except AnalysisError as e:
            logger.error("Task %s analysis failed: %s", task_id, e)
            self._task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e.detail) if e.detail else str(e),
            )

        except Exception as e:
            logger.exception("Task %s failed with unexpected error", task_id)
            self._task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )

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

    def delete_task(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task UUID.

        Returns:
            True if task was deleted, False if not found.
        """
        return self._task_manager.delete_task(task_id)