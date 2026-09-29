"""Task manager for async analysis pipeline.

Provides an in-memory task manager with deduplication logic
to prevent redundant analysis tasks with the same parameters,
and cancellation of queued and running tasks.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4

from tradingagents.api.schemas.task import (
    FINISHED_STATUSES,
    TaskCreateRequest,
    TaskResponse,
    TaskStatus,
)

logger = logging.getLogger(__name__)

CANCELLED_BEFORE_START = "Cancelled before it started."
CANCELLED_WHILE_RUNNING = "Cancelled while running; stopped at its next LLM call."


class CancelOutcome(str, Enum):
    """What :meth:`TaskManager.cancel_task` did."""

    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"  # pending/queued: cancelled at once, never runs
    REQUESTED = "requested"  # processing: stops at its next LLM call
    FINISHED = "finished"  # already completed, failed or cancelled


class TaskManager:
    """In-memory task manager with deduplication and TTL cleanup.

    Thread-safe: worker threads mutate tasks while request handlers read them,
    so every access to ``_tasks`` is guarded by ``_lock``.
    """

    def __init__(self, ttl_minutes: int = 60, max_tasks: int = 100):
        """Initialize the task manager.

        Args:
            ttl_minutes: Time-to-live for completed/failed tasks in minutes.
            max_tasks: Maximum number of tasks to keep in memory.
        """
        self._tasks: dict[str, TaskResponse] = {}
        self._ttl = ttl_minutes
        self._max_tasks = max_tasks
        # Per task: the event that cancels it (checked when it starts, and at
        # every LLM call while it runs) and the worker Future while it waits.
        self._cancel_events: dict[str, threading.Event] = {}
        self._futures: dict[str, Future] = {}
        # Processing tasks deleted while their run is still stopping: hidden
        # from every lookup, but kept (and counted as active and processing)
        # until the worker records the end, so the busy counts stay true.
        self._deleted: set[str] = set()
        # Reentrant: create_task() calls find_active_task() and _cleanup_expired().
        self._lock = threading.RLock()

    def find_active_task(
        self, ticker: str, trade_date: str, asset_type: str
    ) -> TaskResponse | None:
        """Find an existing task with the same parameters that is still running.

        Args:
            ticker: Ticker symbol.
            trade_date: Trade date in YYYY-MM-DD format.
            asset_type: Asset type ('stock' or 'crypto').

        Returns:
            Existing TaskResponse if found, None otherwise.
        """
        with self._lock:
            for task in self._tasks.values():
                if (
                    task.task_id not in self._deleted
                    and task.ticker == ticker
                    and task.trade_date == trade_date
                    and task.asset_type == asset_type
                    and task.status
                    in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.PROCESSING)
                    # A task being cancelled is not reused: a new request
                    # gets a fresh task instead of one about to end.
                    and not task.cancel_requested
                ):
                    logger.info(
                        "Found existing active task %s for %s/%s/%s",
                        task.task_id,
                        ticker,
                        trade_date,
                        asset_type,
                    )
                    return task
            return None

    def create_task(self, request: TaskCreateRequest) -> tuple[TaskResponse, bool]:
        """Create a new analysis task or return existing one if duplicate.

        Args:
            request: Task creation request.

        Returns:
            Tuple of (TaskResponse, is_new).
            is_new is True if a new task was created, False if an existing
            task was returned (deduplication).
        """
        with self._lock:
            # Check for duplicate active task
            existing = self.find_active_task(
                request.ticker, request.trade_date, request.asset_type
            )
            if existing is not None:
                return existing, False

            # Cleanup expired tasks
            self._cleanup_expired()

            # Check capacity
            if len(self._tasks) >= self._max_tasks:
                raise ValueError(
                    f"Task queue is full (max_tasks={self._max_tasks}). "
                    "Wait for some tasks to complete or delete finished tasks."
                )

            task_id = str(uuid4())
            now = datetime.now()
            task = TaskResponse(
                task_id=task_id,
                status=TaskStatus.PENDING,
                ticker=request.ticker,
                trade_date=request.trade_date,
                asset_type=request.asset_type,
                created_at=now,
                updated_at=now,
            )
            self._tasks[task_id] = task
            self._cancel_events[task_id] = threading.Event()
            logger.info("Created new task %s for %s/%s/%s", task_id, request.ticker, request.trade_date, request.asset_type)
            return task, True

    def get_task(self, task_id: str) -> TaskResponse | None:
        """Get a task by ID.

        Args:
            task_id: Task UUID.

        Returns:
            TaskResponse if found, None otherwise.
        """
        with self._lock:
            if task_id in self._deleted:
                return None
            return self._tasks.get(task_id)

    def list_tasks(
        self, status: TaskStatus | None = None, ticker: str | None = None
    ) -> list[TaskResponse]:
        """List tasks with optional filters.

        Args:
            status: Filter by task status.
            ticker: Filter by ticker symbol.

        Returns:
            List of matching TaskResponse objects.
        """
        with self._lock:
            tasks = [t for t in self._tasks.values() if t.task_id not in self._deleted]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if ticker is not None:
            tasks = [t for t in tasks if t.ticker == ticker]
        # Sort by created_at descending (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def update_task(self, task_id: str, **kwargs) -> bool:
        """Update task fields.

        Args:
            task_id: Task UUID.
            **kwargs: Fields to update.

        Returns:
            True if task was found and updated, False otherwise.
        """
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

            task.updated_at = datetime.now()

            status = kwargs.get("status")
            if status in FINISHED_STATUSES:
                task.completed_at = datetime.now()
                # A cancelled task's one INFO line comes from whoever cancelled
                # or stopped it; this would only repeat it.
                logger.log(
                    logging.DEBUG if status == TaskStatus.CANCELLED else logging.INFO,
                    "Task %s completed with status: %s",
                    task_id,
                    status,
                )
                if task_id in self._deleted:
                    # Deleted while it ran: its worker has now stopped.
                    self._forget(task_id)

            return True

    def attach_future(self, task_id: str, future: Future) -> None:
        """Keep the worker's Future of a queued task, so cancelling can drop it.

        A task cancelled before its Future arrived gets it cancelled at once.
        """
        with self._lock:
            event = self._cancel_events.get(task_id)
            if task_id not in self._tasks or event is None:
                return
            if event.is_set():
                future.cancel()
                return
            self._futures[task_id] = future
        future.add_done_callback(lambda done: self._forget_future(task_id, done))

    def _forget_future(self, task_id: str, future: Future) -> None:
        with self._lock:
            if self._futures.get(task_id) is future:
                del self._futures[task_id]

    def start_task(self, task_id: str) -> threading.Event | None:
        """Mark a task processing as its worker picks it up.

        Returns:
            The task's cancel event, to hand to the run; None when the task was
            cancelled or deleted while it waited -- the run must be skipped.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            event = self._cancel_events.get(task_id)
            if task is None or event is None or event.is_set():
                return None
            if task.status in FINISHED_STATUSES:
                return None
            task.status = TaskStatus.PROCESSING
            task.updated_at = datetime.now()
            return event

    def cancel_task(self, task_id: str) -> CancelOutcome:
        """Cancel a task.

        A pending or queued task is cancelled at once: its Future is cancelled
        and, should a worker pick it up anyway, the cancel event stops it before
        it starts. A processing task gets ``cancel_requested`` and keeps its
        status until the run stops at its next LLM call, when the worker marks
        it cancelled.

        A cancel can race the run's end, so a task with ``cancel_requested``
        may still end ``completed`` or ``failed``: the run checks the cancel
        event last right after the graph returns (or raises), and a cancel that
        lands after that check, but before the worker records the outcome,
        cannot stop it any more.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task_id in self._deleted:
                return CancelOutcome.NOT_FOUND
            if task.status in FINISHED_STATUSES:
                return CancelOutcome.FINISHED
            now = datetime.now()
            event = self._cancel_events.setdefault(task_id, threading.Event())
            event.set()
            task.cancel_requested = True
            task.updated_at = now
            if task.status == TaskStatus.PROCESSING:
                # Its INFO line is logged when the run stops (TaskService).
                logger.debug("Task %s: cancel requested; it stops at its next LLM call", task_id)
                return CancelOutcome.REQUESTED
            future = self._futures.pop(task_id, None)
            if future is not None:
                future.cancel()
            task.status = TaskStatus.CANCELLED
            task.message = CANCELLED_BEFORE_START
            task.completed_at = now
            logger.info("Task %s cancelled before it started", task_id)
            return CancelOutcome.CANCELLED

    def delete_task(self, task_id: str) -> bool:
        """Delete a task, cancelling it first if it has not finished.

        The task is gone for every caller at once. A processing one, though,
        still occupies its worker until the run stops at its next LLM call, so
        it keeps counting in ``active_task_count`` and ``processing_count``
        (the symbols refresher reads them to tell when the app is idle) until
        the worker records its end.

        Args:
            task_id: Task UUID.

        Returns:
            True if task was deleted, False if not found.
        """
        with self._lock:
            if task_id not in self._tasks or task_id in self._deleted:
                return False

            if self._tasks[task_id].status not in FINISHED_STATUSES:
                logger.warning("Deleting active task %s; cancelling it", task_id)
                if self.cancel_task(task_id) == CancelOutcome.REQUESTED:
                    self._deleted.add(task_id)
                    return True

            self._forget(task_id)
            return True

    def _forget(self, task_id: str) -> None:
        """Drop a task's record, cancel event and Future (lock held)."""
        self._tasks.pop(task_id, None)
        self._cancel_events.pop(task_id, None)
        self._futures.pop(task_id, None)
        self._deleted.discard(task_id)

    def cleanup(self) -> int:
        """Remove expired completed/failed/cancelled tasks.

        Returns:
            Number of tasks removed.
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=self._ttl)

        with self._lock:
            to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in FINISHED_STATUSES:
                    if task.completed_at and task.completed_at < cutoff:
                        to_remove.append(task_id)
                # Only clean up stale pending tasks when TTL is enabled.
                # Avoids removing freshly created tasks when ttl=0 (test scenarios).
                elif task.status == TaskStatus.PENDING and self._ttl > 0 and task.created_at < cutoff:
                    to_remove.append(task_id)

            for task_id in to_remove:
                self._forget(task_id)

        if to_remove:
            logger.info("Cleaned up %d expired tasks", len(to_remove))

        return len(to_remove)

    def _cleanup_expired(self) -> None:
        """Internal: remove expired tasks to free capacity."""
        self.cleanup()

    @property
    def active_task_count(self) -> int:
        """Get the number of active (pending/processing/queued) tasks."""
        with self._lock:
            return sum(
                1
                for task in self._tasks.values()
                if task.status
                in (TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.QUEUED)
            )

    @property
    def processing_count(self) -> int:
        """Get the number of tasks currently occupying a worker slot."""
        with self._lock:
            return sum(
                1
                for task in self._tasks.values()
                if task.status == TaskStatus.PROCESSING
            )

    @property
    def total_task_count(self) -> int:
        """Get the total number of tasks in memory (deleted ones still stopping included)."""
        with self._lock:
            return len(self._tasks)
