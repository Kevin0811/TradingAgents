"""Task manager for async analysis pipeline.

Provides an in-memory task manager with deduplication logic
to prevent redundant analysis tasks with the same parameters.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    """In-memory task manager with deduplication and TTL cleanup."""

    def __init__(self, ttl_minutes: int = 60, max_tasks: int = 100):
        """Initialize the task manager.

        Args:
            ttl_minutes: Time-to-live for completed/failed tasks in minutes.
            max_tasks: Maximum number of tasks to keep in memory.
        """
        self._tasks: dict[str, TaskResponse] = {}
        self._ttl = ttl_minutes
        self._max_tasks = max_tasks

    def find_active_task(
        self, ticker: str, trade_date: str, asset_type: str
    ) -> Optional[TaskResponse]:
        """Find an existing task with the same parameters that is still running.

        Args:
            ticker: Ticker symbol.
            trade_date: Trade date in YYYY-MM-DD format.
            asset_type: Asset type ('stock' or 'crypto').

        Returns:
            Existing TaskResponse if found, None otherwise.
        """
        for task in self._tasks.values():
            if (
                task.ticker == ticker
                and task.trade_date == trade_date
                and task.asset_type == asset_type
                and task.status in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.PROCESSING)
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
        logger.info("Created new task %s for %s/%s/%s", task_id, request.ticker, request.trade_date, request.asset_type)
        return task, True

    def get_task(self, task_id: str) -> Optional[TaskResponse]:
        """Get a task by ID.

        Args:
            task_id: Task UUID.

        Returns:
            TaskResponse if found, None otherwise.
        """
        return self._tasks.get(task_id)

    def list_tasks(
        self, status: Optional[TaskStatus] = None, ticker: Optional[str] = None
    ) -> list[TaskResponse]:
        """List tasks with optional filters.

        Args:
            status: Filter by task status.
            ticker: Filter by ticker symbol.

        Returns:
            List of matching TaskResponse objects.
        """
        tasks = list(self._tasks.values())
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
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        task.updated_at = datetime.now()

        if kwargs.get("status") in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            task.completed_at = datetime.now()
            logger.info(
                "Task %s completed with status: %s",
                task_id,
                kwargs.get("status"),
            )

        return True

    def delete_task(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task UUID.

        Returns:
            True if task was deleted, False if not found.
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        if task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
            logger.warning("Deleting active task %s", task_id)

        del self._tasks[task_id]
        return True

    def cleanup(self) -> int:
        """Remove expired completed/failed tasks.

        Returns:
            Number of tasks removed.
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=self._ttl)

        to_remove = []
        for task_id, task in self._tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                if task.completed_at and task.completed_at < cutoff:
                    to_remove.append(task_id)
            elif task.status == TaskStatus.PENDING and self._ttl > 0:
                # Only clean up stale pending tasks when TTL is enabled
                # Avoids removing freshly created tasks when ttl=0 (test scenarios)
                if task.created_at < cutoff:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]

        if to_remove:
            logger.info("Cleaned up %d expired tasks", len(to_remove))

        return len(to_remove)

    def _cleanup_expired(self) -> None:
        """Internal: remove expired tasks to free capacity."""
        self.cleanup()

    @property
    def active_task_count(self) -> int:
        """Get the number of active (pending/processing/queued) tasks."""
        return sum(
            1
            for task in self._tasks.values()
            if task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.QUEUED)
        )

    @property
    def processing_count(self) -> int:
        """Get the number of currently processing tasks (holding a semaphore slot)."""
        return sum(
            1
            for task in self._tasks.values()
            if task.status == TaskStatus.PROCESSING
        )

    @property
    def total_task_count(self) -> int:
        """Get the total number of tasks in memory."""
        return len(self._tasks)