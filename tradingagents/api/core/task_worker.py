"""Background worker pool for long-running analysis tasks.

Analysis runs take minutes. They must not execute on the event loop, and they
must not execute on the AnyIO threadpool that Starlette shares with sync
dependencies and sync endpoints -- a queued task blocking there starves request
handling for the whole process.

This worker owns a dedicated ``ThreadPoolExecutor`` sized to the configured
concurrency limit. Tasks beyond that limit wait in the executor's internal
queue, so waiting costs no thread at all.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class TaskWorker:
    """Bounded thread pool that runs analysis tasks off the request path."""

    def __init__(self, max_workers: int = 2):
        """Initialize the worker pool.

        Args:
            max_workers: Maximum number of concurrently executing tasks.
                Additional submissions queue inside the executor.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ta-task",
        )

    @property
    def max_workers(self) -> int:
        """Maximum number of concurrently executing tasks."""
        return self._max_workers

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Queue a callable for execution and return immediately.

        Args:
            fn: Callable to run in a worker thread.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            The Future for the submitted work.
        """
        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self._log_failure)
        return future

    @staticmethod
    def _log_failure(future: Future) -> None:
        """Surface exceptions that would otherwise die silently in the Future."""
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.exception("Background task raised an exception", exc_info=exc)

    def shutdown(self, wait: bool = False) -> None:
        """Stop the pool, dropping tasks that have not started yet.

        Args:
            wait: Block until in-flight tasks finish. Left False by default so
                shutdown is not held hostage by a multi-minute analysis.
        """
        self._executor.shutdown(wait=wait, cancel_futures=True)
