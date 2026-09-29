"""The synchronous ``POST /analyze`` runs in flight, so they can be cancelled.

A sync run has no task record; its cancel event lives here while its request
waits for it. The app's lifespan cancels them all on shutdown, and the request
cancels its own when the client disconnects (see ``routers.analyze``).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class SyncRuns:
    """Thread-safe set of the cancel events of the sync runs in flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: set[threading.Event] = set()

    @contextmanager
    def track(self, event: threading.Event) -> Iterator[threading.Event]:
        """Register ``event`` for the duration of one run."""
        with self._lock:
            self._events.add(event)
        try:
            yield event
        finally:
            with self._lock:
                self._events.discard(event)

    def cancel_all(self) -> int:
        """Set every run's cancel event; return how many there were."""
        with self._lock:
            events = list(self._events)
        for event in events:
            event.set()
        return len(events)

    @property
    def active_count(self) -> int:
        """Sync runs in flight (queued or running)."""
        with self._lock:
            return len(self._events)
