"""When the supported-symbols refresher may talk to Yahoo: only while idle.

TradingAgents shares Yahoo Finance with its own work: analysis tasks and the
``/data`` endpoints fetch prices, news and fundamentals from it. The symbol
lists are bulk fetches (up to ~55 screener pages for the US), so they are
kept out of the way of that work:

* ``ActivityMonitor`` says whether the API is idle: no analysis task pending,
  queued or processing in the TaskManager, no Yahoo-backed request in flight
  (``/data``, ``/analysts``, the synchronous ``POST /analyze``), and none
  finished within the idle grace period (``symbols_idle_grace_minutes``).
* ``RefreshGate`` is consulted by the symbol source between pages: while the
  API is busy it pauses the refresh in small steps and resumes the same
  market once idle again. Only ``shutdown()`` abandons it.

Deciding *whether to start* an automatic refresh (idle, plus the optional
refresh window) is the catalog's job; the gate only makes a running one yield.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# How long a paused refresh sleeps before it checks for idleness again.
GATE_POLL_SECONDS = 5.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RefreshInterrupted(RuntimeError):
    """A paused refresh was abandoned because the catalog is shutting down."""


class ActivityMonitor:
    """Tracks Yahoo-backed API activity to tell when the API is idle."""

    def __init__(
        self,
        busy_tasks: Callable[[], int],
        grace: Callable[[], timedelta],
        clock: Callable[[], datetime] = _utcnow,
    ):
        """Initialize the monitor.

        Args:
            busy_tasks: Number of analysis tasks pending, queued or processing.
            grace: How long after the last Yahoo-backed request the API stays
                busy (read on every check, so a settings change applies live).
            clock: Returns the current UTC time (tests).
        """
        self._busy_tasks = busy_tasks
        self._grace = grace
        self._clock = clock
        self._lock = threading.Lock()
        self._in_flight = 0
        self._last_request_at: datetime | None = None

    def request_started(self) -> None:
        with self._lock:
            self._in_flight += 1
            self._last_request_at = self._clock()

    def request_finished(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._last_request_at = self._clock()

    def active_tasks(self) -> int:
        try:
            return int(self._busy_tasks())
        except Exception:  # pragma: no cover - never let a probe break the refresher
            logger.exception("Could not count active analysis tasks; assuming busy")
            return 1

    def is_idle(self) -> bool:
        """True when no task is active and no Yahoo-backed request is recent."""
        if self.active_tasks() > 0:
            return False
        with self._lock:
            if self._in_flight:
                return False
            last = self._last_request_at
        return last is None or self._clock() - last >= self._grace()

    def state(self) -> dict:
        """Snapshot for ``GET /symbols/settings``."""
        with self._lock:
            in_flight, last = self._in_flight, self._last_request_at
        return {
            "idle": self.is_idle(),
            "active_tasks": self.active_tasks(),
            "data_requests_in_flight": in_flight,
            "last_data_request_at": last,
        }


class RefreshGate:
    """Makes a running refresh wait between pages while the API is busy."""

    def __init__(
        self,
        is_idle: Callable[[], bool] = lambda: True,
        *,
        poll_seconds: float = GATE_POLL_SECONDS,
        wait: Callable[[float], bool] | None = None,
    ):
        """Initialize the gate.

        Args:
            is_idle: Whether the API is idle now.
            poll_seconds: Pause between idleness checks while waiting.
            wait: ``wait(seconds) -> stopped``; defaults to waiting on the
                gate's stop event, so ``stop()`` ends a pause at once (tests
                pass a fake that advances a clock instead of sleeping).
        """
        self._is_idle = is_idle
        self._poll = poll_seconds
        self._stopped = threading.Event()
        self._wait = wait or self._stopped.wait
        # Set by the catalog for each fetch: a manual refresh does not yield.
        self.active = True
        self.paused = False

    def stop(self) -> None:
        self._stopped.set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def wait_until_idle(self, label: str = "") -> None:
        """Return once idle (at once if inactive); raise RefreshInterrupted on stop."""
        if self._stopped.is_set():
            raise RefreshInterrupted("the symbol catalog is shutting down")
        if not self.active or self._is_idle():
            return
        where = f" ({label})" if label else ""
        logger.info("Pausing the symbol refresh%s: TradingAgents is busy", where)
        self.paused = True
        try:
            while not self._is_idle():
                if self._wait(self._poll) or self._stopped.is_set():
                    raise RefreshInterrupted("the symbol catalog is shutting down")
        finally:
            self.paused = False
        logger.info("Resuming the symbol refresh%s", where)
