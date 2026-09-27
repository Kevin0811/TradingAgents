"""When the supported-symbols refresher may talk to Yahoo: only while idle.

TradingAgents shares Yahoo Finance with its own work: analysis tasks and the
``/data`` endpoints fetch prices, news and fundamentals from it. The symbol
lists are bulk fetches (up to ~55 screener pages for the US), so they are
kept out of the way of that work:

* ``ActivityMonitor`` says whether the API is idle: no analysis task pending,
  queued or processing in the TaskManager, no Yahoo-backed request in flight
  (``/data``, ``/analysts``, the synchronous ``POST /analyze``), and none
  finished within the idle grace period (``symbols_idle_grace_minutes``).
* ``RefreshGate`` is consulted by the symbol source between pages of a fetch
  that yields (an automatic refresh of an existing list): while the API is
  busy it pauses the refresh in small steps and resumes the same market once
  idle again. Whether a fetch yields is decided per fetch by the catalog and
  passed to the source (``fetch(market, yield_when_busy=...)``); a manual
  refresh and a missing list never yield.
* Manual work preempts a yielding fetch: ``release()`` lets it run on without
  pausing (a manual refresh of the same market), ``make_way()`` abandons it
  with ``RefreshPreempted`` instead of pausing (manual work for another
  market is waiting; the catalog queues it again after that work).
  ``stop()`` (shutdown) abandons it with ``RefreshInterrupted``.

Deciding *whether to start* an automatic refresh (idle, plus the optional
refresh window) is the catalog's job; the gate only makes a running one yield.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# How long a paused refresh sleeps before it checks for idleness again.
GATE_POLL_SECONDS = 5.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RefreshInterrupted(RuntimeError):
    """A refresh was abandoned because the catalog is shutting down."""


class RefreshPreempted(RuntimeError):
    """An automatic refresh gave way to manual work instead of pausing."""


@dataclass(frozen=True)
class PauseResult:
    """What ``RefreshGate.wait_until_idle`` did before returning."""

    paused_seconds: float = 0.0  # how long it paused (0 when it did not)
    released: bool = False  # manual work released the fetch: stop yielding


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
    """Makes a yielding refresh wait between pages while the API is busy.

    The per-fetch flags (``release`` / ``make_way``) are reset by
    ``begin_fetch()``; the catalog calls all three under its own lock, so a
    flag never outlives the fetch it was meant for.
    """

    def __init__(
        self,
        is_idle: Callable[[], bool] = lambda: True,
        *,
        poll_seconds: float | None = None,
        wait: Callable[[float], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Initialize the gate.

        Args:
            is_idle: Whether the API is idle now.
            poll_seconds: Pause between idleness checks while waiting
                (default ``GATE_POLL_SECONDS``).
            wait: ``wait(seconds) -> stopped``; defaults to waiting on the
                gate's wake-up event, so ``stop()``, ``release()`` and
                ``make_way()`` end a pause at once (tests pass a fake that
                advances a clock instead of sleeping).
            clock: Monotonic seconds, to measure how long a pause lasted.
        """
        self._is_idle = is_idle
        self._poll = GATE_POLL_SECONDS if poll_seconds is None else poll_seconds
        self._stopped = threading.Event()
        self._wake = threading.Event()
        self._wait = wait or self._wait_for_wake
        self._clock = clock
        self._released = False
        self._make_way = False
        self.paused = False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def check_stopped(self) -> None:
        """Raise RefreshInterrupted once the gate is stopped."""
        if self._stopped.is_set():
            raise RefreshInterrupted("the symbol catalog is shutting down")

    def sleep(self, seconds: float) -> None:
        """Sleep up to ``seconds``; returns early when the gate is stopped."""
        self._stopped.wait(seconds)

    # ------------------------------------------------------------------
    # Per-fetch preemption
    # ------------------------------------------------------------------

    def begin_fetch(self) -> None:
        """Forget the previous fetch's release / make-way request."""
        self._released = False
        self._make_way = False
        self._wake.clear()

    def release(self) -> None:
        """Let the current fetch run to its end without pausing again."""
        self._released = True
        self._wake.set()

    def make_way(self) -> None:
        """Manual work waits: abandon the current fetch rather than pause it."""
        self._make_way = True
        self._wake.set()

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    def wait_until_idle(self, label: str = "") -> PauseResult:
        """Return once idle, or once released; see PauseResult.

        Raises:
            RefreshInterrupted: The gate was stopped (shutdown).
            RefreshPreempted: The API is busy and manual work is waiting.
        """
        self.check_stopped()
        if self._released:
            return PauseResult(released=True)
        if self._is_idle():
            return PauseResult()
        where = f" ({label})" if label else ""
        if self._make_way:
            raise RefreshPreempted(f"the refresh{where} made way for a manual refresh")
        logger.info("Pausing the symbol refresh%s: TradingAgents is busy", where)
        started = self._clock()
        self.paused = True
        try:
            while not self._is_idle():
                if self._released:
                    logger.info(
                        "Resuming the symbol refresh%s without pausing: a manual refresh "
                        "asked for it",
                        where,
                    )
                    return PauseResult(self._clock() - started, released=True)
                if self._make_way:
                    raise RefreshPreempted(f"the refresh{where} made way for a manual refresh")
                if self._wait(self._poll) or self._stopped.is_set():
                    raise RefreshInterrupted("the symbol catalog is shutting down")
        finally:
            self.paused = False
        logger.info("Resuming the symbol refresh%s", where)
        return PauseResult(self._clock() - started)

    def _wait_for_wake(self, seconds: float) -> bool:
        self._wake.wait(seconds)
        self._wake.clear()
        return self._stopped.is_set()
