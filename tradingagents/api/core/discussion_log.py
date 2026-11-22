"""The per-task discussion event log (Plan 08, event schema v1).

A run's discussion -- who spoke, in which phase, which tools they called and
what they finally said -- is recorded as a list of events, one log per task.
The log is append-only and lives in memory as long as its task does (the task
TTL); a finished run's transcript is also written to a file (see
``infrastructure/transcript_files.py``).

Envelope (every event)::

    {"id": 7, "v": 1, "type": "turn_started", "ms": 1234, "at": "2026-...Z", "data": {...}}

``id`` counts from 1 per task with no gaps, so an event's position in the log
is ``id - 1`` and a subscriber needs nothing but the last id it has seen (its
cursor): the log is never copied per subscriber and a client that stops
reading costs nothing but that integer. ``ms`` counts from the start of the
run (0 before it).

Bounds: at most :data:`MAX_EVENTS` events per log. The first event past that
is replaced by one ``gap`` event, ``{"dropped_from_id": N}`` (N is the gap's
own id: nothing from there on but the terminal event is kept), so a resuming
client sees the loss; later events are only counted, and the count is
reported in the terminal ``task_status`` as ``events_dropped``. The gap and
the terminal ``task_status`` always fit, so a log holds at most
``MAX_EVENTS + 2`` events. Turn text is capped by the emitter (see
``infrastructure/discussion_capture.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1

# Persisted event types (all kept in the log and the transcript).
RUN_STARTED = "run_started"
PHASE_CHANGED = "phase_changed"
TURN_STARTED = "turn_started"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
TURN_COMPLETED = "turn_completed"
TURN_ABORTED = "turn_aborted"
TASK_STATUS = "task_status"
# Sent once, in the log, when the log is full (see the module docstring).
GAP = "gap"
PERSISTED_TYPES = frozenset({
    RUN_STARTED, PHASE_CHANGED, TURN_STARTED, TOOL_CALL, TOOL_RESULT,
    TURN_COMPLETED, TURN_ABORTED, TASK_STATUS, GAP,
})
# Live-only types: part of schema v1, never stored and not emitted yet
# (token streaming is off and unimplemented).
TOKEN = "token"
LIVE_ONLY_TYPES = frozenset({TOKEN})

# task_status values. "running" is sent when the run starts; the others end the log.
# A log whose run has not started reports "queued" in its transcript.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# Hard ceiling on one task's events. A default run makes about 60 (12 turns, ~25
# tool calls); the longest allowed one (4 analysts, 5/5 rounds: 32 turns) about
# 400 with the tool cap. Anything past this is counted, not kept.
MAX_EVENTS = 1000


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class DiscussionLog:
    """One task's append-only event log. Thread-safe; appends never block on I/O.

    Writers are the run's callback threads and the task manager; readers are
    SSE subscribers on the event loop. A subscriber waits with :meth:`wait`,
    which registers one ``asyncio.Event`` for as long as it waits and nothing
    else.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._status: str = STATUS_QUEUED
        self._finished = False
        self._dropped = 0
        self._notes: dict[str, Any] = {}
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()

    # -- writing ---------------------------------------------------------------

    def start_clock(self) -> None:
        """Start ``ms`` at 0 (the run's start). Idempotent."""
        with self._lock:
            if self._t0 is None:
                self._t0 = time.monotonic()

    def append(self, type_: str, data: dict[str, Any]) -> dict[str, Any] | None:
        """Append one event; return it, or None when the log is finished or full."""
        with self._lock:
            if self._finished:
                return None
            if len(self._events) >= MAX_EVENTS:
                if not self._dropped:   # the first loss: one marker, past the cap
                    self._append_locked(GAP, {"dropped_from_id": len(self._events) + 1})
                self._dropped += 1
                return None
            if type_ == TASK_STATUS:
                self._status = data.get("status", self._status)
            return self._append_locked(type_, data)

    def note(self, key: str, value: Any) -> None:
        """Keep a counter for the terminal ``task_status`` (e.g. ``tools_dropped``)."""
        with self._lock:
            self._notes[key] = value

    def finish(self, status: str, **codes: str) -> dict[str, Any] | None:
        """Append the terminal ``task_status`` and close the log. Idempotent.

        ``codes`` are the task's ``error_code`` / ``cancel_reason`` when set;
        they go into the event's data as they are.
        """
        with self._lock:
            if self._finished:
                return None
            data: dict[str, Any] = {"status": status, **self._notes, **codes}
            if self._dropped:
                data["events_dropped"] = self._dropped
            self._status = status
            self._finished = True
            return self._append_locked(TASK_STATUS, data)

    def _append_locked(self, type_: str, data: dict[str, Any]) -> dict[str, Any]:
        ms = 0 if self._t0 is None else int((time.monotonic() - self._t0) * 1000)
        event = {
            "id": len(self._events) + 1,
            "v": SCHEMA_VERSION,
            "type": type_,
            "ms": ms,
            "at": _now_iso(),
            "data": data,
        }
        self._events.append(event)
        self._wake_locked()
        return event

    def _wake_locked(self) -> None:
        for loop, waiter in self._waiters:
            with contextlib.suppress(RuntimeError):  # the subscriber's loop has closed
                loop.call_soon_threadsafe(waiter.set)

    def wake(self) -> None:
        """Wake every waiting subscriber (e.g. so it sees the server is closing)."""
        with self._lock:
            self._wake_locked()

    # -- reading ---------------------------------------------------------------

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def last_id(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def waiter_count(self) -> int:
        with self._lock:
            return len(self._waiters)

    def read(self, after: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
        """The events with ``id > after`` (at most ``limit``), shared, never copied."""
        start = max(0, int(after))
        with self._lock:
            end = len(self._events) if limit is None else min(len(self._events), start + limit)
            return self._events[start:end]

    async def wait(
        self, after: int, timeout: float, stop: threading.Event | None = None
    ) -> bool:
        """Wait until there is an event past ``after``, the log has finished or ``stop`` is set.

        Returns True if there is something to read (or the log has finished,
        or ``stop`` is set), False on timeout. ``stop`` is checked under the
        log's lock, so a setter that then calls :meth:`wake` is never missed.
        """
        loop = asyncio.get_running_loop()
        waiter = asyncio.Event()
        key = (loop, waiter)
        with self._lock:
            if len(self._events) > after or self._finished or (stop is not None and stop.is_set()):
                return True
            self._waiters.add(key)
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
            return True
        except TimeoutError:
            return False
        finally:
            with self._lock:
                self._waiters.discard(key)

    def transcript(self, mapping_version: str | None = None) -> dict[str, Any]:
        """The transcript built from the log (the ``/transcript`` body)."""
        events = self.read()
        return build_transcript(self.task_id, events, self.status, mapping_version)


def build_transcript(
    task_id: str,
    events: list[dict[str, Any]],
    status: str,
    mapping_version: str | None = None,
) -> dict[str, Any]:
    """Fold a task's events into its transcript: one entry per turn, in turn order."""
    run_started: dict[str, Any] | None = None
    turns: dict[int, dict[str, Any]] = {}
    open_by_agent: dict[str, int] = {}
    tools_dropped = 0
    events_dropped = 0
    for event in events:
        type_, data = event["type"], event["data"]
        if type_ == RUN_STARTED:
            run_started = data
            mapping_version = data.get("mapping_version", mapping_version)
        elif type_ == TURN_STARTED:
            turns[data["turn"]] = {
                "turn": data["turn"],
                "agent_id": data["agent_id"],
                "role": data.get("role"),
                "phase": data.get("phase"),
                "started_at": event["at"],
                "completed_at": None,
                "aborted": False,
                "content": None,
                "truncated": False,
                "tools": [],
            }
            open_by_agent[data["agent_id"]] = data["turn"]
        elif type_ == TOOL_CALL:
            turn = turns.get(open_by_agent.get(data["agent_id"], -1))
            if turn is not None:
                turn["tools"].append(data["tool"])
        elif type_ in (TURN_COMPLETED, TURN_ABORTED):
            turn = turns.get(data["turn"])
            if turn is None:
                continue
            turn["completed_at"] = event["at"]
            if type_ == TURN_COMPLETED:
                turn["content"] = data.get("content")
                turn["truncated"] = bool(data.get("truncated"))
            else:
                turn["aborted"] = True
            if open_by_agent.get(data["agent_id"]) == data["turn"]:
                del open_by_agent[data["agent_id"]]
        elif type_ == TASK_STATUS:
            tools_dropped = data.get("tools_dropped", tools_dropped)
            events_dropped = data.get("events_dropped", events_dropped)
    return {
        "task_id": task_id,
        "v": SCHEMA_VERSION,
        "mapping_version": mapping_version,
        "status": status,
        "run_started": run_started,
        "turns": [turns[n] for n in sorted(turns)],
        "tools_dropped": tools_dropped,
        "events_dropped": events_dropped,
    }


class DiscussionStore:
    """The app's discussion logs, keyed by task id.

    Logs are created with their task and dropped with it (the task manager
    calls :meth:`create`, :meth:`finish` and :meth:`drop`), so the store holds
    at most as many logs as the task manager holds tasks (``task_max_tasks``).
    """

    def __init__(self) -> None:
        self._logs: dict[str, DiscussionLog] = {}
        self._lock = threading.Lock()
        # Set when the server shuts down: open event streams end at once.
        self.closing = threading.Event()

    def close(self) -> None:
        """Mark the store closing and wake every waiting subscriber so its stream ends."""
        self.closing.set()
        with self._lock:
            logs = list(self._logs.values())
        for log in logs:
            log.wake()

    def create(self, task_id: str) -> DiscussionLog:
        with self._lock:
            log = self._logs.get(task_id)
            if log is None:
                log = self._logs[task_id] = DiscussionLog(task_id)
            return log

    def get(self, task_id: str) -> DiscussionLog | None:
        with self._lock:
            return self._logs.get(task_id)

    def finish(self, task_id: str, status: str, **codes: str) -> None:
        log = self.get(task_id)
        if log is not None:
            log.finish(status, **codes)

    def drop(self, task_id: str) -> None:
        """Forget a task's log. A log not yet finished (a queued task deleted
        before it ran) is first finished as cancelled, so an open stream on it
        still receives a terminal ``task_status`` instead of hanging."""
        with self._lock:
            log = self._logs.pop(task_id, None)
        if log is not None:
            log.finish("cancelled")

    def __len__(self) -> int:
        with self._lock:
            return len(self._logs)
