"""Routes for a task's discussion: a live event stream (SSE) and its transcript."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from tradingagents.api.core.discussion_log import DiscussionLog, DiscussionStore
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.dependencies import (
    get_discussion_store,
    get_task_manager,
    get_transcript_files,
)
from tradingagents.api.infrastructure.discussion_capture import MAPPING_VERSION
from tradingagents.api.infrastructure.transcript_files import TranscriptFiles

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# A comment line keeps idle connections (and the proxies on their way) open.
PING_INTERVAL_SECONDS = 15.0
# Events sent per write; a reconnecting client catches up in chunks of this.
READ_BATCH = 100

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _not_found(task_id: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")


def _live_log(task_id: str, task_manager: TaskManager, store: DiscussionStore) -> DiscussionLog | None:
    """The task's log while the task itself is known (not deleted, not expired)."""
    if task_manager.get_task(task_id) is None:
        return None
    return store.get(task_id)


def _sse(event: dict[str, Any]) -> str:
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event['id']}\ndata: {data}\n\n"


async def stream_events(log: DiscussionLog, after: int) -> AsyncIterator[str]:
    """Send every event past ``after``, then each new one, until the log has ended.

    The subscriber's whole state is its cursor (the last id sent). Cancelling
    the generator (the client went away) leaves nothing behind.
    """
    loop = asyncio.get_running_loop()
    cursor = after
    last_write = loop.time()
    while True:
        batch = log.read(cursor, READ_BATCH)
        if batch:
            yield "".join(_sse(e) for e in batch)
            cursor = batch[-1]["id"]
            last_write = loop.time()
            continue
        # finished is set with the terminal event, so a finished log has
        # nothing past last_id: done once the cursor has reached it.
        if log.finished and log.last_id <= cursor:
            return
        interval = PING_INTERVAL_SECONDS
        remaining = interval - (loop.time() - last_write)
        if remaining <= 0:
            yield ": ping\n\n"
            last_write = loop.time()
            continue
        await log.wait(cursor, remaining)


def _cursor(last_event_id: str | None, after: int) -> int:
    """``Last-Event-ID`` (a reconnect) wins over ``?after``; an unreadable one is ignored."""
    if last_event_id is not None:
        try:
            return max(0, int(last_event_id.strip()))
        except ValueError:
            pass
    return after


@router.get(
    "/tasks/{task_id}/events",
    summary="Stream a task's discussion (SSE)",
    description=(
        "Server-sent events of the task's discussion, event schema v1. Each event's "
        "`data` is one JSON envelope `{id, v, type, ms, at, data}` and its SSE `id` is "
        "the envelope's `id`, counting from 1 per task with no gaps.\n\n"
        "Persisted types: `run_started` (`ticker`, `trade_date`, `analysts`, "
        "`debate_rounds`, `risk_rounds`, `expected_turns`, `mapping_version`), "
        "`phase_changed`, `turn_started`, `tool_call`, `tool_result` (tool names only), "
        "`turn_completed` (`content` capped at 32 KiB, `truncated`), `turn_aborted` and "
        "`task_status` (`running`, then one of `completed`, `failed`, `cancelled`). "
        "`token` and `gap` are reserved for live-only use and not sent yet.\n\n"
        "Resume with the `Last-Event-ID` header (preferred) or `?after=N`: only events "
        "with a higher id are sent. `?tokens=1` is accepted and treated as `0`: token "
        "streaming is not implemented. An idle stream gets a `: ping` comment every "
        "15 s. The stream ends once the terminal `task_status` has been sent.\n\n"
        "Like every endpoint of this API, this one has no authentication: keep the "
        "API on a trusted network."
    ),
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "The event stream"},
        404: {"description": "Task not found"},
    },
    response_class=StreamingResponse,
)
async def stream_task_events(
    task_id: str,
    after: int = Query(default=0, ge=0, description="Send only events with a higher id"),
    tokens: int = Query(default=0, ge=0, le=1, description="Reserved; 1 is treated as 0"),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    task_manager: TaskManager = Depends(get_task_manager),
    store: DiscussionStore = Depends(get_discussion_store),
) -> StreamingResponse:
    """Stream the task's discussion events."""
    log = _live_log(task_id, task_manager, store)
    if log is None:
        raise _not_found(task_id)
    return StreamingResponse(
        stream_events(log, _cursor(last_event_id, after)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get(
    "/tasks/{task_id}/transcript",
    summary="Get a task's discussion transcript",
    description=(
        "The task's discussion as JSON: `{task_id, v, mapping_version, status, "
        "run_started, turns, tools_dropped, events_dropped}`, each turn "
        "`{turn, agent_id, role, phase, started_at, completed_at, aborted, content, "
        "truncated, tools}` (`tools`: the names of the tools it called, in order).\n\n"
        "Served from memory while the task is kept (its TTL), then from the "
        "transcript file written next to the run's results when it ended. A task "
        "cancelled before it started has no file. 404 when neither exists."
    ),
    responses={404: {"description": "Task not found"}},
)
async def get_task_transcript(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager),
    store: DiscussionStore = Depends(get_discussion_store),
    transcripts: TranscriptFiles = Depends(get_transcript_files),
) -> dict[str, Any]:
    """Return the task's transcript."""
    log = _live_log(task_id, task_manager, store)
    if log is not None:
        return log.transcript(MAPPING_VERSION)
    data = await run_in_threadpool(transcripts.read, task_id)
    if data is None:
        raise _not_found(task_id)
    return data
