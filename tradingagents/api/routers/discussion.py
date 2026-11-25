"""Routes for a task's discussion: a live event stream (SSE) and its transcript.

An event stream never holds up shutdown for long: it ends on its own after
:data:`MAX_STREAM_SECONDS` (the client resumes with ``Last-Event-ID``), and at
once when the app's discussion store is closed (the lifespan shutdown).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from tradingagents.api.core.discussion_log import DiscussionLog, DiscussionStore
from tradingagents.api.core.exceptions import (
    DataNotFoundError,
    StateLogUnreadableError,
    UnprocessableRequestError,
)
from tradingagents.api.core.rebuilt_transcript import build_rebuilt_transcript
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.dependencies import (
    get_discussion_store,
    get_file_state_repository,
    get_task_manager,
    get_transcript_files,
)
from tradingagents.api.domain.symbols import api_forex_symbol
from tradingagents.api.infrastructure.discussion_capture import MAPPING_VERSION
from tradingagents.api.infrastructure.repositories.file_state_repository import (
    FileStateRepository,
    StateLogIncompleteError,
)
from tradingagents.api.infrastructure.transcript_files import TranscriptFiles
from tradingagents.dataflows.symbols import safe_ticker_component

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# A comment line keeps idle connections (and the proxies on their way) open.
PING_INTERVAL_SECONDS = 15.0
# Events sent per write; a reconnecting client catches up in chunks of this.
READ_BATCH = 100
# A stream ends cleanly after this long, between two events; the client
# reconnects with Last-Event-ID and loses nothing. Keeps a server shutdown
# (which waits for open responses) from waiting on a stream forever.
MAX_STREAM_SECONDS = 300.0

# Characters that some line readers (Python's str.splitlines, JavaScript
# source) take as line breaks although SSE does not. JSON may escape them, so
# a data: line never contains them raw.
_LINE_BREAK_ESCAPES = str.maketrans({
    "\u2028": "\\u2028", "\u2029": "\\u2029", "\x85": "\\u0085",
})

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
    return f"id: {event['id']}\ndata: {data.translate(_LINE_BREAK_ESCAPES)}\n\n"


async def stream_events(
    log: DiscussionLog, after: int, closing: threading.Event | None = None
) -> AsyncIterator[str]:
    """Send every event past ``after``, then each new one, until the log has ended.

    A cursor past the end is taken as the end (``log.last_id``), so the
    client still gets the rest of the run. The stream also ends, between two
    events, after :data:`MAX_STREAM_SECONDS` or once ``closing`` is set.

    The subscriber's whole state is its cursor (the last id sent). Cancelling
    the generator (the client went away) leaves nothing behind.
    """
    loop = asyncio.get_running_loop()
    cursor = min(after, log.last_id)
    started = last_write = loop.time()
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
        now = loop.time()
        lifetime_left = MAX_STREAM_SECONDS - (now - started)
        if (closing is not None and closing.is_set()) or lifetime_left <= 0:
            return
        remaining = PING_INTERVAL_SECONDS - (now - last_write)
        if remaining <= 0:
            yield ": ping\n\n"
            last_write = loop.time()
            continue
        await log.wait(cursor, min(remaining, lifetime_left), stop=closing)


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
        "`task_status` (`running`, then one of `completed`, `failed`, `cancelled`; "
        "the terminal one also carries the task's `error_code` on `failed` or its "
        "`cancel_reason` on `cancelled`, the same codes as the task's fields; "
        "`cancel_reason` may be absent on a `cancelled` that ends a log removed "
        "after its run started). "
        "When the log is full (1000 events) one `gap` event `{dropped_from_id}` "
        "marks where events stop being kept; only the terminal `task_status` "
        "follows it. `token` is reserved for live-only use and not sent yet.\n\n"
        "Resume with the `Last-Event-ID` header (preferred) or `?after=N`: only events "
        "with a higher id are sent. `?tokens=1` is accepted and treated as `0`: token "
        "streaming is not implemented. An idle stream gets a `: ping` comment every "
        "15 s. The stream ends once the terminal `task_status` has been sent; it "
        "also ends, between two events, after 300 s or when the server shuts "
        "down: reconnect with `Last-Event-ID`. A cursor past the end is taken "
        "as the end.\n\n"
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
        stream_events(log, _cursor(last_event_id, after), store.closing),
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


TRADE_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


# A state log that is empty or not complete JSON, written less than
# STATE_LOG_FRESH_SECONDS ago, may still be being written: it is read once more
# after STATE_LOG_RETRY_DELAY seconds before the 503. An older one is corrupt: 500.
STATE_LOG_RETRY_DELAY = 0.2
STATE_LOG_FRESH_SECONDS = 60


def _unprocessable(detail: str) -> UnprocessableRequestError:
    return UnprocessableRequestError(message="Invalid request", detail=detail)


def state_log_tickers(ticker: str) -> list[str]:
    """The ticker folders a run of ``ticker`` may have logged under, in lookup order.

    First the form ``POST /analyze`` hands the graph (stripped; a forex pair
    the core leaves alone rewritten, ``USDTWD`` -> ``USDTWD=X``, as
    ``validate_ticker_shape`` does), then the ticker as given. Raises
    ValueError when one is not a safe path component.
    """
    raw = ticker.strip()
    forms: list[str] = []
    for form in (api_forex_symbol(raw) or raw, raw):
        if form not in forms:
            forms.append(safe_ticker_component(form))
    return forms


def _being_written(exc: Exception) -> bool:
    """Whether ``exc`` is an empty or truncated log written in the last minute."""
    if not isinstance(exc, StateLogIncompleteError):
        return False
    age = (datetime.now(UTC) - exc.logged_at).total_seconds()
    return abs(age) < STATE_LOG_FRESH_SECONDS


def _load_once_more(
    repo: FileStateRepository, ticker: str, trade_date: str
) -> tuple[dict[str, Any], datetime]:
    """Load the log; an empty or truncated file written just now is read again once."""
    try:
        return repo.load_state_logged(ticker, trade_date)
    except StateLogIncompleteError as exc:
        if not _being_written(exc):
            raise
        time.sleep(STATE_LOG_RETRY_DELAY)
        return repo.load_state_logged(ticker, trade_date)


def _load_first(
    repo: FileStateRepository, tickers: list[str], trade_date: str
) -> tuple[dict[str, Any], datetime] | None:
    """The state and mtime of the first readable log among ``tickers``; None if none exists.

    An unreadable log under one ticker form does not hide a good one under the
    next; when none is readable the first form's error is raised.
    """
    first_error: Exception | None = None
    for ticker in tickers:
        try:
            state, logged_at = _load_once_more(repo, ticker, trade_date)
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            first_error = first_error or exc
            continue
        return state, logged_at
    if first_error is not None:
        raise first_error
    return None


@router.get(
    "/rebuilt-transcript",
    summary="Rebuild a run's discussion transcript from its saved state log",
    description=(
        "The discussion of a completed run, rebuilt from the core's "
        "`full_states_log_<trade_date>.json` for runs that have no live transcript. "
        "Read-only: no graph, no model, no network.\n\n"
        "The body has the `/tasks/{task_id}/transcript` shape without `task_id`: "
        "`{v, mapping_version, status: \"completed\", run_started, turns, "
        "tools_dropped: 0, events_dropped: 0, investment_horizon, time_horizon, rebuilt}`. "
        "`investment_horizon` is always null (the state log does not keep the "
        "request's horizon); `time_horizon` is the portfolio manager's own, read "
        "from the logged `final_trade_decision` as the `/analyze` result's is "
        "(null when it gave none). Turns: the analysts "
        "(the run's order), the bull/bear debate, the research manager, the "
        "trader, the aggressive/conservative/neutral debate and the portfolio "
        "manager. With `run_settings` in the log, each seat the run had keeps "
        "a turn, with `\"\"` content when its field is empty (as live capture "
        "records it); without them, empty fields are skipped and the analysts "
        "come in the default order market, sentiment, news, fundamentals. "
        "Rebuilt turn numbers and counts do not equal a live transcript's: live "
        "capture opens a turn on each analyst node pass, tool loops included, so "
        "clients must not compare turn numbers between the two. "
        "A log has no tools or timings: "
        "`tools` is `[]`, `started_at`/`completed_at` are null. Content is "
        "capped as live (32 KiB a turn, the task budget, the final decision's "
        "share reserved).\n\n"
        "`rebuilt`: `logged_at` (the log file's mtime, UTC ISO: when the run "
        "wrote it, or when the file was last touched), "
        "`settings_known` (the log has `run_settings`), `split_exact` (true only "
        "when every debate boundary was confirmed by the speakers' side histories "
        "or the run's round count; when false a debate is one turn per side, or "
        "its pieces by speaker prefix, and no text of the debate's `history` is "
        "dropped) and "
        "`fingerprints` (sha256 hex of the raw UTF-8 text of `market_report`, "
        "`sentiment_report`, `news_report`, `fundamentals_report`, "
        "`investment_plan`, `trader_investment_plan`, `final_trade_decision`, "
        "for each present; the same names as the `/analyze` result). There is "
        "one log per ticker and date, overwritten by a re-run: compare the "
        "fingerprints to know it is your run's.\n\n"
        "The ticker is looked up as `POST /analyze` runs it (`USDTWD` -> "
        "`USDTWD=X`), then as given; an unreadable log under one form does not "
        "hide a readable one under the other.\n\n"
        "Errors are `{error, detail}`: 404 when there is no log (or its real "
        "path, symlinks resolved, is outside the results directory); 422 for a "
        "ticker that is not a safe path component or a date that is not a real "
        "`YYYY-MM-DD` date (a missing parameter or one that fails the query "
        "pattern is FastAPI's own 422, whose `detail` is a list); 503 when the "
        "log, last written under a minute ago, is empty or not complete JSON "
        "after one retry (a run may be writing it: transient, try again); 500 "
        "when it is not UTF-8, not a JSON object, over 64 MiB, or empty or not "
        "complete JSON and written a minute or more ago (corrupt: no retry).\n\n"
        "Like every endpoint of this API, this one has no authentication: keep the "
        "API on a trusted network."
    ),
    responses={
        404: {"description": "No state log for this ticker and date"},
        422: {"description": "Invalid ticker or trade date"},
        500: {"description": "The state log could not be read"},
        503: {"description": "The state log is empty or incomplete (being written); try again"},
    },
)
async def get_rebuilt_transcript(
    ticker: str = Query(..., description="Ticker symbol, as analysed"),
    trade_date: str = Query(
        ..., pattern=TRADE_DATE_PATTERN, description="Trading date in YYYY-MM-DD format"),
    repo: FileStateRepository = Depends(get_file_state_repository),
) -> dict[str, Any]:
    """Return the transcript rebuilt from the run's state log."""
    try:
        datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        raise _unprocessable("trade_date is not a valid YYYY-MM-DD date") from None
    try:
        tickers = state_log_tickers(ticker)
    except ValueError:
        raise _unprocessable("ticker is not a valid symbol") from None
    try:
        # Read and parse off the event loop.
        loaded = await run_in_threadpool(_load_first, repo, tickers, trade_date)
    except (OSError, ValueError) as exc:
        # The exception text can name the file: only its class is logged.
        logger.warning("Could not read the state log of %s on %s: %s",
                       tickers[0], trade_date, type(exc).__name__)
        if _being_written(exc):
            raise StateLogUnreadableError(
                message="The saved state log is not complete JSON",
                detail="The saved state log is being written; try again.",
                retryable=True,
            ) from None
        raise StateLogUnreadableError(
            message="Could not read the saved state log",
            detail="The state log for this ticker and date is unreadable or not valid JSON.",
        ) from None
    if loaded is None:
        raise DataNotFoundError(
            message=f"No state log for {tickers[0]} on {trade_date}",
            detail="Only completed runs have one.",
        )
    # Outside the handler above: a converter bug is a real 500, with its traceback.
    return await run_in_threadpool(build_rebuilt_transcript, *loaded)
