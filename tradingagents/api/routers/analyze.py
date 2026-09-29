"""Router for the full analysis pipeline endpoint."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from tradingagents.api.config import ApiConfig, get_config
from tradingagents.api.core.exceptions import AnalysisCancelled
from tradingagents.api.core.sync_runs import SyncRuns
from tradingagents.api.core.task_manager import CancelOutcome, TaskManager
from tradingagents.api.core.task_worker import TaskWorker
from tradingagents.api.dependencies import (
    get_analysis_service,
    get_task_manager,
    get_task_service,
    get_task_worker,
)
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.config_overrides import resolve_overrides
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.schemas.request import AnalyzeRequest
from tradingagents.api.schemas.response import AnalyzeResponse
from tradingagents.api.schemas.symbols import TICKER_NOT_SUPPORTED_RESPONSES
from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
)

# How often a waiting synchronous request checks whether its client has gone.
DISCONNECT_POLL_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Synchronous analysis endpoint (legacy)
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=AnalyzeResponse,
    summary="Run full analysis pipeline (synchronous)",
    description=(
        "Execute the complete multi-agent trading analysis pipeline for a given "
        "ticker and date. This runs all selected analysts, the research debate, "
        "trader proposal, and risk management discussion to produce a final "
        "trade decision.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `asset_type`: 'stock' or 'crypto' (default: 'stock'), case-insensitive\n"
        "- `selected_analysts`: List of 'market', 'social', 'news', 'fundamentals' "
        "(default: all four; 'sentiment' is accepted as an alias of 'social')\n"
        "- `debug`: Enable debug mode (default: false)\n\n"
        "**Performance overrides (all optional; see `GET /config` for current server "
        "defaults):**\n"
        "- `research_depth`: 'shallow'/'medium'/'deep' preset for debate/risk rounds "
        "(1/3/5)\n"
        "- `max_debate_rounds`, `max_risk_discuss_rounds`: explicit round counts "
        "(1-5), take precedence over `research_depth`\n"
        "- `deep_think_llm`, `quick_think_llm`: override the models used for this run\n"
        "- `max_tokens`, `llm_max_retries`: cap output tokens / retry attempts for "
        "this run\n\n"
        "**Pipeline Steps:**\n"
        "1. **Analysts** (market, social, news, fundamentals) - gather data and produce reports\n"
        "2. **Researchers** (bull/bear) - debate the investment thesis\n"
        "3. **Research Manager** - synthesize debate into an investment plan\n"
        "4. **Trader** - convert investment plan into a transaction proposal\n"
        "5. **Risk Management** (aggressive/conservative/neutral) - debate risk factors\n"
        "6. **Portfolio Manager** - produce final trade decision\n\n"
        "Note: This endpoint may take significant time to respond (30s+) depending "
        "on the LLM provider and analysis depth. It never blocks other requests, "
        "but it holds the connection open for the whole run. With `llm_provider` "
        "`ollama` it runs on the task worker pool, queued with `POST /analyze/tasks` "
        "tasks and subject to `task_max_concurrent` (1 by default with Ollama); "
        "with any other provider it runs on the threadpool and is not. "
        "For async operation, use `POST /analyze/tasks` instead.\n\n"
        "The run is cancelled -- it stops at its next LLM call, or never starts if "
        "still queued -- when the client disconnects (a client or proxy timeout "
        "included) or the server shuts down; the request then ends with 503 "
        "(`Analysis cancelled`), which a disconnected client never sees. As with "
        "tasks, a cancel during the final decision may still leave it in the "
        "core's decision log.\n\n"
        "The ticker is checked against the supported-symbols list first; see the 422 "
        "response (`ticker_not_supported`) and `GET /symbols/check`."
    ),
    responses=TICKER_NOT_SUPPORTED_RESPONSES,
)
async def analyze(
    request: AnalyzeRequest,
    http_request: Request,
    analysis_service: AnalysisService = Depends(get_analysis_service),
    task_worker: TaskWorker = Depends(get_task_worker),
    config: ApiConfig = Depends(get_config),
) -> AnalyzeResponse:
    """Run the full trading analysis pipeline (synchronous)."""
    # Convert enum values to strings for the service function
    selected = tuple(a.value for a in request.selected_analysts)
    cancel_event = threading.Event()
    run = functools.partial(
        analysis_service.run_analysis,
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type.value,
        selected_analysts=selected,
        overrides=resolve_overrides(request),
        cancel_event=cancel_event,
    )

    sync_runs: SyncRuns = http_request.app.state.sync_runs
    with sync_runs.track(cancel_event):
        if config.uses_ollama:
            # Serialised with the queued tasks: one local model, one worker
            # by default (see ApiConfig.task_max_concurrent).
            future: Any = task_worker.submit_for_caller(run)
        else:
            future = asyncio.ensure_future(run_in_threadpool(run))
        try:
            result = await _await_run(future, http_request, cancel_event)
        except AnalysisCancelled:
            logger.info(
                "Synchronous analysis of %s on %s cancelled", request.ticker, request.trade_date
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Analysis cancelled: the client disconnected or the server is shutting down"
                ),
            ) from None

    return AnalyzeResponse(**result.to_dict())


async def _await_run(
    future: concurrent.futures.Future | asyncio.Future,
    http_request: Request,
    cancel_event: threading.Event,
) -> Any:
    """Wait for a sync run, cancelling it once its client has disconnected.

    The run is cancelled through ``cancel_event`` (and, if it is still queued,
    by cancelling its Future), and this keeps waiting until it has stopped, so
    the worker slot is free again when the request ends. Should this coroutine
    itself be cancelled (a server past its graceful-shutdown timeout), the run
    is cancelled the same way before that propagates.

    Raises:
        AnalysisCancelled: the run was cancelled, queued or running.
    """
    waiter = (
        asyncio.wrap_future(future) if isinstance(future, concurrent.futures.Future) else future
    )

    def cancel_run() -> None:
        cancel_event.set()
        if isinstance(future, concurrent.futures.Future):
            future.cancel()  # dropped from the queue if it has not started

    try:
        while True:
            done, _ = await asyncio.wait({waiter}, timeout=DISCONNECT_POLL_SECONDS)
            if done:
                break
            if not cancel_event.is_set() and await http_request.is_disconnected():
                logger.info("Client of a synchronous analysis disconnected; cancelling the run")
                cancel_run()
    except asyncio.CancelledError:
        cancel_run()
        raise
    if waiter.cancelled():
        raise AnalysisCancelled("analysis cancelled before it started")
    return waiter.result()


# ---------------------------------------------------------------------------
# Async task endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/tasks",
    response_model=TaskResponse,
    summary="Create an analysis task (async)",
    description=(
        "Create a new analysis task that runs in the background. "
        "Returns immediately with a task_id. "
        "If a task with the same ticker, trade_date, and asset_type is already "
        "active (pending, queued, or processing), returns the existing task_id instead "
        "of creating a duplicate.\n\n"
        "If the task queue is full (task_max_tasks limit exceeded), returns "
        "429 Too Many Requests.\n\n"
        "Tasks are queued automatically when concurrency limit is reached "
        "(`task_max_concurrent`; 1 by default with Ollama, see `GET /config`). "
        "Use `GET /analyze/tasks/{task_id}` to check the status and retrieve "
        "results when completed, and `POST /analyze/tasks/{task_id}/cancel` to "
        "stop it.\n\n"
        "The ticker is checked against the supported-symbols list before anything is "
        "queued; see the 422 response (`ticker_not_supported`) and `GET /symbols/check`."
    ),
    responses=TICKER_NOT_SUPPORTED_RESPONSES,
)
async def create_analysis_task(
    request: TaskCreateRequest,
    task_service: TaskService = Depends(get_task_service),
    task_manager: TaskManager = Depends(get_task_manager),
    task_worker: TaskWorker = Depends(get_task_worker),
    config: ApiConfig = Depends(get_config),
) -> JSONResponse:
    """Create an analysis task or return existing one if duplicate."""
    max_tasks = int(config.config.get("task_max_tasks", 100))

    def queue_full_response() -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Task queue is full",
                "detail": (
                    f"Maximum task limit ({max_tasks}) reached. "
                    "Wait for some tasks to complete or delete finished tasks."
                ),
            },
        )

    # Only in-flight tasks count against the limit; completed ones are just
    # results waiting for their TTL to expire.
    if task_manager.active_task_count >= max_tasks:
        return queue_full_response()

    try:
        task, is_new = task_service.create_task(request)
    except ValueError:
        # TaskManager enforces its own max_tasks on total retained records.
        return queue_full_response()

    if is_new:
        # Queued in the worker pool; it starts once a slot frees up.
        task_manager.update_task(task.task_id, status=TaskStatus.QUEUED)
        future = task_worker.submit(task_service.execute_analysis, task.task_id, request)
        # Kept so cancelling a queued task drops it from the worker's queue.
        task_manager.attach_future(task.task_id, future)
        status_code = status.HTTP_202_ACCEPTED
    else:
        # Existing active task found: return it without queuing
        status_code = status.HTTP_200_OK

    return JSONResponse(
        status_code=status_code,
        content=task.model_dump(mode="json"),
    )


@router.get(
    "/tasks",
    response_model=list[TaskResponse],
    summary="List analysis tasks",
    description=(
        "List analysis tasks with optional filters for status and ticker. Statuses: "
        "`pending`, `queued`, `processing`, `completed`, `failed`, `cancelled`."
    ),
)
async def list_analysis_tasks(
    status_filter: TaskStatus | None = Query(
        default=None,
        alias="status",
        description="Filter by task status",
    ),
    ticker: str | None = Query(
        default=None,
        description="Filter by ticker symbol",
    ),
    task_service: TaskService = Depends(get_task_service),
) -> list[TaskResponse]:
    """List analysis tasks with optional filters."""
    return task_service.list_tasks(status=status_filter, ticker=ticker)


@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="Get analysis task status",
    description=(
        "Get the current status and result of an analysis task. "
        "Poll this endpoint until status is 'completed', 'failed' or 'cancelled'. "
        "A processing task that was asked to cancel reports `cancel_requested: true` "
        "until it stops."
    ),
)
async def get_analysis_task(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Get the status and result of an analysis task."""
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    return task


@router.post(
    "/tasks/{task_id}/cancel",
    response_model=TaskResponse,
    summary="Cancel an analysis task",
    description=(
        "Cancel an analysis task and return it.\n\n"
        "- **pending/queued**: cancelled at once; it never runs. The task comes back "
        "with status `cancelled`.\n"
        "- **processing**: the run stops at its next LLM call (the analysis cannot "
        "be interrupted mid-call). The task comes back with status `processing` and "
        "`cancel_requested: true`; poll `GET /analyze/tasks/{task_id}` until the "
        "status is `cancelled`.\n"
        "- **completed/failed/cancelled**: nothing to cancel; 409.\n\n"
        "A cancelled task is never reported as `failed`; its `message` says when "
        "it was cancelled and it has no `result`. A cancel can race the run's end, "
        "though: a processing task with `cancel_requested: true` may still end "
        "`completed` or `failed` when the run had already passed its last check. "
        "A cancel during the final decision may also leave that decision in the "
        "core's decision log (the core stores it before the API can discard the "
        "result)."
    ),
    responses={
        404: {"description": "Task not found"},
        409: {"description": "Task already finished (completed, failed or cancelled)"},
    },
)
async def cancel_analysis_task(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Cancel a queued or running analysis task."""
    outcome = task_service.cancel_task(task_id)
    task = task_service.get_task(task_id)
    if outcome == CancelOutcome.NOT_FOUND or task is None:
        # task is None: deleted (or expired) between the cancel and this read.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    if outcome == CancelOutcome.FINISHED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task {task_id} already finished with status '{task.status.value}'",
        )
    return task


@router.delete(
    "/tasks/{task_id}",
    summary="Delete an analysis task",
    description=(
        "Delete an analysis task. A pending, queued or processing task is cancelled "
        "first (see `POST /analyze/tasks/{task_id}/cancel`): a queued one never runs, "
        "and a running one stops at its next LLM call, its result discarded. "
        "Finished tasks (completed, failed, cancelled) are simply removed. The task "
        "is gone (404) at once; a running one still occupies its worker, and counts "
        "as active, until it has stopped."
    ),
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_analysis_task(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
) -> Response:
    """Delete an analysis task, cancelling it first if it is still active."""
    deleted = task_service.delete_task(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
