"""Router for the full analysis pipeline endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse

from tradingagents.api.config import ApiConfig, get_config
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.config_overrides import resolve_overrides
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.dependencies import (
    get_analysis_service,
    get_task_manager,
    get_task_service,
    get_task_worker,
)
from tradingagents.api.schemas.request import AnalyzeRequest
from tradingagents.api.schemas.response import AnalyzeResponse
from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.core.task_worker import TaskWorker

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
)


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
        "- `asset_type`: 'stock' or 'crypto' (default: 'stock')\n"
        "- `selected_analysts`: List of 'market', 'sentiment', 'news', 'fundamentals'\n"
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
        "1. **Analysts** (market, sentiment, news, fundamentals) - gather data and produce reports\n"
        "2. **Researchers** (bull/bear) - debate the investment thesis\n"
        "3. **Research Manager** - synthesize debate into an investment plan\n"
        "4. **Trader** - convert investment plan into a transaction proposal\n"
        "5. **Risk Management** (aggressive/conservative/neutral) - debate risk factors\n"
        "6. **Portfolio Manager** - produce final trade decision\n\n"
        "Note: This endpoint may take significant time to respond (30s+) depending "
        "on the LLM provider and analysis depth. It runs on the threadpool, so it "
        "does not block other requests, but it holds the connection open for the "
        "whole run and is not subject to `task_max_concurrent`. "
        "For async operation, use `POST /analyze/tasks` instead."
    ),
)
def analyze(
    request: AnalyzeRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Run the full trading analysis pipeline (synchronous)."""
    # Convert enum values to strings for the service function
    selected = tuple(a.value for a in request.selected_analysts)

    result = analysis_service.run_analysis(
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type.value,
        selected_analysts=selected,
        overrides=resolve_overrides(request),
    )

    return AnalyzeResponse(**result.to_dict())


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
        "Tasks are queued automatically when concurrency limit is reached. "
        "Use `GET /analyze/tasks/{task_id}` to check the status and retrieve "
        "results when completed."
    ),
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
        task_worker.submit(task_service.execute_analysis, task.task_id, request)
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
    description="List analysis tasks with optional filters for status and ticker.",
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
        "Poll this endpoint until status is 'completed' or 'failed'."
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


@router.delete(
    "/tasks/{task_id}",
    summary="Delete an analysis task",
    description=(
        "Delete an analysis task. "
        "Warning: Deleting a queued/processing task will not stop the "
        "background execution, but the result will be discarded."
    ),
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_analysis_task(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
) -> Response:
    """Delete an analysis task."""
    deleted = task_service.delete_task(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)