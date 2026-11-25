"""Task schemas for async analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from tradingagents.api.schemas.enums import InvestmentHorizon
from tradingagents.api.schemas.overrides import RunOverridesMixin
from tradingagents.api.schemas.request import (
    INVESTMENT_HORIZON_DESCRIPTION,
    TRADE_DATE_DESCRIPTION,
)
from tradingagents.api.schemas.token_usage import TokenUsage
from tradingagents.api.schemas.validators import (
    normalize_asset_type,
    validate_ticker_shape,
    validate_ticker_supported,
    validate_trade_date,
)


class TaskStatus(StrEnum):
    """Status of an analysis task."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# A task in one of these states has ended and never changes again.
FINISHED_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})


class TaskCreateRequest(RunOverridesMixin):
    """Request body for creating an analysis task."""

    ticker: str = Field(..., description="Ticker symbol to analyze (e.g. 'AAPL', 'BTC')")
    trade_date: str = Field(..., description=TRADE_DATE_DESCRIPTION)
    asset_type: str = Field(
        default="stock",
        description="Asset type: 'stock' or 'crypto'",
    )
    selected_analysts: list[str] = Field(
        default=["market", "social", "news", "fundamentals"],
        description=(
            "List of analysts to include: 'market', 'social', 'news', 'fundamentals' "
            "('sentiment' is accepted as an alias of 'social')"
        ),
    )
    debug: bool = Field(default=False, description="Enable debug mode with verbose output")
    investment_horizon: InvestmentHorizon | None = Field(
        default=None,
        description=INVESTMENT_HORIZON_DESCRIPTION,
    )

    _validate_ticker = field_validator("ticker")(validate_ticker_shape)
    _validate_trade_date = field_validator("trade_date")(validate_trade_date)
    _normalize_asset_type = field_validator("asset_type", mode="before")(normalize_asset_type)
    _check_ticker_supported = model_validator(mode="after")(validate_ticker_supported)


class TaskResponse(BaseModel):
    """Response body for an analysis task."""

    task_id: str
    status: TaskStatus
    ticker: str
    trade_date: str
    asset_type: str
    investment_horizon: InvestmentHorizon | None = Field(
        default=None,
        description=(
            "The request's investment_horizon (null when none was sent). Two "
            "requests with different horizons are different tasks; the result "
            "echoes it too."
        ),
    )
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    error_code: str | None = Field(
        default=None,
        description=(
            "Why the task failed, as a stable code; set only when status is 'failed' "
            "(null otherwise). 'error' keeps the free-text detail. One of: "
            "TA_LLM_TIMEOUT (an LLM call timed out), TA_LLM_UNREACHABLE (the LLM "
            "provider could not be reached), TA_LLM_AUTH (the provider rejected the "
            "key: 401/403), TA_LLM_MODEL_NOT_FOUND (the configured model does not "
            "exist: 404, or Ollama's 'model ... not found'), TA_LLM_RATE_LIMITED "
            "(429), TA_LLM_RESPONSE_INVALID (the model's reply did not validate or "
            "parse), TA_DATA_VENDOR_FAILED (a market data vendor failed), "
            "TA_RUN_FAILED (anything else). New codes may be added; treat an "
            "unknown one as TA_RUN_FAILED."
        ),
    )
    result: dict | None = None
    message: str | None = None
    cancel_reason: str | None = Field(
        default=None,
        description=(
            "Why the task was cancelled, as a stable code; set only when status is "
            "'cancelled' (null otherwise). 'message' keeps the free text. One of: "
            "TA_CANCELLED_BEFORE_START (cancelled while pending or queued; it never "
            "ran), TA_CANCELLED_WHILE_RUNNING (cancelled while processing; it stopped "
            "at its next LLM call)."
        ),
    )
    cancel_requested: bool = Field(
        default=False,
        description=(
            "True once the task was cancelled (POST /analyze/tasks/{task_id}/cancel). "
            "A processing task keeps status 'processing' with this flag until it stops "
            "at its next LLM call, then ends as 'cancelled' -- or, when the cancel "
            "raced the run's end, as 'completed' or 'failed'."
        ),
    )
    token_usage: TokenUsage | None = Field(
        default=None,
        description=(
            "Token usage of the run's LLM calls, per model tier and model. Set when the "
            "run ends, whether it completed, failed or was cancelled (then with "
            "partial: true); null until then, and for a task cancelled before it ran."
        ),
    )
