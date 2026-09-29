"""Task schemas for async analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from tradingagents.api.schemas.overrides import RunOverridesMixin
from tradingagents.api.schemas.validators import (
    normalize_asset_type,
    validate_ticker_shape,
    validate_ticker_supported,
)


class TaskStatus(str, Enum):
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
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
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

    _validate_ticker = field_validator("ticker")(validate_ticker_shape)
    _normalize_asset_type = field_validator("asset_type", mode="before")(normalize_asset_type)
    _check_ticker_supported = model_validator(mode="after")(validate_ticker_supported)


class TaskResponse(BaseModel):
    """Response body for an analysis task."""

    task_id: str
    status: TaskStatus
    ticker: str
    trade_date: str
    asset_type: str
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict | None = None
    message: str | None = None
    cancel_requested: bool = Field(
        default=False,
        description=(
            "True once the task was cancelled (POST /analyze/tasks/{task_id}/cancel). "
            "A processing task keeps status 'processing' with this flag until it stops "
            "at its next LLM call, then ends as 'cancelled' -- or, when the cancel "
            "raced the run's end, as 'completed' or 'failed'."
        ),
    )
