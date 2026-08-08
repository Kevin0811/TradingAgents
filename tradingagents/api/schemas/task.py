"""Task schemas for async analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from tradingagents.api.schemas.validators import validate_ticker_shape


class TaskStatus(str, Enum):
    """Status of an analysis task."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskCreateRequest(BaseModel):
    """Request body for creating an analysis task."""

    ticker: str = Field(..., description="Ticker symbol to analyze (e.g. 'AAPL', 'BTC')")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    asset_type: str = Field(
        default="stock",
        description="Asset type: 'stock' or 'crypto'",
    )
    selected_analysts: list[str] = Field(
        default=["market", "social", "news", "fundamentals"],
        description="List of analysts to include",
    )
    debug: bool = Field(default=False, description="Enable debug mode with verbose output")

    _validate_ticker = field_validator("ticker")(validate_ticker_shape)


class TaskResponse(BaseModel):
    """Response body for an analysis task."""

    task_id: str
    status: TaskStatus
    ticker: str
    trade_date: str
    asset_type: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    message: Optional[str] = None