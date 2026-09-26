"""Schemas for the supported-symbols endpoints (``/symbols``)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from tradingagents.api.schemas.enums import SymbolListStatus, SymbolMarket, SymbolType


class SymbolEntryResponse(BaseModel):
    """One supported instrument, in Yahoo Finance symbol form."""

    symbol: str = Field(..., description="Yahoo symbol, e.g. '2330.TW', 'AAPL', 'BTC-USD', 'TWD=X'")
    name: str = Field(..., description="Yahoo long name, else short name ('' if Yahoo has none)")
    exchange: str = Field(..., description="Yahoo exchange code, e.g. 'TAI', 'TWO', 'NMS', 'JPX'")
    type: SymbolType
    market: SymbolMarket


class SymbolListResponse(BaseModel):
    """A page of one market's supported-symbols list."""

    market: SymbolMarket
    status: SymbolListStatus = Field(
        ...,
        description=(
            "'ready' when the list is loaded; 'loading' when there is no list yet and a "
            "background fetch is queued or running; 'unavailable' when there is no list "
            "and no fetch in progress (e.g. the last fetch failed)"
        ),
    )
    fetched_at: datetime | None = Field(None, description="When the list was fetched (UTC)")
    stale: bool = Field(..., description="True when the list is older than the cache TTL")
    refreshing: bool = Field(..., description="True while a refresh is queued or running")
    total: int = Field(..., description="Number of matching entries before paging")
    limit: int
    offset: int
    symbols: list[SymbolEntryResponse]


class SymbolNotFoundResponse(BaseModel):
    """404 body for ``GET /symbols/{symbol}``."""

    error: str
    detail: str
    symbol: str = Field(..., description="The normalised symbol that was looked up")
    market: SymbolMarket | None = Field(
        None, description="The market whose list was searched; null if no list covers it"
    )
    suggestions: list[str] = Field(default_factory=list, description="Up to 3 closest symbols")


class SymbolListUnavailableResponse(BaseModel):
    """503 body for ``GET /symbols/{symbol}`` when the relevant list is not loaded."""

    error: str
    detail: str
    symbol: str
    market: SymbolMarket
    status: SymbolListStatus


class SymbolRefreshResponse(BaseModel):
    """202 body for ``POST /symbols/refresh``."""

    queued: list[SymbolMarket] = Field(..., description="Markets queued or already refreshing")
