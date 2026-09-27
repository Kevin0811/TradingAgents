"""Schemas for the supported-symbols endpoints (``/symbols``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
    last_error: str | None = Field(
        None,
        description=(
            "Why the last refresh failed or was refused (a truncated-looking fetch); null "
            "once a refresh succeeds. The previous list, if any, is still served."
        ),
    )
    total: int = Field(..., description="Number of matching entries before paging")
    limit: int
    offset: int
    symbols: list[SymbolEntryResponse]


_LAST_ERROR_DESCRIPTION = "The market list's last refresh error, if any (see GET /symbols)"


class SymbolNotFoundResponse(BaseModel):
    """404 body for ``GET /symbols/{symbol}``."""

    error: str
    detail: str
    symbol: str = Field(..., description="The normalised symbol that was looked up")
    market: SymbolMarket | None = Field(
        None, description="The market whose list was searched; null if no list covers it"
    )
    status: SymbolListStatus = Field(
        ...,
        description=(
            "Load state of that market's list: 'ready', 'loading', 'unavailable' (so the "
            "miss proves nothing yet), or 'uncovered' when no list covers the symbol"
        ),
    )
    suggestions: list[str] = Field(default_factory=list, description="Up to 3 closest symbols")
    last_error: str | None = Field(None, description=_LAST_ERROR_DESCRIPTION)


class SymbolCheckResponse(BaseModel):
    """200 body for ``GET /symbols/check``: the analyze validator's decision."""

    ticker: str = Field(..., description="The ticker as given (trimmed)")
    normalized: str = Field(
        ..., description="The list form checked, e.g. 'USDTWD' -> 'TWD=X', 'BTC' -> 'BTC-USD'"
    )
    market: SymbolMarket | None = Field(
        None, description="The market whose list decides; null when no list covers the ticker"
    )
    supported: bool | None = Field(
        ...,
        description=(
            "true: on the list. false: not on the loaded list (or a suffix-less TW/JP "
            "code). null: cannot tell (no list covers it, or the list is not loaded)"
        ),
    )
    enforced: bool = Field(
        ...,
        description=(
            "Whether supported=false rejects an analyze request (422 ticker_not_supported). "
            "true for tw and jp, false for us, crypto and fx, where a miss is only a hint"
        ),
    )
    status: SymbolListStatus = Field(
        ..., description="Load state of the market's list, or 'uncovered'"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Up to 3 listed symbols close to the ticker"
    )
    entry: SymbolEntryResponse | None = Field(
        None, description="The list entry when supported (null for a cross pair like EURJPY=X)"
    )
    last_error: str | None = Field(None, description=_LAST_ERROR_DESCRIPTION)


class SymbolRefreshResponse(BaseModel):
    """202 body for ``POST /symbols/refresh``."""

    queued: list[SymbolMarket] = Field(..., description="Markets queued or already refreshing")
    skipped: list[SymbolMarket] = Field(
        default_factory=list,
        description="Markets fetched within the refresh cooldown (pass force=true to refresh)",
    )


# ---------------------------------------------------------------------------
# 422 ticker_not_supported (POST /analyze, POST /analyze/tasks)
# ---------------------------------------------------------------------------


class TickerValidationErrorItem(BaseModel):
    """One item of a 422 ``detail`` list."""

    type: str = Field(..., description="'ticker_not_supported', or a standard pydantic type")
    loc: list[str | int]
    msg: str
    input: Any = None
    ctx: dict[str, Any] | None = Field(
        None,
        description=(
            "For ticker_not_supported: 'ticker' (normalised), 'market', 'suggestions' "
            "(up to 3 listed symbols) and 'hint' (the text appended to msg)"
        ),
    )


class TickerValidationErrorResponse(BaseModel):
    """422 body of the analyze endpoints."""

    detail: list[TickerValidationErrorItem]


TICKER_NOT_SUPPORTED_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {
        "model": TickerValidationErrorResponse,
        "description": (
            "Validation error. `type: ticker_not_supported` means the ticker is not on the "
            "loaded supported-symbols list of a market that enforces it (tw, jp), or is a "
            "suffix-less TW/JP code whose suffixed symbol is listed; `ctx.suggestions` "
            "names listed symbols. A miss in us, crypto or fx is not rejected. "
            "`GET /symbols/check` returns the same decision without submitting."
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "type": "ticker_not_supported",
                            "loc": ["body"],
                            "msg": (
                                "'2331.TW' is not on the supported tw symbol list (see GET "
                                "/symbols). Did you mean: 2330.TW, 2303.TW?"
                            ),
                            "input": {"ticker": "2331.TW", "trade_date": "2026-09-25"},
                            "ctx": {
                                "ticker": "2331.TW",
                                "market": "tw",
                                "suggestions": ["2330.TW", "2303.TW"],
                                "hint": " Did you mean: 2330.TW, 2303.TW?",
                            },
                        }
                    ]
                }
            }
        },
    }
}
