"""Schemas for the supported-symbols endpoints (``/symbols``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tradingagents.api.domain.services.symbol_settings import SETTING_SPECS, validate_setting
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
    next_refresh_after: datetime | None = Field(
        None,
        description=(
            "Set while a stale list (or one a miss asked to refresh) is served and waits "
            "for the refresh window: when that window next opens (UTC). Null when nothing "
            "is waiting, when the window is open now, or when there is no window"
        ),
    )
    refresh_window: str | None = Field(
        None,
        description=(
            "The optional window automatic refreshes of stale lists also wait for, e.g. "
            "'02:00-06:00 Asia/Taipei'; null when none is set. Missing lists and POST "
            "/symbols/refresh ignore it"
        ),
    )
    waiting_for: str | None = Field(
        None,
        description=(
            "Why a stale list is not being refreshed yet: 'activity' (TradingAgents is "
            "busy), 'window', 'auto_refresh_off', 'retry_limit' or 'tick' (due at the next "
            "tick); null when nothing is waiting"
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
        description=(
            "Markets not queued: fetched within the 30-minute cooldown (pass force=true), "
            "or, with force=true, already forced within the last 5 minutes"
        ),
    )
    skip_reasons: dict[SymbolMarket, str] = Field(
        default_factory=dict,
        description="Why each skipped market was skipped: 'cooldown' or 'forced_recently'",
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


# ---------------------------------------------------------------------------
# GET / PUT /symbols/settings
# ---------------------------------------------------------------------------

_SOURCE_DESCRIPTION = "'default', 'env' (a TRADINGAGENTS_* variable), 'config' or 'api'"


class SymbolSettingsValues(BaseModel):
    """Effective refresher settings."""

    ttl_days: float = Field(..., description="Age (days) after which a list is stale")
    auto_refresh: bool = Field(..., description="Refresh missing/stale lists automatically")
    idle_grace_minutes: float = Field(
        ..., description="Minutes without Yahoo-backed activity before TradingAgents counts as idle"
    )
    refresh_window: str = Field(
        ..., description="Optional daily window 'HH:MM-HH:MM' for automatic refreshes; '' = off"
    )
    refresh_timezone: str = Field(..., description="IANA time zone of the refresh window")
    page_delay_seconds: float = Field(..., description="Pause between Yahoo requests")


class SymbolSettingsSources(BaseModel):
    """Where each effective value comes from."""

    ttl_days: str = Field(..., description=_SOURCE_DESCRIPTION)
    auto_refresh: str
    idle_grace_minutes: str
    refresh_window: str
    refresh_timezone: str
    page_delay_seconds: str


class SymbolMarketRefreshState(BaseModel):
    """Refresh state of one market's list."""

    status: SymbolListStatus
    stale: bool
    refreshing: bool = Field(..., description="Queued or being fetched")
    paused: bool = Field(..., description="Being fetched, but paused while TradingAgents is busy")
    waiting_for: str | None = Field(
        None,
        description=(
            "Why a stale list is not refreshed yet: 'activity', 'window', "
            "'auto_refresh_off', 'retry_limit' or 'tick'; null when nothing is waiting"
        ),
    )
    next_refresh_after: datetime | None = Field(
        None, description="When the refresh window next opens, while the list waits for it"
    )
    last_error: str | None = None


class SymbolRefreshState(BaseModel):
    """Read-only refresher state."""

    idle: bool = Field(..., description="Whether TradingAgents counts as idle now")
    active_tasks: int = Field(..., description="Analysis tasks pending, queued or processing")
    data_requests_in_flight: int = Field(
        ..., description="Yahoo-backed requests (/data, /analysts, sync /analyze) running now"
    )
    last_data_request_at: datetime | None = Field(
        None, description="When the last Yahoo-backed request started or ended (UTC)"
    )
    window_open: bool = Field(..., description="True when no window is set or now is inside it")
    markets: dict[SymbolMarket, SymbolMarketRefreshState]


class SymbolSettingsResponse(BaseModel):
    """200 body for ``GET`` / ``PUT /symbols/settings``."""

    values: SymbolSettingsValues
    sources: SymbolSettingsSources
    state: SymbolRefreshState


def _limits(name: str) -> dict[str, float]:
    spec = SETTING_SPECS[name]
    return {"ge": spec.minimum, "le": spec.maximum}


class SymbolSettingsUpdate(BaseModel):
    """Body of ``PUT /symbols/settings``: a partial update.

    Omitted fields are left alone; a field set to null goes back to its
    default / env / config value.
    """

    model_config = ConfigDict(extra="forbid")

    ttl_days: float | None = Field(None, **_limits("ttl_days"), description="0.5-90")
    auto_refresh: bool | None = None
    idle_grace_minutes: float | None = Field(
        None, **_limits("idle_grace_minutes"), description="0-240"
    )
    refresh_window: str | None = Field(
        None, max_length=20, description="'HH:MM-HH:MM' (may wrap midnight) or '' for none"
    )
    refresh_timezone: str | None = Field(
        None, max_length=64, description="IANA time zone, e.g. 'Asia/Taipei'"
    )
    page_delay_seconds: float | None = Field(
        None, **_limits("page_delay_seconds"), description="0.5-10"
    )

    @field_validator("refresh_window", "refresh_timezone")
    @classmethod
    def _check_text(cls, value: str | None, info) -> str | None:
        return None if value is None else validate_setting(info.field_name, value)
