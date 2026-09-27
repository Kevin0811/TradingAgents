"""Router for the supported-symbols list (``/symbols``)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tradingagents.api.dependencies import (
    get_activity_monitor,
    get_symbol_catalog,
    get_symbol_settings,
)
from tradingagents.api.domain.services.refresh_activity import ActivityMonitor
from tradingagents.api.domain.services.symbol_catalog import (
    SKIP_COOLDOWN,
    SKIP_FORCED_RECENTLY,
    SymbolCatalog,
)
from tradingagents.api.domain.services.symbol_settings import SymbolSettings
from tradingagents.api.domain.symbols import MARKETS
from tradingagents.api.schemas.enums import AssetType, SymbolMarket, SymbolType
from tradingagents.api.schemas.symbols import (
    SymbolCheckResponse,
    SymbolEntryResponse,
    SymbolListResponse,
    SymbolMarketRefreshState,
    SymbolNotFoundResponse,
    SymbolRefreshResponse,
    SymbolRefreshState,
    SymbolSettingsResponse,
    SymbolSettingsSources,
    SymbolSettingsUpdate,
    SymbolSettingsValues,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/symbols",
    tags=["Symbols"],
)

_ASSET_TYPES = tuple(a.value for a in AssetType)


def _entry(entry) -> SymbolEntryResponse | None:
    return SymbolEntryResponse(**entry.to_dict()) if entry is not None else None


@router.get(
    "",
    response_model=SymbolListResponse,
    summary="List or search supported symbols",
    description=(
        "Page through, or search, one market's supported-symbols list: the Yahoo "
        "Finance symbols TradingAgents accepts for analysis. The list is served from "
        "memory and never triggers a synchronous network call.\n\n"
        "**Parameters:**\n"
        "- `market`: 'tw', 'us', 'jp', 'crypto' or 'fx' (required)\n"
        "- `q`: case-insensitive symbol prefix or name substring; an exact symbol match "
        "comes first, then symbol-prefix matches, then name matches\n"
        "- `type`: 'equity', 'etf', 'crypto' or 'currency'\n"
        "- `limit` (0-1000, default 50), `offset` (default 0)\n\n"
        "Always 200. When the market has no list yet, `symbols` is empty and `status` "
        "is 'loading' (a background fetch is queued or running) or 'unavailable' (no "
        "fetch in progress, e.g. the last one failed); `stale` is true once the list is "
        "older than the cache TTL (it is still served, and refreshed in the background once "
        "TradingAgents is idle and, if one is set, inside `refresh_window`; `waiting_for` "
        "says what it waits for, `next_refresh_after` when the window opens). "
        "`last_error` says why the last refresh failed or was refused."
    ),
)
def list_symbols(
    market: SymbolMarket = Query(..., description="Market to list"),
    q: str | None = Query(default=None, max_length=100, description="Symbol prefix or name"),
    symbol_type: SymbolType | None = Query(default=None, alias="type", description="Type"),
    limit: int = Query(default=50, ge=0, le=1000, description="Page size"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
) -> SymbolListResponse:
    """List or search one market's supported symbols."""
    page, total = catalog.search(
        market.value,
        q=q,
        symbol_type=symbol_type.value if symbol_type else None,
        limit=limit,
        offset=offset,
    )
    symbol_list = catalog.get_list(market.value)
    return SymbolListResponse(
        market=market,
        status=catalog.status(market.value),
        fetched_at=symbol_list.fetched_at if symbol_list else None,
        stale=catalog.is_stale(market.value),
        refreshing=catalog.is_refreshing(market.value),
        last_error=catalog.last_error(market.value),
        next_refresh_after=catalog.next_refresh_after(market.value),
        refresh_window=catalog.refresh_window.describe() if catalog.refresh_window else None,
        waiting_for=catalog.waiting_for(market.value),
        total=total,
        limit=limit,
        offset=offset,
        symbols=[SymbolEntryResponse(**e.to_dict()) for e in page],
    )


@router.get(
    "/check",
    response_model=SymbolCheckResponse,
    summary="Check a ticker the way analyze requests are checked",
    description=(
        "Return exactly the decision `POST /analyze` and `POST /analyze/tasks` make for "
        "this ticker (the same code path), without submitting anything. Always 200; 422 "
        "only for invalid query parameters.\n\n"
        "- `ticker` (required), `asset_type`: 'stock' (default) or 'crypto', "
        "case-insensitive\n\n"
        "An analyze request is rejected (422 `ticker_not_supported`) exactly when "
        "`supported` is false and `enforced` is true. `enforced` is true for tw and jp; "
        "a miss in us, crypto or fx is allowed and `suggestions` is only a hint. "
        "`supported` is null when no list covers the ticker (`status` 'uncovered', e.g. "
        "'0700.HK', '^GSPC') or its market's list is not loaded ('loading' / "
        "'unavailable'); such tickers get the shape check only. A suffix-less TW/JP code "
        "('2330', '2881A', '130A') is reported unsupported and enforced, with the listed "
        "'.TW' / '.TWO' / '.T' symbol in `suggestions` when there is one."
    ),
)
def check_symbol(
    ticker: str = Query(..., min_length=1, max_length=64, description="Ticker to check"),
    asset_type: str | None = Query(default=None, description="'stock' (default) or 'crypto'"),
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
) -> SymbolCheckResponse:
    """Return the analyze validator's decision for one ticker."""
    errors = []
    if not ticker.strip():
        errors.append(
            {
                "type": "value_error",
                "loc": ("query", "ticker"),
                "msg": "Value error, ticker is required",
                "input": ticker,
            }
        )
    normalized_type = asset_type.strip().lower() if asset_type is not None else None
    if normalized_type is not None and normalized_type not in _ASSET_TYPES:
        errors.append(
            {
                "type": "enum",
                "loc": ("query", "asset_type"),
                "msg": "Input should be " + " or ".join(f"'{a}'" for a in _ASSET_TYPES),
                "input": asset_type,
                "ctx": {"expected": " or ".join(f"'{a}'" for a in _ASSET_TYPES)},
            }
        )
    if errors:
        raise RequestValidationError(errors)

    decision = catalog.decide(ticker, normalized_type)
    return SymbolCheckResponse(
        ticker=decision.ticker,
        normalized=decision.normalized,
        market=decision.market,
        supported=decision.supported,
        enforced=decision.enforced,
        status=decision.status,
        suggestions=list(decision.suggestions),
        entry=_entry(decision.entry),
        last_error=decision.last_error,
    )


@router.post(
    "/refresh",
    response_model=SymbolRefreshResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Refresh supported-symbols lists",
    description=(
        "Queue a background refresh from Yahoo Finance, right away: it neither waits for "
        "TradingAgents to be idle nor for the refresh window, and does not pause while "
        "busy. Returns 202 immediately; "
        "markets refresh one at a time, manual ones ahead of automatic ones (an automatic "
        "refresh paused for activity makes way and resumes later; one of the same market "
        "just carries on without pausing). `market` limits it to one market; omit it to "
        "refresh all. A market whose list was fetched within the last 30 minutes is "
        "listed in `skipped` instead of `queued` (`skip_reasons` 'cooldown'), unless "
        "`force=true`; a market already forced within the last 5 minutes is skipped even "
        "with `force=true` ('forced_recently'). A failed "
        "refresh keeps the previous list and sets its `last_error`, and so does one that "
        "looks truncated (under 80% of the previous entries, or an exchange/type group "
        "gone empty) unless `force=true`, which accepts such a shrink deliberately. A "
        "fetch that fails or comes back incomplete is never saved, forced or not. Poll `GET /symbols?market=...&limit=0` and watch `fetched_at` / "
        "`refreshing` / `last_error`."
    ),
)
async def refresh_symbols(
    market: SymbolMarket | None = Query(default=None, description="Market; omit for all"),
    force: bool = Query(
        default=False,
        description=(
            "Ignore the 30-minute cooldown and accept a shrunken list "
            "(at most once per 5 minutes per market)"
        ),
    ),
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
) -> SymbolRefreshResponse:
    """Queue a background refresh of one or all markets."""
    markets = [market.value] if market else list(MARKETS)
    queued, skipped = catalog.request_manual_refresh(markets, force=force)
    # Without force the only reason to skip is the cooldown; with it, the forced gap.
    reason = SKIP_FORCED_RECENTLY if force else SKIP_COOLDOWN
    logger.info(
        "Symbol list refresh requested%s: queued %s; skipped (%s) %s",
        " (forced)" if force else "",
        ", ".join(queued) or "none",
        reason,
        ", ".join(skipped) or "none",
    )
    return SymbolRefreshResponse(
        queued=queued, skipped=skipped, skip_reasons=dict.fromkeys(skipped, reason)
    )


def _settings_response(
    catalog: SymbolCatalog, settings: SymbolSettings, monitor: ActivityMonitor
) -> SymbolSettingsResponse:
    activity = monitor.state()
    markets = {}
    for market in MARKETS:
        markets[market] = SymbolMarketRefreshState(
            status=catalog.status(market),
            stale=catalog.is_stale(market),
            refreshing=catalog.is_refreshing(market),
            paused=catalog.is_paused(market),
            waiting_for=catalog.waiting_for(market),
            next_refresh_after=catalog.next_refresh_after(market),
            last_error=catalog.last_error(market),
        )
    return SymbolSettingsResponse(
        values=SymbolSettingsValues(**settings.values()),
        sources=SymbolSettingsSources(**settings.sources()),
        state=SymbolRefreshState(
            idle=activity["idle"],
            active_tasks=activity["active_tasks"],
            data_requests_in_flight=activity["data_requests_in_flight"],
            last_data_request_at=activity["last_data_request_at"],
            window_open=catalog.window_open(),
            markets=markets,
        ),
    )


_SETTINGS_NOTE = (
    "Like every endpoint of this API, this one has no authentication: keep the API on "
    "a trusted network."
)


@router.get(
    "/settings",
    response_model=SymbolSettingsResponse,
    summary="Get the symbol refresher settings",
    description=(
        "The effective refresher settings, where each comes from ('default', 'env', "
        "'config' or 'api'), and the read-only refresher state: whether TradingAgents "
        "counts as idle (no analysis task pending/queued/processing and no Yahoo-backed "
        "request - /data, /analysts, sync /analyze - within `idle_grace_minutes`), and "
        "per market what a stale list waits for.\n\n" + _SETTINGS_NOTE
    ),
)
def get_settings(
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
    settings: SymbolSettings = Depends(get_symbol_settings),
    monitor: ActivityMonitor = Depends(get_activity_monitor),
) -> SymbolSettingsResponse:
    """Return the effective refresher settings and state."""
    return _settings_response(catalog, settings, monitor)


@router.put(
    "/settings",
    response_model=SymbolSettingsResponse,
    summary="Change the symbol refresher settings",
    description=(
        "Partial update: omitted fields stay as they are; a field set to null goes back "
        "to its default / env / config value. Ranges: `ttl_days` 0.5-90, "
        "`idle_grace_minutes` 0-240, `page_delay_seconds` 0.5-10; `refresh_window` is "
        "'HH:MM-HH:MM' (may wrap midnight) or '' for none; `refresh_timezone` an IANA "
        "zone. Invalid input is a 422 and changes nothing. Changes apply at once (the "
        "refresher reads them on every tick and page) and are saved to `settings.json` "
        "in the symbols cache dir, so they survive a restart and win over env values.\n\n"
        + _SETTINGS_NOTE
    ),
)
def put_settings(
    body: SymbolSettingsUpdate,
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
    settings: SymbolSettings = Depends(get_symbol_settings),
    monitor: ActivityMonitor = Depends(get_activity_monitor),
) -> SymbolSettingsResponse:
    """Apply a partial settings update."""
    changes = {name: getattr(body, name) for name in body.model_fields_set}
    try:
        settings.update(changes)
    except ValueError as exc:
        field, message = exc.args if len(exc.args) == 2 else ("body", str(exc))
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body", field),
                    "msg": f"Value error, {message}",
                    "input": changes.get(field),
                }
            ]
        ) from exc
    return _settings_response(catalog, settings, monitor)


@router.get(
    "/{symbol}",
    response_model=SymbolEntryResponse,
    summary="Look up one supported symbol",
    description=(
        "Plain exact lookup across every market's list. The symbol is normalised first "
        "(case, broker aliases such as 'BTCUSD' -> 'BTC-USD', 'USDJPY' -> 'JPY=X').\n\n"
        "- **200**: the entry.\n"
        "- **404**: not on any loaded list. `suggestions` holds up to 3 close symbols "
        "from the list of the symbol's market (for a suffix-less TW/JP code, the listed "
        "'.TW' / '.TWO' / '.T' symbol), and `status` that list's load state: when it is "
        "'loading' or 'unavailable' the miss proves nothing yet; 'uncovered' means no "
        "list covers the symbol (e.g. '0700.HK', '^GSPC'), which may still work for "
        "analysis.\n\n"
        "Whether an analyze request would accept a ticker is `GET /symbols/check`."
    ),
    responses={404: {"model": SymbolNotFoundResponse}},
)
def get_symbol(
    symbol: str,
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
):
    """Look up one symbol, with suggestions when it is unknown."""
    found = catalog.lookup(symbol)
    if found.entry is not None:
        return SymbolEntryResponse(**found.entry.to_dict())
    if found.market is None:
        detail = f"'{found.symbol}' belongs to a market the supported-symbols list does not cover."
    elif found.status == "ready":
        detail = f"'{found.symbol}' is not on the {found.market} supported-symbols list."
    else:
        detail = (
            f"'{found.symbol}' was not found; the {found.market} symbol list is not loaded "
            f"yet ({found.status})."
        )
    body = SymbolNotFoundResponse(
        error="Symbol not found",
        detail=detail,
        symbol=found.symbol,
        market=found.market,
        status=found.status,
        suggestions=list(found.suggestions),
        last_error=found.last_error,
    )
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=body.model_dump(mode="json"))
