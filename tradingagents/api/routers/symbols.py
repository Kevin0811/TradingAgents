"""Router for the supported-symbols list (``/symbols``)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tradingagents.api.dependencies import get_symbol_catalog
from tradingagents.api.domain.services.symbol_catalog import SymbolCatalog
from tradingagents.api.domain.symbols import MARKETS
from tradingagents.api.schemas.enums import AssetType, SymbolMarket, SymbolType
from tradingagents.api.schemas.symbols import (
    SymbolCheckResponse,
    SymbolEntryResponse,
    SymbolListResponse,
    SymbolNotFoundResponse,
    SymbolRefreshResponse,
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
        "older than the cache TTL (it is still served, and refreshed in the background). "
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
        "Queue a background refresh from Yahoo Finance. Returns 202 immediately; "
        "markets refresh one at a time. `market` limits it to one market; omit it to "
        "refresh all. A market whose list was fetched within the last 30 minutes is "
        "listed in `skipped` instead of `queued`, unless `force=true`. A failed "
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
        description="Ignore the 30-minute cooldown and accept a shrunken list",
    ),
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
) -> SymbolRefreshResponse:
    """Queue a background refresh of one or all markets."""
    markets = [market.value] if market else list(MARKETS)
    queued, skipped = catalog.request_manual_refresh(markets, force=force)
    logger.info(
        "Symbol list refresh requested: queued %s; skipped (cooldown) %s",
        ", ".join(queued) or "none",
        ", ".join(skipped) or "none",
    )
    return SymbolRefreshResponse(queued=queued, skipped=skipped)


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
