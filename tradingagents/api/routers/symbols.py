"""Router for the supported-symbols list (``/symbols``)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse

from tradingagents.api.dependencies import get_symbol_catalog
from tradingagents.api.domain.services.symbol_catalog import SymbolCatalog
from tradingagents.api.domain.symbols import MARKETS, list_key
from tradingagents.api.schemas.enums import SymbolMarket, SymbolType
from tradingagents.api.schemas.symbols import (
    SymbolEntryResponse,
    SymbolListResponse,
    SymbolListUnavailableResponse,
    SymbolNotFoundResponse,
    SymbolRefreshResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/symbols",
    tags=["Symbols"],
)


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
        "older than the cache TTL (it is still served, and refreshed in the background)."
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
        total=total,
        limit=limit,
        offset=offset,
        symbols=[SymbolEntryResponse(**e.to_dict()) for e in page],
    )


@router.post(
    "/refresh",
    response_model=SymbolRefreshResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Refresh supported-symbols lists",
    description=(
        "Queue a background refresh from Yahoo Finance, regardless of the lists' age. "
        "Returns 202 immediately; markets refresh one at a time. `market` limits it to "
        "one market; omit it to refresh all. A failed refresh keeps the previous list. "
        "Poll `GET /symbols?market=...&limit=0` and watch `fetched_at` / `refreshing`."
    ),
)
async def refresh_symbols(
    market: SymbolMarket | None = Query(default=None, description="Market; omit for all"),
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
) -> SymbolRefreshResponse:
    """Queue a background refresh of one or all markets."""
    markets = [market.value] if market else list(MARKETS)
    queued = catalog.request_refresh(markets)
    logger.info("Symbol list refresh requested for %s", ", ".join(queued))
    return SymbolRefreshResponse(queued=queued)


@router.get(
    "/{symbol}",
    response_model=SymbolEntryResponse,
    summary="Look up one supported symbol",
    description=(
        "Exact lookup across every market's list. The symbol is normalised first "
        "(case, broker aliases such as 'BTCUSD' -> 'BTC-USD', 'USDJPY' -> 'JPY=X').\n\n"
        "- **200**: the entry.\n"
        "- **404**: not on the list; `suggestions` holds up to 3 close symbols from "
        "the list of the market the symbol belongs to. `market` is null when no list "
        "covers it (e.g. '0700.HK', '^GSPC'); it may still work for analysis.\n"
        "- **503**: the list of the symbol's market is not loaded yet, so it cannot be "
        "checked (`status` is 'loading' or 'unavailable')."
    ),
    responses={
        404: {"model": SymbolNotFoundResponse},
        503: {"model": SymbolListUnavailableResponse},
    },
)
def get_symbol(
    symbol: str,
    catalog: SymbolCatalog = Depends(get_symbol_catalog),
):
    """Look up one symbol, with suggestions when it is unknown."""
    key = list_key(symbol)
    entry = catalog.get(key)
    if entry is not None:
        return SymbolEntryResponse(**entry.to_dict())
    market = catalog.market_for(key)
    if market is not None and not catalog.is_loaded(market):
        body = SymbolListUnavailableResponse(
            error="Symbol list not loaded",
            detail=f"The {market} symbol list is not loaded yet, so '{key}' cannot be checked.",
            symbol=key,
            market=market,
            status=catalog.status(market),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body.model_dump(mode="json"),
        )
    body = SymbolNotFoundResponse(
        error="Symbol not found",
        detail=(
            f"'{key}' is not on the {market} supported-symbols list."
            if market
            else f"'{key}' belongs to a market the supported-symbols list does not cover."
        ),
        symbol=key,
        market=market,
        suggestions=catalog.suggest(key, market) if market else [],
    )
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=body.model_dump(mode="json"))
