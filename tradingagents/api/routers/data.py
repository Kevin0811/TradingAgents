"""Router for data query endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from tradingagents.api.domain.services.market_data_service import MarketDataService
from tradingagents.api.dependencies import get_market_data_service
from tradingagents.api.schemas.response import (
    FundamentalsResponse,
    GlobalNewsResponse,
    IndicatorsResponse,
    MacroIndicatorsResponse,
    NewsResponse,
    PredictionMarketsResponse,
    PriceHistoryResponse,
    StockDataResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/data",
    tags=["Data"],
)

# Repeated in several endpoint descriptions: the API resolves a bare crypto
# base as the coin, which differs from the rest of the codebase.
_TICKER_NOTE = (
    "`ticker`: Stock/crypto symbol. Broker aliases are resolved automatically "
    "(`XAUUSD` -> `GC=F`, `BTCUSD` -> `BTC-USD`). A bare crypto base such as "
    "`BTC` is read as the coin (`BTC-USD`); use the exchange symbol if you "
    "want a same-named equity or ETF."
)


@router.get(
    "/stock/{ticker}",
    response_model=StockDataResponse,
    summary="Stock OHLCV Data",
    description=(
        "Get historical OHLCV stock data for a ticker, as one record per bar.\n\n"
        "**Parameters:**\n"
        f"- {_TICKER_NOTE}\n"
        "- `trade_date`: End of the window, YYYY-MM-DD\n"
        "- `period`: Window length in calendar days, inclusive of `trade_date` "
        "(1-3650, default: 365). `period=1` requests that single day.\n\n"
        "Bars always carry `date` and OHLCV fields; vendor-specific extras "
        "(`dividends`, `split_coefficient`, ...) are passed through as-is."
    ),
)
def get_stock_data(
    ticker: str,
    trade_date: str = Query(..., description="End of the window, YYYY-MM-DD"),
    period: int = Query(default=365, ge=1, le=3650, description="Window length in days"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> StockDataResponse:
    """Get stock OHLCV data."""
    return StockDataResponse(
        **market_data_service.get_stock_data(ticker, trade_date, period=period)
    )


@router.get(
    "/indicators/{ticker}",
    response_model=IndicatorsResponse,
    summary="Technical Indicators",
    description=(
        "Get technical indicators for a ticker, as one series per indicator.\n\n"
        "**Parameters:**\n"
        f"- {_TICKER_NOTE}\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `indicator_names`: Comma-separated indicator names; omit for a "
        "default set. Available: 'close_50_sma', 'close_200_sma', "
        "'close_10_ema', 'macd', 'macds', 'macdh', 'rsi', 'boll', 'boll_ub', "
        "'boll_lb', 'atr', 'vwma', 'mfi' ('mfi' is yfinance-only)\n"
        "- `look_back_days`: How many days of values to return per indicator "
        "(1-365, default: 30)\n\n"
        "A point's `value` is null on non-trading days, with the vendor's "
        "explanation in `note`. Unsupported indicator names are reported in "
        "`errors` rather than failing the whole request."
    ),
)
def get_indicators(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    indicator_names: str | None = Query(
        default=None,
        description="Comma-separated indicator names (e.g. 'rsi,macd,close_50_sma')",
    ),
    look_back_days: int = Query(default=30, ge=1, le=365, description="Days of values"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> IndicatorsResponse:
    """Get technical indicators."""
    names = [n.strip() for n in indicator_names.split(",") if n.strip()] if indicator_names else None
    return IndicatorsResponse(
        **market_data_service.get_indicators(
            ticker, trade_date, indicator_names=names, look_back_days=look_back_days
        )
    )


@router.get(
    "/history/{ticker}",
    response_model=PriceHistoryResponse,
    summary="Price History for Charting",
    description=(
        "Get OHLCV bars and technical indicators over one window, aligned to "
        "the same dates.\n\n"
        "Prefer this over combining `/data/stock` with `/data/indicators`: "
        "those two describe their windows with different parameters, so a "
        "caller that wants both has to re-derive the overlap.\n\n"
        "**Parameters:**\n"
        f"- {_TICKER_NOTE}\n"
        "- `start_date`: Start of the window, YYYY-MM-DD (inclusive)\n"
        "- `end_date`: End of the window, YYYY-MM-DD (inclusive)\n"
        "- `indicators`: Comma-separated indicator names. Omit for the chart "
        "default set; pass an empty value for bars only. Available: "
        "'close_50_sma', 'close_200_sma', 'close_10_ema', 'macd', 'macds', "
        "'macdh', 'rsi', 'boll', 'boll_ub', 'boll_lb', 'atr', 'vwma', 'mfi' "
        "('mfi' is yfinance-only)\n\n"
        "The window may not exceed 1825 days, which is as far back as the "
        "indicator source reaches. Indicator points carry a null `value` on "
        "non-trading days; unsupported names are reported in `errors` rather "
        "than failing the request."
    ),
)
def get_price_history(
    ticker: str,
    start_date: str = Query(..., description="Start of the window, YYYY-MM-DD"),
    end_date: str = Query(..., description="End of the window, YYYY-MM-DD"),
    indicators: str | None = Query(
        default=None,
        description="Comma-separated indicator names; empty for bars only",
    ),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> PriceHistoryResponse:
    """Get aligned price bars and indicator series."""
    # None and "" mean different things here: no parameter asks for the chart
    # default set, an empty one asks for no indicators at all.
    names = (
        None
        if indicators is None
        else [n.strip() for n in indicators.split(",") if n.strip()]
    )
    return PriceHistoryResponse(
        **market_data_service.get_price_history(
            ticker, start_date, end_date, indicator_names=names
        )
    )


@router.get(
    "/fundamentals/{ticker}",
    response_model=FundamentalsResponse,
    summary="Fundamentals Data",
    description=(
        "Get fundamental data (overview, balance sheet, cashflow, income "
        "statement) for a ticker.\n\n"
        "**Parameters:**\n"
        f"- {_TICKER_NOTE}\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `report_type`: 'all' (default), 'overview', 'balance_sheet', "
        "'cashflow', 'income_statement'\n"
        "- `freq`: Statement frequency, 'annual' or 'quarterly' "
        "(default: 'quarterly'). Ignored for 'overview'."
    ),
)
def get_fundamentals(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    report_type: str = Query(
        default="all",
        description="all, overview, balance_sheet, cashflow, income_statement",
    ),
    freq: str = Query(default="quarterly", description="annual or quarterly"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> FundamentalsResponse:
    """Get fundamentals data."""
    data = market_data_service.get_fundamentals(
        ticker, trade_date, report_type=report_type, freq=freq
    )
    return FundamentalsResponse(ticker=ticker, data=data)


@router.get(
    "/news/{ticker}",
    response_model=NewsResponse,
    summary="Ticker News",
    description=(
        "Get news articles for a ticker.\n\n"
        "**Parameters:**\n"
        f"- {_TICKER_NOTE}\n"
        "- `trade_date`: End of the window, YYYY-MM-DD\n"
        "- `look_back_days`: How many days before `trade_date` to include "
        "(1-365, default: 7)\n\n"
        "For market-wide news use `/data/global-news`."
    ),
)
def get_news(
    ticker: str,
    trade_date: str = Query(..., description="End of the window, YYYY-MM-DD"),
    look_back_days: int = Query(default=7, ge=1, le=365, description="Window length"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> NewsResponse:
    """Get news data for a single ticker."""
    news_items = market_data_service.get_news(
        ticker, trade_date, look_back_days=look_back_days
    )
    return NewsResponse(ticker=ticker, news_items=news_items)


@router.get(
    "/global-news",
    response_model=GlobalNewsResponse,
    summary="Global Market News",
    description=(
        "Get global/macro market news. Not scoped to a ticker or country.\n\n"
        "**Parameters:**\n"
        "- `trade_date`: End of the window, YYYY-MM-DD\n"
        "- `look_back_days`: Window length; omit to use the configured "
        "`global_news_lookback_days`\n"
        "- `limit`: Max articles; omit to use the configured "
        "`global_news_article_limit`"
    ),
)
def get_global_news(
    trade_date: str = Query(..., description="End of the window, YYYY-MM-DD"),
    look_back_days: int | None = Query(default=None, ge=1, le=365),
    limit: int | None = Query(default=None, ge=1, le=100),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> GlobalNewsResponse:
    """Get global market news."""
    news_items = market_data_service.get_global_news(
        trade_date, look_back_days=look_back_days, limit=limit
    )
    return GlobalNewsResponse(trade_date=trade_date, news_items=news_items)


@router.get(
    "/macro-indicators",
    response_model=MacroIndicatorsResponse,
    summary="Macro Indicators",
    description=(
        "Get a macroeconomic time series from FRED (Federal Reserve Economic "
        "Data).\n\n"
        "**Parameters:**\n"
        "- `indicator`: Friendly alias ('cpi', 'core_pce', 'unemployment', "
        "'fed_funds_rate', '10y_treasury', 'yield_curve', 'real_gdp', 'vix') "
        "or a raw FRED series ID such as 'CPIAUCSL'\n"
        "- `trade_date`: End of the window, YYYY-MM-DD\n"
        "- `look_back_days`: Window length; omit for a 1-year window"
    ),
)
def get_macro_indicators(
    indicator: str = Query(..., description="Alias or raw FRED series ID"),
    trade_date: str = Query(..., description="End of the window, YYYY-MM-DD"),
    look_back_days: int | None = Query(default=None, ge=1, le=3650),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> MacroIndicatorsResponse:
    """Get a macro indicator series."""
    data = market_data_service.get_macro_indicators(
        indicator, trade_date, look_back_days=look_back_days
    )
    return MacroIndicatorsResponse(indicator=indicator, data=data)


@router.get(
    "/prediction-markets",
    response_model=PredictionMarketsResponse,
    summary="Prediction Markets",
    description=(
        "Get market-implied probabilities for forward-looking events from "
        "prediction markets (Polymarket). Searched by topic, not by ticker.\n\n"
        "**Parameters:**\n"
        "- `topic`: Event keyword(s), e.g. 'Fed rate cut', 'recession 2026'\n"
        "- `limit`: Max markets to return; omit for the vendor default"
    ),
)
def get_prediction_markets(
    topic: str = Query(..., description="Event topic/keyword"),
    limit: int | None = Query(default=None, ge=1, le=50),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> PredictionMarketsResponse:
    """Get prediction market data."""
    data = market_data_service.get_prediction_markets(topic, limit=limit)
    return PredictionMarketsResponse(topic=topic, markets=data)
