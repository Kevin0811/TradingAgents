"""Router for data query endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from tradingagents.api.domain.services.market_data_service import MarketDataService
from tradingagents.api.dependencies import get_market_data_service
from tradingagents.api.schemas.response import (
    FundamentalsResponse,
    IndicatorsResponse,
    MacroIndicatorsResponse,
    NewsResponse,
    PredictionMarketsResponse,
    StockDataResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/data",
    tags=["Data"],
)


@router.get(
    "/stock/{ticker}",
    response_model=StockDataResponse,
    summary="Stock OHLCV Data",
    description=(
        "Get historical OHLCV stock data for a ticker.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `period`: Number of days of historical data (1-3650, default: 365)"
    ),
)
async def get_stock_data(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    period: int = Query(default=365, ge=1, le=3650, description="Days of historical data"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> StockDataResponse:
    """Get stock OHLCV data."""
    data = market_data_service.get_stock_data(ticker, trade_date, period=period)
    return StockDataResponse(ticker=ticker, data=data)


@router.get(
    "/indicators/{ticker}",
    response_model=IndicatorsResponse,
    summary="Technical Indicators",
    description=(
        "Get technical indicators for a ticker.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `indicator_names`: Comma-separated indicator names. "
        "Available: 'rsi', 'macd', 'signal', 'histogram', 'close_50_sma', "
        "'close_200_sma', 'close_10_sma', 'close_20_sma', 'bollinger_upper', "
        "'bollinger_middle', 'bollinger_lower', 'volume', 'atr', 'adx', "
        "'stoch_k', 'stoch_d'"
    ),
)
async def get_indicators(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    indicator_names: str | None = Query(
        default=None,
        description="Comma-separated indicator names (e.g. 'rsi,macd,close_50_sma')",
    ),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> IndicatorsResponse:
    """Get technical indicators."""
    names = [n.strip() for n in indicator_names.split(",")] if indicator_names else None
    data = market_data_service.get_indicators(ticker, trade_date, indicator_names=names)
    return IndicatorsResponse(ticker=ticker, indicators=data)


@router.get(
    "/fundamentals/{ticker}",
    response_model=FundamentalsResponse,
    summary="Fundamentals Data",
    description=(
        "Get fundamental data (financials, balance sheet, cashflow) for a ticker.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `report_type`: Type of fundamental data. "
        "Available: 'all', 'balance_sheet', 'cashflow', 'income_statement'"
    ),
)
async def get_fundamentals(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    report_type: str = Query(
        default="all",
        description="Type of fundamental data: all, balance_sheet, cashflow, income_statement",
    ),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> FundamentalsResponse:
    """Get fundamentals data."""
    data = market_data_service.get_fundamentals(ticker, trade_date, report_type=report_type)
    return FundamentalsResponse(ticker=ticker, data=data)


@router.get(
    "/news/{ticker}",
    response_model=NewsResponse,
    summary="News Data",
    description=(
        "Get news articles and events for a ticker.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `country`: Country code for global news. "
        "Available: 'US', 'CN', 'JP', 'GB', 'DE', 'FR', 'TW', 'KR', 'SG'"
    ),
)
async def get_news(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    country: str | None = Query(default=None, description="Country code for global news"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> NewsResponse:
    """Get news data."""
    news_items = market_data_service.get_news(ticker, trade_date, country=country)
    return NewsResponse(ticker=ticker, news_items=news_items)


@router.get(
    "/macro-indicators",
    response_model=MacroIndicatorsResponse,
    summary="Macro Indicators",
    description=(
        "Get macroeconomic indicators for a country.\n\n"
        "**Parameters:**\n"
        "- `country`: Country code. "
        "Available: 'US', 'CN', 'JP', 'GB', 'DE', 'FR', 'TW', 'KR', 'SG' (default: 'US')"
    ),
)
async def get_macro_indicators(
    country: str = Query(default="US", description="Country code"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> MacroIndicatorsResponse:
    """Get macro indicators."""
    data = market_data_service.get_macro_indicators(country=country)
    return MacroIndicatorsResponse(country=country, data=data)


@router.get(
    "/prediction-markets/{ticker}",
    response_model=PredictionMarketsResponse,
    summary="Prediction Markets",
    description=(
        "Get prediction market data for a ticker.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format"
    ),
)
async def get_prediction_markets(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> PredictionMarketsResponse:
    """Get prediction market data."""
    data = market_data_service.get_prediction_markets(ticker, trade_date)
    return PredictionMarketsResponse(ticker=ticker, markets=data)