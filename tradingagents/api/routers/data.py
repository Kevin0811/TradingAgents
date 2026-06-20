"""Router for data query endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from tradingagents.api.schemas import (
    ErrorResponse,
    FundamentalsResponse,
    IndicatorsResponse,
    MacroIndicatorsResponse,
    NewsResponse,
    PredictionMarketsResponse,
    StockDataResponse,
)
from tradingagents.api.services import (
    fetch_fundamentals,
    fetch_indicators,
    fetch_macro_indicators,
    fetch_news,
    fetch_prediction_markets,
    fetch_stock_data,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/data",
    tags=["Data"],
    responses={500: {"model": ErrorResponse, "description": "Internal server error"}},
)


@router.get(
    "/stock/{ticker}",
    response_model=StockDataResponse,
    summary="Stock OHLCV Data",
    description="Get historical OHLCV stock data for a ticker.",
)
async def get_stock_data(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    period: int = Query(default=365, ge=1, le=3650, description="Days of historical data"),
) -> StockDataResponse:
    """Get stock OHLCV data."""
    try:
        data = fetch_stock_data(ticker, trade_date, period=period)
        return StockDataResponse(ticker=ticker, data=data)
    except Exception as e:
        logger.exception("Failed to fetch stock data for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/indicators/{ticker}",
    response_model=IndicatorsResponse,
    summary="Technical Indicators",
    description="Get technical indicators for a ticker.",
)
async def get_indicators(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    indicator_names: str | None = Query(
        default=None,
        description="Comma-separated indicator names (e.g. 'rsi,macd,close_50_sma')",
    ),
) -> IndicatorsResponse:
    """Get technical indicators."""
    try:
        names = [n.strip() for n in indicator_names.split(",")] if indicator_names else None
        data = fetch_indicators(ticker, trade_date, indicator_names=names)
        return IndicatorsResponse(ticker=ticker, indicators=data)
    except Exception as e:
        logger.exception("Failed to fetch indicators for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/fundamentals/{ticker}",
    response_model=FundamentalsResponse,
    summary="Fundamentals Data",
    description="Get fundamental data (financials, balance sheet, cashflow) for a ticker.",
)
async def get_fundamentals(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    report_type: str = Query(
        default="all",
        description="Type of fundamental data: all, balance_sheet, cashflow, income_statement",
    ),
) -> FundamentalsResponse:
    """Get fundamentals data."""
    try:
        data = fetch_fundamentals(ticker, trade_date, report_type=report_type)
        return FundamentalsResponse(ticker=ticker, data=data)
    except Exception as e:
        logger.exception("Failed to fetch fundamentals for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/news/{ticker}",
    response_model=NewsResponse,
    summary="News Data",
    description="Get news articles and events for a ticker.",
)
async def get_news(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    country: str | None = Query(default=None, description="Country code for global news"),
) -> NewsResponse:
    """Get news data."""
    try:
        news_items = fetch_news(ticker, trade_date, country=country)
        return NewsResponse(ticker=ticker, news_items=news_items)
    except Exception as e:
        logger.exception("Failed to fetch news for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/macro-indicators",
    response_model=MacroIndicatorsResponse,
    summary="Macro Indicators",
    description="Get macroeconomic indicators for a country.",
)
async def get_macro_indicators(
    country: str = Query(default="US", description="Country code"),
) -> MacroIndicatorsResponse:
    """Get macro indicators."""
    try:
        data = fetch_macro_indicators(country=country)
        return MacroIndicatorsResponse(country=country, data=data)
    except Exception as e:
        logger.exception("Failed to fetch macro indicators for %s", country)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/prediction-markets/{ticker}",
    response_model=PredictionMarketsResponse,
    summary="Prediction Markets",
    description="Get prediction market data for a ticker.",
)
async def get_prediction_markets(
    ticker: str,
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
) -> PredictionMarketsResponse:
    """Get prediction market data."""
    try:
        data = fetch_prediction_markets(ticker, trade_date)
        return PredictionMarketsResponse(ticker=ticker, markets=data)
    except Exception as e:
        logger.exception("Failed to fetch prediction markets for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))