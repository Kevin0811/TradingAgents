"""Request schemas for TradingAgents API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tradingagents.api.schemas.enums import (
    AnalystType,
    AssetType,
    Country,
    IndicatorName,
    ReportType,
)


class AnalyzeRequest(BaseModel):
    """Request body for the full analysis pipeline."""

    ticker: str = Field(..., description="Ticker symbol to analyze (e.g. 'AAPL', 'BTC')")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    asset_type: AssetType = Field(
        default=AssetType.STOCK,
        description="Asset type: 'stock' or 'crypto'",
    )
    selected_analysts: list[AnalystType] = Field(
        default_factory=lambda: [
            AnalystType.MARKET,
            AnalystType.SENTIMENT,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ],
        description="List of analysts to include: 'market', 'sentiment', 'news', 'fundamentals'",
    )
    debug: bool = Field(default=False, description="Enable debug mode with verbose output")


class SingleAnalystRequest(BaseModel):
    """Request body for individual analyst endpoints."""

    ticker: str = Field(..., description="Ticker symbol to analyze")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    asset_type: AssetType = Field(
        default=AssetType.STOCK,
        description="Asset type: 'stock' or 'crypto'",
    )


class StockDataRequest(BaseModel):
    """Request body for stock data queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    period: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Number of days of historical data (1-3650)",
    )


class IndicatorsRequest(BaseModel):
    """Request body for technical indicators queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    indicator_names: list[IndicatorName] = Field(
        default_factory=list,
        description=(
            "List of indicator names. "
            "Available: 'rsi', 'macd', 'signal', 'histogram', 'close_50_sma', "
            "'close_200_sma', 'close_10_sma', 'close_20_sma', 'bollinger_upper', "
            "'bollinger_middle', 'bollinger_lower', 'volume', 'atr', 'adx', "
            "'stoch_k', 'stoch_d'"
        ),
    )


class FundamentalsRequest(BaseModel):
    """Request body for fundamentals queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    report_type: ReportType = Field(
        default=ReportType.ALL,
        description=(
            "Type of fundamental data to retrieve. "
            "Available: 'all', 'balance_sheet', 'cashflow', 'income_statement'"
        ),
    )


class NewsRequest(BaseModel):
    """Request body for news queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    country: Country | None = Field(
        default=None,
        description=(
            "Country code for global news. "
            "Available: 'US', 'CN', 'JP', 'GB', 'DE', 'FR', 'TW', 'KR', 'SG'"
        ),
    )


class MacroIndicatorsRequest(BaseModel):
    """Request body for macro indicators."""

    country: Country = Field(
        default=Country.US,
        description=(
            "Country code. "
            "Available: 'US', 'CN', 'JP', 'GB', 'DE', 'FR', 'TW', 'KR', 'SG'"
        ),
    )


class PredictionMarketRequest(BaseModel):
    """Request body for prediction market queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")