"""Request schemas for TradingAgents API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from tradingagents.api.schemas.enums import (
    AnalystType,
    AssetType,
    IndicatorName,
    ReportFreq,
    ReportType,
)
from tradingagents.api.schemas.overrides import RunOverridesMixin
from tradingagents.api.schemas.validators import validate_ticker_shape


class AnalyzeRequest(RunOverridesMixin):
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

    _validate_ticker = field_validator("ticker")(validate_ticker_shape)


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
            "List of indicator names; empty uses a default set. "
            "Available: 'close_50_sma', 'close_200_sma', 'close_10_ema', 'macd', "
            "'macds', 'macdh', 'rsi', 'boll', 'boll_ub', 'boll_lb', 'atr', "
            "'vwma', 'mfi'"
        ),
    )
    look_back_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="How many days of indicator values to return",
    )


class FundamentalsRequest(BaseModel):
    """Request body for fundamentals queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    report_type: ReportType = Field(
        default=ReportType.ALL,
        description=(
            "Type of fundamental data to retrieve. "
            "Available: 'all', 'overview', 'balance_sheet', 'cashflow', "
            "'income_statement'"
        ),
    )
    freq: ReportFreq = Field(
        default=ReportFreq.QUARTERLY,
        description="Reporting frequency for the statements: 'annual' or 'quarterly'",
    )


class NewsRequest(BaseModel):
    """Request body for ticker news queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="End of the window, YYYY-MM-DD")
    look_back_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="How many days before trade_date to include",
    )


class GlobalNewsRequest(BaseModel):
    """Request body for global/macro news queries."""

    trade_date: str = Field(..., description="End of the window, YYYY-MM-DD")
    look_back_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Window length; omit to use the configured default",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Max articles; omit to use the configured default",
    )


class MacroIndicatorsRequest(BaseModel):
    """Request body for macro indicators (FRED)."""

    indicator: str = Field(
        ...,
        description=(
            "Friendly alias ('cpi', 'core_pce', 'unemployment', 'fed_funds_rate', "
            "'10y_treasury', 'yield_curve', 'real_gdp', 'vix') or a raw FRED "
            "series ID such as 'CPIAUCSL'"
        ),
    )
    trade_date: str = Field(..., description="End of the window, YYYY-MM-DD")
    look_back_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Window length; omit for a 1-year window",
    )


class PredictionMarketRequest(BaseModel):
    """Request body for prediction market queries."""

    topic: str = Field(
        ...,
        description="Event topic/keyword, e.g. 'Fed rate cut', 'recession 2026'",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Max markets to return; omit for the vendor default",
    )
