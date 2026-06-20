"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    SentimentBand,
    SentimentReport,
    TraderAction,
    TraderProposal,
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class AssetType(str, Enum):
    """Supported asset types for analysis."""

    STOCK = "stock"
    CRYPTO = "crypto"


class AnalystType(str, Enum):
    """Available analyst agents."""

    MARKET = "market"
    SENTIMENT = "sentiment"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class AnalyzeRequest(BaseModel):
    """Request body for the full analysis pipeline."""

    ticker: str = Field(..., description="Ticker symbol to analyze (e.g. 'AAPL', 'BTC')")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    asset_type: AssetType = Field(
        default=AssetType.STOCK,
        description="Asset type: stock or crypto",
    )
    selected_analysts: list[AnalystType] = Field(
        default_factory=lambda: [
            AnalystType.MARKET,
            AnalystType.SENTIMENT,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ],
        description="List of analysts to include in the analysis",
    )
    debug: bool = Field(default=False, description="Enable debug mode with verbose output")


class SingleAnalystRequest(BaseModel):
    """Request body for individual analyst endpoints."""

    ticker: str = Field(..., description="Ticker symbol to analyze")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    asset_type: AssetType = Field(default=AssetType.STOCK, description="Asset type")


class StockDataRequest(BaseModel):
    """Request body for stock data queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    period: int = Field(default=365, ge=1, le=3650, description="Number of days of historical data")


class IndicatorsRequest(BaseModel):
    """Request body for technical indicators queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    indicator_names: list[str] = Field(
        default_factory=list,
        description="List of indicator names (e.g. ['rsi', 'macd', 'close_50_sma'])",
    )


class FundamentalsRequest(BaseModel):
    """Request body for fundamentals queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    report_type: Literal["all", "balance_sheet", "cashflow", "income_statement"] = Field(
        default="all",
        description="Type of fundamental data to retrieve",
    )


class NewsRequest(BaseModel):
    """Request body for news queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    country: str | None = Field(default=None, description="Country code for global news")


class MacroIndicatorsRequest(BaseModel):
    """Request body for macro indicators."""

    country: str = Field(default="US", description="Country code (e.g. 'US', 'CN', 'JP')")


class PredictionMarketRequest(BaseModel):
    """Request body for prediction market queries."""

    ticker: str = Field(..., description="Ticker symbol")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class AnalystReportResponse(BaseModel):
    """Generic analyst report response."""

    ticker: str
    trade_date: str
    analyst_type: str
    report: str


class AnalyzeResponse(BaseModel):
    """Full analysis pipeline response."""

    ticker: str
    trade_date: str
    asset_type: str
    market_report: str | None = None
    sentiment_report: str | None = None
    news_report: str | None = None
    fundamentals_report: str | None = None
    investment_plan: str | None = None
    trader_investment_plan: str | None = None
    final_trade_decision: str | None = None
    signal: str | None = None


class StockDataResponse(BaseModel):
    """Stock OHLCV data response."""

    ticker: str
    data: str | None = None  # CSV format


class IndicatorsResponse(BaseModel):
    """Technical indicators response."""

    ticker: str
    indicators: str | None = None  # CSV format


class FundamentalsResponse(BaseModel):
    """Fundamentals data response."""

    ticker: str
    data: dict | str | None = None


class NewsResponse(BaseModel):
    """News data response."""

    ticker: str | None = None
    news_items: list[dict] | str | None = None


class MacroIndicatorsResponse(BaseModel):
    """Macro indicators response."""

    country: str
    data: str | None = None


class PredictionMarketsResponse(BaseModel):
    """Prediction market data response."""

    ticker: str
    markets: str | None = None


# Structured decision responses (reuse schemas from agents)
class ResearchPlanResponse(ResearchPlan):
    """Research plan with metadata."""

    ticker: str | None = None
    trade_date: str | None = None


class TraderProposalResponse(TraderProposal):
    """Trader proposal with metadata."""

    ticker: str | None = None
    trade_date: str | None = None


class PortfolioDecisionResponse(PortfolioDecision):
    """Portfolio decision with metadata."""

    ticker: str | None = None
    trade_date: str | None = None


class SentimentReportResponse(SentimentReport):
    """Sentiment report with metadata."""

    ticker: str | None = None
    trade_date: str | None = None


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None