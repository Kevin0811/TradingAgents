"""Response schemas for TradingAgents API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Re-use structured schemas from agents package
from tradingagents.agents.schemas import (
    PortfolioDecision,
    ResearchPlan,
    SentimentReport,
    TraderProposal,
)


# ---------------------------------------------------------------------------
# Generic analyst report response
# ---------------------------------------------------------------------------


class AnalystReportResponse(BaseModel):
    """Generic analyst report response."""

    ticker: str
    trade_date: str
    analyst_type: str
    report: str


# ---------------------------------------------------------------------------
# Full analysis pipeline response
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data query responses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Structured decision responses
# ---------------------------------------------------------------------------


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