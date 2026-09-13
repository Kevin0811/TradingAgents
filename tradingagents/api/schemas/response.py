"""Response schemas for TradingAgents API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# Re-use structured schemas from agents package
from tradingagents.agents.schemas import SentimentReport, TraderProposal

# ---------------------------------------------------------------------------
# Rating enums for markdown-parsed decisions
# ---------------------------------------------------------------------------
#
# The core agents.schemas.PortfolioRating enum is intentionally kept to the
# strict 5-tier scale because it is also the LLM structured-output schema
# (json_schema/response_schema/tool-use) the Research Manager and Portfolio
# Manager are constrained to answer with -- widening it to include REVIEW
# would let a model choose "no rating" as if it were a legitimate answer.
# REVIEW is instead a sentinel the API adds on its own when it cannot parse a
# rating out of the rendered markdown, so it needs its own, API-only enum.


class ResearchPlanRecommendation(str, Enum):
    """5-tier recommendation plus REVIEW when it could not be parsed from the report."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"
    REVIEW = "REVIEW"


class PortfolioDecisionRating(str, Enum):
    """5-tier rating plus REVIEW when it could not be parsed from the report."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"
    REVIEW = "REVIEW"


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


class OHLCVBar(BaseModel):
    """A single price observation.

    Extra keys are allowed on purpose: vendors carry their own columns
    (``dividends``, ``split_coefficient``, ...) and dropping them would lose
    data the caller asked for. ``date`` plus OHLCV are the guaranteed fields.
    """

    model_config = ConfigDict(extra="allow")

    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    adj_close: float | None = None


class StockDataResponse(BaseModel):
    """Stock OHLCV data response."""

    ticker: str = Field(description="Ticker exactly as requested")
    symbol: str = Field(description="Canonical symbol the vendor was queried with")
    start_date: str
    end_date: str
    count: int
    rows: list[OHLCVBar] = Field(default_factory=list)


class IndicatorPoint(BaseModel):
    """One indicator observation.

    ``value`` is null when the vendor reported a placeholder instead of a
    number (a non-trading day, or a window where the indicator is undefined);
    the placeholder text is kept in ``note`` so the reason is not lost.
    """

    date: str
    value: float | None = None
    note: str | None = None


class IndicatorSeries(BaseModel):
    """One indicator's values over the requested window."""

    name: str
    description: str | None = None
    points: list[IndicatorPoint] = Field(default_factory=list)


class IndicatorsResponse(BaseModel):
    """Technical indicators response."""

    ticker: str = Field(description="Ticker exactly as requested")
    symbol: str = Field(description="Canonical symbol the vendor was queried with")
    trade_date: str
    look_back_days: int
    indicators: list[IndicatorSeries] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list,
        description="Per-indicator failures, e.g. an unsupported indicator name",
    )


class PriceHistoryResponse(BaseModel):
    """OHLCV bars and indicator series over one window, aligned for charting.

    ``bars`` and every series in ``indicators`` cover the same window, which is
    what separates this from combining ``StockDataResponse`` with
    ``IndicatorsResponse`` by hand — those two describe their windows with
    different parameters.
    """

    ticker: str = Field(description="Ticker exactly as requested")
    symbol: str = Field(description="Canonical symbol the vendor was queried with")
    start_date: str
    end_date: str
    count: int = Field(description="Number of bars, not of calendar days")
    bars: list[OHLCVBar] = Field(default_factory=list)
    indicators: list[IndicatorSeries] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list,
        description="Per-indicator failures, e.g. an unsupported indicator name",
    )


class FundamentalsResponse(BaseModel):
    """Fundamentals data response."""

    ticker: str
    data: dict | str | None = None


class NewsResponse(BaseModel):
    """News data response."""

    ticker: str | None = None
    news_items: list[dict] | str | None = None


class GlobalNewsResponse(BaseModel):
    """Global/macro market news response."""

    trade_date: str
    news_items: str | None = None


class MacroIndicatorsResponse(BaseModel):
    """Macro indicators response."""

    indicator: str
    data: str | None = None


class PredictionMarketsResponse(BaseModel):
    """Prediction market data response."""

    topic: str
    markets: str | None = None


# ---------------------------------------------------------------------------
# Structured decision responses
# ---------------------------------------------------------------------------


class ResearchPlanResponse(BaseModel):
    """Research plan with metadata.

    ``recommendation`` may be ``REVIEW`` when the saved report's recommendation
    could not be parsed -- see ResearchPlanRecommendation.
    """

    recommendation: ResearchPlanRecommendation
    rationale: str
    strategic_actions: str
    ticker: str | None = None
    trade_date: str | None = None


class TraderProposalResponse(TraderProposal):
    """Trader proposal with metadata."""

    ticker: str | None = None
    trade_date: str | None = None


class PortfolioDecisionResponse(BaseModel):
    """Portfolio decision with metadata.

    ``rating`` may be ``REVIEW`` when the saved report's rating could not be
    parsed -- see PortfolioDecisionRating.
    """

    rating: PortfolioDecisionRating
    executive_summary: str
    investment_thesis: str
    price_target: float | None = None
    time_horizon: str | None = None
    ticker: str | None = None
    trade_date: str | None = None


class SentimentReportResponse(SentimentReport):
    """Sentiment report with metadata."""

    ticker: str | None = None
    trade_date: str | None = None
