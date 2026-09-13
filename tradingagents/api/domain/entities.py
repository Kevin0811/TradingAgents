"""Domain entities for TradingAgents API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MarketData:
    """Market data entity."""

    ticker: str
    ohlcv_data: str  # CSV format
    indicators: dict[str, float] = field(default_factory=dict)
    fundamentals: dict[str, Any] | None = None


@dataclass
class AnalystReport:
    """Analyst report entity."""

    ticker: str
    trade_date: str
    analyst_type: str
    report_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Full analysis result entity."""

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "asset_type": self.asset_type,
            "market_report": self.market_report,
            "sentiment_report": self.sentiment_report,
            "news_report": self.news_report,
            "fundamentals_report": self.fundamentals_report,
            "investment_plan": self.investment_plan,
            "trader_investment_plan": self.trader_investment_plan,
            "final_trade_decision": self.final_trade_decision,
            "signal": self.signal,
        }


@dataclass
class DecisionState:
    """Saved decision state from a previous analysis."""

    ticker: str
    trade_date: str
    research_plan: str = ""
    trader_proposal: str = ""
    portfolio_decision: str = ""
    sentiment_report: str = ""
    raw_state: dict[str, Any] = field(default_factory=dict)
