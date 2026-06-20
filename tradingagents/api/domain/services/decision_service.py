"""Decision service - parses and manages trading decisions."""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.agents.schemas import (
    PortfolioDecision,
    ResearchPlan,
    SentimentReport,
    TraderProposal,
)
from tradingagents.api.core.exceptions import DataNotFoundError
from tradingagents.api.domain.entities import DecisionState
from tradingagents.api.domain.repositories import StateRepository

logger = logging.getLogger(__name__)


def _clean_enum_value(value: str) -> str:
    """Remove markdown formatting symbols (e.g., ** or *) and return the cleaned value.

    Args:
        value: A string that may contain markdown bold/italic markers.

    Returns:
        The cleaned string with markdown symbols removed.
    """
    return value.replace("**", "").replace("*", "").strip()


class DecisionService:
    """Service for parsing and managing trading decisions."""

    def __init__(self, state_repository: StateRepository):
        """Initialize the decision service.

        Args:
            state_repository: Repository for loading saved analysis states.
        """
        self._state_repo = state_repository

    def get_decision_state(self, ticker: str, trade_date: str) -> DecisionState:
        """Load the saved decision state for a ticker and date.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date.

        Returns:
            DecisionState with all decision components.

        Raises:
            DataNotFoundError: If no state exists.
        """
        try:
            raw_state = self._state_repo.load_state(ticker, trade_date)
        except FileNotFoundError as e:
            raise DataNotFoundError(
                message=f"No analysis state found for {ticker} on {trade_date}",
                detail=f"Run the analysis first: POST /analyze",
            ) from e

        return DecisionState(
            ticker=ticker,
            trade_date=trade_date,
            research_plan=raw_state.get("investment_plan", ""),
            trader_proposal=raw_state.get("trader_investment_plan", ""),
            portfolio_decision=raw_state.get("final_trade_decision", ""),
            sentiment_report=raw_state.get("sentiment_report", ""),
            raw_state=raw_state,
        )

    def parse_research_plan(self, text: str) -> ResearchPlan:
        """Parse a research plan from markdown text.

        Args:
            text: Markdown-formatted research plan.

        Returns:
            Parsed ResearchPlan.
        """
        recommendation = "Hold"
        rationale = ""
        strategic_actions = ""

        for line in text.splitlines():
            line_lower = line.lower().strip()
            if "**recommendation**" in line_lower or "recommendation:" in line_lower:
                recommendation = _clean_enum_value(line.split(":", 1)[-1].strip())
            elif "**rationale**" in line_lower or "rationale:" in line_lower:
                rationale = line.split(":", 1)[-1].strip()
            elif "**strategic actions**" in line_lower or "strategic actions:" in line_lower:
                strategic_actions = line.split(":", 1)[-1].strip()

        return ResearchPlan(
            recommendation=recommendation,
            rationale=rationale or "Could not parse rationale from report.",
            strategic_actions=strategic_actions or "Could not parse strategic actions from report.",
        )

    def parse_trader_proposal(self, text: str) -> TraderProposal:
        """Parse a trader proposal from markdown text.

        Args:
            text: Markdown-formatted trader proposal.

        Returns:
            Parsed TraderProposal.
        """
        action = "Hold"
        reasoning = ""
        entry_price = None
        stop_loss = None
        position_sizing = None

        for line in text.splitlines():
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if "**action**" in line_lower or "action:" in line_lower:
                action = line_stripped.split(":", 1)[-1].strip()
            elif "**reasoning**" in line_lower or "reasoning:" in line_lower:
                reasoning = line_stripped.split(":", 1)[-1].strip()
            elif "**entry price**" in line_lower or "entry price:" in line_lower:
                try:
                    entry_price = float(line_stripped.split(":", 1)[-1].strip())
                except (ValueError, TypeError):
                    pass
            elif "**stop loss**" in line_lower or "stop loss:" in line_lower:
                try:
                    stop_loss = float(line_stripped.split(":", 1)[-1].strip())
                except (ValueError, TypeError):
                    pass
            elif "**position sizing**" in line_lower or "position sizing:" in line_lower:
                position_sizing = line_stripped.split(":", 1)[-1].strip()

        # Extract final proposal if present
        if "FINAL TRANSACTION PROPOSAL:" in text:
            for line in text.splitlines():
                if "FINAL TRANSACTION PROPOSAL:" in line:
                    action = _clean_enum_value(line.split("**")[-2]) if "**" in line else "Hold"
                    break

        # Clean action of any markdown formatting
        action = _clean_enum_value(action)

        return TraderProposal(
            action=action,
            reasoning=reasoning or "Could not parse reasoning from report.",
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_sizing=position_sizing,
        )

    def parse_portfolio_decision(self, text: str) -> PortfolioDecision:
        """Parse a portfolio decision from markdown text.

        Args:
            text: Markdown-formatted portfolio decision.

        Returns:
            Parsed PortfolioDecision.
        """
        rating = "Hold"
        executive_summary = ""
        investment_thesis = ""
        price_target = None
        time_horizon = None

        for line in text.splitlines():
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if "**rating**" in line_lower or "rating:" in line_lower:
                rating = _clean_enum_value(line_stripped.split(":", 1)[-1].strip())
            elif "**executive summary**" in line_lower or "executive summary:" in line_lower:
                executive_summary = line_stripped.split(":", 1)[-1].strip()
            elif "**investment thesis**" in line_lower or "investment thesis:" in line_lower:
                investment_thesis = line_stripped.split(":", 1)[-1].strip()
            elif "**price target**" in line_lower or "price target:" in line_lower:
                try:
                    price_target = float(line_stripped.split(":", 1)[-1].strip())
                except (ValueError, TypeError):
                    pass
            elif "**time horizon**" in line_lower or "time horizon:" in line_lower:
                time_horizon = line_stripped.split(":", 1)[-1].strip()

        return PortfolioDecision(
            rating=rating,
            executive_summary=executive_summary or "Could not parse executive summary from report.",
            investment_thesis=investment_thesis or "Could not parse investment thesis from report.",
            price_target=price_target,
            time_horizon=time_horizon,
        )

    def parse_sentiment_report(self, text: str) -> SentimentReport:
        """Parse a sentiment report from markdown text.

        Args:
            text: Markdown-formatted sentiment report.

        Returns:
            Parsed SentimentReport.
        """
        overall_band = "Neutral"
        overall_score = 5.0
        confidence = "medium"
        narrative = ""

        for line in text.splitlines():
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if "overall sentiment" in line_lower or "band:" in line_lower:
                if "**" in line_stripped:
                    overall_band = _clean_enum_value(line_stripped.split("**")[-2])
                elif ":" in line_stripped:
                    overall_band = _clean_enum_value(line_stripped.split(":", 1)[-1].strip())
            elif "score:" in line_lower:
                try:
                    score_part = line_stripped.split(":", 1)[-1].strip()
                    overall_score = float(score_part.split("/")[0].strip())
                except (ValueError, TypeError, IndexError):
                    pass
            elif "**confidence**" in line_lower or "confidence:" in line_lower:
                conf_value = line_stripped.split(":", 1)[-1].strip().lower()
                if conf_value in ("low", "medium", "high"):
                    confidence = conf_value
                else:
                    confidence = "medium"
            elif not line_lower.startswith("**"):
                narrative += line_stripped + "\n"

        return SentimentReport(
            overall_band=overall_band,
            overall_score=overall_score,
            confidence=confidence,
            narrative=narrative.strip() or "Could not parse narrative from report.",
        )