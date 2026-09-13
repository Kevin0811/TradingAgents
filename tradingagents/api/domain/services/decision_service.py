"""Decision service - parses and manages trading decisions."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from tradingagents.agents.schemas import (
    SentimentReport,
    TraderProposal,
    _coerce_optional_float,
)
from tradingagents.agents.utils.rating import RATING_REVIEW, RATINGS_5_TIER, extract_rating
from tradingagents.api.core.exceptions import DataNotFoundError
from tradingagents.api.domain.entities import DecisionState
from tradingagents.api.domain.repositories import StateRepository
from tradingagents.api.schemas.response import (
    PortfolioDecisionRating,
    ResearchPlanRecommendation,
)

logger = logging.getLogger(__name__)

_RATING_WORDS = {r.lower() for r in RATINGS_5_TIER}


def _clean_enum_value(value: str) -> str:
    """Remove markdown formatting symbols (e.g., ** or *) and return the cleaned value.

    Args:
        value: A string that may contain markdown bold/italic markers.

    Returns:
        The cleaned string with markdown symbols removed.
    """
    return value.replace("**", "").replace("*", "").strip()


def _parse_price(raw: str) -> float | None:
    """Clean and parse an optional price field the same way the core Trader/Portfolio
    schemas do: strip currency symbols and thousands separators, and never salvage
    a percentage into an absolute price (see tradingagents.agents.schemas._coerce_optional_float).
    """
    cleaned = _coerce_optional_float(raw)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


class ParsedResearchPlan(BaseModel):
    """Research plan parsed from markdown; recommendation may be REVIEW."""

    recommendation: ResearchPlanRecommendation
    rationale: str
    strategic_actions: str


class ParsedPortfolioDecision(BaseModel):
    """Portfolio decision parsed from markdown; rating may be REVIEW."""

    rating: PortfolioDecisionRating
    executive_summary: str
    investment_thesis: str
    price_target: float | None = None
    time_horizon: str | None = None


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
                detail="Run the analysis first: POST /analyze",
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

    def parse_research_plan(self, text: str) -> ParsedResearchPlan:
        """Parse a research plan from markdown text.

        Args:
            text: Markdown-formatted research plan.

        Returns:
            Parsed ResearchPlan. ``recommendation`` is ``"REVIEW"`` rather than a
            fabricated ``"Hold"`` when no recognizable recommendation is found
            (mirrors the core graph's rating fix, tradingagents.agents.utils.rating).
        """
        recommendation_raw: str | None = None
        rationale = ""
        strategic_actions = ""

        for line in text.splitlines():
            line_lower = line.lower().strip()
            if "**recommendation**" in line_lower or "recommendation:" in line_lower:
                recommendation_raw = _clean_enum_value(line.split(":", 1)[-1].strip())
            elif "**rationale**" in line_lower or "rationale:" in line_lower:
                rationale = line.split(":", 1)[-1].strip()
            elif "**strategic actions**" in line_lower or "strategic actions:" in line_lower:
                strategic_actions = line.split(":", 1)[-1].strip()

        recommendation = (
            recommendation_raw.capitalize()
            if recommendation_raw and recommendation_raw.lower() in _RATING_WORDS
            else RATING_REVIEW
        )

        return ParsedResearchPlan(
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
                entry_price = _parse_price(line_stripped.split(":", 1)[-1].strip())
            elif "**stop loss**" in line_lower or "stop loss:" in line_lower:
                stop_loss = _parse_price(line_stripped.split(":", 1)[-1].strip())
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

    def parse_portfolio_decision(self, text: str) -> ParsedPortfolioDecision:
        """Parse a portfolio decision from markdown text.

        Args:
            text: Markdown-formatted portfolio decision.

        Returns:
            Parsed PortfolioDecision. ``rating`` is ``"REVIEW"`` rather than a
            fabricated ``"Hold"`` when no recognizable rating is found (see
            tradingagents.agents.utils.rating.extract_rating).
        """
        executive_summary = ""
        investment_thesis = ""
        price_target = None
        time_horizon = None

        for line in text.splitlines():
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if "**executive summary**" in line_lower or "executive summary:" in line_lower:
                executive_summary = line_stripped.split(":", 1)[-1].strip()
            elif "**investment thesis**" in line_lower or "investment thesis:" in line_lower:
                investment_thesis = line_stripped.split(":", 1)[-1].strip()
            elif "**price target**" in line_lower or "price target:" in line_lower:
                price_target = _parse_price(line_stripped.split(":", 1)[-1].strip())
            elif "**time horizon**" in line_lower or "time horizon:" in line_lower:
                time_horizon = line_stripped.split(":", 1)[-1].strip()

        rating = extract_rating(text) or RATING_REVIEW

        return ParsedPortfolioDecision(
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
                confidence = conf_value if conf_value in ("low", "medium", "high") else "medium"
            elif not line_lower.startswith("**"):
                narrative += line_stripped + "\n"

        return SentimentReport(
            overall_band=overall_band,
            overall_score=overall_score,
            confidence=confidence,
            narrative=narrative.strip() or "Could not parse narrative from report.",
        )
