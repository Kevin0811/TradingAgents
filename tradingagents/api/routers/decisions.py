"""Router for structured decision endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from tradingagents.api.dependencies import get_decision_service
from tradingagents.api.domain.services.decision_service import DecisionService
from tradingagents.api.schemas.response import (
    PortfolioDecisionResponse,
    ResearchPlanResponse,
    SentimentReportResponse,
    TraderProposalResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/decisions",
    tags=["Decisions"],
)


@router.get(
    "/research-plan",
    response_model=ResearchPlanResponse,
    summary="Research Plan",
    description=(
        "Get the structured research plan produced by the Research Manager. "
        "Contains the investment recommendation, rationale, and strategic actions.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n\n"
        "Note: This endpoint reads the saved state from a previous analysis run. "
        "Run the /analyze endpoint first to generate the data."
    ),
)
def get_research_plan(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    decision_service: DecisionService = Depends(get_decision_service),
) -> ResearchPlanResponse:
    """Get the research plan for a ticker."""
    decision_state = decision_service.get_decision_state(ticker, trade_date)
    plan = decision_service.parse_research_plan(decision_state.research_plan)
    return ResearchPlanResponse(
        **plan.model_dump(),
        ticker=ticker,
        trade_date=trade_date,
    )


@router.get(
    "/trader-proposal",
    response_model=TraderProposalResponse,
    summary="Trader Proposal",
    description=(
        "Get the structured transaction proposal produced by the Trader. "
        "Contains the proposed action (Buy/Hold/Sell), reasoning, and trading levels.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n\n"
        "Note: This endpoint reads the saved state from a previous analysis run. "
        "Run the /analyze endpoint first to generate the data."
    ),
)
def get_trader_proposal(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    decision_service: DecisionService = Depends(get_decision_service),
) -> TraderProposalResponse:
    """Get the trader proposal for a ticker."""
    decision_state = decision_service.get_decision_state(ticker, trade_date)
    proposal = decision_service.parse_trader_proposal(decision_state.trader_proposal)
    return TraderProposalResponse(
        **proposal.model_dump(),
        ticker=ticker,
        trade_date=trade_date,
    )


@router.get(
    "/portfolio-decision",
    response_model=PortfolioDecisionResponse,
    summary="Portfolio Decision",
    description=(
        "Get the final portfolio decision produced by the Portfolio Manager. "
        "Contains the final rating, executive summary, investment thesis, "
        "price target, and time horizon.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n\n"
        "Note: This endpoint reads the saved state from a previous analysis run. "
        "Run the /analyze endpoint first to generate the data."
    ),
)
def get_portfolio_decision(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    decision_service: DecisionService = Depends(get_decision_service),
) -> PortfolioDecisionResponse:
    """Get the portfolio decision for a ticker."""
    decision_state = decision_service.get_decision_state(ticker, trade_date)
    decision = decision_service.parse_portfolio_decision(decision_state.portfolio_decision)
    return PortfolioDecisionResponse(
        **decision.model_dump(),
        ticker=ticker,
        trade_date=trade_date,
    )


@router.get(
    "/sentiment-report",
    response_model=SentimentReportResponse,
    summary="Sentiment Report",
    description=(
        "Get the structured sentiment report produced by the Sentiment Analyst. "
        "Contains overall sentiment band, score, confidence level, and narrative.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n\n"
        "Note: This endpoint reads the saved state from a previous analysis run. "
        "Run the /analyze endpoint first to generate the data."
    ),
)
def get_sentiment_report(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
    decision_service: DecisionService = Depends(get_decision_service),
) -> SentimentReportResponse:
    """Get the sentiment report for a ticker."""
    decision_state = decision_service.get_decision_state(ticker, trade_date)
    report = decision_service.parse_sentiment_report(decision_state.sentiment_report)
    return SentimentReportResponse(
        **report.model_dump(),
        ticker=ticker,
        trade_date=trade_date,
    )
