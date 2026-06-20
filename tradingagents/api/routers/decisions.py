"""Router for structured decision endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from tradingagents.agents.schemas import (
    PortfolioDecision,
    ResearchPlan,
    SentimentReport,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)
from tradingagents.api.schemas import (
    ErrorResponse,
    PortfolioDecisionResponse,
    ResearchPlanResponse,
    SentimentReportResponse,
    TraderProposalResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/decisions",
    tags=["Decisions"],
    responses={500: {"model": ErrorResponse, "description": "Internal server error"}},
)


@router.get(
    "/research-plan",
    response_model=ResearchPlanResponse,
    summary="Research Plan",
    description=(
        "Get the structured research plan produced by the Research Manager. "
        "Contains the investment recommendation, rationale, and strategic actions."
    ),
)
async def get_research_plan(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
) -> ResearchPlanResponse:
    """Get the research plan for a ticker.

    Note: This endpoint reads the saved state from a previous analysis run.
    Run the /analyze endpoint first to generate the data.
    """
    try:
        from tradingagents.api.config import get_config

        config = get_config()
        state = _load_state(ticker, trade_date, config.config.get("results_dir"))
        investment_plan = state.get("investment_plan", "")

        plan = parse_research_plan(investment_plan)
        return ResearchPlanResponse(
            **plan.model_dump(),
            ticker=ticker,
            trade_date=trade_date,
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="Unable to load state management")
    except Exception as e:
        logger.exception("Failed to get research plan for %s on %s", ticker, trade_date)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/trader-proposal",
    response_model=TraderProposalResponse,
    summary="Trader Proposal",
    description=(
        "Get the structured transaction proposal produced by the Trader. "
        "Contains the proposed action (Buy/Hold/Sell), reasoning, and trading levels."
    ),
)
async def get_trader_proposal(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
) -> TraderProposalResponse:
    """Get the trader proposal for a ticker."""
    try:
        from tradingagents.api.config import get_config

        config = get_config()
        state = _load_state(ticker, trade_date, config.config.get("results_dir"))
        trader_plan = state.get("trader_investment_plan", state.get("trader_investment_decision", ""))

        proposal = parse_trader_proposal(trader_plan)
        return TraderProposalResponse(
            **proposal.model_dump(),
            ticker=ticker,
            trade_date=trade_date,
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="Unable to load state management")
    except Exception as e:
        logger.exception("Failed to get trader proposal for %s on %s", ticker, trade_date)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/portfolio-decision",
    response_model=PortfolioDecisionResponse,
    summary="Portfolio Decision",
    description=(
        "Get the final portfolio decision produced by the Portfolio Manager. "
        "Contains the final rating, executive summary, investment thesis, "
        "price target, and time horizon."
    ),
)
async def get_portfolio_decision(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
) -> PortfolioDecisionResponse:
    """Get the portfolio decision for a ticker."""
    try:
        from tradingagents.api.config import get_config

        config = get_config()
        state = _load_state(ticker, trade_date, config.config.get("results_dir"))
        final_decision = state.get("final_trade_decision", "")

        decision = parse_portfolio_decision(final_decision)
        return PortfolioDecisionResponse(
            **decision.model_dump(),
            ticker=ticker,
            trade_date=trade_date,
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="Unable to load state management")
    except Exception as e:
        logger.exception("Failed to get portfolio decision for %s on %s", ticker, trade_date)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sentiment-report",
    response_model=SentimentReportResponse,
    summary="Sentiment Report",
    description=(
        "Get the structured sentiment report produced by the Sentiment Analyst. "
        "Contains overall sentiment band, score, confidence level, and narrative."
    ),
)
async def get_sentiment_report(
    ticker: str = Query(..., description="Ticker symbol"),
    trade_date: str = Query(..., description="Trading date in YYYY-MM-DD format"),
) -> SentimentReportResponse:
    """Get the sentiment report for a ticker."""
    try:
        from tradingagents.api.config import get_config

        config = get_config()
        state = _load_state(ticker, trade_date, config.config.get("results_dir"))
        sentiment_text = state.get("sentiment_report", "")

        report = parse_sentiment_report(sentiment_text)
        return SentimentReportResponse(
            **report.model_dump(),
            ticker=ticker,
            trade_date=trade_date,
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="Unable to load state management")
    except Exception as e:
        logger.exception("Failed to get sentiment report for %s on %s", ticker, trade_date)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def parse_research_plan(text: str) -> ResearchPlan:
    """Parse a research plan from markdown text.

    Falls back to default values if parsing fails.
    """
    recommendation = "Hold"
    rationale = ""
    strategic_actions = ""

    for line in text.splitlines():
        line_lower = line.lower().strip()
        if "**recommendation**" in line_lower or "recommendation:" in line_lower:
            recommendation = line.split(":", 1)[-1].strip()
        elif "**rationale**" in line_lower or "rationale:" in line_lower:
            rationale = line.split(":", 1)[-1].strip()
        elif "**strategic actions**" in line_lower or "strategic actions:" in line_lower:
            strategic_actions = line.split(":", 1)[-1].strip()

    return ResearchPlan(
        recommendation=recommendation,
        rationale=rationale or "Could not parse rationale from report.",
        strategic_actions=strategic_actions or "Could not parse strategic actions from report.",
    )


def parse_trader_proposal(text: str) -> TraderProposal:
    """Parse a trader proposal from markdown text.

    Falls back to default values if parsing fails.
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
                action = line.split("**")[-2] if "**" in line else "Hold"
                break

    return TraderProposal(
        action=action,
        reasoning=reasoning or "Could not parse reasoning from report.",
        entry_price=entry_price,
        stop_loss=stop_loss,
        position_sizing=position_sizing,
    )


def parse_portfolio_decision(text: str) -> PortfolioDecision:
    """Parse a portfolio decision from markdown text.

    Falls back to default values if parsing fails.
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
            rating = line_stripped.split(":", 1)[-1].strip()
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


def parse_sentiment_report(text: str) -> SentimentReport:
    """Parse a sentiment report from markdown text.

    Falls back to default values if parsing fails.
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
                overall_band = line_stripped.split("**")[-2]
            elif ":" in line_stripped:
                overall_band = line_stripped.split(":", 1)[-1].strip()
        elif "score:" in line_lower:
            try:
                score_part = line_stripped.split(":", 1)[-1].strip()
                overall_score = float(score_part.split("/")[0].strip())
            except (ValueError, TypeError, IndexError):
                pass
        elif "**confidence**" in line_lower or "confidence:" in line_lower:
            conf_value = line_stripped.split(":", 1)[-1].strip().lower()
            # Map confidence string to valid enum value
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


# ---------------------------------------------------------------------------
# State loading helper
# ---------------------------------------------------------------------------


def _load_state(
    ticker: str, trade_date: str, results_dir: str
) -> dict:
    """Load the saved state for a ticker and date from disk.

    Args:
        ticker: Ticker symbol.
        trade_date: Trading date.
        results_dir: Results directory path from config.

    Returns:
        Dict with saved state.

    Raises:
        FileNotFoundError: If no state file exists.
    """
    from pathlib import Path

    from tradingagents.dataflows.utils import safe_ticker_component

    safe_ticker = safe_ticker_component(ticker)
    log_path = (
        Path(results_dir)
        / safe_ticker
        / "TradingAgentsStrategy_logs"
        / f"full_states_log_{trade_date}.json"
    )

    if not log_path.exists():
        raise FileNotFoundError(
            f"No state file found for {ticker} on {trade_date}. "
            f"Run the analysis first: POST /analyze"
        )

    import json

    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)