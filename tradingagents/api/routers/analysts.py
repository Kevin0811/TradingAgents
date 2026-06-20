"""Router for individual analyst endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from tradingagents.api.config import get_config
from tradingagents.api.schemas import (
    AnalystReportResponse,
    ErrorResponse,
    SingleAnalystRequest,
)
from tradingagents.api.services import (
    run_fundamentals_analyst,
    run_market_analyst,
    run_news_analyst,
    run_sentiment_analyst,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analysts",
    tags=["Analysts"],
    responses={500: {"model": ErrorResponse, "description": "Internal server error"}},
)

# Map of analyst type to service function
ANALYST_SERVICES = {
    "market": run_market_analyst,
    "sentiment": run_sentiment_analyst,
    "news": run_news_analyst,
    "fundamentals": run_fundamentals_analyst,
}


def _run_analyst(analyst_type: str, request: SingleAnalystRequest) -> AnalystReportResponse:
    """Run a single analyst.

    Args:
        analyst_type: Type of analyst to run.
        request: Analyst request parameters.

    Returns:
        Analyst report response.

    Raises:
        HTTPException: If analyst type is unknown or execution fails.
    """
    try:
        config = get_config()
        service = ANALYST_SERVICES.get(analyst_type)

        if service is None:
            raise ValueError(f"Unknown analyst type: {analyst_type}")

        result = service(
            ticker=request.ticker,
            trade_date=request.trade_date,
            asset_type=request.asset_type.value,
            config=config.config,
        )

        return AnalystReportResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "%s analyst failed for %s on %s",
            analyst_type, request.ticker, request.trade_date,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/market",
    response_model=AnalystReportResponse,
    summary="Market Analysis",
    description=(
        "Run the market analyst to analyze technical indicators, price action, "
        "and market conditions. The analyst selects relevant indicators and "
        "produces a detailed market report."
    ),
)
async def market_analyst(request: SingleAnalystRequest) -> AnalystReportResponse:
    """Run market analysis."""
    return _run_analyst("market", request)


@router.post(
    "/sentiment",
    response_model=AnalystReportResponse,
    summary="Sentiment Analysis",
    description=(
        "Run the sentiment analyst to analyze social media, stocktwits, and "
        "other sentiment sources. Produces an overall sentiment band, score, "
        "confidence level, and detailed narrative."
    ),
)
async def sentiment_analyst(request: SingleAnalystRequest) -> AnalystReportResponse:
    """Run sentiment analysis."""
    return _run_analyst("sentiment", request)


@router.post(
    "/news",
    response_model=AnalystReportResponse,
    summary="News Analysis",
    description=(
        "Run the news analyst to analyze news articles, insider transactions, "
        "and macro indicators. Produces a comprehensive news report with "
        "actionable insights."
    ),
)
async def news_analyst(request: SingleAnalystRequest) -> AnalystReportResponse:
    """Run news analysis."""
    return _run_analyst("news", request)


@router.post(
    "/fundamentals",
    response_model=AnalystReportResponse,
    summary="Fundamentals Analysis",
    description=(
        "Run the fundamentals analyst to analyze financial statements, "
        "balance sheets, cashflow, and income statements. Produces a detailed "
        "fundamentals report with valuation insights."
    ),
)
async def fundamentals_analyst(request: SingleAnalystRequest) -> AnalystReportResponse:
    """Run fundamentals analysis."""
    return _run_analyst("fundamentals", request)