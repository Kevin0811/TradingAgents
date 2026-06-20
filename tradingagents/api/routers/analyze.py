"""Router for the full analysis pipeline endpoint."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from tradingagents.api.config import get_config
from tradingagents.api.schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from tradingagents.api.services import run_full_analysis

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
    responses={500: {"model": ErrorResponse, "description": "Internal server error"}},
)


@router.post(
    "",
    response_model=AnalyzeResponse,
    summary="Run full analysis pipeline",
    description=(
        "Execute the complete multi-agent trading analysis pipeline for a given "
        "ticker and date. This runs all selected analysts, the research debate, "
        "trader proposal, and risk management discussion to produce a final "
        "trade decision."
    ),
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the full trading analysis pipeline.

    This endpoint orchestrates all agent teams:
    1. **Analysts** (market, sentiment, news, fundamentals) - gather data and produce reports
    2. **Researchers** (bull/bear) - debate the investment thesis
    3. **Research Manager** - synthesize debate into an investment plan
    4. **Trader** - convert investment plan into a transaction proposal
    5. **Risk Management** (aggressive/conservative/neutral) - debate risk factors
    6. **Portfolio Manager** - produce final trade decision

    Note: This endpoint may take significant time to respond (30s+) depending
    on the LLM provider and analysis depth.

    Args:
        request: Analysis request with ticker, date, and options.

    Returns:
        Complete analysis results including all reports and final decision.

    Raises:
        HTTPException: If the analysis fails.
    """
    try:
        config = get_config()

        # Convert enum values to strings for the service function
        selected = tuple(a.value for a in request.selected_analysts)

        result = run_full_analysis(
            ticker=request.ticker,
            trade_date=request.trade_date,
            asset_type=request.asset_type.value,
            selected_analysts=selected,
            debug=request.debug,
            config=config.config,
        )

        return AnalyzeResponse(**result)

    except Exception as e:
        logger.exception("Analysis failed for %s on %s", request.ticker, request.trade_date)
        raise HTTPException(status_code=500, detail=str(e))