"""Router for the full analysis pipeline endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.dependencies import get_analysis_service
from tradingagents.api.schemas.request import AnalyzeRequest
from tradingagents.api.schemas.response import AnalyzeResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
)


@router.post(
    "",
    response_model=AnalyzeResponse,
    summary="Run full analysis pipeline",
    description=(
        "Execute the complete multi-agent trading analysis pipeline for a given "
        "ticker and date. This runs all selected analysts, the research debate, "
        "trader proposal, and risk management discussion to produce a final "
        "trade decision.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `asset_type`: 'stock' or 'crypto' (default: 'stock')\n"
        "- `selected_analysts`: List of 'market', 'sentiment', 'news', 'fundamentals'\n"
        "- `debug`: Enable debug mode (default: false)\n\n"
        "**Pipeline Steps:**\n"
        "1. **Analysts** (market, sentiment, news, fundamentals) - gather data and produce reports\n"
        "2. **Researchers** (bull/bear) - debate the investment thesis\n"
        "3. **Research Manager** - synthesize debate into an investment plan\n"
        "4. **Trader** - convert investment plan into a transaction proposal\n"
        "5. **Risk Management** (aggressive/conservative/neutral) - debate risk factors\n"
        "6. **Portfolio Manager** - produce final trade decision\n\n"
        "Note: This endpoint may take significant time to respond (30s+) depending "
        "on the LLM provider and analysis depth."
    ),
)
async def analyze(
    request: AnalyzeRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Run the full trading analysis pipeline."""
    # Convert enum values to strings for the service function
    selected = tuple(a.value for a in request.selected_analysts)

    result = analysis_service.run_analysis(
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type.value,
        selected_analysts=selected,
    )

    return AnalyzeResponse(**result.to_dict())