"""Router for individual analyst endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from tradingagents.api.dependencies import get_analyst_service
from tradingagents.api.domain.services.analyst_service import AnalystService
from tradingagents.api.schemas.request import SingleAnalystRequest
from tradingagents.api.schemas.response import AnalystReportResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analysts",
    tags=["Analysts"],
)


@router.post(
    "/market",
    response_model=AnalystReportResponse,
    summary="Market Analysis",
    description=(
        "Run the market analyst to analyze technical indicators, price action, "
        "and market conditions. The analyst selects relevant indicators and "
        "produces a detailed market report.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `asset_type`: 'stock' or 'crypto' (default: 'stock')"
    ),
)
def market_analyst(
    request: SingleAnalystRequest,
    analyst_service: AnalystService = Depends(get_analyst_service),
) -> AnalystReportResponse:
    """Run market analysis."""
    report = analyst_service.run_analyst(
        analyst_type="market",
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type.value,
    )
    return AnalystReportResponse(
        ticker=report.ticker,
        trade_date=report.trade_date,
        analyst_type=report.analyst_type,
        report=report.report_content,
    )


@router.post(
    "/sentiment",
    response_model=AnalystReportResponse,
    summary="Sentiment Analysis",
    description=(
        "Run the sentiment analyst to analyze social media, stocktwits, and "
        "other sentiment sources. Produces an overall sentiment band, score, "
        "confidence level, and detailed narrative.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `asset_type`: 'stock' or 'crypto' (default: 'stock')"
    ),
)
def sentiment_analyst(
    request: SingleAnalystRequest,
    analyst_service: AnalystService = Depends(get_analyst_service),
) -> AnalystReportResponse:
    """Run sentiment analysis."""
    report = analyst_service.run_analyst(
        analyst_type="sentiment",
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type.value,
    )
    return AnalystReportResponse(
        ticker=report.ticker,
        trade_date=report.trade_date,
        analyst_type=report.analyst_type,
        report=report.report_content,
    )


@router.post(
    "/news",
    response_model=AnalystReportResponse,
    summary="News Analysis",
    description=(
        "Run the news analyst to analyze news articles, insider transactions, "
        "and macro indicators. Produces a comprehensive news report with "
        "actionable insights.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `asset_type`: 'stock' or 'crypto' (default: 'stock')"
    ),
)
def news_analyst(
    request: SingleAnalystRequest,
    analyst_service: AnalystService = Depends(get_analyst_service),
) -> AnalystReportResponse:
    """Run news analysis."""
    report = analyst_service.run_analyst(
        analyst_type="news",
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type.value,
    )
    return AnalystReportResponse(
        ticker=report.ticker,
        trade_date=report.trade_date,
        analyst_type=report.analyst_type,
        report=report.report_content,
    )


@router.post(
    "/fundamentals",
    response_model=AnalystReportResponse,
    summary="Fundamentals Analysis",
    description=(
        "Run the fundamentals analyst to analyze financial statements, "
        "balance sheets, cashflow, and income statements. Produces a detailed "
        "fundamentals report with valuation insights.\n\n"
        "**Parameters:**\n"
        "- `ticker`: Stock/crypto symbol (e.g., 'AAPL', 'BTC')\n"
        "- `trade_date`: Date in YYYY-MM-DD format\n"
        "- `asset_type`: 'stock' or 'crypto' (default: 'stock')"
    ),
)
def fundamentals_analyst(
    request: SingleAnalystRequest,
    analyst_service: AnalystService = Depends(get_analyst_service),
) -> AnalystReportResponse:
    """Run fundamentals analysis."""
    report = analyst_service.run_analyst(
        analyst_type="fundamentals",
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type.value,
    )
    return AnalystReportResponse(
        ticker=report.ticker,
        trade_date=report.trade_date,
        analyst_type=report.analyst_type,
        report=report.report_content,
    )
