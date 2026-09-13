"""Pydantic schemas for API request/response models."""

from tradingagents.api.schemas.enums import AnalystType, AssetType
from tradingagents.api.schemas.error import ErrorResponse
from tradingagents.api.schemas.request import (
    AnalyzeRequest,
    FundamentalsRequest,
    GlobalNewsRequest,
    IndicatorsRequest,
    MacroIndicatorsRequest,
    NewsRequest,
    PredictionMarketRequest,
    SingleAnalystRequest,
    StockDataRequest,
)
from tradingagents.api.schemas.response import (
    AnalystReportResponse,
    AnalyzeResponse,
    FundamentalsResponse,
    GlobalNewsResponse,
    IndicatorPoint,
    IndicatorSeries,
    IndicatorsResponse,
    MacroIndicatorsResponse,
    NewsResponse,
    OHLCVBar,
    PortfolioDecisionResponse,
    PredictionMarketsResponse,
    PriceHistoryResponse,
    ResearchPlanResponse,
    SentimentReportResponse,
    StockDataResponse,
    TraderProposalResponse,
)

__all__ = [
    # Requests
    "AnalyzeRequest",
    "SingleAnalystRequest",
    "StockDataRequest",
    "IndicatorsRequest",
    "FundamentalsRequest",
    "NewsRequest",
    "GlobalNewsRequest",
    "MacroIndicatorsRequest",
    "PredictionMarketRequest",
    # Responses
    "AnalystReportResponse",
    "AnalyzeResponse",
    "StockDataResponse",
    "OHLCVBar",
    "IndicatorsResponse",
    "IndicatorSeries",
    "IndicatorPoint",
    "PriceHistoryResponse",
    "FundamentalsResponse",
    "NewsResponse",
    "GlobalNewsResponse",
    "MacroIndicatorsResponse",
    "PredictionMarketsResponse",
    "ResearchPlanResponse",
    "TraderProposalResponse",
    "PortfolioDecisionResponse",
    "SentimentReportResponse",
    # Enums
    "AssetType",
    "AnalystType",
    # Error
    "ErrorResponse",
]
