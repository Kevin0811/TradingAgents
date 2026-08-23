"""Pydantic schemas for API request/response models."""

from tradingagents.api.schemas.request import (
    AnalyzeRequest,
    SingleAnalystRequest,
    StockDataRequest,
    IndicatorsRequest,
    FundamentalsRequest,
    NewsRequest,
    GlobalNewsRequest,
    MacroIndicatorsRequest,
    PredictionMarketRequest,
)
from tradingagents.api.schemas.response import (
    AnalystReportResponse,
    AnalyzeResponse,
    StockDataResponse,
    OHLCVBar,
    IndicatorsResponse,
    IndicatorSeries,
    IndicatorPoint,
    PriceHistoryResponse,
    FundamentalsResponse,
    NewsResponse,
    GlobalNewsResponse,
    MacroIndicatorsResponse,
    PredictionMarketsResponse,
    ResearchPlanResponse,
    TraderProposalResponse,
    PortfolioDecisionResponse,
    SentimentReportResponse,
)
from tradingagents.api.schemas.enums import AssetType, AnalystType
from tradingagents.api.schemas.error import ErrorResponse

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