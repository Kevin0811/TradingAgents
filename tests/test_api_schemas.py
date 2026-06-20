"""Tests for the TradingAgents API request/response schemas and enums.

Covers:
- Enum values and membership
- Request schema validation (required fields, defaults, constraints)
- Response schema serialization
- Entity dataclass behavior
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradingagents.api.schemas.enums import (
    AnalystType,
    AssetType,
    Country,
    IndicatorName,
    ReportType,
)
from tradingagents.api.schemas.request import (
    AnalyzeRequest,
    FundamentalsRequest,
    IndicatorsRequest,
    MacroIndicatorsRequest,
    NewsRequest,
    PredictionMarketRequest,
    SingleAnalystRequest,
    StockDataRequest,
)
from tradingagents.api.schemas.response import (
    AnalyzeResponse,
    FundamentalsResponse,
    IndicatorsResponse,
    MacroIndicatorsResponse,
    NewsResponse,
    PredictionMarketsResponse,
    StockDataResponse,
)
from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus
from tradingagents.api.domain.entities import (
    AnalysisResult,
    AnalystReport,
    DecisionState,
    MarketData,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssetTypeEnum:
    def test_stock_value(self):
        assert AssetType.STOCK.value == "stock"

    def test_crypto_value(self):
        assert AssetType.CRYPTO.value == "crypto"

    def test_all_members(self):
        assert set(AssetType) == {AssetType.STOCK, AssetType.CRYPTO}


@pytest.mark.unit
class TestAnalystTypeEnum:
    def test_market_value(self):
        assert AnalystType.MARKET.value == "market"

    def test_sentiment_value(self):
        assert AnalystType.SENTIMENT.value == "sentiment"

    def test_news_value(self):
        assert AnalystType.NEWS.value == "news"

    def test_fundamentals_value(self):
        assert AnalystType.FUNDAMENTALS.value == "fundamentals"

    def test_all_members(self):
        expected = {
            AnalystType.MARKET,
            AnalystType.SENTIMENT,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        }
        assert set(AnalystType) == expected


@pytest.mark.unit
class TestReportTypeEnum:
    def test_all_value(self):
        assert ReportType.ALL.value == "all"

    def test_balance_sheet_value(self):
        assert ReportType.BALANCE_SHEET.value == "balance_sheet"

    def test_cashflow_value(self):
        assert ReportType.CASHFLOW.value == "cashflow"

    def test_income_statement_value(self):
        assert ReportType.INCOME_STATEMENT.value == "income_statement"


@pytest.mark.unit
class TestCountryEnum:
    def test_all_country_values(self):
        expected = {"US", "CN", "JP", "GB", "DE", "FR", "TW", "KR", "SG"}
        actual = {c.value for c in Country}
        assert actual == expected


@pytest.mark.unit
class TestIndicatorNameEnum:
    def test_rsi_value(self):
        assert IndicatorName.RSI.value == "rsi"

    def test_macd_value(self):
        assert IndicatorName.MACD.value == "macd"

    def test_bollinger_upper_value(self):
        assert IndicatorName.BOLLINGER_UPPER.value == "bollinger_upper"

    def test_volume_value(self):
        assert IndicatorName.VOLUME.value == "volume"


@pytest.mark.unit
class TestTaskStatusEnum:
    def test_all_status_values(self):
        expected = {"pending", "queued", "processing", "completed", "failed"}
        actual = {s.value for s in TaskStatus}
        assert actual == expected


# ---------------------------------------------------------------------------
# Request schema tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeRequest:
    def test_required_fields(self):
        req = AnalyzeRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_asset_type(self):
        req = AnalyzeRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.asset_type == AssetType.STOCK

    def test_default_analysts(self):
        req = AnalyzeRequest(ticker="AAPL", trade_date="2026-06-01")
        expected = [
            AnalystType.MARKET,
            AnalystType.SENTIMENT,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]
        assert req.selected_analysts == expected

    def test_default_debug_false(self):
        req = AnalyzeRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.debug is False

    def test_custom_values(self):
        req = AnalyzeRequest(
            ticker="BTC",
            trade_date="2026-06-01",
            asset_type=AssetType.CRYPTO,
            selected_analysts=[AnalystType.MARKET],
            debug=True,
        )
        assert req.ticker == "BTC"
        assert req.asset_type == AssetType.CRYPTO
        assert req.selected_analysts == [AnalystType.MARKET]
        assert req.debug is True

    def test_missing_ticker_raises(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(trade_date="2026-06-01")

    def test_missing_trade_date_raises(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(ticker="AAPL")


@pytest.mark.unit
class TestSingleAnalystRequest:
    def test_required_fields(self):
        req = SingleAnalystRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_asset_type(self):
        req = SingleAnalystRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.asset_type == AssetType.STOCK


@pytest.mark.unit
class TestStockDataRequest:
    def test_required_fields(self):
        req = StockDataRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_period(self):
        req = StockDataRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.period == 365

    def test_period_bounds_min(self):
        req = StockDataRequest(ticker="AAPL", trade_date="2026-06-01", period=1)
        assert req.period == 1

    def test_period_bounds_max(self):
        req = StockDataRequest(ticker="AAPL", trade_date="2026-06-01", period=3650)
        assert req.period == 3650

    def test_period_below_min_raises(self):
        with pytest.raises(ValidationError):
            StockDataRequest(ticker="AAPL", trade_date="2026-06-01", period=0)

    def test_period_above_max_raises(self):
        with pytest.raises(ValidationError):
            StockDataRequest(ticker="AAPL", trade_date="2026-06-01", period=3651)


@pytest.mark.unit
class TestIndicatorsRequest:
    def test_required_fields(self):
        req = IndicatorsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_indicator_names_empty(self):
        req = IndicatorsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.indicator_names == []

    def test_custom_indicator_names(self):
        req = IndicatorsRequest(
            ticker="AAPL",
            trade_date="2026-06-01",
            indicator_names=[IndicatorName.RSI, IndicatorName.MACD],
        )
        assert req.indicator_names == [IndicatorName.RSI, IndicatorName.MACD]


@pytest.mark.unit
class TestFundamentalsRequest:
    def test_required_fields(self):
        req = FundamentalsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_report_type(self):
        req = FundamentalsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.report_type == ReportType.ALL


@pytest.mark.unit
class TestNewsRequest:
    def test_required_fields(self):
        req = NewsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_country_none(self):
        req = NewsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.country is None

    def test_custom_country(self):
        req = NewsRequest(ticker="AAPL", trade_date="2026-06-01", country=Country.US)
        assert req.country == Country.US


@pytest.mark.unit
class TestMacroIndicatorsRequest:
    def test_default_country_us(self):
        req = MacroIndicatorsRequest()
        assert req.country == Country.US

    def test_custom_country(self):
        req = MacroIndicatorsRequest(country=Country.JP)
        assert req.country == Country.JP


@pytest.mark.unit
class TestPredictionMarketRequest:
    def test_required_fields(self):
        req = PredictionMarketRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"


@pytest.mark.unit
class TestTaskCreateRequest:
    def test_required_fields(self):
        req = TaskCreateRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_asset_type(self):
        req = TaskCreateRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.asset_type == "stock"

    def test_default_analysts(self):
        req = TaskCreateRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.selected_analysts == ["market", "social", "news", "fundamentals"]

    def test_default_debug_false(self):
        req = TaskCreateRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.debug is False


# ---------------------------------------------------------------------------
# Response schema tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeResponse:
    def test_serialization(self):
        resp = AnalyzeResponse(
            ticker="AAPL",
            trade_date="2026-06-01",
            asset_type="stock",
            market_report="Market report",
            signal="Buy",
        )
        data = resp.model_dump()
        assert data["ticker"] == "AAPL"
        assert data["signal"] == "Buy"
        assert data["sentiment_report"] is None

    def test_all_fields_optional(self):
        resp = AnalyzeResponse(ticker="AAPL", trade_date="2026-06-01", asset_type="stock")
        assert resp.market_report is None
        assert resp.sentiment_report is None
        assert resp.news_report is None
        assert resp.fundamentals_report is None
        assert resp.investment_plan is None
        assert resp.trader_investment_plan is None
        assert resp.final_trade_decision is None
        assert resp.signal is None


@pytest.mark.unit
class TestStockDataResponse:
    def test_data_can_be_none(self):
        resp = StockDataResponse(ticker="AAPL")
        assert resp.data is None

    def test_data_can_be_string(self):
        resp = StockDataResponse(ticker="AAPL", data="Date,Close\n2026-06-01,100")
        assert resp.data == "Date,Close\n2026-06-01,100"


@pytest.mark.unit
class TestIndicatorsResponse:
    def test_indicators_can_be_none(self):
        resp = IndicatorsResponse(ticker="AAPL")
        assert resp.indicators is None


@pytest.mark.unit
class TestFundamentalsResponse:
    def test_data_can_be_none(self):
        resp = FundamentalsResponse(ticker="AAPL")
        assert resp.data is None

    def test_data_can_be_dict(self):
        resp = FundamentalsResponse(ticker="AAPL", data={"revenue": 100})
        assert resp.data == {"revenue": 100}

    def test_data_can_be_string(self):
        resp = FundamentalsResponse(ticker="AAPL", data="CSV data")
        assert resp.data == "CSV data"


@pytest.mark.unit
class TestNewsResponse:
    def test_ticker_can_be_none(self):
        resp = NewsResponse()
        assert resp.ticker is None
        assert resp.news_items is None

    def test_news_items_can_be_list(self):
        resp = NewsResponse(ticker="AAPL", news_items=[{"title": "News"}])
        assert len(resp.news_items) == 1


@pytest.mark.unit
class TestTaskResponse:
    def test_serialization(self):
        from datetime import datetime

        now = datetime.now()
        resp = TaskResponse(
            task_id="abc-123",
            status=TaskStatus.COMPLETED,
            ticker="AAPL",
            trade_date="2026-06-01",
            asset_type="stock",
            created_at=now,
            updated_at=now,
            completed_at=now,
        )
        data = resp.model_dump(mode="json")
        assert data["task_id"] == "abc-123"
        assert data["status"] == "completed"
        assert data["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Entity dataclass tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarketData:
    def test_required_fields(self):
        md = MarketData(ticker="AAPL", ohlcv_data="Date,Close\n2026-06-01,100")
        assert md.ticker == "AAPL"
        assert md.ohlcv_data == "Date,Close\n2026-06-01,100"

    def test_default_indicators_empty(self):
        md = MarketData(ticker="AAPL", ohlcv_data="data")
        assert md.indicators == {}

    def test_default_fundamentals_none(self):
        md = MarketData(ticker="AAPL", ohlcv_data="data")
        assert md.fundamentals is None


@pytest.mark.unit
class TestAnalystReport:
    def test_required_fields(self):
        report = AnalystReport(
            ticker="AAPL",
            trade_date="2026-06-01",
            analyst_type="market",
            report_content="Report",
        )
        assert report.ticker == "AAPL"
        assert report.analyst_type == "market"
        assert report.report_content == "Report"

    def test_default_metadata_empty(self):
        report = AnalystReport(
            ticker="AAPL",
            trade_date="2026-06-01",
            analyst_type="market",
            report_content="Report",
        )
        assert report.metadata == {}


@pytest.mark.unit
class TestAnalysisResult:
    def test_required_fields(self):
        result = AnalysisResult(
            ticker="AAPL",
            trade_date="2026-06-01",
            asset_type="stock",
        )
        assert result.ticker == "AAPL"
        assert result.trade_date == "2026-06-01"
        assert result.asset_type == "stock"

    def test_to_dict_includes_all_fields(self):
        result = AnalysisResult(
            ticker="AAPL",
            trade_date="2026-06-01",
            asset_type="stock",
            market_report="Market",
            signal="Buy",
        )
        d = result.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["trade_date"] == "2026-06-01"
        assert d["asset_type"] == "stock"
        assert d["market_report"] == "Market"
        assert d["signal"] == "Buy"
        assert d["sentiment_report"] is None

    def test_all_optional_fields_default_none(self):
        result = AnalysisResult(ticker="AAPL", trade_date="2026-06-01", asset_type="stock")
        assert result.market_report is None
        assert result.sentiment_report is None
        assert result.news_report is None
        assert result.fundamentals_report is None
        assert result.investment_plan is None
        assert result.trader_investment_plan is None
        assert result.final_trade_decision is None
        assert result.signal is None


@pytest.mark.unit
class TestDecisionState:
    def test_required_fields(self):
        state = DecisionState(ticker="AAPL", trade_date="2026-06-01")
        assert state.ticker == "AAPL"
        assert state.trade_date == "2026-06-01"

    def test_default_research_plan_empty(self):
        state = DecisionState(ticker="AAPL", trade_date="2026-06-01")
        assert state.research_plan == ""

    def test_default_trader_proposal_empty(self):
        state = DecisionState(ticker="AAPL", trade_date="2026-06-01")
        assert state.trader_proposal == ""

    def test_default_portfolio_decision_empty(self):
        state = DecisionState(ticker="AAPL", trade_date="2026-06-01")
        assert state.portfolio_decision == ""

    def test_default_sentiment_report_empty(self):
        state = DecisionState(ticker="AAPL", trade_date="2026-06-01")
        assert state.sentiment_report == ""

    def test_default_raw_state_empty(self):
        state = DecisionState(ticker="AAPL", trade_date="2026-06-01")
        assert state.raw_state == {}