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
    IndicatorName,
    ReportFreq,
    ReportType,
)
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
    AnalyzeResponse,
    FundamentalsResponse,
    GlobalNewsResponse,
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
class TestReportFreqEnum:
    def test_values(self):
        assert {f.value for f in ReportFreq} == {"annual", "quarterly"}


@pytest.mark.unit
class TestIndicatorNameEnum:
    def test_rsi_value(self):
        assert IndicatorName.RSI.value == "rsi"

    def test_macd_value(self):
        assert IndicatorName.MACD.value == "macd"

    def test_values_match_vendor_keys(self):
        """The enum must track the vendors' own indicator keys exactly.

        A friendly-looking alias here (e.g. 'bollinger_upper') would be
        accepted by the API and then rejected by the vendor.
        """
        expected = {
            "close_50_sma", "close_200_sma", "close_10_ema",
            "macd", "macds", "macdh", "rsi",
            "boll", "boll_ub", "boll_lb",
            "atr", "vwma", "mfi",
        }
        assert {i.value for i in IndicatorName} == expected


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

    def test_default_look_back_days(self):
        req = IndicatorsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.look_back_days == 30

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

    def test_default_freq_quarterly(self):
        req = FundamentalsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.freq == ReportFreq.QUARTERLY


@pytest.mark.unit
class TestNewsRequest:
    def test_required_fields(self):
        req = NewsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.ticker == "AAPL"
        assert req.trade_date == "2026-06-01"

    def test_default_look_back_days(self):
        req = NewsRequest(ticker="AAPL", trade_date="2026-06-01")
        assert req.look_back_days == 7

    def test_look_back_days_lower_bound(self):
        with pytest.raises(ValidationError):
            NewsRequest(ticker="AAPL", trade_date="2026-06-01", look_back_days=0)


@pytest.mark.unit
class TestGlobalNewsRequest:
    def test_required_fields(self):
        req = GlobalNewsRequest(trade_date="2026-06-01")
        assert req.trade_date == "2026-06-01"

    def test_optional_fields_default_to_none(self):
        """None means "inherit the configured default", not "zero"."""
        req = GlobalNewsRequest(trade_date="2026-06-01")
        assert req.look_back_days is None
        assert req.limit is None


@pytest.mark.unit
class TestMacroIndicatorsRequest:
    def test_required_fields(self):
        req = MacroIndicatorsRequest(indicator="cpi", trade_date="2026-06-01")
        assert req.indicator == "cpi"
        assert req.trade_date == "2026-06-01"

    def test_indicator_is_required(self):
        with pytest.raises(ValidationError):
            MacroIndicatorsRequest(trade_date="2026-06-01")

    def test_raw_fred_series_id_accepted(self):
        req = MacroIndicatorsRequest(indicator="CPIAUCSL", trade_date="2026-06-01")
        assert req.indicator == "CPIAUCSL"

    def test_default_look_back_days_none(self):
        req = MacroIndicatorsRequest(indicator="cpi", trade_date="2026-06-01")
        assert req.look_back_days is None


@pytest.mark.unit
class TestPredictionMarketRequest:
    def test_required_fields(self):
        req = PredictionMarketRequest(topic="Fed rate cut")
        assert req.topic == "Fed rate cut"
        assert req.limit is None

    def test_topic_is_required(self):
        with pytest.raises(ValidationError):
            PredictionMarketRequest()


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
    def _resp(self, **rows):
        return StockDataResponse(
            ticker="AAPL",
            symbol="AAPL",
            start_date="2026-06-01",
            end_date="2026-06-01",
            count=1,
            **rows,
        )

    def test_rows_are_records(self):
        resp = self._resp(rows=[{"date": "2026-06-01", "close": 100.5}])
        assert resp.rows[0].date == "2026-06-01"
        assert resp.rows[0].close == 100.5
        assert resp.rows[0].open is None

    def test_vendor_extra_columns_are_preserved(self):
        """Dropping unknown vendor columns would lose requested data."""
        resp = self._resp(rows=[{"date": "2026-06-01", "dividends": 0.24}])
        assert resp.rows[0].model_dump()["dividends"] == 0.24

    def test_rows_default_to_empty(self):
        assert self._resp().rows == []

    def test_date_is_required_on_a_bar(self):
        with pytest.raises(ValidationError):
            self._resp(rows=[{"close": 100.5}])


@pytest.mark.unit
class TestIndicatorsResponse:
    def _resp(self, **kwargs):
        return IndicatorsResponse(
            ticker="AAPL",
            symbol="AAPL",
            trade_date="2026-06-01",
            look_back_days=30,
            **kwargs,
        )

    def test_series_are_records(self):
        resp = self._resp(
            indicators=[
                {
                    "name": "rsi",
                    "description": "RSI: ...",
                    "points": [{"date": "2026-06-01", "value": 43.2}],
                }
            ]
        )
        assert resp.indicators[0].name == "rsi"
        assert resp.indicators[0].points[0].value == 43.2

    def test_point_value_may_be_null_with_a_note(self):
        resp = self._resp(
            indicators=[
                {
                    "name": "rsi",
                    "points": [{"date": "2026-06-01", "note": "Not a trading day"}],
                }
            ]
        )
        point = resp.indicators[0].points[0]
        assert point.value is None
        assert point.note == "Not a trading day"

    def test_collections_default_to_empty(self):
        resp = self._resp()
        assert resp.indicators == []
        assert resp.errors == []


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
class TestGlobalNewsResponse:
    def test_keyed_by_date_not_ticker(self):
        resp = GlobalNewsResponse(trade_date="2026-06-01", news_items="## News")
        assert resp.trade_date == "2026-06-01"
        assert resp.news_items == "## News"


@pytest.mark.unit
class TestMacroIndicatorsResponse:
    def test_keyed_by_indicator_not_country(self):
        resp = MacroIndicatorsResponse(indicator="cpi", data="## CPI")
        assert resp.indicator == "cpi"
        assert resp.data == "## CPI"


@pytest.mark.unit
class TestPredictionMarketsResponse:
    def test_keyed_by_topic_not_ticker(self):
        resp = PredictionMarketsResponse(topic="Fed rate cut", markets="## Markets")
        assert resp.topic == "Fed rate cut"
        assert resp.markets == "## Markets"


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