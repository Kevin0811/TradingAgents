"""Tests for the TradingAgents API /data router and MarketDataService.

These endpoints were entirely non-functional before: the service called the
``@tool``-decorated data functions as if they were plain callables (raising
"'StructuredTool' object is not callable"), and the arguments it passed did
not match the tools' signatures — ``trade_date`` landed in ``freq``, an OHLCV
CSV was passed as a ticker, and ``country``/``date`` parameters were invented
for tools that never accepted them.

So the assertions here deliberately focus on the *exact payload handed to each
tool*. That is the shape the bug took, and a response-only assertion would not
have caught any of it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tradingagents.api.app import create_app
from tradingagents.api.domain.services import market_data_service as mds
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.interface import NO_DATA_SENTINEL, UNAVAILABLE_SENTINEL


TOOL_PATH = "tradingagents.api.domain.services.market_data_service"


@pytest.fixture
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


OHLCV_CSV = (
    "# Stock data for BTC-USD from 2026-08-01 to 2026-08-01\n"
    "# Total records: 1\n"
    "# Data retrieved on: 2026-08-01 16:56:41\n"
    "\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-08-01,62823.3,63085.02,62823.3,62828.44,14471152640,0.0,0.0\n"
)

INDICATOR_REPORT = (
    "## rsi values from 2026-07-30 to 2026-07-31:\n"
    "\n"
    "2026-07-31: 43.24595007666022\n"
    "2026-07-30: 61.66273641699562\n"
    "\n"
    "\n"
    "RSI: Measures momentum to flag overbought/oversold conditions."
)


class _FakeTool:
    """Stand-in for a LangChain StructuredTool that records its payload."""

    def __init__(self, result="OK", error=None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def payload(self):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        return self.calls[0]


def _patch(name, tool):
    """Patch one tool symbol inside the service module."""
    return patch.object(mds, name, tool)


# ---------------------------------------------------------------------------
# GET /data/stock/{ticker}
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStockDataEndpoint:
    def test_passes_a_date_range_not_a_period(self, client):
        """The tool takes start_date/end_date; `period` must be converted."""
        tool = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/AAPL",
                params={"trade_date": "2026-06-30", "period": 30},
            )

        assert resp.status_code == 200
        assert tool.payload == {
            "symbol": "AAPL",
            "start_date": "2026-06-01",
            "end_date": "2026-06-30",
        }

    def test_returns_structured_bars_not_csv(self, client):
        tool = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/BTC",
                params={"trade_date": "2026-08-01", "period": 1},
            )

        body = resp.json()
        assert body["ticker"] == "BTC"
        assert body["symbol"] == "BTC-USD"
        assert body["count"] == 1
        assert body["rows"] == [
            {
                "date": "2026-08-01",
                "open": 62823.3,
                "high": 63085.02,
                "low": 62823.3,
                "close": 62828.44,
                "volume": 14471152640,
                "adj_close": None,
                # Vendor extras survive rather than being dropped.
                "dividends": 0.0,
                "split_coefficient": 0.0,
            }
        ]

    def test_period_one_is_a_single_day(self, client):
        """period=1 means just trade_date — the user's original repro."""
        tool = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/BTC",
                params={"trade_date": "2026-08-01", "period": 1},
            )

        assert resp.status_code == 200
        assert tool.payload["start_date"] == "2026-08-01"
        assert tool.payload["end_date"] == "2026-08-01"
        assert resp.json()["start_date"] == "2026-08-01"
        assert resp.json()["end_date"] == "2026-08-01"

    def test_bare_crypto_base_resolves_to_usd_pair(self, client):
        """A bare `BTC` means the coin at the API layer, not the ETF."""
        tool = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/BTC", params={"trade_date": "2026-08-01"}
            )

        assert tool.payload["symbol"] == "BTC-USD"
        # The response reports what was queried, not just what was asked for.
        assert resp.json()["ticker"] == "BTC"
        assert resp.json()["symbol"] == "BTC-USD"

    def test_broker_alias_is_resolved(self, client):
        tool = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/XAUUSD", params={"trade_date": "2026-08-01"}
            )

        assert tool.payload["symbol"] == "GC=F"
        assert resp.json()["symbol"] == "GC=F"

    def test_equity_ticker_is_left_alone(self, client):
        tool = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", tool):
            client.get("/api/v1/data/stock/AAPL", params={"trade_date": "2026-08-01"})

        assert tool.payload["symbol"] == "AAPL"

    def test_unparsable_report_is_a_503_not_a_silent_empty_list(self, client):
        """A genuine no-data result arrives as a sentinel, so this is a defect."""
        tool = _FakeTool("something that is not a price table")
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/AAPL", params={"trade_date": "2026-08-01"}
            )

        assert resp.status_code == 503

    def test_no_data_returns_404(self, client):
        tool = _FakeTool(error=NoMarketDataError("NOPE", "NOPE", "no rows"))
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/NOPE", params={"trade_date": "2026-08-01"}
            )

        assert resp.status_code == 404
        assert "NOPE" in resp.json()["detail"]

    def test_rate_limited_vendor_returns_503(self, client):
        tool = _FakeTool(error=VendorRateLimitError("throttled"))
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/AAPL", params={"trade_date": "2026-08-01"}
            )

        assert resp.status_code == 503

    def test_malformed_date_returns_400(self, client):
        tool = _FakeTool()
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/AAPL", params={"trade_date": "06/30/2026"}
            )

        assert resp.status_code == 400
        assert tool.calls == []


# ---------------------------------------------------------------------------
# GET /data/indicators/{ticker}
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIndicatorsEndpoint:
    def test_passes_symbol_not_ohlcv_data(self, client):
        """Regression: the service used to fetch OHLCV and pass the CSV as `symbol`."""
        tool = _FakeTool(INDICATOR_REPORT)
        with _patch("get_indicators", tool):
            resp = client.get(
                "/api/v1/data/indicators/AAPL",
                params={
                    "trade_date": "2026-08-01",
                    "indicator_names": "rsi,macd",
                    "look_back_days": 10,
                },
            )

        assert resp.status_code == 200
        assert tool.payload == {
            "symbol": "AAPL",
            "indicator": "rsi,macd",
            "curr_date": "2026-08-01",
            "look_back_days": 10,
        }

    def test_returns_structured_series(self, client):
        tool = _FakeTool(INDICATOR_REPORT)
        with _patch("get_indicators", tool):
            resp = client.get(
                "/api/v1/data/indicators/AAPL",
                params={"trade_date": "2026-08-01", "indicator_names": "rsi"},
            )

        body = resp.json()
        assert body["symbol"] == "AAPL"
        assert body["errors"] == []
        assert len(body["indicators"]) == 1
        series = body["indicators"][0]
        assert series["name"] == "rsi"
        assert series["description"].startswith("RSI: Measures momentum")
        assert series["points"] == [
            {"date": "2026-07-30", "value": 61.66273641699562, "note": None},
            {"date": "2026-07-31", "value": 43.24595007666022, "note": None},
        ]

    def test_non_trading_day_value_is_null_with_a_note(self, client):
        report = (
            "## rsi values from 2026-08-01 to 2026-08-01:\n\n"
            "2026-08-01: N/A: Not a trading day (weekend or holiday)\n\n\n"
            "RSI: Measures momentum."
        )
        tool = _FakeTool(report)
        with _patch("get_indicators", tool):
            resp = client.get(
                "/api/v1/data/indicators/AAPL",
                params={"trade_date": "2026-08-01", "indicator_names": "rsi"},
            )

        point = resp.json()["indicators"][0]["points"][0]
        assert point["value"] is None
        assert "Not a trading day" in point["note"]

    def test_omitted_names_use_the_default_set(self, client):
        tool = _FakeTool(INDICATOR_REPORT)
        with _patch("get_indicators", tool):
            client.get(
                "/api/v1/data/indicators/AAPL", params={"trade_date": "2026-08-01"}
            )

        assert tool.payload["indicator"] == ",".join(mds.DEFAULT_INDICATORS)
        assert tool.payload["look_back_days"] == 30

    def test_per_indicator_failure_lands_in_errors(self, client):
        """The tool appends unsupported-name errors as bare text; keep them visible."""
        tool = _FakeTool("Indicator adx is not supported.\n\n" + INDICATOR_REPORT)
        with _patch("get_indicators", tool):
            resp = client.get(
                "/api/v1/data/indicators/AAPL",
                params={"trade_date": "2026-08-01", "indicator_names": "adx,rsi"},
            )

        body = resp.json()
        assert body["errors"] == ["Indicator adx is not supported."]
        assert [s["name"] for s in body["indicators"]] == ["rsi"]

    def test_unsupported_indicator_returns_400(self, client):
        tool = _FakeTool(error=ValueError("Indicator adx is not supported"))
        with _patch("get_indicators", tool):
            resp = client.get(
                "/api/v1/data/indicators/AAPL",
                params={"trade_date": "2026-08-01", "indicator_names": "adx"},
            )

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /data/history/{ticker}
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPriceHistoryEndpoint:
    """The charting endpoint, which is the only one that fans out to two tools.

    Its whole reason to exist is that the bars and the indicator series come
    back over the *same* window, so the payload assertions here are mostly
    about the window arithmetic: the indicator tool counts backwards from
    ``curr_date``, so that has to be the price window's end, and its
    ``look_back_days`` has to be the width of that window.
    """

    def test_returns_bars_and_indicators_together(self, client):
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2026-07-30", "end_date": "2026-08-01"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert body["symbol"] == "AAPL"
        assert body["start_date"] == "2026-07-30"
        assert body["end_date"] == "2026-08-01"
        assert body["count"] == 1
        assert body["bars"][0]["close"] == 62828.44
        assert [s["name"] for s in body["indicators"]] == ["rsi"]
        assert body["errors"] == []

    def test_both_tools_see_the_same_window(self, client):
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2026-07-02", "end_date": "2026-08-01"},
            )

        assert stock.payload == {
            "symbol": "AAPL",
            "start_date": "2026-07-02",
            "end_date": "2026-08-01",
        }
        # The indicator window walks backwards from curr_date, so it starts
        # where the price window ends and reaches back across its full width.
        assert indicators.payload["curr_date"] == "2026-08-01"
        assert indicators.payload["look_back_days"] == 30

    def test_defaults_to_the_chart_indicator_set(self, client):
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2026-07-30", "end_date": "2026-08-01"},
            )

        # A chart needs the Bollinger envelope and the MACD signal/histogram,
        # which the summary-oriented DEFAULT_INDICATORS deliberately omits.
        assert indicators.payload["indicator"] == ",".join(mds.CHART_INDICATORS)
        assert "boll_ub" in indicators.payload["indicator"]
        assert "macds" in indicators.payload["indicator"]

    def test_named_indicators_replace_the_default_set(self, client):
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            client.get(
                "/api/v1/data/history/AAPL",
                params={
                    "start_date": "2026-07-30",
                    "end_date": "2026-08-01",
                    "indicators": "rsi, macd",
                },
            )

        assert indicators.payload["indicator"] == "rsi,macd"

    def test_empty_indicators_skips_the_tool_entirely(self, client):
        """An empty value and an absent one mean different things.

        Absent asks for the default set; empty asks for none at all. A caller
        drawing a bare price line should not pay for nine vendor calls only to
        discard the results.
        """
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={
                    "start_date": "2026-07-30",
                    "end_date": "2026-08-01",
                    "indicators": "",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["indicators"] == []
        assert indicators.calls == []
        # The bars still come back — only the indicator half was skipped.
        assert resp.json()["count"] == 1

    def test_reversed_range_is_rejected(self, client):
        stock = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", stock):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2026-08-01", "end_date": "2026-07-30"},
            )

        assert resp.status_code == 400
        # Rejected before any vendor call, not after one that returns nothing.
        assert stock.calls == []

    def test_range_beyond_the_indicator_source_is_rejected(self, client):
        """Indicators come from a fixed five-year download.

        A longer window would return bars with no indicator values attached,
        which reads as "this symbol has no RSI" rather than "you asked for more
        history than exists".
        """
        stock = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", stock):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2016-08-01", "end_date": "2026-08-01"},
            )

        assert resp.status_code == 400
        assert stock.calls == []

    def test_range_at_the_limit_is_accepted(self, client):
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2021-08-01", "end_date": "2026-07-31"},
            )

        assert resp.status_code == 200
        assert indicators.payload["look_back_days"] == mds.MAX_HISTORY_DAYS

    def test_malformed_date_is_rejected(self, client):
        stock = _FakeTool(OHLCV_CSV)
        with _patch("get_stock_data", stock):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "01/08/2026", "end_date": "2026-08-01"},
            )

        assert resp.status_code == 400
        assert stock.calls == []

    def test_unsupported_indicator_name_is_reported_not_fatal(self, client):
        """One bad name must not cost the caller the whole chart."""
        report = (
            "Indicator adx is not supported. Please choose from: [...]\n\n"
            + INDICATOR_REPORT
        )
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(report)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={
                    "start_date": "2026-07-30",
                    "end_date": "2026-08-01",
                    "indicators": "adx,rsi",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert [s["name"] for s in body["indicators"]] == ["rsi"]
        assert any("adx" in err for err in body["errors"])

    def test_non_trading_days_keep_their_explanation(self, client):
        report = (
            "## rsi values from 2026-08-01 to 2026-08-02:\n"
            "\n"
            "2026-08-02: N/A: Not a trading day (weekend or holiday)\n"
            "2026-08-01: 43.24595007666022\n"
        )
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(report)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2026-08-01", "end_date": "2026-08-02"},
            )

        points = {p["date"]: p for p in resp.json()["indicators"][0]["points"]}
        assert points["2026-08-02"]["value"] is None
        assert "Not a trading day" in points["2026-08-02"]["note"]
        assert points["2026-08-01"]["value"] == 43.24595007666022

    def test_unknown_symbol_is_404(self, client):
        stock = _FakeTool(error=NoMarketDataError("ZZZZ", "ZZZZ", "delisted"))
        with _patch("get_stock_data", stock):
            resp = client.get(
                "/api/v1/data/history/ZZZZ",
                params={"start_date": "2026-07-30", "end_date": "2026-08-01"},
            )

        assert resp.status_code == 404

    def test_unparsable_price_report_is_503_not_404(self, client):
        """An empty result raises upstream, so a report we cannot read is ours.

        Reporting it as "no data" would hide a vendor format change behind a
        status code that says the symbol does not exist.
        """
        stock = _FakeTool("not a csv at all")
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            resp = client.get(
                "/api/v1/data/history/AAPL",
                params={"start_date": "2026-07-30", "end_date": "2026-08-01"},
            )

        assert resp.status_code == 503
        # Bars failed, so the indicator call is never worth making.
        assert indicators.calls == []

    def test_vendor_sentinel_becomes_a_status_code(self, client):
        stock = _FakeTool(f"{NO_DATA_SENTINEL} nothing for ZZZZ")
        with _patch("get_stock_data", stock):
            resp = client.get(
                "/api/v1/data/history/ZZZZ",
                params={"start_date": "2026-07-30", "end_date": "2026-08-01"},
            )

        assert resp.status_code == 404

    def test_bare_crypto_base_resolves_for_both_tools(self, client):
        stock = _FakeTool(OHLCV_CSV)
        indicators = _FakeTool(INDICATOR_REPORT)
        with _patch("get_stock_data", stock), _patch("get_indicators", indicators):
            resp = client.get(
                "/api/v1/data/history/BTC",
                params={"start_date": "2026-07-30", "end_date": "2026-08-01"},
            )

        assert stock.payload["symbol"] == "BTC-USD"
        assert indicators.payload["symbol"] == "BTC-USD"
        assert resp.json()["ticker"] == "BTC"
        assert resp.json()["symbol"] == "BTC-USD"


# ---------------------------------------------------------------------------
# GET /data/fundamentals/{ticker}
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFundamentalsEndpoint:
    def test_statement_receives_freq_and_curr_date_by_name(self, client):
        """Regression: positionally, trade_date landed in the tool's `freq` slot."""
        tool = _FakeTool("balance sheet")
        with _patch("get_balance_sheet", tool):
            resp = client.get(
                "/api/v1/data/fundamentals/AAPL",
                params={
                    "trade_date": "2026-08-01",
                    "report_type": "balance_sheet",
                    "freq": "annual",
                },
            )

        assert resp.status_code == 200
        assert tool.payload == {
            "ticker": "AAPL",
            "freq": "annual",
            "curr_date": "2026-08-01",
        }

    def test_freq_defaults_to_quarterly(self, client):
        tool = _FakeTool()
        with _patch("get_cashflow", tool):
            client.get(
                "/api/v1/data/fundamentals/AAPL",
                params={"trade_date": "2026-08-01", "report_type": "cashflow"},
            )

        assert tool.payload["freq"] == "quarterly"

    def test_report_type_all_bundles_every_source(self, client):
        overview = _FakeTool("overview")
        balance = _FakeTool("balance")
        cash = _FakeTool("cash")
        income = _FakeTool("income")

        with _patch("get_fundamentals", overview), _patch(
            "get_balance_sheet", balance
        ), _patch("get_cashflow", cash), _patch("get_income_statement", income):
            resp = client.get(
                "/api/v1/data/fundamentals/AAPL", params={"trade_date": "2026-08-01"}
            )

        assert resp.status_code == 200
        assert resp.json()["data"] == {
            "fundamentals": "overview",
            "balance_sheet": "balance",
            "cashflow": "cash",
            "income_statement": "income",
        }
        # The overview tool has no `freq` parameter.
        assert overview.payload == {"ticker": "AAPL", "curr_date": "2026-08-01"}

    def test_unknown_report_type_falls_back_to_overview(self, client):
        tool = _FakeTool("overview")
        with _patch("get_fundamentals", tool):
            resp = client.get(
                "/api/v1/data/fundamentals/AAPL",
                params={"trade_date": "2026-08-01", "report_type": "nonsense"},
            )

        assert resp.status_code == 200
        assert tool.payload == {"ticker": "AAPL", "curr_date": "2026-08-01"}


# ---------------------------------------------------------------------------
# GET /data/news/{ticker} and /data/global-news
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNewsEndpoints:
    def test_ticker_news_passes_a_window(self, client):
        tool = _FakeTool("## News")
        with _patch("get_news", tool):
            resp = client.get(
                "/api/v1/data/news/AAPL",
                params={"trade_date": "2026-08-01", "look_back_days": 7},
            )

        assert resp.status_code == 200
        assert tool.payload == {
            "ticker": "AAPL",
            "start_date": "2026-07-25",
            "end_date": "2026-08-01",
        }

    def test_ticker_news_resolves_bare_crypto(self, client):
        tool = _FakeTool()
        with _patch("get_news", tool):
            client.get("/api/v1/data/news/ETH", params={"trade_date": "2026-08-01"})

        assert tool.payload["ticker"] == "ETH-USD"

    def test_global_news_is_not_ticker_or_country_scoped(self, client):
        tool = _FakeTool("## Global News")
        with _patch("get_global_news", tool):
            resp = client.get(
                "/api/v1/data/global-news", params={"trade_date": "2026-08-01"}
            )

        assert resp.status_code == 200
        # None means "inherit the configured default" in the tool itself.
        assert tool.payload == {
            "curr_date": "2026-08-01",
            "look_back_days": None,
            "limit": None,
        }
        assert resp.json() == {
            "trade_date": "2026-08-01",
            "news_items": "## Global News",
        }

    def test_global_news_forwards_explicit_overrides(self, client):
        tool = _FakeTool()
        with _patch("get_global_news", tool):
            client.get(
                "/api/v1/data/global-news",
                params={"trade_date": "2026-08-01", "look_back_days": 3, "limit": 5},
            )

        assert tool.payload["look_back_days"] == 3
        assert tool.payload["limit"] == 5


# ---------------------------------------------------------------------------
# GET /data/macro-indicators
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMacroIndicatorsEndpoint:
    def test_queries_by_indicator_not_country(self, client):
        tool = _FakeTool("## CPI")
        with _patch("get_macro_indicators", tool):
            resp = client.get(
                "/api/v1/data/macro-indicators",
                params={"indicator": "cpi", "trade_date": "2026-08-01"},
            )

        assert resp.status_code == 200
        assert tool.payload == {
            "indicator": "cpi",
            "curr_date": "2026-08-01",
            "look_back_days": None,
        }
        assert resp.json() == {"indicator": "cpi", "data": "## CPI"}

    def test_indicator_is_required(self, client):
        resp = client.get(
            "/api/v1/data/macro-indicators", params={"trade_date": "2026-08-01"}
        )
        assert resp.status_code == 422

    def test_unconfigured_vendor_returns_503(self, client):
        tool = _FakeTool(error=VendorNotConfiguredError("FRED_API_KEY missing"))
        with _patch("get_macro_indicators", tool):
            resp = client.get(
                "/api/v1/data/macro-indicators",
                params={"indicator": "cpi", "trade_date": "2026-08-01"},
            )

        # VendorNotConfiguredError is also a ValueError; it must not be
        # misreported as a 400 client error.
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /data/prediction-markets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredictionMarketsEndpoint:
    def test_queries_by_topic_not_ticker(self, client):
        tool = _FakeTool("## Markets")
        with _patch("get_prediction_markets", tool):
            resp = client.get(
                "/api/v1/data/prediction-markets",
                params={"topic": "Fed rate cut", "limit": 3},
            )

        assert resp.status_code == 200
        assert tool.payload == {"topic": "Fed rate cut", "limit": 3}
        assert resp.json() == {"topic": "Fed rate cut", "markets": "## Markets"}

    def test_topic_is_required(self, client):
        resp = client.get("/api/v1/data/prediction-markets")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Routing sentinels
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRoutingSentinels:
    """`route_to_vendor` returns prose for these, it does not raise.

    The agent path reads that prose and carries on. An HTTP caller needs a
    status code, so without translation every unknown symbol and every
    unconfigured optional vendor would be a 200 carrying an apology.
    """

    def test_no_data_sentinel_becomes_404(self, client):
        tool = _FakeTool(
            f"{NO_DATA_SENTINEL} No usable market data for 'NOTAREALTICKER' from "
            "any configured vendor. The symbol may be invalid, delisted, ..."
        )
        with _patch("get_stock_data", tool):
            resp = client.get(
                "/api/v1/data/stock/NOTAREALTICKER",
                params={"trade_date": "2026-08-01"},
            )

        assert resp.status_code == 404
        assert "NOTAREALTICKER" in resp.json()["detail"]

    def test_unavailable_sentinel_becomes_503(self, client):
        tool = _FakeTool(
            f"{UNAVAILABLE_SENTINEL} optional macro_data could not be retrieved "
            "(FRED_API_KEY environment variable is not set.). Proceed without it."
        )
        with _patch("get_macro_indicators", tool):
            resp = client.get(
                "/api/v1/data/macro-indicators",
                params={"indicator": "cpi", "trade_date": "2026-08-01"},
            )

        assert resp.status_code == 503
        assert "FRED_API_KEY" in resp.json()["detail"]

    def test_sentinel_is_caught_on_every_endpoint(self, client):
        """The check lives in one place, so news gets it too."""
        tool = _FakeTool(f"{NO_DATA_SENTINEL} nothing for 'ZZZZ'")
        with _patch("get_news", tool):
            resp = client.get(
                "/api/v1/data/news/ZZZZ", params={"trade_date": "2026-08-01"}
            )

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# The tools really are StructuredTools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolsAreNotPlainCallables:
    def test_data_tools_must_be_invoked(self):
        """Guards the original bug: these are StructuredTools, not functions.

        If someone re-imports a plain function here the service would still
        work, but a future `@tool` decoration would silently break it again.
        """
        for name in (
            "get_stock_data",
            "get_indicators",
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
            "get_news",
            "get_global_news",
            "get_macro_indicators",
            "get_prediction_markets",
        ):
            tool = getattr(mds, name)
            assert hasattr(tool, "invoke"), f"{name} has no .invoke()"
            assert not callable(tool), f"{name} is directly callable; update the service"
