"""Tests for parsing vendor report text into structured records.

The dataflow layer formats results as text for an LLM to read; the HTTP API
needs records. These tests pin the two report formats the parsers accept, in
both vendor flavours, so a wording change downstream fails here loudly instead
of silently emptying an API response.
"""

from __future__ import annotations

import pytest

from tradingagents.api.domain.vendor_reports import (
    parse_indicator_report,
    parse_ohlcv_csv,
)


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

YFINANCE_CSV = (
    "# Stock data for BTC-USD from 2026-08-01 to 2026-08-02\n"
    "# Total records: 2\n"
    "# Data retrieved on: 2026-08-02 16:56:41\n"
    "\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-08-01,62823.3,63085.02,62823.3,62828.44,14471152640,0.0,0.0\n"
    "2026-08-02,62828.44,63500.0,62700.0,63210.11,15003000000,0.0,0.0\n"
)

# Alpha Vantage returns bare CSV with its own column names.
ALPHA_VANTAGE_CSV = (
    "timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient\n"
    "2026-08-01,205.5,207.25,204.9,206.8,206.8,48000000,0.0,1.0\n"
)


@pytest.mark.unit
class TestParseOhlcvCsv:
    def test_skips_comment_header_and_blank_line(self):
        rows = parse_ohlcv_csv(YFINANCE_CSV)
        assert len(rows) == 2
        assert rows[0]["date"] == "2026-08-01"

    def test_numbers_are_numbers_not_strings(self):
        row = parse_ohlcv_csv(YFINANCE_CSV)[0]
        assert row["close"] == 62828.44
        assert isinstance(row["close"], float)
        # Volume has no decimal point, so it stays an integer.
        assert row["volume"] == 14471152640
        assert isinstance(row["volume"], int)

    def test_keeps_vendor_specific_columns(self):
        row = parse_ohlcv_csv(YFINANCE_CSV)[0]
        assert row["dividends"] == 0.0
        # "Stock Splits" and "split_coefficient" are the same concept.
        assert row["split_coefficient"] == 0.0

    def test_alpha_vantage_columns_are_canonicalized(self):
        row = parse_ohlcv_csv(ALPHA_VANTAGE_CSV)[0]
        assert row["date"] == "2026-08-01"  # from "timestamp"
        assert row["adj_close"] == 206.8  # from "adjusted_close"
        assert row["close"] == 206.8

    def test_blank_cells_become_none(self):
        csv_text = "Date,Open,Close\n2026-08-01,,206.8\n"
        assert parse_ohlcv_csv(csv_text)[0]["open"] is None

    def test_rows_without_a_date_are_dropped(self):
        csv_text = "Date,Close\n2026-08-01,206.8\n,\n"
        assert len(parse_ohlcv_csv(csv_text)) == 1

    @pytest.mark.parametrize("text", ["", "   ", "header only\n", "# just a comment\n"])
    def test_unusable_input_yields_no_rows(self, text):
        assert parse_ohlcv_csv(text) == []


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

YFINANCE_INDICATORS = (
    "## rsi values from 2026-07-30 to 2026-07-31:\n"
    "\n"
    "2026-07-31: 43.24595007666022\n"
    "2026-07-30: 61.66273641699562\n"
    "\n"
    "\n"
    "RSI: Measures momentum to flag overbought/oversold conditions."
    "\n\n"
    "## macd values from 2026-07-30 to 2026-07-31:\n"
    "\n"
    "2026-07-31: -1.5\n"
    "2026-07-30: N/A: Not a trading day (weekend or holiday)\n"
    "\n"
    "\n"
    "MACD: Computes momentum via differences of EMAs."
)

# Alpha Vantage upper-cases the indicator name in its heading.
ALPHA_VANTAGE_INDICATORS = (
    "## RSI values from 2026-07-31 to 2026-07-31:\n"
    "\n"
    "2026-07-31: 43.2459\n"
    "\n"
    "\n"
    "RSI: Measures momentum."
)


@pytest.mark.unit
class TestParseIndicatorReport:
    def test_splits_one_series_per_indicator(self):
        series, errors = parse_indicator_report(YFINANCE_INDICATORS)
        assert [s["name"] for s in series] == ["rsi", "macd"]
        assert errors == []

    def test_points_are_sorted_ascending_by_date(self):
        series, _ = parse_indicator_report(YFINANCE_INDICATORS)
        dates = [p["date"] for p in series[0]["points"]]
        assert dates == sorted(dates)

    def test_numeric_values_are_floats(self):
        series, _ = parse_indicator_report(YFINANCE_INDICATORS)
        assert series[0]["points"][-1]["value"] == 43.24595007666022
        assert series[0]["points"][-1]["note"] is None

    def test_placeholder_value_becomes_null_with_note(self):
        series, _ = parse_indicator_report(YFINANCE_INDICATORS)
        macd_point = series[1]["points"][0]
        assert macd_point["date"] == "2026-07-30"
        assert macd_point["value"] is None
        assert "Not a trading day" in macd_point["note"]

    def test_description_is_captured(self):
        series, _ = parse_indicator_report(YFINANCE_INDICATORS)
        assert series[0]["description"].startswith("RSI: Measures momentum")

    def test_uppercase_vendor_heading_is_normalized(self):
        series, _ = parse_indicator_report(ALPHA_VANTAGE_INDICATORS)
        assert series[0]["name"] == "rsi"

    def test_text_before_the_first_section_is_an_error(self):
        report = "Indicator adx is not supported.\n\n" + ALPHA_VANTAGE_INDICATORS
        series, errors = parse_indicator_report(report)
        assert errors == ["Indicator adx is not supported."]
        assert [s["name"] for s in series] == ["rsi"]

    def test_report_with_no_sections_is_all_error(self):
        series, errors = parse_indicator_report("Indicator adx is not supported.")
        assert series == []
        assert errors == ["Indicator adx is not supported."]

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_input_yields_nothing(self, text):
        assert parse_indicator_report(text) == ([], [])
