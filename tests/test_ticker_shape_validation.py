"""A bare numeric ticker (e.g. "2330") reaches Yahoo Finance as-is and fails
deep inside the graph ("possibly delisted; no price data found") only after
a background worker has already run the job. These guard the fail-fast check
at the API boundary (tradingagents/api/schemas/validators.py) that rejects
that specific shape synchronously, on POST, instead.
"""
import pytest
from pydantic import ValidationError

from tradingagents.api.schemas.request import AnalyzeRequest
from tradingagents.api.schemas.task import TaskCreateRequest


@pytest.mark.unit
@pytest.mark.parametrize("request_cls", [AnalyzeRequest, TaskCreateRequest])
class TestTickerShapeValidation:
    def test_bare_numeric_ticker_is_rejected(self, request_cls):
        with pytest.raises(ValidationError, match="2330.TW"):
            request_cls(ticker="2330", trade_date="2026-08-07")

    def test_empty_ticker_is_rejected(self, request_cls):
        with pytest.raises(ValidationError):
            request_cls(ticker="", trade_date="2026-08-07")

    @pytest.mark.parametrize("ticker", ["2330.TW", "AAPL", "BTC-USD", "0700.HK"])
    def test_qualified_or_alphabetic_tickers_pass(self, request_cls, ticker):
        req = request_cls(ticker=ticker, trade_date="2026-08-07")
        assert req.ticker == ticker
