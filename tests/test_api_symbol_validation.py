"""Request ticker validation against the supported-symbols list.

With a loaded tw or jp list, an off-list ticker fails the request with a 422
that names close symbols. A miss in the us, crypto or fx list is allowed (the
per-market policy), with the suggestions only reported by GET /symbols/check.
When the ticker's market has no loaded list (cold start, failed fetch without
a cache) or no list covers it (HK, indices, futures), the old shape-only check
applies, so an empty cache never blocks an analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tradingagents.api.app import create_app
from tradingagents.api.dependencies import get_analysis_service
from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.services.symbol_catalog import (
    SymbolCatalog,
    get_active_catalog,
    set_active_catalog,
)
from tradingagents.api.domain.symbols import (
    api_forex_symbol,
    bare_crypto_to_pair,
    classify_market,
    list_key,
)
from tradingagents.api.infrastructure.repositories.file_symbol_cache_repository import (
    FileSymbolCacheRepository,
)
from tradingagents.api.schemas.request import AnalyzeRequest
from tradingagents.api.schemas.task import TaskCreateRequest
from tradingagents.dataflows.symbols import normalize_symbol

DATE = "2026-09-25"
LISTS = {
    "tw": ["2330.TW", "2303.TW", "0050.TW", "6488.TWO", "9958A.TW"],
    "jp": ["7203.T", "130A.T"],
    "us": ["AAPL", "SPY", "BRK-B"],
    "crypto": ["BTC-USD", "ETH-USD", "PEPE24478-USD"],
    "fx": ["TWD=X", "JPY=X", "EUR=X"],
}
TYPES = {"tw": "equity", "jp": "equity", "us": "equity", "crypto": "crypto", "fx": "currency"}


class _NoSource:
    def fetch(self, market):  # pragma: no cover - never called
        raise AssertionError("no network in tests")

    def source_label(self, market):  # pragma: no cover
        return ""


@pytest.fixture
def catalog(tmp_path):
    repo = FileSymbolCacheRepository(tmp_path)
    for market, symbols in LISTS.items():
        entries = tuple(SymbolEntry(s, s, "X", TYPES[market], market) for s in symbols)
        repo.save(SymbolList(market, entries, datetime.now(timezone.utc)))
    catalog = SymbolCatalog(repo, _NoSource(), auto_refresh=False)
    catalog.start()
    set_active_catalog(catalog)
    yield catalog
    set_active_catalog(None)


@pytest.fixture
def no_catalog():
    set_active_catalog(None)
    yield
    set_active_catalog(None)


REQUESTS = [AnalyzeRequest, TaskCreateRequest]


@pytest.mark.unit
@pytest.mark.parametrize("request_cls", REQUESTS)
class TestWithLoadedList:
    @pytest.mark.parametrize(
        ("ticker", "asset_type"),
        [
            ("2330.TW", "stock"),
            ("6488.TWO", "stock"),
            ("AAPL", "stock"),
            ("BRK-B", "stock"),
            ("BTC-USD", "crypto"),
            ("BTCUSD", "crypto"),  # normalize_symbol -> BTC-USD
            ("BTC", "crypto"),  # bare base read as the coin
            ("eth-usdt", "crypto"),  # USDT pair -> ETH-USD
            ("TWD=X", "stock"),
            ("USDJPY", "stock"),  # core forex rule -> USDJPY=X == JPY=X
            ("EURUSD", "stock"),
            ("0700.HK", "stock"),  # no list covers HK: shape check only
            ("^GSPC", "stock"),
            ("GC=F", "stock"),
            ("XAUUSD", "stock"),  # alias -> GC=F
            ("SAP.F", "stock"),  # a suffix outside tw/jp never routes: uncovered
            # Soft markets: a miss is allowed.
            ("APPL", "stock"),
            ("ZZZZ", "stock"),
            ("BTX-USD", "crypto"),
            ("DOGE", "crypto"),
            ("USDKRW", "stock"),
            ("7203.T", "STOCK"),  # asset_type is case-insensitive
        ],
    )
    def test_on_list_or_uncovered_passes(self, catalog, request_cls, ticker, asset_type):
        req = request_cls(ticker=ticker, trade_date=DATE, asset_type=asset_type)
        assert req.ticker == ticker  # the request keeps what the caller sent

    @pytest.mark.parametrize(
        ("ticker", "asset_type", "market", "suggestion"),
        [
            ("2331.TW", "stock", "tw", "2330.TW"),
            ("6489.TWO", "stock", "tw", "6488.TWO"),
            ("7204.T", "stock", "jp", "7203.T"),
            ("9999.T", "Stock", "jp", None),
            # Suffix-less codes whose suffixed symbol is listed.
            ("9958A", "stock", "tw", "9958A.TW"),
            ("130A", "stock", "jp", "130A.T"),
        ],
    )
    def test_off_list_is_rejected_with_suggestions(
        self, catalog, request_cls, ticker, asset_type, market, suggestion
    ):
        with pytest.raises(ValidationError) as excinfo:
            request_cls(ticker=ticker, trade_date=DATE, asset_type=asset_type)
        (error,) = excinfo.value.errors()
        assert error["type"] == "ticker_not_supported"
        assert error["ctx"]["market"] == market
        assert "GET /symbols" in error["msg"]
        if suggestion:
            assert error["ctx"]["suggestions"][0] == suggestion
            assert suggestion in error["msg"]

    def test_bare_digits_still_rejected_and_point_at_the_listed_symbol(self, catalog, request_cls):
        with pytest.raises(ValidationError, match=r"On the supported list: 2330\.TW\."):
            request_cls(ticker="2330", trade_date=DATE)
        with pytest.raises(ValidationError, match="bare numeric"):
            request_cls(ticker="999999", trade_date=DATE)

    def test_usdtwd_is_mapped_to_the_yahoo_form(self, catalog, request_cls):
        assert request_cls(ticker="usdtwd", trade_date=DATE).ticker == "USDTWD=X"

    def test_suffixless_code_error_names_the_listed_symbol(self, catalog, request_cls):
        with pytest.raises(ValidationError) as excinfo:
            request_cls(ticker="9958A", trade_date=DATE)
        (error,) = excinfo.value.errors()
        assert error["ctx"] == {
            "ticker": "9958A",
            "market": "tw",
            "suggestions": ["9958A.TW"],
            "hint": " It needs an exchange suffix; listed as: 9958A.TW.",
        }

    def test_asset_type_is_case_insensitive(self, catalog, request_cls):
        req = request_cls(ticker="BTC", trade_date=DATE, asset_type=" CRYPTO ")
        assert getattr(req.asset_type, "value", req.asset_type) == "crypto"


@pytest.mark.unit
@pytest.mark.parametrize("request_cls", REQUESTS)
class TestFallbackWithoutList:
    @pytest.mark.parametrize("ticker", ["ZZZZ", "2331.TW", "7203.T", "NOPE-USD", "USDKRW"])
    def test_no_catalog_means_shape_check_only(self, no_catalog, request_cls, ticker):
        assert request_cls(ticker=ticker, trade_date=DATE).ticker == ticker

    def test_market_not_loaded_falls_back(self, tmp_path, request_cls):
        # Only tw has a list: any .T ticker passes, although jp is enforced.
        repo = FileSymbolCacheRepository(tmp_path)
        entries = (SymbolEntry("2330.TW", "TSMC", "TAI", "equity", "tw"),)
        repo.save(SymbolList("tw", entries, datetime.now(timezone.utc)))
        catalog = SymbolCatalog(repo, _NoSource(), auto_refresh=False)
        catalog.start()
        set_active_catalog(catalog)
        try:
            assert request_cls(ticker="0000.T", trade_date=DATE).ticker == "0000.T"
        finally:
            set_active_catalog(None)

    def test_bare_digits_rejected_without_a_list(self, no_catalog, request_cls):
        with pytest.raises(ValidationError, match="2330.TW"):
            request_cls(ticker="2330", trade_date=DATE)


@pytest.mark.unit
class TestOverHttp:
    @pytest.fixture
    def client(self, tmp_path, catalog):
        app = create_app(
            overrides={"symbols_cache_dir": str(tmp_path), "symbols_auto_refresh": False}
        )
        set_active_catalog(catalog)  # create_app registered its own, empty one
        # If validation ever let a request through, nothing may actually run.
        analysis = MagicMock()
        analysis.run_analysis.side_effect = AssertionError("request was not rejected")
        app.dependency_overrides[get_analysis_service] = lambda: analysis
        app.state.task_worker.submit = MagicMock(side_effect=AssertionError("not rejected"))
        app.state.symbol_catalog = catalog  # the check endpoint reads the same catalog
        return TestClient(app, raise_server_exceptions=False)

    @pytest.mark.parametrize("path", ["/api/v1/analyze/tasks", "/api/v1/analyze"])
    def test_off_list_ticker_is_422_with_suggestions(self, client, path):
        resp = client.post(path, json={"ticker": "2331.TW", "trade_date": DATE})

        assert resp.status_code == 422
        (error,) = resp.json()["detail"]
        assert error["type"] == "ticker_not_supported"
        assert error["ctx"] == {
            "ticker": "2331.TW",
            "market": "tw",
            "suggestions": ["2330.TW", "2303.TW"],
            "hint": " Did you mean: 2330.TW, 2303.TW?",
        }

    # One code path: an analyze request is rejected exactly when the check
    # endpoint reports supported=false and enforced=true.
    @pytest.mark.parametrize(
        ("ticker", "asset_type"),
        [
            ("2330.TW", "stock"),
            ("2331.TW", "stock"),
            ("7204.T", "stock"),
            ("0000.T", "stock"),
            ("9958A", "stock"),
            ("130A", "stock"),
            ("9959A", "stock"),
            ("APPL", "stock"),
            ("USDKRW", "stock"),
            ("DOGE", "crypto"),
            ("BTC", "CRYPTO"),
            ("0700.HK", "stock"),
            ("SAP.F", "stock"),
        ],
    )
    def test_check_endpoint_matches_the_validator(self, client, ticker, asset_type):
        check = client.get(
            "/api/v1/symbols/check", params={"ticker": ticker, "asset_type": asset_type}
        ).json()
        resp = client.post(
            "/api/v1/analyze/tasks",
            json={"ticker": ticker, "trade_date": DATE, "asset_type": asset_type},
        )
        rejected = check["supported"] is False and check["enforced"] is True
        if rejected:
            assert resp.status_code == 422
            (error,) = resp.json()["detail"]
            assert error["type"] == "ticker_not_supported"
            assert error["ctx"]["suggestions"] == check["suggestions"]
        else:
            # Not rejected: validation passed and the (mocked) worker was reached.
            assert resp.status_code == 500, resp.text

    def test_create_app_registers_an_empty_catalog(self, tmp_path, no_catalog):
        app = create_app(
            overrides={"symbols_cache_dir": str(tmp_path), "symbols_auto_refresh": False}
        )
        assert get_active_catalog() is app.state.symbol_catalog
        # Nothing loaded until the lifespan runs: validation falls back.
        assert TaskCreateRequest(ticker="ZZZZ", trade_date=DATE).ticker == "ZZZZ"


@pytest.mark.unit
class TestSymbolRules:
    def test_usdtwd_is_mapped_like_the_core_maps_usdjpy(self):
        # The core leaves USDTWD alone (TWD is not in its upstream currency set)...
        assert normalize_symbol("USDTWD") == "USDTWD"
        # ...so the API maps it the same way the core maps USDJPY.
        assert normalize_symbol("USDJPY") == "USDJPY=X"
        assert api_forex_symbol("USDTWD") == "USDTWD=X"
        assert api_forex_symbol("twdjpy") == "TWDJPY=X"
        assert api_forex_symbol("USDJPY") is None  # the core already handles it
        assert api_forex_symbol("TWDXYZ") is None  # XYZ is not a currency
        assert api_forex_symbol("TWD=X") is None  # already Yahoo form

    @pytest.mark.parametrize(
        ("raw", "key"),
        [
            ("usdtwd", "TWD=X"),
            ("USDTWD=X", "TWD=X"),
            ("TWD=X", "TWD=X"),
            ("EURUSD", "EURUSD=X"),
            ("BTCUSD", "BTC-USD"),
            ("2330.tw", "2330.TW"),
        ],
    )
    def test_list_key(self, raw, key):
        assert list_key(raw) == key

    @pytest.mark.parametrize(
        ("key", "market"),
        [
            ("2330.TW", "tw"),
            ("6488.TWO", "tw"),
            ("7203.T", "jp"),
            ("AAPL", "us"),
            ("BRK-B", "us"),
            ("BTC-USD", "crypto"),
            ("TWD=X", "fx"),
            ("EURJPY=X", "fx"),
            ("SAP.F", None),
            ("9999.S", None),
            ("BTC-EUR", None),
            ("0700.HK", None),
            ("^GSPC", None),
            ("GC=F", None),
            ("USD=X", None),
        ],
    )
    def test_classify_market(self, key, market):
        assert classify_market(key) == market


class TestBareCryptoToPair:
    """Opt-in convenience for API callers whose input is known to be a coin.

    Moved here from the core (``symbol_utils.bare_crypto_to_pair``) when the
    core was re-synced from upstream, which has no such helper.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("BTC", "BTC-USD"), ("eth", "ETH-USD"), ("  SOL  ", "SOL-USD"), ("BTC+", "BTC-USD")],
    )
    def test_resolves_bare_bases(self, raw, expected):
        assert bare_crypto_to_pair(raw) == expected

    @pytest.mark.parametrize("raw", ["AAPL", "BTC-USD", "BTCUSD", "EURUSD", "GOLD", "", "-", None])
    def test_non_bases_return_none(self, raw):
        assert bare_crypto_to_pair(raw) is None

    def test_core_normalizer_still_leaves_bare_bases_alone(self):
        # BTC is also a listed spot-bitcoin ETF, so the global normalizer must
        # not rewrite it; only the API's crypto path opts in.
        assert normalize_symbol("BTC") == "BTC"
