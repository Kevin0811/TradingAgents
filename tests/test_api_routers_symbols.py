"""The /symbols endpoints: list/search, loading state, exact lookup, refresh.

The app is built with a temp cache dir and auto-refresh off, and its Yahoo
source is swapped for a fake, so nothing here reaches the network.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tradingagents.api import app as app_module
from tradingagents.api.app import create_app
from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.services.symbol_catalog import set_active_catalog
from tradingagents.api.infrastructure.repositories.file_symbol_cache_repository import (
    FileSymbolCacheRepository,
)

BASE = "/api/v1/symbols"
FETCHED = datetime.now(timezone.utc) - timedelta(days=1)


class FakeSource:
    def __init__(self, *args, **kwargs):
        self.calls: list[str] = []
        self.gate = threading.Event()
        self.gate.set()

    def fetch(self, market):
        self.calls.append(market)
        self.gate.wait(5)
        return [SymbolEntry("NEW1", "New", "NYQ", "equity", market)]

    def source_label(self, market):
        return "fake"


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "YahooSymbolSource", FakeSource)
    repo = FileSymbolCacheRepository(tmp_path)

    def _make(markets=("tw", "us", "fx")):
        data = {
            "tw": [
                SymbolEntry("0050.TW", "Yuanta Taiwan Top 50 ETF", "TAI", "etf", "tw"),
                SymbolEntry("2303.TW", "United Microelectronics", "TAI", "equity", "tw"),
                SymbolEntry("2330.TW", "Taiwan Semiconductor Manufacturing", "TAI", "equity", "tw"),
                SymbolEntry("6488.TWO", "GlobalWafers", "TWO", "equity", "tw"),
            ],
            "us": [
                SymbolEntry("AAPL", "Apple Inc.", "NMS", "equity", "us"),
                SymbolEntry("SPY", "SPDR S&P 500 ETF Trust", "PCX", "etf", "us"),
            ],
            "fx": [
                SymbolEntry("TWD=X", "USD/TWD", "CCY", "currency", "fx"),
                SymbolEntry("JPY=X", "USD/JPY", "CCY", "currency", "fx"),
            ],
        }
        for market in markets:
            repo.save(SymbolList(market, tuple(data[market]), FETCHED, "seeded"))
        app = create_app(
            overrides={"symbols_cache_dir": str(tmp_path), "symbols_auto_refresh": False}
        )
        app.state.symbol_catalog.start()
        return TestClient(app), app

    yield _make
    set_active_catalog(None)


@pytest.mark.unit
class TestListSymbols:
    def test_list_shape(self, make_client):
        client, _ = make_client()
        resp = client.get(BASE, params={"market": "tw", "limit": 2})

        assert resp.status_code == 200
        body = resp.json()
        assert body["market"] == "tw"
        assert body["status"] == "ready"
        assert body["stale"] is False
        assert body["refreshing"] is False
        assert datetime.fromisoformat(body["fetched_at"]) == FETCHED
        assert body["total"] == 4
        assert (body["limit"], body["offset"]) == (2, 0)
        assert body["symbols"] == [
            {
                "symbol": "0050.TW",
                "name": "Yuanta Taiwan Top 50 ETF",
                "exchange": "TAI",
                "type": "etf",
                "market": "tw",
            },
            {
                "symbol": "2303.TW",
                "name": "United Microelectronics",
                "exchange": "TAI",
                "type": "equity",
                "market": "tw",
            },
        ]

    def test_search_type_and_paging(self, make_client):
        client, _ = make_client()

        def symbols(**params):
            body = client.get(BASE, params={"market": "tw", **params}).json()
            return body["total"], [s["symbol"] for s in body["symbols"]]

        assert symbols(q="23") == (2, ["2303.TW", "2330.TW"])
        assert symbols(q="taiwan") == (2, ["0050.TW", "2330.TW"])
        assert symbols(q="GLOBAL") == (1, ["6488.TWO"])
        assert symbols(type="etf") == (1, ["0050.TW"])
        assert symbols(offset=3) == (4, ["6488.TWO"])
        assert symbols(limit=0) == (4, [])

    def test_market_not_loaded_answers_empty_with_status(self, make_client):
        client, app = make_client(markets=("tw",))
        resp = client.get(BASE, params={"market": "jp"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unavailable"
        assert body["symbols"] == [] and body["total"] == 0
        assert body["fetched_at"] is None and body["stale"] is False

    def test_market_loading_while_a_fetch_is_in_flight(self, make_client):
        client, app = make_client(markets=("tw",))
        source = app.state.symbol_catalog._source
        source.gate.clear()
        client.post(f"{BASE}/refresh", params={"market": "jp"})

        body = client.get(BASE, params={"market": "jp"}).json()
        assert body["status"] == "loading"
        assert body["refreshing"] is True
        assert body["symbols"] == []

        source.gate.set()
        assert app.state.symbol_catalog.wait_idle(5)
        body = client.get(BASE, params={"market": "jp"}).json()
        assert body["status"] == "ready"
        assert [s["symbol"] for s in body["symbols"]] == ["NEW1"]

    @pytest.mark.parametrize(
        "params",
        [{}, {"market": "hk"}, {"market": "tw", "type": "bond"}, {"market": "tw", "limit": 1001}],
    )
    def test_invalid_params_are_422(self, make_client, params):
        client, _ = make_client()
        assert client.get(BASE, params=params).status_code == 422


@pytest.mark.unit
class TestGetSymbol:
    @pytest.mark.parametrize(
        ("raw", "symbol", "market"),
        [
            ("2330.TW", "2330.TW", "tw"),
            ("2330.tw", "2330.TW", "tw"),
            ("aapl", "AAPL", "us"),
            ("TWD=X", "TWD=X", "fx"),
            ("USDTWD", "TWD=X", "fx"),
            ("USDJPY=X", "JPY=X", "fx"),
        ],
    )
    def test_exact_lookup_normalises(self, make_client, raw, symbol, market):
        client, _ = make_client()
        resp = client.get(f"{BASE}/{raw}")
        assert resp.status_code == 200
        assert resp.json()["symbol"] == symbol
        assert resp.json()["market"] == market

    def test_unknown_symbol_is_404_with_suggestions(self, make_client):
        client, _ = make_client()
        resp = client.get(f"{BASE}/2331.TW")

        assert resp.status_code == 404
        body = resp.json()
        assert body["error"] == "Symbol not found"
        assert body["symbol"] == "2331.TW"
        assert body["market"] == "tw"
        assert body["suggestions"][0] == "2330.TW"
        assert len(body["suggestions"]) <= 3

    def test_uncovered_market_is_404_without_market(self, make_client):
        client, _ = make_client()
        body = client.get(f"{BASE}/0700.HK").json()
        assert body["market"] is None
        assert body["suggestions"] == []

    def test_market_not_loaded_is_503(self, make_client):
        client, _ = make_client(markets=("tw",))
        resp = client.get(f"{BASE}/7203.T")

        assert resp.status_code == 503
        assert resp.json() == {
            "error": "Symbol list not loaded",
            "detail": "The jp symbol list is not loaded yet, so '7203.T' cannot be checked.",
            "symbol": "7203.T",
            "market": "jp",
            "status": "unavailable",
        }


@pytest.mark.unit
class TestRefresh:
    def test_refresh_one_market(self, make_client):
        client, app = make_client()
        resp = client.post(f"{BASE}/refresh", params={"market": "us"})

        assert resp.status_code == 202
        assert resp.json() == {"queued": ["us"]}
        assert app.state.symbol_catalog.wait_idle(5)
        assert app.state.symbol_catalog._source.calls == ["us"]
        body = client.get(BASE, params={"market": "us"}).json()
        assert [s["symbol"] for s in body["symbols"]] == ["NEW1"]

    def test_refresh_all_markets(self, make_client):
        client, app = make_client()
        resp = client.post(f"{BASE}/refresh")

        assert resp.status_code == 202
        assert resp.json() == {"queued": ["tw", "us", "jp", "crypto", "fx"]}
        assert app.state.symbol_catalog.wait_idle(5)
        assert app.state.symbol_catalog._source.calls == ["tw", "us", "jp", "crypto", "fx"]

    def test_refresh_rejects_unknown_market(self, make_client):
        client, _ = make_client()
        assert client.post(f"{BASE}/refresh", params={"market": "hk"}).status_code == 422
