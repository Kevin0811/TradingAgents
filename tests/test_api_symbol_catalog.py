"""SymbolCatalog and its file cache: load, TTL, atomic writes, refresh policy.

No network: the Yahoo source is replaced by a fake that returns canned
entries, raises, or blocks until released (to prove refreshes run in the
background and never hold up start-up).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.services.symbol_catalog import SymbolCatalog
from tradingagents.api.infrastructure.repositories.file_symbol_cache_repository import (
    FileSymbolCacheRepository,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def entry(symbol: str, market: str = "tw", type_: str = "equity", name: str = "") -> SymbolEntry:
    return SymbolEntry(symbol, name or symbol, "X", type_, market)


TW = [
    entry("2330.TW", name="Taiwan Semiconductor"),
    entry("0050.TW", type_="etf"),
    entry("6488.TWO"),
]


class FakeSource:
    """Fetch results per market: a list, an exception, or a gate to wait on."""

    def __init__(self, results: dict | None = None):
        self.results = results or {}
        self.calls: list[str] = []
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    def fetch(self, market: str):
        self.calls.append(market)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(5), "test never released the gate"
        result = self.results.get(market, RuntimeError(f"no data for {market}"))
        if isinstance(result, Exception):
            raise result
        return list(result)

    def source_label(self, market: str) -> str:
        return "fake source"


def make_catalog(tmp_path, source=None, clock=lambda: NOW, **kwargs):
    repo = FileSymbolCacheRepository(tmp_path)
    catalog = SymbolCatalog(repo, source or FakeSource(), clock=clock, **kwargs)
    return catalog, repo


def seed_cache(repo, market="tw", entries=TW, fetched_at=NOW - timedelta(days=1)):
    repo.save(SymbolList(market, tuple(entries), fetched_at, "seeded"))


# ---------------------------------------------------------------------------
# File cache
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFileCache:
    def test_round_trip_and_file_format(self, tmp_path):
        repo = FileSymbolCacheRepository(tmp_path)
        seed_cache(repo)

        payload = json.loads((tmp_path / "tw.json").read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert payload["market"] == "tw"
        assert payload["count"] == 3
        assert payload["source"] == "seeded"
        assert payload["fetched_at"] == (NOW - timedelta(days=1)).isoformat()
        assert payload["entries"][0] == {
            "symbol": "2330.TW",
            "name": "Taiwan Semiconductor",
            "exchange": "X",
            "type": "equity",
            "market": "tw",
        }

        loaded = repo.load("tw")
        assert loaded.entries == tuple(TW)
        assert loaded.fetched_at == NOW - timedelta(days=1)

    def test_missing_and_corrupt_files_load_as_none(self, tmp_path):
        repo = FileSymbolCacheRepository(tmp_path)
        assert repo.load("us") is None
        (tmp_path / "us.json").write_text("{not json", encoding="utf-8")
        assert repo.load("us") is None
        (tmp_path / "us.json").write_text(json.dumps({"version": 99}), encoding="utf-8")
        assert repo.load("us") is None

    def test_write_is_atomic(self, tmp_path, monkeypatch):
        repo = FileSymbolCacheRepository(tmp_path)
        seed_cache(repo)
        before = (tmp_path / "tw.json").read_bytes()

        def failing_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            seed_cache(repo, entries=[entry("9999.TW")])

        # The old file is intact and no temp file is left behind.
        assert (tmp_path / "tw.json").read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["tw.json"]

    def test_save_goes_through_a_temp_file_in_the_same_dir(self, tmp_path, monkeypatch):
        repo = FileSymbolCacheRepository(tmp_path / "symbols")
        seen = []
        real_replace = os.replace

        def spy(src, dst):
            seen.append((os.path.dirname(src), os.path.basename(src), str(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        seed_cache(repo)

        ((src_dir, src_name, dst),) = seen
        assert src_dir == str(tmp_path / "symbols")
        assert src_name.startswith(".tw.") and src_name.endswith(".tmp")
        assert dst == str(tmp_path / "symbols" / "tw.json")


# ---------------------------------------------------------------------------
# Load, TTL, status
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadAndTtl:
    def test_start_loads_cached_lists(self, tmp_path):
        catalog, repo = make_catalog(tmp_path, auto_refresh=False)
        seed_cache(repo)

        catalog.start()

        assert catalog.is_loaded("tw")
        assert catalog.status("tw") == "ready"
        assert not catalog.is_loaded("us")
        assert catalog.status("us") == "unavailable"

    def test_stale_after_ttl(self, tmp_path):
        clock = {"now": NOW}
        catalog, repo = make_catalog(tmp_path, clock=lambda: clock["now"], auto_refresh=False)
        seed_cache(repo, fetched_at=NOW)
        catalog.start()

        assert not catalog.is_stale("tw")
        clock["now"] = NOW + timedelta(days=7, seconds=1)
        assert catalog.is_stale("tw")
        # A stale list is still served.
        assert catalog.status("tw") == "ready"
        assert catalog.search("tw", "2330")[1] == 1

    def test_ttl_is_configurable(self, tmp_path):
        catalog, repo = make_catalog(tmp_path, ttl=timedelta(days=1), auto_refresh=False)
        seed_cache(repo, fetched_at=NOW - timedelta(days=2))
        catalog.start()
        assert catalog.is_stale("tw")


# ---------------------------------------------------------------------------
# Refresh policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefresh:
    def test_failed_refresh_keeps_the_old_list(self, tmp_path, caplog):
        source = FakeSource({"tw": RuntimeError("Yahoo said 429")})
        catalog, repo = make_catalog(tmp_path, source, auto_refresh=False)
        seed_cache(repo)
        before = (tmp_path / "tw.json").read_bytes()
        catalog.start()

        assert catalog.refresh_market("tw") is False

        assert [e.symbol for e in catalog.get_list("tw").entries] == [e.symbol for e in TW]
        assert (tmp_path / "tw.json").read_bytes() == before
        assert "429" in catalog.last_error("tw")
        assert "keeping the 3-entry list" in caplog.text

    def test_successful_refresh_swaps_and_persists(self, tmp_path):
        source = FakeSource({"tw": [entry("2317.TW")]})
        catalog, repo = make_catalog(tmp_path, source, auto_refresh=False)
        seed_cache(repo)
        catalog.start()

        assert catalog.refresh_market("tw") is True

        assert [e.symbol for e in catalog.get_list("tw").entries] == ["2317.TW"]
        assert catalog.get_list("tw").fetched_at == NOW
        assert [e.symbol for e in repo.load("tw").entries] == ["2317.TW"]
        assert repo.load("tw").source == "fake source"

    def test_start_refreshes_missing_and_stale_markets_in_the_background(self, tmp_path):
        source = FakeSource({m: [entry(f"{m.upper()}1", m)] for m in ("us", "jp", "crypto", "fx")})
        source.results["tw"] = [entry("2317.TW")]
        source.gate = threading.Event()
        catalog, repo = make_catalog(tmp_path, source)
        seed_cache(repo, market="jp", fetched_at=NOW)  # fresh: not refetched
        seed_cache(repo, market="tw", fetched_at=NOW - timedelta(days=30))  # stale

        catalog.start()  # must return while the first fetch is still blocked
        assert source.entered.wait(5)
        assert catalog.status("us") == "loading"
        assert catalog.is_refreshing("tw")
        assert catalog.search("tw", "2330")[1] == 1  # old list still served meanwhile

        source.gate.set()
        assert catalog.wait_idle(5)
        # One market at a time, in a fixed order; the fresh one is skipped.
        assert source.calls == ["tw", "us", "crypto", "fx"]
        assert catalog.status("us") == "ready"
        assert [e.symbol for e in catalog.get_list("tw").entries] == ["2317.TW"]

    def test_request_refresh_does_not_queue_twice(self, tmp_path):
        source = FakeSource({"tw": TW})
        source.gate = threading.Event()
        catalog, _ = make_catalog(tmp_path, source, auto_refresh=False)

        assert catalog.request_refresh(["tw"]) == ["tw"]
        assert source.entered.wait(5)
        assert catalog.request_refresh(["tw", "tw", "bogus"]) == ["tw"]
        source.gate.set()
        assert catalog.wait_idle(5)
        # The second request came while "tw" was being fetched: that fetch serves it.
        assert source.calls == ["tw"]

    def test_stale_read_requeues_at_most_once_per_retry_window(self, tmp_path):
        clock = {"now": NOW}
        source = FakeSource({"tw": RuntimeError("down")})
        catalog, repo = make_catalog(tmp_path, source, clock=lambda: clock["now"])
        seed_cache(repo, fetched_at=NOW - timedelta(days=30))
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 1

        catalog.search("tw", "2330")  # within the retry window: no new attempt
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 1

        clock["now"] = NOW + timedelta(hours=2)
        catalog.search("tw", "2330")
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 2

    def test_nothing_is_fetched_before_start_or_with_auto_refresh_off(self, tmp_path):
        source = FakeSource()
        catalog, _ = make_catalog(tmp_path, source)
        catalog.search("tw")
        catalog.check("AAPL")
        assert catalog.wait_idle(5)
        assert source.calls == []

        off, _ = make_catalog(tmp_path, source, auto_refresh=False)
        off.start()
        off.search("tw")
        assert off.wait_idle(5)
        assert source.calls == []

    def test_shutdown_stops_the_queue(self, tmp_path):
        source = FakeSource({m: [entry("A", m)] for m in ("tw", "us")})
        source.gate = threading.Event()
        catalog, _ = make_catalog(tmp_path, source, auto_refresh=False)
        catalog.request_refresh(["tw", "us"])
        assert source.entered.wait(5)

        catalog.shutdown()
        source.gate.set()
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueries:
    @pytest.fixture
    def catalog(self, tmp_path):
        catalog, repo = make_catalog(tmp_path, auto_refresh=False)
        seed_cache(
            repo,
            entries=[
                entry("0050.TW", type_="etf", name="Yuanta Taiwan Top 50 ETF"),
                entry("2330.TW", name="Taiwan Semiconductor Manufacturing"),
                entry("2303.TW", name="United Microelectronics"),
                entry("6488.TWO", name="GlobalWafers"),
                entry("2330A.TW", name="Pref share"),
            ],
        )
        seed_cache(repo, market="us", entries=[entry("AAPL", "us", name="Apple Inc.")])
        seed_cache(
            repo,
            market="fx",
            entries=[entry("TWD=X", "fx", "currency"), entry("JPY=X", "fx", "currency")],
        )
        seed_cache(repo, market="jp", entries=[entry("7203.T", "jp"), entry("9999.S", "jp")])
        catalog.start()
        return catalog

    def test_search_exact_then_prefix_then_name(self, catalog):
        page, total = catalog.search("tw", "2330")
        assert [e.symbol for e in page] == ["2330.TW", "2330A.TW"]
        page, _ = catalog.search("tw", "2330.tw")  # case-insensitive
        assert [e.symbol for e in page] == ["2330.TW"]
        page, _ = catalog.search("tw", "taiwan")
        assert [e.symbol for e in page] == ["0050.TW", "2330.TW"]

    def test_search_type_and_paging(self, catalog):
        assert [e.symbol for e in catalog.search("tw", symbol_type="etf")[0]] == ["0050.TW"]
        page, total = catalog.search("tw", limit=2, offset=1)
        assert total == 5
        assert [e.symbol for e in page] == ["2330.TW", "2303.TW"]

    def test_get_and_suggest(self, catalog):
        assert catalog.get("AAPL").market == "us"
        assert catalog.get("NOPE") is None
        assert catalog.suggest("2331.TW", "tw")[0] == "2330.TW"
        assert len(catalog.suggest("2331.TW", "tw")) <= 3

    def test_suffixes_seen_in_the_jp_list_route_to_jp(self, catalog):
        assert catalog.market_for("1234.S") == "jp"
        assert catalog.market_for("0700.HK") is None

    @pytest.mark.parametrize(
        ("ticker", "asset_type", "supported", "market"),
        [
            ("2330.TW", "stock", True, "tw"),
            ("2331.TW", "stock", False, "tw"),
            ("aapl", "stock", True, "us"),
            ("USDTWD", "stock", True, "fx"),
            ("TWD=X", "stock", True, "fx"),
            ("USDJPY", "stock", True, "fx"),
            ("JPYTWD=X", "stock", True, "fx"),
            ("USDKRW", "stock", False, "fx"),
            ("0700.HK", "stock", None, None),  # no list covers HK
            ("^GSPC", "stock", None, None),
            ("GC=F", "stock", None, None),
            ("BTC-USD", "crypto", None, "crypto"),  # crypto list not loaded
            ("BTC", "crypto", None, "crypto"),
        ],
    )
    def test_check(self, catalog, ticker, asset_type, supported, market):
        result = catalog.check(ticker, asset_type)
        assert (result.supported, result.market) == (supported, market)

    def test_check_suggests_close_symbols(self, catalog):
        result = catalog.check("2331.TW")
        assert result.suggestions[0] == "2330.TW"
