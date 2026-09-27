"""SymbolCatalog and its file cache: load, TTL, atomic writes, refresh policy.

No network: the Yahoo source is replaced by a fake that returns canned
entries, raises, or blocks until released (to prove refreshes run in the
background and never hold up start-up).
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.services.symbol_catalog import SymbolCatalog
from tradingagents.api.domain.symbols import REJECT_UNLISTED_TICKERS
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
        source = FakeSource({"tw": [*TW, entry("2317.TW")]})
        catalog, repo = make_catalog(tmp_path, source, auto_refresh=False)
        seed_cache(repo)
        catalog.start()

        assert catalog.refresh_market("tw") is True

        symbols = [e.symbol for e in TW] + ["2317.TW"]
        assert [e.symbol for e in catalog.get_list("tw").entries] == symbols
        assert catalog.get_list("tw").fetched_at == NOW
        assert [e.symbol for e in repo.load("tw").entries] == symbols
        assert repo.load("tw").source == "fake source"

    def test_start_refreshes_missing_and_stale_markets_in_the_background(self, tmp_path):
        source = FakeSource({m: [entry(f"{m.upper()}1", m)] for m in ("us", "jp", "crypto", "fx")})
        source.results["tw"] = [*TW, entry("2317.TW")]
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
        assert "2317.TW" in {e.symbol for e in catalog.get_list("tw").entries}

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
        catalog.decide("AAPL")
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

    def test_only_hard_coded_suffixes_route(self, catalog):
        # "9999.S" is on the jp list, but a suffix seen in a list never routes.
        assert catalog.market_for("1234.S") is None
        assert catalog.market_for("SAP.F") is None
        assert catalog.market_for("0700.HK") is None
        assert (catalog.market_for("2330.TW"), catalog.market_for("6488.TWO")) == ("tw", "tw")
        assert catalog.market_for("7203.T") == "jp"
        # Such an entry is still searchable and found by an exact lookup...
        assert [e.symbol for e in catalog.search("jp", "9999")[0]] == ["9999.S"]
        assert catalog.get("9999.S").market == "jp"
        # ...but a ticker with that suffix is uncovered: shape check only.
        decision = catalog.decide("1234.S")
        assert (decision.market, decision.supported, decision.rejected) == (None, None, False)

    @pytest.mark.parametrize(
        ("ticker", "asset_type", "supported", "market", "rejected"),
        [
            ("2330.TW", "stock", True, "tw", False),
            ("2331.TW", "stock", False, "tw", True),  # tw is enforced
            ("7203.T", "stock", True, "jp", False),
            ("7204.T", "stock", False, "jp", True),  # jp is enforced
            ("aapl", "stock", True, "us", False),
            ("APPL", "stock", False, "us", False),  # us is soft
            ("USDTWD", "stock", True, "fx", False),
            ("TWD=X", "stock", True, "fx", False),
            ("USDJPY", "stock", True, "fx", False),
            ("JPYTWD=X", "stock", True, "fx", False),
            ("USDKRW", "stock", False, "fx", False),  # fx is soft
            ("0700.HK", "stock", None, None, False),  # no list covers HK
            ("^GSPC", "stock", None, None, False),
            ("GC=F", "stock", None, None, False),
            ("BTC-USD", "crypto", None, "crypto", False),  # crypto list not loaded
            ("BTC", "crypto", None, "crypto", False),
            ("BTC", "CRYPTO", None, "crypto", False),  # asset_type is case-insensitive
        ],
    )
    def test_decide(self, catalog, ticker, asset_type, supported, market, rejected):
        result = catalog.decide(ticker, asset_type)
        assert (result.supported, result.market, result.rejected) == (supported, market, rejected)

    def test_decide_suggests_close_symbols(self, catalog):
        result = catalog.decide("2331.TW")
        assert result.suggestions[0] == "2330.TW"
        # A soft miss still carries the suggestions, as a hint.
        assert catalog.decide("APPL").suggestions == ("AAPL",)


# ---------------------------------------------------------------------------
# Cache payload validation and temp-file sweep
# ---------------------------------------------------------------------------

GOOD_ENTRY = {"symbol": "2330.TW", "name": "TSMC", "exchange": "TAI", "type": "equity"}


def _payload(**overrides):
    payload = {
        "version": 1,
        "market": "tw",
        "fetched_at": NOW.isoformat(),
        "entries": [GOOD_ENTRY],
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
class TestCachePayloadValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            [],  # not an object
            "tw",
            42,
            None,
            _payload(entries=[]),  # empty counts as missing
            _payload(entries=None),
            _payload(entries={"2330.TW": GOOD_ENTRY}),
            _payload(entries=["2330.TW"]),  # entry not an object
            _payload(entries=[GOOD_ENTRY, {"name": "no symbol", "type": "equity"}]),
            _payload(entries=[{**GOOD_ENTRY, "symbol": ""}]),
            _payload(entries=[{**GOOD_ENTRY, "type": "bond"}]),  # unknown type
            _payload(entries=[{k: v for k, v in GOOD_ENTRY.items() if k != "type"}]),
            _payload(entries=[{**GOOD_ENTRY, "name": 7}]),
            _payload(fetched_at=None),
            _payload(fetched_at=1727000000),
            _payload(fetched_at="yesterday"),
        ],
    )
    def test_bad_payload_loads_as_none(self, tmp_path, payload):
        (tmp_path / "tw.json").write_text(json.dumps(payload), encoding="utf-8")
        assert FileSymbolCacheRepository(tmp_path).load("tw") is None

    def test_missing_optional_fields_are_fine(self, tmp_path):
        payload = _payload(entries=[{"symbol": "2330.TW", "type": "equity", "name": None}])
        del payload["market"]
        (tmp_path / "tw.json").write_text(json.dumps(payload), encoding="utf-8")
        loaded = FileSymbolCacheRepository(tmp_path).load("tw")
        assert loaded.entries == (SymbolEntry("2330.TW", "", "", "equity", "tw"),)

    def test_startup_survives_every_bad_file(self, tmp_path):
        for market, payload in zip(
            ("tw", "us", "jp", "crypto", "fx"),
            [[], "x", _payload(entries=[]), _payload(entries=[{"type": "equity"}]), None],
            strict=True,
        ):
            (tmp_path / f"{market}.json").write_text(json.dumps(payload), encoding="utf-8")
        catalog, _ = make_catalog(tmp_path, auto_refresh=False)

        catalog.start()

        assert all(catalog.status(m) == "unavailable" for m in ("tw", "us", "jp", "crypto", "fx"))

    def test_startup_survives_a_repository_that_raises(self, tmp_path):
        class Exploding(FileSymbolCacheRepository):
            def load(self, market):
                raise RuntimeError("boom")

        catalog = SymbolCatalog(Exploding(tmp_path), FakeSource(), auto_refresh=False)
        catalog.start()
        assert not catalog.is_loaded("tw")


@pytest.mark.unit
class TestTempFileSweep:
    def test_start_removes_orphaned_market_temp_files_only(self, tmp_path):
        old = time.time() - 3600
        names = {
            ".tw.abc123.tmp": old,  # orphaned by a crashed write: removed
            ".crypto.x.tmp": old,  # removed
            ".us.fresh.tmp": time.time(),  # may be a write in progress: kept
            ".hk.abc.tmp": old,  # not a market of ours: kept
            "notes.tmp": old,  # kept
        }
        for name, mtime in names.items():
            path = tmp_path / name
            path.write_text("partial", encoding="utf-8")
            os.utime(path, (mtime, mtime))
        seed_cache(FileSymbolCacheRepository(tmp_path))
        catalog, _ = make_catalog(tmp_path, auto_refresh=False)

        catalog.start()

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            ".hk.abc.tmp",
            ".us.fresh.tmp",
            "notes.tmp",
            "tw.json",
        ]
        assert catalog.is_loaded("tw")

    def test_sweep_of_a_missing_dir_is_a_no_op(self, tmp_path):
        assert FileSymbolCacheRepository(tmp_path / "nope").sweep_temp_files() == 0


# ---------------------------------------------------------------------------
# Truncated-refresh guard
# ---------------------------------------------------------------------------


def _tw_list(n_equity: int, n_etf: int = 0) -> list[SymbolEntry]:
    return [entry(f"{1000 + i}.TW") for i in range(n_equity)] + [
        entry(f"00{50 + i}.TW", type_="etf") for i in range(n_etf)
    ]


@pytest.mark.unit
class TestRefreshGuard:
    def _catalog(self, tmp_path, old, new):
        source = FakeSource({"tw": new})
        catalog, repo = make_catalog(tmp_path, source, auto_refresh=False)
        seed_cache(repo, entries=old)
        catalog.start()
        return catalog, repo

    def test_a_list_shrunk_below_80_percent_is_refused(self, tmp_path, caplog):
        catalog, repo = self._catalog(tmp_path, _tw_list(9, 1), _tw_list(6, 1))
        before = (tmp_path / "tw.json").read_bytes()

        assert catalog.refresh_market("tw") is False

        assert len(catalog.get_list("tw").entries) == 10
        assert (tmp_path / "tw.json").read_bytes() == before
        assert "7 entries, fewer than 80% of the previous 10" in catalog.last_error("tw")
        assert "keeping the 10-entry list" in caplog.text

    def test_80_percent_is_accepted(self, tmp_path):
        catalog, repo = self._catalog(tmp_path, _tw_list(9, 1), _tw_list(7, 1))
        assert catalog.refresh_market("tw") is True
        assert len(repo.load("tw").entries) == 8
        assert catalog.last_error("tw") is None

    def test_an_exchange_type_group_gone_empty_is_refused(self, tmp_path):
        # Plenty of rows, but every ETF is gone: the ETF query came back empty.
        catalog, _ = self._catalog(tmp_path, _tw_list(9, 1), _tw_list(12))

        assert catalog.refresh_market("tw") is False

        assert "no rows any more for X/etf (had 1)" in catalog.last_error("tw")
        assert any(e.type == "etf" for e in catalog.get_list("tw").entries)

    def test_the_first_list_is_accepted_whatever_its_size(self, tmp_path):
        source = FakeSource({"tw": _tw_list(1)})
        catalog, _ = make_catalog(tmp_path, source, auto_refresh=False)
        assert catalog.refresh_market("tw") is True

    def test_a_good_refresh_clears_the_last_error(self, tmp_path):
        catalog, _ = self._catalog(tmp_path, _tw_list(9, 1), _tw_list(1))
        assert catalog.refresh_market("tw") is False
        catalog._source.results["tw"] = _tw_list(10, 1)
        assert catalog.refresh_market("tw") is True
        assert catalog.last_error("tw") is None


# ---------------------------------------------------------------------------
# Miss-triggered refresh and the manual-refresh cooldown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefreshOnMiss:
    def _catalog(self, tmp_path, fetched_at, results=None):
        clock = {"now": NOW}
        source = FakeSource(results or {"tw": RuntimeError("down")})
        catalog, repo = make_catalog(tmp_path, source, clock=lambda: clock["now"])
        seed_cache(repo, fetched_at=fetched_at)
        catalog.start()  # also queues the other (missing) markets
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 0  # not stale by the 7-day TTL
        return catalog, source, clock

    def test_a_miss_in_a_list_older_than_a_day_queues_a_refresh(self, tmp_path):
        catalog, source, clock = self._catalog(tmp_path, NOW - timedelta(days=2))

        catalog.decide("2331.TW")
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 1

        # Rate-limited per market, like any automatic attempt.
        catalog.decide("2332.TW")
        catalog.lookup("2333.TW")
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 1

        clock["now"] = NOW + timedelta(hours=2)
        catalog.lookup("2333.TW")  # a plain lookup miss counts too
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 2

    def test_a_miss_in_a_fresh_list_or_a_hit_queues_nothing(self, tmp_path):
        catalog, source, _ = self._catalog(tmp_path, NOW - timedelta(hours=2))
        catalog.decide("2331.TW")
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 0

        old, old_source, _ = self._catalog(tmp_path / "old", NOW - timedelta(days=2))
        old.decide("2330.TW")  # on the list
        assert old.wait_idle(5)
        assert old_source.calls.count("tw") == 0

    def test_the_refresh_picks_up_a_new_listing(self, tmp_path):
        results = {"tw": [*TW, entry("7777.TW")]}
        catalog, _, _ = self._catalog(tmp_path, NOW - timedelta(days=2), results)

        assert catalog.decide("7777.TW").rejected  # not listed yet...
        assert catalog.wait_idle(5)
        assert catalog.decide("7777.TW").supported is True  # ...and now it is


@pytest.mark.unit
class TestManualRefreshCooldown:
    def test_recently_fetched_markets_are_skipped_unless_forced(self, tmp_path):
        source = FakeSource({"tw": TW, "us": [entry("A", "us")]})
        catalog, repo = make_catalog(tmp_path, source, auto_refresh=False)
        seed_cache(repo, fetched_at=NOW - timedelta(minutes=10))
        seed_cache(
            repo, market="us", entries=[entry("A", "us")], fetched_at=NOW - timedelta(minutes=31)
        )
        catalog.start()

        assert catalog.request_manual_refresh(["tw", "us", "jp"]) == (["us", "jp"], ["tw"])
        assert catalog.wait_idle(5)
        assert source.calls == ["us", "jp"]

        assert catalog.request_manual_refresh(["tw"], force=True) == (["tw"], [])
        assert catalog.wait_idle(5)
        assert source.calls == ["us", "jp", "tw"]

    def test_a_successful_refresh_starts_the_cooldown(self, tmp_path):
        source = FakeSource({"us": [entry("A", "us")]})
        catalog, _ = make_catalog(tmp_path, source, auto_refresh=False)
        assert catalog.request_manual_refresh(["us"]) == (["us"], [])
        assert catalog.wait_idle(5)
        assert catalog.request_manual_refresh(["us"]) == ([], ["us"])


# ---------------------------------------------------------------------------
# Per-market reject policy and suffix-less TW/JP codes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRejectPolicy:
    def test_the_policy_constant(self):
        assert dict(REJECT_UNLISTED_TICKERS) == {
            "tw": True,
            "jp": True,
            "us": False,
            "crypto": False,
            "fx": False,
        }

    def _catalog(self, tmp_path, **kwargs):
        catalog, repo = make_catalog(tmp_path, auto_refresh=False, **kwargs)
        seed_cache(repo)
        seed_cache(repo, market="us", entries=[entry("AAPL", "us")])
        seed_cache(repo, market="crypto", entries=[entry("BTC-USD", "crypto", "crypto")])
        seed_cache(repo, market="fx", entries=[entry("TWD=X", "fx", "currency")])
        catalog.start()
        return catalog

    @pytest.mark.parametrize(
        ("ticker", "asset_type", "enforced"),
        [
            ("2331.TW", "stock", True),
            ("APPL", "stock", False),
            ("DOGE", "crypto", False),
            ("USDKRW", "stock", False),
        ],
    )
    def test_misses(self, tmp_path, ticker, asset_type, enforced):
        decision = self._catalog(tmp_path).decide(ticker, asset_type)
        assert decision.supported is False
        assert decision.status == "ready"
        assert (decision.enforced, decision.rejected) == (enforced, enforced)

    def test_an_enforced_market_without_a_list_falls_back(self, tmp_path):
        decision = self._catalog(tmp_path).decide("7203.T")
        assert (decision.market, decision.supported, decision.enforced) == ("jp", None, True)
        assert (decision.status, decision.rejected) == ("unavailable", False)

    def test_the_policy_can_be_tightened_per_market(self, tmp_path):
        catalog = self._catalog(tmp_path, reject_unlisted={**REJECT_UNLISTED_TICKERS, "us": True})
        assert catalog.decide("APPL").rejected
        assert catalog.decide("AAPL").supported is True

    def test_a_hit_carries_the_entry(self, tmp_path):
        decision = self._catalog(tmp_path).decide("2330.tw")
        assert (decision.ticker, decision.normalized) == ("2330.tw", "2330.TW")
        assert decision.entry.symbol == "2330.TW"


@pytest.mark.unit
class TestSuffixlessLocalCodes:
    @pytest.fixture
    def catalog(self, tmp_path):
        catalog, repo = make_catalog(tmp_path, auto_refresh=False)
        seed_cache(
            repo,
            entries=[entry("2330.TW"), entry("2881A.TW"), entry("6488.TWO"), entry("00679B.TW")],
        )
        seed_cache(repo, market="jp", entries=[entry("7203.T", "jp"), entry("130A.T", "jp")])
        seed_cache(repo, market="us", entries=[entry("AAPL", "us")])
        catalog.start()
        return catalog

    @pytest.mark.parametrize(
        ("ticker", "market", "listed"),
        [
            ("2330", "tw", ("2330.TW",)),
            ("6488", "tw", ("6488.TWO",)),
            ("2881A", "tw", ("2881A.TW",)),  # not "a miss on the us list"
            ("00679b", "tw", ("00679B.TW",)),
            ("7203", "jp", ("7203.T",)),
            ("130A", "jp", ("130A.T",)),
            (" 130a ", "jp", ("130A.T",)),
        ],
    )
    def test_a_listed_suffixed_symbol_is_named(self, catalog, ticker, market, listed):
        decision = catalog.decide(ticker)
        assert (decision.market, decision.suggestions) == (market, listed)
        assert decision.supported is False and decision.rejected
        assert decision.reason == "missing_suffix"
        assert catalog.lookup(ticker).suggestions == listed

    def test_a_bare_numeric_code_is_rejected_even_when_unlisted(self, catalog):
        decision = catalog.decide("999999")
        assert (decision.market, decision.status, decision.suggestions) == (None, "uncovered", ())
        assert decision.rejected

    def test_an_unlisted_letter_code_falls_to_the_soft_us_check(self, catalog):
        decision = catalog.decide("2882A")
        assert (decision.market, decision.supported, decision.rejected) == ("us", False, False)

    def test_crypto_requests_are_not_read_as_local_codes(self, catalog):
        assert catalog.decide("7203", "crypto").market == "crypto"
