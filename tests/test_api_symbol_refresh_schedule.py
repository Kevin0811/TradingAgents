"""When automatic symbol-list refreshes run: idleness, the optional window, settings.

Everything runs on an injected clock and fakes: no test sleeps for real or
touches the network. Background fetches go to in-memory fake sources, and a
paused refresh is driven by a fake ``wait`` instead of real time.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tradingagents.api import app as app_module
from tradingagents.api.app import create_app
from tradingagents.api.config import ApiConfig
from tradingagents.api.core.activity_middleware import yahoo_backed_paths
from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.refresh_window import RefreshWindow
from tradingagents.api.domain.services.refresh_activity import (
    ActivityMonitor,
    RefreshGate,
    RefreshInterrupted,
)
from tradingagents.api.domain.services.symbol_catalog import SymbolCatalog, set_active_catalog
from tradingagents.api.domain.services.symbol_settings import SymbolSettings
from tradingagents.api.infrastructure.repositories.file_symbol_cache_repository import (
    FileSymbolCacheRepository,
)
from tradingagents.api.infrastructure.yahoo_symbol_source import YahooSymbolSource

# 2026-09-27 12:00 UTC = 20:00 in Taipei.
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
TAIPEI_03 = datetime(2026, 9, 27, 19, 0, tzinfo=timezone.utc)  # 03:00 Taipei, next day


def entry(symbol: str, market: str = "tw", type_: str = "equity") -> SymbolEntry:
    return SymbolEntry(symbol, symbol, "X", type_, market)


TW = [entry("2330.TW"), entry("2303.TW"), entry("0050.TW", type_="etf")]
US = [entry("AAPL", "us"), entry("MSFT", "us")]


class Clock:
    def __init__(self, now: datetime = NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FakeSource:
    """Returns each market's seeded list (plus one new row) and records calls."""

    def __init__(self):
        self.calls: list[str] = []

    fail: set[str] = set()

    def fetch(self, market):
        self.calls.append(market)
        if market in self.fail:
            raise RuntimeError(f"{market} is down")
        base = {"tw": TW, "us": US}.get(market, [])
        return [*base, entry(f"NEW.{market}", market)]

    def source_label(self, market):
        return "fake"


class Idle:
    """A switchable idle check."""

    def __init__(self, idle: bool = True):
        self.idle = idle

    def __call__(self) -> bool:
        return self.idle


@pytest.fixture
def make_catalog(tmp_path):
    made = []

    def _make(*, lists=None, window=None, idle=None, clock=None, ttl_days=7.0, auto=True):
        repo = FileSymbolCacheRepository(tmp_path)
        for market, (entries, age) in (lists or {}).items():
            repo.save(SymbolList(market, tuple(entries), (clock or Clock())() - age, "seeded"))
        settings = SymbolSettings(
            {
                "ttl_days": ttl_days,
                "auto_refresh": auto,
                "idle_grace_minutes": 10.0,
                "refresh_window": window or "",
                "refresh_timezone": "Asia/Taipei",
                "page_delay_seconds": 2.0,
            },
            store_dir=None,
        )
        source = FakeSource()
        catalog = SymbolCatalog(
            repo,
            source,
            settings=settings,
            idle_check=idle or Idle(),
            tick_seconds=3600,
            clock=clock or Clock(),
        )
        made.append(catalog)
        return catalog, source, settings

    yield _make
    for catalog in made:
        catalog.shutdown()


ALL_FRESH_BUT_TW = {
    "us": (US, timedelta(hours=1)),
    "jp": ([entry("7203.T", "jp")], timedelta(hours=1)),
    "crypto": ([entry("BTC-USD", "crypto", "crypto")], timedelta(hours=1)),
    "fx": ([entry("TWD=X", "fx", "currency")], timedelta(hours=1)),
}


def _stale_tw(**extra):
    return {**ALL_FRESH_BUT_TW, "tw": (TW, timedelta(days=8)), **extra}


# ---------------------------------------------------------------------------
# RefreshWindow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefreshWindow:
    def test_empty_disables(self):
        assert RefreshWindow.parse("", "Asia/Taipei") is None
        assert RefreshWindow.parse("  ", "Asia/Taipei") is None
        assert RefreshWindow.parse(None, None) is None

    def test_inside_and_outside(self):
        window = RefreshWindow.parse("02:00-06:00", "Asia/Taipei")
        assert window.describe() == "02:00-06:00 Asia/Taipei"
        assert window.contains(datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc))  # 02:00
        assert window.contains(datetime(2026, 9, 27, 21, 59, tzinfo=timezone.utc))  # 05:59
        assert not window.contains(datetime(2026, 9, 27, 22, 0, tzinfo=timezone.utc))  # 06:00
        assert not window.contains(NOW)  # 20:00

    def test_wraps_midnight(self):
        window = RefreshWindow.parse("22:00-04:00", "UTC")
        for hour, inside in [(21, False), (22, True), (23, True), (0, True), (3, True), (4, False)]:
            assert window.contains(datetime(2026, 9, 27, hour, 30, tzinfo=timezone.utc)) is inside

    def test_24_00_end(self):
        window = RefreshWindow.parse("20:00-24:00", "UTC")
        assert window.contains(datetime(2026, 9, 27, 23, 59, tzinfo=timezone.utc))
        assert not window.contains(datetime(2026, 9, 27, 0, 0, tzinfo=timezone.utc))

    def test_next_open(self):
        window = RefreshWindow.parse("02:00-06:00", "Asia/Taipei")
        assert window.next_open(NOW) == datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)
        assert window.next_open(TAIPEI_03) == TAIPEI_03  # open now

    @pytest.mark.parametrize(
        ("spec", "tz"),
        [
            ("2-6", "Asia/Taipei"),
            ("02:00", "Asia/Taipei"),
            ("25:00-06:00", "Asia/Taipei"),
            ("02:60-06:00", "Asia/Taipei"),
            ("02:00-02:00", "Asia/Taipei"),
            ("02:00-06:00", "Mars/Olympus_Mons"),
            ("02:00-06:00", ""),
        ],
    )
    def test_bad_values(self, spec, tz):
        with pytest.raises(ValueError):
            RefreshWindow.parse(spec, tz)


# ---------------------------------------------------------------------------
# Idleness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestActivityMonitor:
    def test_idle_busy_transitions(self):
        clock, tasks, grace = Clock(), {"n": 0}, {"g": timedelta(minutes=10)}
        monitor = ActivityMonitor(lambda: tasks["n"], lambda: grace["g"], clock)
        assert monitor.is_idle()

        tasks["n"] = 1  # an analysis task queued or processing
        assert not monitor.is_idle()
        tasks["n"] = 0

        monitor.request_started()  # a /data request in flight
        clock.now += timedelta(hours=1)
        assert not monitor.is_idle()
        monitor.request_finished()
        assert not monitor.is_idle()  # within the grace period
        clock.now += timedelta(minutes=9, seconds=59)
        assert not monitor.is_idle()
        clock.now += timedelta(seconds=1)
        assert monitor.is_idle()

        # The grace is read live.
        grace["g"] = timedelta(minutes=30)
        assert not monitor.is_idle()
        assert monitor.state()["last_data_request_at"] == NOW + timedelta(hours=1)

    def test_tracked_paths(self):
        tracked = yahoo_backed_paths("/api/v1")
        assert tracked("GET", "/api/v1/data/stock")
        assert tracked("GET", "/api/v1/analysts/market")
        assert tracked("POST", "/api/v1/analyze")
        assert not tracked("POST", "/api/v1/analyze/tasks")  # counted via the TaskManager
        assert not tracked("GET", "/api/v1/symbols")
        assert not tracked("GET", "/health")


# ---------------------------------------------------------------------------
# When automatic refreshes start
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAutomaticRefreshConditions:
    def test_a_stale_list_waits_while_busy_and_refreshes_once_idle(self, make_catalog):
        idle = Idle(False)
        catalog, source, _ = make_catalog(lists=_stale_tw(), idle=idle)
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls == []
        catalog.search("tw")  # a read does not start it either
        assert catalog.wait_idle(5)
        assert source.calls == []
        assert catalog.waiting_for("tw") == "activity"
        assert catalog.tick() == []

        idle.idle = True
        assert catalog.waiting_for("tw") == "tick"
        assert catalog.tick() == ["tw"]
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]
        assert catalog.waiting_for("tw") is None

    def test_a_missing_list_is_fetched_at_once_even_when_busy_or_outside_the_window(
        self, make_catalog
    ):
        lists = {m: v for m, v in _stale_tw().items() if m != "jp"}
        catalog, source, _ = make_catalog(lists=lists, idle=Idle(False), window="02:00-06:00")
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls == ["jp"]  # tw is stale but waits

    def test_a_missing_list_is_retried_while_busy(self, make_catalog):
        clock = Clock()
        lists = {m: v for m, v in _stale_tw().items() if m != "jp"}
        catalog, source, _ = make_catalog(lists=lists, idle=Idle(False), clock=clock)
        source.fail = {"jp"}
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls == ["jp"]  # failed

        clock.now += timedelta(hours=2)  # past the retry limit, still busy
        assert catalog.tick() == ["jp"]
        assert catalog.wait_idle(5)
        assert source.calls == ["jp", "jp"]

    def test_the_optional_window_is_an_extra_condition(self, make_catalog):
        clock, idle = Clock(), Idle(True)
        catalog, source, _ = make_catalog(
            lists=_stale_tw(), idle=idle, clock=clock, window="02:00-06:00"
        )
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls == []  # idle, but 20:00 in Taipei
        assert catalog.waiting_for("tw") == "window"
        assert catalog.next_refresh_after("tw") == datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)

        clock.now = TAIPEI_03
        idle.idle = False
        assert catalog.tick() == []  # inside the window, but busy
        assert catalog.waiting_for("tw") == "activity"
        assert catalog.next_refresh_after("tw") is None

        idle.idle = True
        assert catalog.tick() == ["tw"]
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]

    def test_a_wrap_around_window(self, make_catalog):
        clock = Clock(datetime(2026, 9, 27, 21, 0, tzinfo=timezone.utc))
        catalog, source, _ = make_catalog(lists=_stale_tw(), clock=clock, window="23:00-01:00")
        catalog.settings.update({"refresh_timezone": "UTC"})
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls == []
        clock.now = datetime(2026, 9, 28, 0, 30, tzinfo=timezone.utc)
        assert catalog.tick() == ["tw"]

    def test_no_window_means_idle_is_enough(self, make_catalog):
        catalog, source, _ = make_catalog(lists=_stale_tw())
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]

    def test_manual_refresh_is_immediate_even_when_busy_and_outside_the_window(self, make_catalog):
        catalog, source, _ = make_catalog(lists=_stale_tw(), idle=Idle(False), window="02:00-06:00")
        catalog.start()
        assert catalog.request_manual_refresh(["tw"]) == (["tw"], [])
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]

    def test_auto_refresh_off_waits_with_its_reason(self, make_catalog):
        catalog, source, _ = make_catalog(lists=_stale_tw(), auto=False)
        catalog.start()
        assert catalog.wait_idle(5)
        assert source.calls == []
        assert catalog.waiting_for("tw") == "auto_refresh_off"


@pytest.mark.unit
class TestMissTriggeredRefresh:
    def _lists(self):
        day_old = timedelta(days=2)  # older than 24h, fresher than the TTL
        return {
            **ALL_FRESH_BUT_TW,
            "tw": (TW, day_old),
            "us": (US, day_old),
        }

    def test_an_enforced_miss_waits_until_idle(self, make_catalog):
        idle = Idle(False)
        catalog, source, _ = make_catalog(lists=self._lists(), idle=idle)
        catalog.start()

        assert catalog.decide("2331.TW").rejected
        assert catalog.wait_idle(5)
        assert source.calls == []
        assert catalog.waiting_for("tw") == "activity"  # flagged by the miss

        idle.idle = True
        assert catalog.tick() == ["tw"]
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]

    def test_an_enforced_miss_when_idle_queues_at_once(self, make_catalog):
        catalog, source, _ = make_catalog(lists=self._lists())
        catalog.start()
        catalog.decide("2331.TW")
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]

    @pytest.mark.parametrize(("ticker", "asset_type"), [("ZZZZ", "stock"), ("DOGE", "crypto")])
    def test_a_soft_market_miss_never_triggers_a_refresh(self, make_catalog, ticker, asset_type):
        lists = {
            **self._lists(),
            "crypto": ([entry("BTC-USD", "crypto", "crypto")], timedelta(days=2)),
        }
        catalog, source, _ = make_catalog(lists=lists)
        catalog.start()

        decision = catalog.decide(ticker, asset_type)
        catalog.lookup(ticker)
        assert decision.supported is False and not decision.rejected
        assert catalog.wait_idle(5)
        assert source.calls == []
        assert catalog.tick() == []
        assert catalog.waiting_for(decision.market) is None

    def test_the_policy_is_read_live_from_the_reject_mapping(self, make_catalog, tmp_path):
        repo = FileSymbolCacheRepository(tmp_path / "strict")
        for market, (entries, age) in self._lists().items():
            repo.save(SymbolList(market, tuple(entries), NOW - age, "seeded"))
        source = FakeSource()
        catalog = SymbolCatalog(
            repo,
            source,
            reject_unlisted={"tw": True, "jp": True, "us": True},
            clock=Clock(),
            tick_seconds=3600,
        )
        try:
            catalog.start()
            catalog.decide("ZZZZ")  # us is enforced here: a day-old miss refreshes
            assert catalog.wait_idle(5)
            assert source.calls == ["us"]
        finally:
            catalog.shutdown()


@pytest.mark.unit
class TestTick:
    def test_the_tick_thread_stops_on_shutdown(self, make_catalog):
        catalog, _, _ = make_catalog(lists=_stale_tw())
        catalog.start()
        thread = catalog._tick_thread
        assert thread is not None and thread.is_alive()

        catalog.shutdown()
        thread.join(2)  # Event.wait returns at once: no waiting out the interval
        assert not thread.is_alive()
        assert catalog.tick() == []

    def test_a_catalog_without_auto_refresh_starts_no_tick_until_enabled(self, make_catalog):
        catalog, source, settings = make_catalog(lists=_stale_tw(), auto=False)
        catalog.start()
        assert catalog._tick_thread is None

        settings.update({"auto_refresh": True})  # live: starts the tick and runs it
        assert catalog._tick_thread is not None
        assert catalog.wait_idle(5)
        assert source.calls == ["tw"]


# ---------------------------------------------------------------------------
# A running refresh yields between pages
# ---------------------------------------------------------------------------


class PagedScreen:
    """A screener with ``pages`` full pages on TAI equities, nothing elsewhere."""

    def __init__(self, pages: int, on_page=None):
        self.rows = [{"symbol": f"{i:04d}.TW"} for i in range(250 * pages)]
        self.offsets: list[int] = []
        self.on_page = on_page

    def __call__(self, query, offset=0, size=25, sortField=None, sortAsc=None):
        exchange = json.dumps(query.to_dict())
        if '"TAI"' not in exchange or "intradaymarketcap" not in exchange:
            return {"quotes": [], "total": 0}
        self.offsets.append(offset)
        if self.on_page:
            self.on_page(offset)
        return {"quotes": self.rows[offset : offset + size], "total": len(self.rows)}


@pytest.mark.unit
class TestPauseAndResume:
    def test_a_refresh_pauses_mid_market_and_resumes_the_same_market(self, caplog):
        caplog.set_level(logging.INFO)
        idle = Idle(True)
        polls = []

        def fake_wait(seconds):
            polls.append(seconds)
            if len(polls) == 3:
                idle.idle = True  # the task finished
            return False

        def on_page(offset):
            if offset == 250:
                idle.idle = False  # an analysis task starts after page 2

        gate = RefreshGate(idle, poll_seconds=5.0, wait=fake_wait)
        screen = PagedScreen(3, on_page)
        source = YahooSymbolSource(screen=screen, sleep=lambda s: None, gate=gate)

        entries = source.fetch("tw")

        assert len(entries) == 750
        assert screen.offsets == [0, 250, 500]  # the same market carried on
        assert polls == [5.0, 5.0, 5.0]
        assert not gate.paused
        assert "Pausing the symbol refresh (tw, after 2 request(s))" in caplog.text
        assert "Resuming the symbol refresh" in caplog.text

    def test_shutdown_abandons_a_paused_refresh_and_keeps_the_old_list(self, tmp_path):
        gate = RefreshGate(Idle(False), wait=lambda s: True)  # "stopped" while waiting
        source = YahooSymbolSource(screen=PagedScreen(3), sleep=lambda s: None, gate=gate)
        repo = FileSymbolCacheRepository(tmp_path)
        repo.save(SymbolList("tw", tuple(TW), NOW - timedelta(days=8), "seeded"))
        catalog = SymbolCatalog(repo, source, auto_refresh=False, gate=gate, clock=Clock())
        catalog.start()

        assert catalog.refresh_market("tw") is False

        assert "RefreshInterrupted" in catalog.last_error("tw")
        assert [e.symbol for e in catalog.get_list("tw").entries] == [e.symbol for e in TW]

    def test_catalog_shutdown_ends_a_paused_refresh_at_once(self, tmp_path):
        busy_seen = threading.Event()
        state = {"idle": True}

        def is_idle():
            if not state["idle"]:
                busy_seen.set()
            return state["idle"]

        def on_page(offset):
            state["idle"] = False  # busy from the first page on

        gate = RefreshGate(is_idle, poll_seconds=30.0)  # the real, stop-aware wait
        source = YahooSymbolSource(screen=PagedScreen(3, on_page), sleep=lambda s: None, gate=gate)
        repo = FileSymbolCacheRepository(tmp_path)
        repo.save(SymbolList("tw", tuple(TW), NOW - timedelta(days=8), "seeded"))
        catalog = SymbolCatalog(repo, source, auto_refresh=False, gate=gate, clock=Clock())
        catalog.start()
        catalog.request_refresh(["tw"])
        assert busy_seen.wait(5)

        catalog.shutdown()  # must not wait out the 30 s poll

        assert catalog.wait_idle(5)
        assert "RefreshInterrupted" in catalog.last_error("tw")
        assert len(catalog.get_list("tw").entries) == len(TW)

    def test_a_stopped_gate_refuses_to_continue(self):
        gate = RefreshGate()
        gate.stop()
        with pytest.raises(RefreshInterrupted):
            gate.wait_until_idle()

    def test_a_manual_refresh_does_not_pause(self, tmp_path):
        waits = []

        def fake_wait(seconds):
            waits.append(seconds)
            return len(waits) > 20  # never hang the test if it did pause

        gate = RefreshGate(Idle(False), wait=fake_wait)
        source = YahooSymbolSource(screen=PagedScreen(3), sleep=lambda s: None, gate=gate)
        catalog = SymbolCatalog(
            FileSymbolCacheRepository(tmp_path), source, auto_refresh=False, gate=gate
        )
        catalog.start()

        assert catalog.request_manual_refresh(["tw"]) == (["tw"], [])
        assert catalog.wait_idle(5)

        assert waits == []
        assert len(catalog.get_list("tw").entries) == 750

    def test_page_delay_is_read_live(self):
        delays, sleeps = {"d": 2.0}, []
        source = YahooSymbolSource(
            screen=PagedScreen(3, on_page=lambda o: delays.update(d=5.0) if o == 250 else None),
            sleep=sleeps.append,
            page_delay_seconds=lambda: delays["d"],
        )
        source.fetch("tw")
        assert sleeps[:2] == [2.0, 5.0]
        assert set(sleeps[2:]) == {5.0}

    def test_the_default_page_delay_is_two_seconds(self):
        assert YahooSymbolSource()._page_delay == 2.0
        assert ApiConfig().config["symbols_page_delay_seconds"] == 2.0


# ---------------------------------------------------------------------------
# The app: middleware, settings API, persistence
# ---------------------------------------------------------------------------

BASE = "/api/v1/symbols"


class AppSource(FakeSource):
    def __init__(self, *args, **kwargs):
        super().__init__()


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "YahooSymbolSource", AppSource)
    apps = []

    def _make(**overrides):
        app = create_app(
            overrides={
                "symbols_cache_dir": str(tmp_path),
                "symbols_auto_refresh": False,
                **overrides,
            }
        )
        apps.append(app)
        return TestClient(app, raise_server_exceptions=False), app

    yield _make
    for app in apps:
        app.state.symbol_catalog.shutdown()
    set_active_catalog(None)


EXPECTED_DEFAULTS = {
    "ttl_days": 7.0,
    "auto_refresh": False,
    "idle_grace_minutes": 10.0,
    "refresh_window": "",
    "refresh_timezone": "Asia/Taipei",
    "page_delay_seconds": 2.0,
}


@pytest.mark.unit
class TestSettingsApi:
    def test_get_shape(self, make_app):
        client, _ = make_app()
        body = client.get(f"{BASE}/settings").json()

        assert body["values"] == EXPECTED_DEFAULTS
        assert body["sources"] == {
            **dict.fromkeys(EXPECTED_DEFAULTS, "default"),
            "auto_refresh": "config",  # set through create_app(overrides=...)
        }
        state = body["state"]
        assert state["idle"] is True
        assert (state["active_tasks"], state["data_requests_in_flight"]) == (0, 0)
        assert state["last_data_request_at"] is None
        assert state["window_open"] is True
        assert set(state["markets"]) == {"tw", "us", "jp", "crypto", "fx"}
        assert state["markets"]["tw"] == {
            "status": "unavailable",
            "stale": False,
            "refreshing": False,
            "paused": False,
            "waiting_for": None,
            "next_refresh_after": None,
            "last_error": None,
        }

    def test_put_updates_persists_and_resets(self, make_app, tmp_path):
        client, app = make_app()
        resp = client.put(
            f"{BASE}/settings",
            json={"ttl_days": 3, "refresh_window": "01:00-05:00", "idle_grace_minutes": 0},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["values"]["ttl_days"] == 3.0
        assert body["values"]["refresh_window"] == "01:00-05:00"
        assert body["values"]["idle_grace_minutes"] == 0.0
        assert body["sources"]["ttl_days"] == "api"
        assert body["sources"]["page_delay_seconds"] == "default"
        assert json.loads((tmp_path / "settings.json").read_text())["values"] == {
            "idle_grace_minutes": 0.0,
            "refresh_window": "01:00-05:00",
            "ttl_days": 3.0,
        }

        # Survives a restart, and wins over config values.
        client2, _ = make_app(symbols_cache_ttl_days=9)
        again = client2.get(f"{BASE}/settings").json()
        assert again["values"]["ttl_days"] == 3.0 and again["sources"]["ttl_days"] == "api"

        # null resets a field to its base value.
        reset = client2.put(f"{BASE}/settings", json={"ttl_days": None}).json()
        assert reset["values"]["ttl_days"] == 9.0 and reset["sources"]["ttl_days"] == "config"
        assert "ttl_days" not in json.loads((tmp_path / "settings.json").read_text())["values"]

        # An empty body changes nothing.
        assert client2.put(f"{BASE}/settings", json={}).json()["values"] == reset["values"]

    @pytest.mark.parametrize(
        "body",
        [
            {"ttl_days": 0.1},
            {"ttl_days": 91},
            {"idle_grace_minutes": -1},
            {"idle_grace_minutes": 241},
            {"page_delay_seconds": 0.2},
            {"page_delay_seconds": 11},
            {"refresh_window": "2-6"},
            {"refresh_window": "25:00-06:00"},
            {"refresh_window": "02:00-02:00"},
            {"refresh_timezone": "Mars/Olympus_Mons"},
            {"refresh_timezone": ""},
            {"auto_refresh": "maybe"},
            {"ttl_days": "soon"},
            {"unknown": 1},
            {"ttl_days": 3, "page_delay_seconds": 99},  # all or nothing
        ],
    )
    def test_invalid_input_is_422_and_changes_nothing(self, make_app, tmp_path, body):
        client, _ = make_app()
        resp = client.put(f"{BASE}/settings", json=body)
        assert resp.status_code == 422, resp.text
        assert client.get(f"{BASE}/settings").json()["values"] == EXPECTED_DEFAULTS
        assert not (tmp_path / "settings.json").exists()

    def test_changes_apply_live(self, make_app, tmp_path):
        repo = FileSymbolCacheRepository(tmp_path)
        repo.save(SymbolList("tw", tuple(TW), datetime.now(timezone.utc) - timedelta(days=3)))
        client, app = make_app(symbols_auto_refresh=True)
        catalog = app.state.symbol_catalog
        catalog.start()
        assert catalog.wait_idle(5)
        source = catalog._source
        assert "tw" not in source.calls  # 3 days old: fresh under the 7-day TTL

        # A shorter TTL makes it stale now, and the update runs a tick at once.
        client.put(f"{BASE}/settings", json={"ttl_days": 2})
        assert catalog.wait_idle(5)
        assert source.calls.count("tw") == 1

        # The page delay reaches the source's live getter.
        client.put(f"{BASE}/settings", json={"page_delay_seconds": 4.5})
        assert app.state.symbol_settings.page_delay_seconds == 4.5

    def test_a_corrupt_settings_file_is_ignored(self, make_app, tmp_path, caplog):
        (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
        client, _ = make_app()
        assert client.get(f"{BASE}/settings").json()["values"] == EXPECTED_DEFAULTS
        assert "Ignoring the symbol settings file" in caplog.text

    def test_a_bad_field_in_the_file_is_dropped(self, make_app, tmp_path, caplog):
        (tmp_path / "settings.json").write_text(
            json.dumps({"version": 1, "values": {"ttl_days": 500, "page_delay_seconds": 3}}),
            encoding="utf-8",
        )
        client, _ = make_app()
        body = client.get(f"{BASE}/settings").json()
        assert body["values"]["ttl_days"] == 7.0
        assert body["values"]["page_delay_seconds"] == 3.0
        assert body["sources"]["page_delay_seconds"] == "api"
        assert "Ignoring 'ttl_days'" in caplog.text

    def test_a_bad_configured_window_fails_app_creation(self, tmp_path):
        with pytest.raises(ValueError, match="HH:MM-HH:MM"):
            create_app(
                overrides={"symbols_cache_dir": str(tmp_path), "symbols_refresh_window": "2am"}
            )
        set_active_catalog(None)

    def test_list_response_reports_what_a_stale_list_waits_for(self, make_app, tmp_path):
        repo = FileSymbolCacheRepository(tmp_path)
        repo.save(SymbolList("tw", tuple(TW), datetime.now(timezone.utc) - timedelta(days=8)))
        client, app = make_app()
        app.state.symbol_catalog.start()
        body = client.get(BASE, params={"market": "tw", "limit": 0}).json()
        assert (body["stale"], body["waiting_for"]) == (True, "auto_refresh_off")
        assert body["refresh_window"] is None and body["next_refresh_after"] is None


@pytest.mark.unit
class TestDataActivityMiddleware:
    def test_a_data_request_makes_the_api_busy_for_the_grace_period(self, make_app):
        client, app = make_app()
        monitor = app.state.activity_monitor
        assert monitor.is_idle()

        # An unknown /data path: no handler runs (so no Yahoo call), yet it counts.
        client.get("/api/v1/data/__nothing__")
        assert not monitor.is_idle()
        state = client.get(f"{BASE}/settings").json()["state"]
        assert state["idle"] is False and state["last_data_request_at"] is not None

        client.put(f"{BASE}/settings", json={"idle_grace_minutes": 0})
        assert monitor.is_idle()

    def test_symbol_requests_do_not_count(self, make_app):
        client, app = make_app()
        client.get(BASE, params={"market": "tw"})
        assert app.state.activity_monitor.state()["last_data_request_at"] is None

    def test_active_tasks_make_the_api_busy(self, make_app):
        client, app = make_app()
        from tradingagents.api.schemas.task import TaskCreateRequest

        app.state.task_manager.create_task(
            TaskCreateRequest(ticker="AAPL", trade_date="2026-09-25")
        )
        assert not app.state.activity_monitor.is_idle()
        assert client.get(f"{BASE}/settings").json()["state"]["active_tasks"] == 1


@pytest.mark.unit
def test_refresher_threads_are_daemon_and_named(make_catalog):
    catalog, _, _ = make_catalog(lists=_stale_tw())
    catalog.start()
    assert catalog.wait_idle(5)
    names = {t.name for t in threading.enumerate() if t.name.startswith("symbol-catalog")}
    assert "symbol-catalog-tick" in names
    assert catalog._tick_thread.daemon


@pytest.mark.unit
class TestConfigAndEnv:
    def test_defaults(self):
        config = ApiConfig().config
        assert config["symbols_refresh_window"] == ""
        assert config["symbols_refresh_timezone"] == "Asia/Taipei"
        assert config["symbols_idle_grace_minutes"] == 10.0

    def test_env_parsing_and_sources(self, monkeypatch, make_app):
        from tradingagents.api import config as config_module

        monkeypatch.setattr(config_module, "SYMBOLS_ENV_KEYS", set())
        monkeypatch.setenv("TRADINGAGENTS_SYMBOLS_REFRESH_WINDOW", "")  # empty = off
        monkeypatch.setenv("TRADINGAGENTS_SYMBOLS_IDLE_GRACE_MINUTES", "2.5")
        monkeypatch.setenv("TRADINGAGENTS_SYMBOLS_CACHE_TTL_DAYS", "")  # ignored: not text
        defaults = config_module._symbols_defaults()
        assert defaults["symbols_refresh_window"] == ""
        assert defaults["symbols_idle_grace_minutes"] == 2.5
        assert defaults["symbols_cache_ttl_days"] == 7.0
        env_keys = set(config_module.SYMBOLS_ENV_KEYS)
        assert env_keys == {
            "symbols_refresh_window",
            "symbols_idle_grace_minutes",
        }

        client, _ = make_app(symbols_idle_grace_minutes=2.5)
        sources = client.get(f"{BASE}/settings").json()["sources"]
        assert sources["refresh_window"] == "env"
        assert sources["idle_grace_minutes"] == "config"  # an override wins the label
