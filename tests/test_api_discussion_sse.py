"""The discussion endpoints (Plan 08, 08-A): SSE event stream and transcript.

Runs go through the real app, task worker and graph, under the fake model of
``tests/discussion_fakes.py``; nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from tests.discussion_fakes import TICKER, TRADE_DATE, install_offline
from tradingagents.api import config as api_config
from tradingagents.api.app import create_app
from tradingagents.api.core.discussion_log import MAX_EVENTS, DiscussionLog, DiscussionStore
from tradingagents.api.routers import discussion
from tradingagents.api.schemas.task import TaskCreateRequest

TASKS = "/api/v1/analyze/tasks"


@pytest.fixture(autouse=True)
def _not_ollama(monkeypatch):
    monkeypatch.setitem(api_config.DEFAULT_CONFIG, "llm_provider", "openai")


@pytest.fixture
def app(tmp_path):
    return create_app(overrides={"results_dir": str(tmp_path / "results")})


def _start(client) -> str:
    resp = client.post(TASKS, json={"ticker": TICKER, "trade_date": TRADE_DATE})
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _read(client, task_id, headers=None, params=None):
    """Every event of the stream, its SSE ids and its comment lines, until it ends."""
    events, ids, comments = [], [], []
    with client.stream("GET", f"{TASKS}/{task_id}/events", headers=headers,
                       params=params) as resp:
        assert resp.status_code == 200, resp.read()
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
            elif line.startswith("id: "):
                ids.append(int(line[len("id: "):]))
            elif line.startswith(":"):
                comments.append(line)
    assert ids == [e["id"] for e in events]
    return events, comments


def _finished_run(app, monkeypatch):
    install_offline(monkeypatch)
    client = TestClient(app)
    task_id = _start(client)
    app.state.task_worker.shutdown(wait=True)
    assert client.get(f"{TASKS}/{task_id}").json()["status"] == "completed"
    return client, task_id


@pytest.mark.unit
class TestEventStream:
    def test_a_finished_task_streams_every_event_then_closes(self, app, monkeypatch):
        client, task_id = _finished_run(app, monkeypatch)
        events, _ = _read(client, task_id)
        assert [e["id"] for e in events] == list(range(1, len(events) + 1))
        assert events[0]["type"] == "run_started"
        assert events[0]["data"]["expected_turns"] == 12
        assert events[-1]["type"] == "task_status"
        assert events[-1]["data"] == {"status": "completed"}
        assert len([e for e in events if e["type"] == "turn_completed"]) == 12

    @pytest.mark.parametrize("how", ["header", "query"])
    def test_resuming_loses_and_repeats_nothing(self, app, monkeypatch, how):
        client, task_id = _finished_run(app, monkeypatch)
        full, _ = _read(client, task_id)
        cut = 7
        if how == "header":
            rest, _ = _read(client, task_id, headers={"Last-Event-ID": str(cut)})
        else:
            rest, _ = _read(client, task_id, params={"after": cut})
        assert [e["id"] for e in rest] == list(range(cut + 1, len(full) + 1))
        assert full[:cut] + rest == full

    def test_last_event_id_wins_over_after(self, app, monkeypatch):
        client, task_id = _finished_run(app, monkeypatch)
        rest, _ = _read(client, task_id, headers={"Last-Event-ID": "10"}, params={"after": 2})
        assert rest[0]["id"] == 11

    def test_tokens_is_accepted_and_ignored(self, app, monkeypatch):
        client, task_id = _finished_run(app, monkeypatch)
        with_tokens, _ = _read(client, task_id, params={"tokens": 1})
        assert with_tokens == _read(client, task_id)[0]
        assert not {"token", "gap"} & {e["type"] for e in with_tokens}
        assert client.get(f"{TASKS}/{task_id}/events", params={"tokens": 2}).status_code == 422

    def test_two_live_subscribers_get_the_same_whole_stream(self, app, monkeypatch):
        release = threading.Event()
        model = install_offline(monkeypatch)
        model.hook = lambda n: n == 1 and release.wait(10)
        client = TestClient(app)
        task_id = _start(client)
        log = app.state.discussions.get(task_id)
        results: list = [None, None]

        def subscribe(i):
            results[i] = _read(client, task_id)[0]

        threads = [threading.Thread(target=subscribe, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 10
        while log.waiter_count < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert log.waiter_count == 2   # both caught up and waiting while the run is held
        release.set()
        for t in threads:
            t.join(15)
        app.state.task_worker.shutdown(wait=True)
        assert results[0] == results[1] == log.read()
        assert results[0][-1]["data"] == {"status": "completed"}
        assert log.waiter_count == 0

    def test_pings_keep_an_idle_stream_open_until_the_task_ends(self, app, monkeypatch):
        monkeypatch.setattr(discussion, "PING_INTERVAL_SECONDS", 0.05)
        client = TestClient(app)
        # A task that never reaches the worker: its log stays silent until cancelled.
        task, _ = app.state.task_manager.create_task(
            TaskCreateRequest(ticker=TICKER, trade_date=TRADE_DATE))
        pings, data = 0, []
        with client.stream("GET", f"{TASKS}/{task.task_id}/events") as resp:
            for line in resp.iter_lines():
                if line == ": ping":
                    pings += 1
                    if pings == 3:   # the reader cancels once it has seen three
                        app.state.task_manager.cancel_task(task.task_id)
                elif line.startswith("data: "):
                    data.append(json.loads(line[len("data: "):])["data"])
        assert pings >= 3
        assert data == [{"status": "cancelled"}]

    def test_a_cursor_past_the_end_still_gets_the_rest_of_the_run(self, app, monkeypatch):
        release = threading.Event()
        model = install_offline(monkeypatch)
        model.hook = lambda n: release.wait(10)   # every analyst holds at its first call
        client = TestClient(app)
        task_id = _start(client)
        log = app.state.discussions.get(task_id)
        deadline = time.monotonic() + 10
        while len(model.calls) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(model.calls) == 4   # the log is still until release
        last = log.last_id
        out: list = []
        reader = threading.Thread(
            target=lambda: out.append(_read(client, task_id, params={"after": last + 5})[0]))
        reader.start()
        deadline = time.monotonic() + 10
        while log.waiter_count < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        reader.join(15)
        app.state.task_worker.shutdown(wait=True)
        (events,) = out
        assert [e["id"] for e in events] == list(range(last + 1, log.last_id + 1))
        assert events[-1]["type"] == "task_status"
        assert events[-1]["data"] == {"status": "completed"}

    def test_a_stream_ends_at_its_max_lifetime_and_resumes(self, app, monkeypatch):
        monkeypatch.setattr(discussion, "MAX_STREAM_SECONDS", 1.0)
        release = threading.Event()
        model = install_offline(monkeypatch)
        model.hook = lambda n: release.wait(10)
        client = TestClient(app)
        task_id = _start(client)
        log = app.state.discussions.get(task_id)
        started = time.monotonic()
        first, _ = _read(client, task_id)   # a clean EOF while the run is held
        assert time.monotonic() - started < 5
        assert first and not log.finished
        assert {"completed"}.isdisjoint(e["data"].get("status") for e in first)
        release.set()
        app.state.task_worker.shutdown(wait=True)
        rest, _ = _read(client, task_id, headers={"Last-Event-ID": str(first[-1]["id"])})
        assert first + rest == log.read()
        assert rest[-1]["data"] == {"status": "completed"}

    def test_closing_the_store_ends_open_streams_at_once(self, app):
        with TestClient(app):
            assert not app.state.discussions.closing.is_set()
        assert app.state.discussions.closing.is_set()   # the lifespan shutdown sets it

    async def test_a_waiting_stream_ends_when_the_store_closes(self):
        store = DiscussionStore()
        log = store.create("t")
        log.append("task_status", {"status": "running"})
        stream = discussion.stream_events(log, 0, store.closing)
        assert '"id":1' in await anext(stream)
        waiting = asyncio.ensure_future(anext(stream))
        deadline = time.monotonic() + 5
        while log.waiter_count < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert log.waiter_count == 1
        store.close()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(waiting, 2)   # long before the 15 s ping
        assert log.waiter_count == 0
        # A stream opened after the close ends once it has caught up.
        assert [c async for c in discussion.stream_events(log, 0, store.closing)] != []

    def test_line_separators_stay_inside_one_data_line(self, app):
        client = TestClient(app)
        task, _ = app.state.task_manager.create_task(
            TaskCreateRequest(ticker=TICKER, trade_date=TRADE_DATE))
        text = "a\u2028b\u2029c\x85d"
        log = app.state.discussions.get(task.task_id)
        log.append("turn_completed", {"turn": 1, "agent_id": "market", "content": text,
                                      "truncated": False})
        app.state.task_manager.cancel_task(task.task_id)
        raw = client.get(f"{TASKS}/{task.task_id}/events").text
        # Even a reader that breaks lines on U+2028/U+2029/U+0085 sees one data: line.
        lines = raw.splitlines()
        data = [json.loads(line[len("data: "):]) for line in lines if line.startswith("data: ")]
        assert len(data) == 2 and len([ln for ln in lines if ln]) == 4
        assert data[0]["data"]["content"] == text
        assert data == log.read()

    def test_a_cancelled_run_ends_with_aborted_turns_and_cancelled(self, app, monkeypatch):
        reached, release = threading.Event(), threading.Event()
        model = install_offline(monkeypatch)

        def hook(n):
            if n == 3:   # mid-analysis: several analyst turns open
                reached.set()
                release.wait(10)

        model.hook = hook
        client = TestClient(app)
        task_id = _start(client)
        assert reached.wait(10)
        assert client.post(f"{TASKS}/{task_id}/cancel").json()["cancel_requested"] is True
        release.set()
        app.state.task_worker.shutdown(wait=True)
        assert client.get(f"{TASKS}/{task_id}").json()["status"] == "cancelled"

        events, _ = _read(client, task_id)
        started = {e["data"]["turn"] for e in events if e["type"] == "turn_started"}
        completed = {e["data"]["turn"] for e in events if e["type"] == "turn_completed"}
        aborted = {e["data"]["turn"] for e in events if e["type"] == "turn_aborted"}
        assert aborted and started == completed | aborted
        assert events[-1]["data"] == {"status": "cancelled"}
        assert events[-2]["type"] == "turn_aborted"

    def test_unknown_tasks_are_404(self, app):
        client = TestClient(app)
        for suffix in ("events", "transcript"):
            for task_id in ("nope", "00000000-0000-4000-8000-000000000000"):
                resp = client.get(f"{TASKS}/{task_id}/{suffix}")
                assert resp.status_code == 404
                assert resp.json() == {"detail": f"Task {task_id} not found"}


@pytest.mark.unit
class TestSubscriberMemory:
    async def test_a_subscriber_that_stops_reading_costs_only_its_cursor(self):
        log = DiscussionLog("t")
        log.append("task_status", {"status": "running"})
        stalled = discussion.stream_events(log, 0)
        first = await anext(stalled)   # reads once, then never again
        assert '"id":1' in first
        for _ in range(MAX_EVENTS + 500):
            log.append("phase_changed", {"phase": "analysis"})
        # The log stays bounded and is shared, never copied per subscriber.
        assert log.last_id == MAX_EVENTS + 1   # the kept events and the gap marker
        assert all(a is b for a, b in zip(log.read(), log.read(), strict=True))
        assert log.waiter_count == 0   # a subscriber that is not waiting holds no waiter
        reader = discussion.stream_events(log, MAX_EVENTS)
        chunk = await anext(reader)
        assert chunk.count("id: ") == 1 and '"type":"gap"' in chunk
        waiting = asyncio.ensure_future(anext(reader))
        await asyncio.sleep(0.05)
        assert log.waiter_count == 1
        waiting.cancel()   # the client disconnected
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert log.waiter_count == 0
        await stalled.aclose()


@pytest.mark.unit
class TestTranscript:
    def test_the_transcript_outlives_the_task_in_its_file(self, app, monkeypatch, tmp_path):
        client, task_id = _finished_run(app, monkeypatch)
        live = client.get(f"{TASKS}/{task_id}/transcript")
        assert live.status_code == 200
        body = live.json()
        assert (body["task_id"], body["v"], body["status"]) == (task_id, 1, "completed")
        assert body["run_started"]["expected_turns"] == 12
        assert [t["turn"] for t in body["turns"]] == list(range(1, 13))
        assert all(t["content"] and t["completed_at"] and not t["aborted"] for t in body["turns"])
        assert body["turns"][0]["agent_id"] in {"market", "sentiment", "news", "fundamentals"}
        market = next(t for t in body["turns"] if t["agent_id"] == "market")
        assert "get_stock_data" in market["tools"]

        path = (tmp_path / "results" / TICKER / "TradingAgentsStrategy_logs"
                / f"discussion_{task_id}.json")
        assert json.loads(path.read_text(encoding="utf-8")) == body
        assert not list(path.parent.glob(".discussion-*"))   # no temporary file left

        app.state.task_manager._ttl = 0   # the task expires
        app.state.task_manager.cleanup()
        assert client.get(f"{TASKS}/{task_id}").status_code == 404
        assert app.state.discussions.get(task_id) is None
        assert client.get(f"{TASKS}/{task_id}/events").status_code == 404
        from_file = client.get(f"{TASKS}/{task_id}/transcript")
        assert from_file.status_code == 200
        assert from_file.json() == body

    def test_a_task_cancelled_before_it_ran_writes_no_file(self, app, tmp_path):
        client = TestClient(app)
        task, _ = app.state.task_manager.create_task(
            TaskCreateRequest(ticker=TICKER, trade_date=TRADE_DATE))
        app.state.task_manager.cancel_task(task.task_id)
        body = client.get(f"{TASKS}/{task.task_id}/transcript").json()
        assert (body["status"], body["run_started"], body["turns"]) == ("cancelled", None, [])
        assert not list((tmp_path / "results").glob("*/*/discussion_*.json"))
