"""Tests for the synchronous POST /analyze path (issue #4, review M4).

With Ollama the sync run goes through the app's TaskWorker, queued with the
tasks; every sync run gets a cancel event that a server shutdown or a client
disconnect sets. The graph is a stand-in driving a real ``langchain_core`` fake
chat model with the callbacks the API passes, and the Ollama trimmer is off
(no test reaches Ollama; see ``ollama_http_guard`` in conftest).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tradingagents.api.app import create_app
from tradingagents.api.routers import analyze as analyze_module

GRAPH = "tradingagents.api.domain.services.analysis_service.TradingAgentsGraph"
ANALYZE = "/api/v1/analyze"
TASKS = "/api/v1/analyze/tasks"


class Script:
    """A TradingAgentsGraph stand-in: two LLM calls per run, the ``hold``
    ticker pausing between them until ``proceed`` is set."""

    def __init__(self, hold: str):
        self.hold = hold
        self.held = threading.Event()
        self.proceed = threading.Event()
        self.log: list[str] = []
        script = self

        class Graph:
            def __init__(self, *args, callbacks=None, **kwargs):
                self.model = FakeListChatModel(responses=["a", "b"], callbacks=callbacks)

            def propagate(self, ticker, trade_date, asset_type="stock"):
                script.log.append(f"{ticker} start")
                self.model.invoke("first")
                if ticker == script.hold:
                    script.held.set()
                    assert script.proceed.wait(5)
                self.model.invoke("second")
                script.log.append(f"{ticker} end")
                return {"final_trade_decision": "Buy"}, "Buy"

        self.graph = Graph


def _app(**overrides):
    app = create_app(overrides=overrides)
    app.state.ollama_cache_trimmer = None  # this file tests queueing and cancel only
    return app


def _post_in_thread(client, path, body):
    out = {}

    def post():
        out["resp"] = client.post(path, json=body)

    thread = threading.Thread(target=post)
    thread.start()
    return thread, out


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.unit
class TestSyncRunsQueueWithOllama:
    def test_an_ollama_sync_run_waits_for_the_running_task(self):
        script = Script(hold="AAPL")
        with patch(GRAPH, script.graph):
            app = _app(llm_provider="ollama")
            assert app.state.task_worker.max_workers == 1
            client = TestClient(app)
            client.post(TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"})
            assert script.held.wait(5)
            thread, out = _post_in_thread(
                client, ANALYZE, {"ticker": "MSFT", "trade_date": "2026-06-01"}
            )
            _wait_for(lambda: app.state.sync_runs.active_count == 1)
            time.sleep(0.2)  # time enough to start, were it not queued
            assert script.log == ["AAPL start"]
            script.proceed.set()
            thread.join(5)
            app.state.task_worker.shutdown(wait=True)

        assert script.log == ["AAPL start", "AAPL end", "MSFT start", "MSFT end"]
        resp = out["resp"]
        assert resp.status_code == 200, resp.text
        assert resp.json()["signal"] == "Buy" and resp.json()["ticker"] == "MSFT"
        assert app.state.sync_runs.active_count == 0

    def test_other_providers_keep_running_sync_runs_on_the_threadpool(self):
        script = Script(hold="AAPL")
        with patch(GRAPH, script.graph):
            app = _app(llm_provider="openai", task_max_concurrent=1)
            client = TestClient(app)
            client.post(TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"})
            assert script.held.wait(5)
            resp = client.post(ANALYZE, json={"ticker": "MSFT", "trade_date": "2026-06-01"})
            script.proceed.set()
            app.state.task_worker.shutdown(wait=True)
        assert resp.status_code == 200
        assert script.log.index("MSFT end") < script.log.index("AAPL end")


@pytest.mark.unit
class TestSyncRunCancel:
    @pytest.mark.parametrize("provider", ["ollama", "openai"])
    def test_a_shutdown_stops_a_running_sync_run(self, provider):
        script = Script(hold="MSFT")
        with patch(GRAPH, script.graph):
            app = _app(llm_provider=provider)
            client = TestClient(app)
            thread, out = _post_in_thread(
                client, ANALYZE, {"ticker": "MSFT", "trade_date": "2026-06-01"}
            )
            assert script.held.wait(5)
            assert app.state.sync_runs.cancel_all() == 1  # what the lifespan does
            script.proceed.set()
            thread.join(5)
            app.state.task_worker.shutdown(wait=True)
        assert script.log == ["MSFT start"]  # the second LLM call never ran
        assert out["resp"].status_code == 503
        assert "cancelled" in out["resp"].json()["detail"]

    def test_the_lifespan_cancels_sync_runs_and_drops_queued_ones(self):
        script = Script(hold="MSFT")
        with patch(GRAPH, script.graph):
            app = _app(llm_provider="ollama")
            runs = app.state.sync_runs

            def release_once_cancelled():
                # The held run resumes only once the shutdown has cancelled it.
                _wait_for(lambda: any(e.is_set() for e in list(runs._events)))
                script.proceed.set()

            with TestClient(app) as client:
                running, out_running = _post_in_thread(
                    client, ANALYZE, {"ticker": "MSFT", "trade_date": "2026-06-01"}
                )
                assert script.held.wait(5)
                queued, out_queued = _post_in_thread(
                    client, ANALYZE, {"ticker": "NVDA", "trade_date": "2026-06-01"}
                )
                _wait_for(lambda: runs.active_count == 2)
                releaser = threading.Thread(target=release_once_cancelled)
                releaser.start()
                # Leaving the block runs the lifespan's shutdown.
            for thread in (running, queued, releaser):
                thread.join(5)
            app.state.task_worker.shutdown(wait=True)
        assert out_running["resp"].status_code == 503
        assert out_queued["resp"].status_code == 503
        assert script.log == ["MSFT start"]  # stopped at its next call; NVDA never ran

    def test_a_client_disconnect_cancels_the_run(self, monkeypatch):
        monkeypatch.setattr(analyze_module, "DISCONNECT_POLL_SECONDS", 0.01)
        script = Script(hold="MSFT")
        with patch(GRAPH, script.graph):
            app = _app(llm_provider="ollama")
            sent = asyncio.run(_post_then_disconnect(app, script))
            app.state.task_worker.shutdown(wait=True)
        assert script.log == ["MSFT start"]  # stopped at its next LLM call
        assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 503
        assert app.state.sync_runs.active_count == 0

    def test_a_queued_run_whose_client_left_never_starts(self, monkeypatch):
        monkeypatch.setattr(analyze_module, "DISCONNECT_POLL_SECONDS", 0.01)
        script = Script(hold="AAPL")
        with patch(GRAPH, script.graph):
            app = _app(llm_provider="ollama")
            TestClient(app).post(TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"})
            assert script.held.wait(5)
            sent = asyncio.run(_post_then_disconnect(app, script, wait_held=False))
            script.proceed.set()
            app.state.task_worker.shutdown(wait=True)
        assert sent[0]["status"] == 503
        assert script.log == ["AAPL start", "AAPL end"]


async def _post_then_disconnect(app, script, wait_held=True):
    """POST /analyze for MSFT straight to the ASGI app, then drop the client."""
    body = json.dumps({"ticker": "MSFT", "trade_date": "2026-06-01"}).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    gone = asyncio.Event()

    async def receive():
        if messages:
            return messages.pop(0)
        await gone.wait()
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": ANALYZE,
        "raw_path": ANALYZE.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    call = asyncio.create_task(app(scope, receive, send))
    runs = app.state.sync_runs

    async def until(condition):
        for _ in range(500):
            if condition():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out")

    await until(lambda: runs.active_count == 1)
    if wait_held:
        await until(script.held.is_set)
    gone.set()
    await until(lambda: runs.active_count == 0 or any(e.is_set() for e in runs._events))
    script.proceed.set()
    await asyncio.wait_for(call, 5)
    return sent
