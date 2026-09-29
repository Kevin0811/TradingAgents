"""Tests for the Ollama prompt-cache trimmer and its LLM callback (issue #5).

All HTTP goes through a fake; nothing reaches the network or a real model.
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tradingagents.api.app import create_app
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.infrastructure.llm_callbacks import OllamaCacheTrimHandler
from tradingagents.api.infrastructure.ollama_cache_trimmer import (
    BACKOFF_CHECKS,
    FAILURES_BEFORE_BACKOFF,
    NOT_TAKEN_BACKOFF_CHECKS,
    WEIGHTS_OVERHEAD_MB,
    OllamaCacheTrimmer,
    normalize_model_name,
    ollama_native_url,
)

MB = 1 << 20
URL = "http://ollama.test:11434"
MODEL = "qwen3.5:4b-mlx"


class FakeHttp:
    """Scripted /api/ps sizes and unload answers; records every request.

    ``sizes`` are MB per /api/ps read (None = not loaded), or ``(MB,
    context_length)`` pairs. ``weights`` is the model's /api/tags size in MB
    (None = not listed); ``tags_error`` makes /api/tags fail.
    """

    def __init__(self, sizes=None, unload_answer=None, weights=None):
        self.sizes = list(sizes or [])
        self.ps_error: Exception | None = None
        self.unload_answer = unload_answer or {"done_reason": "unload"}
        self.unload_error: Exception | None = None
        self.weights = weights
        self.tags_error: Exception | None = None
        self.gets: list[str] = []
        self.tags_gets: list[str] = []
        self.posts: list[tuple[str, dict]] = []
        self.extra_models: list[dict] = []

    def get_json(self, url, timeout):
        assert timeout <= 5
        if url.endswith("/api/tags"):
            self.tags_gets.append(url)
            if self.tags_error is not None:
                raise self.tags_error
            if self.weights is None:
                return {"models": []}
            return {"models": [{"name": MODEL, "model": MODEL, "size": self.weights * MB}]}
        self.gets.append(url)
        if self.ps_error is not None:
            raise self.ps_error
        size = self.sizes.pop(0) if self.sizes else None
        models = list(self.extra_models)
        if size is not None:
            size, ctx = size if isinstance(size, tuple) else (size, None)
            entry = {"name": MODEL, "model": MODEL, "size": size * MB, "size_vram": 0}
            if ctx is not None:
                entry["context_length"] = ctx
            models.append(entry)
        return {"models": models}

    def post_json(self, url, payload, timeout):
        self.posts.append((url, payload))
        assert timeout <= 10
        if self.unload_error is not None:
            raise self.unload_error
        return self.unload_answer


def _trimmer(http, budget_mb=1024):
    return OllamaCacheTrimmer(budget_mb, http=http)


@pytest.mark.unit
class TestTrimDecision:
    def test_unloads_once_growth_exceeds_the_budget(self):
        http = FakeHttp(sizes=[4000, 4600, 5100])
        trimmer = _trimmer(http)
        assert not trimmer.check(URL, [MODEL])[0].trimmed  # baseline 4000
        assert not trimmer.check(URL, [MODEL])[0].trimmed  # +600
        result = trimmer.check(URL, [MODEL])[0]  # +1100 > 1024
        assert result.trimmed and result.baseline == 4000 * MB
        assert http.posts == [(f"{URL}/api/generate", {"model": MODEL, "keep_alive": 0})]
        assert http.gets == [f"{URL}/api/ps"] * 3
        assert trimmer.trims == 1

    def test_growth_exactly_at_the_budget_is_kept(self):
        http = FakeHttp(sizes=[4000, 5024])
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL])
        assert not trimmer.check(URL, [MODEL])[0].trimmed
        assert http.posts == []

    def test_a_smaller_size_lowers_the_baseline(self):
        http = FakeHttp(sizes=[4500, 4000, 5100])
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL])
        assert trimmer.check(URL, [MODEL])[0].baseline == 4000 * MB
        assert trimmer.check(URL, [MODEL])[0].trimmed

    def test_a_model_not_loaded_starts_over(self):
        # 4000 -> gone -> 5500: the new load's baseline is 5500, not 4000.
        http = FakeHttp(sizes=[4000, None, 5500])
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL])
        assert trimmer.check(URL, [MODEL])[0].loaded is False
        result = trimmer.check(URL, [MODEL])[0]
        assert result.baseline == 5500 * MB and not result.trimmed

    def test_budget_zero_never_reads_or_unloads(self):
        http = FakeHttp(sizes=[4000, 20000])
        trimmer = _trimmer(http, budget_mb=0)
        assert trimmer.check(URL, [MODEL]) == []
        assert trimmer.check(URL, [MODEL]) == []
        assert http.gets == [] and http.posts == []

    def test_only_the_runs_models_are_unloaded(self):
        http = FakeHttp(sizes=[4000, 4100])
        http.extra_models = [{"name": "other:7b", "model": "other:7b", "size": 30000 * MB}]
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL])
        trimmer.check(URL, [MODEL])
        assert http.posts == []

    def test_latest_tag_is_normalised(self):
        http = FakeHttp()
        http.extra_models = [{"name": "qwen3.5:latest", "model": "qwen3.5:latest", "size": 0}]
        trimmer = _trimmer(http)
        sizes = iter([4000, 6000])

        def get_json(url, timeout):
            if url.endswith("/api/tags"):
                return {"models": []}
            http.extra_models[0]["size"] = next(sizes) * MB
            return {"models": http.extra_models}

        http.get_json = get_json
        trimmer.check(URL, ["qwen3.5"])
        assert trimmer.check(URL, ["qwen3.5"])[0].trimmed
        assert http.posts[0][1]["model"] == "qwen3.5:latest"

    def test_quick_and_deep_models_are_both_checked_with_one_read(self):
        http = FakeHttp(sizes=[4000, 6000])
        http.extra_models = [{"name": "deep:14b", "size": 9000 * MB}]
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL, "deep:14b"])
        http.extra_models[0]["size"] = 10100 * MB
        results = trimmer.check(URL, [MODEL, "deep:14b", MODEL])
        assert [r.model for r in results] == [MODEL, "deep:14b"]  # de-duplicated
        assert all(r.trimmed for r in results)
        assert len(http.gets) == 2
        assert {p[1]["model"] for p in http.posts} == {MODEL, "deep:14b"}


@pytest.mark.unit
class TestSharedModelAndFailures:
    def test_within_budget_after_a_trim_is_taken_as_reloaded(self):
        # Trimmed at 5100; the next read (our reload, or fin-insight's) is 4100.
        http = FakeHttp(sizes=[4000, 5100, 4100, 4200])
        trimmer = _trimmer(http)
        for _ in range(4):
            trimmer.check(URL, [MODEL])
        assert len(http.posts) == 1

    def test_as_large_after_a_trim_is_retried_once(self):
        http = FakeHttp(sizes=[4000, 5100, 5200])
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL])
        assert trimmer.check(URL, [MODEL])[0].trimmed
        assert trimmer.check(URL, [MODEL])[0].trimmed
        assert len(http.posts) == 2

    def test_an_unload_without_done_reason_unload_is_a_failure(self):
        http = FakeHttp(sizes=[4000, 5100, 5100], unload_answer={"done_reason": "stop"})
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL])
        result = trimmer.check(URL, [MODEL])[0]
        assert not result.trimmed and "done_reason" in result.error
        assert trimmer.trims == 0

    def test_failed_unloads_back_off(self):
        http = FakeHttp(sizes=[4000] + [6000] * 30)
        http.unload_error = OSError("connection refused")
        trimmer = _trimmer(http)
        trimmer.check(URL, [MODEL])
        for _ in range(FAILURES_BEFORE_BACKOFF):
            trimmer.check(URL, [MODEL])
        assert len(http.posts) == FAILURES_BEFORE_BACKOFF
        deferred = [trimmer.check(URL, [MODEL])[0] for _ in range(BACKOFF_CHECKS - 1)]
        assert all(r.deferred for r in deferred)
        assert len(http.posts) == FAILURES_BEFORE_BACKOFF
        trimmer.check(URL, [MODEL])  # the tenth due check tries again
        assert len(http.posts) == FAILURES_BEFORE_BACKOFF + 1

    def test_failed_reads_never_raise_and_back_off(self):
        http = FakeHttp()
        http.ps_error = TimeoutError("timed out")
        trimmer = _trimmer(http)
        for _ in range(FAILURES_BEFORE_BACKOFF):
            assert trimmer.check(URL, [MODEL])[0].error == "timed out"
        for _ in range(BACKOFF_CHECKS - 1):
            assert trimmer.check(URL, [MODEL])[0].deferred
        assert len(http.gets) == FAILURES_BEFORE_BACKOFF
        trimmer.check(URL, [MODEL])
        assert len(http.gets) == FAILURES_BEFORE_BACKOFF + 1

    def test_a_malformed_answer_never_raises(self):
        http = FakeHttp()
        http.get_json = lambda url, timeout: ["not", "a", "dict"]
        [result] = _trimmer(http).check(URL, [MODEL])
        assert not result.loaded and result.error

    def test_checks_are_serialised_across_threads(self):
        active, peak = 0, 0
        guard = threading.Lock()

        class SlowHttp(FakeHttp):
            def get_json(self, url, timeout):
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with guard:
                    active -= 1
                return {"models": []}

        trimmer = _trimmer(SlowHttp())
        threads = [threading.Thread(target=trimmer.check, args=(URL, [MODEL])) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak == 1


@pytest.mark.unit
class TestReloadsByAnotherApp:
    """M1: fin-insight shares the model and reloads it with its own num_ctx."""

    def test_a_smaller_reload_past_the_old_budget_is_re_anchored_not_unloaded(self):
        # Trimmed at 5100; reloaded (by the other app) at 5060: smaller than at
        # the trim, yet more than 1024 MB past the old baseline of 4000.
        http = FakeHttp(sizes=[4000, 5100, 5060, 5200, 6100])
        trimmer = _trimmer(http)
        results = [trimmer.check(URL, [MODEL])[0] for _ in range(5)]
        assert [r.trimmed for r in results] == [False, True, False, False, True]
        assert results[2].baseline == 5060 * MB
        assert len(http.posts) == 2

    def test_a_reload_with_another_context_length_starts_over(self):
        # The reviewer's sequence, with /api/ps reporting the loads' contexts.
        http = FakeHttp(sizes=[(4000, 4096), (5100, 4096), (5200, 8192), (5300, 8192)])
        trimmer = _trimmer(http)
        results = [trimmer.check(URL, [MODEL])[0] for _ in range(4)]
        assert [r.trimmed for r in results] == [False, True, False, False]
        assert results[2].baseline == 5200 * MB
        assert len(http.posts) == 1

    def test_a_context_length_change_without_a_trim_starts_over(self):
        http = FakeHttp(sizes=[(4000, 4096), (4500, 4096), (6000, 32768), (6100, 32768)])
        trimmer = _trimmer(http)
        results = [trimmer.check(URL, [MODEL])[0] for _ in range(4)]
        assert not any(r.trimmed for r in results)
        assert results[2].baseline == 6000 * MB

    def test_an_unload_not_taken_is_retried_once_then_backs_off(self, caplog):
        # Answered unloads, but the model stays as large (or keeps coming back
        # larger): one retry at once, then only every NOT_TAKEN_BACKOFF_CHECKS-th.
        http = FakeHttp(sizes=[4000, 5100] + [5200] * (NOT_TAKEN_BACKOFF_CHECKS + 1))
        trimmer = _trimmer(http)
        logger = "tradingagents.api.infrastructure.ollama_cache_trimmer"
        with caplog.at_level(logging.INFO, logger=logger):
            trimmer.check(URL, [MODEL])
            assert trimmer.check(URL, [MODEL])[0].trimmed  # the trim
            assert trimmer.check(URL, [MODEL])[0].trimmed  # not taken #1: retried
            deferred = [trimmer.check(URL, [MODEL])[0] for _ in range(NOT_TAKEN_BACKOFF_CHECKS - 1)]
            assert all(r.deferred for r in deferred)
            assert len(http.posts) == 2
            assert trimmer.check(URL, [MODEL])[0].trimmed  # the 20th due check
        not_taken = [r for r in caplog.records if "as large after its cache trim" in r.message]
        assert [r.levelno for r in not_taken] == [logging.INFO, logging.WARNING]

    def test_not_taken_unloads_do_not_count_as_failures(self):
        http = FakeHttp(sizes=[4000, 5100, 5200, 5200, 4100, 5200])
        trimmer = _trimmer(http)
        for _ in range(5):
            trimmer.check(URL, [MODEL])
        state = trimmer._models[f"{URL} {MODEL}"]
        assert state.failures == 0
        # 4100 < 5200 at the last trim: a reload, so the counter starts over.
        assert state.not_taken == 0 and state.baseline == 4100 * MB
        assert trimmer.check(URL, [MODEL])[0].trimmed  # +1100 over the new baseline


@pytest.mark.unit
class TestWeightsReference:
    """M2: a model already grown at first sight is trimmed on that first check."""

    def test_a_bloated_model_is_trimmed_on_the_first_check(self):
        http = FakeHttp(sizes=[12000, 4100, 4300], weights=3000)
        trimmer = _trimmer(http)
        first = trimmer.check(URL, [MODEL])[0]
        assert first.trimmed
        assert first.baseline == (3000 + WEIGHTS_OVERHEAD_MB) * MB
        after = trimmer.check(URL, [MODEL])[0]  # reloaded: measured from here
        assert not after.trimmed and after.baseline == 4100 * MB
        assert not trimmer.check(URL, [MODEL])[0].trimmed
        assert len(http.posts) == 1
        assert http.tags_gets == [f"{URL}/api/tags"]

    def test_a_fresh_load_below_the_reference_keeps_its_own_baseline(self):
        http = FakeHttp(sizes=[4000, 4600], weights=3000)
        trimmer = _trimmer(http)
        assert trimmer.check(URL, [MODEL])[0].baseline == 4000 * MB
        assert not trimmer.check(URL, [MODEL])[0].trimmed

    def test_the_weights_are_read_once_per_server_and_model(self):
        http = FakeHttp(sizes=[4000, None, 4000, 4100], weights=3000)
        trimmer = _trimmer(http)
        for _ in range(4):
            trimmer.check(URL, [MODEL])
        assert http.tags_gets == [f"{URL}/api/tags"]
        other = "http://other.test:11434"
        http.sizes = [4000]
        trimmer.check(other, [MODEL])
        assert http.tags_gets == [f"{URL}/api/tags", f"{other}/api/tags"]

    @pytest.mark.parametrize("broken", ["error", "unlisted"])
    def test_without_the_weights_the_first_read_is_the_baseline(self, broken):
        http = FakeHttp(sizes=[12000, 12500], weights=None if broken == "unlisted" else 3000)
        if broken == "error":
            http.tags_error = OSError("connection refused")
        trimmer = _trimmer(http)
        assert trimmer.check(URL, [MODEL])[0].baseline == 12000 * MB
        assert not trimmer.check(URL, [MODEL])[0].trimmed
        assert http.posts == []

    def test_a_model_whose_loads_are_that_large_is_trimmed_once_at_most(self):
        # A long context (or a preallocated cache) makes every fresh load
        # 9000 MB: the reference trim is taken back, and not used again.
        http = FakeHttp(sizes=[9000, 9000, 9100, None, 9000, 9050], weights=3000)
        trimmer = _trimmer(http)
        results = [trimmer.check(URL, [MODEL])[0] for _ in range(6)]
        assert [r.trimmed for r in results] == [True, False, False, False, False, False]
        assert results[4].baseline == 9000 * MB
        assert trimmer._models[f"{URL} {MODEL}"].not_taken == 0


@pytest.mark.unit
class TestMisnamedModel:
    def test_warns_once_when_other_models_are_loaded_but_not_ours(self, caplog):
        http = FakeHttp()
        http.extra_models = [{"name": "qwen3.5:4b", "model": "qwen3.5:4b", "size": 4000 * MB}]
        trimmer = _trimmer(http)
        with caplog.at_level(logging.WARNING):
            trimmer.check(URL, ["qwen3.5:4b-mlxx"])
            trimmer.check(URL, ["qwen3.5:4b-mlxx"])
        warnings = [r for r in caplog.records if "not among the loaded ones" in r.message]
        assert len(warnings) == 1
        assert "qwen3.5:4b-mlxx" in warnings[0].message and "qwen3.5:4b" in warnings[0].message

    def test_nothing_loaded_is_not_a_misnamed_model(self, caplog):
        trimmer = _trimmer(FakeHttp())
        with caplog.at_level(logging.WARNING):
            trimmer.check(URL, [MODEL])
        assert not [r for r in caplog.records if "not among the loaded ones" in r.message]


@pytest.mark.unit
class TestNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("qwen3.5", "qwen3.5:latest"),
            ("qwen3.5:4b-mlx", "qwen3.5:4b-mlx"),
            ("localhost:5000/ns/name", "localhost:5000/ns/name:latest"),
            (" ", ""),
        ],
    )
    def test_normalize_model_name(self, name, expected):
        assert normalize_model_name(name) == expected

    @pytest.mark.parametrize(
        ("backend_url", "expected"),
        [
            ("http://host.docker.internal:11434/v1", "http://host.docker.internal:11434"),
            ("http://host.docker.internal:11434/v1/", "http://host.docker.internal:11434"),
            ("http://gpu.test:11434", "http://gpu.test:11434"),
        ],
    )
    def test_native_url_strips_v1(self, backend_url, expected, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        assert ollama_native_url(backend_url) == expected

    def test_native_url_falls_back_to_env_then_default(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env.test:11434/v1")
        assert ollama_native_url(None) == "http://env.test:11434"
        monkeypatch.delenv("OLLAMA_BASE_URL")
        assert ollama_native_url(None) == "http://localhost:11434"


@pytest.mark.unit
class TestCallbackWiring:
    def test_handler_checks_after_every_llm_call(self):
        trimmer = MagicMock()
        handler = OllamaCacheTrimHandler(trimmer, URL, [MODEL])
        model = FakeListChatModel(responses=["a", "b"], callbacks=[handler])
        model.invoke("hi")
        model.invoke("hi")
        assert trimmer.check.call_count == 2
        trimmer.check.assert_called_with(URL, (MODEL,))

    def test_handler_checks_after_a_failed_call(self):
        trimmer = MagicMock()
        handler = OllamaCacheTrimHandler(trimmer, URL, [MODEL])
        handler.on_llm_error(RuntimeError("boom"))
        assert trimmer.check.call_count == 1

    def test_handler_never_fails_the_call(self):
        trimmer = MagicMock()
        trimmer.check.side_effect = RuntimeError("trimmer bug")
        handler = OllamaCacheTrimHandler(trimmer, URL, [MODEL])
        model = FakeListChatModel(responses=["a"], callbacks=[handler])
        assert model.invoke("hi").content == "a"

    def _callbacks_for(self, config, trimmer, overrides=None):
        with patch(
            "tradingagents.api.domain.services.analysis_service.TradingAgentsGraph"
        ) as graph_cls:
            graph_cls.return_value.propagate.return_value = ({}, "Hold")
            AnalysisService(config=config, cache_trimmer=trimmer).run_analysis(
                "AAPL", "2026-06-01", overrides=overrides
            )
        return graph_cls.call_args.kwargs["callbacks"] or []

    def test_ollama_runs_get_the_trim_handler_for_their_models(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        trimmer = OllamaCacheTrimmer(1024, http=FakeHttp())
        config = {
            "llm_provider": "Ollama",
            "backend_url": "http://host.docker.internal:11434/v1",
            "quick_think_llm": "qwen3.5:4b-mlx",
            "deep_think_llm": "qwen3.5:4b-mlx",
        }
        callbacks = self._callbacks_for(config, trimmer, overrides={"deep_think_llm": "deep:14b"})
        [handler] = [c for c in callbacks if isinstance(c, OllamaCacheTrimHandler)]
        assert handler._native_url == "http://host.docker.internal:11434"
        assert handler._models == ("qwen3.5:4b-mlx", "deep:14b")

    def test_other_providers_get_no_trim_handler(self):
        trimmer = OllamaCacheTrimmer(1024, http=FakeHttp())
        callbacks = self._callbacks_for({"llm_provider": "openai"}, trimmer)
        assert not [c for c in callbacks if isinstance(c, OllamaCacheTrimHandler)]

    def test_app_shares_one_trimmer_and_budget_zero_disables_it(self):
        app = create_app(overrides={"llm_provider": "ollama", "ollama_cache_trim_mb": 1024})
        assert app.state.ollama_cache_trimmer.budget_mb == 1024
        off = create_app(overrides={"llm_provider": "ollama", "ollama_cache_trim_mb": 0})
        assert off.state.ollama_cache_trimmer is None

    def test_task_runs_use_the_apps_trimmer(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        captured = {}

        def fake_graph(*args, **kwargs):
            captured["callbacks"] = kwargs.get("callbacks") or []
            graph = MagicMock()
            graph.propagate.return_value = ({}, "Hold")
            return graph

        with patch(
            "tradingagents.api.domain.services.analysis_service.TradingAgentsGraph",
            side_effect=fake_graph,
        ):
            app = create_app(overrides={"llm_provider": "ollama"})
            client = TestClient(app)
            client.post(
                "/api/v1/analyze/tasks", json={"ticker": "AAPL", "trade_date": "2026-06-01"}
            )
            app.state.task_worker.shutdown(wait=True)
        [handler] = [c for c in captured["callbacks"] if isinstance(c, OllamaCacheTrimHandler)]
        assert handler._trimmer is app.state.ollama_cache_trimmer


@pytest.mark.unit
class TestNoTestReachesOllama:
    """M3: the conftest guard refuses the trimmer's real HTTP client."""

    def test_the_guard_refuses_real_ollama_calls(self, ollama_http_guard, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")

        class Graph:
            def __init__(self, *args, callbacks=None, **kwargs):
                self.model = FakeListChatModel(responses=["a"], callbacks=callbacks)

            def propagate(self, *args, **kwargs):
                self.model.invoke("x")  # the trim handler checks after it
                return {"final_trade_decision": "Buy"}, "Buy"

        app = create_app(overrides={"llm_provider": "ollama"})
        assert type(app.state.ollama_cache_trimmer._http).__name__ == "UrllibOllamaHttp"
        with patch("tradingagents.api.domain.services.analysis_service.TradingAgentsGraph", Graph):
            resp = TestClient(app).post(
                "/api/v1/analyze", json={"ticker": "AAPL", "trade_date": "2026-06-01"}
            )
            app.state.task_worker.shutdown(wait=True)
        assert resp.status_code == 200  # the refused read never fails the run
        assert ollama_http_guard == ["http://host.docker.internal:11434/api/ps"]
        ollama_http_guard.clear()  # seen: don't fail this test at teardown
