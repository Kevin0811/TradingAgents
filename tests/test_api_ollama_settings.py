"""Tests for the Ollama-related API settings: the worker default and the trim budget.

Every test pins the values a contributor's environment or ``.env`` could set
(``TRADINGAGENTS_TASK_MAX_CONCURRENT``, ``TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB``
and what the package folded in from them at import), so none can change an
outcome here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tradingagents.api import config as config_module
from tradingagents.api.app import create_app
from tradingagents.api.config import ApiConfig


@pytest.fixture(autouse=True)
def _pinned_settings(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_TASK_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(config_module, "TASK_MAX_CONCURRENT_FROM_ENV", False)
    monkeypatch.setitem(config_module.DEFAULT_API_CONFIG, "task_max_concurrent", 2)
    monkeypatch.setattr(config_module, "OLLAMA_DEFAULT_CONFIG", config_module._ollama_defaults())


@pytest.mark.unit
class TestTaskMaxConcurrent:
    def test_ollama_defaults_to_one_worker(self):
        assert ApiConfig({"llm_provider": "ollama"}).task_max_concurrent == 1
        assert ApiConfig({"llm_provider": " Ollama "}).task_max_concurrent == 1

    def test_other_providers_keep_the_default(self):
        assert ApiConfig({"llm_provider": "openai"}).task_max_concurrent == 2

    def test_an_explicit_config_value_wins(self):
        config = ApiConfig({"llm_provider": "ollama", "task_max_concurrent": 3})
        assert config.task_max_concurrent == 3

    def test_an_explicit_env_value_wins(self, monkeypatch):
        # DEFAULT_API_CONFIG folded the env value in at import; stand in for it.
        monkeypatch.setattr(config_module, "TASK_MAX_CONCURRENT_FROM_ENV", True)
        monkeypatch.setitem(config_module.DEFAULT_API_CONFIG, "task_max_concurrent", 3)
        assert ApiConfig({"llm_provider": "ollama"}).task_max_concurrent == 3

    def test_a_none_override_counts_as_unset(self):
        overrides = {"llm_provider": "ollama", "task_max_concurrent": None}
        assert ApiConfig(overrides).task_max_concurrent == 1
        overrides["llm_provider"] = "openai"
        assert ApiConfig(overrides).task_max_concurrent == 2
        app = create_app(overrides={"llm_provider": "ollama", "task_max_concurrent": None})
        assert app.state.task_worker.max_workers == 1

    def test_the_worker_and_get_config_use_the_effective_value(self):
        app = create_app(overrides={"llm_provider": "ollama"})
        assert app.state.task_worker.max_workers == 1
        body = TestClient(app).get("/api/v1/config").json()
        assert body["task_max_concurrent"] == 1
        assert body["ollama_cache_trim_mb"] == 1024
        assert body["ollama_cache_trim_active"] is True


@pytest.mark.unit
class TestOllamaCacheTrimSetting:
    def test_default_is_1024_mb(self):
        assert ApiConfig().ollama_cache_trim_mb == 1024

    def test_env_var_sets_it_and_empty_means_unset(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB", "512")
        assert config_module._ollama_defaults()["ollama_cache_trim_mb"] == 512
        monkeypatch.setenv("TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB", " ")
        assert config_module._ollama_defaults()["ollama_cache_trim_mb"] == 1024
        monkeypatch.setenv("TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB", "0")
        assert config_module._ollama_defaults()["ollama_cache_trim_mb"] == 0

    def test_a_malformed_env_value_names_the_env_var(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB", "lots")
        with pytest.raises(ValueError, match="TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB='lots'"):
            config_module._ollama_defaults()

    def test_config_override_wins_over_the_env(self, monkeypatch):
        monkeypatch.setitem(config_module.OLLAMA_DEFAULT_CONFIG, "ollama_cache_trim_mb", 512)
        assert ApiConfig().ollama_cache_trim_mb == 512
        assert ApiConfig({"ollama_cache_trim_mb": 0}).ollama_cache_trim_mb == 0

    @pytest.mark.parametrize(
        ("overrides", "active"),
        [
            ({"llm_provider": "ollama"}, True),
            ({"llm_provider": "ollama", "ollama_cache_trim_mb": 0}, False),
            ({"llm_provider": "openai"}, False),
            ({"llm_provider": "anthropic", "ollama_cache_trim_mb": 2048}, False),
        ],
    )
    def test_the_trim_is_active_only_with_ollama(self, overrides, active):
        app = create_app(overrides=overrides)
        assert (app.state.ollama_cache_trimmer is not None) is active
        body = TestClient(app).get("/api/v1/config").json()
        assert body["ollama_cache_trim_active"] is active
        assert body["ollama_cache_trim_mb"] == overrides.get("ollama_cache_trim_mb", 1024)
