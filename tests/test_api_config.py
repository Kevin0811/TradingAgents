"""Tests for the TradingAgents API configuration layer.

Covers:
- ApiConfig default values and property accessors
- Override merging
- ensure_directories() side effects
- Global get_config() / set_config() singleton isolation
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.api.config import ApiConfig, get_config, set_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.default_api_config import DEFAULT_API_CONFIG


@pytest.fixture(autouse=True)
def _reset_global_config():
    """Reset the global config singleton before and after each test."""
    import tradingagents.api.config as config_module
    config_module._global_config = None
    yield
    config_module._global_config = None


@pytest.mark.unit
class TestApiConfigDefaults:
    """Verify ApiConfig exposes the expected defaults from DEFAULT_CONFIG + DEFAULT_API_CONFIG."""

    def test_config_merges_core_and_api_defaults(self):
        cfg = ApiConfig()
        # Core config keys must be present
        assert "llm_provider" in cfg.config
        assert "results_dir" in cfg.config
        # API-specific keys must be present
        assert "api_title" in cfg.config
        assert "api_version" in cfg.config
        assert "api_host" in cfg.config
        assert "api_port" in cfg.config

    def test_config_allows_overrides(self):
        overrides = {
            "llm_provider": "google",
            "api_title": "Custom API",
            "api_version": "v2",
            "api_host": "127.0.0.1",
            "api_port": 9000,
        }
        cfg = ApiConfig(overrides=overrides)
        assert cfg.config["llm_provider"] == "google"
        assert cfg.config["api_title"] == "Custom API"
        assert cfg.config["api_version"] == "v2"
        assert cfg.config["api_host"] == "127.0.0.1"
        assert cfg.config["api_port"] == 9000

    def test_config_overrides_do_not_clobber_unrelated_defaults(self):
        overrides = {"llm_provider": "anthropic"}
        cfg = ApiConfig(overrides=overrides)
        # Unrelated API defaults should remain at their defaults
        assert cfg.config["api_version"] == DEFAULT_API_CONFIG["api_version"]
        assert cfg.config["api_host"] == DEFAULT_API_CONFIG["api_host"]


@pytest.mark.unit
class TestApiConfigProperties:
    """Property accessors must return the correct types and values."""

    def test_data_cache_dir_returns_path(self):
        cfg = ApiConfig()
        assert isinstance(cfg.data_cache_dir, Path)

    def test_results_dir_returns_path(self):
        cfg = ApiConfig()
        assert isinstance(cfg.results_dir, Path)

    def test_llm_provider_returns_string(self):
        cfg = ApiConfig()
        assert isinstance(cfg.llm_provider, str)
        assert cfg.llm_provider == DEFAULT_CONFIG.get("llm_provider", "openai")

    def test_debug_returns_bool(self):
        cfg = ApiConfig()
        assert isinstance(cfg.debug, bool)

    def test_api_title_returns_string(self):
        cfg = ApiConfig()
        assert isinstance(cfg.api_title, str)
        assert cfg.api_title == DEFAULT_API_CONFIG.get("api_title", "TradingAgents API")

    def test_api_version_returns_string(self):
        cfg = ApiConfig()
        assert isinstance(cfg.api_version, str)
        assert cfg.api_version == DEFAULT_API_CONFIG.get("api_version", "v1")

    def test_api_host_returns_string(self):
        cfg = ApiConfig()
        assert isinstance(cfg.api_host, str)
        assert cfg.api_host == DEFAULT_API_CONFIG.get("api_host", "0.0.0.0")

    def test_api_port_returns_int(self):
        cfg = ApiConfig()
        assert isinstance(cfg.api_port, int)
        assert cfg.api_port == DEFAULT_API_CONFIG.get("api_port", 8000)

    def test_analyst_concurrency_limit_returns_int(self):
        cfg = ApiConfig()
        assert isinstance(cfg.analyst_concurrency_limit, int)


@pytest.mark.unit
class TestApiConfigEnsureDirectories:
    """ensure_directories() must create missing directories."""

    def test_creates_data_cache_dir(self, tmp_path):
        overrides = {"data_cache_dir": str(tmp_path / "cache")}
        cfg = ApiConfig(overrides=overrides)
        cache_dir = cfg.data_cache_dir
        assert not cache_dir.exists()
        cfg.ensure_directories()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_creates_results_dir(self, tmp_path):
        overrides = {"results_dir": str(tmp_path / "results")}
        cfg = ApiConfig(overrides=overrides)
        results_dir = cfg.results_dir
        assert not results_dir.exists()
        cfg.ensure_directories()
        assert results_dir.exists()
        assert results_dir.is_dir()

    def test_idempotent_when_dirs_exist(self, tmp_path):
        cache = tmp_path / "cache"
        results = tmp_path / "results"
        cache.mkdir()
        results.mkdir()
        cfg = ApiConfig(overrides={"data_cache_dir": str(cache), "results_dir": str(results)})
        cfg.ensure_directories()  # must not raise
        assert cache.exists()
        assert results.exists()


@pytest.mark.unit
class TestGlobalConfigSingleton:
    """get_config() / set_config() must manage a global singleton."""

    def test_get_config_creates_new_instance(self):
        cfg = get_config()
        assert isinstance(cfg, ApiConfig)

    def test_get_config_returns_same_instance(self):
        first = get_config()
        second = get_config()
        assert first is second

    def test_set_config_replaces_singleton(self):
        custom = ApiConfig(overrides={"api_title": "Custom"})
        set_config(custom)
        assert get_config() is custom
        assert get_config().api_title == "Custom"

    def test_set_config_isolation_does_not_leak(self):
        """After one test sets config, the next test (via autouse fixture) gets a fresh one."""
        custom = ApiConfig(overrides={"api_title": "LeakTest"})
        set_config(custom)
        assert get_config().api_title == "LeakTest"
        # The autouse fixture resets _global_config to None after this test,
        # so subsequent tests start fresh.