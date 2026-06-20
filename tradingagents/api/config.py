"""API configuration and settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.default_api_config import DEFAULT_API_CONFIG


class ApiConfig:
    """Configuration for the TradingAgents API.

    Wraps the core DEFAULT_CONFIG with API-specific settings.
    """

    def __init__(self, overrides: dict[str, Any] | None = None):
        """Initialize API configuration.

        Merges core DEFAULT_CONFIG with API-specific DEFAULT_API_CONFIG.

        Args:
            overrides: Optional dict to override default config values.
        """
        self._config: dict[str, Any] = {**DEFAULT_CONFIG, **DEFAULT_API_CONFIG}
        if overrides:
            self._config.update(overrides)

    @property
    def config(self) -> dict[str, Any]:
        """Return the full config dict."""
        return self._config

    @property
    def data_cache_dir(self) -> Path:
        """Return the data cache directory."""
        return Path(self._config.get("data_cache_dir", ".cache"))

    @property
    def results_dir(self) -> Path:
        """Return the results directory."""
        return Path(self._config.get("results_dir", ".results"))

    @property
    def llm_provider(self) -> str:
        """Return the LLM provider name."""
        return self._config.get("llm_provider", "openai")

    @property
    def debug(self) -> bool:
        """Return debug flag."""
        return bool(self._config.get("debug", False))

    @property
    def api_title(self) -> str:
        """Return the API title for OpenAPI docs."""
        return self._config.get("api_title", "TradingAgents API")

    @property
    def api_version(self) -> str:
        """Return the API version."""
        return self._config.get("api_version", "v1")

    @property
    def api_host(self) -> str:
        """Return the API host."""
        return self._config.get("api_host", "0.0.0.0")

    @property
    def api_port(self) -> int:
        """Return the API port."""
        return int(self._config.get("api_port", 8000))

    @property
    def analyst_concurrency_limit(self) -> int:
        """Return the analyst concurrency limit."""
        return int(self._config.get("analyst_concurrency_limit", 4))

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Global config singleton
_global_config: ApiConfig | None = None


def get_config() -> ApiConfig:
    """Get the global API config, creating it if necessary."""
    global _global_config
    if _global_config is None:
        _global_config = ApiConfig()
    return _global_config


def set_config(config: ApiConfig) -> None:
    """Set the global API config."""
    global _global_config
    _global_config = config