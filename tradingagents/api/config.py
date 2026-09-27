"""API configuration and settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tradingagents.default_api_config import DEFAULT_API_CONFIG, _coerce
from tradingagents.default_config import DEFAULT_CONFIG

# Supported-symbols list settings. Kept in the API package (rather than in
# default_api_config.py) so the feature stays self-contained; the env vars
# follow the same TRADINGAGENTS_* convention and are coerced the same way. An
# empty value (e.g. compose's ``${VAR:-}``) counts as unset, as in the core.
_SYMBOLS_ENV_OVERRIDES = {
    "TRADINGAGENTS_SYMBOLS_CACHE_DIR": "symbols_cache_dir",
    "TRADINGAGENTS_SYMBOLS_CACHE_TTL_DAYS": "symbols_cache_ttl_days",
    "TRADINGAGENTS_SYMBOLS_AUTO_REFRESH": "symbols_auto_refresh",
    "TRADINGAGENTS_SYMBOLS_INCLUDE_OTC": "symbols_include_otc",
    "TRADINGAGENTS_SYMBOLS_PAGE_DELAY_SECONDS": "symbols_page_delay_seconds",
    "TRADINGAGENTS_SYMBOLS_MAX_PAGES": "symbols_max_pages",
    "TRADINGAGENTS_SYMBOLS_REFRESH_WINDOW": "symbols_refresh_window",
    "TRADINGAGENTS_SYMBOLS_REFRESH_TIMEZONE": "symbols_refresh_timezone",
    "TRADINGAGENTS_SYMBOLS_IDLE_GRACE_MINUTES": "symbols_idle_grace_minutes",
}
# Config keys whose value came from an env var (reported as source "env").
SYMBOLS_ENV_KEYS: set[str] = set()


def _symbols_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "symbols_cache_dir": None,  # None -> <data_cache_dir>/symbols
        "symbols_cache_ttl_days": 7.0,  # float, so "0.5" from the env var works
        "symbols_auto_refresh": True,
        "symbols_include_otc": False,
        "symbols_page_delay_seconds": 2.0,  # pause between Yahoo requests
        "symbols_max_pages": 200,  # per screener query (250 rows a page)
        # Automatic refreshes of stale lists wait until TradingAgents has been
        # idle (no analysis task, no Yahoo-backed request) for this long.
        "symbols_idle_grace_minutes": 10.0,
        # Optional extra condition: a daily window ("HH:MM-HH:MM", may wrap
        # midnight, local to the zone below). "" (the default) = any time.
        "symbols_refresh_window": "",
        "symbols_refresh_timezone": "Asia/Taipei",
    }
    for env_var, key in _SYMBOLS_ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or not raw.strip():
            continue  # unset or empty: the default applies
        try:
            defaults[key] = _coerce(raw, defaults[key])
        except ValueError as exc:
            raise ValueError(f"invalid {env_var}={raw!r}: {exc}") from exc
        SYMBOLS_ENV_KEYS.add(key)
    return defaults


def symbols_env_var(key: str) -> str | None:
    """The ``TRADINGAGENTS_SYMBOLS_*`` env var that sets config ``key``, if any."""
    return next((var for var, k in _SYMBOLS_ENV_OVERRIDES.items() if k == key), None)


SYMBOLS_DEFAULT_CONFIG = _symbols_defaults()


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
        self._config: dict[str, Any] = {
            **DEFAULT_CONFIG,
            **DEFAULT_API_CONFIG,
            **SYMBOLS_DEFAULT_CONFIG,
        }
        self._override_keys = set(overrides or ())
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

    @property
    def symbols_cache_dir(self) -> Path:
        """Return the directory holding the supported-symbols cache files."""
        configured = self._config.get("symbols_cache_dir")
        return Path(configured) if configured else self.data_cache_dir / "symbols"

    def source_of(self, key: str) -> str:
        """Where ``key``'s value came from: ``config`` (overrides), ``env`` or ``default``.

        Precedence: ``config`` > ``env`` > ``default`` (``PUT
        /symbols/settings`` overrides all three; see ``SymbolSettings``).
        """
        if key in self._override_keys:
            return "config"
        if key in SYMBOLS_ENV_KEYS:
            return "env"
        return "default"

    def origin_of(self, key: str) -> str:
        """``key``'s origin for an error message: the env var or the config key."""
        source = self.source_of(key)
        if source == "env":
            return f"env var {symbols_env_var(key)}"
        if source == "config":
            return f"config key {key!r}"
        return f"default of {key!r}"

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
