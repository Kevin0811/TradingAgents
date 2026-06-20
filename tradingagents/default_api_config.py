"""Default configuration for TradingAgents API.

This module provides API-specific configuration that is merged with the
core DEFAULT_CONFIG from default_config.py.  Environment variable overrides
follow the same pattern as the main config.
"""

import os

# Environment variable overrides for API config
_ENV_API_OVERRIDES = {
    "TRADINGAGENTS_API_TITLE": "api_title",
    "TRADINGAGENTS_API_VERSION": "api_version",
    "TRADINGAGENTS_API_HOST": "api_host",
    "TRADINGAGENTS_API_PORT": "api_port",
    "TRADINGAGENTS_TASK_TTL_MINUTES": "task_ttl_minutes",
    "TRADINGAGENTS_TASK_MAX_TASKS": "task_max_tasks",
    "TRADINGAGENTS_TASK_MAX_CONCURRENT": "task_max_concurrent",
    "TRADINGAGENTS_ANALYST_CONCURRENCY_LIMIT": "analyst_concurrency_limit",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_API_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


DEFAULT_API_CONFIG = _apply_env_overrides({
    # API server settings
    "api_title": "TradingAgents API",
    "api_version": "v1",
    "api_host": "0.0.0.0",
    "api_port": 8000,

    # Task Queue settings (async analysis)
    "task_ttl_minutes": 60,          # Task retention time in minutes
    "task_max_tasks": 100,           # Maximum number of tasks in memory
    "task_max_concurrent": 2,        # Maximum concurrent task executions

    # Analyst concurrency
    "analyst_concurrency_limit": 4,
})