"""LLM provider factory for creating LLM clients.

Configuration is sourced from .env file via DEFAULT_CONFIG,
which applies TRADINGAGENTS_* environment variable overrides.

Clients are built by the core's ``create_tier_client`` so the API follows the
core's provider handling (per-tier providers and endpoints, thinking/effort
settings, ``max_tokens``, ``llm_max_retries``) instead of keeping its own copy.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.factory import create_tier_client, tier_provider

TIERS: tuple[str, ...] = ("quick", "deep")


def _full_config(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` on top of DEFAULT_CONFIG: the core reads keys a partial dict may omit.

    A key missing from ``config`` takes DEFAULT_CONFIG's value, which includes
    TRADINGAGENTS_* env and .env overrides (before, a partial dict fell back to
    hard-coded "openai" / "o3-mini" / "gpt-4o-mini"). The API always passes a
    full ApiConfig, so only direct callers with a partial dict see this.
    """
    return {**DEFAULT_CONFIG, **config}


def check_llm_config(config: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a model tier's settings would be refused on every call.

    Catches before a run what otherwise fails it midway: an unknown provider,
    and a ``temperature`` the tier's Anthropic model does not accept (Claude
    Sonnet 5.5 and Fable 5.1 reject any non-default value). Which models refuse
    it is left to ``langchain-anthropic``, by building one request payload
    offline; nothing is sent.
    """
    config = _full_config(config)
    temperature = config.get("temperature")
    for tier in TIERS:
        try:
            client = create_tier_client(config, tier)
            if tier_provider(config, tier) != "anthropic" or temperature in (None, ""):
                continue
            llm = client.get_llm()
        # TypeError/AttributeError: a value of the wrong type (e.g. a None provider
        # or a list temperature); pydantic's ValidationError is a ValueError.
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{tier}-thinking tier: {exc}") from exc
        build_payload = getattr(llm, "_get_request_payload", None)
        if build_payload is None:
            continue
        try:
            build_payload([HumanMessage(content="ping")])
        except ValueError as exc:
            raise ValueError(
                f"{tier}-thinking model {config[f'{tier}_think_llm']!r} rejects the server's "
                f"temperature setting ({exc}). Choose a model that accepts it, or unset "
                "TRADINGAGENTS_TEMPERATURE on the server."
            ) from exc


class LLMProviderFactory:
    """Factory for creating LLM clients based on .env configuration.

    Configuration is read from DEFAULT_CONFIG which includes:
    - Default values from default_config.py
    - Overrides from .env file (TRADINGAGENTS_* variables)
    - API keys from environment (OPENAI_API_KEY, etc.)

    A partial ``config`` is completed from DEFAULT_CONFIG (env and .env
    included), not from hard-coded OpenAI defaults; see ``_full_config``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the LLM provider factory.

        Args:
            config: Optional configuration dict to override .env settings.
                   When None, uses DEFAULT_CONFIG from .env file.
        """
        self._config = config if config is not None else DEFAULT_CONFIG

    def _tier_llm(self, tier: str, model: str | None = None) -> Any:
        """The LangChain model for ``tier``, optionally with another model ID."""
        config = _full_config(self._config)
        if model:
            config[f"{tier}_think_llm"] = model
        return create_tier_client(config, tier).get_llm()

    def create_deep_thinking_llm(self) -> Any:
        """Create a deep-thinking LLM client.

        Returns:
            LangChain LLM instance.
        """
        return self._tier_llm("deep")

    def create_quick_thinking_llm(self) -> Any:
        """Create a quick-thinking LLM client.

        Returns:
            LangChain LLM instance.
        """
        return self._tier_llm("quick")

    def create_llm(self, model: str | None = None) -> Any:
        """Create an LLM client with optional model override.

        Uses the deep-thinking tier's provider and settings.

        Args:
            model: Optional model name override.

        Returns:
            LangChain LLM instance.
        """
        return self._tier_llm("deep", model)
