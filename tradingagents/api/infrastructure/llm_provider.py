"""LLM provider factory for creating LLM clients.

Configuration is sourced from .env file via DEFAULT_CONFIG,
which applies TRADINGAGENTS_* environment variable overrides.
"""

from __future__ import annotations

from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client


class LLMProviderFactory:
    """Factory for creating LLM clients based on .env configuration.

    Configuration is read from DEFAULT_CONFIG which includes:
    - Default values from default_config.py
    - Overrides from .env file (TRADINGAGENTS_* variables)
    - API keys from environment (OPENAI_API_KEY, etc.)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the LLM provider factory.

        Args:
            config: Optional configuration dict to override .env settings.
                   When None, uses DEFAULT_CONFIG from .env file.
        """
        self._config = config if config is not None else DEFAULT_CONFIG

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Extract provider-specific kwargs from config."""
        kwargs = {}
        provider = self._config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self._config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level
        elif provider == "openai":
            reasoning_effort = self._config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        elif provider == "anthropic":
            effort = self._config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        temperature = self._config.get("temperature")
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)

        return kwargs

    def create_deep_thinking_llm(self) -> Any:
        """Create a deep-thinking LLM client.

        Returns:
            LangChain LLM instance.
        """
        kwargs = self._get_provider_kwargs()
        client = create_llm_client(
            provider=self._config.get("llm_provider", "openai"),
            model=self._config.get("deep_think_llm", "o3-mini"),
            base_url=self._config.get("backend_url"),
            **kwargs,
        )
        return client.get_llm()

    def create_quick_thinking_llm(self) -> Any:
        """Create a quick-thinking LLM client.

        Returns:
            LangChain LLM instance.
        """
        kwargs = self._get_provider_kwargs()
        client = create_llm_client(
            provider=self._config.get("llm_provider", "openai"),
            model=self._config.get("quick_think_llm", "gpt-4o-mini"),
            base_url=self._config.get("backend_url"),
            **kwargs,
        )
        return client.get_llm()

    def create_llm(self, model: str | None = None) -> Any:
        """Create an LLM client with optional model override.

        Args:
            model: Optional model name override.

        Returns:
            LangChain LLM instance.
        """
        kwargs = self._get_provider_kwargs()
        model_name = model or self._config.get("deep_think_llm", "o3-mini")
        client = create_llm_client(
            provider=self._config.get("llm_provider", "openai"),
            model=model_name,
            base_url=self._config.get("backend_url"),
            **kwargs,
        )
        return client.get_llm()
