"""FastAPI dependency injection for the TradingAgents API."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends

from tradingagents.agents import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)
from tradingagents.api.config import ApiConfig, get_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients import create_llm_client


# ---------------------------------------------------------------------------
# LLM client dependencies
# ---------------------------------------------------------------------------


@lru_cache
def get_llm_deep(
    config: ApiConfig = Depends(get_config),
) -> Any:
    """Create a deep-thinking LLM client."""
    llm_kwargs = _get_provider_kwargs(config.config)
    client = create_llm_client(
        provider=config.llm_provider,
        model=config.config.get("deep_thinkllm", config.config.get("deep_think_lmm", "o3-mini")),
        base_url=config.config.get("backend_url"),
        **llm_kwargs,
    )
    return client.get_llm()


@lru_cache
def get_llm_quick(
    config: ApiConfig = Depends(get_config),
) -> Any:
    """Create a quick-thinking LLM client."""
    llm_kwargs = _get_provider_kwargs(config.config)
    client = create_llm_client(
        provider=config.llm_provider,
        model=config.config.get("quick_think_llm", "gpt-4o-mini"),
        base_url=config.config.get("backend_url"),
        **llm_kwargs,
    )
    return client.get_llm()


def _get_provider_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract provider-specific kwargs from config."""
    kwargs = {}
    provider = config.get("llm_provider", "").lower()

    if provider == "google":
        thinking_level = config.get("google_thinking_level")
        if thinking_level:
            kwargs["thinking_level"] = thinking_level
    elif provider == "openai":
        reasoning_effort = config.get("openai_reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    elif provider == "anthropic":
        effort = config.get("anthropic_effort")
        if effort:
            kwargs["effort"] = effort

    temperature = config.get("temperature")
    if temperature is not None and temperature != "":
        kwargs["temperature"] = float(temperature)

    return kwargs


# ---------------------------------------------------------------------------
# Analyst dependencies
# ---------------------------------------------------------------------------


def get_market_analyst(
    config: ApiConfig = Depends(get_config),
) -> callable:
    """Create a market analyst node factory."""
    from tradingagents.api.services import _create_market_analyst_node
    llm_kwargs = _get_provider_kwargs(config.config)
    client = create_llm_client(
        provider=config.llm_provider,
        model=config.config.get("deep_think_llm", "o3-mini"),
        base_url=config.config.get("backend_url"),
        **llm_kwargs,
    )
    return lambda: _create_market_analyst_node(client.get_llm())


def get_sentiment_analyst(
    config: ApiConfig = Depends(get_config),
) -> callable:
    """Create a sentiment analyst node factory."""
    llm_kwargs = _get_provider_kwargs(config.config)
    client = create_llm_client(
        provider=config.llm_provider,
        model=config.config.get("deep_think_llm", "o3-mini"),
        base_url=config.config.get("backend_url"),
        **llm_kwargs,
    )
    return lambda: create_sentiment_analyst(client.get_llm())


def get_news_analyst(
    config: ApiConfig = Depends(get_config),
) -> callable:
    """Create a news analyst node factory."""
    llm_kwargs = _get_provider_kwargs(config.config)
    client = create_llm_client(
        provider=config.llm_provider,
        model=config.config.get("deep_think_llm", "o3-mini"),
        base_url=config.config.get("backend_url"),
        **llm_kwargs,
    )
    return lambda: create_news_analyst(client.get_llm())


def get_fundamentals_analyst(
    config: ApiConfig = Depends(get_config),
) -> callable:
    """Create a fundamentals analyst node factory."""
    llm_kwargs = _get_provider_kwargs(config.config)
    client = create_llm_client(
        provider=config.llm_provider,
        model=config.config.get("deep_think_llm", "o3-mini"),
        base_url=config.config.get("backend_url"),
        **llm_kwargs,
    )
    return lambda: create_fundamentals_analyst(client.get_llm())


# ---------------------------------------------------------------------------
# Graph dependency
# ---------------------------------------------------------------------------


def get_trading_graph(
    config: ApiConfig = Depends(get_config),
) -> TradingAgentsGraph:
    """Create a TradingAgentsGraph instance.

    Note: This creates a new graph on each call. For production, consider
    caching or using a singleton pattern.
    """
    return TradingAgentsGraph(
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=config.debug,
        config=config.config,
    )