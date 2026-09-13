"""Tests for TradingAgents API dependency-injection providers.

Covers:
- get_llm_factory must honor ApiConfig overrides (regression: it used to build
  a bare LLMProviderFactory() that silently ignored config.config, so single-
  analyst endpoints never saw llm_provider/deep_think_llm/etc. overrides).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.api.config import ApiConfig
from tradingagents.api.dependencies import get_llm_factory


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_llm_factory_honors_config_overrides():
    with patch(
        "tradingagents.api.infrastructure.llm_provider.create_llm_client"
    ) as mock_create:
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        config = ApiConfig(
            overrides={"llm_provider": "anthropic", "deep_think_llm": "claude-x"}
        )
        factory = await get_llm_factory(config)
        factory()

        assert mock_create.call_args.kwargs["provider"] == "anthropic"
        assert mock_create.call_args.kwargs["model"] == "claude-x"
