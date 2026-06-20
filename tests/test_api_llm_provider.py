"""Tests for the TradingAgents API LLMProviderFactory.

Covers:
- Factory reads config and creates LLM clients
- Provider-specific kwargs (thinking_level, reasoning_effort, effort, temperature)
- Model override via create_llm(model=...)
- Default config fallback when no config is passed
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.api.infrastructure.llm_provider import LLMProviderFactory


@pytest.fixture(autouse=True)
def _mock_create_llm_client():
    """Mock the underlying create_llm_client to avoid real API calls."""
    with patch("tradingagents.api.infrastructure.llm_provider.create_llm_client") as mock:
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock.return_value = mock_client
        yield mock


@pytest.mark.unit
class TestLLMProviderFactoryConfig:
    """Factory must read config and pass correct kwargs."""

    def test_uses_default_config_when_none_passed(self, _mock_create_llm_client):
        """When config is None, factory uses DEFAULT_CONFIG."""
        factory = LLMProviderFactory()
        factory.create_llm()
        # create_llm_client should have been called with defaults
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert "provider" in call_kwargs
        assert "model" in call_kwargs

    def test_uses_custom_config_when_passed(self, _mock_create_llm_client):
        config = {
            "llm_provider": "google",
            "deep_think_llm": "gemini-2.5-flash",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["provider"] == "google"
        assert call_kwargs["model"] == "gemini-2.5-flash"

    def test_model_override_uses_passed_model(self, _mock_create_llm_client):
        config = {"llm_provider": "openai", "deep_think_llm": "gpt-5.5"}
        factory = LLMProviderFactory(config=config)
        factory.create_llm(model="custom-model")
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["model"] == "custom-model"


@pytest.mark.unit
class TestLLMProviderFactoryProviderKwargs:
    """Provider-specific kwargs must be passed correctly."""

    def test_google_thinking_level_passed(self, _mock_create_llm_client):
        config = {
            "llm_provider": "google",
            "google_thinking_level": "high",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["thinking_level"] == "high"

    def test_openai_reasoning_effort_passed(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "openai_reasoning_effort": "high",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    def test_anthropic_effort_passed(self, _mock_create_llm_client):
        config = {
            "llm_provider": "anthropic",
            "anthropic_effort": "medium",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["effort"] == "medium"

    def test_temperature_passed_as_float(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "temperature": "0.5",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_temperature_empty_string_omitted(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "temperature": "",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert "temperature" not in call_kwargs

    def test_temperature_none_omitted(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "temperature": None,
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert "temperature" not in call_kwargs

    def test_backend_url_passed(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "backend_url": "http://proxy/v1",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["base_url"] == "http://proxy/v1"


@pytest.mark.unit
class TestLLMProviderFactoryCreateMethods:
    """Different create_* methods must use the correct model from config."""

    def test_create_deep_thinking_llm_uses_deep_think_llm(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "deep_think_llm": "o3-mini",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_deep_thinking_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["model"] == "o3-mini"

    def test_create_quick_thinking_llm_uses_quick_think_llm(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "quick_think_llm": "gpt-4o-mini",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_quick_thinking_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_create_llm_defaults_to_deep_think_llm(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "deep_think_llm": "o3-mini",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm()
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["model"] == "o3-mini"

    def test_create_llm_with_override(self, _mock_create_llm_client):
        config = {
            "llm_provider": "openai",
            "deep_think_llm": "o3-mini",
        }
        factory = LLMProviderFactory(config=config)
        factory.create_llm(model="gpt-5.5")
        call_kwargs = _mock_create_llm_client.call_args[1]
        assert call_kwargs["model"] == "gpt-5.5"