"""Tests for running the API on Claude 5.5 models (issue #11).

Covers:
- LLMProviderFactory builds each tier through the core's create_tier_client
  (per-tier providers, max_tokens, llm_max_retries)
- check_llm_config refuses a temperature the tier's Anthropic model rejects,
  offline (no API key, nothing sent), and an unknown provider
- the per-request provider / anthropic_effort overrides, GET /config, and the
  400 that /analyze and /analyze/tasks return for a configuration that would
  fail every LLM call
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tradingagents.api.app import create_app
from tradingagents.api.core.exceptions import InvalidRequestError
from tradingagents.api.domain.services.config_overrides import (
    check_run_config,
    resolve_overrides,
)
from tradingagents.api.infrastructure.llm_provider import (
    LLMProviderFactory,
    check_llm_config,
)
from tradingagents.api.schemas.request import AnalyzeRequest

ANTHROPIC = {
    "llm_provider": "anthropic",
    "deep_think_llm": "claude-opus-5-5",
    "quick_think_llm": "claude-haiku-5-5",
}


@pytest.fixture(autouse=True)
def _no_anthropic_key(monkeypatch):
    """The checks build clients offline; they must not need a real key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def mock_create_llm_client():
    with patch("tradingagents.llm_clients.factory.create_llm_client") as mock:
        mock.return_value.get_llm.return_value = MagicMock()
        yield mock


@pytest.mark.unit
class TestFactoryUsesTheCoreTiers:
    def test_each_tier_runs_on_its_own_provider(self, mock_create_llm_client):
        config = {
            "llm_provider": "ollama",
            "quick_think_llm": "qwen3",
            "deep_think_provider": "anthropic",
            "deep_think_llm": "claude-opus-5-5",
        }
        factory = LLMProviderFactory(config=config)

        factory.create_quick_thinking_llm()
        assert mock_create_llm_client.call_args.kwargs["provider"] == "ollama"
        assert mock_create_llm_client.call_args.kwargs["model"] == "qwen3"

        factory.create_deep_thinking_llm()
        assert mock_create_llm_client.call_args.kwargs["provider"] == "anthropic"
        assert mock_create_llm_client.call_args.kwargs["model"] == "claude-opus-5-5"

    def test_create_llm_uses_the_deep_tier_provider(self, mock_create_llm_client):
        config = {"llm_provider": "ollama", "deep_think_provider": "anthropic"}
        LLMProviderFactory(config=config).create_llm(model="claude-sonnet-5-5")
        assert mock_create_llm_client.call_args.kwargs["provider"] == "anthropic"
        assert mock_create_llm_client.call_args.kwargs["model"] == "claude-sonnet-5-5"

    def test_effort_max_tokens_and_retries_reach_the_client(self, mock_create_llm_client):
        config = {**ANTHROPIC, "anthropic_effort": "high", "max_tokens": 4000, "llm_max_retries": 0}
        LLMProviderFactory(config=config).create_deep_thinking_llm()
        kwargs = mock_create_llm_client.call_args.kwargs
        assert kwargs["effort"] == "high"
        assert kwargs["max_tokens"] == 4000
        assert kwargs["max_retries"] == 0


@pytest.mark.unit
class TestCheckLlmConfig:
    @pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-fable-5-1"])
    def test_temperature_a_claude_model_rejects_is_refused(self, model):
        with pytest.raises(ValueError, match=f"{model}.*TRADINGAGENTS_TEMPERATURE"):
            check_llm_config({**ANTHROPIC, "deep_think_llm": model, "temperature": 0.2})

    def test_temperature_an_older_claude_model_accepts_passes(self):
        check_llm_config(
            {**ANTHROPIC, "deep_think_llm": "claude-haiku-4-5",
             "quick_think_llm": "claude-haiku-4-5", "temperature": 0.2}
        )

    def test_claude_5_5_without_temperature_passes(self):
        check_llm_config(ANTHROPIC)

    def test_a_tier_on_anthropic_is_checked_when_the_main_provider_is_not(self):
        config = {
            "llm_provider": "ollama",
            "quick_think_llm": "qwen3",
            "deep_think_provider": "anthropic",
            "deep_think_llm": "claude-sonnet-5-5",
            "temperature": 0.3,
        }
        with pytest.raises(ValueError, match="deep-thinking model 'claude-sonnet-5-5'"):
            check_llm_config(config)

    def test_temperature_on_a_non_anthropic_provider_is_left_alone(self):
        check_llm_config({"llm_provider": "ollama", "temperature": 0.3})

    def test_unknown_provider_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            check_llm_config({"llm_provider": "nope"})


@pytest.mark.unit
class TestRunOverrides:
    def test_provider_and_effort_overrides_pass_through(self):
        request = AnalyzeRequest(
            ticker="AAPL",
            trade_date="2026-06-01",
            deep_think_provider="anthropic",
            quick_think_provider="ollama",
            anthropic_effort="xhigh",
        )
        assert resolve_overrides(request) == {
            "deep_think_provider": "anthropic",
            "quick_think_provider": "ollama",
            "anthropic_effort": "xhigh",
        }

    def test_unknown_effort_is_a_validation_error(self):
        with pytest.raises(ValueError):
            AnalyzeRequest(ticker="AAPL", trade_date="2026-06-01", anthropic_effort="extreme")

    def test_an_override_that_breaks_the_run_raises_invalid_request(self):
        server = {**ANTHROPIC, "temperature": 0.2, "deep_think_llm": "claude-haiku-4-5",
                  "quick_think_llm": "claude-haiku-4-5"}
        check_run_config(server, {})
        with pytest.raises(InvalidRequestError) as exc_info:
            check_run_config(server, {"deep_think_llm": "claude-sonnet-5-5"})
        assert "claude-sonnet-5-5" in exc_info.value.detail


@pytest.mark.unit
class TestRoutes:
    SERVER = {**ANTHROPIC, "temperature": 0.2, "deep_think_llm": "claude-haiku-4-5",
              "quick_think_llm": "claude-haiku-4-5"}
    BODY = {"ticker": "AAPL", "trade_date": "2026-06-01", "deep_think_llm": "claude-sonnet-5-5"}

    def test_sync_analyze_refuses_the_run_before_starting_it(self):
        with patch(
            "tradingagents.api.domain.services.analysis_service.AnalysisService.run_analysis"
        ) as run:
            client = TestClient(create_app(self.SERVER))
            resp = client.post("/api/v1/analyze", json=self.BODY)
        assert resp.status_code == 400
        assert "claude-sonnet-5-5" in resp.json()["detail"]
        run.assert_not_called()

    def test_task_create_refuses_the_task_before_queuing_it(self):
        app = create_app(self.SERVER)
        client = TestClient(app)
        resp = client.post("/api/v1/analyze/tasks", json=self.BODY)
        assert resp.status_code == 400
        assert app.state.task_manager.active_task_count == 0

    def test_get_config_shows_the_tier_providers_and_effort(self):
        client = TestClient(
            create_app({**ANTHROPIC, "deep_think_provider": "anthropic",
                        "quick_think_provider": "ollama", "anthropic_effort": "high"})
        )
        body = client.get("/api/v1/config").json()
        assert body["deep_think_provider"] == "anthropic"
        assert body["quick_think_provider"] == "ollama"
        assert body["anthropic_effort"] == "high"

    def test_startup_logs_a_configuration_that_fails_every_run(self, caplog):
        create_app({**ANTHROPIC, "deep_think_llm": "claude-sonnet-5-5", "temperature": 0.2})
        assert "LLM configuration will fail every analysis" in caplog.text
