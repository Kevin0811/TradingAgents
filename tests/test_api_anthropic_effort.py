"""Tests for tradingagents/api/infrastructure/anthropic_effort.py.

The core sends ``effort`` only to Opus/Sonnet/Fable; ``install()`` extends its
``_supports_effort`` predicate so Claude Haiku 5.5+ gets it too, while Haiku
4.5 and older stay excluded (they return 400 for effort). Nothing is sent to
Anthropic: the request payload is built offline.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from tradingagents.api.infrastructure import anthropic_effort, secret_keys
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph
from tradingagents.llm_clients import anthropic_client, factory

pytestmark = [
    pytest.mark.unit,
    # claude-haiku-5-5 is not in the core's model catalog; get_llm() warns.
    pytest.mark.filterwarnings("ignore:Model .* is not in the known model list"),
]

CORE_PREDICATE = anthropic_effort.original(anthropic_client._supports_effort)


@pytest.fixture(autouse=True)
def core_predicate(monkeypatch):
    """Start every test from the core's own predicate (an earlier create_app()
    may have installed the wrapper); the previous binding is restored after."""
    monkeypatch.setattr(anthropic_client, "_supports_effort", CORE_PREDICATE)


@pytest.fixture
def secrets(tmp_path, monkeypatch):
    """A temp secrets directory holding a dummy Anthropic key; env keys unset."""
    directory = tmp_path / "secrets"
    directory.mkdir()
    (directory / "ANTHROPIC_API_KEY").write_text("test-key\n")
    monkeypatch.setenv(secret_keys.DIR_VAR, str(directory))
    for name in secret_keys.ON_DEMAND_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    return directory


def tier_config(**overrides):
    return {**DEFAULT_CONFIG, "llm_provider": "anthropic", "backend_url": None,
            "deep_think_provider": None, "quick_think_provider": None,
            "deep_think_backend_url": None, "quick_think_backend_url": None,
            "deep_think_llm": "claude-opus-5-5", "quick_think_llm": "claude-haiku-5-5",
            "temperature": None, "anthropic_effort": None, **overrides}


def sent_effort(llm):
    """The ``effort`` the ChatAnthropic would send (``output_config.effort``), or None.

    langchain-anthropic keeps it in the field ``reasoning_effort`` (alias
    ``effort``) and writes it to the request as ``output_config.effort``; the
    payload is built offline, nothing is sent.
    """
    payload = llm._get_request_payload([HumanMessage(content="ping")])
    return (payload.get("output_config") or {}).get("effort")


def build(model, **kwargs):
    client = factory.create_llm_client("anthropic", model, api_key="test", **kwargs)
    return client.get_llm()


# ---- the predicate -------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["claude-haiku-5-5", "claude-haiku-5-5-20261001", "CLAUDE-HAIKU-5-5", "claude-haiku-5-6",
     "claude-haiku-6", "claude-haiku-6-0", "claude-haiku-6-1-20270101"],
)
def test_api_anthropic_effort_haiku_5_5_and_later_supported(model):
    assert anthropic_effort.haiku_supports_effort(model)
    assert anthropic_effort._wrap(CORE_PREDICATE)(model)


@pytest.mark.parametrize(
    "model",
    ["claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-haiku-5", "claude-haiku-5-0",
     "claude-haiku-5-4", "claude-haiku-5-20261001", "claude-3-5-haiku-20241022",
     "claude-3-haiku-20240307", "claude-haiku-4-7-preview", "claude-haiku-5-5-preview",
     "claude-haiku-latest", "claude-haiku-5-5-2026", "", "haiku-5-5",
     # A trailing newline is not supported (fullmatch, not match with "$").
     "claude-haiku-5-5\n", "claude-haiku-5-5-20261001\n"],
)
def test_api_anthropic_effort_older_or_odd_haiku_not_supported(model):
    assert not anthropic_effort.haiku_supports_effort(model)
    assert not anthropic_effort._wrap(CORE_PREDICATE)(model)


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-5", "claude-opus-4-1", "claude-opus-5-5", "claude-sonnet-4-5",
     "claude-sonnet-4-6", "claude-sonnet-5", "claude-sonnet-5-5", "claude-fable-5",
     "claude-fable-5-1", "claude-mythos-preview", "claude-mythos-5", "claude-experimental-x",
     "claude-opus-4-5-20251101", "gpt-5"],
)
def test_api_anthropic_effort_other_families_unchanged(model):
    assert anthropic_effort._wrap(CORE_PREDICATE)(model) is CORE_PREDICATE(model)


# ---- guard: a real client sends effort to Haiku 5.5 ----------------------------------


def test_api_anthropic_effort_guard_real_client_sends_effort_to_haiku_5_5():
    """With install(), a real AnthropicClient for claude-haiku-5-5 sends effort "low".

    Fails if install() is removed or made a no-op, if the core renames
    ``_supports_effort`` (install() raises), or if ``get_llm`` stops consulting
    that module-level name.
    """
    anthropic_effort.install()
    llm = build("claude-haiku-5-5", effort="low")
    assert llm.reasoning_effort == "low"  # langchain-anthropic field; alias "effort"
    assert sent_effort(llm) == "low"


def test_api_anthropic_effort_without_install_core_drops_it():
    # Documents the gap the install closes; if the core starts sending effort to
    # Haiku 5.5 itself, this fails and the API module can be retired.
    assert sent_effort(build("claude-haiku-5-5", effort="low")) is None


def test_api_anthropic_effort_dated_haiku_5_5_gets_effort():
    anthropic_effort.install()
    assert sent_effort(build("claude-haiku-5-5-20261001", effort="max")) == "max"


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-haiku-4-5-20251001"])
def test_api_anthropic_effort_haiku_4_5_still_gets_none(model):
    anthropic_effort.install()
    llm = build(model, effort="low")
    assert llm.reasoning_effort is None
    assert sent_effort(llm) is None


def test_api_anthropic_effort_unset_sends_nothing():
    anthropic_effort.install()
    assert sent_effort(build("claude-haiku-5-5")) is None


def test_api_anthropic_effort_opus_still_gets_effort():
    anthropic_effort.install()
    assert sent_effort(build("claude-opus-5-5", effort="high")) == "high"


# ---- config level --------------------------------------------------------------


def test_api_anthropic_effort_config_quick_tier_gets_effort():
    anthropic_effort.install()
    config = tier_config(anthropic_effort="low")
    llm = factory.create_tier_client(config, "quick", api_key="test").get_llm()
    assert llm.model == "claude-haiku-5-5"
    assert sent_effort(llm) == "low"


def test_api_anthropic_effort_config_unset_effort_sends_nothing():
    anthropic_effort.install()
    llm = factory.create_tier_client(tier_config(), "quick", api_key="test").get_llm()
    assert sent_effort(llm) is None


def test_api_anthropic_effort_api_llm_provider_gets_effort(secrets):
    from tradingagents.api.infrastructure.llm_provider import LLMProviderFactory

    anthropic_effort.install()
    llm = LLMProviderFactory(config=tier_config(anthropic_effort="low")).create_quick_thinking_llm()
    assert sent_effort(llm) == "low"


def test_api_anthropic_effort_real_graph_quick_tier_gets_effort(secrets, tmp_path, monkeypatch):
    """A real TradingAgentsGraph's Haiku 5.5 tier sends the configured effort."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    monkeypatch.setattr(trading_graph, "create_tier_client", factory.create_tier_client)
    secret_keys.install()
    anthropic_effort.install()
    config = tier_config(
        anthropic_effort="low",
        results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "memory" / "log.md"),
    )
    graph = TradingAgentsGraph(selected_analysts=["market"], config=config)
    assert sent_effort(graph.quick_thinking_llm) == "low"
    assert sent_effort(graph.deep_thinking_llm) == "low"


# ---- install -------------------------------------------------------------------


def test_api_anthropic_effort_install_is_idempotent():
    anthropic_effort.install()
    first = anthropic_client._supports_effort
    assert first is not CORE_PREDICATE
    assert anthropic_effort.is_wrapped(first)
    assert anthropic_effort.original(first) is CORE_PREDICATE
    anthropic_effort.install()
    assert anthropic_client._supports_effort is first


class _AutoAttributes:
    """Makes up any attribute asked for, as a proxy or a lenient fake might."""

    def __getattr__(self, name):
        return object()

    def __call__(self, model):
        return False


def test_api_anthropic_effort_install_wraps_an_object_with_made_up_attributes(monkeypatch):
    fake = _AutoAttributes()
    monkeypatch.setattr(anthropic_client, "_supports_effort", fake)
    assert not anthropic_effort.is_wrapped(fake)
    anthropic_effort.install()
    wrapped = anthropic_client._supports_effort
    assert wrapped is not fake and anthropic_effort.is_wrapped(wrapped)
    assert wrapped("claude-haiku-5-5") and not wrapped("claude-opus-5-5")


def test_api_anthropic_effort_install_fails_loudly_without_core_predicate(monkeypatch):
    monkeypatch.delattr(anthropic_client, "_supports_effort")
    with pytest.raises(RuntimeError, match="_supports_effort"):
        anthropic_effort.install()


def test_api_anthropic_effort_create_app_installs(monkeypatch):
    from tradingagents.api.app import create_app

    monkeypatch.setattr(trading_graph, "create_tier_client", factory.create_tier_client)
    create_app()
    assert anthropic_effort.is_wrapped(anthropic_client._supports_effort)


def test_api_anthropic_effort_create_app_fails_without_core_predicate(monkeypatch):
    from tradingagents.api.app import create_app

    monkeypatch.setattr(trading_graph, "create_tier_client", factory.create_tier_client)
    monkeypatch.delattr(anthropic_client, "_supports_effort")
    with pytest.raises(RuntimeError, match="_supports_effort"):
        create_app()
