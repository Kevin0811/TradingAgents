"""Tests for the per-tier token usage of an analysis run (issue #20).

The core hands one callback list to both tiers' models, so the API tags each
tier's model with its tier after building the graph, and ``TokenUsageRecorder``
reads the tag from the metadata LangChain passes to its callbacks. The models
here are fakes reporting usage the way LangChain providers do; no test reaches
a provider.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from tradingagents.agents.structured import bind_structured, invoke_structured_or_freetext
from tradingagents.api.app import create_app
from tradingagents.api.core.exceptions import AnalysisCancelled, AnalysisError
from tradingagents.api.domain.services.analysis_service import AnalysisService, _tag_tiers
from tradingagents.api.infrastructure.llm_callbacks import (
    TIER_METADATA_KEY,
    TokenUsageRecorder,
    usage_counts,
)
from tradingagents.api.schemas.token_usage import TokenUsage
from tradingagents.default_config import DEFAULT_CONFIG

GRAPH = "tradingagents.api.domain.services.analysis_service.TradingAgentsGraph"
ANALYZE = "/api/v1/analyze"
TASKS = "/api/v1/analyze/tasks"

# Usage as langchain-anthropic reports it: input_tokens counts the cache reads
# and writes too (the API's own input_tokens, 100 here, leaves them out).
ANTHROPIC_USAGE = {
    "input_tokens": 1150,
    "output_tokens": 40,
    "total_tokens": 1190,
    "input_token_details": {"cache_read": 1000, "cache_creation": 50},
}
# Usage as an OpenAI-compatible provider such as Ollama reports it: no details.
OLLAMA_USAGE = {"input_tokens": 300, "output_tokens": 20, "total_tokens": 320}

TIERS = {"quick": ("ollama", "qwen3:4b"), "deep": ("anthropic", "claude-opus-5-5")}


class Plan(BaseModel):
    action: str = "Buy"


class UsageModel(BaseChatModel):
    """A chat model answering with fixed usage, like a provider does.

    ``served`` is the model the response names (none when None); ``usage`` is
    its ``usage_metadata`` (none when None). It supports ``bind_tools`` and
    ``with_structured_output`` the way providers do: as bindings of itself.
    """

    served: str | None = None
    usage: dict | None = None
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "usage-fake"

    def bind_tools(self, tools, **kwargs):
        names = [t.__name__ if isinstance(t, type) else t.name for t in tools]
        return self.bind(tools=names, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return self.bind_tools([schema]) | RunnableLambda(
            lambda message: schema(**message.tool_calls[0]["args"])
        )

    def _generate(self, messages, stop=None, run_manager=None, tools=None, **kwargs):
        if self.fail:
            raise RuntimeError("provider unavailable")
        calls = [{"name": tools[0], "args": {}, "id": "call-1"}] if tools else []
        message = AIMessage(
            content="" if calls else "text",
            tool_calls=calls,
            usage_metadata=self.usage,
            response_metadata={"model_name": self.served} if self.served else {},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tagged(tier: str, recorder: TokenUsageRecorder, **fields) -> UsageModel:
    model = UsageModel(callbacks=[recorder], **fields)
    model.metadata = {TIER_METADATA_KEY: tier}
    return model


def _entries(summary: dict) -> list[tuple]:
    return [(t["tier"], t["provider"], t["model"], t["calls"]) for t in summary["tiers"]]


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTokenUsageRecorder:
    def test_anthropic_and_ollama_usage_add_up_per_tier_and_model(self):
        recorder = TokenUsageRecorder(TIERS)
        deep = _tagged("deep", recorder, served="claude-opus-5-5-20261001", usage=ANTHROPIC_USAGE)
        quick = _tagged("quick", recorder, served="qwen3:4b", usage=OLLAMA_USAGE)
        deep.invoke("a")
        deep.invoke("b")
        quick.invoke("c")

        summary = recorder.summary(partial=False)
        assert summary == {
            "tiers": [
                {
                    "tier": "quick", "provider": "ollama", "model": "qwen3:4b", "calls": 1,
                    "input_tokens": 300, "output_tokens": 20,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                    "calls_without_usage": 0,
                },
                {
                    "tier": "deep", "provider": "anthropic",
                    "model": "claude-opus-5-5-20261001", "calls": 2,
                    "input_tokens": 2300, "output_tokens": 80,
                    "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 100,
                    "calls_without_usage": 0,
                },
            ],
            "totals": {
                "calls": 3, "input_tokens": 2600, "output_tokens": 100,
                "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 100,
            },
            "partial": False,
        }
        TokenUsage.model_validate(summary)  # the shape the API serves

    def test_two_tiers_on_one_model_are_two_entries(self):
        recorder = TokenUsageRecorder({"quick": ("ollama", "qwen3"), "deep": ("ollama", "qwen3")})
        quick = _tagged("quick", recorder, served="qwen3", usage=OLLAMA_USAGE)
        deep = _tagged("deep", recorder, served="qwen3", usage=OLLAMA_USAGE)
        quick.invoke("a")
        quick.invoke("b")
        deep.invoke("c")

        summary = recorder.summary(partial=False)
        assert _entries(summary) == [("quick", "ollama", "qwen3", 2), ("deep", "ollama", "qwen3", 1)]
        assert [t["input_tokens"] for t in summary["tiers"]] == [600, 300]

    def test_one_tier_answered_by_two_models_is_two_entries_in_model_order(self):
        recorder = TokenUsageRecorder(TIERS)
        _tagged("deep", recorder, served="model-b", usage=OLLAMA_USAGE).invoke("a")
        _tagged("deep", recorder, served="model-a", usage=OLLAMA_USAGE).invoke("b")
        assert [t["model"] for t in recorder.summary(partial=False)["tiers"]] == [
            "model-a", "model-b",
        ]

    def test_a_response_naming_no_model_counts_under_the_configured_model(self):
        recorder = TokenUsageRecorder(TIERS)
        _tagged("deep", recorder, usage=OLLAMA_USAGE).invoke("a")
        assert _entries(recorder.summary(partial=False)) == [
            ("deep", "anthropic", "claude-opus-5-5", 1),
        ]

    def test_calls_without_usage_still_count_as_calls(self):
        recorder = TokenUsageRecorder(TIERS)
        quick = _tagged("quick", recorder, served="qwen3:4b", usage=OLLAMA_USAGE)
        silent = _tagged("quick", recorder, served="qwen3:4b")
        failing = _tagged("quick", recorder, fail=True)
        quick.invoke("a")
        silent.invoke("b")
        with pytest.raises(RuntimeError):
            failing.invoke("c")

        [entry] = recorder.summary(partial=True)["tiers"]
        assert entry["calls"] == 3
        assert entry["calls_without_usage"] == 2
        assert entry["input_tokens"] == 300

    def test_a_call_without_usage_adds_to_calls_only(self):
        """No usage data: calls and calls_without_usage go up, no token field
        does, so the token counts are a lower bound."""
        recorder = TokenUsageRecorder(TIERS)
        _tagged("deep", recorder, served="claude-opus-5-5").invoke("a")
        summary = recorder.summary(partial=False)
        assert summary["tiers"] == [{
            "tier": "deep", "provider": "anthropic", "model": "claude-opus-5-5", "calls": 1,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "calls_without_usage": 1,
        }]
        assert summary["totals"] == {
            "calls": 1, "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }

    def test_input_tokens_include_the_cache_tokens(self):
        """The rule fin-insight prices by: input_tokens INCLUDES both cache fields
        (uncached input = input_tokens - cache_read - cache_creation), so adding
        the cache fields to it would double count them."""
        uncached, read, written = 100, 1000, 50
        recorder = TokenUsageRecorder(TIERS)
        _tagged("deep", recorder, served="claude-opus-5-5", usage=ANTHROPIC_USAGE).invoke("a")
        [entry] = recorder.summary(partial=False)["tiers"]
        assert entry["cache_read_input_tokens"] == read
        assert entry["cache_creation_input_tokens"] == written
        assert entry["input_tokens"] == uncached + read + written
        assert (entry["input_tokens"] - entry["cache_read_input_tokens"]
                - entry["cache_creation_input_tokens"]) == uncached

    def test_a_call_without_a_tier_is_not_counted(self):
        recorder = TokenUsageRecorder(TIERS)
        UsageModel(callbacks=[recorder], usage=OLLAMA_USAGE).invoke("a")
        assert recorder.summary(partial=False)["tiers"] == []
        assert recorder.summary(partial=False)["totals"]["calls"] == 0

    def test_concurrent_calls_add_up_exactly(self):
        """The core's analysts call the models from several threads at once.

        On CPython with the GIL this passes even without the recorder's lock
        (its updates do not yield mid-way); it guards the sums, not the lock.
        """
        recorder = TokenUsageRecorder(TIERS)
        threads, calls = 8, 1500
        message = AIMessage(content="x", usage_metadata=ANTHROPIC_USAGE,
                            response_metadata={"model_name": "m"})
        result = LLMResult(generations=[[ChatGeneration(message=message)]])
        start = threading.Barrier(threads)

        def work(tier):
            start.wait()
            for _ in range(calls):
                run_id = uuid.uuid4()
                recorder.on_chat_model_start({}, [[]], run_id=run_id,
                                             metadata={TIER_METADATA_KEY: tier})
                recorder.on_llm_end(result, run_id=run_id)

        workers = [threading.Thread(target=work, args=(("quick", "deep")[i % 2],))
                   for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        summary = recorder.summary(partial=False)
        assert [t["calls"] for t in summary["tiers"]] == [threads // 2 * calls] * 2
        assert summary["totals"]["input_tokens"] == threads * calls * 1150
        assert summary["totals"]["cache_read_input_tokens"] == threads * calls * 1000


@pytest.mark.unit
class TestUsageCounts:
    def test_missing_details_count_as_zero(self):
        assert usage_counts({"input_tokens": 5, "output_tokens": 2}) == {
            "input_tokens": 5, "output_tokens": 2,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }

    def test_cache_writes_split_by_ttl_are_added_up(self):
        usage = {
            "input_tokens": 80, "output_tokens": 1,
            "input_token_details": {
                "cache_read": 0, "cache_creation": 0,
                "ephemeral_5m_input_tokens": 30, "ephemeral_1h_input_tokens": 20,
            },
        }
        assert usage_counts(usage)["cache_creation_input_tokens"] == 50

    @pytest.mark.parametrize("ttl_split", [False, True])
    def test_langchain_anthropics_own_usage_counts_cached_input_in_input_tokens(self, ttl_split):
        """Pin the rule against the installed langchain-anthropic: its
        ``input_tokens`` is Anthropic's plus the cache reads and writes."""
        chat_models = pytest.importorskip("langchain_anthropic.chat_models")
        create = getattr(chat_models, "_create_usage_metadata", None)
        if create is None:
            pytest.skip("langchain-anthropic no longer has _create_usage_metadata")
        from anthropic.types import Usage

        raw = {
            "input_tokens": 100, "output_tokens": 40,
            "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 50,
        }
        if ttl_split:
            raw["cache_creation"] = {"ephemeral_5m_input_tokens": 30, "ephemeral_1h_input_tokens": 20}
        counts = usage_counts(create(Usage.model_validate(raw)))
        assert counts == {
            "input_tokens": 1150, "output_tokens": 40,
            "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 50,
        }


# ---------------------------------------------------------------------------
# Tier tagging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTierTagging:
    def test_tags_reach_bindings_made_before_them(self):
        """The core binds tools and structured output while it builds the graph,
        before the API tags the models; the tags still reach those calls."""
        recorder = TokenUsageRecorder(TIERS)

        class Graph:
            quick_thinking_llm = UsageModel(callbacks=[recorder], served="q", usage=OLLAMA_USAGE)
            deep_thinking_llm = UsageModel(callbacks=[recorder], served="d", usage=OLLAMA_USAGE,
                                           metadata={"kept": True})

        structured = bind_structured(Graph.deep_thinking_llm, Plan, "Portfolio Manager")
        with_tools = Graph.quick_thinking_llm.bind_tools([Plan])
        _tag_tiers(Graph)

        metadata = Graph.deep_thinking_llm.metadata
        assert metadata["kept"] is True and metadata[TIER_METADATA_KEY] == "deep"
        text = invoke_structured_or_freetext(
            structured, Graph.deep_thinking_llm, "prompt", lambda plan: plan.action, "PM"
        )
        assert text == "Buy"  # the structured call answered
        assert with_tools.invoke("prompt").tool_calls
        assert _entries(recorder.summary(partial=False)) == [
            ("quick", "ollama", "q", 1), ("deep", "anthropic", "d", 1),
        ]

    def test_the_core_graph_hands_its_tagged_models_to_its_agents(self, tmp_path):
        """A real TradingAgentsGraph (its models faked): the API tags the very
        objects the core's agents and Reflector call."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        models = {
            "quick": UsageModel(served="qwen3", usage=OLLAMA_USAGE),
            "deep": UsageModel(served="qwen3", usage=ANTHROPIC_USAGE),
        }

        class Client:
            def __init__(self, tier, callbacks):
                self.model = models[tier].model_copy(update={"callbacks": callbacks})

            def get_llm(self):
                return self.model

        def create_tier_client(config, tier, callbacks=None, **extra):
            return Client(tier, callbacks)

        def propagate(graph, ticker, trade_date, asset_type="stock"):
            assert graph.reflector.quick_thinking_llm is graph.quick_thinking_llm
            graph.reflector.reflect_on_final_decision("Buy", 0.01, 0.002)
            graph.quick_thinking_llm.bind_tools([Plan]).invoke("analyst turn")
            structured = bind_structured(graph.deep_thinking_llm, Plan, "Portfolio Manager")
            assert structured.invoke("decide").action == "Buy"
            return {"final_trade_decision": "Buy"}, "Buy"

        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(
            llm_provider="ollama", quick_think_llm="qwen3", deep_think_llm="qwen3",
            results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
            memory_log_path=str(tmp_path / "memory.md"), checkpoint_enabled=False,
        )
        with (
            patch("tradingagents.graph.trading_graph.create_tier_client", create_tier_client),
            patch.object(TradingAgentsGraph, "propagate", propagate),
        ):
            result = AnalysisService(config=config).run_analysis("AAPL", "2026-06-01")

        usage = result.token_usage
        assert usage["partial"] is False
        assert _entries(usage) == [("quick", "ollama", "qwen3", 2), ("deep", "ollama", "qwen3", 1)]
        assert usage["totals"]["cache_read_input_tokens"] == 1000
        assert usage["totals"]["input_tokens"] == 2 * 300 + 1150


# ---------------------------------------------------------------------------
# Through the service and the API
# ---------------------------------------------------------------------------


def _graph(script: list, hold: threading.Event | None = None,
           proceed: threading.Event | None = None):
    """A TradingAgentsGraph stand-in whose run calls its two tier models.

    ``script`` lists the run's steps: "quick"/"deep" call that tier's model,
    "hold" waits for ``proceed`` (signalling ``hold``), "raise" fails the run.
    """

    class Graph:
        def __init__(self, *args, config=None, callbacks=None, **kwargs):
            self.quick_thinking_llm = UsageModel(
                callbacks=callbacks, served=config["quick_think_llm"], usage=OLLAMA_USAGE
            )
            self.deep_thinking_llm = UsageModel(
                callbacks=callbacks, served=config["deep_think_llm"], usage=ANTHROPIC_USAGE
            )

        def propagate(self, ticker, trade_date, asset_type="stock"):
            for step in script:
                if step == "hold":
                    hold.set()
                    assert proceed.wait(5)
                elif step == "raise":
                    raise RuntimeError("provider unavailable")
                else:
                    getattr(self, f"{step}_thinking_llm").invoke(step)
            return {"final_trade_decision": "Buy"}, "Buy"

    return Graph


SERVER = {
    "llm_provider": "ollama",
    "quick_think_llm": "qwen3:4b",
    "deep_think_llm": "claude-opus-5-5",
    "deep_think_provider": "anthropic",
}


def _service_config():
    return {**copy.deepcopy(DEFAULT_CONFIG), **SERVER}


@pytest.mark.unit
class TestAnalysisServiceUsage:
    def test_a_completed_run_reports_its_usage(self):
        with patch(GRAPH, _graph(["quick", "quick", "deep"])):
            result = AnalysisService(config=_service_config()).run_analysis("AAPL", "2026-06-01")
        usage = result.token_usage
        assert usage["partial"] is False
        assert _entries(usage) == [
            ("quick", "ollama", "qwen3:4b", 2), ("deep", "anthropic", "claude-opus-5-5", 1),
        ]
        assert usage["totals"] == {
            "calls": 3, "input_tokens": 1750, "output_tokens": 80,
            "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 50,
        }
        assert "token_usage" not in result.to_dict()  # tasks report it beside result

    def test_a_failed_run_carries_its_usage_so_far(self):
        with (
            patch(GRAPH, _graph(["quick", "deep", "raise", "deep"])),
            pytest.raises(AnalysisError) as exc_info,
        ):
            AnalysisService(config=_service_config()).run_analysis("AAPL", "2026-06-01")
        usage = exc_info.value.token_usage
        assert usage["partial"] is True
        assert usage["totals"]["calls"] == 2

    def test_a_cancelled_run_carries_its_usage_so_far(self):
        hold, proceed, cancel = threading.Event(), threading.Event(), threading.Event()
        out = {}

        def run():
            try:
                AnalysisService(config=_service_config()).run_analysis(
                    "AAPL", "2026-06-01", cancel_event=cancel
                )
            except AnalysisCancelled as e:
                out["usage"] = e.token_usage

        with patch(GRAPH, _graph(["quick", "hold", "deep"], hold, proceed)):
            thread = threading.Thread(target=run)
            thread.start()
            assert hold.wait(5)
            cancel.set()
            proceed.set()
            thread.join(5)
        assert out["usage"]["partial"] is True
        # The deep call was refused at its start: not counted.
        assert _entries(out["usage"]) == [("quick", "ollama", "qwen3:4b", 1)]


@pytest.fixture
def app():
    app = create_app(overrides=SERVER)
    app.state.ollama_cache_trimmer = None  # no test reaches Ollama
    yield app
    app.state.task_worker.shutdown(wait=True)


def _wait_for_task(client, task_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        task = client.get(f"{TASKS}/{task_id}").json()
        if task["status"] in ("completed", "failed", "cancelled"):
            return task
        assert time.monotonic() < deadline, f"task still {task['status']}"
        time.sleep(0.02)


BODY = {"ticker": "AAPL", "trade_date": "2026-06-01"}


@pytest.mark.unit
class TestTaskUsage:
    def test_a_completed_task_reports_its_usage(self, app):
        with patch(GRAPH, _graph(["quick", "deep", "deep"])):
            client = TestClient(app)
            task_id = client.post(TASKS, json=BODY).json()["task_id"]
            task = _wait_for_task(client, task_id)
        assert task["status"] == "completed"
        usage = task["token_usage"]
        assert usage["partial"] is False
        assert _entries(usage) == [
            ("quick", "ollama", "qwen3:4b", 1), ("deep", "anthropic", "claude-opus-5-5", 2),
        ]
        assert "token_usage" not in task["result"]
        listed = client.get(TASKS).json()
        assert listed[0]["token_usage"] == usage

    def test_a_task_failing_midway_keeps_its_usage_so_far(self, app):
        with patch(GRAPH, _graph(["quick", "deep", "raise"])):
            client = TestClient(app)
            task_id = client.post(TASKS, json=BODY).json()["task_id"]
            task = _wait_for_task(client, task_id)
        assert task["status"] == "failed"
        assert task["token_usage"]["partial"] is True
        assert task["token_usage"]["totals"]["calls"] == 2
        assert task["token_usage"]["totals"]["cache_read_input_tokens"] == 1000

    def test_a_task_cancelled_midway_keeps_its_usage_so_far(self, app):
        hold, proceed = threading.Event(), threading.Event()
        with patch(GRAPH, _graph(["quick", "hold", "deep"], hold, proceed)):
            client = TestClient(app)
            task_id = client.post(TASKS, json=BODY).json()["task_id"]
            assert hold.wait(5)
            running = client.get(f"{TASKS}/{task_id}").json()
            assert running["status"] == "processing"
            assert running["token_usage"] is None  # set when the run ends
            client.post(f"{TASKS}/{task_id}/cancel")
            proceed.set()
            task = _wait_for_task(client, task_id)
        assert task["status"] == "cancelled"
        assert task["token_usage"]["partial"] is True
        assert _entries(task["token_usage"]) == [("quick", "ollama", "qwen3:4b", 1)]

    def test_a_task_not_yet_started_has_no_usage(self, app):
        hold, proceed = threading.Event(), threading.Event()
        with patch(GRAPH, _graph(["hold"], hold, proceed)):
            client = TestClient(app)
            running = client.post(TASKS, json=BODY).json()  # holds the only worker
            assert hold.wait(5)
            queued = client.post(TASKS, json={**BODY, "ticker": "MSFT"}).json()
            assert queued["token_usage"] is None
            assert client.get(f"{TASKS}/{queued['task_id']}").json()["token_usage"] is None
            proceed.set()
            # Both end before the stand-in graph is unpatched.
            for task in (running, queued):
                assert _wait_for_task(client, task["task_id"])["token_usage"] is not None


@pytest.mark.unit
class TestSyncUsage:
    def test_the_sync_response_reports_the_usage(self, app):
        with patch(GRAPH, _graph(["quick", "deep"])):
            resp = TestClient(app).post(ANALYZE, json=BODY)
        assert resp.status_code == 200, resp.text
        usage = resp.json()["token_usage"]
        assert usage["partial"] is False
        assert usage["totals"]["calls"] == 2

    def test_a_failed_sync_run_reports_its_usage_so_far(self, app):
        with patch(GRAPH, _graph(["quick", "raise"])):
            resp = TestClient(app, raise_server_exceptions=False).post(ANALYZE, json=BODY)
        assert resp.status_code == 500
        assert resp.json()["token_usage"]["partial"] is True
        assert resp.json()["token_usage"]["totals"]["calls"] == 1


@pytest.mark.unit
class TestSchema:
    def test_the_field_is_in_the_openapi_schema_with_descriptions(self, app):
        schemas = TestClient(app).get("/api/v1/openapi.json").json()["components"]["schemas"]
        for name in ("TaskResponse", "AnalyzeResponse"):
            field = schemas[name]["properties"]["token_usage"]
            assert field["description"]
            assert {"$ref": "#/components/schemas/TokenUsage"} in field["anyOf"]
        usage = schemas["TokenUsage"]["properties"]
        assert set(usage) == {"tiers", "totals", "partial"}
        tier = schemas["TierTokenUsage"]["properties"]
        assert list(tier) == [
            "tier", "provider", "model", "calls", "input_tokens", "output_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens", "calls_without_usage",
        ]
        including = "INCLUDING cache_read_input_tokens and cache_creation_input_tokens"
        assert including in tier["input_tokens"]["description"]
        assert including in schemas["TokenUsageTotals"]["properties"]["input_tokens"]["description"]
        assert "lower bound" in tier["calls_without_usage"]["description"]
        assert "lower bound" in usage["totals"]["description"]
        assert all(field.get("description") for field in tier.values())
