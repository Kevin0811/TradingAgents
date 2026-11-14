"""Offline harness for the discussion-capture tests (Plan 08, stage 08-A).

The REAL graph runs under a fake chat model and canned data, as in
``test_graph_end_to_end.py``: no model, no network, no real data. The fake
records every prompt it is sent, so a test can check none reaches an event.
Marker strings can be planted in the instrument, past and portfolio contexts
and in every tool's output.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from tradingagents.agents import context, schemas
from tradingagents.agents.analysts import sentiment_analyst
from tradingagents.dataflows import router
from tradingagents.dataflows.vendors.yahoo import market as yahoo_market, snapshot
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import propagation, trading_graph
from tradingagents.memory import log as memory_log

TRADE_DATE = "2026-01-09"
TICKER = "SPKE"

MARK_INSTRUMENT = "QQ-INSTRUMENT-MARKER-QQ"
MARK_PAST = "QQ-PAST-MARKER-QQ"
MARK_PORTFOLIO = "QQ-PORTFOLIO-MARKER-QQ"
MARK_TOOL = "QQ-TOOL-OUTPUT-MARKER-QQ"
MARKERS = (MARK_INSTRUMENT, MARK_PAST, MARK_PORTFOLIO, MARK_TOOL)

ARGS = {"symbol": TICKER, "ticker": TICKER, "curr_date": TRADE_DATE, "start_date": "2026-01-02",
        "end_date": TRADE_DATE, "indicator": "rsi", "topic": "rates", "freq": "quarterly",
        "query": "rates", "look_back_days": 7, "limit": 5}

FILLER = "Synthetic sentence about a fictional instrument, with no real data in it. "


def _structured(name: str) -> dict:
    canned = {
        "ResearchPlan": schemas.ResearchPlan(
            recommendation=schemas.PortfolioRating.OVERWEIGHT,
            rationale="RM rationale. " + FILLER * 3, strategic_actions="RM actions."),
        "TraderProposal": schemas.TraderProposal(
            action=schemas.TraderAction.BUY, reasoning="Trader reasoning. " + FILLER * 3),
        "PortfolioDecision": schemas.PortfolioDecision(
            rating=schemas.PortfolioRating.OVERWEIGHT, executive_summary="PM summary.",
            investment_thesis="PM thesis. " + FILLER * 3),
        "SentimentReport": schemas.SentimentReport(
            overall_band=schemas.SentimentBand.NEUTRAL, overall_score=5.0, confidence="low",
            narrative="Sentiment. " + FILLER * 3),
    }
    return canned[name].model_dump(mode="json")


class FakeChat(BaseChatModel):
    """Tool-calling, structured-output-capable fake (the 08-0 spike's).

    * bound to tools without ``tool_choice`` (an analyst): calls every tool once, then answers;
    * bound with ``tool_choice`` (``with_structured_output``): returns the schema as a tool call;
    * otherwise: prose naming the node it ran in, ``words`` sentences long.
    """

    words: int = 5
    prompts: list = Field(default_factory=list)   # every prompt text it was sent
    calls: list = Field(default_factory=list)     # one entry per call (shared by copies)
    hook: Any = None                              # hook(call_number) before each answer

    @property
    def _llm_type(self) -> str:
        return "discussion-fake"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, tools=None, tool_choice=None, **kw):
        for m in messages:
            if isinstance(m.content, str):
                self.prompts.append(m.content)
        self.calls.append(1)
        if self.hook is not None:
            self.hook(len(self.calls))
        md = getattr(run_manager, "metadata", {}) or {}
        where = (md.get("checkpoint_ns") or md.get("langgraph_node") or "?").split(":")[0]
        if tools and tool_choice:
            name = tools[0]["function"]["name"]
            msg = AIMessage(content="", tool_calls=[
                {"name": name, "id": "call_s", "args": _structured(name)}])
        elif tools and not isinstance(messages[-1], ToolMessage):
            calls = []
            for i, t in enumerate(tools):
                props = t["function"]["parameters"].get("properties", {})
                calls.append({"name": t["function"]["name"], "id": f"call_{i}",
                              "args": {k: v for k, v in ARGS.items() if k in props}})
            msg = AIMessage(content="", tool_calls=calls)
        else:
            msg = AIMessage(content=(f"[{where}] prose turn. " + FILLER * self.words
                                     + "\n\n**Rating**: Overweight\n\nFINAL TRANSACTION PROPOSAL: **BUY**"))
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


def install_offline(monkeypatch, model: FakeChat | None = None, markers: bool = False) -> FakeChat:
    """Canned data everywhere and ``model`` (with the API's callbacks) for every tier."""
    model = model or FakeChat()
    tool_text = (MARK_TOOL + " ") if markers else ""
    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in list(vendors):
            monkeypatch.setitem(vendors, vendor,
                                lambda *a, _m=method, **k: f"{tool_text}canned {_m} data")
    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=60),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000,
    })
    monkeypatch.setattr(snapshot, "load_ohlcv", lambda *a, **k: prices.copy())
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages",
                        lambda *a, **k: f"{tool_text}no posts")
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts",
                        lambda *a, **k: f"{tool_text}no posts")
    monkeypatch.setattr(yahoo_market.yf, "Ticker",
                        lambda s: type("T", (), {"info": {"longName": "Spike Corp"}})())
    context._identity.cache_clear()

    def client(config, tier, callbacks=None, **kwargs):
        # The API's callbacks (cancel, usage) ride on the model, as with a real client.
        return _Client(model.model_copy(update={"callbacks": callbacks}) if callbacks else model)

    monkeypatch.setattr(trading_graph, "create_tier_client", client)
    if markers:
        original = propagation.Propagator.create_initial_state

        def initial_state(self, *args, **kwargs):
            state = original(self, *args, **kwargs)
            state["instrument_context"] = f"{state['instrument_context']} {MARK_INSTRUMENT}"
            state["portfolio_context"] = MARK_PORTFOLIO
            return state

        monkeypatch.setattr(propagation.Propagator, "create_initial_state", initial_state)
        monkeypatch.setattr(memory_log.TradingMemoryLog, "get_past_context",
                            lambda self, *a, **k: f"Past lessons: {MARK_PAST}")
    return model


def run_config(tmp_path, debate: int = 1, risk: int = 1, **extra: Any) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
               memory_log_path=str(tmp_path / "log.md"), max_debate_rounds=debate,
               max_risk_discuss_rounds=risk, checkpoint_enabled=False, llm_provider="openai",
               **extra)
    return cfg
