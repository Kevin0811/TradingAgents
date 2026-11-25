"""A transcript rebuilt from the core's full_states_log (Plan 08, 08-D3).

Every log here is synthetic, built the way the agents build theirs: each turn
is ``"<Speaker> Analyst: <text>"`` appended to the history with ``"\\n"``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_neutral_debator,
)
from tradingagents.api import config as api_config
from tradingagents.api.app import create_app
from tradingagents.api.core.discussion_log import DiscussionLog
from tradingagents.api.core.rebuilt_transcript import build_rebuilt_transcript
from tradingagents.api.dependencies import get_state_repository
from tradingagents.api.infrastructure.discussion_capture import (
    MAPPING,
    MAPPING_VERSION,
    MAX_TASK_CONTENT_BYTES,
    MAX_TURN_CONTENT_BYTES,
    TaskEventEmitter,
    analyst_agent_ids,
)
from tradingagents.api.infrastructure.repositories import file_state_repository
from tradingagents.api.infrastructure.repositories.file_state_repository import (
    FileStateRepository,
)
from tradingagents.api.routers import discussion
from tradingagents.graph import trading_graph
from tradingagents.graph.propagation import Propagator

ROUTE = "/api/v1/analyze/rebuilt-transcript"
TICKER = "SPKE"
TRADE_DATE = "2026-01-09"
LOGGED_AT = datetime(2026, 1, 10, 3, 4, 5, tzinfo=UTC)

REPORT_FIELDS = {
    "market": "market_report", "social": "sentiment_report",
    "news": "news_report", "fundamentals": "fundamentals_report",
}
SEAT_BY_AGENT = {seat.agent_id: seat for seat in MAPPING.values()}


@pytest.fixture(autouse=True)
def _no_graph(monkeypatch):
    """Nothing here may build the graph (and so no model)."""
    def refuse(*args, **kwargs):
        raise AssertionError("TradingAgentsGraph must not be built")

    monkeypatch.setattr(trading_graph.TradingAgentsGraph, "__init__", refuse)
    monkeypatch.setitem(api_config.DEFAULT_CONFIG, "llm_provider", "openai")


def _debate(speakers, rounds, text=lambda name, i: f"{name.lower()} point {i}", keys=None):
    """A debate state the way the agents write it, and its turns in order."""
    keys = keys or [f"{name.lower()}_history" for name in speakers]
    state = {"history": "", **dict.fromkeys(keys, "")}
    turns = []
    for i in range(len(speakers) * rounds):
        name, key = speakers[i % len(speakers)], keys[i % len(speakers)]
        argument = f"{name} Analyst: {text(name, i)}"
        state["history"] += "\n" + argument
        state[key] += "\n" + argument
        turns.append(argument)
    return state, turns


def make_log(analysts=("market", "social", "news", "fundamentals"), debate_rounds=1,
             risk_rounds=1, settings=True, old=False, **overrides):
    research, research_turns = _debate(["Bull", "Bear"], debate_rounds)
    if old:
        risk, risk_turns = _debate(["Risky", "Safe", "Neutral"], risk_rounds,
                                   keys=["risky_history", "safe_history", "neutral_history"])
    else:
        risk, risk_turns = _debate(["Aggressive", "Conservative", "Neutral"], risk_rounds)
    log = {
        "company_of_interest": TICKER,
        "trade_date": TRADE_DATE,
        **dict.fromkeys(REPORT_FIELDS.values(), ""),
        **{REPORT_FIELDS[a]: f"{a} report body" for a in analysts},
        "investment_debate_state": {**research, "current_response": research_turns[-1]},
        "risk_debate_state": risk,
        "investment_plan": "research manager plan",
        ("trader_investment_decision" if old else "trader_investment_plan"): "trader plan",
        "final_trade_decision": "portfolio manager decision",
    }
    if settings:
        log["final_rating"] = "Hold"
        log["run_settings"] = {
            "analysts": list(analysts), "max_debate_rounds": debate_rounds,
            "max_risk_discuss_rounds": risk_rounds, "llm_provider": "openai",
        }
    log.update(overrides)
    return log, research_turns, risk_turns


def _agents(transcript):
    return [t["agent_id"] for t in transcript["turns"]]


def _assert_turn_shape(transcript):
    for number, turn in enumerate(transcript["turns"], start=1):
        seat = SEAT_BY_AGENT[turn["agent_id"]]
        assert turn["turn"] == number
        assert (turn["role"], turn["phase"]) == (seat.role, seat.phase)
        assert turn["started_at"] is None and turn["completed_at"] is None
        assert turn["aborted"] is False and turn["tools"] == []


class TestConverter:
    def test_four_analysts_one_round_each_give_twelve_turns_in_order(self):
        log, research, risk = make_log()
        t = build_rebuilt_transcript(log, LOGGED_AT)

        assert _agents(t) == [
            "market", "sentiment", "news", "fundamentals", "bull", "bear",
            "research_manager", "trader", "aggressive", "conservative", "neutral",
            "portfolio_manager",
        ]
        _assert_turn_shape(t)
        contents = [turn["content"] for turn in t["turns"]]
        assert contents[:4] == [f"{a} report body" for a in REPORT_FIELDS]
        assert contents[4:6] == research and contents[8:11] == risk
        assert contents[4].startswith("Bull Analyst: ")
        assert contents[6:8] == ["research manager plan", "trader plan"]
        assert contents[11] == "portfolio manager decision"
        assert not any(turn["truncated"] for turn in t["turns"])
        assert {k: t[k] for k in ("v", "mapping_version", "status", "tools_dropped",
                                  "events_dropped")} == {
            "v": 1, "mapping_version": MAPPING_VERSION, "status": "completed",
            "tools_dropped": 0, "events_dropped": 0,
        }
        assert t["run_started"] == {
            "ticker": TICKER, "trade_date": TRADE_DATE,
            "analysts": ["market", "sentiment", "news", "fundamentals"],
            "debate_rounds": 1, "risk_rounds": 1, "expected_turns": 12,
            "mapping_version": MAPPING_VERSION,
        }
        assert t["rebuilt"]["logged_at"] == "2026-01-10T03:04:05Z"
        assert t["rebuilt"]["settings_known"] is True
        assert t["rebuilt"]["split_exact"] is True

    def test_two_debate_and_two_risk_rounds(self):
        log, research, risk = make_log(debate_rounds=2, risk_rounds=2)
        t = build_rebuilt_transcript(log, None)

        assert _agents(t)[4:8] == ["bull", "bear", "bull", "bear"]
        assert _agents(t)[10:16] == ["aggressive", "conservative", "neutral"] * 2
        assert [turn["content"] for turn in t["turns"][4:8]] == research
        assert [turn["content"] for turn in t["turns"][10:16]] == risk
        assert len(t["turns"]) == t["run_started"]["expected_turns"] == 4 + 4 + 6 + 3
        assert t["rebuilt"]["split_exact"] is True
        assert t["rebuilt"]["logged_at"] is None
        _assert_turn_shape(t)

    def test_a_subset_of_analysts_in_the_run_order(self):
        log, _, _ = make_log(analysts=("news", "market"))
        t = build_rebuilt_transcript(log, LOGGED_AT)

        assert _agents(t)[:3] == ["news", "market", "bull"]
        assert t["run_started"]["analysts"] == ["news", "market"]
        assert t["run_started"]["expected_turns"] == 10 == len(t["turns"])

    def test_no_run_settings(self):
        log, _, _ = make_log(analysts=("fundamentals", "social", "market"), settings=False,
                             debate_rounds=2)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        # The default order, only the reports present.
        assert _agents(t)[:4] == ["market", "sentiment", "fundamentals", "bull"]
        assert t["rebuilt"]["settings_known"] is False
        assert t["rebuilt"]["split_exact"] is True
        assert t["run_started"]["analysts"] == ["market", "sentiment", "fundamentals"]
        assert t["run_started"]["expected_turns"] == len(t["turns"]) == 3 + 4 + 2 + 3 + 1
        assert (t["run_started"]["debate_rounds"], t["run_started"]["risk_rounds"]) == (2, 1)
        _assert_turn_shape(t)

    def test_old_keys_risky_safe_and_trader_investment_decision(self):
        log, _, risk = make_log(old=True, settings=False)
        assert "Risky Analyst: " in log["risk_debate_state"]["history"]
        t = build_rebuilt_transcript(log, LOGGED_AT)

        assert _agents(t)[6:12] == [
            "research_manager", "trader", "aggressive", "conservative", "neutral",
            "portfolio_manager",
        ]
        assert t["turns"][7]["content"] == "trader plan"
        assert [turn["content"] for turn in t["turns"][8:11]] == risk
        assert t["turns"][8]["content"].startswith("Risky Analyst: ")
        assert t["turns"][9]["content"].startswith("Safe Analyst: ")
        assert t["rebuilt"]["split_exact"] is True

    def test_old_keys_fall_back_to_the_risky_and_safe_sides(self):
        log, _, _ = make_log(old=True, risk_rounds=2)
        log["run_settings"]["max_risk_discuss_rounds"] = 1  # the count no longer matches
        t = build_rebuilt_transcript(log, LOGGED_AT)

        risk = [turn for turn in t["turns"] if turn["phase"] == "risk_debate"]
        assert [turn["agent_id"] for turn in risk] == ["aggressive", "conservative", "neutral"]
        assert risk[0]["content"] == log["risk_debate_state"]["risky_history"][1:]
        assert risk[1]["content"] == log["risk_debate_state"]["safe_history"][1:]
        assert t["rebuilt"]["split_exact"] is False

    @pytest.mark.parametrize("settings", [True, False])
    def test_a_quoted_prefix_over_two_rounds_splits_exactly(self, settings):
        # Each speaker quotes the other's prefix (and its own after it): the side
        # histories still place every boundary.
        def text(name, i):
            if name == "Bear":
                return f"rebuttal {i}\nBull Analyst: quoted"
            return f"point {i}\nBear Analyst: you said\nBull Analyst: but"

        research, turns = _debate(["Bull", "Bear"], 2, text=text)
        log, _, _ = make_log(settings=settings, debate_rounds=2, investment_debate_state=research)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        debate = [turn for turn in t["turns"] if turn["phase"] == "research_debate"]
        assert [turn["agent_id"] for turn in debate] == ["bull", "bear", "bull", "bear"]
        assert [turn["content"] for turn in debate] == turns
        assert t["rebuilt"]["split_exact"] is True
        _assert_turn_shape(t)

    def test_one_round_with_every_side_is_one_exact_turn_per_side(self):
        def text(name, i):
            return f"{name} says\nNeutral Analyst: quoted\nAggressive Analyst: too"

        risk, turns = _debate(["Aggressive", "Conservative", "Neutral"], 1, text=text)
        log, _, _ = make_log(risk_debate_state=risk)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        debate = [turn for turn in t["turns"] if turn["phase"] == "risk_debate"]
        assert [turn["content"] for turn in debate] == turns
        assert t["rebuilt"]["split_exact"] is True

    def test_empty_sides_keep_all_the_debate_text_unverified(self):
        def text(name, i):
            return "rebuttal\nBull Analyst: quoted" if name == "Bear" else f"point {i}"

        research, _ = _debate(["Bull", "Bear"], 1, text=text)
        research.update(bull_history="", bear_history="")
        log, _, _ = make_log(settings=False, investment_debate_state=research)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        debate = [turn for turn in t["turns"] if turn["phase"] == "research_debate"]
        # Each piece to the speaker whose prefix it starts with; nothing dropped.
        assert [turn["agent_id"] for turn in debate] == ["bull", "bear", "bull"]
        assert "\n" + "\n".join(turn["content"] for turn in debate) == research["history"]
        assert t["rebuilt"]["split_exact"] is False

    @pytest.mark.parametrize(("settings", "exact"), [(True, True), (False, False)])
    def test_without_sides_only_the_known_round_count_confirms_the_split(self, settings, exact):
        research, turns = _debate(["Bull", "Bear"], 2)
        research.update(bull_history="", bear_history="")
        log, _, _ = make_log(settings=settings, debate_rounds=2, investment_debate_state=research)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        debate = [turn for turn in t["turns"] if turn["phase"] == "research_debate"]
        assert [turn["content"] for turn in debate] == turns
        assert t["rebuilt"]["split_exact"] is exact

    def test_a_count_that_disagrees_with_the_sides_is_not_exact(self):
        research, _ = _debate(["Bull", "Bear"], 2)
        research["bear_history"] = "\nBear Analyst: something else"
        log, _, _ = make_log(settings=False, debate_rounds=2, investment_debate_state=research)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        debate = [turn for turn in t["turns"] if turn["phase"] == "research_debate"]
        # The sides lack history's bear turns: history by prefix, nothing of it dropped.
        assert [turn["agent_id"] for turn in debate] == ["bull", "bear", "bull", "bear"]
        assert "\n" + "\n".join(turn["content"] for turn in debate) == research["history"]
        assert t["rebuilt"]["split_exact"] is False

    def test_sides_holding_all_of_a_disagreeing_history_are_one_turn_each(self):
        research, _ = _debate(["Bull", "Bear"], 2)
        research["bear_history"] += "\nBear Analyst: a turn history lacks"
        log, _, _ = make_log(settings=False, investment_debate_state=research)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        debate = [turn for turn in t["turns"] if turn["phase"] == "research_debate"]
        assert [turn["agent_id"] for turn in debate] == ["bull", "bear"]
        assert debate[1]["content"] == research["bear_history"][1:]
        assert t["rebuilt"]["split_exact"] is False

    @pytest.mark.parametrize("shape", ["20000 one-character turns", "a 1 MB turn of quotes"])
    def test_the_split_is_linear_on_pathological_logs(self, shape):
        if shape.startswith("20000"):
            research, turns = _debate(["Bull", "Bear"], 10_000, text=lambda name, i: "x")
        else:
            quote = "q\nBear Analyst: q"  # a prefix of the next speaker, quoted
            research, turns = _debate(
                ["Bull", "Bear"], 1,
                text=lambda name, i: quote * (1_000_000 // len(quote)) if i == 0 else "y")
        log, _, _ = make_log(settings=False, investment_debate_state=research)

        began = time.perf_counter()
        t = build_rebuilt_transcript(log, LOGGED_AT)
        elapsed = time.perf_counter() - began

        assert t["rebuilt"]["split_exact"] is True
        assert sum(turn["phase"] == "research_debate" for turn in t["turns"]) == len(turns)
        assert elapsed < 1.5, f"{shape}: {elapsed:.2f}s"

    def test_with_settings_an_empty_field_keeps_its_seat_with_empty_content(self):
        log, _, _ = make_log(investment_plan="", trader_investment_plan=None,
                             sentiment_report="")
        t = build_rebuilt_transcript(log, LOGGED_AT)

        assert len(t["turns"]) == t["run_started"]["expected_turns"] == 12
        by_agent = {turn["agent_id"]: turn for turn in t["turns"]}
        for agent_id in ("sentiment", "research_manager", "trader"):
            assert by_agent[agent_id]["content"] == ""
            assert by_agent[agent_id]["truncated"] is False

    def test_without_settings_empty_fields_are_skipped(self):
        log, _, _ = make_log(settings=False, investment_plan="", trader_investment_plan=None)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        assert "research_manager" not in _agents(t) and "trader" not in _agents(t)
        assert [turn["turn"] for turn in t["turns"]] == list(range(1, 11))
        assert t["run_started"]["expected_turns"] == 10

    def test_a_turn_is_capped_at_32_kib_on_a_character_boundary(self):
        long = "界" * 20000  # 60000 bytes
        log, _, _ = make_log(market_report=long)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        market = t["turns"][0]
        assert market["truncated"] is True
        assert len(market["content"].encode("utf-8")) == MAX_TURN_CONTENT_BYTES - 2
        assert long.startswith(market["content"])

    def test_the_task_budget_keeps_the_final_decision_share(self):
        # 70 debate rounds: 150 expected turns, a 4 MiB budget (128 full turns).
        rounds = 70
        research, _ = _debate(["Bull", "Bear"], rounds,
                              text=lambda name, i: f"{i} " + "x" * 40000)
        log, _, _ = make_log(debate_rounds=rounds, investment_debate_state=research,
                             final_trade_decision="P" * 40000)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        assert t["run_started"]["expected_turns"] == 150
        sizes = [len(turn["content"].encode("utf-8")) for turn in t["turns"]]
        assert sum(sizes) == MAX_TASK_CONTENT_BYTES
        final = t["turns"][-1]
        assert final["agent_id"] == "portfolio_manager"
        assert final["content"] == "P" * MAX_TURN_CONTENT_BYTES and final["truncated"]
        # The turns before it ran out, leaving the final decision's 32 KiB free.
        assert t["turns"][-2]["content"] == "" and t["turns"][-2]["truncated"]
        assert sum(sizes[:-1]) == MAX_TASK_CONTENT_BYTES - MAX_TURN_CONTENT_BYTES

    def test_fingerprints_are_sha256_of_the_raw_uncapped_fields(self):
        long = "決策 " * 20000
        log, _, _ = make_log(analysts=("market", "news"), final_trade_decision=long)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        def sha(text):
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        assert t["rebuilt"]["fingerprints"] == {
            "market_report": sha("market report body"),
            "news_report": sha("news report body"),
            "investment_plan": sha("research manager plan"),
            "trader_investment_plan": sha("trader plan"),
            "final_trade_decision": sha(long),
        }
        assert t["turns"][-1]["truncated"]  # the turn is capped, its fingerprint is not

    def test_the_horizons_beside_the_transcript(self):
        """investment_horizon is unknown to a log; time_horizon is the PM's, parsed."""
        log, _, _ = make_log(final_trade_decision=(
            "**Rating**: Buy\n\n**Time Horizon**: 3-6 months"))
        t = build_rebuilt_transcript(log, LOGGED_AT)
        assert t["investment_horizon"] is None
        assert t["time_horizon"] == "3-6 months"
        assert "time_horizon" not in t["run_started"]
        assert "investment_horizon" not in t["run_started"]

        not_provided, _, _ = make_log(final_trade_decision=(
            "**Rating**: Buy\n\n**Time Horizon**: not provided"))
        assert build_rebuilt_transcript(not_provided, LOGGED_AT)["time_horizon"] is None
        assert build_rebuilt_transcript(make_log()[0], LOGGED_AT)["time_horizon"] is None

    def test_old_trader_key_is_fingerprinted_under_the_result_name(self):
        log, _, _ = make_log(old=True)
        fingerprints = build_rebuilt_transcript(log, LOGGED_AT)["rebuilt"]["fingerprints"]
        assert fingerprints["trader_investment_plan"] == hashlib.sha256(b"trader plan").hexdigest()


# -- parity with live capture -------------------------------------------------

NODE_BY_AGENT = {seat.agent_id: node for node, seat in MAPPING.items()}


class _StubLLM:
    """The only stub: the LLM boundary. Each call returns the next fixed text."""

    def __init__(self, texts):
        self._texts = iter(texts)

    def invoke(self, prompt):
        return SimpleNamespace(content=next(self._texts))


def _live_and_logged(rounds, reports):
    """One synthetic run, driven through the real debate nodes.

    Every node's output goes to the live TaskEventEmitter as LangGraph reports
    it and is merged into the state as the graph does; the managers, the trader
    and the analysts (tool-using or structured-output nodes) have their fields
    set directly. Returns the live transcript and the log ``_log_state`` writes.
    """
    core = ["market", "social", "news", "fundamentals"]
    state = Propagator().create_initial_state(TICKER, TRADE_DATE)
    log = DiscussionLog("parity")
    emitter = TaskEventEmitter(log)
    emitter.start(ticker=TICKER, trade_date=TRADE_DATE, analysts=analyst_agent_ids(core),
                  debate_rounds=rounds, risk_rounds=rounds)
    run_ids = iter(range(1000))

    def step(agent_id, node_or_output):
        node = NODE_BY_AGENT[agent_id]
        run_id = next(run_ids)
        emitter.on_chain_start({}, {}, run_id=run_id, name=node,
                               metadata={"langgraph_node": node, "checkpoint_ns": ""})
        output = node_or_output(state) if callable(node_or_output) else node_or_output
        state.update(output)
        emitter.on_chain_end(output, run_id=run_id)

    research_says = [f"bull {i}" if i % 2 == 0 else f"bear {i}\nBull Analyst: quoted"
                     for i in range(2 * rounds)]
    risk_says = [f"risk {i}\nNeutral Analyst: quoted" for i in range(3 * rounds)]
    research_llm, risk_llm = _StubLLM(research_says), _StubLLM(risk_says)
    bull, bear = create_bull_researcher(research_llm), create_bear_researcher(research_llm)
    debaters = [create_aggressive_debator(risk_llm), create_conservative_debator(risk_llm),
                create_neutral_debator(risk_llm)]

    for agent_id, field in [("market", "market_report"), ("sentiment", "sentiment_report"),
                            ("news", "news_report"), ("fundamentals", "fundamentals_report")]:
        step(agent_id, {field: reports.get(field, "")})
    for _ in range(rounds):  # conditional_logic: bull, bear, ...
        step("bull", bull)
        step("bear", bear)
    step("research_manager", {"investment_plan": "research manager plan"})
    step("trader", {"trader_investment_plan": "trader plan"})
    for _ in range(rounds):  # aggressive, conservative, neutral, ...
        for agent_id, node in zip(["aggressive", "conservative", "neutral"], debaters,
                                  strict=True):
            step(agent_id, node)
    step("portfolio_manager", {"final_trade_decision": "P" * 40000})
    emitter.close()
    log.finish("completed")

    debate, risk = state["investment_debate_state"], state["risk_debate_state"]
    logged = {  # the fields _log_state writes
        "company_of_interest": state["company_of_interest"],
        "trade_date": state["trade_date"],
        **{field: state[field] for field in REPORT_FIELDS.values()},
        "investment_debate_state": {
            key: debate[key] for key in ("bull_history", "bear_history", "history",
                                         "current_response")},
        "trader_investment_plan": state["trader_investment_plan"],
        "risk_debate_state": {
            key: risk[key] for key in ("aggressive_history", "conservative_history",
                                       "neutral_history", "history")},
        "investment_plan": state["investment_plan"],
        "final_trade_decision": state["final_trade_decision"],
        "final_rating": "Hold",
        "run_settings": {"analysts": core, "max_debate_rounds": rounds,
                         "max_risk_discuss_rounds": rounds},
    }
    return log.transcript(), json.loads(json.dumps(logged, ensure_ascii=False))


class TestParityWithLiveCapture:
    @pytest.mark.parametrize("rounds", [1, 2])
    def test_the_rebuilt_turns_equal_the_live_ones(self, rounds):
        reports = {"market_report": "界" * 15000, "sentiment_report": "",  # capped; empty
                   "news_report": "news body", "fundamentals_report": "fundamentals body"}
        live, logged = _live_and_logged(rounds, reports)
        rebuilt = build_rebuilt_transcript(logged, LOGGED_AT)

        def view(transcript):
            return [(t["agent_id"], t["role"], t["phase"], t["content"], t["truncated"])
                    for t in transcript["turns"]]

        assert len(live["turns"]) == 4 + 2 * rounds + 2 + 3 * rounds + 1
        assert view(rebuilt) == view(live)
        assert [set(t) for t in rebuilt["turns"]] == [set(t) for t in live["turns"]]
        assert rebuilt["run_started"] == live["run_started"]
        assert rebuilt["rebuilt"]["split_exact"] is True
        assert live["turns"][0]["truncated"] and live["turns"][1]["content"] == ""


# -- the route ----------------------------------------------------------------


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def client(results):
    app = create_app(overrides={"results_dir": str(results)})
    yield TestClient(app)
    app.state.task_worker.shutdown(wait=True)


def _write_log(results, ticker, body, trade_date=TRADE_DATE):
    folder = results / ticker / "TradingAgentsStrategy_logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"full_states_log_{trade_date}.json"
    path.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    os.utime(path, (LOGGED_AT.timestamp(), LOGGED_AT.timestamp()))
    return path


class TestRoute:
    def test_200_serves_the_rebuilt_transcript(self, client, results):
        log, _, _ = make_log()
        _write_log(results, TICKER, log)

        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == build_rebuilt_transcript(log, LOGGED_AT)
        assert set(body) == {"v", "mapping_version", "status", "run_started", "turns",
                             "tools_dropped", "events_dropped", "investment_horizon",
                             "time_horizon", "rebuilt"}
        assert body["rebuilt"]["logged_at"] == "2026-01-10T03:04:05Z"
        assert len(body["turns"]) == 12

    def test_404_without_a_log(self, client, results):
        _write_log(results, TICKER, make_log()[0], trade_date="2026-01-08")
        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})
        assert resp.status_code == 404
        assert set(resp.json()) == {"error", "detail"}

    @pytest.mark.parametrize("ticker", ["../etc", "..", "a/b", "..%2Fx", "", " ", "x" * 40])
    def test_422_for_an_unsafe_ticker(self, client, results, ticker):
        _write_log(results, "x", make_log()[0])  # a file a traversal could reach
        resp = client.get(ROUTE, params={"ticker": ticker, "trade_date": TRADE_DATE})
        assert resp.status_code == 422
        assert resp.json() == {"error": "Invalid request", "detail": "ticker is not a valid symbol"}

    @pytest.mark.parametrize("trade_date", ["2026-13-01", "2026-02-30", "20260109",
                                            "2026-1-9", "../2026-01-09", ""])
    def test_422_for_a_bad_trade_date(self, client, results, trade_date):
        _write_log(results, TICKER, make_log()[0])
        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": trade_date})
        assert resp.status_code == 422
        if trade_date in ("2026-13-01", "2026-02-30"):  # past the query pattern: ours
            assert resp.json() == {"error": "Invalid request",
                                   "detail": "trade_date is not a valid YYYY-MM-DD date"}
        else:  # FastAPI's own validation 422
            assert isinstance(resp.json()["detail"], list)

    @pytest.mark.parametrize("body", ["[1, 2]", "not-utf8", "too-big"])
    def test_an_unreadable_log_is_a_safe_500(self, client, results, tmp_path, body,
                                             monkeypatch, caplog):
        path = _write_log(results, TICKER, make_log()[0])
        if body == "not-utf8":
            path.write_bytes(b'{"market_report": "\xff\xfe"}')  # not UTF-8
        elif body == "too-big":
            monkeypatch.setattr(file_state_repository, "MAX_STATE_LOG_BYTES", 64)
        else:
            path.write_text(body, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})

        assert resp.status_code == 500
        assert resp.json() == {
            "error": "Could not read the saved state log",
            "detail": "The state log for this ticker and date is unreadable or not valid JSON.",
        }
        assert str(tmp_path) not in resp.text and "full_states_log" not in resp.text
        # Not an analysis failure: one WARNING naming only the exception class.
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        warning = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning) == 1 and str(tmp_path) not in warning[0]

    def test_the_size_cap_is_checked_before_parsing(self, client, results, monkeypatch):
        _write_log(results, TICKER, make_log()[0])
        monkeypatch.setattr(file_state_repository, "MAX_STATE_LOG_BYTES", 64)
        monkeypatch.setattr(file_state_repository.json, "load",
                            lambda f: pytest.fail("parsed a file over the cap"))
        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})
        assert resp.status_code == 500

    @pytest.mark.parametrize("body", ["", '{"market_report": "cut sho'])
    def test_an_empty_or_truncated_log_is_read_again_then_a_503(self, client, results,
                                                                 monkeypatch, body):
        path = _write_log(results, TICKER, body)
        os.utime(path)  # written just now: maybe still being written
        sleeps = []
        monkeypatch.setattr(discussion, "STATE_LOG_RETRY_DELAY", 0)
        monkeypatch.setattr(discussion.time, "sleep", sleeps.append)

        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})

        assert resp.status_code == 503
        assert resp.json() == {"error": "The saved state log is not complete JSON",
                               "detail": "The saved state log is being written; try again."}
        assert sleeps == [0]

    @pytest.mark.parametrize("body", ["", '{"market_report": "cut sho'])
    def test_an_old_empty_or_truncated_log_is_a_500_without_a_retry(self, client, results,
                                                                     monkeypatch, body):
        path = _write_log(results, TICKER, body)
        old = time.time() - discussion.STATE_LOG_FRESH_SECONDS - 5
        os.utime(path, (old, old))  # corrupt, not being written
        monkeypatch.setattr(discussion.time, "sleep",
                            lambda delay: pytest.fail("retried an old corrupt log"))

        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})

        assert resp.status_code == 500
        assert resp.json()["error"] == "Could not read the saved state log"

    def test_a_converter_bug_is_a_real_500_with_its_traceback(self, results, monkeypatch,
                                                               caplog):
        _write_log(results, TICKER, make_log()[0])

        def broken(state, logged_at):
            raise ValueError("converter bug")

        monkeypatch.setattr(discussion, "build_rebuilt_transcript", broken)
        app = create_app(overrides={"results_dir": str(results)})
        try:
            with caplog.at_level(logging.ERROR):
                resp = TestClient(app, raise_server_exceptions=False).get(
                    ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})
        finally:
            app.state.task_worker.shutdown(wait=True)

        assert resp.status_code == 500
        assert resp.json()["error"] == "Internal server error"
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors and errors[0].exc_info is not None

    def test_a_log_finished_during_the_retry_is_served(self, client, results, monkeypatch):
        log, _, _ = make_log()
        path = _write_log(results, TICKER, json.dumps(log)[:50])  # half written
        os.utime(path)
        monkeypatch.setattr(discussion.time, "sleep",
                            lambda delay: path.write_text(json.dumps(log), encoding="utf-8"))

        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})

        assert resp.status_code == 200, resp.text
        assert len(resp.json()["turns"]) == 12

    @pytest.mark.parametrize("link", ["folder", "file"])
    def test_a_symlink_out_of_results_is_not_found(self, client, results, tmp_path, link):
        outside = _write_log(tmp_path / "outside", TICKER, make_log()[0])
        if link == "folder":
            results.mkdir(parents=True)
            (results / TICKER).symlink_to(outside.parent.parent, target_is_directory=True)
        else:
            path = _write_log(results, TICKER, "{}")
            path.unlink()
            path.symlink_to(outside)

        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})

        assert resp.status_code == 404
        assert set(resp.json()) == {"error", "detail"} and "outside" not in resp.text

    def test_a_symlink_inside_results_is_followed(self, client, results):
        target = _write_log(results, "OTHER", make_log()[0])
        path = _write_log(results, TICKER, "{}")
        path.unlink()
        path.symlink_to(target)
        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})
        assert resp.status_code == 200

    def test_the_rebuild_runs_off_the_event_loop(self, client, results, monkeypatch):
        _write_log(results, TICKER, make_log()[0])
        seen = []

        def rebuild(state, logged_at):
            try:
                asyncio.get_running_loop()
                seen.append("event loop")
            except RuntimeError:
                seen.append("worker thread")
            return build_rebuilt_transcript(state, logged_at)

        monkeypatch.setattr(discussion, "build_rebuilt_transcript", rebuild)
        assert client.get(ROUTE, params={"ticker": TICKER,
                                         "trade_date": TRADE_DATE}).status_code == 200
        assert seen == ["worker thread"]

    def test_an_unreadable_forex_log_falls_back_to_the_ticker_as_given(self, client, results):
        _write_log(results, "USDTWD=X", "[1]")
        log, _, _ = make_log(company_of_interest="USDTWD")
        _write_log(results, "USDTWD", log)

        resp = client.get(ROUTE, params={"ticker": "USDTWD", "trade_date": TRADE_DATE})

        assert resp.status_code == 200, resp.text
        assert resp.json()["run_started"]["ticker"] == "USDTWD"

    def test_with_no_readable_form_the_first_error_answers(self, client, results):
        _write_log(results, "USDTWD=X", "[1]")
        _write_log(results, "USDTWD", "")
        resp = client.get(ROUTE, params={"ticker": "USDTWD", "trade_date": TRADE_DATE})
        assert resp.status_code == 500

    def test_the_api_forex_rewrite_is_looked_up(self, client, results):
        # POST /analyze runs USDTWD as USDTWD=X, so its log is under that folder.
        log, _, _ = make_log(company_of_interest="USDTWD=X")
        _write_log(results, "USDTWD=X", log)

        resp = client.get(ROUTE, params={"ticker": "USDTWD", "trade_date": TRADE_DATE})

        assert resp.status_code == 200, resp.text
        assert resp.json()["run_started"]["ticker"] == "USDTWD=X"

    def test_the_ticker_as_given_is_tried_after_the_rewrite(self, client, results):
        log, _, _ = make_log(company_of_interest="USDTWD")
        _write_log(results, "USDTWD", log)  # e.g. a CLI run

        resp = client.get(ROUTE, params={"ticker": "USDTWD", "trade_date": TRADE_DATE})

        assert resp.status_code == 200, resp.text
        assert resp.json()["run_started"]["ticker"] == "USDTWD"

    def test_the_route_is_documented(self, client):
        operation = client.get("/api/v1/openapi.json").json()["paths"][ROUTE]["get"]
        assert "full_states_log" in operation["description"]
        assert {"404", "422", "500", "503"} <= set(operation["responses"])
        assert "mtime" in operation["description"]
        assert "market, sentiment, news, fundamentals" in operation["description"]
        assert "must not compare turn numbers" in operation["description"]


def test_the_state_repository_dependencies_share_one_object():
    repository = FileStateRepository(results_dir="unused")
    assert asyncio.run(get_state_repository(repository)) is repository
