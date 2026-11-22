"""A transcript rebuilt from the core's full_states_log (Plan 08, 08-D3).

Every log here is synthetic, built the way the agents build theirs: each turn
is ``"<Speaker> Analyst: <text>"`` appended to the history with ``"\\n"``.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from tradingagents.api import config as api_config
from tradingagents.api.app import create_app
from tradingagents.api.core.rebuilt_transcript import build_rebuilt_transcript
from tradingagents.api.infrastructure.discussion_capture import (
    MAPPING,
    MAPPING_VERSION,
    MAX_TASK_CONTENT_BYTES,
    MAX_TURN_CONTENT_BYTES,
)
from tradingagents.graph import trading_graph

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
    def test_a_speaker_prefix_inside_a_turn_falls_back_to_one_turn_per_side(self, settings):
        def text(name, i):
            return "rebuttal\nBull Analyst: quoted" if name == "Bear" else f"point {i}"

        research, _ = _debate(["Bull", "Bear"], 1, text=text)
        log, _, _ = make_log(settings=settings, investment_debate_state=research)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        debate = [turn for turn in t["turns"] if turn["phase"] == "research_debate"]
        assert [turn["agent_id"] for turn in debate] == ["bull", "bear"]
        assert debate[0]["content"] == "Bull Analyst: point 0"
        assert debate[1]["content"] == "Bear Analyst: rebuttal\nBull Analyst: quoted"
        assert t["rebuilt"]["split_exact"] is False
        _assert_turn_shape(t)

    def test_empty_fields_are_skipped(self):
        log, _, _ = make_log(investment_plan="", trader_investment_plan=None)
        t = build_rebuilt_transcript(log, LOGGED_AT)

        assert "research_manager" not in _agents(t) and "trader" not in _agents(t)
        assert [turn["turn"] for turn in t["turns"]] == list(range(1, 11))
        assert t["run_started"]["expected_turns"] == 12  # from the settings

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

    def test_old_trader_key_is_fingerprinted_under_the_result_name(self):
        log, _, _ = make_log(old=True)
        fingerprints = build_rebuilt_transcript(log, LOGGED_AT)["rebuilt"]["fingerprints"]
        assert fingerprints["trader_investment_plan"] == hashlib.sha256(b"trader plan").hexdigest()


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
                             "tools_dropped", "events_dropped", "rebuilt"}
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

    @pytest.mark.parametrize("trade_date", ["2026-13-01", "2026-02-30", "20260109",
                                            "2026-1-9", "../2026-01-09", ""])
    def test_422_for_a_bad_trade_date(self, client, results, trade_date):
        _write_log(results, TICKER, make_log()[0])
        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": trade_date})
        assert resp.status_code == 422

    @pytest.mark.parametrize("body", ['{"market_report": "cut sho', "[1, 2]", "not-utf8"])
    def test_an_unreadable_log_is_a_safe_500(self, client, results, tmp_path, body):
        path = _write_log(results, TICKER, "{}")
        if body == "not-utf8":
            path.write_bytes(b'{"market_report": "\xff\xfe"}')  # not UTF-8
        else:
            path.write_text(body, encoding="utf-8")

        resp = client.get(ROUTE, params={"ticker": TICKER, "trade_date": TRADE_DATE})

        assert resp.status_code == 500
        assert set(resp.json()) == {"error", "detail"}
        assert str(tmp_path) not in resp.text and "full_states_log" not in resp.text

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
        assert {"404", "422", "500"} <= set(operation["responses"])
