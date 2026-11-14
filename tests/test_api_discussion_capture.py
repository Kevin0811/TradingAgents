"""Discussion capture (Plan 08, 08-A): the REAL graph under a fake model, no network.

The analysis service runs on a ``ta-task`` worker thread, as ``TaskWorker``
runs it, with a discussion log; the tests read the events it recorded.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.discussion_fakes import (
    MARKERS,
    TICKER,
    TRADE_DATE,
    install_offline,
    run_config,
)
from tradingagents.api.core.discussion_log import (
    MAX_EVENTS,
    DiscussionLog,
    DiscussionStore,
    build_transcript,
)
from tradingagents.api.core.exceptions import AnalysisCancelled, AnalysisError
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.infrastructure import discussion_capture as capture
from tradingagents.api.infrastructure.discussion_capture import (
    IGNORED_NODES,
    MAPPING,
    MAX_TOOL_CALLS,
    MAX_TURN_CONTENT_BYTES,
    TaskEventEmitter,
    cap_utf8,
    unmapped_nodes,
)
from tradingagents.api.infrastructure.transcript_files import TranscriptFiles
from tradingagents.api.schemas.task import TaskCreateRequest, TaskStatus
from tradingagents.graph import setup as graph_setup, trading_graph

FOUR = ("market", "sentiment", "news", "fundamentals")
PHASES = ["analysis", "research_debate", "research_decision", "trading_plan", "risk_debate",
          "final_decision"]
SEQUENTIAL_AGENTS = ["bull", "bear", "research_manager", "trader", "aggressive", "conservative",
                     "neutral", "portfolio_manager"]


@pytest.fixture
def final_states(monkeypatch):
    """Every final state the graph returns, to compare turn content against."""
    states: list = []
    original = trading_graph.TradingAgentsGraph.propagate

    def propagate(self, *args, **kwargs):
        state, signal = original(self, *args, **kwargs)
        states.append(state)
        return state, signal

    monkeypatch.setattr(trading_graph.TradingAgentsGraph, "propagate", propagate)
    return states


def _run(tmp_path, analysts=FOUR, debate=1, risk=1, debug=False, cancel_event=None,
         workers=1, log=None):
    log = log or DiscussionLog("task-1")
    service = AnalysisService(config=run_config(tmp_path, debate, risk), debug=debug)
    with ThreadPoolExecutor(workers, thread_name_prefix="ta-task") as pool:
        pool.submit(service.run_analysis, ticker=TICKER, trade_date=TRADE_DATE,
                    selected_analysts=analysts, cancel_event=cancel_event, events=log).result()
    log.finish("completed")
    return log


def _of(log, type_):
    return [e for e in log.read() if e["type"] == type_]


def _check_stream(log, analysts, debate, risk):
    events = log.read()
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    assert all(e["v"] == 1 and e["ms"] >= 0 and e["at"].endswith("Z") for e in events)
    assert events[0]["type"] == "run_started"
    assert events[1]["data"] == {"status": "running"}
    assert events[-1]["data"] == {"status": "completed"}
    expected = len(analysts) + 2 * debate + 3 * risk + 3
    assert events[0]["data"]["expected_turns"] == expected
    started = _of(log, "turn_started")
    completed = _of(log, "turn_completed")
    assert [s["data"]["turn"] for s in started] == list(range(1, expected + 1))
    assert sorted(c["data"]["turn"] for c in completed) == list(range(1, expected + 1))
    assert not _of(log, "turn_aborted")
    assert [p["data"]["phase"] for p in _of(log, "phase_changed")] == PHASES
    # The analysts open the run, in parallel; everyone after them speaks in turn.
    n = len(analysts)
    analyst_ids = {"sentiment" if a == "social" else a for a in analysts}
    assert {s["data"]["agent_id"] for s in started[:n]} == analyst_ids
    later = [s["data"]["agent_id"] for s in started[n:]]
    assert later == (["bull", "bear"] * debate + ["research_manager", "trader"]
                     + ["aggressive", "conservative", "neutral"] * risk + ["portfolio_manager"])
    by_id = {e["id"]: e for e in events}
    for s in started[n:]:
        done = next(c for c in completed if c["data"]["turn"] == s["data"]["turn"])
        assert all(e["type"] != "turn_started" for i, e in by_id.items() if s["id"] < i < done["id"])
    # Each phase_changed comes right before the first turn of its phase.
    for p in _of(log, "phase_changed"):
        nxt = by_id[p["id"] + 1]
        assert nxt["type"] == "turn_started" and nxt["data"]["phase"] == p["data"]["phase"]
    # Tool events: analysts only, each call answered once, in the analysis phase.
    calls, results = _of(log, "tool_call"), _of(log, "tool_result")
    assert calls and {c["data"]["agent_id"] for c in calls} <= analyst_ids - {"sentiment"}
    assert sorted(c["data"]["call_id"] for c in calls) == sorted(r["data"]["call_id"] for r in results)
    return started, completed


def _max_open(log, phase):
    open_, best = set(), 0
    for e in log.read():
        if e["type"] == "turn_started" and e["data"]["phase"] == phase:
            open_.add(e["data"]["turn"])
        elif e["type"] in ("turn_completed", "turn_aborted"):
            open_.discard(e["data"]["turn"])
        best = max(best, len(open_))
    return best


@pytest.mark.unit
class TestEventStream:
    def test_a_default_run_streams_twelve_turns_in_order(self, tmp_path, monkeypatch):
        model = install_offline(monkeypatch)

        def hook(n):
            # The first analyst's LLM call waits (up to 10 s) until another
            # analyst has made one: their turns overlap however threads run.
            deadline = time.monotonic() + 10
            while n == 1 and len(model.calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
        model.hook = hook
        log = _run(tmp_path)
        started, _ = _check_stream(log, FOUR, 1, 1)
        assert len(started) == 12
        run_started = log.read()[0]["data"]
        assert run_started == {
            "ticker": TICKER, "trade_date": TRADE_DATE, "analysts": list(FOUR),
            "debate_rounds": 1, "risk_rounds": 1, "expected_turns": 12,
            "mapping_version": capture.MAPPING_VERSION,
        }
        assert _max_open(log, "analysis") >= 2   # the analysts' turns overlap
        assert {r["data"]["ok"] for r in _of(log, "tool_result")} == {True}

    def test_two_analysts_and_two_rounds_make_fifteen_turns(self, tmp_path, monkeypatch):
        install_offline(monkeypatch)
        log = _run(tmp_path, analysts=("market", "news"), debate=2, risk=2)
        started, _ = _check_stream(log, ("market", "news"), 2, 2)
        assert len(started) == 15

    def test_the_debug_stream_path_is_captured_too(self, tmp_path, monkeypatch, capsys):
        install_offline(monkeypatch)
        log = _run(tmp_path, debug=True)
        started, _ = _check_stream(log, FOUR, 1, 1)
        assert len(started) == 12

    def test_concurrent_runs_keep_their_events_apart(self, tmp_path, monkeypatch):
        install_offline(monkeypatch).hook = lambda n: time.sleep(0.01)
        log1, log2 = DiscussionLog("one"), DiscussionLog("two")
        s1 = AnalysisService(config=run_config(tmp_path / "1", 1, 1))
        s2 = AnalysisService(config=run_config(tmp_path / "2", 2, 2))
        with ThreadPoolExecutor(2, thread_name_prefix="ta-task") as pool:
            f1 = pool.submit(s1.run_analysis, ticker=TICKER, trade_date=TRADE_DATE,
                             selected_analysts=FOUR, events=log1)
            f2 = pool.submit(s2.run_analysis, ticker=TICKER, trade_date=TRADE_DATE,
                             selected_analysts=("market", "news"), events=log2)
            f1.result()
            f2.result()
        log1.finish("completed")
        log2.finish("completed")
        assert len(_check_stream(log1, FOUR, 1, 1)[0]) == 12
        assert len(_check_stream(log2, ("market", "news"), 2, 2)[0]) == 15
        assert not {"sentiment", "fundamentals"} & {
            e["data"].get("agent_id") for e in log2.read()}

    def test_no_events_without_a_log_or_outside_the_run(self, tmp_path, monkeypatch):
        install_offline(monkeypatch)
        service = AnalysisService(config=run_config(tmp_path))
        log = DiscussionLog("t")
        # One worker thread: the later jobs run where the run set the emitter.
        with ThreadPoolExecutor(1, thread_name_prefix="ta-task") as pool:
            pool.submit(service.run_analysis, ticker=TICKER, trade_date=TRADE_DATE,
                        events=log).result()
            assert _of(log, "turn_completed")   # the emitter was set on this thread
            assert pool.submit(capture.DISCUSSION_EMITTER.get).result() is None
            count = log.last_id
            pool.submit(service.run_analysis, ticker=TICKER, trade_date=TRADE_DATE).result()
            assert pool.submit(capture.DISCUSSION_EMITTER.get).result() is None
        assert log.last_id == count   # a run without a log records nothing in another's


@pytest.mark.unit
class TestContent:
    def test_each_turn_is_the_final_state_field_of_its_seat(self, tmp_path, monkeypatch,
                                                            final_states):
        install_offline(monkeypatch)
        log = _run(tmp_path, debate=2, risk=1)
        (state,) = final_states
        histories = {"bull": "investment_debate_state.bull_history",
                     "bear": "investment_debate_state.bear_history",
                     "aggressive": "risk_debate_state.aggressive_history",
                     "conservative": "risk_debate_state.conservative_history",
                     "neutral": "risk_debate_state.neutral_history"}
        seats = {s.agent_id: s for s in MAPPING.values()}
        spoken: dict[str, list[str]] = {}
        for event in _of(log, "turn_completed"):
            agent, content = event["data"]["agent_id"], event["data"]["content"]
            assert content and not event["data"]["truncated"]
            spoken.setdefault(agent, []).append(content)
            if agent not in histories:
                assert content == capture._dig(state, seats[agent].content_path), agent
        # A debater's turns, in order, are exactly the entries of its history.
        for agent, path in histories.items():
            assert "".join("\n" + c for c in spoken[agent]) == capture._dig(state, path), agent
        assert len(_of(log, "turn_completed")) == 4 + 4 + 3 + 2 + 1

    def test_no_prompt_context_or_tool_output_reaches_any_event(self, tmp_path, monkeypatch):
        model = install_offline(monkeypatch, markers=True)
        log = _run(tmp_path)
        dumped = json.dumps(log.read())
        prompts = "\n".join(model.prompts)
        for marker in MARKERS:
            assert marker in prompts, f"{marker} was never planted"   # the test means something
            assert marker not in dumped, marker
        for prompt in {p for p in model.prompts if len(p) > 200}:
            assert prompt[:120] not in dumped
        transcript = json.dumps(log.transcript())
        assert not any(m in transcript for m in MARKERS)

    def test_a_turn_is_capped_at_32_kib_on_a_character_boundary(self):
        log = DiscussionLog("t")
        emitter = TaskEventEmitter(log)
        md = {"langgraph_node": "Market Analyst", "checkpoint_ns": ""}
        for i, text in enumerate(["é" * 20_000, "a" * MAX_TURN_CONTENT_BYTES]):
            emitter.on_chain_start({}, {}, run_id=i, metadata=md, name="Market Analyst")
            emitter.on_chain_end({"market_report": text}, run_id=i)
        big, exact = [e["data"] for e in _of(log, "turn_completed")]
        assert big["truncated"] is True
        assert len(big["content"].encode("utf-8")) == MAX_TURN_CONTENT_BYTES
        assert set(big["content"]) == {"é"}
        assert exact["truncated"] is False and len(exact["content"]) == MAX_TURN_CONTENT_BYTES
        transcript = log.transcript()
        assert [t["truncated"] for t in transcript["turns"]] == [True, False]

    def test_a_lone_surrogate_is_replaced_not_raised(self):
        assert cap_utf8("a\ud800b", 100) == ("a?b", False)
        assert cap_utf8("\ud800" * 10, 4) == ("????", True)
        log = DiscussionLog("t")
        emitter = TaskEventEmitter(log)
        md = {"langgraph_node": "Market Analyst", "checkpoint_ns": ""}
        emitter.on_chain_start({}, {}, run_id=1, metadata=md, name="Market Analyst")
        emitter.on_chain_end({"market_report": "x\udfffy"}, run_id=1)
        (done,) = _of(log, "turn_completed")
        assert done["data"]["content"] == "x?y"
        json.dumps(done).encode("utf-8")   # the stream can encode it

    @staticmethod
    def _speak(emitter, run_id, node, path, text):
        md = {"langgraph_node": node, "checkpoint_ns": ""}
        emitter.on_chain_start({}, {}, run_id=run_id, metadata=md, name=node)
        out: dict = {}
        cur = out
        *heads, last = path.split(".")
        for h in heads:
            cur = cur.setdefault(h, {})
        cur[last] = text
        emitter.on_chain_end(out, run_id=run_id)

    def test_long_rounds_never_blank_the_final_decision(self, monkeypatch):
        long = "x" * (MAX_TURN_CONTENT_BYTES + 5000)
        decision = "FINAL: " + "d" * 9000
        seats = [(n, s) for n, s in MAPPING.items() if s.agent_id != "portfolio_manager"]
        pm_node = next(n for n, s in MAPPING.items() if s.agent_id == "portfolio_manager")

        def run(n_turns, debate=10, risk=10):
            log = DiscussionLog("t")
            emitter = TaskEventEmitter(log)
            emitter.start(ticker=TICKER, trade_date=TRADE_DATE, analysts=list(FOUR),
                          debate_rounds=debate, risk_rounds=risk)
            for i in range(n_turns):
                node, seat = seats[i % len(seats)]
                self._speak(emitter, i, node, seat.content_path, long)
            self._speak(emitter, "pm", pm_node, "final_trade_decision", decision)
            return [e["data"] for e in _of(log, "turn_completed")]

        # Rounds 10/10: 57 turns, 56 of them over the cap before the decision.
        assert capture.expected_turns(4, 10, 10) == 57
        turns = run(56)
        assert turns[-1]["content"] == decision and not turns[-1]["truncated"]
        assert all(len(t["content"]) == MAX_TURN_CONTENT_BYTES for t in turns[:-1])
        # More text than even the 4 MiB ceiling: the decision's 32 KiB stays reserved.
        assert capture.task_content_budget(57) == 57 * MAX_TURN_CONTENT_BYTES
        assert capture.task_content_budget(1000) == capture.MAX_TASK_CONTENT_BYTES
        assert capture.task_content_budget(None) == capture.MIN_TASK_CONTENT_BYTES
        turns = run(200)
        assert turns[-1]["content"] == decision and not turns[-1]["truncated"]
        used = sum(len(t["content"].encode("utf-8")) for t in turns)
        assert used <= capture.MAX_TASK_CONTENT_BYTES


@pytest.mark.unit
class TestBounds:
    def test_tool_events_stop_at_the_cap_and_the_rest_are_counted(self):
        log = DiscussionLog("t")
        emitter = TaskEventEmitter(log)
        md = {"langgraph_node": "Market Analyst", "checkpoint_ns": ""}
        emitter.on_chain_start({}, {}, run_id="turn", metadata=md, name="Market Analyst")
        tool_md = {"langgraph_node": "tools", "checkpoint_ns": "Market Analyst:x|tools:y"}
        for i in range(MAX_TOOL_CALLS + 50):
            emitter.on_tool_start({}, "secret args", run_id=i, metadata=tool_md, name="get_news")
            emitter.on_tool_end("secret output", run_id=i)
        emitter.on_chain_end({"market_report": "done"}, run_id="turn")
        emitter.close()
        log.finish("completed")
        assert len(_of(log, "tool_call")) == len(_of(log, "tool_result")) == MAX_TOOL_CALLS
        assert {e["data"]["agent_id"] for e in _of(log, "tool_call")} == {"market"}
        assert log.read()[-1]["data"] == {"status": "completed", "tools_dropped": 50}
        transcript = log.transcript()
        assert transcript["tools_dropped"] == 50
        assert transcript["turns"][0]["tools"] == ["get_news"] * MAX_TOOL_CALLS
        assert "secret" not in json.dumps(log.read())

    def test_the_log_keeps_at_most_max_events_and_still_ends(self):
        log = DiscussionLog("t")
        for _ in range(MAX_EVENTS + 25):
            log.append("phase_changed", {"phase": "analysis"})
        log.finish("completed")
        events = log.read()
        assert len(events) == MAX_EVENTS + 2
        assert [e["id"] for e in events] == list(range(1, MAX_EVENTS + 3))
        # One marker at the first loss, past the cap, then only the terminal event.
        assert events[MAX_EVENTS]["type"] == "gap"
        assert events[MAX_EVENTS]["data"] == {"dropped_from_id": MAX_EVENTS + 1}
        assert [e["type"] for e in events].count("gap") == 1
        assert events[-1]["data"] == {"status": "completed", "events_dropped": 25}
        assert log.append("phase_changed", {"phase": "analysis"}) is None   # closed
        assert log.transcript()["events_dropped"] == 25


@pytest.mark.unit
class TestMapping:
    def test_every_graph_node_is_mapped_or_ignored(self, tmp_path, monkeypatch):
        install_offline(monkeypatch)
        graph = trading_graph.TradingAgentsGraph(
            selected_analysts=("market", "social", "news", "fundamentals"),
            config=run_config(tmp_path))
        nodes = set(graph.workflow.nodes)
        assert unmapped_nodes(nodes) == set()
        assert set(MAPPING) | IGNORED_NODES == nodes   # and no stale entries

    def test_a_new_node_fails_the_coverage_check(self, tmp_path, monkeypatch):
        install_offline(monkeypatch)
        original = graph_setup.GraphSetup.setup_graph

        def with_extra_node(self, *args, **kwargs):
            workflow = original(self, *args, **kwargs)
            workflow.add_node("Macro Strategist", lambda state: {})
            return workflow

        monkeypatch.setattr(graph_setup.GraphSetup, "setup_graph", with_extra_node)
        graph = trading_graph.TradingAgentsGraph(config=run_config(tmp_path))
        assert unmapped_nodes(graph.workflow.nodes) == {"Macro Strategist"}

    def test_unknown_top_level_nodes_and_ignored_or_inner_runs(self):
        log = DiscussionLog("t")
        emitter = TaskEventEmitter(log)
        emitter.on_chain_start({}, {}, run_id=1, name="Mystery",
                               metadata={"langgraph_node": "Mystery", "checkpoint_ns": ""})
        emitter.on_chain_start({}, {}, run_id=2, name="Memory Log",
                               metadata={"langgraph_node": "Memory Log", "checkpoint_ns": ""})
        emitter.on_chain_start({}, {}, run_id=3, name="agent",   # inside an analyst subgraph
                               metadata={"langgraph_node": "agent", "checkpoint_ns": "Market Analyst:1"})
        emitter.on_chain_start({}, {}, run_id=4, name="RunnableSequence",
                               metadata={"langgraph_node": "Trader", "checkpoint_ns": ""})
        emitter.on_tool_start({}, "", run_id=5, name="t",
                              metadata={"langgraph_node": "Memory Log", "checkpoint_ns": ""})
        emitter.on_chain_end({"anything": "x"}, run_id=1)
        events = log.read()
        assert [e["type"] for e in events] == ["turn_started", "turn_completed"]
        assert events[0]["data"] == {"turn": 1, "agent_id": "unknown", "role": "unknown",
                                     "phase": None}
        assert events[1]["data"]["content"] is None

    def test_callbacks_never_raise(self):
        class Broken(DiscussionLog):
            def append(self, *a, **k):
                raise RuntimeError("boom")

        emitter = TaskEventEmitter(Broken("t"))
        md = {"langgraph_node": "Trader", "checkpoint_ns": ""}
        emitter.on_chain_start({}, {}, run_id=1, metadata=md, name="Trader")
        emitter.on_chain_end({"trader_investment_plan": "x"}, run_id=1)
        emitter.on_tool_start({}, "", run_id=2, metadata=md, name="t")
        emitter.on_tool_end("", run_id=2)


@pytest.mark.unit
class TestEndings:
    def _cancel_at(self, model, cancel_event, call):
        def hook(n):
            if n == call:
                cancel_event.set()
        model.hook = hook

    def test_a_cancelled_run_aborts_every_open_turn(self, tmp_path, monkeypatch):
        model = install_offline(monkeypatch)
        cancel = threading.Event()
        self._cancel_at(model, cancel, 3)   # mid-analysis: several analyst turns open
        log = DiscussionLog("t")
        with pytest.raises(AnalysisCancelled):
            _run(tmp_path, cancel_event=cancel, log=log)
        log.finish("cancelled")
        started = {e["data"]["turn"] for e in _of(log, "turn_started")}
        completed = {e["data"]["turn"] for e in _of(log, "turn_completed")}
        aborted = [e["data"]["turn"] for e in _of(log, "turn_aborted")]
        assert aborted and len(aborted) == len(set(aborted))
        assert started == completed | set(aborted) and not completed & set(aborted)
        assert log.read()[-1]["data"] == {"status": "cancelled"}
        transcript = log.transcript()
        assert transcript["status"] == "cancelled"
        assert {t["turn"] for t in transcript["turns"] if t["aborted"]} == set(aborted)

    def test_a_failed_run_aborts_its_open_turn(self, tmp_path, monkeypatch):
        model = install_offline(monkeypatch)

        def hook(n):
            if n >= 10:   # every call from here: no fallback can save the run
                raise RuntimeError("provider unavailable")
        model.hook = hook
        log = DiscussionLog("t")
        with pytest.raises(AnalysisError):
            _run(tmp_path, log=log)
        log.finish("failed")
        assert _of(log, "turn_aborted")
        started = {e["data"]["turn"] for e in _of(log, "turn_started")}
        ended = {e["data"]["turn"] for e in log.read()
                 if e["type"] in ("turn_completed", "turn_aborted")}
        assert started == ended
        assert log.read()[-1]["data"]["status"] == "failed"


@pytest.mark.unit
class TestTranscriptSave:
    class _FailingAnalysis:
        """Starts the run's log, then fails: the run reached the transcript step."""

        def run_analysis(self, *, events=None, **kwargs):
            TaskEventEmitter(events).start(
                ticker="../escape", trade_date=TRADE_DATE, analysts=["market"],
                debate_rounds=1, risk_rounds=1)
            raise AnalysisError("provider unavailable")

    def test_an_unsafe_ticker_never_raises_into_the_worker(self, tmp_path, caplog):
        store = DiscussionStore()
        manager = TaskManager(discussions=store)
        results = tmp_path / "results"
        service = TaskService(manager, self._FailingAnalysis(), discussions=store,
                              transcripts=TranscriptFiles(results))
        task, _ = manager.create_task(TaskCreateRequest(ticker=TICKER, trade_date=TRADE_DATE))
        assert store.get(task.task_id).transcript()["status"] == "queued"
        # The ticker the transcript folder is named after fails the path check.
        unsafe = TaskCreateRequest.model_construct(
            ticker="../escape", trade_date=TRADE_DATE, asset_type="stock",
            selected_analysts=["market"])
        service.execute_analysis(task.task_id, unsafe)   # no exception escapes
        assert manager.get_task(task.task_id).status == TaskStatus.FAILED
        assert store.get(task.task_id).transcript()["status"] == "failed"
        # The file writer itself returns None for it rather than raising.
        transcript = store.get(task.task_id).transcript()
        assert TranscriptFiles(results).write("../escape", transcript) is None
        assert not list(tmp_path.rglob("discussion_*.json"))
        assert not list(tmp_path.rglob(".discussion-*"))
        assert any(task.task_id in r.getMessage() for r in caplog.records
                   if r.levelname == "WARNING")

    def test_a_broken_transcript_is_logged_not_raised(self, tmp_path, monkeypatch, caplog):
        store = DiscussionStore()
        manager = TaskManager(discussions=store)
        service = TaskService(manager, self._FailingAnalysis(), discussions=store,
                              transcripts=TranscriptFiles(tmp_path))
        task, _ = manager.create_task(TaskCreateRequest(ticker=TICKER, trade_date=TRADE_DATE))

        def boom(self, mapping_version=None):
            raise RuntimeError("secret turn text")
        monkeypatch.setattr(DiscussionLog, "transcript", boom)
        service.execute_analysis(task.task_id, TaskCreateRequest.model_construct(
            ticker=TICKER, trade_date=TRADE_DATE, asset_type="stock",
            selected_analysts=["market"]))
        assert manager.get_task(task.task_id).status == TaskStatus.FAILED
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any(task.task_id in w for w in warnings)
        assert not any("secret" in w for w in warnings)


def test_transcript_folds_events_into_turns():
    log = DiscussionLog("t")
    emitter = TaskEventEmitter(log)
    emitter.start(ticker=TICKER, trade_date=TRADE_DATE, analysts=["market"], debate_rounds=1,
                  risk_rounds=1, checkpoint_enabled=True)
    md = {"langgraph_node": "Market Analyst", "checkpoint_ns": ""}
    emitter.on_chain_start({}, {}, run_id=1, metadata=md, name="Market Analyst")
    tool_md = {"checkpoint_ns": "Market Analyst:a|tools:b"}
    for i, tool in enumerate(["get_stock_data", "get_indicators"]):
        emitter.on_tool_start({}, "", run_id=10 + i, metadata=tool_md, name=tool)
        emitter.on_tool_end("", run_id=10 + i)
    emitter.on_chain_end({"market_report": "report"}, run_id=1)
    emitter.close()
    log.finish("completed")
    transcript = build_transcript("t", log.read(), log.status)
    assert transcript["run_started"]["expected_turns"] is None   # checkpoint resume
    assert transcript["status"] == "completed"
    (turn,) = transcript["turns"]
    assert turn["tools"] == ["get_stock_data", "get_indicators"]
    assert (turn["agent_id"], turn["role"], turn["phase"], turn["content"]) == (
        "market", "analyst", "analysis", "report")
    assert turn["started_at"] and turn["completed_at"]


def test_dropping_an_unfinished_log_finishes_it_as_cancelled():
    # A queued task deleted before it ran: its log is dropped without a run,
    # and an open stream on it must still get a terminal task_status.
    store = DiscussionStore()
    log = store.create("6f1c6a52-0b7e-4d39-9d2c-1f1e7a3d0c11")
    store.drop("6f1c6a52-0b7e-4d39-9d2c-1f1e7a3d0c11")
    assert log.finished
    assert log.read()[-1]["type"] == "task_status"
    assert log.read()[-1]["data"] == {"status": "cancelled"}
    assert store.get("6f1c6a52-0b7e-4d39-9d2c-1f1e7a3d0c11") is None


def test_dropping_a_finished_log_adds_nothing():
    store = DiscussionStore()
    log = store.create("6f1c6a52-0b7e-4d39-9d2c-1f1e7a3d0c12")
    log.finish("completed")
    before = log.read()
    store.drop("6f1c6a52-0b7e-4d39-9d2c-1f1e7a3d0c12")
    assert log.read() == before
