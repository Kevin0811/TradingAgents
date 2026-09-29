"""Tests for cancelling analysis tasks (issue #4).

A queued task is cancelled at once and never runs; a running one stops at its
next LLM call, through a LangChain callback. The graph is replaced by a stand-in
that drives a real ``langchain_core`` fake chat model with the callbacks the API
passes, so the cancel exception travels the same path it takes in the core.
"""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import Future
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tradingagents.api.app import create_app
from tradingagents.api.core.exceptions import AnalysisCancelled, AnalysisError
from tradingagents.api.core.task_manager import (
    CANCELLED_BEFORE_START,
    CANCELLED_WHILE_RUNNING,
    CancelOutcome,
    TaskManager,
)
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.infrastructure.llm_callbacks import CancelOnLLMStart
from tradingagents.api.schemas.task import TaskCreateRequest, TaskStatus

GRAPH = "tradingagents.api.domain.services.analysis_service.TradingAgentsGraph"
TASKS = "/api/v1/analyze/tasks"


def _request(ticker="AAPL"):
    return TaskCreateRequest(ticker=ticker, trade_date="2026-06-01")


def _scripted_graph(first_call_done: threading.Event, proceed: threading.Event, log: list):
    """A TradingAgentsGraph stand-in making two LLM calls, pausing between them."""

    class Graph:
        def __init__(self, *args, callbacks=None, **kwargs):
            self.model = FakeListChatModel(responses=["one", "two"], callbacks=callbacks)
            log.append("built")

        def propagate(self, ticker, trade_date, asset_type="stock"):
            self.model.invoke("first")
            log.append("first call")
            first_call_done.set()
            assert proceed.wait(5)
            self.model.invoke("second")
            log.append("second call")
            return {"final_trade_decision": "Buy"}, "Buy"

    return Graph


# ---------------------------------------------------------------------------
# LangChain propagation (the mechanism everything else relies on)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCancelPropagatesThroughLangChain:
    @pytest.mark.parametrize(
        "make_model",
        [
            lambda cb: FakeListChatModel(responses=["a", "b"], callbacks=cb),  # on_chat_model_start
            lambda cb: FakeListLLM(responses=["a", "b"], callbacks=cb),  # on_llm_start
        ],
        ids=["chat_model", "llm"],
    )
    def test_cancel_exception_propagates_out_of_the_llm_call(self, make_model):
        event = threading.Event()
        model = make_model([CancelOnLLMStart(event)])
        model.invoke("hi")  # not cancelled: the call runs
        event.set()
        with pytest.raises(AnalysisCancelled):
            model.invoke("hi")

    def test_cancel_exception_unwinds_a_langgraph_graph(self):
        from langgraph.graph import END, START, StateGraph

        class State(TypedDict):
            out: list

        event = threading.Event()
        model = FakeListChatModel(responses=["a", "b"], callbacks=[CancelOnLLMStart(event)])
        reached = []

        def first(state):
            answer = model.invoke("x").content
            event.set()  # cancelled between two nodes' LLM calls
            return {"out": [answer]}

        def second(state):
            answer = model.invoke("y").content
            reached.append("second")
            return {"out": state["out"] + [answer]}

        builder = StateGraph(State)
        builder.add_node("first", first)
        builder.add_node("second", second)
        builder.add_edge(START, "first")
        builder.add_edge("first", "second")
        builder.add_edge("second", END)
        with pytest.raises(AnalysisCancelled):
            builder.compile().invoke({"out": []})
        assert reached == []


# ---------------------------------------------------------------------------
# AnalysisService: a cancelled run ends as AnalysisCancelled, never AnalysisError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalysisServiceCancel:
    def test_a_cancel_the_core_swallowed_still_ends_cancelled(self):
        """Core code that catches the exception (like the structured-output
        fallback) and finishes anyway must not turn a cancel into a result."""
        event = threading.Event()

        class Graph:
            def __init__(self, *args, callbacks=None, **kwargs):
                self.model = FakeListChatModel(responses=["a"], callbacks=callbacks)

            def propagate(self, *args, **kwargs):
                event.set()
                with contextlib.suppress(Exception):
                    self.model.invoke("x")
                return {"final_trade_decision": "Buy"}, "Buy"

        with patch(GRAPH, Graph), pytest.raises(AnalysisCancelled):
            AnalysisService().run_analysis("AAPL", "2026-06-01", cancel_event=event)

    def test_a_wrapped_cancel_is_not_reported_as_a_failure(self):
        event = threading.Event()

        class Graph:
            def __init__(self, *args, callbacks=None, **kwargs):
                self.model = FakeListChatModel(responses=["a"], callbacks=callbacks)

            def propagate(self, *args, **kwargs):
                event.set()
                try:
                    self.model.invoke("x")
                except AnalysisCancelled as exc:
                    raise RuntimeError("node failed") from exc
                return {}, "Hold"

        with patch(GRAPH, Graph), pytest.raises(AnalysisCancelled):
            AnalysisService().run_analysis("AAPL", "2026-06-01", cancel_event=event)

    def test_a_real_failure_is_still_a_failure(self):
        event = threading.Event()
        with patch(GRAPH) as graph_cls, pytest.raises(AnalysisError):
            graph_cls.return_value.propagate.side_effect = RuntimeError("boom")
            AnalysisService().run_analysis("AAPL", "2026-06-01", cancel_event=event)


# ---------------------------------------------------------------------------
# TaskManager / TaskService
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskManagerCancel:
    def test_cancelling_a_queued_task_cancels_its_future(self):
        tm = TaskManager()
        task, _ = tm.create_task(_request())
        tm.update_task(task.task_id, status=TaskStatus.QUEUED)
        future = Future()
        tm.attach_future(task.task_id, future)
        assert tm.cancel_task(task.task_id) == CancelOutcome.CANCELLED
        assert future.cancelled()
        task = tm.get_task(task.task_id)
        assert task.status == TaskStatus.CANCELLED
        assert task.cancel_requested and task.completed_at is not None
        assert task.message == CANCELLED_BEFORE_START

    def test_a_future_attached_after_the_cancel_is_cancelled(self):
        tm = TaskManager()
        task, _ = tm.create_task(_request())
        tm.cancel_task(task.task_id)
        future = Future()
        tm.attach_future(task.task_id, future)
        assert future.cancelled()

    def test_a_cancelled_queued_task_never_runs_even_if_picked_up(self):
        """Backstop: a worker that picked the task up anyway skips the run."""
        tm = TaskManager()
        analysis = MagicMock()
        service = TaskService(task_manager=tm, analysis_service=analysis)
        task, _ = tm.create_task(_request())
        tm.update_task(task.task_id, status=TaskStatus.QUEUED)
        tm.cancel_task(task.task_id)
        service.execute_analysis(task.task_id, _request())
        analysis.run_analysis.assert_not_called()
        assert tm.get_task(task.task_id).status == TaskStatus.CANCELLED

    def test_a_deleted_queued_task_never_runs(self):
        tm = TaskManager()
        analysis = MagicMock()
        service = TaskService(task_manager=tm, analysis_service=analysis)
        task, _ = tm.create_task(_request())
        tm.update_task(task.task_id, status=TaskStatus.QUEUED)
        assert tm.delete_task(task.task_id)
        service.execute_analysis(task.task_id, _request())
        analysis.run_analysis.assert_not_called()

    def test_a_processing_task_gets_cancel_requested(self):
        tm = TaskManager()
        task, _ = tm.create_task(_request())
        event = tm.start_task(task.task_id)
        assert tm.cancel_task(task.task_id) == CancelOutcome.REQUESTED
        assert event.is_set()
        task = tm.get_task(task.task_id)
        assert task.status == TaskStatus.PROCESSING and task.cancel_requested

    def test_the_run_gets_the_cancel_event_and_ends_cancelled(self):
        tm = TaskManager()
        analysis = MagicMock()

        def run(**kwargs):
            tm.cancel_task(task.task_id)
            assert kwargs["cancel_event"].is_set()
            raise AnalysisCancelled("analysis cancelled")

        analysis.run_analysis.side_effect = run
        task, _ = tm.create_task(_request())
        TaskService(task_manager=tm, analysis_service=analysis).execute_analysis(
            task.task_id, _request()
        )
        task = tm.get_task(task.task_id)
        assert task.status == TaskStatus.CANCELLED
        assert task.error is None and task.result is None
        assert task.message == CANCELLED_WHILE_RUNNING
        assert task.completed_at is not None

    @pytest.mark.parametrize("final", [TaskStatus.COMPLETED, TaskStatus.FAILED])
    def test_finished_tasks_cannot_be_cancelled(self, final):
        tm = TaskManager()
        task, _ = tm.create_task(_request())
        tm.update_task(task.task_id, status=final)
        assert tm.cancel_task(task.task_id) == CancelOutcome.FINISHED
        assert tm.cancel_task("nope") == CancelOutcome.NOT_FOUND

    def test_cancelled_tasks_expire_like_finished_ones(self):
        tm = TaskManager(ttl_minutes=0)
        task, _ = tm.create_task(_request())
        tm.update_task(task.task_id, status=TaskStatus.QUEUED)
        tm.cancel_task(task.task_id)
        assert tm.cleanup() == 1
        assert tm.get_task(task.task_id) is None


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCancelEndpoint:
    def test_cancelling_a_running_task_stops_it_at_the_next_llm_call(self):
        first, proceed, log = threading.Event(), threading.Event(), []
        with patch(GRAPH, _scripted_graph(first, proceed, log)):
            app = create_app()
            client = TestClient(app)
            task_id = client.post(
                TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"}
            ).json()["task_id"]
            assert first.wait(5)
            resp = client.post(f"{TASKS}/{task_id}/cancel")
            assert resp.status_code == 200
            assert resp.json()["status"] == "processing"
            assert resp.json()["cancel_requested"] is True
            polled = client.get(f"{TASKS}/{task_id}").json()
            assert (polled["status"], polled["cancel_requested"]) == ("processing", True)
            proceed.set()
            app.state.task_worker.shutdown(wait=True)

            task = client.get(f"{TASKS}/{task_id}").json()
            listed = client.get(TASKS, params={"status": "cancelled"}).json()

        assert log == ["built", "first call"]  # the second LLM call never ran
        assert task["status"] == "cancelled"
        assert task["cancel_requested"] is True
        assert task["error"] is None and task["result"] is None
        assert task["message"] == CANCELLED_WHILE_RUNNING
        assert task["completed_at"] is not None
        assert [t["task_id"] for t in listed] == [task_id]

    def test_cancelling_a_queued_task_means_it_never_starts(self):
        first, proceed, log = threading.Event(), threading.Event(), []
        picked_up = []
        original = TaskService.execute_analysis

        def execute_analysis(self, task_id, request):
            picked_up.append(request.ticker)
            return original(self, task_id, request)

        with (
            patch(GRAPH, _scripted_graph(first, proceed, log)),
            patch.object(TaskService, "execute_analysis", execute_analysis),
        ):
            app = create_app(overrides={"task_max_concurrent": 1})
            client = TestClient(app)
            running = client.post(TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"})
            assert first.wait(5)
            queued = client.post(TASKS, json={"ticker": "MSFT", "trade_date": "2026-06-01"})
            queued_id = queued.json()["task_id"]
            assert client.get(f"{TASKS}/{queued_id}").json()["status"] == "queued"

            resp = client.post(f"{TASKS}/{queued_id}/cancel")
            # Queued behind the cancelled one: once it has run, the single
            # worker has passed the cancelled task's place in the queue.
            last = client.post(TASKS, json={"ticker": "NVDA", "trade_date": "2026-06-01"})
            proceed.set()
            for _ in range(500):
                if client.get(f"{TASKS}/{last.json()['task_id']}").json()["status"] == "completed":
                    break
                time.sleep(0.01)
            app.state.task_worker.shutdown(wait=True)
            after = client.get(f"{TASKS}/{queued_id}").json()
            first_task = client.get(f"{TASKS}/{running.json()['task_id']}").json()

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        assert resp.json()["message"] == CANCELLED_BEFORE_START
        assert after["status"] == "cancelled"
        # The worker never even picked the cancelled task up: its Future was cancelled.
        assert picked_up == ["AAPL", "NVDA"]
        assert log.count("built") == 2
        assert first_task["status"] == "completed"

    def test_finished_tasks_answer_409_and_unknown_ones_404(self):
        app = create_app()
        client = TestClient(app)
        tm: TaskManager = app.state.task_manager
        done, _ = tm.create_task(_request("AAPL"))
        tm.update_task(done.task_id, status=TaskStatus.COMPLETED)
        cancelled, _ = tm.create_task(_request("MSFT"))
        tm.cancel_task(cancelled.task_id)

        resp = client.post(f"{TASKS}/{done.task_id}/cancel")
        assert resp.status_code == 409
        assert "completed" in resp.json()["detail"]
        assert client.post(f"{TASKS}/{cancelled.task_id}/cancel").status_code == 409
        assert client.post(f"{TASKS}/nope/cancel").status_code == 404

    def test_a_task_being_cancelled_is_not_reused_for_a_new_request(self):
        first, proceed, log = threading.Event(), threading.Event(), []
        with patch(GRAPH, _scripted_graph(first, proceed, log)):
            app = create_app()
            client = TestClient(app)
            body = {"ticker": "AAPL", "trade_date": "2026-06-01"}
            old_id = client.post(TASKS, json=body).json()["task_id"]
            assert first.wait(5)
            client.post(f"{TASKS}/{old_id}/cancel")
            again = client.post(TASKS, json=body)
            proceed.set()
            app.state.task_worker.shutdown(wait=True)
        assert again.status_code == 202
        assert again.json()["task_id"] != old_id


@pytest.mark.unit
class TestDeleteCancels:
    def test_deleting_a_running_task_stops_it(self):
        first, proceed, log = threading.Event(), threading.Event(), []
        with patch(GRAPH, _scripted_graph(first, proceed, log)):
            app = create_app()
            client = TestClient(app)
            task_id = client.post(
                TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"}
            ).json()["task_id"]
            assert first.wait(5)
            assert client.delete(f"{TASKS}/{task_id}").status_code == 204
            assert client.get(f"{TASKS}/{task_id}").status_code == 404
            proceed.set()
            app.state.task_worker.shutdown(wait=True)
        assert log == ["built", "first call"]

    def test_deleting_a_finished_task_still_works(self):
        """fin-insight deletes a task after reading its result."""
        with patch(GRAPH) as graph_cls:
            graph_cls.return_value.propagate.return_value = ({"final_trade_decision": "Buy"}, "Buy")
            app = create_app()
            client = TestClient(app)
            task_id = client.post(
                TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"}
            ).json()["task_id"]
            app.state.task_worker.shutdown(wait=True)
            assert client.get(f"{TASKS}/{task_id}").json()["status"] == "completed"
            assert client.delete(f"{TASKS}/{task_id}").status_code == 204
            assert client.get(f"{TASKS}/{task_id}").status_code == 404
