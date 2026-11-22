"""Stable failure codes and cancel reasons on analysis tasks (Plan 08-D2a).

``classify_error`` is a pure table over exception types and status codes; the
lifecycle tests check that a failed task carries ``error_code``, a cancelled one
``cancel_reason``, that both reach the response JSON and the terminal discussion
event, and that ``error`` / ``message`` read as before. Every exception is built
in memory: no network, no LLM.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

from tradingagents.api import config as api_config
from tradingagents.api.app import create_app
from tradingagents.api.core.discussion_log import DiscussionStore
from tradingagents.api.core.error_codes import (
    CANCEL_REASONS,
    ERROR_CODES,
    TA_CANCELLED_BEFORE_START,
    TA_CANCELLED_WHILE_RUNNING,
    TA_DATA_VENDOR_FAILED,
    TA_LLM_AUTH,
    TA_LLM_MODEL_NOT_FOUND,
    TA_LLM_RATE_LIMITED,
    TA_LLM_RESPONSE_INVALID,
    TA_LLM_TIMEOUT,
    TA_LLM_UNREACHABLE,
    TA_RUN_FAILED,
    classify_error,
)
from tradingagents.api.core.exceptions import AnalysisCancelled, AnalysisError
from tradingagents.api.core.safe_errors import error_text
from tradingagents.api.core.task_manager import (
    CANCELLED_BEFORE_START,
    CANCELLED_WHILE_RUNNING,
    TaskManager,
)
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.schemas.task import TaskCreateRequest, TaskStatus
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)

GRAPH = "tradingagents.api.domain.services.analysis_service.TradingAgentsGraph"
TASKS = "/api/v1/analyze/tasks"
REQ = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")


@pytest.fixture(autouse=True)
def _not_ollama(monkeypatch):
    monkeypatch.setitem(api_config.DEFAULT_CONFIG, "llm_provider", "openai")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=REQ)


def _status_error(cls, status: int):
    return cls(f"Error code: {status}", response=_response(status), body=None)


def _validation_error() -> ValidationError:
    class Decision(BaseModel):
        rating: int

    try:
        Decision.model_validate({"rating": "not a number"})
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


class _OllamaResponseError(Exception):
    """Stands in for ``ollama.ResponseError`` (not installed): a status_code attribute."""

    def __init__(self, error: str, status_code: int):
        super().__init__(error)
        self.status_code = status_code


class _YFRateLimitError(Exception):
    pass


_YFRateLimitError.__module__ = "yfinance.exceptions"


def _raised_from(outer: BaseException, inner: BaseException) -> BaseException:
    """``outer`` raised ``from inner``, as a real ``raise ... from`` leaves it."""
    try:
        try:
            raise inner
        except BaseException as caught:
            raise outer from caught
    except BaseException as exc:  # noqa: BLE001 - returned for inspection
        return exc


def _raised_during(outer: BaseException, inner: BaseException) -> BaseException:
    """``outer`` raised while handling ``inner`` (only ``__context__`` set)."""
    try:
        try:
            raise inner
        except BaseException:
            raise outer  # noqa: B904 - the implicit context is the point
    except BaseException as exc:  # noqa: BLE001
        return exc


# ---------------------------------------------------------------------------
# classify_error: the table
# ---------------------------------------------------------------------------

CASES = [
    # timeouts
    ("builtin TimeoutError", TimeoutError("timed out"), TA_LLM_TIMEOUT),
    # asyncio.TimeoutError is the builtin TimeoutError (Python 3.11+), so the case above covers it.
    ("httpx.ReadTimeout", httpx.ReadTimeout("read timed out", request=REQ), TA_LLM_TIMEOUT),
    ("httpx.ConnectTimeout", httpx.ConnectTimeout("connect timed out"), TA_LLM_TIMEOUT),
    ("openai.APITimeoutError", openai.APITimeoutError(request=REQ), TA_LLM_TIMEOUT),
    ("anthropic.APITimeoutError", anthropic.APITimeoutError(request=REQ), TA_LLM_TIMEOUT),
    # unreachable
    ("ConnectionRefusedError", ConnectionRefusedError(111, "refused"), TA_LLM_UNREACHABLE),
    ("ConnectionError", ConnectionError("reset"), TA_LLM_UNREACHABLE),
    ("httpx.ConnectError", httpx.ConnectError("all connection attempts failed"), TA_LLM_UNREACHABLE),
    ("openai.APIConnectionError", openai.APIConnectionError(request=REQ), TA_LLM_UNREACHABLE),
    ("anthropic.APIConnectionError", anthropic.APIConnectionError(request=REQ), TA_LLM_UNREACHABLE),
    # auth
    ("openai.AuthenticationError", _status_error(openai.AuthenticationError, 401), TA_LLM_AUTH),
    ("openai.PermissionDeniedError", _status_error(openai.PermissionDeniedError, 403), TA_LLM_AUTH),
    ("anthropic.AuthenticationError", _status_error(anthropic.AuthenticationError, 401), TA_LLM_AUTH),
    ("anthropic.PermissionDeniedError", _status_error(anthropic.PermissionDeniedError, 403), TA_LLM_AUTH),
    ("google-genai ClientError 401", genai_errors.ClientError(401, {}), TA_LLM_AUTH),
    ("httpx.HTTPStatusError 403",
     httpx.HTTPStatusError("forbidden", request=REQ, response=_response(403)), TA_LLM_AUTH),
    # model not found
    ("openai.NotFoundError", _status_error(openai.NotFoundError, 404), TA_LLM_MODEL_NOT_FOUND),
    ("anthropic.NotFoundError", _status_error(anthropic.NotFoundError, 404), TA_LLM_MODEL_NOT_FOUND),
    ("ollama ResponseError 404",
     _OllamaResponseError("model 'llama9' not found", 404), TA_LLM_MODEL_NOT_FOUND),
    ("ollama text, no status (last resort)",
     RuntimeError('model "llama9" not found, try pulling it first'), TA_LLM_MODEL_NOT_FOUND),
    # rate limited
    ("openai.RateLimitError", _status_error(openai.RateLimitError, 429), TA_LLM_RATE_LIMITED),
    ("anthropic.RateLimitError", _status_error(anthropic.RateLimitError, 429), TA_LLM_RATE_LIMITED),
    ("google-genai ClientError 429", genai_errors.ClientError(429, {}), TA_LLM_RATE_LIMITED),
    # response invalid
    ("pydantic ValidationError", _validation_error(), TA_LLM_RESPONSE_INVALID),
    ("OutputParserException", OutputParserException("not JSON"), TA_LLM_RESPONSE_INVALID),
    ("openai.APIResponseValidationError",
     openai.APIResponseValidationError(response=_response(200), body=None), TA_LLM_RESPONSE_INVALID),
    # data vendors
    ("NoMarketDataError", NoMarketDataError("AAPL", detail="empty"), TA_DATA_VENDOR_FAILED),
    ("VendorUnavailableError", VendorUnavailableError("throttled"), TA_DATA_VENDOR_FAILED),
    ("VendorNotConfiguredError", VendorNotConfiguredError("no key"), TA_DATA_VENDOR_FAILED),
    ("a yfinance exception", _YFRateLimitError("Too Many Requests"), TA_DATA_VENDOR_FAILED),
    # everything else
    ("RuntimeError", RuntimeError("boom"), TA_RUN_FAILED),
    ("ValueError", ValueError("bad"), TA_RUN_FAILED),
    ("KeyError", KeyError("final_trade_decision"), TA_RUN_FAILED),
    ("google-genai ServerError 500", genai_errors.ServerError(500, {}), TA_RUN_FAILED),
    ("openai 500", _status_error(openai.InternalServerError, 500), TA_RUN_FAILED),
    ("bare AnalysisError", AnalysisError("Analysis failed", detail="x"), TA_RUN_FAILED),
]


@pytest.mark.unit
class TestClassifyError:
    @pytest.mark.parametrize(("exc", "code"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
    def test_table(self, exc, code):
        assert classify_error(exc) == code

    def test_every_code_is_covered_and_listed(self):
        assert {c[2] for c in CASES} == set(ERROR_CODES)
        assert len(set(ERROR_CODES)) == len(ERROR_CODES) == 8
        assert CANCEL_REASONS == (TA_CANCELLED_BEFORE_START, TA_CANCELLED_WHILE_RUNNING)

    def test_the_cause_chain_is_walked(self):
        """AnalysisService raises AnalysisError *from* the core's exception."""
        inner = _raised_from(RuntimeError("node failed"), openai.APITimeoutError(request=REQ))
        exc = _raised_from(AnalysisError("Analysis failed", detail="node failed"), inner)
        assert classify_error(exc) == TA_LLM_TIMEOUT

    def test_the_implicit_context_is_walked(self):
        exc = _raised_during(KeyError("x"), _status_error(openai.RateLimitError, 429))
        assert classify_error(exc) == TA_LLM_RATE_LIMITED

    def test_the_outermost_typed_match_wins(self):
        exc = _raised_from(OutputParserException("bad"), httpx.ConnectError("refused"))
        assert classify_error(exc) == TA_LLM_RESPONSE_INVALID

    def test_types_beat_message_text(self):
        """Text is the last resort: a typed match anywhere in the chain wins."""
        exc = _raised_from(
            RuntimeError("model 'x' not found"), _status_error(openai.RateLimitError, 429)
        )
        assert classify_error(exc) == TA_LLM_RATE_LIMITED

    def test_a_cycle_ends(self):
        a, b = RuntimeError("a"), RuntimeError("b")
        a.__context__, b.__context__ = b, a
        assert classify_error(a) == TA_RUN_FAILED

    def test_a_broken_str_is_no_match(self):
        class Weird(Exception):
            def __str__(self):
                raise ValueError("no text")

        assert classify_error(Weird()) == TA_RUN_FAILED


# ---------------------------------------------------------------------------
# The task lifecycle (TaskService + TaskManager + discussion log)
# ---------------------------------------------------------------------------


def _request(ticker="AAPL"):
    return TaskCreateRequest(ticker=ticker, trade_date="2026-06-01")


def _run(analysis, store=None):
    store = store if store is not None else DiscussionStore()
    tm = TaskManager(discussions=store)
    task, _ = tm.create_task(_request())
    TaskService(task_manager=tm, analysis_service=analysis, discussions=store).execute_analysis(
        task.task_id, _request()
    )
    return tm.get_task(task.task_id), store.get(task.task_id)


@pytest.mark.unit
class TestLifecycle:
    def test_a_failed_analysis_gets_an_error_code_and_the_same_error(self):
        analysis = MagicMock()
        cause = openai.APITimeoutError(request=REQ)
        analysis.run_analysis.side_effect = _raised_from(
            AnalysisError("Analysis failed for AAPL on 2026-06-01", detail=error_text(cause)),
            cause,
        )
        task, log = _run(analysis)
        assert task.status == TaskStatus.FAILED
        assert task.error == "Request timed out."   # error_text(cause), as before
        assert task.error_code == TA_LLM_TIMEOUT
        assert task.cancel_reason is None
        assert log.read()[-1]["data"] == {"status": "failed", "error_code": TA_LLM_TIMEOUT}

    def test_an_unexpected_failure_is_ta_run_failed(self):
        analysis = MagicMock()
        analysis.run_analysis.side_effect = RuntimeError("worker blew up")
        task, log = _run(analysis)
        assert (task.status, task.error) == (TaskStatus.FAILED, "worker blew up")
        assert task.error_code == TA_RUN_FAILED
        assert log.read()[-1]["data"] == {"status": "failed", "error_code": TA_RUN_FAILED}

    def test_the_real_analysis_service_keeps_the_cause_for_the_classifier(self):
        cause = _status_error(openai.AuthenticationError, 401)
        with patch(GRAPH) as graph_cls:
            graph_cls.return_value.propagate.side_effect = cause
            task, _ = _run(AnalysisService())
        assert task.status == TaskStatus.FAILED
        assert task.error == error_text(cause)
        assert task.error_code == TA_LLM_AUTH

    def test_a_cancel_before_start(self):
        store = DiscussionStore()
        tm = TaskManager(discussions=store)
        task, _ = tm.create_task(_request())
        tm.cancel_task(task.task_id)
        task = tm.get_task(task.task_id)
        assert task.status == TaskStatus.CANCELLED
        assert task.message == CANCELLED_BEFORE_START
        assert task.cancel_reason == TA_CANCELLED_BEFORE_START
        assert task.error is None and task.error_code is None
        assert store.get(task.task_id).read()[-1]["data"] == {
            "status": "cancelled", "cancel_reason": TA_CANCELLED_BEFORE_START,
        }

    def test_a_cancel_while_running(self):
        store = DiscussionStore()
        tm = TaskManager(discussions=store)
        task, _ = tm.create_task(_request())
        analysis = MagicMock()

        def run(**kwargs):
            tm.cancel_task(task.task_id)
            raise AnalysisCancelled("analysis cancelled")

        analysis.run_analysis.side_effect = run
        TaskService(task_manager=tm, analysis_service=analysis, discussions=store).execute_analysis(
            task.task_id, _request()
        )
        done = tm.get_task(task.task_id)
        assert done.status == TaskStatus.CANCELLED
        assert done.message == CANCELLED_WHILE_RUNNING
        assert done.cancel_reason == TA_CANCELLED_WHILE_RUNNING
        assert done.error is None and done.error_code is None
        assert store.get(task.task_id).read()[-1]["data"] == {
            "status": "cancelled", "cancel_reason": TA_CANCELLED_WHILE_RUNNING,
        }

    def test_a_completed_task_has_neither(self):
        analysis = MagicMock()
        analysis.run_analysis.return_value.to_dict.return_value = {"signal": "Buy"}
        analysis.run_analysis.return_value.token_usage = None
        task, log = _run(analysis)
        assert task.status == TaskStatus.COMPLETED
        assert task.error_code is None and task.cancel_reason is None
        assert log.read()[-1]["data"] == {"status": "completed"}


# ---------------------------------------------------------------------------
# Over HTTP: the response JSON and the OpenAPI schema
# ---------------------------------------------------------------------------


class _FailingGraph:
    def __init__(self, *args, **kwargs):
        pass

    def propagate(self, *args, **kwargs):
        raise _status_error(openai.RateLimitError, 429)


class _HeldGraph:
    """Completes, but only once ``release`` is set (keeps a single worker busy)."""

    started = threading.Event()
    release = threading.Event()

    def __init__(self, *args, **kwargs):
        pass

    def propagate(self, *args, **kwargs):
        type(self).started.set()
        assert type(self).release.wait(10)
        return {"final_trade_decision": "Buy"}, "Buy"


class TestResponseJson:
    def _finish(self, client, app, task_id):
        app.state.task_worker.shutdown(wait=True)
        return client.get(f"{TASKS}/{task_id}").json()

    def test_a_failed_task(self, tmp_path):
        with patch(GRAPH, _FailingGraph):
            app = create_app(overrides={"results_dir": str(tmp_path)})
            client = TestClient(app)
            task_id = client.post(TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"}).json()[
                "task_id"
            ]
            body = self._finish(client, app, task_id)
            listed = client.get(TASKS).json()
        assert body["status"] == "failed"
        assert body["error"] == "Error code: 429"
        assert body["error_code"] == TA_LLM_RATE_LIMITED
        assert body["cancel_reason"] is None
        assert listed[0]["error_code"] == TA_LLM_RATE_LIMITED

    def test_a_completed_and_a_cancelled_task(self, tmp_path):
        _HeldGraph.started.clear()
        _HeldGraph.release.clear()
        with patch(GRAPH, _HeldGraph):
            app = create_app(overrides={"results_dir": str(tmp_path), "task_max_concurrent": 1})
            client = TestClient(app)
            running = client.post(TASKS, json={"ticker": "AAPL", "trade_date": "2026-06-01"})
            assert _HeldGraph.started.wait(5)
            queued = client.post(TASKS, json={"ticker": "MSFT", "trade_date": "2026-06-01"})
            cancelled = client.post(f"{TASKS}/{queued.json()['task_id']}/cancel").json()
            _HeldGraph.release.set()
            completed = self._finish(client, app, running.json()["task_id"])
        assert cancelled["status"] == "cancelled"
        assert cancelled["message"] == CANCELLED_BEFORE_START
        assert cancelled["cancel_reason"] == TA_CANCELLED_BEFORE_START
        assert cancelled["error_code"] is None
        assert completed["status"] == "completed"
        assert "error_code" in completed and "cancel_reason" in completed
        assert completed["error_code"] is None and completed["cancel_reason"] is None

    def test_the_openapi_schema_lists_the_fields_and_codes(self, tmp_path):
        schema = create_app(overrides={"results_dir": str(tmp_path)}).openapi()
        props = schema["components"]["schemas"]["TaskResponse"]["properties"]
        for code in ERROR_CODES:
            assert code in props["error_code"]["description"]
        for reason in CANCEL_REASONS:
            assert reason in props["cancel_reason"]["description"]
