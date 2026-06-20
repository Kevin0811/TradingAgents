"""Tests for the TradingAgents API /analyze router.

Covers:
- POST /analyze (synchronous) calls analysis_service and returns AnalyzeResponse
- POST /analyze/tasks creates a task and queues it for background execution
- GET /analyze/tasks lists tasks with filters
- GET /analyze/tasks/{task_id} returns task status
- DELETE /analyze/tasks/{task_id} removes a task
- 429 response when task queue is full
- 404 response when task not found
"""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tradingagents.api.app import create_app
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.domain.entities import AnalysisResult
from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus


# ---------------------------------------------------------------------------
# Task manager singleton reset fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_task_manager_singleton():
    """Reset the TaskManager singleton between tests so state doesn't leak."""
    import tradingagents.api.dependencies as deps
    original = deps._task_manager
    deps._task_manager = None
    yield
    deps._task_manager = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_response(task_id="test-id", status=TaskStatus.PENDING, ticker="AAPL"):
    return TaskResponse(
        task_id=task_id,
        status=status,
        ticker=ticker,
        trade_date="2026-06-01",
        asset_type="stock",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def _make_analysis_result():
    return AnalysisResult(
        ticker="AAPL",
        trade_date="2026-06-01",
        asset_type="stock",
        market_report="Market report",
        signal="Buy",
    )


# ---------------------------------------------------------------------------
# POST /analyze (synchronous)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeEndpoint:
    """POST /analyze must run the analysis and return the result."""

    def test_analyze_calls_service_and_returns(self, monkeypatch):
        """The endpoint must call analysis_service.run_analysis and return AnalyzeResponse."""
        mock_result = _make_analysis_result()

        # Patch at the AnalysisService class level
        with patch(
            "tradingagents.api.domain.services.analysis_service.AnalysisService.run_analysis",
            return_value=mock_result,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.post(
                "/api/v1/analyze",
                json={
                    "ticker": "AAPL",
                    "trade_date": "2026-06-01",
                    "asset_type": "stock",
                    "selected_analysts": ["market", "sentiment", "news", "fundamentals"],
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert body["signal"] == "Buy"
        assert body["market_report"] == "Market report"

    def test_analyze_converts_enum_to_string(self, monkeypatch):
        """The endpoint must convert enum asset_type to string for the service."""
        captured = {}

        def fake_run_analysis(self, **kwargs):
            captured.update(kwargs)
            return _make_analysis_result()

        with patch(
            "tradingagents.api.domain.services.analysis_service.AnalysisService.run_analysis",
            fake_run_analysis,
        ):
            app = create_app()
            client = TestClient(app)
            client.post(
                "/api/v1/analyze",
                json={
                    "ticker": "AAPL",
                    "trade_date": "2026-06-01",
                    "asset_type": "crypto",
                },
            )

        assert captured["asset_type"] == "crypto"


# ---------------------------------------------------------------------------
# POST /analyze/tasks (async task creation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateAnalysisTask:
    """POST /analyze/tasks must create a task and queue it."""

    def test_create_task_returns_202_accepted(self, monkeypatch):
        """New task must return 202 Accepted."""
        mock_task = _make_task_response(status=TaskStatus.PENDING)

        def fake_create_task(self, request):
            return mock_task, True

        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.create_task",
            fake_create_task,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.post(
                "/api/v1/analyze/tasks",
                json={
                    "ticker": "AAPL",
                    "trade_date": "2026-06-01",
                },
            )

        assert resp.status_code == 202

    def test_create_task_returns_200_for_duplicate(self, monkeypatch):
        """Existing active task must return 200 OK (not 202)."""
        mock_task = _make_task_response(status=TaskStatus.PROCESSING)

        def fake_create_task(self, request):
            return mock_task, False  # is_new=False

        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.create_task",
            fake_create_task,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.post(
                "/api/v1/analyze/tasks",
                json={
                    "ticker": "AAPL",
                    "trade_date": "2026-06-01",
                },
            )

        assert resp.status_code == 200

    def test_create_task_returns_429_when_queue_full(self, monkeypatch):
        """When task queue is full, must return 429."""
        # Patch TaskManager.total_task_count to return 100
        monkeypatch.setattr(
            "tradingagents.api.core.task_manager.TaskManager.total_task_count",
            property(lambda self: 100),
        )

        app = create_app(overrides={"task_max_tasks": 100})
        client = TestClient(app)
        resp = client.post(
            "/api/v1/analyze/tasks",
            json={
                "ticker": "AAPL",
                "trade_date": "2026-06-01",
            },
        )

        assert resp.status_code == 429
        body = resp.json()
        assert "Task queue is full" in body["error"]


# ---------------------------------------------------------------------------
# GET /analyze/tasks (list tasks)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListAnalysisTasks:
    """GET /analyze/tasks must return filtered task list."""

    def test_list_tasks_returns_all(self, monkeypatch):
        mock_tasks = [
            _make_task_response(task_id="1", status=TaskStatus.PENDING),
            _make_task_response(task_id="2", status=TaskStatus.COMPLETED),
        ]

        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.list_tasks",
            return_value=mock_tasks,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/analyze/tasks")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2

    def test_list_tasks_filters_by_status(self, monkeypatch):
        mock_tasks = [
            _make_task_response(task_id="1", status=TaskStatus.PENDING),
        ]

        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.list_tasks",
            return_value=mock_tasks,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/analyze/tasks", params={"status": "pending"})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["status"] == "pending"

    def test_list_tasks_filters_by_ticker(self, monkeypatch):
        mock_tasks = [
            _make_task_response(task_id="1", ticker="AAPL"),
        ]

        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.list_tasks",
            return_value=mock_tasks,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/analyze/tasks", params={"ticker": "AAPL"})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# GET /analyze/tasks/{task_id} (get task status)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAnalysisTask:
    """GET /analyze/tasks/{task_id} must return task or 404."""

    def test_get_task_returns_task(self, monkeypatch):
        mock_task = _make_task_response(task_id="test-id", status=TaskStatus.COMPLETED)

        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.get_task",
            return_value=mock_task,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/analyze/tasks/test-id")

        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "test-id"
        assert body["status"] == "completed"

    def test_get_task_returns_404_when_not_found(self, monkeypatch):
        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.get_task",
            return_value=None,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/analyze/tasks/nonexistent")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /analyze/tasks/{task_id} (delete task)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteAnalysisTask:
    """DELETE /analyze/tasks/{task_id} must remove task or return 404."""

    def test_delete_task_returns_204(self, monkeypatch):
        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.delete_task",
            return_value=True,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.delete("/api/v1/analyze/tasks/test-id")

        assert resp.status_code == 204

    def test_delete_task_returns_404_when_not_found(self, monkeypatch):
        with patch(
            "tradingagents.api.domain.services.task_service.TaskService.delete_task",
            return_value=False,
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.delete("/api/v1/analyze/tasks/nonexistent")

        assert resp.status_code == 404