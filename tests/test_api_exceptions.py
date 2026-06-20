"""Tests for the TradingAgents API exception hierarchy and error handlers.

Covers:
- Exception class hierarchy (TradingAgentsAPIError and subclasses)
- Exception handler status-code mapping
- JSON response body shape
- Exception handler registration on FastAPI app
- Uncaught exception fallback
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from tradingagents.api.core.exceptions import (
    AnalysisError,
    DataNotFoundError,
    ExternalServiceError,
    InvalidRequestError,
    TradingAgentsAPIError,
)
from tradingagents.api.core.error_handlers import (
    generic_exception_handler,
    register_exception_handlers,
    trading_agents_api_error_handler,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExceptionHierarchy:
    """All custom exceptions must derive from TradingAgentsAPIError."""

    def test_base_exception_has_message_and_detail(self):
        exc = TradingAgentsAPIError("something broke", detail="root cause")
        assert exc.message == "something broke"
        assert exc.detail == "root cause"
        assert str(exc) == "something broke"

    def test_analysis_error_is_subclass(self):
        assert issubclass(AnalysisError, TradingAgentsAPIError)

    def test_data_not_found_error_is_subclass(self):
        assert issubclass(DataNotFoundError, TradingAgentsAPIError)

    def test_invalid_request_error_is_subclass(self):
        assert issubclass(InvalidRequestError, TradingAgentsAPIError)

    def test_external_service_error_is_subclass(self):
        assert issubclass(ExternalServiceError, TradingAgentsAPIError)


# ---------------------------------------------------------------------------
# Error handler: status code mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorHandlerStatusMapping:
    """The handler must map exception types to HTTP status codes."""

    @pytest.fixture
    def mock_request(self):
        req = MagicMock(spec=Request)
        req.method = "GET"
        req.url.path = "/test"
        return req

    async def _call_handler(self, exc, mock_request):
        """Helper to call the handler and decode the response body."""
        resp = await trading_agents_api_error_handler(mock_request, exc)
        # JSONResponse.body is bytes; decode and parse JSON
        content = json.loads(resp.body.decode("utf-8"))
        return resp.status_code, content

    @pytest.mark.asyncio
    async def test_data_not_found_returns_404(self, mock_request):
        exc = DataNotFoundError("not found")
        status_code, content = await self._call_handler(exc, mock_request)
        assert status_code == status.HTTP_404_NOT_FOUND
        assert content["error"] == "not found"

    @pytest.mark.asyncio
    async def test_analysis_error_returns_500(self, mock_request):
        exc = AnalysisError("analysis failed")
        status_code, content = await self._call_handler(exc, mock_request)
        assert status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert content["error"] == "analysis failed"

    @pytest.mark.asyncio
    async def test_external_service_error_returns_503(self, mock_request):
        exc = ExternalServiceError("LLM timeout")
        status_code, content = await self._call_handler(exc, mock_request)
        assert status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert content["error"] == "LLM timeout"

    @pytest.mark.asyncio
    async def test_invalid_request_returns_400(self, mock_request):
        exc = InvalidRequestError("bad input")
        status_code, content = await self._call_handler(exc, mock_request)
        assert status_code == status.HTTP_400_BAD_REQUEST
        assert content["error"] == "bad input"

    @pytest.mark.asyncio
    async def test_generic_trading_error_returns_400(self, mock_request):
        exc = TradingAgentsAPIError("generic error")
        status_code, content = await self._call_handler(exc, mock_request)
        assert status_code == status.HTTP_400_BAD_REQUEST
        assert content["error"] == "generic error"


# ---------------------------------------------------------------------------
# Error handler: response body
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorHandlerResponseBody:
    """Response body must include error message and detail."""

    @pytest.fixture
    def mock_request(self):
        req = MagicMock(spec=Request)
        req.method = "GET"
        req.url.path = "/test"
        return req

    @pytest.mark.asyncio
    async def test_body_includes_error_and_detail(self, mock_request):
        exc = AnalysisError("boom", detail="stack trace here")
        resp = await trading_agents_api_error_handler(mock_request, exc)
        body = json.loads(resp.body.decode("utf-8"))
        assert body["error"] == "boom"
        assert body["detail"] == "stack trace here"

    @pytest.mark.asyncio
    async def test_detail_can_be_none(self, mock_request):
        exc = DataNotFoundError("missing")
        resp = await trading_agents_api_error_handler(mock_request, exc)
        body = json.loads(resp.body.decode("utf-8"))
        assert body["error"] == "missing"
        assert body["detail"] is None


# ---------------------------------------------------------------------------
# Error handler: logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorHandlerLogging:
    """Errors must be logged with method + path context."""

    @pytest.mark.asyncio
    async def test_logs_error_with_context(self, caplog):
        req = MagicMock(spec=Request)
        req.method = "POST"
        req.url.path = "/analyze"
        exc = AnalysisError("failed", detail="LLM error")

        with caplog.at_level(logging.ERROR, logger="tradingagents.api.core.error_handlers"):
            await trading_agents_api_error_handler(req, exc)

        assert "failed" in caplog.text
        assert "LLM error" in caplog.text
        assert "POST" in caplog.text
        assert "/analyze" in caplog.text


# ---------------------------------------------------------------------------
# Generic exception handler
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenericExceptionHandler:
    """Uncaught exceptions must return 500 with generic message."""

    @pytest.fixture
    def mock_request(self):
        req = MagicMock(spec=Request)
        req.method = "GET"
        req.url.path = "/test"
        return req

    @pytest.mark.asyncio
    async def test_returns_500(self, mock_request):
        exc = RuntimeError("unexpected crash")
        resp = await generic_exception_handler(mock_request, exc)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_body_has_generic_message(self, mock_request):
        exc = RuntimeError("unexpected crash")
        resp = await generic_exception_handler(mock_request, exc)
        body = json.loads(resp.body.decode("utf-8"))
        assert body["error"] == "Internal server error"
        assert "unexpected crash" in body["detail"]


# ---------------------------------------------------------------------------
# Exception handler registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExceptionHandlerRegistration:
    """register_exception_handlers() must attach handlers to the app."""

    def test_registers_handlers_on_app(self):
        app = FastAPI()
        register_exception_handlers(app)

        # FastAPI stores exception handlers in app.exception_handlers
        handlers = app.exception_handlers
        assert TradingAgentsAPIError in handlers
        assert AnalysisError in handlers
        assert DataNotFoundError in handlers
        assert ExternalServiceError in handlers
        assert InvalidRequestError in handlers
        assert Exception in handlers


# ---------------------------------------------------------------------------
# Integration: FastAPI app with exception handlers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExceptionIntegration:
    """End-to-end: FastAPI app must return proper HTTP responses on errors."""

    @pytest.fixture
    def app(self):
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/raise-data-not-found")
        async def raise_data_not_found():
            raise DataNotFoundError("ticker not found")

        @app.get("/raise-analysis-error")
        async def raise_analysis_error():
            raise AnalysisError("analysis failed")

        @app.get("/raise-external-error")
        async def raise_external_error():
            raise ExternalServiceError("LLM unavailable")

        @app.get("/raise-generic")
        async def raise_generic():
            raise RuntimeError("unexpected")

        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app, raise_server_exceptions=False)

    def test_data_not_found_returns_404(self, client):
        resp = client.get("/raise-data-not-found")
        assert resp.status_code == 404
        body = resp.json()
        assert "ticker not found" in body["error"]

    def test_analysis_error_returns_500(self, client):
        resp = client.get("/raise-analysis-error")
        assert resp.status_code == 500
        body = resp.json()
        assert "analysis failed" in body["error"]

    def test_external_error_returns_503(self, client):
        resp = client.get("/raise-external-error")
        assert resp.status_code == 503
        body = resp.json()
        assert "LLM unavailable" in body["error"]

    def test_generic_error_returns_500(self, client):
        resp = client.get("/raise-generic")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "Internal server error"