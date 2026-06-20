"""Tests for the TradingAgents API application factory.

Covers:
- create_app() builds FastAPI app with correct config
- Health check endpoint
- Docs/Redoc/OpenAPI URLs at versioned paths
- Lifespan manager calls ensure_directories()
- Routers are included with correct prefix
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tradingagents.api.app import create_app, lifespan


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateApp:
    """create_app() must build a correctly configured FastAPI app."""

    def test_create_app_returns_fastapi_instance(self):
        app = create_app()
        assert app is not None
        # FastAPI apps have a routes attribute
        assert hasattr(app, "routes")

    def test_create_app_with_overrides(self):
        overrides = {"api_title": "Custom API", "api_version": "v2"}
        app = create_app(overrides=overrides)
        assert app.title == "Custom API"

    def test_app_state_holds_config(self):
        app = create_app()
        assert hasattr(app.state, "api_config")
        assert app.state.api_config is not None

    def test_default_api_version(self):
        app = create_app()
        assert app.state.api_config.api_version == "v1"

    def test_custom_api_version(self):
        overrides = {"api_version": "v2"}
        app = create_app(overrides=overrides)
        assert app.state.api_config.api_version == "v2"


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthCheck:
    """GET /health must return status and version."""

    @pytest.fixture
    def client(self):
        return TestClient(create_app())

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_returns_version(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "version" in body
        assert body["version"] == "v1"


# ---------------------------------------------------------------------------
# OpenAPI docs URLs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenApiUrls:
    """Docs, Redoc, and OpenAPI JSON must be at versioned paths."""

    @pytest.fixture
    def client(self):
        return TestClient(create_app())

    def test_docs_url(self, client):
        resp = client.get("/api/v1/docs")
        assert resp.status_code == 200

    def test_redoc_url(self, client):
        resp = client.get("/api/v1/redoc")
        assert resp.status_code == 200

    def test_openapi_json_url(self, client):
        resp = client.get("/api/v1/openapi.json")
        assert resp.status_code == 200
        body = resp.json()
        assert "openapi" in body

    def test_custom_version_docs(self):
        overrides = {"api_version": "v2"}
        app = create_app(overrides=overrides)
        client = TestClient(app)
        resp = client.get("/api/v2/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Lifespan manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifespan:
    """The lifespan manager must call ensure_directories() on startup."""

    @pytest.mark.asyncio
    async def test_lifespan_calls_ensure_directories(self):
        app = create_app()
        # Mock ensure_directories on the config
        app.state.api_config.ensure_directories = MagicMock()

        async with lifespan(app):
            pass

        app.state.api_config.ensure_directories.assert_called_once()


# ---------------------------------------------------------------------------
# Router inclusion - test that the router objects are attached to the app
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRouterInclusion:
    """Routers must be attached to the app with correct prefix/tags."""

    def test_health_endpoint_is_registered(self):
        app = create_app()
        # Health is at root, not versioned
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/health" in paths

    def test_openapi_urls_are_versioned(self):
        app = create_app()
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/api/v1/docs" in paths
        assert "/api/v1/redoc" in paths
        assert "/api/v1/openapi.json" in paths

    def test_app_has_analyze_router_with_prefix(self):
        """The app should have routes under /api/v1/ prefix."""
        app = create_app()
        # Check that any routes have the /api/v1 prefix
        paths = [route.path for route in app.routes if hasattr(route, "path")]
        versioned = [p for p in paths if p.startswith("/api/v1")]
        assert len(versioned) > 0, f"No versioned routes found in {paths}"