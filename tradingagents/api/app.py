"""FastAPI application factory for the TradingAgents API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from tradingagents.api.config import ApiConfig, set_config
from tradingagents.api.core.error_handlers import register_exception_handlers
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.core.task_worker import TaskWorker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Ensures required directories exist and tears down the task worker on exit.
    """
    config: ApiConfig = app.state.api_config
    config.ensure_directories()
    logger.info("TradingAgents API started")
    try:
        yield
    finally:
        logger.info("TradingAgents API shutting down")
        # Don't wait: an in-flight analysis can take minutes and would
        # otherwise hold up shutdown.
        app.state.task_worker.shutdown(wait=False)


def create_app(overrides: dict[str, Any] | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        overrides: Optional dict to override default configuration.

    Returns:
        Configured FastAPI application.
    """
    config = ApiConfig(overrides)
    set_config(config)

    # Import routers here to avoid circular imports
    from tradingagents.api.routers.analysts import router as analysts_router
    from tradingagents.api.routers.analyze import router as analyze_router
    from tradingagents.api.routers.config import router as config_router
    from tradingagents.api.routers.data import router as data_router
    from tradingagents.api.routers.decisions import router as decisions_router

    api_version = config.api_version

    app = FastAPI(
        title=config.api_title,
        version=config.api_version,
        description="REST API for the TradingAgents multi-agent trading system",
        lifespan=lifespan,
        docs_url=f"/api/{api_version}/docs",
        redoc_url=f"/api/{api_version}/redoc",
        openapi_url=f"/api/{api_version}/openapi.json",
    )
    app.state.api_config = config

    # Task state and the worker pool are per-app, not module globals, so tests
    # and reloads never share state. Created here rather than in the lifespan
    # handler so they exist even when the app is used without one.
    app.state.task_manager = TaskManager(
        ttl_minutes=config.config.get("task_ttl_minutes", 60),
        max_tasks=config.config.get("task_max_tasks", 100),
    )
    app.state.task_worker = TaskWorker(
        max_workers=int(config.config.get("task_max_concurrent", 2)),
    )

    # Register global exception handlers
    register_exception_handlers(app)

    # Include routers with versioned prefix
    app.include_router(analyze_router, prefix=f"/api/{api_version}")
    app.include_router(analysts_router, prefix=f"/api/{api_version}")
    app.include_router(config_router, prefix=f"/api/{api_version}")
    app.include_router(data_router, prefix=f"/api/{api_version}")
    app.include_router(decisions_router, prefix=f"/api/{api_version}")

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": config.api_version}

    return app
