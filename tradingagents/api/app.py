"""FastAPI application factory for the TradingAgents API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from tradingagents.api.config import ApiConfig, set_config
from tradingagents.api.core.activity_middleware import (
    YahooActivityMiddleware,
    yahoo_backed_paths,
)
from tradingagents.api.core.error_handlers import register_exception_handlers
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.core.task_worker import TaskWorker
from tradingagents.api.domain.refresh_window import RefreshWindow
from tradingagents.api.domain.services.refresh_activity import ActivityMonitor, RefreshGate
from tradingagents.api.domain.services.symbol_catalog import SymbolCatalog, set_active_catalog
from tradingagents.api.domain.services.symbol_settings import (
    SETTING_SPECS,
    SymbolSettings,
    validate_setting,
)
from tradingagents.api.infrastructure.repositories.file_symbol_cache_repository import (
    FileSymbolCacheRepository,
)
from tradingagents.api.infrastructure.yahoo_symbol_source import YahooSymbolSource

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Ensures required directories exist, loads the supported-symbols cache
    (queueing a background refresh for missing/stale markets; startup never
    waits on the network), and tears down the task worker on exit.
    """
    config: ApiConfig = app.state.api_config
    config.ensure_directories()
    app.state.symbol_catalog.start()
    logger.info("TradingAgents API started")
    try:
        yield
    finally:
        logger.info("TradingAgents API shutting down")
        app.state.symbol_catalog.shutdown()
        # Don't wait: an in-flight analysis can take minutes and would
        # otherwise hold up shutdown.
        app.state.task_worker.shutdown(wait=False)


def create_symbol_settings(config: ApiConfig) -> SymbolSettings:
    """Build the refresher settings: config/env values plus persisted API overrides.

    Raises:
        ValueError: A configured value is invalid (e.g. a malformed window);
            the message names the env var or config key it came from.
    """
    base, sources = {}, {}
    for name, spec in SETTING_SPECS.items():
        try:
            base[name] = validate_setting(name, config.config.get(spec.config_key))
        except ValueError as exc:
            raise ValueError(f"invalid {config.origin_of(spec.config_key)}: {exc}") from exc
        sources[name] = config.source_of(spec.config_key)
    try:
        RefreshWindow.parse(base["refresh_window"], base["refresh_timezone"])
    except ValueError as exc:
        window_key = SETTING_SPECS["refresh_window"].config_key
        zone_key = SETTING_SPECS["refresh_timezone"].config_key
        raise ValueError(
            f"invalid {config.origin_of(window_key)} / {config.origin_of(zone_key)}: {exc}"
        ) from exc
    return SymbolSettings(base, sources, store_dir=config.symbols_cache_dir)


def create_symbol_catalog(
    config: ApiConfig,
    settings: SymbolSettings | None = None,
    monitor: ActivityMonitor | None = None,
) -> SymbolCatalog:
    """Build the supported-symbols catalog from the API config."""
    config_values = config.config
    settings = settings or create_symbol_settings(config)
    idle_check = monitor.is_idle if monitor is not None else None
    gate = RefreshGate(idle_check) if idle_check is not None else None
    return SymbolCatalog(
        repository=FileSymbolCacheRepository(config.symbols_cache_dir),
        source=YahooSymbolSource(
            page_delay_seconds=lambda: settings.page_delay_seconds,
            max_pages=int(config_values.get("symbols_max_pages", 200)),
            include_otc=bool(config_values.get("symbols_include_otc", False)),
            gate=gate,
        ),
        settings=settings,
        idle_check=idle_check,
        gate=gate,
    )


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
    from tradingagents.api.routers.symbols import router as symbols_router

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

    # The supported-symbols list. Nothing is loaded until the lifespan runs;
    # until then (and for any market whose list never loads) request
    # validation falls back to the ticker shape check. Its automatic refreshes
    # wait until TradingAgents is idle: no active analysis task and no recent
    # Yahoo-backed request (tracked by the middleware below).
    app.state.symbol_settings = create_symbol_settings(config)
    task_manager = app.state.task_manager
    app.state.activity_monitor = ActivityMonitor(
        busy_tasks=lambda: task_manager.active_task_count,
        grace=lambda: app.state.symbol_settings.idle_grace,
    )
    app.state.symbol_catalog = create_symbol_catalog(
        config, app.state.symbol_settings, app.state.activity_monitor
    )
    set_active_catalog(app.state.symbol_catalog)
    app.add_middleware(
        YahooActivityMiddleware,
        monitor=app.state.activity_monitor,
        is_tracked=yahoo_backed_paths(f"/api/{api_version}"),
    )

    # Register global exception handlers
    register_exception_handlers(app)

    # Include routers with versioned prefix
    app.include_router(analyze_router, prefix=f"/api/{api_version}")
    app.include_router(analysts_router, prefix=f"/api/{api_version}")
    app.include_router(config_router, prefix=f"/api/{api_version}")
    app.include_router(data_router, prefix=f"/api/{api_version}")
    app.include_router(decisions_router, prefix=f"/api/{api_version}")
    app.include_router(symbols_router, prefix=f"/api/{api_version}")

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": config.api_version}

    return app
