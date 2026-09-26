"""FastAPI-based REST API for TradingAgents.

This module provides HTTP endpoints for:
- Running full trading analysis pipelines
- Accessing individual analyst agents
- Querying market data and fundamentals
- Retrieving structured trading decisions
- Listing the supported symbols (``/symbols``) that analysis requests are
  validated against

Supported symbols:
    A positive list of the Yahoo Finance symbols TradingAgents accepts, per
    market: ``tw`` / ``us`` / ``jp`` equities and ETFs (yfinance screener;
    equities with a market cap > 0; US OTC venues excluded unless
    ``TRADINGAGENTS_SYMBOLS_INCLUDE_OTC=true``), ``crypto`` (``BTC-USD``) and
    ``fx`` (``TWD=X`` = USD/TWD) from ``yf.Lookup``. Endpoints, under
    ``/api/{version}``:

    - ``GET /symbols?market=&q=&type=&limit=&offset=`` list/search one market
    - ``GET /symbols/{symbol}`` exact lookup (404 with suggestions, 503 while
      the symbol's market list is not loaded)
    - ``POST /symbols/refresh[?market=]`` queue a background refresh (202)

    Lists are cached as ``<cache dir>/<market>.json`` (default
    ``<data_cache_dir>/symbols``, i.e. ``~/.tradingagents/cache/symbols``) and
    loaded at startup; missing or stale ones refresh in the background, one
    market at a time, and a failed refresh keeps the previous list. Requests
    never wait on the network. Settings (env var -> config key):

    - ``TRADINGAGENTS_SYMBOLS_CACHE_DIR`` -> ``symbols_cache_dir``
    - ``TRADINGAGENTS_SYMBOLS_CACHE_TTL_DAYS`` -> ``symbols_cache_ttl_days`` (7)
    - ``TRADINGAGENTS_SYMBOLS_AUTO_REFRESH`` -> ``symbols_auto_refresh`` (true)
    - ``TRADINGAGENTS_SYMBOLS_INCLUDE_OTC`` -> ``symbols_include_otc`` (false)
    - ``TRADINGAGENTS_SYMBOLS_PAGE_DELAY_SECONDS`` ->
      ``symbols_page_delay_seconds`` (1.0)
    - ``TRADINGAGENTS_SYMBOLS_MAX_PAGES`` -> ``symbols_max_pages`` (200 pages
      of 250 rows per screener query)

    ``POST /analyze`` and ``POST /analyze/tasks`` reject a ticker that is not
    on its market's list (422, ``type: ticker_not_supported``, ``ctx`` with
    ``suggestions``). Tickers of a market whose list is not loaded, or that no
    list covers (``0700.HK``, ``^GSPC``, ``GC=F``), get the shape check only.

Architecture:
    The API follows Clean Architecture principles with the following layers:
    - Core: Global exceptions, error handlers, middlewares
    - Domain: Business logic (services), entities, repository interfaces
    - Infrastructure: LLM provider, agent factory, repository implementations
    - Interface: FastAPI routers, request/response schemas

Usage:
    from tradingagents.api import create_app

    app = create_app()
    # Run with uvicorn:
    # uvicorn tradingagents.api:create_app --factory
"""

from tradingagents.api.app import create_app

# Core exports
from tradingagents.api.core.exceptions import (
    AnalysisError,
    DataNotFoundError,
    TradingAgentsAPIError,
)

# Task management exports
from tradingagents.api.core.task_manager import TaskManager

# Domain exports
from tradingagents.api.domain.entities import (
    AnalysisResult,
    AnalystReport,
    DecisionState,
    MarketData,
)
from tradingagents.api.domain.repositories import StateRepository

# Service exports
from tradingagents.api.domain.services import (
    AnalysisService,
    AnalystService,
    DecisionService,
    MarketDataService,
    TaskService,
)

# Infrastructure exports
from tradingagents.api.infrastructure import (
    AgentFactory,
    FileStateRepository,
    LLMProviderFactory,
)
from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus

__all__ = [
    # App factory
    "create_app",
    # Exceptions
    "AnalysisError",
    "DataNotFoundError",
    "TradingAgentsAPIError",
    # Entities
    "AnalysisResult",
    "AnalystReport",
    "DecisionState",
    "MarketData",
    # Repository interfaces
    "StateRepository",
    # Services
    "AnalysisService",
    "AnalystService",
    "DecisionService",
    "MarketDataService",
    "TaskService",
    # Infrastructure
    "AgentFactory",
    "FileStateRepository",
    "LLMProviderFactory",
    # Task management
    "TaskManager",
    "TaskCreateRequest",
    "TaskResponse",
    "TaskStatus",
]
