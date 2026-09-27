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
    - ``GET /symbols/check?ticker=&asset_type=`` the analyze validator's
      decision for one ticker (always 200: ``supported``, ``enforced``,
      ``status``, ``suggestions``, ``entry``, ``last_error``)
    - ``GET /symbols/{symbol}`` plain exact lookup (404 with suggestions and
      the list status)
    - ``POST /symbols/refresh[?market=][&force=true]`` queue a background
      refresh (202 with ``queued`` / ``skipped``; a market fetched within the
      last 30 minutes is skipped unless forced)

    Lists are cached as ``<cache dir>/<market>.json`` (default
    ``<data_cache_dir>/symbols``, i.e. ``~/.tradingagents/cache/symbols``) and
    loaded at startup; missing or stale ones refresh in the background, one
    market at a time. A failed refresh keeps the previous list, and so does a
    refresh that looks truncated (under 80% of the previous entries, or an
    exchange/type group gone empty); either sets ``last_error``. A miss in a
    list older than a day queues a (rate-limited) refresh of that market, so
    new listings show up. Requests never wait on the network. Settings (env
    var -> config key):

    - ``TRADINGAGENTS_SYMBOLS_CACHE_DIR`` -> ``symbols_cache_dir``
    - ``TRADINGAGENTS_SYMBOLS_CACHE_TTL_DAYS`` -> ``symbols_cache_ttl_days``
      (7.0; fractions such as 0.5 work)
    - ``TRADINGAGENTS_SYMBOLS_AUTO_REFRESH`` -> ``symbols_auto_refresh`` (true)
    - ``TRADINGAGENTS_SYMBOLS_INCLUDE_OTC`` -> ``symbols_include_otc`` (false)
    - ``TRADINGAGENTS_SYMBOLS_PAGE_DELAY_SECONDS`` ->
      ``symbols_page_delay_seconds`` (1.0)
    - ``TRADINGAGENTS_SYMBOLS_MAX_PAGES`` -> ``symbols_max_pages`` (200 pages
      of 250 rows per screener query; a query that needs more fails the
      refresh instead of saving a partial list)

    ``POST /analyze`` and ``POST /analyze/tasks`` check the ticker per market
    (``REJECT_UNLISTED_TICKERS`` in ``tradingagents.api.domain.symbols``): a
    ticker missing from the loaded ``tw`` or ``jp`` list is rejected (422,
    ``type: ticker_not_supported``, ``ctx`` with ``suggestions``), as is a
    suffix-less TW/JP code whose ``.TW`` / ``.TWO`` / ``.T`` symbol is listed
    (``2330``, ``130A``). A miss in the ``us``, ``crypto`` or ``fx`` list is
    allowed; its suggestions are only a hint in ``GET /symbols/check``.
    Tickers of a market whose list is not loaded, or that no list covers
    (``0700.HK``, ``^GSPC``, ``GC=F``, ``SAP.F``), get the shape check only.
    Only ``.TW`` / ``.TWO`` (tw) and ``.T`` (jp) route a dotted symbol to a
    list.

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
