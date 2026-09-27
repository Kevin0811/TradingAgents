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
      refresh right away (202 with ``queued`` / ``skipped`` /
      ``skip_reasons``; a market fetched within the last 30 minutes is skipped
      unless forced, and a forced market forced again within 5 minutes is
      skipped too; ``force`` also accepts a shrunken list, never an
      incomplete fetch)
    - ``GET /symbols/settings`` / ``PUT /symbols/settings`` read and change the
      refresher settings at runtime (see below). Like every endpoint of this
      API they have no authentication: keep the API on a trusted network.

    Lists are cached as ``<cache dir>/<market>.json`` (default
    ``<data_cache_dir>/symbols``, i.e. ``~/.tradingagents/cache/symbols``) and
    loaded at startup; the refresher fetches one market at a time on a
    background thread, and requests never wait on the network.

    - A missing list is fetched right away.
    - A stale list (older than the TTL) keeps being served and is refreshed
      only when TradingAgents is idle: no analysis task pending, queued or
      processing, and no Yahoo-backed request (``/data``, ``/analysts``, the
      synchronous ``POST /analyze``) within ``symbols_idle_grace_minutes``.
      An optional daily window (``symbols_refresh_window``, off by default)
      must also hold when set. A tick every 15 minutes picks up lists that
      became due; ``GET /symbols`` shows ``waiting_for`` and, for the window,
      ``next_refresh_after``.
    - While an automatic refresh of an existing list runs it pauses between
      pages whenever TradingAgents turns busy, and resumes the same market
      once idle again (without re-checking the window); after a pause of
      over 15 minutes the current screener query starts over. A manual
      refresh and the fetch of a missing list never pause. Manual work goes
      first: a paused automatic refresh makes way for it (and is queued
      again), or carries on without pausing if it is the market asked for.
      Turning ``auto_refresh`` off drops queued automatic refreshes.
    - A miss in a tw or jp list (the enforced markets) older than a day asks
      for a refresh of that market under the same conditions, so new listings
      show up. us, crypto and fx misses never do: those lists are incomplete
      by design and refresh on the TTL only.
    - A failed refresh keeps the previous list, and so does one that looks
      truncated (under 80% of the previous entries, or an exchange/type group
      gone empty) unless it was forced; either sets ``last_error``. A screener
      page or lookup failing with a 5xx or 429 is retried up to 3 times (2 s,
      5 s, 10 s, or ``Retry-After``), and once after a timeout or connection
      error, before the market's refresh fails.

    Settings (env var -> config key; the ones marked * can also be changed
    with ``PUT /symbols/settings``, which saves them to
    ``<cache dir>/settings.json`` so they survive a restart). Precedence,
    highest first: ``PUT /symbols/settings`` (source ``api``) >
    ``create_app(overrides=...)`` (``config``) > env var (``env``) > built-in
    default (``default``); null in the PUT drops the API value, so the
    config / env / default value applies again. An empty env value counts as
    unset:

    - ``TRADINGAGENTS_SYMBOLS_CACHE_DIR`` -> ``symbols_cache_dir``
    - ``TRADINGAGENTS_SYMBOLS_CACHE_TTL_DAYS`` -> ``symbols_cache_ttl_days``
      * (7.0; 0.5-90)
    - ``TRADINGAGENTS_SYMBOLS_AUTO_REFRESH`` -> ``symbols_auto_refresh`` *
      (true)
    - ``TRADINGAGENTS_SYMBOLS_IDLE_GRACE_MINUTES`` ->
      ``symbols_idle_grace_minutes`` * (10; 0-240)
    - ``TRADINGAGENTS_SYMBOLS_REFRESH_WINDOW`` -> ``symbols_refresh_window`` *
      (``""`` = off; ``HH:MM-HH:MM``, may wrap midnight)
    - ``TRADINGAGENTS_SYMBOLS_REFRESH_TIMEZONE`` ->
      ``symbols_refresh_timezone`` * (``"Asia/Taipei"``, an IANA zone; only
      resolved when a window is set)
    - ``TRADINGAGENTS_SYMBOLS_PAGE_DELAY_SECONDS`` ->
      ``symbols_page_delay_seconds`` * (2.0; 0.5-10)
    - ``TRADINGAGENTS_SYMBOLS_INCLUDE_OTC`` -> ``symbols_include_otc`` (false)
    - ``TRADINGAGENTS_SYMBOLS_MAX_PAGES`` -> ``symbols_max_pages`` (200 pages
      of 250 rows per screener query; a query that needs more fails the
      refresh instead of saving a partial list)

    The * values are validated at startup too: an out-of-range, non-finite
    or malformed env / config value stops the app from starting, with an
    error naming the env var or config key.

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
