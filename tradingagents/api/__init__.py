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

Analysis tasks:
    ``POST /analyze/tasks`` queues an analysis on a worker pool of
    ``task_max_concurrent`` workers (``TRADINGAGENTS_TASK_MAX_CONCURRENT``).
    Unless it is set explicitly (config or env var, either wins), it is 1
    when ``llm_provider`` is ``ollama`` -- concurrent analyses on one small
    local model mostly contend -- and 2 otherwise. ``GET /config`` reports the
    effective value. The one-worker default, like the cache trim below,
    applies only when ``llm_provider == "ollama"``. With Ollama the
    synchronous ``POST /analyze`` runs on the same worker pool, queued with
    the tasks; with any other provider it runs on the request threadpool.
    A sync run is cancelled (it stops at its next LLM call, or never starts
    if still queued) when its client disconnects or the server shuts down,
    and the request ends with 503.

    A task's ``status`` is ``pending``, ``queued``, ``processing``,
    ``completed``, ``failed`` or ``cancelled``.
    ``POST /analyze/tasks/{task_id}/cancel`` cancels a pending or queued task
    at once (it never runs); a processing task gets ``cancel_requested: true``
    and stops at its next LLM call (the core graph cannot be interrupted
    mid-call; a LangChain callback raises at the start of the next one), then
    ends as ``cancelled``, its result discarded. A finished task answers 409.
    A cancel can race the run's end: a task with ``cancel_requested`` may
    still end ``completed`` or ``failed`` when its run had already passed its
    last check. A cancel during the final decision may also leave that
    decision in the core's decision log (``memory_log.store_decision`` runs
    inside the graph); the API discards the result but cannot undo that.
    ``DELETE /analyze/tasks/{task_id}`` cancels an active task the same way
    before removing it, and just removes a finished one. A deleted running
    task is gone (404) at once but keeps counting as active until its run has
    stopped, so the symbols refresher does not take the app for idle.
    LangChain's and the core's warnings about the cancel exception are
    filtered out; each cancelled task logs one INFO line.

Local model (Ollama) cache trim:
    Ollama's MLX engine keeps a prompt-cache snapshot per request under a
    fixed 8 GiB budget (ollama/ollama#18131), so a model grows call by call.
    Only with ``llm_provider == "ollama"`` (no trimmer exists otherwise, and
    ``GET /config`` reports ``ollama_cache_trim_active: false``), after each
    LLM call of an analysis the API reads ``GET /api/ps`` and unloads the
    run's models (its quick and deep model, nothing else) once one has grown
    more than the budget past its baseline (``POST /api/generate`` with
    ``keep_alive: 0``); the next call reloads it. The baseline is the size at
    the first read of a load, capped at the model's weights (``GET
    /api/tags``) plus 1024 MB, so a model already grown when first seen is
    trimmed at once. The model may be shared with another app (fin-insight),
    so a load with a new ``context_length``, or one smaller than at the last
    trim, is measured afresh rather than unloaded. The native URL is the
    run's ``/v1`` URL (``backend_url``, else ``OLLAMA_BASE_URL``, else the
    default) without ``/v1``. Errors are logged and never fail the analysis.
    See ``infrastructure/ollama_cache_trimmer.py`` for the exact rules.

    - ``TRADINGAGENTS_OLLAMA_CACHE_TRIM_MB`` -> ``ollama_cache_trim_mb``
      (1024; 0 = off). A ``create_app(overrides=...)`` value wins over the
      env var; an empty env value counts as unset.

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
