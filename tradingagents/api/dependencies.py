"""FastAPI dependency injection for the TradingAgents API.

This module provides dependency providers for FastAPI, creating
service instances using the infrastructure layer.

Every provider here is declared ``async`` on purpose. FastAPI resolves sync
dependencies on the shared AnyIO threadpool; since these providers only
construct objects and perform no I/O, keeping them on the event loop means
request handling never competes for a threadpool slot.
"""

from __future__ import annotations

from fastapi import Depends, Request

from tradingagents.api.config import ApiConfig, get_config
from tradingagents.api.core.task_manager import TaskManager
from tradingagents.api.core.task_worker import TaskWorker
from tradingagents.api.domain.services.analysis_service import AnalysisService
from tradingagents.api.domain.services.analyst_service import AnalystService
from tradingagents.api.domain.services.decision_service import DecisionService
from tradingagents.api.domain.services.market_data_service import MarketDataService
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.domain.repositories import StateRepository
from tradingagents.api.infrastructure.llm_provider import LLMProviderFactory
from tradingagents.api.infrastructure.repositories.file_state_repository import FileStateRepository


# ---------------------------------------------------------------------------
# Infrastructure dependencies
# ---------------------------------------------------------------------------


async def get_llm_provider_factory(
    config: ApiConfig = Depends(get_config),
) -> LLMProviderFactory:
    """Create an LLM provider factory."""
    return LLMProviderFactory(config=config.config)


async def get_state_repository(
    config: ApiConfig = Depends(get_config),
) -> StateRepository:
    """Create a file-based state repository."""
    return FileStateRepository(results_dir=config.results_dir)


# ---------------------------------------------------------------------------
# Domain service dependencies
# ---------------------------------------------------------------------------


async def get_analysis_service(
    config: ApiConfig = Depends(get_config),
) -> AnalysisService:
    """Create an analysis service."""
    return AnalysisService(
        config=config.config,
        debug=config.debug,
    )


async def get_llm_factory(
    config: ApiConfig = Depends(get_config),
) -> callable:
    """Create an LLM factory function for AnalystService.
    
    Uses .env configuration via DEFAULT_CONFIG.
    """
    # LLMProviderFactory now uses DEFAULT_CONFIG (.env) by default
    # Pass config.config only if you want to override .env settings
    provider = LLMProviderFactory()
    return provider.create_llm


async def get_analyst_service(
    llm_factory: callable = Depends(get_llm_factory),
    config: ApiConfig = Depends(get_config),
) -> AnalystService:
    """Create an analyst service."""
    return AnalystService(
        llm_factory=llm_factory,
        config=config.config,
    )


async def get_market_data_service() -> MarketDataService:
    """Create a market data service."""
    return MarketDataService()


async def get_decision_service(
    state_repo: StateRepository = Depends(get_state_repository),
) -> DecisionService:
    """Create a decision service."""
    return DecisionService(state_repository=state_repo)


# ---------------------------------------------------------------------------
# Task management dependencies
# ---------------------------------------------------------------------------

# The task manager and worker are owned by the app (created in create_app,
# shut down in the lifespan handler) rather than by module globals, so each
# app instance gets its own and there is no lazy-init race between requests.


async def get_task_manager(request: Request) -> TaskManager:
    """Get the app's task manager."""
    return request.app.state.task_manager


async def get_task_worker(request: Request) -> TaskWorker:
    """Get the app's background task worker."""
    return request.app.state.task_worker


async def get_task_service(
    task_manager: TaskManager = Depends(get_task_manager),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> TaskService:
    """Create a task service."""
    return TaskService(
        task_manager=task_manager,
        analysis_service=analysis_service,
    )
