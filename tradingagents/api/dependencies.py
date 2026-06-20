"""FastAPI dependency injection for the TradingAgents API.

This module provides dependency providers for FastAPI, creating
service instances using the infrastructure layer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends

from tradingagents.api.config import ApiConfig, get_config
from tradingagents.api.core.task_manager import TaskManager
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


def get_llm_provider_factory(
    config: ApiConfig = Depends(get_config),
) -> LLMProviderFactory:
    """Create an LLM provider factory."""
    return LLMProviderFactory(config=config.config)


def get_state_repository(
    config: ApiConfig = Depends(get_config),
) -> StateRepository:
    """Create a file-based state repository."""
    return FileStateRepository(results_dir=config.results_dir)


# ---------------------------------------------------------------------------
# Domain service dependencies
# ---------------------------------------------------------------------------


def get_analysis_service(
    config: ApiConfig = Depends(get_config),
) -> AnalysisService:
    """Create an analysis service."""
    return AnalysisService(
        config=config.config,
        debug=config.debug,
    )


def get_llm_factory(
    config: ApiConfig = Depends(get_config),
) -> callable:
    """Create an LLM factory function for AnalystService.
    
    Uses .env configuration via DEFAULT_CONFIG.
    """
    # LLMProviderFactory now uses DEFAULT_CONFIG (.env) by default
    # Pass config.config only if you want to override .env settings
    provider = LLMProviderFactory()
    return provider.create_llm


def get_analyst_service(
    llm_factory: callable = Depends(get_llm_factory),
    config: ApiConfig = Depends(get_config),
) -> AnalystService:
    """Create an analyst service."""
    return AnalystService(
        llm_factory=llm_factory,
        config=config.config,
    )


def get_market_data_service() -> MarketDataService:
    """Create a market data service."""
    return MarketDataService()


def get_decision_service(
    state_repo: StateRepository = Depends(get_state_repository),
) -> DecisionService:
    """Create a decision service."""
    return DecisionService(state_repository=state_repo)


# ---------------------------------------------------------------------------
# Task management dependencies
# ---------------------------------------------------------------------------

import threading

# Global task manager singleton
_task_manager: TaskManager | None = None

# Global semaphore for concurrency control
_execution_semaphore: threading.Semaphore | None = None


def get_execution_semaphore(
    config: ApiConfig = Depends(get_config),
) -> threading.Semaphore:
    """Get or create the global execution semaphore.

    This semaphore controls the maximum number of concurrent task executions.
    """
    global _execution_semaphore
    if _execution_semaphore is None:
        max_concurrent = int(config.config.get("task_max_concurrent", 4))
        _execution_semaphore = threading.Semaphore(max_concurrent)
    return _execution_semaphore


def get_task_manager(
    config: ApiConfig = Depends(get_config),
) -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(
            ttl_minutes=config.config.get("task_ttl_minutes", 60),
            max_tasks=config.config.get("task_max_tasks", 100),
        )
    return _task_manager


def get_task_service(
    task_manager: TaskManager = Depends(get_task_manager),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    semaphore: threading.Semaphore = Depends(get_execution_semaphore),
) -> TaskService:
    """Create a task service with concurrency control."""
    return TaskService(
        task_manager=task_manager,
        analysis_service=analysis_service,
        semaphore=semaphore,
    )
