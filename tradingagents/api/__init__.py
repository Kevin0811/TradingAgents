"""FastAPI-based REST API for TradingAgents.

This module provides HTTP endpoints for:
- Running full trading analysis pipelines
- Accessing individual analyst agents
- Querying market data and fundamentals
- Retrieving structured trading decisions

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
from tradingagents.api.domain.services.task_service import TaskService
from tradingagents.api.schemas.task import TaskCreateRequest, TaskResponse, TaskStatus

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
