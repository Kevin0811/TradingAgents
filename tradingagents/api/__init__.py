"""FastAPI-based REST API for TradingAgents.

This module provides HTTP endpoints for:
- Running full trading analysis pipelines
- Accessing individual analyst agents
- Querying market data and fundamentals
- Retrieving structured trading decisions

Usage:
    from tradingagents.api import create_app

    app = create_app()
    # Run with uvicorn:
    # uvicorn tradingagents.api:create_app --factory
"""

from tradingagents.api.app import create_app

__all__ = ["create_app"]