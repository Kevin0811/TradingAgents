"""API routers for TradingAgents API."""

from tradingagents.api.routers.analysts import router as analysts_router
from tradingagents.api.routers.analyze import router as analyze_router
from tradingagents.api.routers.data import router as data_router
from tradingagents.api.routers.decisions import router as decisions_router

__all__ = [
    "analyze_router",
    "analysts_router",
    "data_router",
    "decisions_router",
]
