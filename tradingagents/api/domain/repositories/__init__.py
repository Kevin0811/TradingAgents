"""Repository interfaces for data access."""

from tradingagents.api.domain.repositories.base import Repository
from tradingagents.api.domain.repositories.state_repository import StateRepository

__all__ = [
    "Repository",
    "StateRepository",
]
