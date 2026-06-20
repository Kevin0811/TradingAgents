"""Base repository interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """Base repository interface for data access."""

    @abstractmethod
    def get(self, id: str) -> T:
        """Get an entity by ID."""
        pass

    @abstractmethod
    def list(self, **kwargs) -> list[T]:
        """List entities with optional filtering."""
        pass

    @abstractmethod
    def save(self, entity: T) -> None:
        """Save an entity."""
        pass