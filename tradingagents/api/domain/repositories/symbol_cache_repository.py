"""Symbol cache repository interface for the supported-symbols list."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tradingagents.api.domain.entities import SymbolList


class SymbolCacheRepository(ABC):
    """Abstract persistent cache holding one supported-symbols list per market."""

    @abstractmethod
    def load(self, market: str) -> SymbolList | None:
        """Return the cached list for ``market``, or None if there is none.

        An unreadable or malformed cache is reported as None (and logged), so
        a bad file never stops the API from starting.
        """

    @abstractmethod
    def save(self, symbol_list: SymbolList) -> None:
        """Persist ``symbol_list``, replacing the market's previous cache atomically."""
