"""State repository interface for analysis results persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StateRepository(ABC):
    """Abstract repository for accessing saved analysis states."""

    @abstractmethod
    def load_state(self, ticker: str, trade_date: str) -> dict[str, Any]:
        """Load the saved state for a ticker and date.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date in YYYY-MM-DD format.

        Returns:
            Dict with saved analysis state.

        Raises:
            FileNotFoundError: If no state file exists.
        """
        pass