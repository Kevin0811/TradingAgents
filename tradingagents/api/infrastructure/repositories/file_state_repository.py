"""File-based implementation of StateRepository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tradingagents.api.domain.repositories import StateRepository
from tradingagents.dataflows.utils import safe_ticker_component


class FileStateRepository(StateRepository):
    """File-based repository for loading saved analysis states from disk."""

    def __init__(self, results_dir: str | Path):
        """Initialize the file state repository.

        Args:
            results_dir: Path to the results directory.
        """
        self._results_dir = Path(results_dir)

    def load_state(self, ticker: str, trade_date: str) -> dict[str, Any]:
        """Load the saved state for a ticker and date from disk.

        Args:
            ticker: Ticker symbol.
            trade_date: Trading date in YYYY-MM-DD format.

        Returns:
            Dict with saved analysis state.

        Raises:
            FileNotFoundError: If no state file exists.
        """
        safe_ticker = safe_ticker_component(ticker)
        log_path = (
            self._results_dir
            / safe_ticker
            / "TradingAgentsStrategy_logs"
            / f"full_states_log_{trade_date}.json"
        )

        if not log_path.exists():
            raise FileNotFoundError(
                f"No state file found for {ticker} on {trade_date}. "
                f"Run the analysis first: POST /analyze"
            )

        with open(log_path, encoding="utf-8") as f:
            return json.load(f)
