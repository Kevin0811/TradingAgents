"""File-based implementation of StateRepository."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents.api.domain.repositories import StateRepository
from tradingagents.dataflows.symbols import safe_ticker_component


class FileStateRepository(StateRepository):
    """File-based repository for loading saved analysis states from disk."""

    def __init__(self, results_dir: str | Path):
        """Initialize the file state repository.

        Args:
            results_dir: Path to the results directory.
        """
        self._results_dir = Path(results_dir)

    def _log_path(self, ticker: str, trade_date: str) -> Path:
        return (
            self._results_dir
            / safe_ticker_component(ticker)
            / "TradingAgentsStrategy_logs"
            / f"full_states_log_{trade_date}.json"
        )

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
        log_path = self._log_path(ticker, trade_date)

        if not log_path.exists():
            raise FileNotFoundError(
                f"No state file found for {ticker} on {trade_date}. "
                f"Run the analysis first: POST /analyze"
            )

        with open(log_path, encoding="utf-8") as f:
            return json.load(f)

    def load_state_logged(self, ticker: str, trade_date: str) -> tuple[dict[str, Any], datetime]:
        """The saved state, and when its file was last written (its mtime, UTC).

        Raises:
            ValueError: An unsafe ticker (``safe_ticker_component``), a file
                that is not JSON or not UTF-8, or JSON that is not an object.
            FileNotFoundError: If no state file exists.
            OSError: If the file cannot be read.
        """
        log_path = self._log_path(ticker, trade_date)
        if not log_path.is_file():
            raise FileNotFoundError(f"No state file found for {ticker} on {trade_date}.")
        with open(log_path, encoding="utf-8") as f:
            logged_at = datetime.fromtimestamp(os.fstat(f.fileno()).st_mtime, UTC)
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("The state file does not hold a JSON object.")
        return state, logged_at
