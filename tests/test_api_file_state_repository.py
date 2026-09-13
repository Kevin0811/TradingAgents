"""Tests for the TradingAgents API FileStateRepository.

Covers:
- load_state() reads JSON from the correct path
- FileNotFoundError when state file is missing
- safe_ticker_component() is applied to the ticker
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.api.infrastructure.repositories.file_state_repository import (
    FileStateRepository,
)


@pytest.mark.unit
class TestFileStateRepositoryLoadState:
    """load_state() must read JSON from the correct path structure."""

    def test_loads_state_from_correct_path(self, tmp_path):
        """State file path: {results_dir}/{TICKER}/TradingAgentsStrategy_logs/full_states_log_{DATE}.json"""
        results_dir = tmp_path / "results"
        ticker_dir = results_dir / "AAPL" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)

        state_file = ticker_dir / "full_states_log_2026-06-01.json"
        expected_state = {"market_report": "bullish", "signal": "Buy"}
        state_file.write_text(json.dumps(expected_state), encoding="utf-8")

        repo = FileStateRepository(results_dir=results_dir)
        state = repo.load_state("AAPL", "2026-06-01")
        assert state == expected_state

    def test_raises_file_not_found_when_no_state_file(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True)

        repo = FileStateRepository(results_dir=results_dir)
        with pytest.raises(FileNotFoundError, match="No state file found"):
            repo.load_state("AAPL", "2026-06-01")

    def test_raises_file_not_found_when_no_ticker_dir(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True)

        repo = FileStateRepository(results_dir=results_dir)
        with pytest.raises(FileNotFoundError, match="No state file found"):
            repo.load_state("NONEXISTENT", "2026-06-01")

    def test_uses_safe_ticker_component(self, tmp_path):
        """The repository must run the ticker through safe_ticker_component()
        (path-traversal safety) rather than interpolating it directly.

        safe_ticker_component() validates but does not change casing (see
        tradingagents/dataflows/utils.py), and neither does the write side
        (trading_graph.py writes results under whatever case the caller
        passed in) -- so, unlike checkpointer.py's own ticker handling, a
        ticker here is matched case-sensitively end to end. The fixture
        directory case must therefore match the ticker passed to
        load_state().
        """
        results_dir = tmp_path / "results"
        ticker_dir = results_dir / "aapl" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)

        state_file = ticker_dir / "full_states_log_2026-06-01.json"
        state_file.write_text(json.dumps({"test": "value"}), encoding="utf-8")

        repo = FileStateRepository(results_dir=results_dir)
        state = repo.load_state("aapl", "2026-06-01")
        assert state == {"test": "value"}

    def test_handles_nested_path_structure(self, tmp_path):
        """The path includes the TradingAgentsStrategy_logs subdirectory."""
        results_dir = tmp_path / "results"
        ticker_dir = results_dir / "MSFT" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)

        state_file = ticker_dir / "full_states_log_2026-06-01.json"
        expected = {"fundamentals_report": "strong"}
        state_file.write_text(json.dumps(expected), encoding="utf-8")

        repo = FileStateRepository(results_dir=results_dir)
        state = repo.load_state("MSFT", "2026-06-01")
        assert state == expected

    def test_returns_dict_from_json(self, tmp_path):
        """The loaded state must be a Python dict."""
        results_dir = tmp_path / "results"
        ticker_dir = results_dir / "GOOG" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)

        state_file = ticker_dir / "full_states_log_2026-06-01.json"
        complex_state = {
            "market_report": "text",
            "nested": {"key": "value"},
            "list_field": [1, 2, 3],
        }
        state_file.write_text(json.dumps(complex_state), encoding="utf-8")

        repo = FileStateRepository(results_dir=results_dir)
        state = repo.load_state("GOOG", "2026-06-01")
        assert isinstance(state, dict)
        assert state["nested"] == {"key": "value"}
        assert state["list_field"] == [1, 2, 3]