"""Tests for resolve_overrides (per-request config-override resolution).

Covers:
- research_depth preset expansion (shallow/medium/deep -> 1/3/5 rounds)
- explicit max_debate_rounds/max_risk_discuss_rounds take precedence over the preset
- passthrough fields (deep_think_llm, quick_think_llm, max_tokens, llm_max_retries)
- no fields set -> empty overrides dict
- works identically for AnalyzeRequest and TaskCreateRequest
"""

from __future__ import annotations

import pytest

from tradingagents.api.domain.services.config_overrides import resolve_overrides
from tradingagents.api.schemas.request import AnalyzeRequest
from tradingagents.api.schemas.task import TaskCreateRequest


@pytest.mark.unit
@pytest.mark.parametrize("request_cls", [AnalyzeRequest, TaskCreateRequest])
class TestResolveOverrides:
    def test_no_overrides_yields_empty_dict(self, request_cls):
        request = request_cls(ticker="AAPL", trade_date="2026-06-01")
        assert resolve_overrides(request) == {}

    @pytest.mark.parametrize(
        "depth,rounds", [("shallow", 1), ("medium", 3), ("deep", 5)]
    )
    def test_research_depth_preset_expands_to_both_round_counts(
        self, request_cls, depth, rounds
    ):
        request = request_cls(ticker="AAPL", trade_date="2026-06-01", research_depth=depth)
        overrides = resolve_overrides(request)
        assert overrides["max_debate_rounds"] == rounds
        assert overrides["max_risk_discuss_rounds"] == rounds

    def test_explicit_rounds_win_over_research_depth(self, request_cls):
        request = request_cls(
            ticker="AAPL",
            trade_date="2026-06-01",
            research_depth="deep",
            max_debate_rounds=2,
        )
        overrides = resolve_overrides(request)
        assert overrides["max_debate_rounds"] == 2
        assert overrides["max_risk_discuss_rounds"] == 5

    def test_passthrough_fields(self, request_cls):
        request = request_cls(
            ticker="AAPL",
            trade_date="2026-06-01",
            deep_think_llm="model-a",
            quick_think_llm="model-b",
            max_tokens=1000,
            llm_max_retries=0,
        )
        overrides = resolve_overrides(request)
        assert overrides == {
            "deep_think_llm": "model-a",
            "quick_think_llm": "model-b",
            "max_tokens": 1000,
            "llm_max_retries": 0,
        }
