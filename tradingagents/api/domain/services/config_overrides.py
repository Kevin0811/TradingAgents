"""Translate a request's optional per-run override fields into a config-override dict.

Shared by both the sync (/analyze) and async (/analyze/tasks) paths so
AnalysisService.run_analysis has exactly one place that needs to merge
overrides into the config passed to TradingAgentsGraph.
"""

from __future__ import annotations

from typing import Any

# CLI's own Shallow/Medium/Deep presets (cli/utils.py::select_research_depth).
_RESEARCH_DEPTH_PRESETS = {"shallow": 1, "medium": 3, "deep": 5}

_PASSTHROUGH_FIELDS = ("deep_think_llm", "quick_think_llm", "max_tokens", "llm_max_retries")


def resolve_overrides(request: Any) -> dict[str, Any]:
    """Build a config-override dict from a request's RunOverridesMixin fields.

    Explicit max_debate_rounds/max_risk_discuss_rounds always win over
    research_depth when both are supplied. Works for both AnalyzeRequest and
    TaskCreateRequest since they share the same field names.
    """
    overrides: dict[str, Any] = {}

    research_depth = getattr(request, "research_depth", None)
    if research_depth is not None:
        rounds = _RESEARCH_DEPTH_PRESETS[research_depth]
        overrides["max_debate_rounds"] = rounds
        overrides["max_risk_discuss_rounds"] = rounds

    for field in ("max_debate_rounds", "max_risk_discuss_rounds", *_PASSTHROUGH_FIELDS):
        value = getattr(request, field, None)
        if value is not None:
            overrides[field] = value

    return overrides
