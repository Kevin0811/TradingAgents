"""Translate a request's optional per-run override fields into a config-override dict.

Shared by both the sync (/analyze) and async (/analyze/tasks) paths so
AnalysisService.run_analysis has exactly one place that needs to merge
overrides into the config passed to TradingAgentsGraph.
"""

from __future__ import annotations

from typing import Any

from tradingagents.api.core.exceptions import InvalidRequestError
from tradingagents.api.infrastructure.llm_provider import check_llm_config

# CLI's own Shallow/Medium/Deep presets (cli/utils.py::select_research_depth).
_RESEARCH_DEPTH_PRESETS = {"shallow": 1, "medium": 3, "deep": 5}

_PASSTHROUGH_FIELDS = (
    "deep_think_llm",
    "quick_think_llm",
    "deep_think_provider",
    "quick_think_provider",
    "anthropic_effort",
    "max_tokens",
    "llm_max_retries",
)

_TIERS = ("deep", "quick")

# Keys resolve_overrides sets to None on purpose: there a None clears the
# server's value instead of being skipped (see AnalysisService.run_analysis).
CLEARED_KEYS = frozenset(f"{tier}_think_backend_url" for tier in _TIERS)


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

    # A tier moved to another provider must not inherit the server's endpoint for
    # that tier (e.g. Claude calls sent to an Ollama URL): it uses the provider's
    # default endpoint. Requests cannot set endpoints themselves.
    for tier in _TIERS:
        if f"{tier}_think_provider" in overrides:
            overrides[f"{tier}_think_backend_url"] = None

    return overrides


def check_run_config(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Reject a run whose model settings would fail on every LLM call.

    ``config`` is the server's, ``overrides`` what ``resolve_overrides`` built
    for the request. Raised before the run is queued, so the caller gets a 400
    instead of a task that fails midway.
    """
    for tier in _TIERS:
        # The server's model name belongs to the server's provider for the tier.
        if f"{tier}_think_provider" in overrides and f"{tier}_think_llm" not in overrides:
            raise InvalidRequestError(
                "Invalid LLM configuration",
                detail=f"set {tier}_think_llm together with {tier}_think_provider",
            )
    try:
        check_llm_config({**config, **overrides})
    except ValueError as exc:
        raise InvalidRequestError("Invalid LLM configuration", detail=str(exc)) from exc
