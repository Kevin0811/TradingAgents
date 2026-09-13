"""Shared per-request config-override fields for analysis requests.

Mixed into both AnalyzeRequest (sync) and TaskCreateRequest (async) so the
two request shapes carry identical override fields and a single
``resolve_overrides`` helper (domain/services/config_overrides.py) can
translate either one into a TradingAgentsGraph config-override dict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RunOverridesMixin(BaseModel):
    """Optional per-run overrides for the highest-impact runtime/config knobs.

    All fields default to None, meaning "use the server's configured default"
    (see GET /config). Explicit max_debate_rounds/max_risk_discuss_rounds take
    precedence over research_depth when both are supplied.
    """

    research_depth: Literal["shallow", "medium", "deep"] | None = Field(
        default=None,
        description=(
            "Convenience preset mapping to CLI depth tiers: shallow=1, medium=3, "
            "deep=5 debate/risk-discussion rounds. Mutually exclusive with "
            "max_debate_rounds/max_risk_discuss_rounds; if both are given, the "
            "explicit ints win."
        ),
    )
    max_debate_rounds: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "Override the number of bull/bear research debate rounds for this "
            "run (default from server config, normally 1). Fewer rounds run "
            "faster."
        ),
    )
    max_risk_discuss_rounds: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "Override the number of risk-management discussion rounds for this "
            "run (default from server config, normally 1). Fewer rounds run "
            "faster."
        ),
    )
    deep_think_llm: str | None = Field(
        default=None,
        description="Override the deep-thinking model for this run.",
    )
    quick_think_llm: str | None = Field(
        default=None,
        description="Override the quick-thinking model for this run.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Override the per-call max output token cap for this run.",
    )
    llm_max_retries: int | None = Field(
        default=None,
        ge=0,
        description="Override the number of LLM retry attempts on transient failures for this run.",
    )
