"""Token usage of one full analysis run, per model tier (issue #20)."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

_INPUT_TOKENS = (
    "All input (prompt) tokens, INCLUDING cache_read_input_tokens and "
    "cache_creation_input_tokens: the uncached input is input_tokens minus both "
    "cache fields, so do not add the cache fields to it. One rule for every "
    "provider, LangChain's usage_metadata.input_tokens: with Anthropic it is the "
    "API's input_tokens (which leaves cached tokens out) plus the cache reads and "
    "writes; OpenAI-compatible providers (Ollama included) count cached prompt "
    "tokens in it already."
)
_OUTPUT_TOKENS = "Output tokens; thinking/reasoning tokens are part of them."
_CACHE_READ = (
    "Input tokens read from the provider's prompt cache; already counted in "
    "input_tokens. 0 when the provider reports none."
)
_CACHE_CREATION = (
    "Input tokens written to the provider's prompt cache; already counted in "
    "input_tokens. 0 when the provider reports none."
)


class TierTokenUsage(BaseModel):
    """The calls one model made for one tier of a run."""

    tier: Literal["quick", "deep"] = Field(
        description="The model tier the calls came from: 'quick' or 'deep'."
    )
    provider: str = Field(description="The tier's LLM provider for this run (e.g. 'anthropic').")
    model: str = Field(
        description=(
            "The model name the provider reported, else the configured one; a tier may "
            "appear under both (e.g. the configured alias and the dated model a provider "
            "names in its response). Two tiers on one model are two entries."
        )
    )
    calls: int = Field(
        description="LLM calls, including those that failed or reported no usage."
    )
    input_tokens: int = Field(description=_INPUT_TOKENS)
    output_tokens: int = Field(description=_OUTPUT_TOKENS)
    cache_read_input_tokens: int = Field(description=_CACHE_READ)
    cache_creation_input_tokens: int = Field(description=_CACHE_CREATION)
    calls_without_usage: int = Field(
        description=(
            "Calls that ended without usage data (a provider or stream that reports "
            "none, or a call that failed). Each adds 1 to calls and to this field and "
            "nothing to any token field, so when this is above 0 the token counts are "
            "a lower bound."
        )
    )


class TokenUsageTotals(BaseModel):
    """The sums over every entry of ``tiers``.

    A call without usage data adds to ``calls`` only, so the token totals are a
    lower bound whenever an entry has ``calls_without_usage`` above 0.
    """

    calls: int = Field(description="LLM calls, including those without usage data.")
    input_tokens: int = Field(description=_INPUT_TOKENS)
    output_tokens: int = Field(description=_OUTPUT_TOKENS)
    cache_read_input_tokens: int = Field(description=_CACHE_READ)
    cache_creation_input_tokens: int = Field(description=_CACHE_CREATION)


class TokenUsage(BaseModel):
    """Token usage of one full analysis run."""

    tiers: list[TierTokenUsage] = Field(
        default_factory=list,
        description=(
            "One entry per (tier, model), quick before deep, then by model. Empty "
            "when the run made no LLM call."
        ),
    )
    totals: TokenUsageTotals = Field(
        description=(
            "Sums over all entries of tiers. Calls without usage data add to calls "
            "only, so the token totals are a lower bound when any entry has "
            "calls_without_usage above 0."
        )
    )
    partial: bool = Field(
        description=(
            "True when the run failed or was cancelled: the counts cover only the "
            "LLM calls made before it stopped."
        )
    )


def token_usage_model(usage: dict[str, Any] | None) -> TokenUsage | None:
    """A run's usage report (``TokenUsageRecorder.summary``) as a :class:`TokenUsage`.

    A report that does not fit is dropped with a warning, never failing the
    response or the task it describes.
    """
    if usage is None:
        return None
    try:
        return TokenUsage.model_validate(usage)
    except ValidationError:
        logger.warning("Dropping a malformed token usage report", exc_info=True)
        return None
