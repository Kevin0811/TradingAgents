"""Error response schema."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tradingagents.api.schemas.token_usage import TokenUsage


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


class AnalysisErrorResponse(ErrorResponse):
    """500 body of a failed synchronous analysis (``POST /analyze``)."""

    token_usage: TokenUsage | None = Field(
        default=None,
        description=(
            "Token usage of the LLM calls the run made before it failed (partial: true). "
            "Absent when the run failed before it started, or its usage report was "
            "malformed."
        ),
    )
