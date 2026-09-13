"""Response schema for the GET /config endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    """Currently-effective server config, limited to the fields a caller can
    override per-request (see RunOverridesMixin) plus read-only queue/concurrency
    settings.

    Deliberately excludes server-internal paths (project_dir, results_dir,
    data_cache_dir, ...) and never carries API keys/secrets -- those are read
    directly from the environment by LLM clients and data vendors and are never
    stored in this config dict.
    """

    llm_provider: str | None = None
    deep_think_llm: str | None = None
    quick_think_llm: str | None = None
    backend_url: str | None = None
    temperature: float | None = None
    llm_max_retries: int | None = None
    max_tokens: int | None = None
    max_debate_rounds: int | None = None
    max_risk_discuss_rounds: int | None = None
    checkpoint_enabled: bool | None = None
    output_language: str | None = None
    task_ttl_minutes: int | None = Field(
        default=None, description="Task retention time in minutes (async task queue)."
    )
    task_max_tasks: int | None = Field(
        default=None, description="Maximum number of tasks retained in memory."
    )
    task_max_concurrent: int | None = Field(
        default=None, description="Maximum concurrent task executions."
    )
    analyst_concurrency_limit: int | None = Field(
        default=None,
        description="Reserved for future use; analysts currently run sequentially.",
    )
