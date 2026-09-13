"""Router for the server config query endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from tradingagents.api.config import ApiConfig, get_config
from tradingagents.api.schemas.config import ConfigResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/config",
    tags=["Config"],
)


@router.get(
    "",
    response_model=ConfigResponse,
    summary="Get current effective config",
    description=(
        "Return the server's currently-effective runtime settings, limited to "
        "the fields callers can override per-request on `POST /analyze` and "
        "`POST /analyze/tasks` (see their `research_depth`/`max_debate_rounds`/"
        "`deep_think_llm`/etc. fields), plus read-only task-queue settings.\n\n"
        "Use this to see what a request will get if it does not override a "
        "field -- most of these are only otherwise settable via server-side "
        "environment variables.\n\n"
        "Never includes API keys or secrets: those are read directly from the "
        "environment by LLM clients and data vendors and are never part of this "
        "config."
    ),
)
def get_current_config(config: ApiConfig = Depends(get_config)) -> ConfigResponse:
    """Return the currently-effective server config."""
    c = config.config
    return ConfigResponse(
        llm_provider=c.get("llm_provider"),
        deep_think_llm=c.get("deep_think_llm"),
        quick_think_llm=c.get("quick_think_llm"),
        backend_url=c.get("backend_url"),
        temperature=c.get("temperature"),
        llm_max_retries=c.get("llm_max_retries"),
        max_tokens=c.get("max_tokens"),
        max_debate_rounds=c.get("max_debate_rounds"),
        max_risk_discuss_rounds=c.get("max_risk_discuss_rounds"),
        checkpoint_enabled=c.get("checkpoint_enabled"),
        output_language=c.get("output_language"),
        task_ttl_minutes=c.get("task_ttl_minutes"),
        task_max_tasks=c.get("task_max_tasks"),
        task_max_concurrent=c.get("task_max_concurrent"),
        analyst_concurrency_limit=c.get("analyst_concurrency_limit"),
    )
