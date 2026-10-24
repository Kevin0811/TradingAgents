"""LangChain callback handlers the API attaches to an analysis's LLMs.

The core graph (synced from upstream, not edited here) passes the ``callbacks``
given to ``TradingAgentsGraph`` into every LLM it builds, so a handler here sees
every LLM call of a run. That is the API's only hook into a running analysis:

* :class:`CancelOnLLMStart` stops a cancelled analysis at its next LLM call.
* :class:`OllamaCacheTrimHandler` trims a local Ollama model's prompt cache
  after each LLM call (see ``ollama_cache_trimmer``).
* :class:`TokenUsageRecorder` adds up the run's token usage per tier and model.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.api.core.exceptions import AnalysisCancelled
from tradingagents.api.infrastructure.ollama_cache_trimmer import OllamaCacheTrimmer

logger = logging.getLogger(__name__)


class CancelOnLLMStart(BaseCallbackHandler):
    """Raise :class:`AnalysisCancelled` at the next LLM call once ``event`` is set.

    ``raise_error`` makes LangChain re-raise the handler's exception out of the
    LLM call instead of logging it, so it unwinds the graph. The event stays
    set, so a caller in the core that catches the exception and retries (the
    structured-output fallback does) is stopped again at its very next call.
    """

    raise_error = True

    def __init__(self, event: threading.Event):
        self._event = event

    def _stop_if_cancelled(self) -> None:
        if self._event.is_set():
            raise AnalysisCancelled("analysis cancelled")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._stop_if_cancelled()

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        self._stop_if_cancelled()


class OllamaCacheTrimHandler(BaseCallbackHandler):
    """After each LLM call (answered or failed), let the trimmer check the run's models."""

    def __init__(self, trimmer: OllamaCacheTrimmer, native_url: str, models: Sequence[str]):
        self._trimmer = trimmer
        self._native_url = native_url
        self._models = tuple(models)

    def _check(self) -> None:
        try:
            self._trimmer.check(self._native_url, self._models)
        except Exception:  # noqa: BLE001 -- the trimmer never raises; belt and braces
            logger.warning("Ollama cache trim failed", exc_info=True)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self._check()

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._check()


# The metadata key that tells a run's LLM calls apart by tier. The core hands
# one callback list to both tiers' models, so AnalysisService tags each model's
# metadata with its tier after building the graph; LangChain passes a model's
# metadata to every callback of its calls, through bind_tools,
# with_structured_output and other bindings too (they wrap the same object).
TIER_METADATA_KEY = "ta_tier"
TIERS = ("quick", "deep")  # the report's order

_TOKEN_COUNTS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def usage_counts(usage: Mapping[str, Any]) -> dict[str, int]:
    """The token counts of one call's LangChain ``usage_metadata``.

    ``input_tokens`` is LangChain's, which counts cached input too: for
    Anthropic, langchain-anthropic adds the cache reads and writes to the API's
    own ``input_tokens`` (which leaves them out), and OpenAI-compatible
    providers count cached prompt tokens in theirs. The cache counts are the
    parts of it read from and written to the cache. langchain-anthropic reports
    the writes as ``cache_creation``, or, once Anthropic splits them by TTL, as
    ``ephemeral_5m_input_tokens`` plus ``ephemeral_1h_input_tokens`` with
    ``cache_creation`` zeroed (never both, so they are not double counted).
    Missing details count as 0.
    """
    details = usage.get("input_token_details") or {}
    cache_creation = details.get("cache_creation") or (
        (details.get("ephemeral_5m_input_tokens") or 0)
        + (details.get("ephemeral_1h_input_tokens") or 0)
    )
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_input_tokens": int(details.get("cache_read") or 0),
        "cache_creation_input_tokens": int(cache_creation),
    }


class TokenUsageRecorder(BaseCallbackHandler):
    """Add up one run's LLM calls and tokens per (tier, model).

    A call's tier is the ``TIER_METADATA_KEY`` of its model's metadata, read
    when it starts; its model is the one the response names
    (``response_metadata`` ``model_name`` or ``model``), else the tier's
    configured model. A call ending without ``usage_metadata``, or failing,
    counts as a call in ``calls_without_usage``. A call whose model carries no
    tier tag (none of the core's do, as both tiers' models are tagged) is left
    out, with a debug line: guessing its tier could only misreport it.

    Thread-safe: the core runs its analysts concurrently, all on these models.
    """

    def __init__(self, tiers: Mapping[str, tuple[str, str]]):
        """``tiers``: tier -> (provider, configured model) of the run."""
        self._tiers = dict(tiers)
        self._lock = threading.Lock()
        self._run_tiers: dict[UUID, str] = {}
        self._usage: dict[tuple[str, str], dict[str, int]] = {}

    def _start(self, run_id: UUID, metadata: Mapping[str, Any] | None) -> None:
        tier = (metadata or {}).get(TIER_METADATA_KEY)
        if tier not in self._tiers:
            logger.debug("LLM call %s has no model tier; its usage is not counted", run_id)
            return
        with self._lock:
            self._run_tiers[run_id] = tier

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, metadata)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, metadata)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            tier = self._run_tiers.pop(run_id, None)
        if tier is None:
            return
        model, usage = None, None
        # One generation per call in the core. Every generation is scanned and
        # the first one carrying usage answers it, wherever it sits; usages are
        # never summed (a provider asked for several choices reports the call's
        # one usage, possibly repeated on each).
        for generation in (g for gens in response.generations for g in gens):
            message = getattr(generation, "message", None)
            meta = getattr(message, "response_metadata", None) or {}
            model = model or meta.get("model_name") or meta.get("model")
            usage = getattr(message, "usage_metadata", None)
            if usage:
                break
        self._add(tier, model, usage_counts(usage) if usage else None)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            tier = self._run_tiers.pop(run_id, None)
        if tier is not None:
            self._add(tier, None, None)

    def _add(self, tier: str, model: str | None, counts: dict[str, int] | None) -> None:
        key = (tier, str(model or self._tiers[tier][1]))
        with self._lock:
            entry = self._usage.get(key)
            if entry is None:
                entry = self._usage[key] = dict.fromkeys(
                    ("calls", *_TOKEN_COUNTS, "calls_without_usage"), 0
                )
            entry["calls"] += 1
            if counts is None:
                entry["calls_without_usage"] += 1
            else:
                for name, value in counts.items():
                    entry[name] += value

    def summary(self, partial: bool) -> dict[str, Any]:
        """The usage so far, in the shape of ``schemas.token_usage.TokenUsage``."""
        with self._lock:
            usage = {key: dict(entry) for key, entry in self._usage.items()}
        tiers = [
            {"tier": tier, "provider": self._tiers[tier][0], "model": model, **entry}
            for (tier, model), entry in sorted(
                usage.items(), key=lambda item: (TIERS.index(item[0][0]), item[0][1])
            )
        ]
        totals = {name: sum(t[name] for t in tiers) for name in ("calls", *_TOKEN_COUNTS)}
        return {"tiers": tiers, "totals": totals, "partial": partial}
