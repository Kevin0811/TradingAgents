"""LangChain callback handlers the API attaches to an analysis's LLMs.

The core graph (synced from upstream, not edited here) passes the ``callbacks``
given to ``TradingAgentsGraph`` into every LLM it builds, so a handler here sees
every LLM call of a run. That is the API's only hook into a running analysis:

* :class:`CancelOnLLMStart` stops a cancelled analysis at its next LLM call.
* :class:`OllamaCacheTrimHandler` trims a local Ollama model's prompt cache
  after each LLM call (see ``ollama_cache_trimmer``).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

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
