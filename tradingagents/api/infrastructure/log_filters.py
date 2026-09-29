"""Log filters that keep a cancelled analysis from flooding the logs.

A running analysis stops through :class:`~tradingagents.api.infrastructure.
llm_callbacks.CancelOnLLMStart`, which raises ``AnalysisCancelled`` at the next
LLM call. On its way up, two loggers outside the API report it as a problem:

* LangChain's callback manager logs ``Error in CancelOnLLMStart.<event>
  callback: AnalysisCancelled(...)`` at WARNING before re-raising it;
* the core's structured-output fallback (``tradingagents.agents.structured``)
  logs ``<agent>: structured-output invocation failed (analysis cancelled);
  retrying once as free text``, and the retry is stopped the same way.

:class:`CancelNoiseFilter` drops exactly those two records -- matched on the
logger, the message template and the cancel exception -- and nothing else;
the task's own INFO line (``TaskService``) says it was cancelled.
"""

from __future__ import annotations

import logging

from tradingagents.api.core.exceptions import AnalysisCancelled

LANGCHAIN_CALLBACK_LOGGER = "langchain_core.callbacks.manager"
STRUCTURED_FALLBACK_LOGGER = "tradingagents.agents.structured"

_CALLBACK_ERROR_TEMPLATE = "Error in %s.%s callback: %s"
_STRUCTURED_FAILED_PREFIX = "%s: structured-output invocation failed"
_CANCEL_HANDLER = "CancelOnLLMStart"


class CancelNoiseFilter(logging.Filter):
    """Drop the warnings a cancelled analysis's unwinding logs (see the module docs)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _is_cancel_noise(record)


def _is_cancel_noise(record: logging.LogRecord) -> bool:
    args = record.args if isinstance(record.args, tuple) else ()
    if record.name == LANGCHAIN_CALLBACK_LOGGER:
        return (
            record.msg == _CALLBACK_ERROR_TEMPLATE
            and len(args) == 3
            and args[0] == _CANCEL_HANDLER
            and AnalysisCancelled.__name__ in str(args[2])
        )
    if record.name == STRUCTURED_FALLBACK_LOGGER:
        return (
            isinstance(record.msg, str)
            and record.msg.startswith(_STRUCTURED_FAILED_PREFIX)
            and len(args) >= 2
            and isinstance(args[1], AnalysisCancelled)
        )
    return False


def install_cancel_log_filter() -> None:
    """Attach :class:`CancelNoiseFilter` to the two loggers, once each."""
    for name in (LANGCHAIN_CALLBACK_LOGGER, STRUCTURED_FALLBACK_LOGGER):
        target = logging.getLogger(name)
        if not any(isinstance(f, CancelNoiseFilter) for f in target.filters):
            target.addFilter(CancelNoiseFilter())
