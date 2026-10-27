"""Send ``anthropic_effort`` to Claude Haiku 5.5+ too.

The core's ``AnthropicClient.get_llm`` passes ``effort`` (``anthropic_effort``
/ ``TRADINGAGENTS_ANTHROPIC_EFFORT``) to ``ChatAnthropic`` only when
``tradingagents.llm_clients.anthropic_client._supports_effort(model)`` is True,
and that predicate matches Opus, Sonnet and Fable only: it dates from Haiku
4.5, which returns 400 for effort. Claude Haiku 5.5 accepts effort
(``low``..``max``, default ``medium``) and thinks adaptively by default, so a
configured ``low`` was silently dropped and the run thought at ``medium``, with
the thinking counted against ``max_tokens``.

``install()`` (called by ``create_app``) rebinds that module-level predicate to
a wrapper that also accepts ``claude-haiku-<major>-<minor>`` at version 5.5 or
later, optionally with a dated snapshot suffix (``claude-haiku-5-5-20261001``).
No dated Haiku 5.5 ID exists today; the dated form is accepted only to mirror
the shape of Anthropic's dated IDs (``claude-haiku-4-5-20251001``). The core's
own predicate accepts no dated form for any family.
``get_llm`` looks the name up in its module's globals on every call, so the
rebinding reaches every ``AnthropicClient``: the analysis graph's
(``TradingAgentsGraph``) and the API's own (``llm_provider``). Haiku 4.5, 5.0
and older Haiku names (``claude-3-5-haiku-...``) stay excluded. No default
changes: with ``anthropic_effort`` unset the core passes no ``effort`` at all.
``anthropic_effort`` is shared by both tiers, so an effort set for an Opus or
Sonnet tier now also applies to a Haiku 5.5 tier.

This applies only to the API process: the CLI never calls ``create_app``, so it
still drops effort for Haiku 5.5.

Why the predicate rather than the ``create_tier_client`` wrapper of
``secret_keys``: that wrapper returns the core's client before ``get_llm``
builds the ``ChatAnthropic``, and ``get_llm`` drops ``effort`` there; adding it
back would mean wrapping ``get_llm`` and duplicating its kwargs handling. The
predicate is the one decision point the core consults.

``tests/test_api_anthropic_effort.py`` builds a real ``AnthropicClient`` and
fails if the core stops consulting this name.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

# Version shape as in the core's predicate (a single-number or dotted
# version), plus an optional date snapshot (see the module docstring). Used
# with fullmatch (not match with "$", which also accepts a trailing newline).
_HAIKU_MODEL = re.compile(r"claude-haiku-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?")
HAIKU_EFFORT_MIN_VERSION: tuple[int, int] = (5, 5)

# A wrapper made by this module has _WRAPPED_ATTR set to _MARK (so install() is
# idempotent) and the predicate it wraps in _ORIGINAL_ATTR; see secret_keys for
# why the check is an identity test against a private object.
_MARK = object()
_WRAPPED_ATTR = "__tradingagents_anthropic_effort__"
_ORIGINAL_ATTR = "__tradingagents_anthropic_effort_original__"
_PREDICATE = "_supports_effort"
_install_lock = threading.Lock()


def haiku_supports_effort(model: str) -> bool:
    """Whether ``model`` is a Claude Haiku at version 5.5 or later."""
    if not isinstance(model, str):
        return False
    match = _HAIKU_MODEL.fullmatch(model.lower())
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2)) if match.group(2) else 0
    return (major, minor) >= HAIKU_EFFORT_MIN_VERSION


def _wrap(original: Callable[[str], bool]) -> Callable[[str], bool]:
    """``original`` that also accepts Haiku 5.5+."""

    def _supports_effort(model: str) -> bool:
        return bool(original(model)) or haiku_supports_effort(model)

    _supports_effort.__doc__ = original.__doc__
    setattr(_supports_effort, _WRAPPED_ATTR, _MARK)
    setattr(_supports_effort, _ORIGINAL_ATTR, original)
    return _supports_effort


def is_wrapped(func: Any) -> bool:
    """Whether ``func`` is a predicate made by ``_wrap``."""
    return getattr(func, _WRAPPED_ATTR, None) is _MARK


def original(func: Any) -> Any:
    """The predicate ``func`` wraps, or ``func`` itself when it is not a wrapper."""
    return getattr(func, _ORIGINAL_ATTR) if is_wrapped(func) else func


def install() -> None:
    """Make every ``AnthropicClient`` pass ``effort`` to Claude Haiku 5.5+ as well.

    Rebinds ``tradingagents.llm_clients.anthropic_client._supports_effort``; a
    second call is a no-op. Process-wide.

    Raises:
        RuntimeError: The core no longer has that predicate (renamed or
            removed), so effort would silently not reach Haiku 5.5 (fail at
            startup rather than run at the wrong effort).
    """
    from tradingagents.llm_clients import anthropic_client

    with _install_lock:
        current = getattr(anthropic_client, _PREDICATE, None)
        if not callable(current):
            raise RuntimeError(
                f"tradingagents.llm_clients.anthropic_client no longer has {_PREDICATE}: "
                "anthropic_effort cannot be sent to Claude Haiku 5.5+ "
                "(see tradingagents/api/infrastructure/anthropic_effort.py)"
            )
        if is_wrapped(current):
            return
        setattr(anthropic_client, _PREDICATE, _wrap(current))
