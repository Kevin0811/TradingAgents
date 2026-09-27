"""Analyst keys: the API's names versus the core's.

The core's graph (``tradingagents/graph/analyst_execution.py``, synced from
upstream) knows the analysts as ``market``, ``social``, ``news`` and
``fundamentals`` and rejects anything else ("unknown analyst key"). The API
has long called the social-media analyst ``sentiment``; that name is still
accepted and mapped here, in one place, before a request reaches the core.
"""

from __future__ import annotations

from collections.abc import Iterable

# API-only analyst names -> the core's key.
ANALYST_KEY_ALIASES: dict[str, str] = {"sentiment": "social"}


def to_core_analyst_key(key: str) -> str:
    """Map one API analyst name to the core's key (unknown names unchanged)."""
    value = str(getattr(key, "value", key)).strip().lower()
    return ANALYST_KEY_ALIASES.get(value, value)


def to_core_analyst_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Map API analyst names to core keys, dropping duplicates, keeping order."""
    return tuple(dict.fromkeys(to_core_analyst_key(k) for k in keys))
