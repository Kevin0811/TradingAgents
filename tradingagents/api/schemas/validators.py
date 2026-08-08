"""Shared field validators for API request schemas.

Kept separate from symbol_utils.py's ticker normalization on purpose: that
module (tradingagents/dataflows/symbol_utils.py) documents itself as purely
syntactic and network-free, and is safe to call on every request including
inside retry loops. This module does something different — it fails a
request synchronously at the API boundary, before it ever reaches a
background worker — so a caller finds out about an obviously malformed
ticker immediately (a 4xx on submission) instead of discovering it only
after the worker has already run the job and the task shows up "failed".
"""

from __future__ import annotations

import re

_BARE_NUMERIC = re.compile(r"^\d{4,6}$")


def validate_ticker_shape(v: str) -> str:
    """Reject a ticker shape known to fail downstream, with an actionable hint.

    Deliberately narrow: this only catches a bare 4-6 digit numeric code (the
    concrete failure mode behind issue reports like "$2330: possibly
    delisted; no price data found" — Yahoo Finance needs "2330.TW", not
    "2330"). It does not attempt full ticker validation, which would require
    a network call this API layer does not make. Crypto bases (e.g. "BTC")
    are alphabetic and never match, so they pass through untouched.
    """
    v = v.strip()
    if not v:
        raise ValueError("ticker is required")
    if _BARE_NUMERIC.fullmatch(v):
        raise ValueError(
            f"'{v}' looks like a bare numeric code with no exchange suffix. "
            f"Yahoo Finance needs one, e.g. '2330' -> '2330.TW' for the Taiwan "
            f"Stock Exchange. See the README 'Markets and tickers' table for "
            f"other exchanges."
        )
    return v
