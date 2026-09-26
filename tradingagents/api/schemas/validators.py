"""Shared field validators for API request schemas.

Kept separate from symbol_utils.py's ticker normalization on purpose: that
module (tradingagents/dataflows/symbol_utils.py) documents itself as purely
syntactic and network-free, and is safe to call on every request including
inside retry loops. This module does something different — it fails a
request synchronously at the API boundary, before it ever reaches a
background worker — so a caller finds out about a bad ticker immediately (a
4xx on submission) instead of discovering it only after the worker has
already run the job and the task shows up "failed".

Two layers:

* ``validate_ticker_shape`` (field validator): always on, network-free. It
  rejects a bare 4-6 digit code and applies the API-only forex mapping
  (``USDTWD`` -> ``USDTWD=X``).
* ``validate_ticker_supported`` (model validator, needs ``asset_type``):
  checks the ticker against the supported-symbols list of the market it
  belongs to. When that list is not loaded (cold start, fetch failure with no
  cache) or no list covers the ticker's market (``0700.HK``, ``^GSPC``,
  ``GC=F``), it lets the ticker through, so an empty cache never blocks an
  analysis.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic_core import PydanticCustomError

from tradingagents.api.domain.symbols import api_forex_symbol

_BARE_NUMERIC = re.compile(r"^\d{4,6}$")
_BARE_NUMERIC_SUFFIXES = ("TW", "TWO", "T")


def _active_catalog():
    # Imported lazily: domain.services' package __init__ imports the task
    # service, which imports the request schemas that import this module.
    from tradingagents.api.domain.services.symbol_catalog import get_active_catalog

    return get_active_catalog()


def validate_ticker_shape(v: str) -> str:
    """Reject a ticker shape known to fail downstream, with an actionable hint.

    Deliberately narrow: this only catches a bare 4-6 digit numeric code (the
    concrete failure mode behind issue reports like "$2330: possibly
    delisted; no price data found" — Yahoo Finance needs "2330.TW", not
    "2330"). Full validation against the supported-symbols list happens in
    ``validate_ticker_supported``. Crypto bases (e.g. "BTC") are alphabetic
    and never match, so they pass through untouched.

    Returns the stripped ticker, with a forex pair the core does not map
    (``USDTWD``) rewritten to Yahoo's form (``USDTWD=X``).
    """
    v = v.strip()
    if not v:
        raise ValueError("ticker is required")
    if _BARE_NUMERIC.fullmatch(v):
        hint = ""
        catalog = _active_catalog()
        if catalog is not None:
            listed = [
                f"{v}.{suffix}"
                for suffix in _BARE_NUMERIC_SUFFIXES
                if catalog.get(f"{v}.{suffix}") is not None
            ]
            if listed:
                hint = f" On the supported list: {', '.join(listed)}."
        raise ValueError(
            f"'{v}' looks like a bare numeric code with no exchange suffix. "
            f"Yahoo Finance needs one, e.g. '2330' -> '2330.TW' for the Taiwan "
            f"Stock Exchange. See the README 'Markets and tickers' table for "
            f"other exchanges.{hint}"
        )
    return api_forex_symbol(v) or v


def validate_ticker_supported(model: Any) -> Any:
    """Model validator: reject a ticker that is not on its market's list.

    Raises a ``ticker_not_supported`` error (HTTP 422) whose ``ctx`` carries
    ``ticker``, ``market`` and ``suggestions`` (up to 3 close symbols).
    """
    catalog = _active_catalog()
    if catalog is None:
        return model
    asset_type = getattr(model.asset_type, "value", model.asset_type)
    result = catalog.check(model.ticker, asset_type)
    if result.supported is False:
        suggestions = list(result.suggestions)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise PydanticCustomError(
            "ticker_not_supported",
            "'{ticker}' is not on the supported {market} symbol list (see GET /symbols).{hint}",
            {
                "ticker": result.key,
                "market": result.market,
                "suggestions": suggestions,
                "hint": hint,
            },
        )
    return model
