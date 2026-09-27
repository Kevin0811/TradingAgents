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
  asks the supported-symbols catalog for its decision (``SymbolCatalog.decide``,
  the same code path as ``GET /symbols/check``) and rejects the request only
  when the ticker is missing from a loaded list of a market whose policy is
  enforced (tw, jp; see ``REJECT_UNLISTED_TICKERS``), or is a suffix-less
  TW/JP code whose suffixed symbol is listed. A miss in a soft market (us,
  crypto, fx), a market whose list is not loaded (cold start, fetch failure
  with no cache) and a market no list covers (``0700.HK``, ``^GSPC``,
  ``GC=F``) all pass, so an empty cache never blocks an analysis.
"""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticCustomError

from tradingagents.api.domain.symbols import api_forex_symbol, is_bare_numeric

TICKER_NOT_SUPPORTED = "ticker_not_supported"
TICKER_NOT_SUPPORTED_MESSAGE = (
    "'{ticker}' is not on the supported {market} symbol list (see GET /symbols).{hint}"
)


def _active_catalog():
    # Imported lazily: domain.services' package __init__ imports the task
    # service, which imports the request schemas that import this module.
    from tradingagents.api.domain.services.symbol_catalog import get_active_catalog

    return get_active_catalog()


def normalize_asset_type(v: Any) -> Any:
    """Field validator (mode="before"): ``asset_type`` is case-insensitive."""
    return v.strip().lower() if isinstance(v, str) else v


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
    if is_bare_numeric(v):
        hint = ""
        catalog = _active_catalog()
        if catalog is not None:
            listed = catalog.listed_local_variants(v)
            if listed:
                hint = f" On the supported list: {', '.join(listed)}."
        raise ValueError(
            f"'{v}' looks like a bare numeric code with no exchange suffix. "
            f"Yahoo Finance needs one, e.g. '2330' -> '2330.TW' for the Taiwan "
            f"Stock Exchange. See the README 'Markets and tickers' table for "
            f"other exchanges.{hint}"
        )
    return api_forex_symbol(v) or v


def ticker_not_supported_error(decision: Any) -> PydanticCustomError:
    """The 422 error for a rejected ``TickerDecision``.

    ``ctx`` carries ``ticker`` (normalised), ``market``, ``suggestions`` (up to
    3 listed symbols) and ``hint`` (the text appended to the message).
    """
    from tradingagents.api.domain.services.symbol_catalog import REASON_MISSING_SUFFIX

    suggestions = list(decision.suggestions)
    if decision.reason == REASON_MISSING_SUFFIX and suggestions:
        hint = f" It needs an exchange suffix; listed as: {', '.join(suggestions)}."
    elif suggestions:
        hint = f" Did you mean: {', '.join(suggestions)}?"
    else:
        hint = ""
    return PydanticCustomError(
        TICKER_NOT_SUPPORTED,
        TICKER_NOT_SUPPORTED_MESSAGE,
        {
            "ticker": decision.normalized,
            "market": decision.market,
            "suggestions": suggestions,
            "hint": hint,
        },
    )


def validate_ticker_supported(model: Any) -> Any:
    """Model validator: reject a ticker the catalog's decision rejects.

    Raises a ``ticker_not_supported`` error (HTTP 422); see
    ``ticker_not_supported_error`` for its ``ctx``.
    """
    catalog = _active_catalog()
    if catalog is None:
        return model
    asset_type = getattr(model.asset_type, "value", model.asset_type)
    decision = catalog.decide(model.ticker, asset_type)
    if decision.rejected:
        raise ticker_not_supported_error(decision)
    return model
