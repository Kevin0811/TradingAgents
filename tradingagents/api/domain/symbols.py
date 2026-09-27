"""Network-free symbol rules for the API's supported-symbols list.

Everything here is purely syntactic, like the core's
``tradingagents/dataflows/symbols.py`` which it builds on (and never
modifies — the core is synced from upstream). It answers two questions:

* which market's list must hold a given ticker (``classify_market``), so the
  request validator knows which list to consult and when to fall back to the
  shape check because that list is not loaded, and
* how an API-only forex convenience (``USDTWD`` -> ``USDTWD=X``) extends the
  core's broker-symbol mapping for currencies the core does not know.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

from tradingagents.dataflows.symbols import crypto_base, normalize_symbol

MARKETS: tuple[str, ...] = ("tw", "us", "jp", "crypto", "fx")
SYMBOL_TYPES: tuple[str, ...] = ("equity", "etf", "crypto", "currency")

# The only Yahoo suffixes that route a ticker to a market's list. They are
# hard-coded and verified on purpose: a suffix is never learned from a loaded
# list, because the JP screener also returns foreign listings (``SAP.F``,
# ``BMW.F``) whose suffix would otherwise pull whole foreign exchanges into the
# jp list's reject policy. Such entries stay searchable; they just do not route.
SUFFIX_MARKETS: Mapping[str, str] = MappingProxyType({"TW": "tw", "TWO": "tw", "T": "jp"})

# Per-market policy for a ticker that is missing from a *loaded* list.
#
# * True  ("enforced"): the analyze request is rejected (422
#   ``ticker_not_supported`` with suggestions).
# * False ("soft"): the ticker is allowed with the shape check only; the
#   suggestions are still reported by ``GET /symbols/check`` as a hint.
#
# tw and jp lists come from exchange-wide screens and are near complete. The us
# list leaves out OTC venues and the crypto / fx lists come from a ranked
# lookup, so a miss there is not proof that Yahoo has no data. Markets missing
# from this mapping are soft. Tighten a market by flipping its value.
REJECT_UNLISTED_TICKERS: Mapping[str, bool] = MappingProxyType(
    {"tw": True, "jp": True, "us": False, "crypto": False, "fx": False}
)

# Currencies the API maps as forex on top of the core's own table. The core's
# ``normalize_symbol`` turns ``USDJPY`` into ``USDJPY=X`` but leaves ``USDTWD``
# alone because TWD is not in its (upstream-owned) currency set.
API_EXTRA_FOREX_CURRENCIES = frozenset({"TWD"})

# Quote currencies after a dash that mark a non-USD crypto pair (``BTC-EUR``),
# as opposed to a US share-class suffix (``BRK-B``). Those pairs are not on the
# crypto list (it holds ``-USD`` pairs only), so they are left uncovered.
_NON_USD_CRYPTO_QUOTES = frozenset(
    {"EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "KRW", "TWD", "BTC", "ETH", "USDT", "USDC"}
)

_US_SHAPE = re.compile(r"^[A-Z0-9][A-Z0-9\-]*$")

# A bare exchange code: always rejected by the shape check (Yahoo needs the
# suffix). ``2330`` -> ``2330.TW``.
_BARE_NUMERIC = re.compile(r"^\d{4,6}$")
# Suffix-less codes that look like a TW or JP listing. TW: 4-6 digits and an
# optional share-class letter (``2330``, ``00679B``, ``2881A``). JP: four
# characters, digits in the odd positions (``7203``, and the alphanumeric codes
# issued since 2024 such as ``130A``).
_TW_LOCAL_CODE = re.compile(r"^\d{4,6}[A-Z]?$")
_JP_LOCAL_CODE = re.compile(r"^\d[0-9A-Z]\d[0-9A-Z]$")


@lru_cache(maxsize=256)
def bare_crypto_to_pair(raw: str) -> str | None:
    """Return ``<BASE>-USD`` for an unquoted crypto base like ``BTC``, else None.

    The core's ``normalize_symbol`` deliberately leaves a bare base alone:
    several of them double as real exchange tickers (``BTC`` is a listed
    spot-bitcoin ETF), so rewriting them globally would misprice equity
    requests. The API knows when its input is a crypto asset, so it opts in
    here. The base is checked through the core's public ``crypto_base``.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper().rstrip("+")
    if not s or "-" in s:
        return None
    return f"{s}-USD" if crypto_base(f"{s}-USD") == s else None


def _is_core_currency(code: str) -> bool:
    """True when the core's forex rule treats ``code`` as a currency.

    Probes the public ``normalize_symbol`` instead of importing its private
    currency set: a six-letter pair of two known currencies is the only input
    it maps to a ``=X`` symbol (aliases such as ``XAUUSD`` map to futures).
    """
    other = "EUR" if code == "USD" else "USD"
    return normalize_symbol(code + other).endswith("=X")


def is_currency(code: str) -> bool:
    """True for a three-letter currency code known to the core or the API."""
    return (
        len(code) == 3
        and code.isalpha()
        and (code in API_EXTRA_FOREX_CURRENCIES or _is_core_currency(code))
    )


def api_forex_symbol(raw: str) -> str | None:
    """Map a six-letter forex pair the core leaves alone to Yahoo's ``PAIR=X``.

    Only pairs with at least one API-extra currency qualify (``USDTWD`` ->
    ``USDTWD=X``), which matches how the core maps ``USDJPY`` -> ``USDJPY=X``.
    Returns None for everything else, including pairs the core already maps.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper().rstrip("+")
    if len(s) != 6 or not s.isalpha():
        return None
    base, quote = s[:3], s[3:]
    if base not in API_EXTRA_FOREX_CURRENCIES and quote not in API_EXTRA_FOREX_CURRENCIES:
        return None
    if is_currency(base) and is_currency(quote):
        return f"{s}=X"
    return None


def to_yahoo_symbol(raw: str) -> str:
    """The core's ``normalize_symbol`` plus the API-only forex mapping."""
    return api_forex_symbol(raw) or normalize_symbol(raw)


def fx_legs(symbol: str) -> list[str] | None:
    """Return the non-USD currency legs of a Yahoo ``=X`` symbol, else None.

    ``TWD=X`` (USD/TWD) -> ``["TWD"]``; ``USDJPY=X`` -> ``["JPY"]``;
    ``EURUSD=X`` -> ``["EUR"]``; ``EURJPY=X`` -> ``["EUR", "JPY"]``.
    """
    if not symbol.endswith("=X"):
        return None
    body = symbol[:-2]
    if len(body) == 3 and body.isalpha():
        legs = [body]
    elif len(body) == 6 and body.isalpha():
        legs = [body[:3], body[3:]]
    else:
        return None
    legs = [leg for leg in legs if leg != "USD"]
    return legs or None


def canonical_fx_symbol(symbol: str) -> str:
    """``USDXXX=X`` -> ``XXX=X`` (the same USD/XXX quote); anything else unchanged."""
    if symbol.endswith("=X") and len(symbol) == 8 and symbol.startswith("USD"):
        return f"{symbol[3:6]}=X"
    return symbol


def list_key(raw: str, asset_type: str | None = None) -> str:
    """Normalise a ticker to the form the supported-symbols list stores.

    Upper-cases, applies the core broker mapping plus the API forex mapping,
    folds ``USDXXX=X`` into ``XXX=X`` and, for ``asset_type="crypto"``, reads a
    bare base the way the API does elsewhere (``BTC`` -> ``BTC-USD``).
    """
    s = raw.strip().upper()
    if asset_type == "crypto":
        s = bare_crypto_to_pair(s) or s
    return canonical_fx_symbol(to_yahoo_symbol(s))


def symbol_suffix(symbol: str) -> str:
    """The part after the last dot (``2330.TW`` -> ``TW``), or ``""``."""
    return symbol.rsplit(".", 1)[1] if "." in symbol else ""


def is_bare_numeric(code: str) -> bool:
    """True for a bare 4-6 digit code (``2330``), which the shape check rejects."""
    return bool(_BARE_NUMERIC.fullmatch(code.strip()))


def local_code_variants(key: str) -> list[str]:
    """Suffixed TW/JP symbols a suffix-less ``key`` may have meant.

    ``2330`` -> ``["2330.TW", "2330.TWO", "2330.T"]``; ``2881A`` ->
    ``["2881A.TW", "2881A.TWO"]``; ``130A`` -> ``["130A.T"]``; ``AAPL`` -> ``[]``.
    """
    variants = []
    if _TW_LOCAL_CODE.fullmatch(key):
        variants += [f"{key}.TW", f"{key}.TWO"]
    if _JP_LOCAL_CODE.fullmatch(key):
        variants.append(f"{key}.T")
    return variants


def classify_market(key: str) -> str | None:
    """Return the market whose list must hold ``key``, or None if none covers it.

    ``key`` is a ``list_key`` result. None means the ticker belongs to a market
    the list does not cover (``0700.HK``, ``^GSPC``, ``GC=F``, ``BTC-EUR``,
    ``SAP.F``), so the validator applies only the shape check to it. Only the
    suffixes in ``SUFFIX_MARKETS`` route a dotted symbol.
    """
    if not key:
        return None
    if key.endswith("=X"):
        return "fx" if fx_legs(key) else None
    if "=" in key or key.startswith("^"):
        return None
    if "." in key:
        return SUFFIX_MARKETS.get(symbol_suffix(key))
    if "-" in key:
        quote = key.rsplit("-", 1)[1]
        if quote == "USD":
            return "crypto"
        if quote in _NON_USD_CRYPTO_QUOTES:
            return None
    return "us" if _US_SHAPE.fullmatch(key) else None
