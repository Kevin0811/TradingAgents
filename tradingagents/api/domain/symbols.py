"""Network-free symbol rules for the API's supported-symbols list.

Everything here is purely syntactic, like the core's
``tradingagents/dataflows/symbol_utils.py`` which it builds on (and never
modifies — the core is synced from upstream). It answers two questions:

* which market's list must hold a given ticker (``classify_market``), so the
  request validator knows which list to consult and when to fall back to the
  shape check because that list is not loaded, and
* how an API-only forex convenience (``USDTWD`` -> ``USDTWD=X``) extends the
  core's broker-symbol mapping for currencies the core does not know.
"""

from __future__ import annotations

import re
from functools import lru_cache

from tradingagents.dataflows.symbol_utils import bare_crypto_to_pair, normalize_symbol

MARKETS: tuple[str, ...] = ("tw", "us", "jp", "crypto", "fx")
SYMBOL_TYPES: tuple[str, ...] = ("equity", "etf", "crypto", "currency")

# Yahoo suffixes that always route to a market's list. Suffixes seen in a
# loaded tw/jp list are added on top at runtime (see ``classify_market``), so a
# JP regional suffix Yahoo returns (e.g. Fukuoka/Sapporo) routes to "jp" as
# soon as that list is loaded, without hard-coding it here.
BASE_SUFFIX_MARKETS: dict[str, str] = {"TW": "tw", "TWO": "tw", "T": "jp"}

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


@lru_cache(maxsize=256)
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


def classify_market(key: str, extra_suffixes: dict[str, str] | None = None) -> str | None:
    """Return the market whose list must hold ``key``, or None if none covers it.

    ``key`` is a ``list_key`` result. None means the ticker belongs to a market
    the list does not cover (``0700.HK``, ``^GSPC``, ``GC=F``, ``BTC-EUR``), so
    the validator applies only the shape check to it.
    """
    if not key:
        return None
    if key.endswith("=X"):
        return "fx" if fx_legs(key) else None
    if "=" in key or key.startswith("^"):
        return None
    if "." in key:
        suffixes = {**BASE_SUFFIX_MARKETS, **(extra_suffixes or {})}
        return suffixes.get(symbol_suffix(key))
    if "-" in key:
        quote = key.rsplit("-", 1)[1]
        if quote == "USD":
            return "crypto"
        if quote in _NON_USD_CRYPTO_QUOTES:
            return None
    return "us" if _US_SHAPE.fullmatch(key) else None
