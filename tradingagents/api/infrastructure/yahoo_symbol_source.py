"""Fetch the supported-symbols lists from Yahoo Finance via yfinance.

Equities and ETFs come from yfinance's built-in screener (``yf.screen`` with an
``EquityQuery`` / ``ETFQuery``), crypto and currencies from ``yf.Lookup``.
Yahoo's screener is undocumented: it caps a page at 250 rows, may rate-limit,
and may change shape. This module therefore paginates with a pause between
requests and a hard page cap, and raises on any failure so the caller keeps
the previous list instead of saving a partial one.

Exchange codes (Yahoo's own, as listed in yfinance's screener value map):

* tw: TAI (TWSE, ``.TW``), TWO (TPEx, ``.TWO``)
* us: NYQ (NYSE), NMS/NGM/NCM (Nasdaq tiers), ASE (NYSE American),
  PCX (NYSE Arca), BTS (Cboe BZX). OTC venues (PNK, OEM, OQB, OQX) are left
  out unless ``include_otc`` is set: they list tens of thousands of thinly
  traded names most of which have no usable Yahoo history.
* jp: JPX (Tokyo, ``.T``), OSA, FKA (Fukuoka), SAP (Sapporo)

Equities are screened with ``intradaymarketcap > 0``, which drops warrants and
other derivative listings (on TW it cuts ~23k rows to ~2.3k) while keeping
preferred shares and TDRs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from tradingagents.api.domain.entities import SymbolEntry
from tradingagents.api.domain.symbols import fx_legs, is_currency
from tradingagents.dataflows.symbol_utils import bare_crypto_to_pair

logger = logging.getLogger(__name__)

MARKET_EXCHANGES: dict[str, tuple[str, ...]] = {
    "tw": ("TAI", "TWO"),
    "us": ("NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS"),
    "jp": ("JPX", "OSA", "FKA", "SAP"),
}
US_OTC_EXCHANGES: tuple[str, ...] = ("PNK", "OEM", "OQB", "OQX")

PAGE_SIZE = 250  # Yahoo's hard cap per screener page
LOOKUP_QUERY = "USD"
LOOKUP_COUNT = 1000

# Instruments the core pipeline explicitly supports (its crypto bases and forex
# currencies, plus the API's TWD). They are merged into a successful crypto /
# fx fetch so the positive list never rejects a symbol that works today, even
# if Yahoo's lookup ranking leaves one out.
_SEED_CRYPTO_BASES = (
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "LTC",
    "BCH",
    "DOT",
    "AVAX",
    "LINK",
)
_SEED_CURRENCIES = (
    "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "CNH", "HKD", "SGD", "SEK",
    "NOK", "DKK", "PLN", "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB", "TWD",
)  # fmt: skip


class SymbolSourceError(RuntimeError):
    """Raised when a market's list could not be fetched completely."""


def _first(quote: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = quote.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class YahooSymbolSource:
    """Builds each market's supported-symbols list from Yahoo Finance."""

    def __init__(
        self,
        *,
        page_delay_seconds: float = 1.0,
        max_pages: int = 200,
        include_otc: bool = False,
        screen: Callable[..., dict[str, Any]] | None = None,
        lookup_factory: Callable[[str], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Initialize the source.

        Args:
            page_delay_seconds: Pause before every request after the first.
            max_pages: Safety bound on pages per screener query.
            include_otc: Also screen the US OTC venues.
            screen: Replacement for ``yfinance.screen`` (tests).
            lookup_factory: Replacement for ``yfinance.Lookup`` (tests).
            sleep: Replacement for ``time.sleep`` (tests).
        """
        self._page_delay = max(0.0, float(page_delay_seconds))
        self._max_pages = max(1, int(max_pages))
        self._include_otc = include_otc
        self._screen = screen
        self._lookup_factory = lookup_factory
        self._sleep = sleep
        self._requests = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exchanges(self, market: str) -> tuple[str, ...]:
        """Return the Yahoo exchange codes screened for ``market``."""
        exchanges = MARKET_EXCHANGES[market]
        if market == "us" and self._include_otc:
            exchanges = exchanges + US_OTC_EXCHANGES
        return exchanges

    def source_label(self, market: str) -> str:
        """Describe where ``market``'s list comes from (stored in the cache)."""
        kind = "lookup" if market in ("crypto", "fx") else "screener"
        return f"yfinance {self._yf_version()} {kind}"

    def fetch(self, market: str) -> list[SymbolEntry]:
        """Fetch ``market``'s full list, sorted by symbol.

        Raises:
            SymbolSourceError: Nothing usable came back.
            Exception: Any yfinance / HTTP error, unchanged.
        """
        self._requests = 0
        if market in MARKET_EXCHANGES:
            entries = self._fetch_screened(market)
        elif market == "crypto":
            entries = self._fetch_crypto()
        elif market == "fx":
            entries = self._fetch_fx()
        else:
            raise ValueError(f"unknown market {market!r}")
        if not entries:
            raise SymbolSourceError(f"Yahoo returned no symbols for market {market!r}")
        unique = {e.symbol: e for e in entries}
        return sorted(unique.values(), key=lambda e: e.symbol)

    # ------------------------------------------------------------------
    # Screener (tw / us / jp)
    # ------------------------------------------------------------------

    def _fetch_screened(self, market: str) -> list[SymbolEntry]:
        from yfinance import EquityQuery, ETFQuery

        entries: list[SymbolEntry] = []
        for exchange in self.exchanges(market):
            equity_query = EquityQuery(
                "and",
                [
                    EquityQuery("eq", ["exchange", exchange]),
                    EquityQuery("gt", ["intradaymarketcap", 0]),
                ],
            )
            for quote in self._screen_pages(equity_query, f"{market}/{exchange}/equity"):
                # Belt and braces: the server-side filter should already drop these.
                cap = quote.get("marketCap")
                if isinstance(cap, (int, float)) and cap <= 0:
                    continue
                entry = self._to_entry(quote, market, "equity", exchange)
                if entry:
                    entries.append(entry)
            etf_query = ETFQuery("eq", ["exchange", exchange])
            for quote in self._screen_pages(etf_query, f"{market}/{exchange}/etf"):
                entry = self._to_entry(quote, market, "etf", exchange)
                if entry:
                    entries.append(entry)
        return entries

    def _screen_pages(self, query: Any, label: str) -> Iterator[dict[str, Any]]:
        screen = self._screen or self._default_screen()
        offset = 0
        for _ in range(self._max_pages):
            self._pace()
            result = screen(query, offset=offset, size=PAGE_SIZE, sortField="ticker", sortAsc=True)
            quotes = (result or {}).get("quotes") or []
            yield from quotes
            offset += len(quotes)
            total = (result or {}).get("total")
            if len(quotes) < PAGE_SIZE or (isinstance(total, int) and offset >= total):
                return
        logger.warning(
            "Symbol screener %s stopped at the %d-page cap (%d rows); the list may be incomplete",
            label,
            self._max_pages,
            offset,
        )

    @staticmethod
    def _to_entry(
        quote: dict[str, Any], market: str, kind: str, exchange: str
    ) -> SymbolEntry | None:
        symbol = _first(quote, "symbol").upper()
        if not symbol:
            return None
        return SymbolEntry(
            symbol=symbol,
            name=_first(quote, "longName", "shortName", "displayName"),
            exchange=_first(quote, "exchange") or exchange,
            type=kind,
            market=market,
        )

    # ------------------------------------------------------------------
    # Lookup (crypto / fx)
    # ------------------------------------------------------------------

    def _lookup_rows(self, kind: str) -> list[dict[str, Any]]:
        factory = self._lookup_factory or self._default_lookup()
        self._pace()
        lookup = factory(LOOKUP_QUERY)
        if kind == "crypto":
            frame = lookup.get_cryptocurrency(count=LOOKUP_COUNT)
        else:
            frame = lookup.get_currency(count=LOOKUP_COUNT)
        if frame is None or getattr(frame, "empty", True):
            return []
        # Lookup returns a DataFrame indexed by symbol.
        return frame.reset_index().to_dict("records")

    def _fetch_crypto(self) -> list[SymbolEntry]:
        entries = []
        for row in self._lookup_rows("crypto"):
            symbol = _first(row, "symbol").upper()
            base, _, quote = symbol.rpartition("-")
            if quote != "USD" or not base or not base.isalnum():
                continue
            entries.append(
                SymbolEntry(
                    symbol=symbol,
                    name=_first(row, "shortName", "longName", "name") or base,
                    exchange=_first(row, "exchange") or "CCC",
                    type="crypto",
                    market="crypto",
                )
            )
        if entries:
            entries += self._seed_crypto(e.symbol for e in entries)
        return entries

    def _fetch_fx(self) -> list[SymbolEntry]:
        entries = []
        for row in self._lookup_rows("fx"):
            legs = fx_legs(_first(row, "symbol").upper())
            # Keep only pairs quoted against USD; emit them in Yahoo's short
            # ``XXX=X`` form (USD/XXX), whichever way round Yahoo listed them.
            if not legs or len(legs) != 1:
                continue
            code = legs[0]
            entries.append(self._currency_entry(code, _first(row, "exchange") or "CCY"))
        if entries:
            entries += self._seed_fx(e.symbol for e in entries)
        return entries

    @staticmethod
    def _currency_entry(code: str, exchange: str = "CCY") -> SymbolEntry:
        return SymbolEntry(
            symbol=f"{code}=X",
            name=f"USD/{code}",
            exchange=exchange,
            type="currency",
            market="fx",
        )

    @staticmethod
    def _seed_crypto(present: Iterable[str]) -> list[SymbolEntry]:
        have = set(present)
        seeds = []
        for base in _SEED_CRYPTO_BASES:
            symbol = bare_crypto_to_pair(base)
            if symbol and symbol not in have:
                seeds.append(SymbolEntry(symbol, base, "CCC", "crypto", "crypto"))
        return seeds

    @classmethod
    def _seed_fx(cls, present: Iterable[str]) -> list[SymbolEntry]:
        have = set(present)
        return [
            cls._currency_entry(code)
            for code in _SEED_CURRENCIES
            if f"{code}=X" not in have and is_currency(code)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pace(self) -> None:
        if self._requests and self._page_delay:
            self._sleep(self._page_delay)
        self._requests += 1

    @staticmethod
    def _default_screen() -> Callable[..., dict[str, Any]]:
        import yfinance

        return yfinance.screen

    @staticmethod
    def _default_lookup() -> Callable[[str], Any]:
        import yfinance

        return yfinance.Lookup

    @staticmethod
    def _yf_version() -> str:
        try:
            import yfinance

            return getattr(yfinance, "__version__", "unknown")
        except ImportError:  # pragma: no cover - yfinance is a hard dependency
            return "unknown"
