"""Fetch the supported-symbols lists from Yahoo Finance via yfinance.

Equities and ETFs come from yfinance's built-in screener (``yf.screen`` with an
``EquityQuery`` / ``ETFQuery``), crypto and currencies from ``yf.Lookup``.
Yahoo's screener is undocumented: it caps a page at 250 rows, may rate-limit,
and may change shape. This module therefore paginates with a pause between
requests and a hard page cap, and raises on any failure, including a query
that stops short of its reported total or runs into the page cap, so the
caller keeps the previous list instead of saving a partial one.

Yahoo also fails single pages transiently (HTTP 500 "Server caught an
exception" deep into a query, or 429). Each screener page and lookup request
is therefore retried on a 5xx or 429, up to ``PAGE_RETRIES`` times with a
growing pause (2 s, 5 s, 10 s, or the server's ``Retry-After``), and once
(``NETWORK_RETRIES``) on a timeout or connection error; other errors,
including every other 4xx, fail the market at once.

With a ``RefreshGate``, a fetch that yields (``fetch(market,
yield_when_busy=True)``, an automatic refresh) pauses between pages while
TradingAgents is busy. A pause longer than ``RESTART_QUERY_AFTER_PAUSE_SECONDS``
restarts the current screener query from offset 0, since offset paging over
a listing that changed meanwhile could skip or repeat rows. Every sleep
(page delay, retry pause) ends early when the gate is stopped, and the gate's
stop is checked before every request, yielding or not.

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

import importlib
import logging
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from tradingagents.api.domain.entities import SymbolEntry
from tradingagents.api.domain.services.refresh_activity import RefreshGate
from tradingagents.api.domain.symbols import bare_crypto_to_pair, fx_legs, is_currency

logger = logging.getLogger(__name__)


def _yfinance() -> Any:
    """Load yfinance at call time.

    Upstream's ``tests/test_layering.py`` allows vendor-library imports only in
    ``tradingagents/dataflows``, so that a vendor failure is always reported as
    unavailable data, never as a fact about the market. The core has no
    screener or lookup, and it is synced from upstream, so this adapter calls
    yfinance itself. It keeps to that rule's intent: every failure becomes a
    ``SymbolSourceError``, and the catalog reports the list as ``unavailable``.
    """
    return importlib.import_module("yfinance")


MARKET_EXCHANGES: dict[str, tuple[str, ...]] = {
    "tw": ("TAI", "TWO"),
    "us": ("NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS"),
    "jp": ("JPX", "OSA", "FKA", "SAP"),
}
US_OTC_EXCHANGES: tuple[str, ...] = ("PNK", "OEM", "OQB", "OQX")

PAGE_SIZE = 250  # Yahoo's hard cap per screener page

# Retries of one screener page / lookup request after a 5xx or 429 (on top of
# the first attempt). The pause before retry n is the base delay times
# _RETRY_BACKOFF[n]: 2 s, 5 s, 10 s. A Retry-After header, when present,
# replaces it (capped at RETRY_AFTER_MAX_SECONDS).
PAGE_RETRIES = 3
PAGE_RETRY_BASE_DELAY_SECONDS = 2.0
_RETRY_BACKOFF = (1.0, 2.5, 5.0)
RETRY_AFTER_MAX_SECONDS = 60.0
# Retries of one request after a timeout / connection error (curl_cffi,
# requests or the builtin exceptions), after PAGE_RETRY_BASE_DELAY_SECONDS.
NETWORK_RETRIES = 1
# A pause for activity longer than this restarts the current screener query.
RESTART_QUERY_AFTER_PAUSE_SECONDS = 15 * 60
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


_STATUS_IN_MESSAGE = re.compile(r"\bHTTP Error (\d{3})\b|\b(\d{3}) (?:Server|Client) Error\b")
# Error texts that carry no status code: yfinance reports a lookup's error
# payload, its own rate limiting and Yahoo's outage page this way.
_STATUS_BY_TEXT = (
    ("Internal Server Error", 500),
    ("Server caught an exception", 500),
    ("Bad Gateway", 502),
    ("Service Unavailable", 503),
    ("Gateway Timeout", 504),
    ("CURRENTLY DOWN", 503),
    ("Too Many Requests", 429),
)


def http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status of a yfinance / HTTP-client error, else None.

    Reads ``exc.response.status_code`` (curl_cffi and requests HTTPError),
    then ``exc.status_code``, then the status in the message ("HTTP Error
    500"), then the few status-less texts yfinance raises for Yahoo errors.
    """
    response = getattr(exc, "response", None)
    for candidate in (getattr(response, "status_code", None), getattr(exc, "status_code", None)):
        if isinstance(candidate, int) and 100 <= candidate <= 599:
            return candidate
    if type(exc).__name__ == "YFRateLimitError":
        return 429
    text = str(exc)
    match = _STATUS_IN_MESSAGE.search(text)
    if match:
        return int(match.group(1) or match.group(2))
    for needle, status in _STATUS_BY_TEXT:
        if needle in text:
            return status
    return None


def is_retryable_status(status: int | None) -> bool:
    """True for 429 and 5xx: worth retrying the same request."""
    return status is not None and (status == 429 or 500 <= status <= 599)


_NETWORK_ERROR_NAMES = ("Timeout", "ConnectionError")
_NETWORK_ERROR_MODULES = ("curl_cffi", "requests", "urllib3")


def is_network_error(exc: BaseException) -> bool:
    """True for a timeout or connection error (curl_cffi, requests or builtin)."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(
        cls.__name__ in _NETWORK_ERROR_NAMES
        and cls.__module__.split(".", 1)[0] in _NETWORK_ERROR_MODULES
        for cls in type(exc).__mro__
    )


def retry_after_seconds(exc: BaseException) -> float | None:
    """The response's ``Retry-After`` in seconds (numeric form only), capped."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:  # pragma: no cover - exotic header containers
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, RETRY_AFTER_MAX_SECONDS)


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
        page_delay_seconds: float | Callable[[], float] = 2.0,
        max_pages: int = 200,
        include_otc: bool = False,
        screen: Callable[..., dict[str, Any]] | None = None,
        lookup_factory: Callable[[str], Any] | None = None,
        sleep: Callable[[float], None] | None = None,
        gate: RefreshGate | None = None,
    ):
        """Initialize the source.

        Args:
            page_delay_seconds: Pause before every request after the first; a
                callable is read before each pause (live settings).
            max_pages: Safety bound on pages per screener query.
            include_otc: Also screen the US OTC venues.
            screen: Replacement for ``yfinance.screen`` (tests).
            lookup_factory: Replacement for ``yfinance.Lookup`` (tests).
            sleep: Replacement for the sleep between requests (tests); by
                default the gate's stop-aware sleep, or ``time.sleep``
                without a gate.
            gate: Consulted before every request after the first of a
                yielding fetch (pauses it while TradingAgents is busy), and
                for its stop before every request (see RefreshGate).
        """
        self._page_delay = page_delay_seconds
        self._gate = gate
        self._market = ""
        self._yield = False
        self._max_pages = max(1, int(max_pages))
        self._include_otc = include_otc
        self._screen = screen
        self._lookup_factory = lookup_factory
        self._sleep = sleep or (gate.sleep if gate is not None else time.sleep)
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

    def fetch(self, market: str, yield_when_busy: bool = True) -> list[SymbolEntry]:
        """Fetch ``market``'s full list, sorted by symbol.

        ``yield_when_busy`` (only meaningful with a gate) pauses the fetch
        between pages while TradingAgents is busy; the catalog passes False
        for a manual refresh and a missing list.

        Raises:
            SymbolSourceError: Nothing usable came back, or a screener query
                came back incomplete (short of its total, or at the page cap).
            RefreshInterrupted: The gate was stopped (shutdown).
            RefreshPreempted: A paused fetch gave way to manual work.
            Exception: Any yfinance / HTTP error, unchanged.
        """
        self._requests = 0
        self._market = market
        self._yield = bool(yield_when_busy and self._gate is not None)
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
        yf = _yfinance()
        EquityQuery, ETFQuery = yf.EquityQuery, yf.ETFQuery  # noqa: N806

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

    def _screen_pages(self, query: Any, label: str) -> list[dict[str, Any]]:
        """Return every row of one screener query, or raise if that is not possible.

        With a ``total`` in the response, pages are requested until ``offset``
        reaches it; an empty page before that means Yahoo stopped short. With
        no ``total``, only an empty page ends the query (a short page may just
        be Yahoo trimming one). Hitting the page cap first raises too: a
        partial list must never replace a complete one. After a pause longer
        than ``RESTART_QUERY_AFTER_PAUSE_SECONDS`` the query starts over.
        """
        screen = self._screen or self._default_screen()
        rows: list[dict[str, Any]] = []
        offset = 0
        total: int | None = None
        pages = 0
        while pages < self._max_pages:
            paused = self._pace()
            if offset and paused > RESTART_QUERY_AFTER_PAUSE_SECONDS:
                logger.info(
                    "Restarting screener %s from offset 0: the refresh paused for %.0f min "
                    "and the listing may have shifted",
                    label,
                    paused / 60,
                )
                rows, offset, total, pages = [], 0, None, 0
            pages += 1
            result = self._with_retries(
                lambda offset=offset: screen(
                    query, offset=offset, size=PAGE_SIZE, sortField="ticker", sortAsc=True
                ),
                f"screener {label} offset={offset}",
            )
            quotes = (result or {}).get("quotes") or []
            reported = (result or {}).get("total")
            if isinstance(reported, int) and not isinstance(reported, bool):
                total = reported
            rows.extend(quotes)
            offset += len(quotes)
            if total is not None:
                if offset >= total:
                    return rows
                if not quotes:
                    raise SymbolSourceError(
                        f"Symbol screener {label} returned an empty page after {offset} of "
                        f"{total} rows"
                    )
            elif not quotes:
                return rows
        raise SymbolSourceError(
            f"Symbol screener {label} hit the {self._max_pages}-page cap after {offset} rows"
            + (f" of {total}" if total is not None else "")
            + "; raise TRADINGAGENTS_SYMBOLS_MAX_PAGES"
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

        def request():
            # A fresh Lookup per attempt: yfinance caches results per instance.
            lookup = factory(LOOKUP_QUERY)
            if kind == "crypto":
                return lookup.get_cryptocurrency(count=LOOKUP_COUNT)
            return lookup.get_currency(count=LOOKUP_COUNT)

        frame = self._with_retries(request, f"lookup {kind} count={LOOKUP_COUNT}")
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

    def _with_retries(self, request: Callable[[], Any], label: str) -> Any:
        """Run one request, retrying it on a 5xx / 429 or a network error.

        Up to ``PAGE_RETRIES`` retries for a status, ``NETWORK_RETRIES`` for
        a timeout / connection error; the gate's stop ends the pause before a
        retry and is checked before it.
        """
        status_retries = network_retries = 0
        while True:
            try:
                return request()
            except Exception as exc:
                status = http_status(exc)
                if is_retryable_status(status) and status_retries < PAGE_RETRIES:
                    delay = retry_after_seconds(exc)
                    if delay is None:
                        delay = PAGE_RETRY_BASE_DELAY_SECONDS * _RETRY_BACKOFF[status_retries]
                    status_retries += 1
                    reason, attempt, limit = f"HTTP {status}", status_retries, PAGE_RETRIES
                elif status is None and is_network_error(exc) and network_retries < NETWORK_RETRIES:
                    delay = PAGE_RETRY_BASE_DELAY_SECONDS
                    network_retries += 1
                    reason, attempt, limit = "a network error", network_retries, NETWORK_RETRIES
                else:
                    raise
                logger.warning(
                    "Yahoo %s failed with %s (%s: %s); retry %d of %d in %.1fs",
                    label,
                    reason,
                    type(exc).__name__,
                    str(exc)[:200],
                    attempt,
                    limit,
                    delay,
                )
                self._sleep_checked(delay)

    def _pace(self) -> float:
        """Before every request: check for a stop; between requests, yield and pause.

        Returns how long the gate paused this fetch for activity (seconds).
        """
        self._check_stopped()
        paused = 0.0
        if self._requests:
            if self._yield:
                # Between pages only: the first request of a fetch starts at once.
                result = self._gate.wait_until_idle(
                    f"{self._market}, after {self._requests} request(s)"
                )
                paused = result.paused_seconds
                if result.released:
                    self._yield = False  # a manual refresh took this fetch over
            delay = self._page_delay() if callable(self._page_delay) else self._page_delay
            delay = max(0.0, float(delay))
            if delay:
                self._sleep_checked(delay)
        self._requests += 1
        return paused

    def _sleep_checked(self, seconds: float) -> None:
        self._sleep(seconds)
        self._check_stopped()

    def _check_stopped(self) -> None:
        if self._gate is not None:
            self._gate.check_stopped()

    @staticmethod
    def _default_screen() -> Callable[..., dict[str, Any]]:
        return _yfinance().screen

    @staticmethod
    def _default_lookup() -> Callable[[str], Any]:
        return _yfinance().Lookup

    @staticmethod
    def _yf_version() -> str:
        try:
            return getattr(_yfinance(), "__version__", "unknown")
        except ImportError:  # pragma: no cover - yfinance is a hard dependency
            return "unknown"
