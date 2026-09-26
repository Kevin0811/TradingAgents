"""Supported-symbols catalog: the API's positive list of tradable tickers.

The catalog holds one list per market (``tw``, ``us``, ``jp``, ``crypto``,
``fx``) in memory, backed by a file cache and refreshed from Yahoo Finance.
Requests only ever read memory: loading the cache happens once at startup,
and every network fetch runs on a single background thread, one market at a
time. A failed refresh logs a warning and keeps serving the previous list.

Refresh policy:

* ``start()`` (app lifespan) loads the cache files and, when auto-refresh is
  on, queues every market whose list is missing or older than the TTL.
* A read of a missing/stale market re-queues it (auto-refresh only), at most
  once per ``retry_after`` so a failing Yahoo is not hammered.
* ``request_refresh()`` queues markets regardless of age (``POST
  /symbols/refresh``).
"""

from __future__ import annotations

import difflib
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.repositories import SymbolCacheRepository
from tradingagents.api.domain.symbols import (
    MARKETS,
    classify_market,
    fx_legs,
    list_key,
    symbol_suffix,
)

logger = logging.getLogger(__name__)

STATUS_READY = "ready"
STATUS_LOADING = "loading"
STATUS_UNAVAILABLE = "unavailable"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _MarketIndex:
    """One loaded market list plus the lookup structures built from it."""

    def __init__(self, symbol_list: SymbolList):
        self.list = symbol_list
        self.by_symbol = {e.symbol: e for e in symbol_list.entries}
        self.symbols = [e.symbol for e in symbol_list.entries]
        self.lower_names = [e.name.lower() for e in symbol_list.entries]
        self.suffixes = {symbol_suffix(s) for s in self.symbols} - {""}


@dataclass(frozen=True)
class TickerCheck:
    """Outcome of checking one ticker against the catalog.

    ``supported`` is True/False when the relevant market's list is loaded,
    and None when the catalog cannot tell (the ticker's market is not covered
    by any list, or that list is not loaded yet) — callers then fall back to
    their shape-only check.
    """

    key: str
    market: str | None
    supported: bool | None
    suggestions: tuple[str, ...] = ()


class SymbolCatalog:
    """In-memory positive list of supported symbols with a background refresher."""

    def __init__(
        self,
        repository: SymbolCacheRepository,
        source,
        *,
        ttl: timedelta = timedelta(days=7),
        auto_refresh: bool = True,
        retry_after: timedelta = timedelta(hours=1),
        clock: Callable[[], datetime] = _utcnow,
    ):
        """Initialize the catalog.

        Args:
            repository: Persistent cache of the per-market lists.
            source: Object with ``fetch(market) -> list[SymbolEntry]`` and
                ``source_label(market) -> str`` (see YahooSymbolSource).
            ttl: Age after which a list counts as stale.
            auto_refresh: Refresh missing/stale lists in the background.
            retry_after: Minimum gap between automatic attempts per market.
            clock: Returns the current UTC time (tests).
        """
        self._repository = repository
        self._source = source
        self._ttl = ttl
        self._auto_refresh = auto_refresh
        self._retry_after = retry_after
        self._clock = clock

        self._indexes: dict[str, _MarketIndex] = {}
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._refreshing: str | None = None
        self._thread: threading.Thread | None = None
        self._last_attempt: dict[str, datetime] = {}
        self._last_error: dict[str, str] = {}
        self._started = False
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load the cache files and queue missing/stale markets (non-blocking)."""
        self.load()
        self._started = True
        if self._auto_refresh:
            due = [m for m in MARKETS if not self.is_loaded(m) or self.is_stale(m)]
            if due:
                self.request_refresh(due)

    def shutdown(self) -> None:
        """Stop the refresher after the market it is fetching, if any."""
        self._stop.set()

    def load(self) -> None:
        """(Re)load every market's list from the cache."""
        for market in MARKETS:
            symbol_list = self._repository.load(market)
            if symbol_list is not None:
                self._indexes[market] = _MarketIndex(symbol_list)
                logger.info(
                    "Loaded %d %s symbols from cache (fetched %s)",
                    len(symbol_list.entries),
                    market,
                    symbol_list.fetched_at.isoformat(),
                )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_loaded(self, market: str) -> bool:
        return market in self._indexes

    def get_list(self, market: str) -> SymbolList | None:
        index = self._indexes.get(market)
        return index.list if index else None

    def is_stale(self, market: str) -> bool:
        """True when the market has a list older than the TTL."""
        index = self._indexes.get(market)
        if index is None:
            return False
        fetched_at = index.list.fetched_at
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return self._clock() - fetched_at > self._ttl

    def is_refreshing(self, market: str) -> bool:
        with self._lock:
            return market == self._refreshing or market in self._pending

    def status(self, market: str) -> str:
        """``ready`` (list loaded), ``loading`` (queued/fetching) or ``unavailable``."""
        self._touch(market)
        if self.is_loaded(market):
            return STATUS_READY
        if self.is_refreshing(market):
            return STATUS_LOADING
        return STATUS_UNAVAILABLE

    def last_error(self, market: str) -> str | None:
        return self._last_error.get(market)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def search(
        self,
        market: str,
        q: str | None = None,
        symbol_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[SymbolEntry], int]:
        """Search one market's list; returns ``(page, total_matches)``.

        ``q`` matches a symbol prefix or a name substring, case-insensitively.
        An exact symbol match sorts first, then symbol-prefix matches, then
        name matches, each group in symbol order.
        """
        self._touch(market)
        index = self._indexes.get(market)
        if index is None:
            return [], 0
        q_upper = (q or "").strip().upper()
        q_lower = q_upper.lower()
        exact: list[SymbolEntry] = []
        prefix: list[SymbolEntry] = []
        by_name: list[SymbolEntry] = []
        for entry, lower_name in zip(index.list.entries, index.lower_names, strict=True):
            if symbol_type and entry.type != symbol_type:
                continue
            if not q_upper:
                prefix.append(entry)
            elif entry.symbol == q_upper:
                exact.append(entry)
            elif entry.symbol.startswith(q_upper):
                prefix.append(entry)
            elif q_lower in lower_name:
                by_name.append(entry)
        matches = exact + prefix + by_name
        return matches[offset : offset + limit], len(matches)

    def market_for(self, key: str) -> str | None:
        """The market whose list must hold ``key`` (a ``list_key`` result)."""
        indexes = dict(self._indexes)  # snapshot: the refresher may add markets
        extra = {
            suffix: market
            for market in ("tw", "jp")
            if market in indexes
            for suffix in indexes[market].suffixes
        }
        return classify_market(key, extra)

    def get(self, key: str) -> SymbolEntry | None:
        """Exact lookup of a ``list_key`` across every loaded market."""
        market = self.market_for(key)
        if market and market in self._indexes:
            entry = self._indexes[market].by_symbol.get(key)
            if entry:
                return entry
        for index in list(self._indexes.values()):
            entry = index.by_symbol.get(key)
            if entry:
                return entry
        return None

    def suggest(self, key: str, market: str | None = None, n: int = 3) -> list[str]:
        """Up to ``n`` closest symbols to ``key`` in ``market`` (or all loaded lists)."""
        if market is not None:
            index = self._indexes.get(market)
            candidates = index.symbols if index else []
        else:
            candidates = [s for index in list(self._indexes.values()) for s in index.symbols]
        return difflib.get_close_matches(key, candidates, n=n, cutoff=0.6)

    def check(self, ticker: str, asset_type: str | None = None) -> TickerCheck:
        """Check a request ticker against the positive list."""
        key = list_key(ticker, asset_type)
        market = self.market_for(key)
        if asset_type == "crypto" and market in (None, "us") and key.isalnum():
            # A crypto request for a base the core has no pair rule for
            # (e.g. "PEPE") means the coin's USD pair.
            key, market = f"{key}-USD", "crypto"
        if market is None:
            return TickerCheck(key, None, None)
        self._touch(market)
        index = self._indexes.get(market)
        if index is None:
            return TickerCheck(key, market, None)
        if market == "fx":
            legs = fx_legs(key) or []
            supported = all(f"{leg}=X" in index.by_symbol for leg in legs)
            missing = [f"{leg}=X" for leg in legs if f"{leg}=X" not in index.by_symbol]
            suggestions = [s for m in missing for s in self.suggest(m, market, n=1)]
            return TickerCheck(key, market, supported, tuple(suggestions[:3]))
        if key in index.by_symbol:
            return TickerCheck(key, market, True)
        return TickerCheck(key, market, False, tuple(self.suggest(key, market)))

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def request_refresh(self, markets: Iterable[str]) -> list[str]:
        """Queue ``markets`` for a background refresh; returns the queued markets.

        Markets already queued or being fetched are not queued twice but are
        still reported. Never blocks on the network.
        """
        requested = [m for m in dict.fromkeys(markets) if m in MARKETS]
        with self._lock:
            for market in requested:
                if market != self._refreshing and market not in self._pending:
                    self._pending.append(market)
            if requested and self._thread is None and not self._stop.is_set():
                self._thread = threading.Thread(
                    target=self._drain, name="symbol-catalog-refresh", daemon=True
                )
                self._thread.start()
        return requested

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the refresher is idle (tests and scripts). True if idle."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def refresh_market(self, market: str) -> bool:
        """Fetch ``market`` now (blocking) and swap it in; True on success.

        On failure the previous list stays in memory and on disk.
        """
        self._last_attempt[market] = self._clock()
        try:
            entries = self._source.fetch(market)
        except Exception as exc:
            self._last_error[market] = f"{type(exc).__name__}: {exc}"
            kept = self.get_list(market)
            logger.warning(
                "Refreshing the %s symbol list failed (%s); %s",
                market,
                self._last_error[market],
                f"keeping the {len(kept.entries)}-entry list from {kept.fetched_at.isoformat()}"
                if kept
                else "no cached list to fall back on",
            )
            return False
        symbol_list = SymbolList(
            market=market,
            entries=tuple(entries),
            fetched_at=self._clock(),
            source=self._source.source_label(market),
        )
        try:
            self._repository.save(symbol_list)
        except OSError as exc:
            logger.warning("Could not write the %s symbol cache: %s", market, exc)
        self._indexes[market] = _MarketIndex(symbol_list)
        self._last_error.pop(market, None)
        logger.info("Refreshed the %s symbol list: %d entries", market, len(entries))
        return True

    def _drain(self) -> None:
        while True:
            with self._lock:
                if self._stop.is_set():
                    self._pending.clear()
                if not self._pending:
                    self._refreshing = None
                    self._thread = None
                    return
                market = self._pending.pop(0)
                self._refreshing = market
            try:
                self.refresh_market(market)
            except Exception:  # pragma: no cover - refresh_market already guards
                logger.exception("Unexpected error refreshing the %s symbol list", market)
            finally:
                with self._lock:
                    self._refreshing = None

    def _touch(self, market: str) -> None:
        """Queue a missing/stale market after startup, rate-limited per market."""
        if not (self._started and self._auto_refresh) or self._stop.is_set():
            return
        if self.is_loaded(market) and not self.is_stale(market):
            return
        if self.is_refreshing(market):
            return
        last = self._last_attempt.get(market)
        if last is not None and self._clock() - last < self._retry_after:
            return
        self.request_refresh([market])


# ---------------------------------------------------------------------------
# Active catalog
# ---------------------------------------------------------------------------
#
# Pydantic field/model validators run without access to the request or the
# app, so the request validator reaches the app's catalog through this
# module-level slot (the same pattern as tradingagents.api.config.get_config).
# create_app() registers each new app's catalog; None means "no positive list",
# and the validator then applies only its shape check.

_active_catalog: SymbolCatalog | None = None


def get_active_catalog() -> SymbolCatalog | None:
    """Return the catalog request validators check tickers against, if any."""
    return _active_catalog


def set_active_catalog(catalog: SymbolCatalog | None) -> None:
    """Register (or clear, with None) the catalog request validators use."""
    global _active_catalog
    _active_catalog = catalog
