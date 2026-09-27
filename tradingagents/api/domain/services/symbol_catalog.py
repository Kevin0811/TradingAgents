"""Supported-symbols catalog: the API's positive list of tradable tickers.

The catalog holds one list per market (``tw``, ``us``, ``jp``, ``crypto``,
``fx``) in memory, backed by a file cache and refreshed from Yahoo Finance.
Requests only ever read memory: loading the cache happens once at startup,
and every network fetch runs on a single background thread, one market at a
time. A failed refresh logs a warning and keeps serving the previous list, and
so does a refresh whose result looks truncated (see ``refresh_market``).

Ticker decisions (``decide``) are the single code path behind both the analyze
request validator and ``GET /symbols/check``. A miss in a loaded list is
rejected only in the markets ``REJECT_UNLISTED_TICKERS`` enforces (tw, jp);
elsewhere it is allowed and the suggestions are only a hint.

Refresh policy:

* ``start()`` (app lifespan) loads the cache files and, when auto-refresh is
  on, queues every market whose list is missing or older than the TTL.
* A read of a missing/stale market re-queues it (auto-refresh only), at most
  once per ``retry_after`` so a failing Yahoo is not hammered.
* A miss in a list older than ``miss_refresh_age`` (24h) queues that market
  too, under the same per-market rate limit, so a newly listed symbol shows up
  without waiting for the TTL.
* ``request_manual_refresh()`` (``POST /symbols/refresh``) queues markets
  regardless of age, except those fetched within ``refresh_cooldown`` unless
  forced.
"""

from __future__ import annotations

import difflib
import logging
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.repositories import SymbolCacheRepository
from tradingagents.api.domain.symbols import (
    MARKETS,
    REJECT_UNLISTED_TICKERS,
    classify_market,
    fx_legs,
    is_bare_numeric,
    list_key,
    local_code_variants,
)

logger = logging.getLogger(__name__)

STATUS_READY = "ready"
STATUS_LOADING = "loading"
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNCOVERED = "uncovered"  # decisions only: no list covers the ticker

# A refresh must keep at least this share of the previous list's entries, or
# it is treated as a truncated fetch and the previous list is kept.
MIN_REFRESH_KEEP_RATIO = 0.8

REASON_NOT_LISTED = "not_listed"
REASON_MISSING_SUFFIX = "missing_suffix"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class SymbolSource(Protocol):
    """Where the catalog fetches a market's list from (see YahooSymbolSource)."""

    def fetch(self, market: str) -> list[SymbolEntry]:
        """Return ``market``'s complete list; raise on any failure."""
        ...

    def source_label(self, market: str) -> str:
        """Describe the source of ``market``'s list (stored in the cache)."""
        ...


class _MarketIndex:
    """One loaded market list plus the lookup structures built from it."""

    def __init__(self, symbol_list: SymbolList):
        self.list = symbol_list
        self.by_symbol = {e.symbol: e for e in symbol_list.entries}
        self.symbols = [e.symbol for e in symbol_list.entries]
        self.lower_names = [e.name.lower() for e in symbol_list.entries]


@dataclass(frozen=True)
class TickerDecision:
    """The catalog's verdict on one request ticker.

    ``supported`` is True/False when the relevant list is loaded, and None
    when the catalog cannot tell (no list covers the ticker's market, or that
    list is not loaded yet); callers then rely on the shape check alone.
    ``enforced`` says whether a miss in ``market`` is rejected; ``rejected``
    combines the two and is what the analyze validator acts on.
    """

    ticker: str  # as given, stripped
    normalized: str  # the list form (``list_key``), e.g. "TWD=X", "BTC-USD"
    market: str | None
    supported: bool | None
    enforced: bool
    status: str  # ready | loading | unavailable | uncovered
    suggestions: tuple[str, ...] = ()
    entry: SymbolEntry | None = None
    last_error: str | None = None
    reason: str | None = None  # why supported is False: not_listed | missing_suffix

    @property
    def rejected(self) -> bool:
        return self.supported is False and self.enforced


@dataclass(frozen=True)
class SymbolLookup:
    """Outcome of an exact lookup (``GET /symbols/{symbol}``)."""

    symbol: str  # normalised
    market: str | None
    entry: SymbolEntry | None
    status: str
    suggestions: tuple[str, ...] = ()
    last_error: str | None = None


class SymbolCatalog:
    """In-memory positive list of supported symbols with a background refresher."""

    def __init__(
        self,
        repository: SymbolCacheRepository,
        source: SymbolSource,
        *,
        ttl: timedelta = timedelta(days=7),
        auto_refresh: bool = True,
        retry_after: timedelta = timedelta(hours=1),
        miss_refresh_age: timedelta = timedelta(hours=24),
        refresh_cooldown: timedelta = timedelta(minutes=30),
        reject_unlisted: Mapping[str, bool] = REJECT_UNLISTED_TICKERS,
        clock: Callable[[], datetime] = _utcnow,
    ):
        """Initialize the catalog.

        Args:
            repository: Persistent cache of the per-market lists.
            source: Fetches a market's list (see ``SymbolSource``).
            ttl: Age after which a list counts as stale.
            auto_refresh: Refresh missing/stale lists in the background.
            retry_after: Minimum gap between automatic attempts per market.
            miss_refresh_age: A miss in a list older than this queues a
                refresh of that market (rate-limited by ``retry_after``).
            refresh_cooldown: ``request_manual_refresh`` skips a market whose
                list is younger than this, unless forced.
            reject_unlisted: Per-market reject policy for a miss in a loaded
                list (see ``REJECT_UNLISTED_TICKERS``).
            clock: Returns the current UTC time (tests).
        """
        self._repository = repository
        self._source = source
        self._ttl = ttl
        self._auto_refresh = auto_refresh
        self._retry_after = retry_after
        self._miss_refresh_age = miss_refresh_age
        self._refresh_cooldown = refresh_cooldown
        self._reject_unlisted = dict(reject_unlisted)
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
        try:
            removed = self._repository.sweep_temp_files()
            if removed:
                logger.info("Removed %d stale symbol-cache temp file(s)", removed)
        except OSError as exc:
            logger.warning("Could not sweep symbol-cache temp files: %s", exc)
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
        """(Re)load every market's list from the cache; never raises."""
        for market in MARKETS:
            try:
                symbol_list = self._repository.load(market)
            except Exception:
                logger.exception("Could not load the %s symbol cache; ignoring it", market)
                continue
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

    def is_enforced(self, market: str | None) -> bool:
        """Whether a miss in ``market``'s loaded list rejects the request."""
        return bool(market and self._reject_unlisted.get(market, False))

    def _age(self, market: str) -> timedelta | None:
        index = self._indexes.get(market)
        if index is None:
            return None
        return self._clock() - _aware(index.list.fetched_at)

    def is_stale(self, market: str) -> bool:
        """True when the market has a list older than the TTL."""
        age = self._age(market)
        return age is not None and age > self._ttl

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

    def last_error(self, market: str | None) -> str | None:
        """The last failed (or refused) refresh of ``market``; cleared on success."""
        return self._last_error.get(market) if market else None

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
        """The market whose list must hold ``key`` (a ``list_key`` result).

        Only the hard-coded suffixes route (``.TW``/``.TWO`` -> tw, ``.T`` ->
        jp); a suffix seen in a loaded list never does.
        """
        return classify_market(key)

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

    def listed_local_variants(self, code: str) -> list[str]:
        """The listed ``.TW`` / ``.TWO`` / ``.T`` symbols a suffix-less code may mean.

        ``2330`` -> ``["2330.TW"]`` once the tw list holds it. Only the tw and
        jp lists are consulted, so this never guesses from a foreign listing.
        """
        listed = []
        for variant in local_code_variants(code.strip().upper()):
            index = self._indexes.get(classify_market(variant) or "")
            if index is not None and variant in index.by_symbol:
                listed.append(variant)
        return listed

    def decide(self, ticker: str, asset_type: str | None = None) -> TickerDecision:
        """Decide whether a request ticker is supported (validator and check endpoint)."""
        raw = ticker.strip()
        asset = (asset_type or "stock").strip().lower()
        key = list_key(raw, asset)

        if asset != "crypto" and self.get(key) is None:
            # A suffix-less TW/JP code: point at the listed symbol instead of
            # judging it against the us list its shape would otherwise fall to.
            # A bare 4-6 digit code is always rejected (the shape check).
            listed = self.listed_local_variants(key)
            if listed or is_bare_numeric(key):
                market = classify_market(listed[0]) if listed else None
                return TickerDecision(
                    ticker=raw,
                    normalized=key,
                    market=market,
                    supported=False,
                    enforced=True,
                    status=self.status(market) if market else STATUS_UNCOVERED,
                    suggestions=tuple(listed),
                    last_error=self.last_error(market),
                    reason=REASON_MISSING_SUFFIX,
                )

        market = self.market_for(key)
        if asset == "crypto" and market in (None, "us") and key.isalnum():
            # A crypto request for a base the core has no pair rule for
            # (e.g. "PEPE") means the coin's USD pair.
            key, market = f"{key}-USD", "crypto"
        if market is None:
            return TickerDecision(raw, key, None, None, False, STATUS_UNCOVERED)

        enforced = self.is_enforced(market)
        status = self.status(market)
        last_error = self.last_error(market)
        index = self._indexes.get(market)
        if index is None:
            return TickerDecision(raw, key, market, None, enforced, status, last_error=last_error)

        if market == "fx":
            legs = fx_legs(key) or []
            missing = [f"{leg}=X" for leg in legs if f"{leg}=X" not in index.by_symbol]
            if not missing:
                entry = index.by_symbol.get(key)
                return TickerDecision(
                    raw, key, market, True, enforced, status, entry=entry, last_error=last_error
                )
            suggestions = [s for m in missing for s in self.suggest(m, market, n=1)][:3]
        else:
            entry = index.by_symbol.get(key)
            if entry is not None:
                return TickerDecision(
                    raw, key, market, True, enforced, status, entry=entry, last_error=last_error
                )
            suggestions = self.suggest(key, market)

        self._refresh_after_miss(market)
        return TickerDecision(
            raw,
            key,
            market,
            False,
            enforced,
            status,
            suggestions=tuple(suggestions),
            last_error=last_error,
            reason=REASON_NOT_LISTED,
        )

    def lookup(self, symbol: str) -> SymbolLookup:
        """Exact lookup of one symbol across every loaded list, with suggestions."""
        key = list_key(symbol)
        entry = self.get(key)
        market = self.market_for(key)
        if entry is not None:
            market = entry.market
            return SymbolLookup(
                key, market, entry, self.status(market), (), self.last_error(market)
            )
        listed = self.listed_local_variants(key)
        if listed:
            market = classify_market(listed[0])
            suggestions: Sequence[str] = listed
        elif market is not None and self.is_loaded(market):
            suggestions = self.suggest(key, market)
            self._refresh_after_miss(market)
        else:
            suggestions = ()
        status = self.status(market) if market else STATUS_UNCOVERED
        return SymbolLookup(key, market, None, status, tuple(suggestions), self.last_error(market))

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

    def request_manual_refresh(
        self, markets: Iterable[str], force: bool = False
    ) -> tuple[list[str], list[str]]:
        """Queue a user-requested refresh; returns ``(queued, skipped)``.

        A market whose list was fetched within ``refresh_cooldown`` is skipped
        unless ``force`` is set, so repeated calls cannot hammer Yahoo.
        """
        requested = [m for m in dict.fromkeys(markets) if m in MARKETS]
        skipped = [] if force else [m for m in requested if self._fetched_recently(m)]
        queued = self.request_refresh([m for m in requested if m not in skipped])
        return queued, skipped

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

        On failure the previous list stays in memory and on disk. A fetch that
        looks truncated next to the previous list (fewer than 80% of its
        entries, or an exchange/type group that had rows is now empty) is
        refused the same way: logged, recorded as ``last_error``, not saved.
        """
        self._last_attempt[market] = self._clock()
        kept = self.get_list(market)
        try:
            entries = list(self._source.fetch(market))
        except Exception as exc:
            self._last_error[market] = f"{type(exc).__name__}: {exc}"
            self._log_kept(market, self._last_error[market], kept)
            return False
        problem = self._shrink_problem(kept, entries)
        if problem:
            self._last_error[market] = f"refused a suspicious refresh: {problem}"
            self._log_kept(market, self._last_error[market], kept)
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

    @staticmethod
    def _shrink_problem(old: SymbolList | None, new: list[SymbolEntry]) -> str | None:
        """Why ``new`` looks like a truncated version of ``old``, or None."""
        if old is None or not old.entries:
            return None
        if len(new) < MIN_REFRESH_KEEP_RATIO * len(old.entries):
            return (
                f"{len(new)} entries, fewer than {MIN_REFRESH_KEEP_RATIO:.0%} of the "
                f"previous {len(old.entries)}"
            )
        old_buckets = Counter((e.exchange, e.type) for e in old.entries)
        new_buckets = {(e.exchange, e.type) for e in new}
        gone = sorted(b for b in old_buckets if b not in new_buckets)
        if gone:
            return "no rows any more for " + ", ".join(
                f"{exchange}/{type_} (had {old_buckets[(exchange, type_)]})"
                for exchange, type_ in gone
            )
        return None

    @staticmethod
    def _log_kept(market: str, reason: str, kept: SymbolList | None) -> None:
        logger.warning(
            "Refreshing the %s symbol list failed (%s); %s",
            market,
            reason,
            f"keeping the {len(kept.entries)}-entry list from {kept.fetched_at.isoformat()}"
            if kept
            else "no cached list to fall back on",
        )

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

    def _fetched_recently(self, market: str) -> bool:
        age = self._age(market)
        return age is not None and age < self._refresh_cooldown

    def _refresh_after_miss(self, market: str) -> None:
        """A miss may be a new listing: refresh a list older than a day."""
        self._touch(market, max_age=self._miss_refresh_age)

    def _touch(self, market: str, max_age: timedelta | None = None) -> None:
        """Queue a missing list, or one older than ``max_age`` (default: the TTL).

        Only after startup with auto-refresh on, and at most once per
        ``retry_after`` per market.
        """
        if not (self._started and self._auto_refresh) or self._stop.is_set():
            return
        age = self._age(market)
        if age is not None and age <= (max_age if max_age is not None else self._ttl):
            return
        if self.is_refreshing(market):
            return
        last = self._last_attempt.get(market)
        if last is not None and self._clock() - last < self._retry_after:
            return
        self._last_attempt[market] = self._clock()
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
