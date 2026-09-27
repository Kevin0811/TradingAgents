"""File-based implementation of SymbolCacheRepository.

One JSON file per market (``<cache_dir>/<market>.json``)::

    {
      "version": 1,
      "market": "tw",
      "source": "yfinance 1.7.0 screener",
      "fetched_at": "2026-09-26T08:00:00+00:00",
      "count": 2642,
      "entries": [{"symbol": "2330.TW", "name": "...", "exchange": "TAI",
                   "type": "equity", "market": "tw"}, ...]
    }

Writes go to a temporary file in the same directory and are then renamed over
the old file, so a crash mid-write never leaves a truncated cache behind. A
temp file orphaned by a crash (``.<market>.*.tmp``) is swept at startup.

File mode: the temp file comes from ``tempfile.mkstemp`` and is created
``0600``; the rename keeps that mode, so the cache files are readable by the
API's own user only. That is deliberate (nothing else needs them) and left as
is.

Loading is defensive: a file that is not valid JSON, not an object, of another
format version, has an empty ``entries`` list, or holds any malformed entry (a
missing symbol, an unknown ``type``, ...) is ignored as a whole and reported as
missing, so a bad file never stops the API from starting.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.repositories import SymbolCacheRepository
from tradingagents.api.domain.symbols import MARKETS, SYMBOL_TYPES

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1

# Leftover temp files younger than this may belong to a write in progress
# (another worker process sharing the directory) and are left alone.
TEMP_FILE_MIN_AGE_SECONDS = 600.0

_TEMP_FILE = re.compile(r"^\.(" + "|".join(map(re.escape, MARKETS)) + r")\..+\.tmp$")


class _BadCache(ValueError):
    """The cache file parsed but does not hold a usable list."""


def _optional_str(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _BadCache(f"entry field {field!r} is not a string")
    return value


def _parse_entry(raw: Any, market: str) -> SymbolEntry:
    if not isinstance(raw, dict):
        raise _BadCache("an entry is not an object")
    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise _BadCache("an entry has no symbol")
    type_ = raw.get("type")
    if type_ not in SYMBOL_TYPES:
        raise _BadCache(f"entry {symbol!r} has an unknown type {type_!r}")
    return SymbolEntry(
        symbol=symbol,
        name=_optional_str(raw.get("name"), "name"),
        exchange=_optional_str(raw.get("exchange"), "exchange"),
        type=type_,
        market=market,
    )


class FileSymbolCacheRepository(SymbolCacheRepository):
    """Stores each market's supported-symbols list as a JSON file."""

    def __init__(self, cache_dir: str | Path):
        """Initialize the repository.

        Args:
            cache_dir: Directory holding one ``<market>.json`` file per market.
        """
        self._cache_dir = Path(cache_dir)

    def path_for(self, market: str) -> Path:
        """Return the cache file path for ``market``."""
        return self._cache_dir / f"{market}.json"

    def load(self, market: str) -> SymbolList | None:
        path = self.path_for(market)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise _BadCache("the payload is not a JSON object")
            if payload.get("version") != CACHE_FORMAT_VERSION:
                logger.warning(
                    "Ignoring symbol cache %s: unsupported version %r",
                    path,
                    payload.get("version"),
                )
                return None
            raw_entries = payload.get("entries")
            if not isinstance(raw_entries, list) or not raw_entries:
                raise _BadCache("no entries")
            entries = tuple(_parse_entry(e, market) for e in raw_entries)
            fetched_at = payload.get("fetched_at")
            if not isinstance(fetched_at, str):
                raise _BadCache("no fetched_at timestamp")
            source = payload.get("source")
            return SymbolList(
                market=market,
                entries=entries,
                fetched_at=datetime.fromisoformat(fetched_at),
                source=source if isinstance(source, str) else "",
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # _BadCache and json.JSONDecodeError are ValueErrors.
            logger.warning("Ignoring unreadable symbol cache %s: %s", path, exc)
            return None

    def save(self, symbol_list: SymbolList) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_FORMAT_VERSION,
            "market": symbol_list.market,
            "source": symbol_list.source,
            "fetched_at": symbol_list.fetched_at.isoformat(),
            "count": len(symbol_list.entries),
            "entries": [e.to_dict() for e in symbol_list.entries],
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=self._cache_dir, prefix=f".{symbol_list.market}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.path_for(symbol_list.market))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def sweep_temp_files(self) -> int:
        """Delete ``.<market>.*.tmp`` files older than ``TEMP_FILE_MIN_AGE_SECONDS``."""
        if not self._cache_dir.is_dir():
            return 0
        cutoff = time.time() - TEMP_FILE_MIN_AGE_SECONDS
        removed = 0
        for path in self._cache_dir.iterdir():
            if not _TEMP_FILE.match(path.name):
                continue
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("Could not remove stale symbol-cache temp file %s: %s", path, exc)
        return removed
