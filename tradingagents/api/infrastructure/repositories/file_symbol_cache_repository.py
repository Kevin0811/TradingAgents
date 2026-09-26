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
the old file, so a crash mid-write never leaves a truncated cache behind.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from tradingagents.api.domain.entities import SymbolEntry, SymbolList
from tradingagents.api.domain.repositories import SymbolCacheRepository

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


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
            if payload.get("version") != CACHE_FORMAT_VERSION:
                logger.warning(
                    "Ignoring symbol cache %s: unsupported version %r",
                    path,
                    payload.get("version"),
                )
                return None
            entries = tuple(
                SymbolEntry(
                    symbol=e["symbol"],
                    name=e.get("name") or "",
                    exchange=e.get("exchange") or "",
                    type=e["type"],
                    market=market,
                )
                for e in payload["entries"]
            )
            return SymbolList(
                market=market,
                entries=entries,
                fetched_at=datetime.fromisoformat(payload["fetched_at"]),
                source=payload.get("source") or "",
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
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
