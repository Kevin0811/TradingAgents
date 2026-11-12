"""Discussion transcripts on disk, so a finished run's discussion outlives the task TTL.

A transcript is written once, when its task ends, next to the run's other
results: ``<results_dir>/<TICKER>/TradingAgentsStrategy_logs/discussion_<task_id>.json``
(the core keeps ``full_states_log_<date>.json`` in the same folder). The write
is atomic: a temporary file in the same folder, then ``os.replace``.

``<TICKER>`` is the ticker the analysis request named. Transcripts are
durable, like the core's ``full_states_log``: deleting the task (``DELETE``)
or its expiry drops only the in-memory log, never the file.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from tradingagents.dataflows.symbols import safe_ticker_component

logger = logging.getLogger(__name__)

LOGS_FOLDER = "TradingAgentsStrategy_logs"


def _canonical_task_id(task_id: str) -> str | None:
    """The task id if it is a canonical UUID (all the API issues), else None."""
    try:
        canonical = str(UUID(task_id))
    except (ValueError, AttributeError, TypeError):
        return None
    return canonical if canonical == task_id else None


class TranscriptFiles:
    """Reads and writes transcript files under one results directory."""

    def __init__(self, results_dir: str | Path):
        self._results_dir = Path(results_dir)

    def path_for(self, ticker: str, task_id: str) -> Path:
        return (
            self._results_dir
            / safe_ticker_component(ticker)
            / LOGS_FOLDER
            / f"discussion_{task_id}.json"
        )

    def write(self, ticker: str, transcript: dict[str, Any]) -> Path | None:
        """Write ``transcript`` atomically; return its path, or None if it failed.

        A failure is logged, never raised: the task's own outcome stands.
        """
        task_id = _canonical_task_id(str(transcript.get("task_id", "")))
        if task_id is None:
            return None
        tmp_name: str | None = None
        try:
            # Inside the try: an unsafe ticker raises ValueError here.
            path = self.path_for(ticker, task_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, prefix=".discussion-",
                suffix=".tmp", delete=False,
            ) as tmp:
                tmp_name = tmp.name
                json.dump(transcript, tmp, ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
            return path
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not write the discussion transcript of task %s: %s",
                           task_id, exc)
            if tmp_name is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
            return None

    def read(self, task_id: str) -> dict[str, Any] | None:
        """The transcript of ``task_id`` from disk, or None."""
        task_id = _canonical_task_id(task_id)
        if task_id is None or not self._results_dir.is_dir():
            return None
        for path in self._results_dir.glob(f"*/{LOGS_FOLDER}/discussion_{task_id}.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as exc:
                logger.warning("Could not read the discussion transcript %s: %s", path, exc)
                continue
            if isinstance(data, dict) and data.get("task_id") == task_id:
                return data
        return None
