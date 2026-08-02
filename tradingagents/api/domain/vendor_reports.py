"""Parse vendor report text into structured records.

The dataflow layer formats every result as text, because its primary consumer
is an LLM reading a prompt. The HTTP API has a different consumer, so this
module converts the two tabular report formats back into records.

It deliberately covers only the formats that are genuinely tabular — OHLCV and
indicator series. News, fundamentals and macro reports are prose written for a
model to read; reverse-engineering fields out of them would produce a brittle
contract that breaks whenever the wording changes.

Both vendors are handled by one parser per format: yfinance and Alpha Vantage
already emit the same shapes, differing only in column naming and casing.
"""

from __future__ import annotations

import csv
import re
from io import StringIO
from typing import Any

# Vendor column name -> canonical field. Applied after lowercasing and
# replacing spaces with underscores, so "Adj Close" arrives as "adj_close"
# and only genuine cross-vendor differences need a row here.
_COLUMN_ALIASES = {
    "timestamp": "date",  # Alpha Vantage names the date column this way
    "adjusted_close": "adj_close",
    "stock_splits": "split_coefficient",
}

# "## rsi values from 2026-07-28 to 2026-07-31:" — Alpha Vantage upper-cases
# the indicator name, yfinance does not.
_SECTION_RE = re.compile(
    r"^##\s+(?P<name>\S+)\s+values\s+from\s+(?P<start>\S+)\s+to\s+(?P<end>[^:]+):\s*$",
    re.MULTILINE,
)

_POINT_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2}):\s*(?P<value>.*)$")


def _canonical_column(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return _COLUMN_ALIASES.get(key, key)


def _coerce(value: str) -> Any:
    """Return a number for numeric cells, None for blanks, else the string."""
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_ohlcv_csv(report: str) -> list[dict[str, Any]]:
    """Return OHLCV rows from a vendor price report.

    Handles the yfinance flavour, which prefixes ``#`` comment lines and a
    blank line before the CSV header, and the Alpha Vantage flavour, which is
    bare CSV. Unknown vendor columns are kept under their canonical names
    rather than dropped.
    """
    if not report:
        return []

    body = [
        line
        for line in report.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(body) < 2:
        return []

    reader = csv.reader(StringIO("\n".join(body)))
    header = [_canonical_column(col) for col in next(reader)]

    rows: list[dict[str, Any]] = []
    for raw in reader:
        if not any(cell.strip() for cell in raw):
            continue
        row = {key: _coerce(cell) for key, cell in zip(header, raw) if key}
        # A row without a date is a trailing artefact, not an observation.
        if row.get("date") is not None:
            row["date"] = str(row["date"])
            rows.append(row)
    return rows


def parse_indicator_report(report: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Return ``(series, errors)`` from an indicator report.

    Each series is ``{"name", "description", "points"}`` where a point is
    ``{"date", "value", "note"}``; ``value`` is None when the vendor reported
    a non-numeric placeholder such as "N/A: Not a trading day", in which case
    the placeholder is preserved in ``note``.

    The tool concatenates one section per requested indicator and appends any
    per-indicator failure as bare text — those are returned in ``errors``
    instead of being silently dropped, so an unsupported name is still visible
    in the response.
    """
    if not report or not report.strip():
        return [], []

    matches = list(_SECTION_RE.finditer(report))
    if not matches:
        return [], [report.strip()]

    errors: list[str] = []
    preamble = report[: matches[0].start()].strip()
    if preamble:
        errors.append(preamble)

    series: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        block = report[match.end() : end]

        points: list[dict[str, Any]] = []
        description_lines: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            point = _POINT_RE.match(stripped)
            if point:
                raw_value = point.group("value").strip()
                value = _coerce(raw_value)
                numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
                points.append(
                    {
                        "date": point.group("date"),
                        "value": value if numeric else None,
                        "note": None if numeric else (raw_value or None),
                    }
                )
            else:
                # Everything after the values is the indicator's usage blurb.
                description_lines.append(stripped)

        points.sort(key=lambda p: p["date"])
        series.append(
            {
                "name": match.group("name").strip().lower(),
                "description": " ".join(description_lines) or None,
                "points": points,
            }
        )

    return series, errors
