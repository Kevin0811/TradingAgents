"""The daily quiet window in which automatic symbol-list refreshes may run.

Automatic refreshes of a *stale* list (and the refresh a miss in a day-old
list asks for) wait for this window, so the API talks to Yahoo at a quiet
hour instead of whenever a request happens to read the list. A *missing* list
and a manual ``POST /symbols/refresh`` are not bound by it.

The window is ``"HH:MM-HH:MM"`` in an IANA time zone, e.g. ``"02:00-06:00"``
in ``Asia/Taipei``. The start is inclusive and the end exclusive; a window
whose end is before its start wraps midnight (``"22:00-04:00"``); ``24:00``
is accepted as an end. An empty spec disables the window (refresh any time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_SPEC = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def _clock_time(hour: int, minute: int, spec: str, *, is_end: bool) -> time:
    if is_end and (hour, minute) == (24, 0):
        return time(0, 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid time {hour:02d}:{minute:02d} in refresh window {spec!r}")
    return time(hour, minute)


@dataclass(frozen=True)
class RefreshWindow:
    """A daily ``[start, end)`` interval in a time zone."""

    start: time
    end: time
    tz: ZoneInfo
    spec: str

    @classmethod
    def parse(cls, spec: str | None, tz_name: str | None) -> RefreshWindow | None:
        """Parse ``spec`` in ``tz_name``; None (window disabled) for an empty spec.

        Raises:
            ValueError: The spec is not ``HH:MM-HH:MM``, a time is out of range,
                start equals end, or the time zone is unknown or empty.
        """
        times = cls.parse_times(spec)
        if times is None:
            return None
        start, end = times
        spec = str(spec).strip()
        name = (tz_name or "").strip()
        if not name:
            raise ValueError("a refresh window needs a time zone (e.g. 'Asia/Taipei')")
        try:
            tz = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown refresh-window time zone {name!r}") from exc
        return cls(start=start, end=end, tz=tz, spec=spec)

    @staticmethod
    def parse_times(spec: str | None) -> tuple[time, time] | None:
        """The ``(start, end)`` of ``spec`` without resolving any time zone.

        None for an empty spec. Raises ValueError like ``parse``.
        """
        if spec is None or not str(spec).strip():
            return None
        spec = str(spec).strip()
        match = _SPEC.fullmatch(spec)
        if not match:
            raise ValueError(
                f"refresh window {spec!r} is not in HH:MM-HH:MM form (e.g. '02:00-06:00'); "
                "use an empty value to refresh at any time"
            )
        h1, m1, h2, m2 = (int(g) for g in match.groups())
        start = _clock_time(h1, m1, spec, is_end=False)
        end = _clock_time(h2, m2, spec, is_end=True)
        if start == end:
            raise ValueError(f"refresh window {spec!r} is empty (start equals end)")
        return start, end

    def describe(self) -> str:
        """``"02:00-06:00 Asia/Taipei"``."""
        return f"{self.spec} {self.tz.key}"

    def contains(self, now: datetime) -> bool:
        """True when ``now`` (aware; naive counts as UTC) falls inside the window."""
        local = _aware(now).astimezone(self.tz).time().replace(tzinfo=None)
        if self.start < self.end:
            return self.start <= local < self.end
        return local >= self.start or local < self.end  # wraps midnight

    def next_open(self, now: datetime) -> datetime:
        """``now`` if inside the window, else the window's next start (UTC)."""
        now = _aware(now)
        if self.contains(now):
            return now.astimezone(timezone.utc)
        local = now.astimezone(self.tz)
        candidate = local.replace(
            hour=self.start.hour, minute=self.start.minute, second=0, microsecond=0
        )
        if candidate <= local:
            candidate = (local + timedelta(days=1)).replace(
                hour=self.start.hour, minute=self.start.minute, second=0, microsecond=0
            )
        return candidate.astimezone(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
