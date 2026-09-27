"""Runtime-adjustable settings of the supported-symbols refresher.

Each setting has a *base* value and may be overridden at runtime through
``PUT /symbols/settings``. Precedence, highest first:

1. ``api``: a ``PUT /symbols/settings`` value, persisted to
   ``<cache dir>/settings.json`` (written atomically) so it survives a restart;
2. ``config``: a ``create_app(overrides=...)`` value;
3. ``env``: a ``TRADINGAGENTS_SYMBOLS_*`` env var (an empty value counts as
   unset);
4. ``default``: the built-in default.

Setting a field to null in the PUT removes its API override, so the next
level applies again. A corrupt settings file is ignored with a warning (base
values apply); a single invalid field in it is dropped. An invalid base value
stops the app from starting (``create_symbol_settings`` names the env var or
config key).

The time zone is only resolved when a refresh window is set, so a system
without tz data runs fine with the window off; with a window set, an unknown
zone is an error.

Readers (the catalog, the Yahoo source, the activity monitor) read the
current values on every use, so a change applies live: the next tick, the
next page.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from tradingagents.api.domain.refresh_window import RefreshWindow

logger = logging.getLogger(__name__)

SETTINGS_FILE_NAME = "settings.json"
SETTINGS_FORMAT_VERSION = 1

SOURCE_DEFAULT = "default"
SOURCE_ENV = "env"
SOURCE_CONFIG = "config"  # a create_app(overrides=...) value
SOURCE_API = "api"


@dataclass(frozen=True)
class SettingSpec:
    """One adjustable setting: its config key and how to validate it."""

    name: str
    config_key: str
    kind: type
    minimum: float | None = None
    maximum: float | None = None


SETTING_SPECS: dict[str, SettingSpec] = {
    spec.name: spec
    for spec in (
        SettingSpec("ttl_days", "symbols_cache_ttl_days", float, 0.5, 90.0),
        SettingSpec("auto_refresh", "symbols_auto_refresh", bool),
        SettingSpec("idle_grace_minutes", "symbols_idle_grace_minutes", float, 0.0, 240.0),
        SettingSpec("refresh_window", "symbols_refresh_window", str),
        SettingSpec("refresh_timezone", "symbols_refresh_timezone", str),
        SettingSpec("page_delay_seconds", "symbols_page_delay_seconds", float, 0.5, 10.0),
    )
}


def validate_setting(name: str, value: Any) -> Any:
    """Return ``value`` as the setting's type, or raise ValueError."""
    spec = SETTING_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown setting {name!r}")
    if spec.kind is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        return value
    if spec.kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        if spec.minimum is not None and number < spec.minimum:
            raise ValueError(f"{name} must be at least {spec.minimum:g}")
        if spec.maximum is not None and number > spec.maximum:
            raise ValueError(f"{name} must be at most {spec.maximum:g}")
        return number
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if name == "refresh_timezone" and not text:
        # The zone itself is resolved with the window (``RefreshWindow.parse``).
        raise ValueError("refresh_timezone must be an IANA time zone, e.g. 'Asia/Taipei'")
    if name == "refresh_window":
        RefreshWindow.parse_times(text)  # format only; the zone is checked with it
    return text


class SymbolSettings:
    """Effective refresher settings: base values plus persisted API overrides."""

    def __init__(
        self,
        base: Mapping[str, Any],
        sources: Mapping[str, str] | None = None,
        store_dir: str | Path | None = None,
    ):
        """Initialize the settings.

        Args:
            base: Setting name -> base value (default / env / config).
            sources: Setting name -> where the base value came from.
            store_dir: Directory of the persisted ``settings.json``; None keeps
                API overrides in memory only.

        Raises:
            ValueError: A base value is invalid (e.g. a malformed env var);
                ``create_symbol_settings`` validates first to name its origin.
        """
        self._lock = threading.Lock()
        self._base = {name: validate_setting(name, base[name]) for name in SETTING_SPECS}
        self._base_sources = {
            name: (sources or {}).get(name, SOURCE_DEFAULT) for name in self._base
        }
        RefreshWindow.parse(self._base["refresh_window"], self._base["refresh_timezone"])
        self._path = Path(store_dir) / SETTINGS_FILE_NAME if store_dir is not None else None
        self._overrides: dict[str, Any] = self._load()
        self._listeners: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        with self._lock:
            return self._overrides.get(name, self._base[name])

    def values(self) -> dict[str, Any]:
        with self._lock:
            return {name: self._overrides.get(name, self._base[name]) for name in self._base}

    def sources(self) -> dict[str, str]:
        with self._lock:
            return {
                name: SOURCE_API if name in self._overrides else self._base_sources[name]
                for name in self._base
            }

    @property
    def ttl(self) -> timedelta:
        return timedelta(days=self.get("ttl_days"))

    @property
    def auto_refresh(self) -> bool:
        return self.get("auto_refresh")

    @property
    def idle_grace(self) -> timedelta:
        return timedelta(minutes=self.get("idle_grace_minutes"))

    @property
    def page_delay_seconds(self) -> float:
        return self.get("page_delay_seconds")

    @property
    def refresh_window(self) -> RefreshWindow | None:
        with self._lock:
            spec = self._overrides.get("refresh_window", self._base["refresh_window"])
            zone = self._overrides.get("refresh_timezone", self._base["refresh_timezone"])
        return RefreshWindow.parse(spec, zone)  # the zone is resolved only with a window

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` after every successful update."""
        self._listeners.append(callback)

    def update(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a partial update; a None value resets that field to its base.

        All fields are validated (and the window against the resulting time
        zone) before anything changes. The overrides are persisted, then the
        listeners run. Returns the new effective values.

        Raises:
            ValueError: As ``(field, message)`` in ``args``; nothing changed.
            OSError: The settings file could not be written; nothing changed.
        """
        with self._lock:
            overrides = dict(self._overrides)
            for name, value in changes.items():
                if name not in SETTING_SPECS:
                    raise ValueError(name, f"unknown setting {name!r}")
                if value is None:
                    overrides.pop(name, None)
                    continue
                try:
                    overrides[name] = validate_setting(name, value)
                except ValueError as exc:
                    raise ValueError(name, str(exc)) from exc
            window = overrides.get("refresh_window", self._base["refresh_window"])
            zone = overrides.get("refresh_timezone", self._base["refresh_timezone"])
            try:
                RefreshWindow.parse(window, zone)
            except ValueError as exc:
                raise ValueError("refresh_window", str(exc)) from exc
            self._save(overrides)
            self._overrides = overrides
        logger.info("Symbol refresher settings updated: %s", dict(changes))
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:  # pragma: no cover - a listener must not undo the update
                logger.exception("Symbol settings listener failed")
        return self.values()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != SETTINGS_FORMAT_VERSION:
                raise ValueError("not a version-1 settings object")
            values = payload.get("values")
            if not isinstance(values, dict):
                raise ValueError("no 'values' object")
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring the symbol settings file %s: %s", self._path, exc)
            return {}
        loaded: dict[str, Any] = {}
        for name, value in values.items():
            try:
                loaded[name] = validate_setting(name, value)
            except ValueError as exc:
                logger.warning(
                    "Ignoring %r in the symbol settings file %s: %s", name, self._path, exc
                )
        window = loaded.get("refresh_window", self._base["refresh_window"])
        zone = loaded.get("refresh_timezone", self._base["refresh_timezone"])
        try:
            RefreshWindow.parse(window, zone)
        except ValueError as exc:
            logger.warning("Ignoring the window in the symbol settings file: %s", exc)
            loaded.pop("refresh_window", None)
            loaded.pop("refresh_timezone", None)
        if loaded:
            logger.info("Loaded symbol refresher settings from %s: %s", self._path, loaded)
        return loaded

    def _save(self, overrides: Mapping[str, Any]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": SETTINGS_FORMAT_VERSION, "values": dict(overrides)}
        fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, prefix=".settings.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise


def static_settings(
    ttl: timedelta = timedelta(days=7),
    auto_refresh: bool = True,
    refresh_window: RefreshWindow | None = None,
    idle_grace: timedelta = timedelta(minutes=10),
    page_delay_seconds: float = 2.0,
) -> SymbolSettings:
    """In-memory settings from plain values (for a catalog built without the app).

    ``ttl`` is not range-checked here, so tests may use any positive TTL.
    """
    base = {
        "ttl_days": 7.0,
        "auto_refresh": auto_refresh,
        "idle_grace_minutes": idle_grace.total_seconds() / 60,
        "refresh_window": refresh_window.spec if refresh_window else "",
        "refresh_timezone": refresh_window.tz.key if refresh_window else "Asia/Taipei",
        "page_delay_seconds": page_delay_seconds,
    }
    settings = SymbolSettings(base)
    settings._base["ttl_days"] = ttl.total_seconds() / 86400  # bypass the API range
    return settings
