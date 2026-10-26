"""LLM API keys read from Docker/Compose secret files when a client is built.

`make docker-run-api` passes the keys taken from the OS keyring as Compose
secrets, i.e. files ``/run/secrets/<VAR>``. For the providers whose core client
takes an ``api_key`` argument (``ON_DEMAND_PROVIDERS``: anthropic, google), the
key is read from its file each time a model tier's client is built and passed
as ``api_key=``; it is never put in ``os.environ``, so it is not in the server's
initial environment (``/proc/1/environ``, ``docker exec ... env``) and child
processes do not inherit it. ``scripts/container_secrets.py`` (the image's
entrypoint) still loads every other secret into the environment, because the
core reads those keys from there (the OpenAI-compatible clients refuse to start
without their env var even when given ``api_key=``; data vendors call
``os.getenv``).

Precedence: a usable secret file wins over the environment (a key in the
keyring must win over ``.env`` inside the container). With no usable file the
client is built exactly as before and finds its key in the environment itself.

The analysis graphs are built by the core's ``TradingAgentsGraph``, which only
forwards ``callbacks`` to ``create_tier_client``. ``install()`` (called by
``create_app``) therefore rebinds the name ``create_tier_client`` in
``tradingagents.graph.trading_graph`` to a wrapper that adds the key; no core
file changes. ``tests/test_api_secret_keys.py`` builds a real graph and fails
if the core stops calling that module-level name.

A file is read only if it is a regular file: a symlink is never followed (it is
skipped, and the open uses O_NOFOLLOW in case one is swapped in), a FIFO or
device fails the fstat check, and the content must be UTF-8, at most 64 KiB and
free of NUL bytes. Trailing CR/LF are stripped; an empty or whitespace-only
file counts as absent. A missing directory or file is normal (no keyring, or a
run outside Docker) and is silent; any other problem logs one WARNING naming
the file, never its content. Nothing is cached: a file replaced in place is
picked up by the next client (Compose still needs ``--force-recreate`` to
change it, see the Makefile).

Limits: the key still lives in the client object (``ChatAnthropic``) for the
run; the file stays mode 0444 in the container's writable layer; anyone who
can ``docker exec`` into the container can read it.

Environment:
  TRADINGAGENTS_SECRETS_DIR  secrets directory (default /run/secrets; for tests)
"""

from __future__ import annotations

import logging
import os
import re
import stat
import threading
from typing import Any

from tradingagents.llm_clients import factory as _factory
from tradingagents.llm_clients.api_key_env import get_api_key_env

logger = logging.getLogger(__name__)

DEFAULT_DIR = "/run/secrets"
DIR_VAR = "TRADINGAGENTS_SECRETS_DIR"
MAX_BYTES = 64 * 1024
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# O_NOFOLLOW: a symlink swapped in after the lstat fails to open (ELOOP).
# O_NONBLOCK: a FIFO swapped in does not block the open; fstat rejects it.
_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_CLOEXEC", 0)
)

# Providers whose core client honours an ``api_key`` argument. The
# OpenAI-compatible ones do not (they raise when their env var is unset), so
# their keys stay in the environment. scripts/container_secrets.py keeps a copy
# of ON_DEMAND_KEY_NAMES (it cannot import this package); a test checks they match.
ON_DEMAND_PROVIDERS: frozenset[str] = frozenset({"anthropic", "google"})
ON_DEMAND_KEY_NAMES: frozenset[str] = frozenset(
    get_api_key_env(provider) for provider in ON_DEMAND_PROVIDERS
)

# Marks a wrapper made by this module, so install() is idempotent.
_WRAPPED_ATTR = "__tradingagents_secret_keys_original__"
_install_lock = threading.Lock()


def secrets_dir() -> str:
    """The secrets directory: $TRADINGAGENTS_SECRETS_DIR, else /run/secrets."""
    return os.environ.get(DIR_VAR) or DEFAULT_DIR


def _read_fd(fd: int, limit: int) -> bytes:
    """Up to ``limit`` bytes from ``fd`` (fewer only at end of file)."""
    chunks: list[bytes] = []
    size = 0
    while size < limit:
        chunk = os.read(fd, limit - size)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def read_secret(name: str) -> str | None:
    """The value of secret file ``<secrets dir>/<name>``, or None.

    None when the file is missing, empty or unusable (see the module docstring);
    an unusable file logs a WARNING naming it, never its content. Read afresh
    on every call and never copied into ``os.environ``.

    Raises:
        ValueError: ``name`` is not a variable name (``^[A-Z][A-Z0-9_]*$``).
    """
    if not _NAME_RE.match(name):
        raise ValueError(f"not a secret variable name: {name!r}")
    path = os.path.join(secrets_dir(), name)
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            logger.warning("Ignoring secret %s: it is a symlink", name)
            return None
        fd = os.open(path, _OPEN_FLAGS)
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP for a symlink swapped in, EACCES, ENOTDIR, ...
        logger.warning("Ignoring secret %s: cannot read it (%s)", name, type(exc).__name__)
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            logger.warning("Ignoring secret %s: not a regular file", name)
            return None
        data = _read_fd(fd, MAX_BYTES + 1)
    except OSError as exc:
        logger.warning("Ignoring secret %s: cannot read it (%s)", name, type(exc).__name__)
        return None
    finally:
        os.close(fd)
    if len(data) > MAX_BYTES:
        logger.warning("Ignoring secret %s: larger than %d bytes", name, MAX_BYTES)
        return None
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Ignoring secret %s: not valid UTF-8", name)
        return None
    if "\0" in value:
        logger.warning("Ignoring secret %s: contains a NUL byte", name)
        return None
    value = value.rstrip("\r\n")
    return value if value.strip() else None


def api_key_for(provider: str) -> str | None:
    """The key from ``provider``'s secret file, or None to leave the lookup to the client.

    None for a provider not in ON_DEMAND_PROVIDERS, and when its file is
    missing or unusable: the client then reads its env var as before.
    """
    provider = (provider or "").lower()
    if provider not in ON_DEMAND_PROVIDERS:
        return None
    return read_secret(get_api_key_env(provider))


def _with_key(config: dict, tier: str, extra: dict[str, Any]) -> dict[str, Any]:
    """``extra`` plus ``api_key`` from the tier provider's secret file, if any.

    An ``api_key`` already in ``extra`` is kept. A config whose provider cannot
    be read is left to the core, which raises its own error for it.
    """
    if "api_key" in extra:
        return extra
    try:
        provider = _factory.tier_provider(config, tier)
    except Exception:  # the core reports a bad config itself
        return extra
    key = api_key_for(provider)
    return {**extra, "api_key": key} if key is not None else extra


def create_tier_client(config: dict, tier: str, **extra: Any):
    """The core's ``create_tier_client``, with the tier's key from its secret file."""
    return _factory.create_tier_client(config, tier, **_with_key(config, tier, extra))


def _wrap(original):
    """A ``create_tier_client`` that calls ``original`` with the secret key added."""

    def create_tier_client(config, tier, **extra):
        return original(config, tier, **_with_key(config, tier, extra))

    create_tier_client.__doc__ = original.__doc__
    setattr(create_tier_client, _WRAPPED_ATTR, original)
    return create_tier_client


def install() -> None:
    """Make ``TradingAgentsGraph`` build its tier clients with the secret keys.

    Rebinds ``tradingagents.graph.trading_graph.create_tier_client``; a second
    call is a no-op. Process-wide: every thread's graph gets it.

    Raises:
        RuntimeError: The core no longer has that name, so the keys could not
            reach its clients (fail at startup rather than at the first run).
    """
    from tradingagents.graph import trading_graph

    with _install_lock:
        current = getattr(trading_graph, "create_tier_client", None)
        if current is None:
            raise RuntimeError(
                "tradingagents.graph.trading_graph no longer has create_tier_client: "
                "API keys from Docker secrets cannot reach the analysis graph's clients "
                "(see tradingagents/api/infrastructure/secret_keys.py)"
            )
        if hasattr(current, _WRAPPED_ATTR):
            return
        trading_graph.create_tier_client = _wrap(current)
