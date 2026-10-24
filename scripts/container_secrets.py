#!/usr/bin/env python3
"""Container entrypoint: load Compose secrets into the environment, then exec.

  python scripts/container_secrets.py <command> [args...]

`make docker-run-api` passes the API keys taken from the OS keyring as Compose
secrets (`keychain_env.py compose --secrets`): each key is a file
/run/secrets/<VAR> rather than an environment variable, so its value is not in
the container config that `docker inspect` shows. The API reads its keys from
the environment, so this entrypoint (Dockerfile.api's ENTRYPOINT) copies each
file into os.environ and then replaces itself with the command (the image's
CMD, or whatever `docker run <image> <command>` gives).

Which files are loaded:
  - only regular files: a symlink is never followed (it is skipped even when it
    points at a regular file, so a planted link cannot pull in a file from
    elsewhere), and the file is opened with O_NOFOLLOW and checked with fstat
    (S_ISREG) before reading, so a FIFO or device swapped in is not read;
  - only names that are variable names (^[A-Z][A-Z0-9_]*$) AND look like a
    secret, i.e. contain KEY, TOKEN, SECRET, PASSWORD or PASS (every key name
    keychain_env.py looks up does). A secret cannot set arbitrary variables;
  - never a name that changes how the process or Python behaves, even if it
    looks like a secret: PATH, HOME, SHELL, USER, PWD, IFS, LD_*, DYLD_*,
    PYTHON*, SSL_CERT_*, REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE and
    TRADINGAGENTS_SECRETS_DIR.
The content is read as UTF-8 (at most 64 KiB) and trailing CR/LF are stripped;
an empty or whitespace-only result is skipped, so the service's env_file still
supplies that variable.

A secret OVERRIDES a value already in the environment. The service's env_file
(.env) has already set its variables when this runs, and a key in the keyring
must win over .env inside the container, as it does in the env-var mode where
compose's `environment:` overrides `env_file:`. Only keys exported from the
keyring become secrets, so a key that is only in .env is left alone (and a copy
of it left in .env is still in the container config: remove it from .env once
it is in the keyring; `keychain_env.py compose` warns about it).

Skipped, with one line on stderr naming the file but never its content: names
that are not variable names, reserved or non-secret-looking names, entries that
are not regular files (symlinks included), files that cannot be read or decoded
or that contain a NUL byte, and files over 64 KiB. A missing secrets directory
is normal (no keyring, or a plain `docker run`): the command just runs.

Limits, i.e. where the values still are:
  - Compose copies environment-sourced secrets INTO the container, so the files
    /run/secrets/<VAR> live in the container's writable layer: `docker diff`
    lists them, and `docker export` / `docker commit` include them, also while
    the container is stopped. They are gone only when the container is removed
    (`docker compose down` or `rm`), not when it stops.
  - The files are mode 0444, so any user inside the container can read them,
    not only the server's user; so can anyone who can `docker exec` into it,
    and the process environment (/proc/<pid>/environ) as well. Docker access
    on the host is effectively root.
  - On the host, the `docker compose` process that created the container holds
    the values in its environment while it runs (`make docker-run-api` runs in
    the foreground), readable by the same user or root.

Environment:
  TRADINGAGENTS_SECRETS_DIR  secrets directory (default /run/secrets; for tests)

Stdlib only and no import of `tradingagents`: it runs before the app starts.
"""

from __future__ import annotations

import os
import re
import stat
import sys

DEFAULT_DIR = "/run/secrets"
DIR_VAR = "TRADINGAGENTS_SECRETS_DIR"
MAX_BYTES = 64 * 1024
VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# A secret's name must contain one of these (see the docstring).
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS")
# Never loaded, even when they look like a secret.
RESERVED_NAMES = frozenset({
    "PATH", "HOME", "SHELL", "USER", "PWD", "IFS",
    DIR_VAR, "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
})
RESERVED_PREFIXES = ("LD_", "DYLD_", "PYTHON", "SSL_CERT_")
# O_NOFOLLOW: a symlink swapped in after the scan fails to open (ELOOP).
# O_NONBLOCK: a FIFO swapped in does not block the open; fstat rejects it.
OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
USAGE = "usage: container_secrets.py <command> [args...]\n"


def note(msg: str) -> None:
    print(f"container-secrets: {msg}", file=sys.stderr)


def name_problem(name: str) -> str | None:
    """Why `name` may not be loaded as a secret, or None if it may."""
    if not VAR_RE.match(name):
        return "not a variable name"
    if name in RESERVED_NAMES or name.startswith(RESERVED_PREFIXES):
        return "reserved variable name"
    if not any(marker in name for marker in SECRET_MARKERS):
        return "not a secret name (needs KEY, TOKEN, SECRET, PASSWORD or PASS)"
    return None


def read_fd(fd: int, limit: int) -> bytes:
    """Up to `limit` bytes from `fd` (fewer only at end of file)."""
    chunks: list[bytes] = []
    size = 0
    while size < limit:
        chunk = os.read(fd, limit - size)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def read_secret(path: str, name: str) -> str | None:
    """The value of one secret file, or None (with a note) if it is skipped."""
    try:
        fd = os.open(path, OPEN_FLAGS)
    except OSError as exc:  # ELOOP for a symlink swapped in, EACCES, ...
        note(f"skipping {name}: cannot read it ({type(exc).__name__})")
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            note(f"skipping {name}: not a regular file")
            return None
        data = read_fd(fd, MAX_BYTES + 1)
    except OSError as exc:
        note(f"skipping {name}: cannot read it ({type(exc).__name__})")
        return None
    finally:
        os.close(fd)
    if len(data) > MAX_BYTES:
        note(f"skipping {name}: larger than {MAX_BYTES} bytes")
        return None
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError:
        note(f"skipping {name}: not valid UTF-8")
        return None
    if "\0" in value:
        note(f"skipping {name}: contains a NUL byte")
        return None
    return value.rstrip("\r\n")


def load_secrets(directory: str) -> None:
    """Set os.environ[NAME] from each usable file in `directory`."""
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except FileNotFoundError:
        return
    except OSError as exc:
        note(f"cannot list the secrets directory ({type(exc).__name__}); loading none")
        return
    for entry in entries:
        name = entry.name
        problem = name_problem(name)
        if problem:
            note(f"skipping {name!r}: {problem}")
            continue
        try:
            regular = entry.is_file(follow_symlinks=False)  # symlinks: not regular
        except OSError:
            regular = False
        if not regular:
            note(f"skipping {name}: not a regular file")
            continue
        value = read_secret(entry.path, name)
        if value is None or not value.strip():
            continue
        os.environ[name] = value  # overrides env_file on purpose: see the docstring


def main(argv: list[str] | None = None) -> None:
    cmd = sys.argv[1:] if argv is None else list(argv)
    if not cmd:
        sys.stderr.write(USAGE)
        sys.exit(2)
    load_secrets(os.environ.get(DIR_VAR) or DEFAULT_DIR)
    try:
        os.execvp(cmd[0], cmd)
    except FileNotFoundError:
        note(f"command not found: {cmd[0]}")
        sys.exit(127)
    except OSError as exc:
        note(f"cannot run {cmd[0]}: {exc.strerror}")
        sys.exit(126)


if __name__ == "__main__":
    main()
