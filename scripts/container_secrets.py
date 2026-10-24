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

For each regular file in the secrets directory whose name is a variable name
(^[A-Z][A-Z0-9_]*$), the content is read as UTF-8 and trailing CR/LF are
stripped; an empty or whitespace-only result is skipped, so the service's
env_file still supplies that variable.

A secret OVERRIDES a value already in the environment. The service's env_file
(.env) has already set its variables when this runs, and a key in the keyring
must win over .env inside the container, as it does in the env-var mode where
compose's `environment:` overrides `env_file:`. Only keys exported from the
keyring become secrets, so a key that is only in .env is left alone.

Skipped, with one line on stderr naming the file but never its content: names
that are not variable names, entries that are not regular files, files that
cannot be read or decoded or that contain a NUL byte, and files over 64 KiB.
A missing secrets directory is normal (no keyring, or a plain `docker run`):
the command just runs.

Limits: anyone who can `docker exec` into the container can still read
/run/secrets and the process environment (/proc/<pid>/environ); Docker access on
the host is effectively root.

Environment:
  TRADINGAGENTS_SECRETS_DIR  secrets directory (default /run/secrets; for tests)

Stdlib only and no import of `tradingagents`: it runs before the app starts.
"""

from __future__ import annotations

import os
import re
import sys

DEFAULT_DIR = "/run/secrets"
DIR_VAR = "TRADINGAGENTS_SECRETS_DIR"
MAX_BYTES = 64 * 1024
VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
USAGE = "usage: container_secrets.py <command> [args...]\n"


def note(msg: str) -> None:
    print(f"container-secrets: {msg}", file=sys.stderr)


def read_secret(path: str, name: str) -> str | None:
    """The value of one secret file, or None (with a note) if it is skipped."""
    try:
        with open(path, "rb") as f:
            data = f.read(MAX_BYTES + 1)
    except OSError as exc:
        note(f"skipping {name}: cannot read it ({type(exc).__name__})")
        return None
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
        if not VAR_RE.match(name):
            note(f"skipping {name!r}: not a variable name")
            continue
        try:
            regular = entry.is_file()
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
