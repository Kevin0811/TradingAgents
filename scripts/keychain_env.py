#!/usr/bin/env python3
"""Keep API keys in the OS keyring instead of .env (macOS, Linux, Windows).

  python scripts/keychain_env.py run -- <command> [args...]
      run a command with the stored keys exported
  python scripts/keychain_env.py compose [--service NAME]... -- <compose args...>
      run `docker compose` with an override passing those keys into NAME
  python scripts/keychain_env.py add <VAR>       store a key (hidden prompt)
  python scripts/keychain_env.py remove <VAR>    delete a stored key
  python scripts/keychain_env.py list            which keys are stored (no values)

Each key is a `keyring` item: service "tradingagents" (or
$TRADINGAGENTS_KEYRING_SERVICE), username the variable name. That is the macOS
Keychain, the Windows Credential Manager or the Linux Secret Service.

`run` exports only the keys that have a non-empty item and are not already set,
so .env still supplies the rest: python-dotenv never overrides a variable that
is already set, and an empty export would count as set. A whitespace-only value
counts as empty everywhere (`run` skips it, `list` shows `empty`, `add` refuses
it). Without a usable keyring, or when a lookup fails, it prints one hint,
exports nothing and runs the command unchanged: it never fails before the
command starts.

Only a real OS keyring counts as usable. Any backend from `keyrings.alt`
(PlaintextKeyring, EncryptedKeyring, ...) or with a priority below 1 does not,
and neither does a ChainerBackend that has any such member: the chainer reads
from every member and writes to the first that accepts, so one weak member
could receive or serve a key.

Environment:
  TRADINGAGENTS_KEYRING_SERVICE  keyring service name (default "tradingagents")
  TRADINGAGENTS_KEYCHAIN_VARS    extra key names to look up, space-separated
  TRADINGAGENTS_KEYCHAIN_QUIET   "1" silences the hints `run`/`compose` print
                                 when they fall back to .env only

Standalone on purpose: stdlib plus the optional `keyring` package, and no
import of `tradingagents`, so it works before the package is installed.
"""

from __future__ import annotations

import getpass
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SERVICE = "tradingagents"
# `run` records here the names (never values) of the keys it exported from the
# keyring, so `compose` passes only those into a container.
EXPORTED_VAR = "TRADINGAGENTS_KEYCHAIN_EXPORTED"
QUIET_VAR = "TRADINGAGENTS_KEYCHAIN_QUIET"
EXTRA_VARS = ("AZURE_OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY")
VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SERVICE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# Secret-looking names in .env.example, also commented out (`#FOO_KEY=`,
# `  ## FOO_KEY=`) or after `export`.
EXAMPLE_RE = re.compile(r"^\s*#*\s*(export\s+)?([A-Z][A-Z0-9_]*(KEY|TOKEN|SECRET)[A-Z0-9_]*)=")
HINT = (
    "no OS keyring available; using .env only "
    "(pip install keyring; on Ubuntu also gnome-keyring or another Secret Service)"
)
USAGE = """\
usage: keychain_env.py run -- <command> [args...]
       keychain_env.py compose [--service NAME]... -- <docker compose args...>
       keychain_env.py add <VAR>
       keychain_env.py remove <VAR>
       keychain_env.py list
"""


class NoKeyring(Exception):
    """No usable keyring, or a lookup failed; `message` says which (no secrets)."""

    def __init__(self, message: str = HINT):
        super().__init__(message)
        self.message = message


def err(msg: str) -> None:
    print(f"keychain-env: {msg}", file=sys.stderr)


def usage() -> None:
    sys.stderr.write(USAGE)
    sys.exit(2)


def service() -> str:
    return os.environ.get("TRADINGAGENTS_KEYRING_SERVICE") or DEFAULT_SERVICE


def is_var(name: str) -> bool:
    return bool(VAR_RE.match(name))


def check_var(args: list[str]) -> str:
    if len(args) != 1:
        usage()
    if not is_var(args[0]):
        err(f"not a variable name: '{args[0]}'")
        sys.exit(2)
    return args[0]


def key_vars() -> list[str]:
    """Known key names, sorted: .env.example, EXTRA_VARS, $TRADINGAGENTS_KEYCHAIN_VARS."""
    names = set(EXTRA_VARS)
    example = ROOT / ".env.example"
    try:
        text = example.read_text(encoding="utf-8", errors="replace") if example.is_file() else ""
    except OSError as exc:  # unreadable: fall back to the other sources
        err(f"cannot read .env.example ({type(exc).__name__}); looking up the other key names only")
        text = ""
    for line in text.splitlines():
        m = EXAMPLE_RE.match(line)
        if m and not m.group(2).startswith("TRADINGAGENTS_"):
            names.add(m.group(2))
    for word in os.environ.get("TRADINGAGENTS_KEYCHAIN_VARS", "").split():
        if is_var(word):
            names.add(word)
        else:
            err(f"ignoring TRADINGAGENTS_KEYCHAIN_VARS entry '{word}': not a variable name")
    return sorted(names)


def backend_name(backend) -> str:
    cls = type(backend)
    return f"{cls.__module__}.{cls.__qualname__}"


def is_weak(backend) -> bool:
    """True for a backend that is not an OS keyring: keyrings.alt or priority < 1."""
    module = type(backend).__module__ or ""
    if module == "keyrings.alt" or module.startswith("keyrings.alt."):
        return True
    try:
        return not backend.priority >= 1
    except Exception:  # priority raises when the backend is not available
        return True


def chain_members(backend) -> list | None:
    """The members of a ChainerBackend, or None if `backend` is not one."""
    try:
        from keyring.backends.chainer import ChainerBackend
    except Exception:
        return None
    if not isinstance(backend, ChainerBackend):
        return None
    try:
        return list(backend.backends)
    except Exception:
        return []


def load_keyring():
    """The `keyring` module with a usable backend, or raise NoKeyring."""
    try:
        import keyring
        import keyring.backends.fail
    except Exception as exc:  # ImportError, or a backend failing at import
        raise NoKeyring from exc
    try:
        backend = keyring.get_keyring()
    except Exception as exc:
        raise NoKeyring from exc
    if isinstance(backend, keyring.backends.fail.Keyring):
        raise NoKeyring
    weak = f"keyring backend {backend_name(backend)} is not an OS keyring; using .env only"
    members = chain_members(backend)
    if members is not None:
        # Chainer: every member must be an OS keyring (see the module docstring).
        if not members or any(is_weak(b) for b in members):
            names = ", ".join(backend_name(b) for b in members) or "none"
            raise NoKeyring(f"{weak} (members: {names})")
    elif is_weak(backend):
        raise NoKeyring(weak)
    return keyring


def get_value(kr, var: str) -> str | None:
    """The stored value without trailing CR/LF, or None without an item.

    Raises NoKeyring when the backend fails (no Secret Service, locked keyring)
    or returns something other than text. The message names only the exception
    type: str(exc) could quote backend details.
    """
    try:
        value = kr.get_password(service(), var)
    except Exception as exc:  # keyring.errors.KeyringError, D-Bus errors, ...
        raise NoKeyring(
            f"keyring lookup failed ({type(exc).__name__}); using .env only"
        ) from exc
    if value is None:
        return None
    if not isinstance(value, str):
        raise NoKeyring(f"keyring returned a non-text value for {var}; using .env only")
    return value.rstrip("\r\n")


def is_empty(value: str | None) -> bool:
    return value is None or not value.strip()


def keyring_env() -> dict[str, str]:
    """The child environment, with the keys found in the keyring exported.

    Only keys that are unset (or empty) here and have a non-empty item are set;
    a missing or empty item is not set at all. Without a usable keyring, or if
    any lookup fails, the environment is returned unchanged (no partial
    exports), after one hint on stderr.
    """
    env = dict(os.environ)
    found: dict[str, str] = {}
    try:
        kr = load_keyring()
        for var in key_vars():
            if not is_empty(env.get(var)):
                continue
            value = get_value(kr, var)
            if not is_empty(value):
                found[var] = value
    except NoKeyring as exc:
        if os.environ.get(QUIET_VAR) != "1":
            err(exc.message)
        return env
    env.update(found)
    inherited = env.get(EXPORTED_VAR, "")
    env[EXPORTED_VAR] = " ".join(w for w in (inherited, *found) if w)
    return env


def exit_code(returncode: int) -> int:
    if os.name == "nt":
        # Windows reports the exit status as an unsigned DWORD (0xC000013A after
        # Ctrl-C); sys.exit needs it as a signed 32-bit int.
        return returncode - 2**32 if returncode > 2**31 - 1 else returncode
    # A child killed by signal N (negative on POSIX) exits like a shell would.
    return 128 - returncode if returncode < 0 else returncode


def wait_child(argv: list[str], env: dict[str, str], cwd: Path | None = None) -> int:
    """Run argv and return its exit code, letting it finish its own Ctrl-C.

    Ctrl-C in a terminal reaches the child as well (the POSIX foreground process
    group, every process on a Windows console), so a KeyboardInterrupt here only
    waits on: docker compose, say, may still be stopping containers. A SIGINT
    sent to this Python process alone (`kill -INT <pid>`, not a terminal Ctrl-C)
    is therefore NOT forwarded: the child never sees it and keeps running.

    SIGTERM and SIGHUP are passed on to the child. Their handlers are installed
    before the child is spawned, so a signal arriving in between is kept and
    sent as soon as the child exists, rather than killing this process and
    orphaning the child.
    """
    child: list[subprocess.Popen] = []
    pending: list[int] = []
    previous = {}
    if os.name != "nt":
        def forward(signum, _frame):
            if child:
                child[0].send_signal(signum)
            else:
                pending.append(signum)

        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, forward)
    try:
        try:
            child.append(subprocess.Popen(argv, env=env, cwd=cwd))
        except FileNotFoundError:
            err(f"command not found: {argv[0]}")
            return 127
        proc = child[0]
        while pending:
            proc.send_signal(pending.pop(0))
        while True:
            try:
                return exit_code(proc.wait())
            except KeyboardInterrupt:
                continue
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def strip_dashdash(args: list[str]) -> list[str]:
    return args[1:] if args[:1] == ["--"] else args


def cmd_run(args: list[str]) -> None:
    argv = strip_dashdash(args)
    if not argv:
        usage()
    env = keyring_env()
    if os.name == "nt":
        sys.exit(wait_child(argv, env))
    try:
        # exec: signals and Ctrl-C go straight to the command, and its exit
        # code is ours.
        os.execvpe(argv[0], argv, env)
    except FileNotFoundError:
        err(f"command not found: {argv[0]}")
        sys.exit(127)
    except OSError as exc:
        err(f"cannot run {argv[0]}: {exc.strerror}")
        sys.exit(126)


def compose_override(services: list[str], names: list[str]) -> str:
    """A compose override passing `names` (never values) into each service."""
    lines = ["services:"]
    for svc in services:
        if names:
            lines += [f"  {svc}:", "    environment:"]
            lines += [f"      - {name}" for name in names]
        else:
            lines.append(f"  {svc}: {{}}")
    return "\n".join(lines) + "\n"


def remove_override(path: str) -> None:
    """Delete the compose override, even if Ctrl-C lands meanwhile.

    A KeyboardInterrupt raised inside the deletion just retries it. An OSError
    (a Windows virus scanner or another process holding the file) is reported
    by type only and not raised, so the child's exit code is still returned.
    """
    while True:
        try:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                err(f"could not delete the temporary compose override {path} "
                    f"({type(exc).__name__}); it holds key names only")
            return
        except KeyboardInterrupt:
            continue


def cmd_compose(args: list[str]) -> None:
    """`docker compose` with an override naming the keys taken from the keyring.

    Only keys exported from the keyring (in this call, or by an outer `run` in
    $TRADINGAGENTS_KEYCHAIN_EXPORTED) are listed: compose reads their values
    from the environment. Any other key, even one set in the shell, is left out
    so the service's env_file (.env) still supplies it, as without a keyring.
    """
    services: list[str] = []
    i = 0
    while i < len(args) and args[i] != "--":
        if args[i] == "--service" and i + 1 < len(args) and SERVICE_RE.match(args[i + 1]):
            services.append(args[i + 1])
            i += 2
        else:
            usage()
    compose_args = args[i + 1 :]
    if not services or not compose_args:
        usage()

    env = keyring_env()
    names = sorted(
        {n for n in env.get(EXPORTED_VAR, "").split() if is_var(n) and env.get(n)}
    )
    fd, path = tempfile.mkstemp(suffix=".yml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(compose_override(services, names))
        argv = ["docker", "compose", "-f", "docker-compose.yml", "-f", path, *compose_args]
        code = wait_child(argv, env, cwd=ROOT)
    finally:
        # The first statement of `finally` sits inside remove_override's
        # retry loop, so a late Ctrl-C cannot skip the deletion.
        remove_override(path)
    sys.exit(code)


def need_keyring():
    try:
        return load_keyring()
    except NoKeyring as exc:
        err(exc.message)
        sys.exit(1)


def cmd_add(args: list[str]) -> None:
    var = check_var(args)
    kr = need_keyring()
    try:
        value = getpass.getpass(f"Value for {var} (hidden): ")
        confirm = getpass.getpass(f"Confirm {var}: ")
    except (KeyboardInterrupt, EOFError):
        print(file=sys.stderr)
        err("cancelled")
        sys.exit(130)
    value, confirm = value.rstrip("\r\n"), confirm.rstrip("\r\n")
    if not value.strip():
        err("empty value; nothing stored")
        sys.exit(1)
    if value != confirm:
        err("the values do not match; nothing stored")
        sys.exit(1)
    try:
        kr.set_password(service(), var, value)
    except Exception as exc:
        err(f"could not store {var}: {type(exc).__name__}")
        sys.exit(1)
    print(f"Stored {var} in the OS keyring (service '{service()}').")


def cmd_remove(args: list[str]) -> None:
    var = check_var(args)
    kr = need_keyring()
    import keyring.errors

    try:
        kr.delete_password(service(), var)
    except keyring.errors.PasswordDeleteError:
        err(f"no stored item for {var} (service '{service()}')")
        sys.exit(1)
    except Exception as exc:
        err(f"could not remove {var}: {type(exc).__name__}")
        sys.exit(1)
    print(f"Removed {var} from the OS keyring (service '{service()}').")


def cmd_list(args: list[str]) -> None:
    """Same lookup as `run`, so an empty item, which `run` skips, shows as `empty`."""
    if args:
        usage()
    kr = need_keyring()
    rows = []
    try:
        for var in key_vars():
            value = get_value(kr, var)
            state = "-" if value is None else ("empty" if is_empty(value) else "stored")
            rows.append(f"  {state:<8} {var}")
    except NoKeyring as exc:
        err(exc.message)
        sys.exit(1)
    print("\n".join(rows))


COMMANDS = {
    "run": cmd_run,
    "compose": cmd_compose,
    "add": cmd_add,
    "remove": cmd_remove,
    "list": cmd_list,
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in COMMANDS:
        usage()
    try:
        COMMANDS[args[0]](args[1:])
    except KeyboardInterrupt:
        # Ctrl-C before any child started (say, during a keyring lookup).
        print(file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
