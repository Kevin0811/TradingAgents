"""Tests for scripts/keychain_env.py and the tasks.py helpers that call it.

The real OS keyring is never touched: the script runs in-process with an
in-memory ``keyring`` backend (the ``store`` fixture), a fake ``keyring``
package, or ``keyring`` made unimportable (the default of the ``h`` fixture for
tests without ``store``, so they pass without ``keyring`` installed), with
``os.execvpe`` and ``subprocess.Popen`` replaced to capture the child's argv and
environment. The few subprocess tests point ``PYTHON_KEYRING_BACKEND`` at a
JSON-file backend in a temp dir.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from invoke.exceptions import Exit

import tasks

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "keychain_env.py"
SERVICE = "tradingagents-unit-test"
EXPORTED = "TRADINGAGENTS_KEYCHAIN_EXPORTED"
HINT = "keychain-env: no OS keyring available"
SECRETISH = re.compile(r"KEY|TOKEN|SECRET|^TRADINGAGENTS_|^PYTHON_KEYRING")


def load_script():
    spec = importlib.util.spec_from_file_location("keychain_env_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Execd(BaseException):
    """Raised by the fake os.execvpe in place of replacing the process."""

    def __init__(self, file, argv, env):
        super().__init__(file)
        self.file, self.argv, self.env = file, list(argv), dict(env)


class FakePopen:
    """Records the child; reads the compose override while it still exists."""

    instances: list[FakePopen] = []
    returncode = 0
    waits: list = []  # exceptions to raise from wait() before returning
    on_spawn = None  # called while "spawning", before the child exists

    def __init__(self, argv, env=None, cwd=None):
        if FakePopen.on_spawn:
            FakePopen.on_spawn()
        self.argv, self.env, self.cwd = list(argv), dict(env), cwd
        self.signals: list[int] = []
        self.override = None
        if "-f" in self.argv:
            path = self.argv[self.argv.index("-f", self.argv.index("-f") + 1) + 1]
            self.override_path = path
            self.override = Path(path).read_text()
        FakePopen.instances.append(self)

    def wait(self):
        if FakePopen.waits:
            raise FakePopen.waits.pop(0)
        return FakePopen.returncode

    def send_signal(self, signum):
        self.signals.append(signum)


class OsProxy:
    """``os`` with another ``name``, without changing the real os.name."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        return getattr(os, attr)


class Result:
    def __init__(self, code, out, err, child):
        self.code, self.out, self.err, self.child = code, out, err, child

    @property
    def env(self):
        return self.child.env

    @property
    def output(self):
        return self.out + self.err


class Harness:
    """The script in-process, with its own .env.example and a clean environment."""

    def __init__(self, tmp_path, monkeypatch, capsys):
        self.mp, self.capsys = monkeypatch, capsys
        self.mod = load_script()
        self.root = tmp_path / "root"
        self.root.mkdir()
        self.mod.ROOT = self.root
        self.env_example("OPENAI_API_KEY=\nANTHROPIC_API_KEY=\n")
        for name in list(os.environ):
            if SECRETISH.search(name):
                monkeypatch.delenv(name)
        monkeypatch.setenv("TRADINGAGENTS_KEYRING_SERVICE", SERVICE)
        self.values: list[str] = []
        monkeypatch.setattr(os, "execvpe", self._execvpe)
        FakePopen.instances, FakePopen.returncode, FakePopen.waits = [], 0, []
        FakePopen.on_spawn = None
        monkeypatch.setattr(subprocess, "Popen", FakePopen)

    @staticmethod
    def _execvpe(file, argv, env):
        raise Execd(file, argv, env)

    def env_example(self, text):
        (self.root / ".env.example").write_text(text)

    def secret(self, value):
        """Remember a value that must never show up in any output."""
        self.values.append(value)
        return value

    def main(self, *args):
        child, code = None, 0
        try:
            self.mod.main(list(args))
        except Execd as e:
            child = e
        except SystemExit as e:
            code = e.code
        if child is None and FakePopen.instances:
            child = FakePopen.instances[-1]
        cap = self.capsys.readouterr()
        res = Result(code, cap.out, cap.err, child)
        for value in self.values:
            if value.strip():
                assert value.strip() not in res.output
        return res


@pytest.fixture
def h(request, tmp_path, monkeypatch, capsys):
    if "store" not in request.fixturenames:
        # Without the in-memory store, never reach the real OS keyring.
        monkeypatch.setitem(sys.modules, "keyring", None)
    return Harness(tmp_path, monkeypatch, capsys)


@pytest.fixture
def store(monkeypatch):
    """An in-memory keyring backend installed as the keyring; its dict."""
    keyring = pytest.importorskip("keyring")
    from keyring.backend import KeyringBackend
    from keyring.errors import PasswordDeleteError

    items: dict[tuple[str, str], str] = {}

    class MemoryKeyring(KeyringBackend):
        priority = 1

        def get_password(self, service, username):
            return items.get((service, username))

        def set_password(self, service, username, password):
            items[(service, username)] = password

        def delete_password(self, service, username):
            if items.pop((service, username), None) is None:
                raise PasswordDeleteError("not found")

    core = importlib.import_module("keyring.core")
    previous = core._keyring_backend
    keyring.set_keyring(MemoryKeyring())
    yield items
    core._keyring_backend = previous


def put(items, h, var, value):
    items[(SERVICE, var)] = value
    if value.strip():
        h.secret(value)


# ---- run ----------------------------------------------------------------------


def test_api_keychain_run_exports_stored_key(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    res = h.main("run", "--", "mycmd", "--flag", "x y")
    assert res.code == 0
    assert res.child.file == "mycmd"
    assert res.child.argv == ["mycmd", "--flag", "x y"]
    assert res.env["OPENAI_API_KEY"] == "sk-fake-openai"
    assert res.env[EXPORTED] == "OPENAI_API_KEY"
    assert HINT not in res.err


def test_api_keychain_run_missing_and_empty_items_are_not_set(h, store):
    put(store, h, "OPENAI_API_KEY", "")
    res = h.main("run", "--", "mycmd")
    # Not even an empty export: python-dotenv would treat that as authoritative.
    assert "OPENAI_API_KEY" not in res.env
    assert "ANTHROPIC_API_KEY" not in res.env
    assert "AZURE_OPENAI_API_KEY" not in res.env
    assert res.env[EXPORTED] == ""


@pytest.mark.parametrize("blank", ["   ", " \t\r\n", "\n"])
def test_api_keychain_whitespace_only_counts_as_empty(h, store, blank):
    put(store, h, "OPENAI_API_KEY", blank)
    put(store, h, "ANTHROPIC_API_KEY", "sk-fake-anthropic")
    res = h.main("run", "--", "mycmd")
    assert "OPENAI_API_KEY" not in res.env
    assert res.env[EXPORTED] == "ANTHROPIC_API_KEY"
    assert listing(h.main("list"))["OPENAI_API_KEY"] == "empty"

    answers = iter([blank, blank])
    h.mp.setattr(h.mod.getpass, "getpass", lambda prompt="": next(answers))
    res = h.main("add", "AZURE_OPENAI_API_KEY")
    assert res.code == 1
    assert "empty value" in res.err
    assert (SERVICE, "AZURE_OPENAI_API_KEY") not in store


def test_api_keychain_run_already_set_variable_wins(h, store, monkeypatch):
    put(store, h, "OPENAI_API_KEY", "sk-from-keyring")
    put(store, h, "ANTHROPIC_API_KEY", "sk-anthropic-keyring")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    monkeypatch.setenv("ANTHROPIC_API_KEY", " ")  # set but blank: filled in
    res = h.main("run", "--", "mycmd")
    assert res.env["OPENAI_API_KEY"] == "sk-from-shell"
    assert res.env["ANTHROPIC_API_KEY"] == "sk-anthropic-keyring"
    assert res.env[EXPORTED].split() == ["ANTHROPIC_API_KEY"]


def test_api_keychain_run_appends_to_inherited_exports(h, store, monkeypatch):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    monkeypatch.setenv(EXPORTED, "EARLIER_KEY")
    res = h.main("run", "--", "mycmd")
    assert res.env[EXPORTED] == "EARLIER_KEY OPENAI_API_KEY"


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("trail\n", "trail"),
        ("crlf\r\n", "crlf"),
        ("many\r\n\n", "many"),
        ("héllo", "héllo"),
        ('quo"te', 'quo"te'),
        ("0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef"),
    ],
)
def test_api_keychain_run_strips_trailing_newlines(h, store, stored, expected):
    put(store, h, "OPENAI_API_KEY", stored)
    res = h.main("run", "--", "mycmd")
    assert res.env["OPENAI_API_KEY"] == expected


def test_api_keychain_run_on_windows_waits_and_propagates(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    h.mp.setattr(h.mod, "os", OsProxy("nt"))
    FakePopen.returncode = 3
    # Ctrl-C reaches the child too; the script keeps waiting for it to exit.
    FakePopen.waits = [KeyboardInterrupt()]
    res = h.main("run", "--", "mycmd", "a")
    assert res.code == 3
    assert res.child.argv == ["mycmd", "a"]
    assert res.env["OPENAI_API_KEY"] == "sk-fake-openai"
    assert "Traceback" not in res.err


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0xC000013A, -1073741510), (0xFFFFFFFF, -1), (2**31 - 1, 2**31 - 1), (1, 1)],
)
def test_api_keychain_windows_exit_code_is_signed(h, returncode, expected):
    h.mp.setattr(h.mod, "os", OsProxy("nt"))
    FakePopen.returncode = returncode
    assert h.main("run", "--", "mycmd").code == expected
    assert h.main("compose", "--service", "api", "--", "up").code == expected


def test_api_keychain_posix_signal_exit_code(h):
    h.mp.setattr(h.mod, "os", OsProxy("posix"))
    assert h.mod.exit_code(-15) == 143
    assert h.mod.exit_code(0xC000013A) == 0xC000013A


def test_api_keychain_usage_errors_exit_2(h):
    for args in ((), ("bogus",), ("run",), ("run", "--"), ("add",), ("list", "x"),
                 ("compose", "--", "up"), ("compose", "--service", "api"),
                 ("compose", "--service", "bad name!", "--", "up")):
        res = h.main(*args)
        assert res.code == 2, args
        assert "usage:" in res.err
        assert res.child is None


# ---- list / add / remove ------------------------------------------------------


def listing(res):
    rows = [line.split() for line in res.out.splitlines() if line.strip()]
    return {name: state for state, name in rows}


def test_api_keychain_list_states_without_values(h, store):
    put(store, h, "OPENAI_API_KEY", "")
    put(store, h, "ANTHROPIC_API_KEY", "sk-fake-anthropic")
    res = h.main("list")
    assert res.code == 0
    assert listing(res) == {
        "ALPHA_VANTAGE_API_KEY": "-",
        "ANTHROPIC_API_KEY": "stored",
        "AZURE_OPENAI_API_KEY": "-",
        "OPENAI_API_KEY": "empty",
    }


def test_api_keychain_add_stores_after_confirmation(h, store):
    answers = iter([h.secret("sk-typed-value"), "sk-typed-value"])
    h.mp.setattr(h.mod.getpass, "getpass", lambda prompt="": next(answers))
    res = h.main("add", "OPENAI_API_KEY")
    assert res.code == 0
    assert store[(SERVICE, "OPENAI_API_KEY")] == "sk-typed-value"
    assert res.out == f"Stored OPENAI_API_KEY in the OS keyring (service '{SERVICE}').\n"


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [("sk-one-value", "sk-other-value", "do not match"), ("", "", "empty value")],
)
def test_api_keychain_add_refuses_mismatch_or_empty(h, store, first, second, message):
    h.secret(first)
    h.secret(second)
    answers = iter([first, second])
    h.mp.setattr(h.mod.getpass, "getpass", lambda prompt="": next(answers))
    res = h.main("add", "OPENAI_API_KEY")
    assert res.code == 1
    assert message in res.err
    assert store == {}


def test_api_keychain_add_rejects_bad_name(h):
    res = h.main("add", "lower_case")
    assert res.code == 2
    assert "not a variable name" in res.err


def test_api_keychain_remove(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    res = h.main("remove", "OPENAI_API_KEY")
    assert res.code == 0
    assert store == {}
    res = h.main("remove", "OPENAI_API_KEY")
    assert res.code == 1
    assert "no stored item for OPENAI_API_KEY" in res.err


# ---- compose ------------------------------------------------------------------


def test_api_keychain_compose_lists_only_keyring_exports(h, store, monkeypatch):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    # Set in the shell (say ~/.zshrc), not from the keyring: .env must still
    # win inside the container, so the override must not name it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", h.secret("sk-shell-anthropic"))
    # Exported by an outer `run`, so it is still listed.
    monkeypatch.setenv("EARLIER_KEY", h.secret("sk-earlier"))
    monkeypatch.setenv(EXPORTED, "EARLIER_KEY")
    res = h.main("compose", "--service", "api", "--", "up", "api")
    assert res.code == 0
    child = res.child
    assert child.override == (
        "services:\n  api:\n    environment:\n      - EARLIER_KEY\n      - OPENAI_API_KEY\n"
    )
    for value in h.values:
        assert value not in child.override
    assert child.argv == [
        "docker", "compose", "-f", "docker-compose.yml", "-f", child.override_path, "up", "api",
    ]
    assert child.override_path.endswith(".yml")
    assert Path(child.cwd) == h.root
    assert child.env["OPENAI_API_KEY"] == "sk-fake-openai"
    assert not Path(child.override_path).exists()


def test_api_keychain_compose_empty_override_and_global_flags(h, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", h.secret("sk-shell-openai"))
    res = h.main(
        "compose", "--service", "a", "--service", "b",
        "--", "--profile", "ollama", "run", "--rm", "a",
    )
    assert res.child.override == "services:\n  a: {}\n  b: {}\n"
    assert res.child.argv[2:4] == ["-f", "docker-compose.yml"]
    assert res.child.argv[6:] == ["--profile", "ollama", "run", "--rm", "a"]


SECRETS_OVERRIDE = (
    "services:\n"
    "  api:\n"
    "    secrets:\n"
    "      - source: tradingagents_earlier_key\n"
    "        target: EARLIER_KEY\n"
    "      - source: tradingagents_openai_api_key\n"
    "        target: OPENAI_API_KEY\n"
    "secrets:\n"
    "  tradingagents_earlier_key:\n"
    "    environment: EARLIER_KEY\n"
    "  tradingagents_openai_api_key:\n"
    "    environment: OPENAI_API_KEY\n"
)


def test_api_keychain_compose_secrets_lists_only_keyring_exports(h, store, monkeypatch):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    # Only in the shell: .env must still win, so it is not a secret.
    monkeypatch.setenv("ANTHROPIC_API_KEY", h.secret("sk-shell-anthropic"))
    monkeypatch.setenv("EARLIER_KEY", h.secret("sk-earlier"))
    monkeypatch.setenv(EXPORTED, "EARLIER_KEY")
    res = h.main("compose", "--secrets", "--service", "api", "--", "up", "--build", "api")
    assert res.code == 0
    child = res.child
    assert child.override == SECRETS_OVERRIDE
    assert "ANTHROPIC" not in child.override
    assert child.argv == [
        "docker", "compose", "-f", "docker-compose.yml", "-f", child.override_path,
        "up", "--build", "api",
    ]
    # Compose reads the secret values from its own environment.
    assert child.env["OPENAI_API_KEY"] == "sk-fake-openai"
    assert child.env["EARLIER_KEY"] == "sk-earlier"
    assert not Path(child.override_path).exists()


def test_api_keychain_compose_secrets_names_only_no_environment(h, store, monkeypatch):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    monkeypatch.setenv("EARLIER_KEY", h.secret("sk-earlier"))
    monkeypatch.setenv(EXPORTED, "EARLIER_KEY")
    # --secrets after --service is accepted too.
    res = h.main("compose", "--service", "api", "--secrets", "--", "up")
    override = res.child.override
    for value in h.values:
        assert value not in override
    assert "sk-" not in override
    # The only `environment:` keys are the secret sources (top level), never a
    # service-level environment list that would land in Config.Env.
    service_part, _, secrets_part = override.partition("\nsecrets:\n")
    assert "environment" not in service_part
    assert not re.search(r"^\s*- (EARLIER_KEY|OPENAI_API_KEY)$", override, re.M)
    assert secrets_part.count("environment:") == 2


def test_api_keychain_compose_secrets_empty_override(h, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", h.secret("sk-shell-openai"))
    res = h.main("compose", "--secrets", "--service", "a", "--service", "b", "--", "up")
    assert res.child.override == "services:\n  a: {}\n  b: {}\n"
    assert not Path(res.child.override_path).exists()


def test_api_keychain_compose_secrets_two_services_share_top_level(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    res = h.main("compose", "--secrets", "--service", "a", "--service", "b", "--", "up")
    assert res.child.override == (
        "services:\n"
        "  a:\n    secrets:\n"
        "      - source: tradingagents_openai_api_key\n        target: OPENAI_API_KEY\n"
        "  b:\n    secrets:\n"
        "      - source: tradingagents_openai_api_key\n        target: OPENAI_API_KEY\n"
        "secrets:\n"
        "  tradingagents_openai_api_key:\n    environment: OPENAI_API_KEY\n"
    )


def test_api_keychain_compose_secrets_deletes_override_when_child_fails(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    FakePopen.returncode = 3
    res = h.main("compose", "--secrets", "--service", "api", "--", "up")
    assert res.code == 3
    assert "secrets:" in res.child.override
    assert not Path(res.child.override_path).exists()


def test_api_keychain_compose_default_mode_unchanged_by_secrets_flag(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    res = h.main("compose", "--service", "api", "--", "up")
    assert "secrets" not in res.child.override
    assert "environment:" in res.child.override


def test_api_keychain_compose_deletes_override_when_child_fails(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    FakePopen.returncode = 17
    res = h.main("compose", "--service", "api", "--", "up")
    assert res.code == 17
    assert not Path(res.child.override_path).exists()


def test_api_keychain_compose_deletes_override_when_spawn_raises(h, monkeypatch):
    seen = []

    def boom(argv, **kwargs):
        seen.append(argv[argv.index("-f", 3) + 1])
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(RuntimeError):
        h.mod.main(["compose", "--service", "api", "--", "up"])
    assert seen and not Path(seen[0]).exists()


def test_api_keychain_compose_interrupt_waits_for_child(h):
    FakePopen.waits = [KeyboardInterrupt(), KeyboardInterrupt()]
    FakePopen.returncode = 130
    res = h.main("compose", "--service", "api", "--", "up")
    assert res.code == 130
    assert not Path(res.child.override_path).exists()


def test_api_keychain_compose_unlink_survives_late_interrupt(h, monkeypatch):
    real_unlink, calls = os.unlink, []

    def unlink(path):
        calls.append(path)
        if len(calls) == 1:
            raise KeyboardInterrupt  # Ctrl-C lands inside the cleanup
        real_unlink(path)

    monkeypatch.setattr(os, "unlink", unlink)
    FakePopen.returncode = 9
    res = h.main("compose", "--service", "api", "--", "up")
    assert res.code == 9
    assert len(calls) == 2
    assert not Path(res.child.override_path).exists()


def test_api_keychain_compose_unlink_error_keeps_exit_code(h, monkeypatch):
    real_unlink = os.unlink

    def locked(path):
        raise PermissionError("held-by-scanner-detail")

    monkeypatch.setattr(os, "unlink", locked)
    FakePopen.returncode = 17
    res = h.main("compose", "--service", "api", "--", "up")
    monkeypatch.setattr(os, "unlink", real_unlink)
    path = Path(res.child.override_path)
    try:
        assert res.code == 17
        assert "could not delete the temporary compose override" in res.err
        assert "PermissionError" in res.err
        assert "held-by-scanner-detail" not in res.err
        assert "Traceback" not in res.err
    finally:
        path.unlink()


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="POSIX signals")
@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP"])
def test_api_keychain_signal_before_spawn_is_forwarded(h, sig):
    """The handler is in place before Popen; a signal meanwhile reaches the child."""
    signum = getattr(signal, sig)
    before = signal.getsignal(signum)

    def deliver():
        handler = signal.getsignal(signum)
        assert callable(handler) and handler is not before
        handler(signum, None)

    FakePopen.on_spawn = deliver
    res = h.main("compose", "--service", "api", "--", "up")
    assert res.code == 0
    assert res.child.signals == [signum]
    assert signal.getsignal(signum) is before


# ---- key names ----------------------------------------------------------------


def test_api_keychain_names_from_env_example_variants(h):
    h.env_example(
        "export FOO_KEY=\n"
        "  #BAR_TOKEN=\n"
        "##BAZ_SECRET=\n"
        "# QUX_API_KEY=x\n"
        "  ## export QUUX_TOKEN=\n"
        "PLAIN_API_KEY=\n"
        "TRADINGAGENTS_API_KEY=\n"
        "LOG_LEVEL=\n"
        "lower_key=\n"
    )
    names = h.mod.key_vars()
    assert names == sorted(set(names))
    for name in ("FOO_KEY", "BAR_TOKEN", "BAZ_SECRET", "QUX_API_KEY", "QUUX_TOKEN",
                 "PLAIN_API_KEY", "AZURE_OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        assert name in names
    for name in ("TRADINGAGENTS_API_KEY", "LOG_LEVEL", "lower_key"):
        assert name not in names


def test_api_keychain_extra_vars_rejects_bad_names(h, store, monkeypatch):
    put(store, h, "EXTRA_KEY", "sk-extra")
    monkeypatch.setenv("TRADINGAGENTS_KEYCHAIN_VARS", " EXTRA_KEY\tbad-name  *  EXTRA_KEY ")
    res = h.main("run", "--", "mycmd")
    assert res.env["EXTRA_KEY"] == "sk-extra"
    assert res.env[EXPORTED] == "EXTRA_KEY"
    assert "'bad-name'" in res.err
    assert "'*'" in res.err
    assert h.mod.key_vars().count("EXTRA_KEY") == 1


# ---- no usable keyring --------------------------------------------------------


class UnusedBackend:
    """A backend the script must reject without ever querying it."""

    priority = 5

    def get_password(self, service, username):
        raise AssertionError("an unusable backend must not be queried")

    set_password = delete_password = get_password


def make_backend(name, module, priority=5):
    cls = type(name, (UnusedBackend,), {"priority": priority})
    cls.__module__ = module
    return cls


class FailKeyring(UnusedBackend):
    priority = 0


class FakeChainer:
    """Like keyring's ChainerBackend: reads from each member in turn."""

    priority = 10

    def __init__(self, members):
        self.backends = members

    def get_password(self, service, username):
        for member in self.backends:
            value = member.get_password(service, username)
            if value is not None:
                return value
        return None


class SecretServiceLike:
    """An OS-keyring-like backend holding fixed items."""

    priority = 5

    def __init__(self, items):
        self.items = items

    def get_password(self, service, username):
        return self.items.get(username)


SecretServiceLike.__module__ = "keyring.backends.SecretService"


def fake_keyring(monkeypatch, backend):
    """A stand-in `keyring` package whose get_keyring() returns `backend`."""
    pkg = types.ModuleType("keyring")
    backends = types.ModuleType("keyring.backends")
    fail = types.ModuleType("keyring.backends.fail")
    chainer = types.ModuleType("keyring.backends.chainer")
    fail.Keyring, chainer.ChainerBackend = FailKeyring, FakeChainer
    pkg.backends, backends.fail, backends.chainer = backends, fail, chainer
    pkg.get_keyring = lambda: backend
    # Like keyring's module-level API, which the script calls.
    pkg.get_password = lambda service, username: backend.get_password(service, username)
    for name, module in (("keyring", pkg), ("keyring.backends", backends),
                         ("keyring.backends.fail", fail),
                         ("keyring.backends.chainer", chainer)):
        monkeypatch.setitem(sys.modules, name, module)


def fake_fail_keyring(monkeypatch):
    fake_keyring(monkeypatch, FailKeyring())


class RaisingPriority(UnusedBackend):
    @property
    def priority(self):
        raise RuntimeError("backend not available")


Plaintext = make_backend("PlaintextKeyring", "keyrings.alt.file", priority=1)
Encrypted = make_backend("EncryptedKeyring", "keyrings.alt.file", priority=5)
AltRoot = make_backend("Thing", "keyrings.alt", priority=5)
Weak = make_backend("WeakKeyring", "some.vendor", priority=0.5)
Strong = make_backend("Keyring", "keyring.backends.macOS", priority=5)

WEAK_CASES = {
    "alt-plaintext": (lambda: Plaintext(), "keyrings.alt.file.PlaintextKeyring"),
    "alt-encrypted": (lambda: Encrypted(), "keyrings.alt.file.EncryptedKeyring"),
    "alt-root-module": (lambda: AltRoot(), "keyrings.alt.Thing"),
    "low-priority": (lambda: Weak(), "some.vendor.WeakKeyring"),
    "priority-raises": (lambda: RaisingPriority(), "RaisingPriority"),
    "chainer-with-alt": (lambda: FakeChainer([Strong(), Plaintext()]), "FakeChainer"),
    "chainer-weak-only": (lambda: FakeChainer([Weak()]), "FakeChainer"),
    "chainer-empty": (lambda: FakeChainer([]), "FakeChainer"),
}


@pytest.mark.parametrize("case", list(WEAK_CASES))
def test_api_keychain_weak_backend_is_no_keyring(h, monkeypatch, case):
    make, name = WEAK_CASES[case]
    fake_keyring(monkeypatch, make())
    res = h.main("run", "--", "mycmd", "arg")
    assert res.child.argv == ["mycmd", "arg"]
    assert res.env == dict(os.environ)
    assert res.err.count("is not an OS keyring; using .env only") == 1, res.err
    assert f"keyring backend {name}" in res.err or name in res.err
    assert HINT not in res.err

    res = h.main("compose", "--service", "api", "--", "up")
    assert res.child.override == "services:\n  api: {}\n"

    answers = iter([h.secret("sk-never-stored"), "sk-never-stored"])
    h.mp.setattr(h.mod.getpass, "getpass", lambda prompt="": next(answers))
    for args in (("list",), ("remove", "OPENAI_API_KEY"), ("add", "OPENAI_API_KEY")):
        res = h.main(*args)
        assert res.code == 1, args
        assert "is not an OS keyring" in res.err


def test_api_keychain_chainer_of_os_keyrings_is_usable(h, monkeypatch):
    members = [
        SecretServiceLike({"OPENAI_API_KEY": h.secret("sk-from-chain")}),
        SecretServiceLike({}),
    ]
    fake_keyring(monkeypatch, FakeChainer(members))
    res = h.main("run", "--", "mycmd")
    assert res.env["OPENAI_API_KEY"] == "sk-from-chain"
    assert res.err == ""


@pytest.mark.parametrize("how", ["missing", "fail-backend"])
def test_api_keychain_no_keyring_runs_command_with_one_hint(h, monkeypatch, how):
    if how == "missing":
        monkeypatch.setitem(sys.modules, "keyring", None)
    else:
        fake_fail_keyring(monkeypatch)
    monkeypatch.setenv("TRADINGAGENTS_KEYCHAIN_VARS", "EXTRA_KEY")
    res = h.main("run", "--", "mycmd", "arg")
    assert res.child.argv == ["mycmd", "arg"]
    assert res.env == dict(os.environ)
    assert EXPORTED not in res.env
    assert res.err.count(HINT) == 1

    monkeypatch.setenv("TRADINGAGENTS_KEYCHAIN_QUIET", "1")
    res = h.main("run", "--", "mycmd")
    assert res.child.argv == ["mycmd"]
    assert res.err == ""

    res = h.main("compose", "--service", "api", "--", "up")
    assert res.child.override == "services:\n  api: {}\n"
    assert not Path(res.child.override_path).exists()

    for args in (("list",), ("remove", "OPENAI_API_KEY"), ("add", "OPENAI_API_KEY")):
        res = h.main(*args)
        assert res.code == 1, args
        assert HINT in res.err


def test_api_keychain_lookup_error_runs_command_without_partial_exports(h, store):
    from keyring.errors import KeyringLocked

    put(store, h, "ANTHROPIC_API_KEY", "sk-fake-anthropic")
    backend = __import__("keyring").get_keyring()
    real_get = backend.get_password

    def locked(service, username):
        if username == "OPENAI_API_KEY":
            raise KeyringLocked("dbus-detail-must-not-leak")
        return real_get(service, username)

    h.mp.setattr(backend, "get_password", locked)
    res = h.main("run", "--", "mycmd")
    assert res.child.argv == ["mycmd"]
    assert "ANTHROPIC_API_KEY" not in res.env  # looked up before the failure
    assert EXPORTED not in res.env
    assert HINT not in res.err
    assert res.err == "keychain-env: keyring lookup failed (KeyringLocked); using .env only\n"
    assert "dbus-detail-must-not-leak" not in res.err

    res = h.main("list")
    assert res.code == 1
    assert "keyring lookup failed (KeyringLocked)" in res.err
    assert "dbus-detail-must-not-leak" not in res.err


@pytest.mark.parametrize("weird", [b"sk-bytes", 42, ["sk"]])
def test_api_keychain_non_str_value_falls_back_and_runs(h, store, weird):
    put(store, h, "ANTHROPIC_API_KEY", "sk-fake-anthropic")
    store[(SERVICE, "OPENAI_API_KEY")] = weird
    res = h.main("run", "--", "mycmd", "x")
    assert res.child.argv == ["mycmd", "x"]
    assert "ANTHROPIC_API_KEY" not in res.env
    assert EXPORTED not in res.env
    assert "non-text value for OPENAI_API_KEY" in res.err
    assert "Traceback" not in res.err
    res = h.main("compose", "--service", "api", "--", "up")
    assert res.child.override == "services:\n  api: {}\n"
    res = h.main("list")
    assert res.code == 1
    assert "non-text value for OPENAI_API_KEY" in res.err


def unreadable_env_example(monkeypatch):
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == ".env.example":
            raise PermissionError("denied-detail")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


def test_api_keychain_unreadable_env_example_still_runs(h, store, monkeypatch):
    put(store, h, "OPENAI_API_KEY", "sk-only-in-env-example")
    put(store, h, "AZURE_OPENAI_API_KEY", "sk-fake-azure")
    unreadable_env_example(monkeypatch)
    res = h.main("run", "--", "mycmd")
    assert res.child.argv == ["mycmd"]
    assert "cannot read .env.example (PermissionError)" in res.err
    assert "denied-detail" not in res.err
    assert res.env["AZURE_OPENAI_API_KEY"] == "sk-fake-azure"  # EXTRA_VARS still work
    assert "OPENAI_API_KEY" not in res.env

    res = h.main("list")
    assert res.code == 0
    assert listing(res) == {"ALPHA_VANTAGE_API_KEY": "-", "AZURE_OPENAI_API_KEY": "stored"}


# ---- real processes ----------------------------------------------------------


JSON_BACKEND = '''
import json, os
from keyring.backend import KeyringBackend

class JsonKeyring(KeyringBackend):
    priority = 1

    def _load(self):
        with open(os.environ["TEST_KEYRING_FILE"]) as f:
            return json.load(f)

    def get_password(self, service, username):
        return self._load().get(service + "/" + username)

    def set_password(self, service, username, password):
        raise NotImplementedError

    def delete_password(self, service, username):
        raise NotImplementedError
'''

PRINT_ENV = (
    "import json, os, sys; "
    "sys.stdout.write(json.dumps({k: v for k, v in os.environ.items() if k.endswith('_KEY')"
    " or k.startswith('TRADINGAGENTS_KEYCHAIN')})); sys.exit(int(sys.argv[1]))"
)


def run_script(tmp_path, *args, items=None, keyring_importable=True):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not SECRETISH.search(k)}
    env["TRADINGAGENTS_KEYRING_SERVICE"] = SERVICE
    if keyring_importable:
        (lib / "jsonring_under_test.py").write_text(JSON_BACKEND)
        store_file = tmp_path / "items.json"
        store_file.write_text(json.dumps({f"{SERVICE}/{k}": v for k, v in (items or {}).items()}))
        env["TEST_KEYRING_FILE"] = str(store_file)
        env["PYTHON_KEYRING_BACKEND"] = "jsonring_under_test.JsonKeyring"
    else:
        (lib / "keyring").mkdir(exist_ok=True)
        (lib / "keyring" / "__init__.py").write_text("raise ImportError('hidden by test')\n")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(lib), env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env, capture_output=True, text=True, timeout=60, cwd=tmp_path,
    )


def test_api_keychain_real_exec_exports_and_propagates_exit_code(tmp_path):
    pytest.importorskip("keyring")
    res = run_script(
        tmp_path, "run", "--", sys.executable, "-c", PRINT_ENV, "7",
        items={"OPENAI_API_KEY": "sk-real-exec\r\n", "ANTHROPIC_API_KEY": ""},
    )
    assert res.returncode == 7, res.stderr
    env = json.loads(res.stdout)
    assert env["OPENAI_API_KEY"] == "sk-real-exec"
    assert "ANTHROPIC_API_KEY" not in env
    assert env[EXPORTED] == "OPENAI_API_KEY"
    assert "sk-real-exec" not in res.stderr


def test_api_keychain_real_exec_without_keyring(tmp_path):
    res = run_script(
        tmp_path, "run", "--", sys.executable, "-c", PRINT_ENV, "5", keyring_importable=False,
    )
    assert res.returncode == 5
    assert json.loads(res.stdout) == {}
    assert res.stderr.count(HINT) == 1


def test_api_keychain_script_is_standalone():
    tree = ast.parse(SCRIPT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "tradingagents" not in imported
    assert imported - {"keyring"} <= set(sys.stdlib_module_names)


# ---- tasks.py -----------------------------------------------------------------


def test_api_keychain_with_keys_posix(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("posix"))
    argv = shlex.split(tasks.with_keys("python -m cli.main"))
    assert argv == [sys.executable, str(SCRIPT), "run", "--", "python", "-m", "cli.main"]
    assert Path(argv[1]).is_absolute()

    monkeypatch.setattr(tasks, "KEYCHAIN_ENV", Path("/tmp/dir with space/keychain_env.py"))
    monkeypatch.setattr(tasks.sys, "executable", "/opt/py thon/bin/python")
    argv = shlex.split(tasks.with_keys("true"))
    assert argv == [
        "/opt/py thon/bin/python", "/tmp/dir with space/keychain_env.py", "run", "--", "true",
    ]


def test_api_keychain_with_keys_windows(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("nt"))
    monkeypatch.setattr(tasks, "KEYCHAIN_ENV", r"C:\src\Trading Agents\scripts\keychain_env.py")
    monkeypatch.setattr(tasks.sys, "executable", r"C:\Program Files\Python311\python.exe")
    assert tasks.with_keys("python -m cli.main") == (
        r'"C:\Program Files\Python311\python.exe" '
        r'"C:\src\Trading Agents\scripts\keychain_env.py" run -- python -m cli.main'
    )


def run_cmd(task, *args, **kwargs):
    c = MagicMock()
    task.body(c, *args, **kwargs)
    (cmd,), _ = c.run.call_args
    return cmd


def test_api_keychain_analyze_quotes_ticker_and_date_posix(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("posix"))
    argv = shlex.split(run_cmd(tasks.analyze, ticker="^GSPC", date="2024-01-02"))
    assert argv[:4] == [sys.executable, str(SCRIPT), "run", "--"]
    assert argv[4:] == ["tradingagents", "analyze", "^GSPC", "--date", "2024-01-02"]
    argv = shlex.split(run_cmd(tasks.analyze, ticker="BRK.B", checkpoint=True))
    assert argv[4:] == ["tradingagents", "analyze", "--checkpoint", "BRK.B"]


def test_api_keychain_analyze_quotes_every_arg_on_windows(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("nt"))
    cmd = run_cmd(tasks.analyze, ticker="^GSPC", date="2024-01-02")
    assert cmd.endswith('tradingagents analyze "^GSPC" --date "2024-01-02"')
    cmd = run_cmd(tasks.analyze, ticker="ES=F")
    assert cmd.endswith('tradingagents analyze "ES=F"')


@pytest.mark.parametrize("os_name", ["posix", "nt"])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ticker": "A&calc"}, "invalid ticker"),
        ({"ticker": "AAPL; touch /tmp/pwned"}, "invalid ticker"),
        ({"ticker": "BRK B"}, "invalid ticker"),
        ({"ticker": "A" * 33}, "invalid ticker"),
        ({"ticker": "%PATH%"}, "invalid ticker"),
        ({"ticker": "AAPL", "date": "2024-01-02 $(id)"}, "invalid date"),
        ({"ticker": "AAPL", "date": "2024-1-2"}, "invalid date"),
        ({"ticker": "AAPL", "date": "2024-01-02&calc"}, "invalid date"),
    ],
)
def test_api_keychain_analyze_rejects_bad_input(monkeypatch, os_name, kwargs, message):
    monkeypatch.setattr(tasks, "os", OsProxy(os_name))
    c = MagicMock()
    with pytest.raises(Exit) as exc:
        tasks.analyze.body(c, **kwargs)
    assert message in str(exc.value.message)
    assert exc.value.code == 2
    c.run.assert_not_called()


def test_api_keychain_quote_windows(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("nt"))
    assert tasks.quote("AAPL") == '"AAPL"'
    assert tasks.quote("a&b|c<d>e^f") == '"a&b|c<d>e^f"'
    assert tasks.quote(8000) == '"8000"'
    assert tasks.quote("C:\\dir with space\\") == '"C:\\dir with space\\\\"'
    assert tasks.quote("C:\\a\\b.py") == '"C:\\a\\b.py"'
    with pytest.raises(Exit):
        tasks.quote('x" & calc & "')


def test_api_keychain_api_and_cli_use_this_interpreter(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("posix"))
    monkeypatch.setattr(tasks.sys, "executable", "/opt/py thon/bin/python")
    argv = shlex.split(run_cmd(tasks.cli))
    assert argv[3:] == ["--", "/opt/py thon/bin/python", "-m", "cli.main"]
    argv = shlex.split(run_cmd(tasks.api, reload=True))
    assert argv[3:] == [
        "--", "/opt/py thon/bin/python", "run_server.py",
        "--host", "0.0.0.0", "--port", "8000", "--reload",
    ]

    monkeypatch.setattr(tasks, "os", OsProxy("nt"))
    monkeypatch.setattr(tasks.sys, "executable", r"C:\Program Files\Python311\python.exe")
    assert run_cmd(tasks.cli).endswith(r' -- "C:\Program Files\Python311\python.exe" -m cli.main')


@pytest.mark.parametrize(
    ("ollama", "compose"),
    [
        (False, ["run", "--rm", "tradingagents"]),
        (True, ["--profile", "ollama", "run", "--rm", "tradingagents-ollama"]),
    ],
)
def test_api_keychain_docker_task_goes_through_compose(monkeypatch, ollama, compose):
    monkeypatch.setattr(tasks, "os", OsProxy("posix"))
    argv = shlex.split(run_cmd(tasks.docker, ollama=ollama))
    service = compose[-1]
    assert argv == [
        sys.executable, str(SCRIPT), "compose", "--service", service, "--", *compose,
    ]


def test_api_keychain_keychain_remove_task(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("posix"))
    argv = shlex.split(run_cmd(tasks.keychain_remove, "OPENAI_API_KEY"))
    assert argv == [sys.executable, str(SCRIPT), "remove", "OPENAI_API_KEY"]
    for task in (tasks.keychain_remove, tasks.keychain_add):
        with pytest.raises(Exit):
            task.body(MagicMock(), 'X" & calc')
