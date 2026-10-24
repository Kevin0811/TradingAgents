"""Tests for scripts/keychain_env.py and the tasks.py helpers that call it.

The real OS keyring is never touched: the script runs in-process with an
in-memory ``keyring`` backend (or ``keyring`` made unimportable), with
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
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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

    def __init__(self, argv, env=None, cwd=None):
        self.argv, self.env, self.cwd = list(argv), dict(env), cwd
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
        pass


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

    def main(self, *args, stdin_lines=None):
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
def h(tmp_path, monkeypatch, capsys):
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


def test_api_keychain_run_already_set_variable_wins(h, store, monkeypatch):
    put(store, h, "OPENAI_API_KEY", "sk-from-keyring")
    put(store, h, "ANTHROPIC_API_KEY", "sk-anthropic-keyring")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # set but empty: filled in
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


def test_api_keychain_add_rejects_bad_name(h, store):
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


def test_api_keychain_compose_empty_override_and_global_flags(h, store, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", h.secret("sk-shell-openai"))
    res = h.main(
        "compose", "--service", "a", "--service", "b",
        "--", "--profile", "ollama", "run", "--rm", "a",
    )
    assert res.child.override == "services:\n  a: {}\n  b: {}\n"
    assert res.child.argv[2:4] == ["-f", "docker-compose.yml"]
    assert res.child.argv[6:] == ["--profile", "ollama", "run", "--rm", "a"]


def test_api_keychain_compose_deletes_override_when_child_fails(h, store):
    put(store, h, "OPENAI_API_KEY", "sk-fake-openai")
    FakePopen.returncode = 17
    res = h.main("compose", "--service", "api", "--", "up")
    assert res.code == 17
    assert not Path(res.child.override_path).exists()


def test_api_keychain_compose_deletes_override_when_spawn_raises(h, store, monkeypatch, tmp_path):
    seen = []

    def boom(argv, **kwargs):
        seen.append(argv[argv.index("-f", 3) + 1])
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(RuntimeError):
        h.mod.main(["compose", "--service", "api", "--", "up"])
    assert seen and not Path(seen[0]).exists()


def test_api_keychain_compose_interrupt_waits_for_child(h, store):
    FakePopen.waits = [KeyboardInterrupt(), KeyboardInterrupt()]
    FakePopen.returncode = 130
    res = h.main("compose", "--service", "api", "--", "up")
    assert res.code == 130
    assert not Path(res.child.override_path).exists()


# ---- key names ----------------------------------------------------------------


def test_api_keychain_names_from_env_example_variants(h, store):
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


def fake_fail_keyring(monkeypatch):
    """A stand-in `keyring` package whose only backend is the fail backend."""
    pkg = types.ModuleType("keyring")
    backends = types.ModuleType("keyring.backends")
    fail = types.ModuleType("keyring.backends.fail")

    class Keyring:
        def get_password(self, service, username):
            raise AssertionError("the fail backend must not be queried")

    fail.Keyring = Keyring
    pkg.backends, backends.fail = backends, fail
    pkg.get_keyring = Keyring
    for name, module in (("keyring", pkg), ("keyring.backends", backends),
                         ("keyring.backends.fail", fail)):
        monkeypatch.setitem(sys.modules, name, module)


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
            raise KeyringLocked("locked")
        return real_get(service, username)

    h.mp.setattr(backend, "get_password", locked)
    res = h.main("run", "--", "mycmd")
    assert res.child.argv == ["mycmd"]
    assert "ANTHROPIC_API_KEY" not in res.env  # looked up before the failure
    assert EXPORTED not in res.env
    assert res.err.count(HINT) == 1


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


def test_api_keychain_analyze_quotes_ticker_and_date(monkeypatch):
    monkeypatch.setattr(tasks, "os", OsProxy("posix"))
    c = MagicMock()
    tasks.analyze.body(c, ticker="AAPL; touch /tmp/pwned", date="2024-01-02 $(id)")
    (cmd,), _ = c.run.call_args
    argv = shlex.split(cmd)
    assert argv[:4] == [sys.executable, str(SCRIPT), "run", "--"]
    assert argv[-4:] == ["analyze", "AAPL; touch /tmp/pwned", "--date", "2024-01-02 $(id)"]

    monkeypatch.setattr(tasks, "os", OsProxy("nt"))
    tasks.analyze.body(c, ticker="BRK B", date="2024-01-02")
    (cmd,), _ = c.run.call_args
    assert cmd.endswith('tradingagents analyze "BRK B" --date 2024-01-02')
