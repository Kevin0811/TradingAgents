"""Tests for scripts/keychain-env and the tasks.py helpers that call it.

The real Keychain is never touched: every test puts a fake ``security`` (items
kept as files in a temp dir) on a PATH built from scratch, which holds only the
tools the script needs.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tasks

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "keychain-env"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else shutil.which("bash")
PREFIX = "tradingagents."

# Tools the script (and the commands the tests run through it) may call.
TOOLS = ("sed", "grep", "sort", "dirname", "xxd", "env", "cat")

FAKE_SECURITY = r'''#!{python}
"""A stand-in for macOS `security`, serving generic-password items from files."""
import os
import sys

store = os.environ["FAKE_KEYCHAIN_DIR"]
with open(os.path.join(store, "..", "security.log"), "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")

cmd, args = sys.argv[1], sys.argv[2:]
opts, flags, i = {{}}, set(), 0
while i < len(args):
    if args[i] in ("-a", "-s", "-l") and i + 1 < len(args):
        opts[args[i]] = args[i + 1]
        i += 2
    else:
        flags.add(args[i])
        i += 1
path = os.path.join(store, opts.get("-s", ""))


def missing():
    sys.stderr.write("security: SecKeychainSearchCopyNext: The specified item "
                     "could not be found in the keychain.\n")
    sys.exit(44)


if cmd == "find-generic-password":
    if not os.path.exists(path):
        missing()
    value = open(path, "rb").read()
    encoded = any(b < 0x20 or b > 0x7E for b in value)
    if "-w" in flags:
        out = value.hex() if encoded else value.decode()
        sys.stdout.write(out + "\n")
    else:
        sys.stdout.write('keychain: "fake"\nattributes:\n    "svce"<blob>="%s"\n'
                         % opts.get("-s"))
        if "-g" in flags:
            if not value:
                sys.stderr.write("password: \n")
            elif encoded:
                esc = "".join(chr(b) if 0x20 <= b <= 0x7E else "\\%03o" % b for b in value)
                sys.stderr.write('password: 0x%s  "%s"\n' % (value.hex().upper(), esc))
            else:
                sys.stderr.write('password: "%s"\n' % value.decode())
elif cmd == "add-generic-password":
    value = sys.stdin.readline().rstrip("\n")
    with open(path, "w") as f:
        f.write(value)
elif cmd == "delete-generic-password":
    if not os.path.exists(path):
        missing()
    os.remove(path)
    sys.stderr.write("password has been deleted.\n")
else:
    sys.exit(2)
'''


class Env:
    """A copy of the script in a temp root, its own .env.example, and a fake Keychain."""

    def __init__(self, tmp_path: Path, keychain: bool = True):
        self.root = tmp_path / "root"
        (self.root / "scripts").mkdir(parents=True)
        self.script = self.root / "scripts" / "keychain-env"
        shutil.copy2(SCRIPT, self.script)
        self.env_example("OPENAI_API_KEY=\nANTHROPIC_API_KEY=\n")
        self.store = tmp_path / "kc" / "items"
        self.store.mkdir(parents=True)
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found, f"{tool} is needed by keychain-env"
            (self.bin / tool).symlink_to(found)
        if keychain:
            fake = self.bin / "security"
            fake.write_text(FAKE_SECURITY.format(python=sys.executable))
            fake.chmod(0o755)
        self.cwd = tmp_path / "cwd"
        self.cwd.mkdir()
        self.values: list[str] = []

    def env_example(self, text: str) -> None:
        (self.root / ".env.example").write_text(text)

    def item(self, var: str, value: str | bytes) -> None:
        raw = value.encode() if isinstance(value, str) else value
        (self.store / f"{PREFIX}{var}").write_bytes(raw)
        if value:
            self.values.append(raw.decode())

    def run(self, *args: str, **extra: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": str(self.bin),
            "HOME": str(self.cwd),
            "USER": "tester",
            "FAKE_KEYCHAIN_DIR": str(self.store),
            **extra,
        }
        return subprocess.run(
            [BASH, str(self.script), *args],
            cwd=self.cwd, env=env, capture_output=True, timeout=30,
        )

    def run_env(self, **extra: str) -> dict[str, str]:
        """Run `env -0` through `keychain-env run`; the resulting environment."""
        res = self.run("run", "env", "-0", **extra)
        assert res.returncode == 0, res.stderr
        pairs = [p.decode() for p in res.stdout.split(b"\0") if p]
        return dict(p.split("=", 1) for p in pairs)

    def assert_no_values(self, res: subprocess.CompletedProcess) -> None:
        output = res.stdout.decode(errors="replace") + res.stderr.decode(errors="replace")
        for value in self.values:
            assert value.strip() not in output
        for value in self.values:
            assert value.strip().encode().hex() not in output.lower()


@pytest.fixture
def kc(tmp_path):
    return Env(tmp_path)


def listing(res: subprocess.CompletedProcess) -> dict[str, str]:
    rows = [line.split() for line in res.stdout.decode().splitlines() if line.strip()]
    return {name: state for state, name in rows}


# ---- run ----------------------------------------------------------------------


def test_api_keychain_run_exports_stored_key(kc):
    kc.item("OPENAI_API_KEY", "sk-fake-openai")
    env = kc.run_env()
    assert env["OPENAI_API_KEY"] == "sk-fake-openai"
    assert env["TRADINGAGENTS_KEYCHAIN_EXPORTED"] == "OPENAI_API_KEY"


def test_api_keychain_run_leaves_missing_key_unset(kc):
    kc.item("OPENAI_API_KEY", "sk-fake-openai")
    env = kc.run_env()
    # Not even an empty export: python-dotenv would treat that as authoritative.
    assert "ANTHROPIC_API_KEY" not in env


def test_api_keychain_run_already_set_variable_wins(kc):
    kc.item("OPENAI_API_KEY", "sk-from-keychain")
    env = kc.run_env(OPENAI_API_KEY="sk-from-shell")
    assert env["OPENAI_API_KEY"] == "sk-from-shell"
    assert "OPENAI_API_KEY" not in env["TRADINGAGENTS_KEYCHAIN_EXPORTED"].split()


def test_api_keychain_empty_item_not_exported_and_listed_empty(kc):
    kc.item("OPENAI_API_KEY", "")
    kc.item("ANTHROPIC_API_KEY", "sk-fake-anthropic")
    env = kc.run_env()
    assert "OPENAI_API_KEY" not in env
    res = kc.run("list")
    assert res.returncode == 0, res.stderr
    rows = listing(res)
    assert rows["OPENAI_API_KEY"] == "empty"
    assert rows["ANTHROPIC_API_KEY"] == "stored"
    assert rows["AZURE_OPENAI_API_KEY"] == "-"
    kc.assert_no_values(res)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (b"trail\n", "trail"),
        (b"crlf\r\n", "crlf"),
        ("héllo".encode(), "héllo"),
        (b'quo"te', 'quo"te'),
        # A real key can be pure lowercase hex (FRED): it must not be decoded.
        (b"0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef"),
    ],
)
def test_api_keychain_run_decodes_values(kc, stored, expected):
    kc.item("OPENAI_API_KEY", stored)
    env = kc.run_env()
    assert env["OPENAI_API_KEY"] == expected


def test_api_keychain_value_never_in_security_argv(kc):
    kc.item("OPENAI_API_KEY", "sk-argv-check")
    kc.run_env()
    kc.run("list")
    log = (kc.store.parent / "security.log").read_text()
    assert "sk-argv-check" not in log


def test_api_keychain_run_without_security_just_runs(tmp_path):
    kc = Env(tmp_path, keychain=False)
    kc.item("OPENAI_API_KEY", "sk-unreachable")
    env = kc.run_env()
    assert "OPENAI_API_KEY" not in env
    assert "TRADINGAGENTS_KEYCHAIN_EXPORTED" not in env


# ---- compose-override ------------------------------------------------------------


def test_api_keychain_compose_override_lists_only_keychain_exports(kc):
    kc.item("OPENAI_API_KEY", "sk-fake-openai")
    # Set in the shell (say ~/.zshrc), not from the Keychain: .env must still
    # win inside the container, so the override must not name it.
    res = kc.run(
        "run", BASH, str(kc.script), "compose-override", "api",
        ANTHROPIC_API_KEY="sk-shell-anthropic",
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.decode()
    assert out == "services:\n  api:\n    environment:\n      - OPENAI_API_KEY\n"
    kc.values.append("sk-shell-anthropic")
    kc.assert_no_values(res)


def test_api_keychain_compose_override_empty_without_exports(kc):
    res = kc.run("compose-override", "api", OPENAI_API_KEY="sk-shell-openai")
    assert res.returncode == 0, res.stderr
    assert res.stdout.decode() == "services:\n  api: {}\n"
    kc.values.append("sk-shell-openai")
    kc.assert_no_values(res)


# ---- key names ----------------------------------------------------------------


def test_api_keychain_vars_rejects_bad_names_and_does_not_glob(kc):
    # A file in the working directory that `*` would glob to, and a matching
    # item: with globbing on, `*` would turn into a valid name and be exported.
    (kc.cwd / "GLOBBED_KEY").write_text("")
    kc.item("GLOBBED_KEY", "sk-glob")
    kc.item("EXTRA_KEY", "sk-extra")
    extra = {"TRADINGAGENTS_KEYCHAIN_VARS": "EXTRA_KEY bad-name *"}
    res = kc.run("run", "env", "-0", **extra)
    assert res.returncode == 0, res.stderr
    names = {p.split(b"=", 1)[0].decode() for p in res.stdout.split(b"\0") if p}
    assert "EXTRA_KEY" in names
    assert "GLOBBED_KEY" not in names
    err = res.stderr.decode()
    assert "'bad-name'" in err
    assert "'*'" in err
    rows = listing(kc.run("list", **extra))
    assert "EXTRA_KEY" in rows
    assert "GLOBBED_KEY" not in rows
    assert "bad-name" not in rows


def test_api_keychain_names_from_env_example_variants(kc):
    kc.env_example(
        "export FOO_KEY=\n"
        "  #BAR_TOKEN=\n"
        "##BAZ_SECRET=\n"
        "# QUX_API_KEY=x\n"
        "PLAIN_API_KEY=\n"
        "TRADINGAGENTS_API_KEY=\n"
        "LOG_LEVEL=\n"
    )
    res = kc.run("list")
    assert res.returncode == 0, res.stderr
    rows = listing(res)
    for name in ("FOO_KEY", "BAR_TOKEN", "BAZ_SECRET", "QUX_API_KEY", "PLAIN_API_KEY"):
        assert name in rows
    assert "TRADINGAGENTS_API_KEY" not in rows
    assert "LOG_LEVEL" not in rows


def test_api_keychain_remove_requires_security(tmp_path):
    kc = Env(tmp_path, keychain=False)
    res = kc.run("remove", "OPENAI_API_KEY")
    assert res.returncode == 1
    assert b"'security' tool is not available" in res.stderr


# ---- tasks.py ---------------------------------------------------------------------


def test_api_keychain_with_keys_absolute_quoted_path(monkeypatch):
    cmd = tasks.with_keys("python -m cli.main")
    argv = shlex.split(cmd)
    assert Path(argv[0]).is_absolute()
    assert Path(argv[0]) == SCRIPT
    assert argv[1:] == ["run", "python", "-m", "cli.main"]

    monkeypatch.setattr(tasks, "KEYCHAIN_ENV", Path("/tmp/dir with space/keychain-env"))
    argv = shlex.split(tasks.with_keys("true"))
    assert argv == ["/tmp/dir with space/keychain-env", "run", "true"]


def test_api_keychain_with_keys_windows_passthrough(monkeypatch):
    monkeypatch.setattr(tasks.os, "name", "nt")
    assert tasks.with_keys("python -m cli.main") == "python -m cli.main"


def test_api_keychain_analyze_quotes_ticker_and_date():
    c = MagicMock()
    tasks.analyze.body(c, ticker="AAPL; touch /tmp/pwned", date="2024-01-02 $(id)")
    (cmd,), _ = c.run.call_args
    argv = shlex.split(cmd)
    assert argv[-4:] == ["analyze", "AAPL; touch /tmp/pwned", "--date", "2024-01-02 $(id)"]
