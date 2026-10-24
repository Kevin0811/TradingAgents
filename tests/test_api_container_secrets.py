"""Tests for scripts/container_secrets.py, the entrypoint of Dockerfile.api.

The script runs in-process against a temp secrets directory
($TRADINGAGENTS_SECRETS_DIR), with ``os.execvp`` replaced to capture the command
and the environment it would run with.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "container_secrets.py"
DIR_VAR = "TRADINGAGENTS_SECRETS_DIR"
VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OTHER_KEY", "BIG_KEY", "BAD_UTF8_KEY")


def load_script():
    spec = importlib.util.spec_from_file_location("container_secrets_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Execd(BaseException):
    """Raised by the fake os.execvp in place of replacing the process."""

    def __init__(self, file, argv):
        super().__init__(file)
        self.file, self.argv, self.env = file, list(argv), dict(os.environ)


class Harness:
    def __init__(self, tmp_path, monkeypatch, capsys):
        self.capsys = capsys
        self.mod = load_script()
        self.dir = tmp_path / "secrets"
        self.dir.mkdir()
        monkeypatch.setenv(DIR_VAR, str(self.dir))
        for name in VARS:
            monkeypatch.delenv(name, raising=False)
        self.values: list[str] = []

        def execvp(file, argv):
            raise Execd(file, argv)

        monkeypatch.setattr(os, "execvp", execvp)

    def put(self, name, content, secret=True):
        path = self.dir / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_bytes(content.encode("utf-8"))
            if secret and content.strip():
                self.values.append(content.strip())
        return path

    def main(self, *argv):
        child, code = None, None
        try:
            self.mod.main(list(argv))
        except Execd as e:
            child = e
        except SystemExit as e:
            code = e.code
        cap = self.capsys.readouterr()
        for value in self.values:
            assert value not in cap.out
            assert value not in cap.err
        return child, code, cap


@pytest.fixture
def h(tmp_path, monkeypatch, capsys):
    return Harness(tmp_path, monkeypatch, capsys)


def test_api_secrets_valid_file_sets_env_and_overrides(h, monkeypatch):
    # env_file (.env) already set it: the keyring value in the secret wins.
    monkeypatch.setenv("OPENAI_API_KEY", "from-dotenv")
    h.put("OPENAI_API_KEY", "sk-from-keyring")
    h.put("ANTHROPIC_API_KEY", "sk-ant-keyring")
    child, code, cap = h.main("python", "run_server.py")
    assert code is None
    assert child.env["OPENAI_API_KEY"] == "sk-from-keyring"
    assert child.env["ANTHROPIC_API_KEY"] == "sk-ant-keyring"
    assert cap.out == cap.err == ""


@pytest.mark.parametrize("content", ["sk-line\n", "sk-line\r\n", "sk-line\n\n\r\n"])
def test_api_secrets_trailing_newline_stripped(h, content):
    h.put("OPENAI_API_KEY", content)
    child, _, _ = h.main("true")
    assert child.env["OPENAI_API_KEY"] == "sk-line"


def test_api_secrets_inner_and_leading_whitespace_kept(h):
    h.put("OPENAI_API_KEY", " a b\n")
    child, _, _ = h.main("true")
    assert child.env["OPENAI_API_KEY"] == " a b"


@pytest.mark.parametrize("content", ["", "\n", "   \n", "\t\r\n"])
def test_api_secrets_empty_or_whitespace_file_skipped(h, monkeypatch, content):
    monkeypatch.setenv("OPENAI_API_KEY", "from-dotenv")
    h.put("OPENAI_API_KEY", content)
    child, _, _ = h.main("true")
    # The env_file value is left alone.
    assert child.env["OPENAI_API_KEY"] == "from-dotenv"


@pytest.mark.parametrize("name", ["openai_api_key", "1KEY", "BAD-NAME", "_KEY", ".hidden"])
def test_api_secrets_bad_name_skipped_with_note(h, name):
    h.put(name, "sk-bad-name-value")
    child, _, cap = h.main("true")
    assert name not in child.env
    assert cap.err.count("\n") == 1
    assert "skipping" in cap.err and name in cap.err
    assert "not a variable name" in cap.err


def test_api_secrets_oversized_file_skipped(h, monkeypatch):
    monkeypatch.setenv("BIG_KEY", "from-dotenv")
    h.put("BIG_KEY", "x" * (64 * 1024 + 1), secret=False)
    h.put("OTHER_KEY", "sk-other")
    child, _, cap = h.main("true")
    assert child.env["BIG_KEY"] == "from-dotenv"
    assert child.env["OTHER_KEY"] == "sk-other"
    assert "skipping BIG_KEY: larger than 65536 bytes" in cap.err
    assert "x" * 16 not in cap.err


def test_api_secrets_exactly_64k_is_loaded(h):
    h.put("BIG_KEY", "y" * (64 * 1024), secret=False)
    child, _, cap = h.main("true")
    assert child.env["BIG_KEY"] == "y" * (64 * 1024)
    assert cap.err == ""


def test_api_secrets_undecodable_file_skipped(h):
    h.put("BAD_UTF8_KEY", b"sk-\xff\xfe-raw")
    child, _, cap = h.main("true")
    assert "BAD_UTF8_KEY" not in child.env
    assert "skipping BAD_UTF8_KEY: not valid UTF-8" in cap.err
    assert "sk-" not in cap.err


def test_api_secrets_nul_byte_skipped(h):
    h.put("OTHER_KEY", "sk-a\0b")
    child, _, cap = h.main("true")
    assert "OTHER_KEY" not in child.env
    assert "skipping OTHER_KEY: contains a NUL byte" in cap.err


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_api_secrets_unreadable_file_skipped(h):
    path = h.put("OTHER_KEY", "sk-unreadable")
    path.chmod(0)
    try:
        child, _, cap = h.main("true")
    finally:
        path.chmod(0o600)
    assert "OTHER_KEY" not in child.env
    assert "skipping OTHER_KEY: cannot read it (PermissionError)" in cap.err


def test_api_secrets_directory_entry_skipped(h):
    (h.dir / "OTHER_KEY").mkdir()
    child, _, cap = h.main("true")
    assert "OTHER_KEY" not in child.env
    assert "skipping OTHER_KEY: not a regular file" in cap.err


def test_api_secrets_missing_dir_just_execs(h, monkeypatch, tmp_path):
    monkeypatch.setenv(DIR_VAR, str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("OPENAI_API_KEY", "from-dotenv")
    child, code, cap = h.main("python", "run_server.py", "--port", "8000")
    assert code is None
    assert child.argv == ["python", "run_server.py", "--port", "8000"]
    assert child.env["OPENAI_API_KEY"] == "from-dotenv"
    assert cap.out == cap.err == ""


def test_api_secrets_default_dir_is_run_secrets(h, monkeypatch):
    seen = []
    monkeypatch.delenv(DIR_VAR)
    monkeypatch.setattr(h.mod, "load_secrets", seen.append)
    h.main("true")
    assert seen == ["/run/secrets"]


def test_api_secrets_values_never_printed(h):
    h.put("OPENAI_API_KEY", "sk-good-value\n")
    h.put("lower_case", "sk-lower-value")
    h.put("BAD_UTF8_KEY", b"sk-raw-\xff")
    h.put("BIG_KEY", "sk-big-" + "z" * 70000)
    # Harness.main asserts no recorded value reaches stdout or stderr.
    child, _, cap = h.main("true")
    assert child.env["OPENAI_API_KEY"] == "sk-good-value"
    assert "sk-" not in cap.out + cap.err
    assert cap.err.count("\n") == 3


def test_api_secrets_argv_passed_through_unchanged(h):
    argv = ["python", "-c", "print('a b')", "--flag=1", "", "$HOME", "*"]
    h.put("OPENAI_API_KEY", "sk-x")
    child, _, _ = h.main(*argv)
    assert child.file == "python"
    assert child.argv == argv


def test_api_secrets_no_args_exits_2(h):
    child, code, cap = h.main()
    assert child is None
    assert code == 2
    assert cap.err.startswith("usage: container_secrets.py")


def test_api_secrets_command_not_found_exits_127(h, monkeypatch):
    def missing(file, argv):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(os, "execvp", missing)
    _, code, cap = h.main("no-such-command-xyz")
    assert code == 127
    assert "command not found: no-such-command-xyz" in cap.err


@pytest.mark.skipif(os.name == "nt", reason="exec semantics are POSIX")
def test_api_secrets_real_exec(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "OPENAI_API_KEY").write_text("sk-real-exec\n")
    env = {k: v for k, v in os.environ.items() if "KEY" not in k}
    env.update({DIR_VAR: str(secrets), "OPENAI_API_KEY": "from-dotenv"})
    res = subprocess.run(
        [sys.executable, str(SCRIPT), sys.executable, "-c",
         "import os,sys; print(os.environ['OPENAI_API_KEY']=='sk-real-exec'); sys.exit(7)"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 7
    assert res.stdout == "True\n"
    assert "sk-real-exec" not in res.stderr


def test_api_secrets_script_is_stdlib_only():
    tree = ast.parse(SCRIPT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "tradingagents" not in imported
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def test_api_secrets_dockerfile_uses_entrypoint():
    text = (REPO / "Dockerfile.api").read_text()
    assert 'ENTRYPOINT ["python", "/home/appuser/app/scripts/container_secrets.py"]' in text
    assert 'CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]' in text
    dockerignore = (REPO / ".dockerignore").read_text().split()
    assert "scripts" not in dockerignore and "scripts/" not in dockerignore
