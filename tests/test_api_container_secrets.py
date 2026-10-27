"""Tests for scripts/container_secrets.py, the entrypoint of Dockerfile.api.

The script runs in-process against a temp secrets directory
($TRADINGAGENTS_SECRETS_DIR), with ``os.execvp`` replaced to capture the command
and the environment it would run with.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "container_secrets.py"
DIR_VAR = "TRADINGAGENTS_SECRETS_DIR"
VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OTHER_KEY", "BIG_KEY",
        "BAD_UTF8_KEY")


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
    h.put("OTHER_KEY", "sk-other-keyring")
    child, code, cap = h.main("python", "run_server.py")
    assert code is None
    assert child.env["OPENAI_API_KEY"] == "sk-from-keyring"
    assert child.env["OTHER_KEY"] == "sk-other-keyring"
    assert cap.out == cap.err == ""


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "GOOGLE_API_KEY"])
def test_api_secrets_on_demand_keys_not_loaded(h, name):
    # The API reads these from their file when it builds a client
    # (tradingagents/api/infrastructure/secret_keys.py): never in the environment.
    h.put(name, "sk-on-demand-value")
    h.put("OPENAI_API_KEY", "sk-openai-keyring")
    child, code, cap = h.main("python", "run_server.py")
    assert code is None
    assert name not in child.env
    assert "sk-on-demand-value" not in child.env.values()
    assert child.env["OPENAI_API_KEY"] == "sk-openai-keyring"
    assert cap.out == cap.err == ""


def test_api_secrets_on_demand_key_leaves_env_file_value_alone(h, monkeypatch):
    # A .env value is neither overwritten nor removed; the API prefers the file.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-dotenv")
    h.put("ANTHROPIC_API_KEY", "sk-ant-keyring")
    child, _, _ = h.main("true")
    assert child.env["ANTHROPIC_API_KEY"] == "from-dotenv"


def test_api_secrets_on_demand_names_match_the_api():
    # The script cannot import tradingagents, so it keeps its own copy.
    from tradingagents.api.infrastructure.secret_keys import ON_DEMAND_KEY_NAMES

    assert load_script().ON_DEMAND_NAMES == ON_DEMAND_KEY_NAMES


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
    (secrets / "ANTHROPIC_API_KEY").write_text("sk-ant-real-exec\n")
    env = {k: v for k, v in os.environ.items() if "KEY" not in k}
    env.update({DIR_VAR: str(secrets), "OPENAI_API_KEY": "from-dotenv"})
    res = subprocess.run(
        [sys.executable, str(SCRIPT), sys.executable, "-c",
         "import os,sys; print(os.environ['OPENAI_API_KEY']=='sk-real-exec',"
         " 'ANTHROPIC_API_KEY' in os.environ); sys.exit(7)"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 7
    assert res.stdout == "True False\n"
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


# ---- symlinks and special files (never followed or read) ----------------------


@pytest.mark.skipif(os.name == "nt", reason="symlinks need POSIX")
def test_api_secrets_symlink_to_outside_file_skipped(h, tmp_path, monkeypatch):
    outside = tmp_path / "outside.txt"
    outside.write_text("dummy-outside-value\n")
    h.values.append("dummy-outside-value")
    monkeypatch.setenv("OPENAI_API_KEY", "from-dotenv")
    (h.dir / "OPENAI_API_KEY").symlink_to(outside)
    h.put("OTHER_KEY", "dummy-regular")
    child, _, cap = h.main("true")
    assert child.env["OPENAI_API_KEY"] == "from-dotenv"
    assert child.env["OTHER_KEY"] == "dummy-regular"
    assert cap.err == "container-secrets: skipping OPENAI_API_KEY: not a regular file\n"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_api_secrets_symlink_to_fifo_skipped(h, tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    (h.dir / "OTHER_KEY").symlink_to(fifo)
    child, _, cap = h.main("true")  # would block forever if the FIFO were opened
    assert "OTHER_KEY" not in child.env
    assert "skipping OTHER_KEY: not a regular file" in cap.err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_api_secrets_fifo_swapped_in_is_not_read(tmp_path):
    # A FIFO that replaces the file between the scan and the open: O_NONBLOCK
    # keeps the open from blocking and fstat rejects it. In a subprocess, so a
    # regression fails on the timeout instead of hanging the test run.
    fifo = tmp_path / "OTHER_KEY"
    os.mkfifo(fifo)
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('cs', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(m.read_secret(sys.argv[1], 'OTHER_KEY'))\n"
    )
    res = subprocess.run([sys.executable, "-c", code, str(fifo)],
                         capture_output=True, text=True, timeout=15)
    assert res.stdout == "None\n"
    assert "skipping OTHER_KEY: not a regular file" in res.stderr


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="needs O_NOFOLLOW")
def test_api_secrets_symlink_swapped_in_is_not_followed(h, tmp_path):
    # A symlink that replaces the file after the scan: the open itself refuses it.
    outside = tmp_path / "outside.txt"
    outside.write_text("dummy-outside-value")
    (h.dir / "OTHER_KEY").symlink_to(outside)
    assert h.mod.read_secret(str(h.dir / "OTHER_KEY"), "OTHER_KEY") is None
    err = h.capsys.readouterr().err
    assert "skipping OTHER_KEY: cannot read it" in err
    assert "dummy-outside-value" not in err


# ---- name policy ---------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "PATH", "HOME", "SHELL", "USER", "PWD", "IFS", "PYTHONPATH", "PYTHONSTARTUP",
    "SSLKEYLOGFILE",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "TRADINGAGENTS_SECRETS_DIR",
    # Reserved even though they look like secrets:
    "PYTHON_KEYRING_BACKEND", "LD_SECRET_KEY", "SSL_CERT_PASSWORD",
])
def test_api_secrets_reserved_name_rejected(h, monkeypatch, name):
    before = os.environ.get(name)
    h.put(name, "dummy-reserved-value")
    child, _, cap = h.main("true")
    assert child.env.get(name) == before
    assert cap.err == f"container-secrets: skipping {name!r}: reserved variable name\n"


@pytest.mark.parametrize("name", ["FOO", "DEBUG", "OPENAI_BASE_URL", "TRADINGAGENTS_LLM_PROVIDER"])
def test_api_secrets_non_secret_name_rejected(h, monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    h.put(name, "dummy-plain-value")
    child, _, cap = h.main("true")
    assert name not in child.env
    assert cap.err.count("\n") == 1
    assert f"skipping {name!r}: not a secret name" in cap.err


@pytest.mark.parametrize("name", ["DB_PASSWORD", "SMTP_PASS", "CLIENT_SECRET", "GH_TOKEN"])
def test_api_secrets_secret_markers_accepted(h, monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    h.put(name, "dummy-accepted")
    child, _, cap = h.main("true")
    assert child.env[name] == "dummy-accepted"
    assert cap.err == ""


def test_api_secrets_every_keychain_key_name_is_accepted(monkeypatch):
    # keychain_env.py turns these names (from the repo's .env.example) into
    # secrets; container_secrets.py must load every one of them.
    spec = importlib.util.spec_from_file_location(
        "keychain_env_for_names", REPO / "scripts" / "keychain_env.py")
    keychain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(keychain)
    monkeypatch.delenv("TRADINGAGENTS_KEYCHAIN_VARS", raising=False)
    names = keychain.key_vars()
    assert {"OPENAI_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "OPENAI_COMPATIBLE_API_KEY"} <= set(names)
    mod = load_script()
    assert {n: mod.name_problem(n) for n in names} == dict.fromkeys(names)


# ---- Makefile ---------------------------------------------------------------------

API_TARGETS = ("docker-run-api", "docker-run-api-detached")


def makefile_recipes(text):
    """{target: [recipe lines]} for plain `target: ...` rules of a Makefile."""
    recipes, current = {}, None
    for line in text.splitlines():
        if line.startswith("\t") and current:
            recipes[current].append(line.strip())
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", line)
        current = m.group(1) if m else None
        if current:
            recipes[current] = []
    return recipes


@pytest.mark.parametrize("target", API_TARGETS)
def test_api_secrets_makefile_api_targets_force_recreate(target):
    # Compose does not hash secret values: without --force-recreate, `up` keeps
    # the existing container and its old key after a keyring rotation.
    recipe = makefile_recipes((REPO / "Makefile").read_text())[target]
    assert recipe, target
    line = " ".join(recipe)
    assert "--secrets" in line
    assert re.search(r"\bup\b.*\s--force-recreate\b", line), line


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_api_secrets_make_dry_run_force_recreate():
    res = subprocess.run(
        ["make", "-n", "-C", str(REPO), *API_TARGETS],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr
    lines = [ln for ln in res.stdout.splitlines() if "compose" in ln]
    assert len(lines) == 2, res.stdout
    assert all(" up " in ln and "--force-recreate" in ln for ln in lines), lines
