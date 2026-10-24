"""Invoke tasks for TradingAgents.

Cross-platform task runner replacing Makefile.

Usage:
    pip install invoke          # install invoke (once)
    invoke --list               # show all tasks
    invoke install-api          # install API dependencies
    invoke api                  # start API server
    invoke api -r               # start API server with auto-reload
    invoke test                 # run test suite
    invoke test-unit            # run unit tests only
    invoke test-api             # run API tests only
    invoke docker               # run with Docker
    invoke env                  # copy .env.example to .env
"""

import os
import re
import shlex
import sys
from pathlib import Path

from invoke.exceptions import Exit
from invoke.tasks import task

# Absolute, so `invoke` works from any subdirectory of the repository.
KEYCHAIN_ENV = Path(__file__).resolve().parent / "scripts" / "keychain_env.py"
TICKER_RE = re.compile(r"^[A-Za-z0-9.^=\-]{1,32}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


@task
def install(c, api=False, dev=False, bedrock=False):
    """Install the package and optional dependencies."""
    extras = []
    if api or (not dev and not bedrock):
        # default: base package only
        pass
    if dev:
        extras.append("dev")
    if bedrock:
        extras.append("bedrock")
    extra_str = f"[{','.join(extras)}]" if extras else ""
    c.run(f'pip install -e ".{extra_str}"')


@task
def install_api(c):
    """Install API dependencies (fastapi + uvicorn)."""
    c.run('pip install -e ".[api]"')


@task
def install_dev(c):
    """Install all dependencies (api + dev + bedrock)."""
    c.run('pip install -e ".[api,dev,bedrock]"')


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


@task(help={"enterprise": "Create .env.enterprise instead of .env"})
def env(c, enterprise=False):
    """Copy .env example file for configuration."""
    if enterprise:
        src, dst = ".env.enterprise.example", ".env.enterprise"
    else:
        src, dst = ".env.example", ".env"
    if os.path.exists(dst):
        print(f"{dst} already exists, skipping.")
        return
    if os.name == "nt":
        c.run(f"copy {src} {dst}")
    else:
        c.run(f"cp {src} {dst}")
    print(f"Created {dst}")


def quote(arg):
    """Quote one argument for the shell invoke runs: cmd.exe on Windows.

    On Windows every argument is wrapped in double quotes, so cmd.exe takes
    ``^ & | < >`` literally; a ``"`` inside would end the quoting, so it is
    refused (it cannot occur in a Windows path). cmd.exe still expands
    ``%VAR%`` inside quotes: callers validate user input (see ``analyze``).
    """
    s = str(arg)
    if os.name == "nt":
        if '"' in s:
            raise Exit(f"cannot quote {s!r} for cmd.exe: it contains a double quote", code=2)
        # Backslashes are literal except before the closing quote: double those.
        trailing = s[len(s.rstrip("\\")) :]
        return '"' + s + trailing + '"'
    return shlex.quote(s)


def keychain_cmd(*args):
    """``python scripts/keychain_env.py <args>`` with this interpreter, quoted."""
    return " ".join([quote(sys.executable), quote(KEYCHAIN_ENV), *args])


def with_keys(cmd):
    """Prefix ``cmd`` so API keys stored in the OS keyring reach it.

    See scripts/keychain_env.py: only stored keys are exported, .env supplies
    the rest; without ``keyring`` the command runs as is. Same on every OS.
    """
    return f"{keychain_cmd('run', '--')} {cmd}"


# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------


@task(
    help={
        "host": "Host to bind (default: 0.0.0.0)",
        "port": "Port to bind (default: 8000)",
        "reload": "Enable auto-reload for development",
    }
)
def api(c, host="0.0.0.0", port=8000, reload=False, r=False):
    """Start the TradingAgents API server."""
    reload_flag = "--reload" if (reload or r) else ""
    python = quote(sys.executable)
    cmd = f"{python} run_server.py --host {quote(host)} --port {quote(port)} {reload_flag}".strip()
    print(f"Starting API server: {host}:{port}")
    print(f"Docs: http://localhost:{port}/api/v1/docs")
    c.run(with_keys(cmd))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@task
def cli(c):
    """Launch the interactive CLI."""
    c.run(with_keys(f"{quote(sys.executable)} -m cli.main"))


@task(
    help={
        "ticker": "Ticker symbol to analyze",
        "date": "Analysis date (YYYY-MM-DD)",
        "checkpoint": "Enable checkpoint resume",
        "clear_checkpoints": "Clear all checkpoints before running",
    }
)
def analyze(c, ticker=None, date=None, checkpoint=False, clear_checkpoints=False):
    """Run trading analysis via CLI."""
    if ticker and not TICKER_RE.match(ticker):
        raise Exit(
            f"invalid ticker {ticker!r}: use 1-32 of A-Z a-z 0-9 . ^ = -", code=2
        )
    if date and not DATE_RE.match(date):
        raise Exit(f"invalid date {date!r}: use YYYY-MM-DD", code=2)
    flags = ""
    if checkpoint:
        flags += " --checkpoint"
    if clear_checkpoints:
        flags += " --clear-checkpoints"
    if ticker:
        flags += f" {quote(ticker)}"
    if date:
        flags += f" --date {quote(date)}"
    cmd = f"tradingagents analyze{flags}"
    print(f"Running: {cmd}")
    c.run(with_keys(cmd.strip()))


def check_var(var):
    if not VAR_RE.match(var or ""):
        raise Exit(f"not a variable name: {var!r} (e.g. ANTHROPIC_API_KEY)", code=2)
    return var


@task(help={"var": "Variable name, e.g. ANTHROPIC_API_KEY"})
def keychain_add(c, var):
    """Store an API key in the OS keyring (hidden prompt)."""
    c.run(keychain_cmd("add", quote(check_var(var))), pty=os.name != "nt")


@task(help={"var": "Variable name, e.g. ANTHROPIC_API_KEY"})
def keychain_remove(c, var):
    """Delete an API key stored in the OS keyring."""
    c.run(keychain_cmd("remove", quote(check_var(var))))


@task
def keychain_list(c):
    """Show which API keys are stored in the OS keyring (no values)."""
    c.run(keychain_cmd("list"))


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


@task(help={"ollama": "Use Ollama profile for local models"})
def docker(c, ollama=False):
    """Run with Docker, passing in the API keys stored in the OS keyring.

    Goes through ``keychain_env.py compose``: a temporary override (key names
    only, always deleted) hands the keyring keys to the service; .env supplies
    the rest, as without a keyring.
    """
    profile = ["--profile", "ollama"] if ollama else []
    target = "tradingagents-ollama" if ollama else "tradingagents"
    c.run(keychain_cmd("compose", "--service", target, "--", *profile, "run", "--rm", target))


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------


@task(help={"mark": "Run only tests with specific marker (unit, integration, smoke)"})
def test(c, mark=None, v=True):
    """Run test suite."""
    cmd = "python -m pytest tests/"
    if v:
        cmd += " -v"
    if mark:
        cmd += f" -m {mark}"
    c.run(cmd)


@task
def test_unit(c):
    """Run unit tests only."""
    c.run("python -m pytest tests/ -v -m unit")


@task
def test_integration(c):
    """Run integration tests only."""
    c.run("python -m pytest tests/ -v -m integration")


@task
def test_smoke(c):
    """Run smoke tests only."""
    c.run("python -m pytest tests/ -v -m smoke")


@task
def test_api(c):
    """Run API tests only."""
    c.run("python -m pytest tests/ -v -k test_api")


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


@task
def docs(c):
    """Open API documentation in browser."""
    print("Start the API server first: invoke api")
    print("Then open: http://localhost:8000/api/v1/docs")
    if os.name == "nt":
        c.run(
            'python -c "import webbrowser; webbrowser.open(\'http://localhost:8000/api/v1/docs\')"'
        )
    else:
        c.run("python -c \"import webbrowser; webbrowser.open('http://localhost:8000/api/v1/docs')\"")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@task
def clean(c):
    """Remove Python cache files and test artifacts."""
    patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".mypy_cache",
        "*.egg-info",
    ]
    for pattern in patterns:
        if os.name == "nt":
            c.run(f'for /r %i in ({pattern}) do @if exist "%i" echo Removing "{pattern}"', warn=True)
        else:
            c.run(f"find . -type d -name {pattern} -exec rm -rf {{}} + 2>/dev/null || true", warn=True)
            c.run(f"find . -type f -name {pattern} -delete 2>/dev/null || true", warn=True)
    print("Cleaned cache files.")


@task
def lint(c):
    """Run ruff linter."""
    c.run("ruff check .")


@task
def format_code(c):
    """Run ruff formatter."""
    c.run("ruff format .")
