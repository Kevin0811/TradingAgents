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
    invoke docker               # run with Docker
    invoke env                  # copy .env.example to .env
"""

from invoke import task
import os
import sys


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
    cmd = f"python run_server.py --host {host} --port {port} {reload_flag}".strip()
    print(f"Starting API server: {host}:{port}")
    print(f"Docs: http://localhost:{port}/api/v1/docs")
    c.run(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@task
def cli(c):
    """Launch the interactive CLI."""
    c.run("python -m cli.main")


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
    flags = ""
    if checkpoint:
        flags += " --checkpoint"
    if clear_checkpoints:
        flags += " --clear-checkpoints"
    if ticker:
        flags += f" {ticker}"
    if date:
        flags += f" --date {date}"
    cmd = f"tradingagents analyze{flags}"
    print(f"Running: {cmd}")
    c.run(cmd.strip())


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


@task(help={"ollama": "Use Ollama profile for local models"})
def docker(c, ollama=False):
    """Run with Docker."""
    profile = "--profile ollama" if ollama else ""
    target = "tradingagents-ollama" if ollama else "tradingagents"
    c.run(f"docker compose {profile} run --rm {target}")


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