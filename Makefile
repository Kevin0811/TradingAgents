# Makefile for TradingAgents
#
# This is an alternative, dependency-free entry point to the same tasks
# defined in tasks.py (invoke). Use whichever you prefer:
#   make <target>      # no extra Python packages required
#   invoke <task>       # cross-platform, requires `pip install -e ".[dev]"`
#
# Usage:
#   make help            # list all targets
#   make docker-build    # build the CLI image
#   make api             # run the API server locally
#   make cli             # launch the interactive CLI
#   make keychain-add VAR=ANTHROPIC_API_KEY   # keep a key in the OS keyring

# Windows (cmd.exe / Git Bash) ships `python`; elsewhere `python3` is the safe name.
ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= python3
endif
IMAGE ?= tradingagents
IMAGE_API ?= tradingagents-api
TAG ?= dev

# Targets that need API keys run through this: it exports the keys stored in
# the OS keyring (macOS Keychain, Windows Credential Manager, Linux Secret
# Service; see `make keychain-add`) and leaves the rest to .env. Without
# `keyring` or a usable backend it prints a hint and runs the command as is.
# Environment knobs (see scripts/keychain_env.py):
#   TRADINGAGENTS_KEYRING_SERVICE=name  keyring service (default tradingagents)
#   TRADINGAGENTS_KEYCHAIN_VARS="A B"   extra key names to look up
#   TRADINGAGENTS_KEYCHAIN_QUIET=1      no hint when falling back to .env only
KEYCHAIN = $(PYTHON) scripts/keychain_env.py
KEYCHAIN_ENV := $(KEYCHAIN) run --
# `docker compose` plus a temporary override (names only, deleted afterwards)
# passing into the service only the keys taken from the keyring; a key merely
# set in the shell still comes from .env inside the container, as before.
#   $(KEYCHAIN_COMPOSE) [--secrets] --service <svc> -- <docker compose args...>
# Where the values end up:
#   - API (docker-run-api*): with --secrets, as Compose secrets, i.e. files
#     /run/secrets/<VAR>. The API reads ANTHROPIC_API_KEY and GOOGLE_API_KEY
#     from their file whenever it builds an LLM client (never in its
#     environment); Dockerfile.api's entrypoint (scripts/container_secrets.py)
#     loads the others into the server's environment. They are not in the
#     container config, so `docker inspect` does not show them.
#   - CLI (docker-run, docker-run-ollama): as environment variables, since they
#     run upstream's Dockerfile, which has no such entrypoint; `docker inspect`
#     shows those values (Config.Env).
# Either way, anyone who can `docker exec` into the container can still read
# them (/run/secrets, the process environment): Docker access is root access.
# Secrets are not hidden everywhere either:
#   - Compose copies them into the container, so /run/secrets/* is in its
#     writable layer: `docker diff` lists the files and `docker export` /
#     `docker commit` include them, also while the container is stopped. They
#     are gone only when the container is removed (`docker compose down`).
#   - The files are mode 0444: any user inside the container can read them.
#   - While `docker compose` runs, its process environment on the host holds
#     the values, readable by the same user or root.
# Once a key is in the keyring, remove it from .env: env_file still copies the
# .env value into the container config (`docker inspect` shows it, whichever
# value wins). `compose` warns, naming the keys, when .env still sets one.
KEYCHAIN_COMPOSE := $(KEYCHAIN) compose

.PHONY: help \
	install install-api install-dev env \
	docker-build docker-build-api docker-run docker-run-ollama docker-run-api \
	docker-run-api-detached \
	api api-reload cli \
	keychain-add keychain-remove keychain-list \
	test test-unit test-integration test-smoke test-api \
	lint format clean

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

# `keyring` (for the keychain-* targets and KEYCHAIN_ENV) is installed here, not
# declared in pyproject.toml: that file comes from upstream, and leaving it
# alone keeps fork syncs conflict-free. Everything still works without it.
install: ## Install the base package
	pip install -e . keyring

install-api: ## Install with API extras (fastapi + uvicorn)
	pip install -e ".[api]" keyring

install-dev: ## Install with api + dev + bedrock extras
	pip install -e ".[api,dev,bedrock]" keyring

env: ## Copy .env.example to .env (skips if .env already exists)
	@test -f .env || cp .env.example .env

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the CLI/core Docker image
	docker build -t $(IMAGE):$(TAG) -f Dockerfile .

docker-build-api: ## Build the API server Docker image
	docker build -t $(IMAGE_API):$(TAG) -f Dockerfile.api .

docker-run: ## Run the CLI in Docker (interactive)
	$(KEYCHAIN_COMPOSE) --service tradingagents -- run --rm tradingagents

docker-run-ollama: ## Run the CLI in Docker with the Ollama profile
	$(KEYCHAIN_COMPOSE) --service tradingagents-ollama -- --profile ollama run --rm tradingagents-ollama

# --build: the image must contain the secrets entrypoint (an image built before
# it would ignore /run/secrets); unchanged layers come from the build cache.
# --force-recreate: Compose's config hash covers the secret's declaration, not
# the value it takes from the environment, so without it `up` leaves a running
# container as it is, and its /run/secrets and server keep the OLD key after
# the key is rotated in the keyring. (A stopped one gets the current value
# when `up` starts it again; recreating covers both.)
docker-run-api: ## Run the API server in Docker (foreground, port 8000; keys as secrets)
	$(KEYCHAIN_COMPOSE) --secrets --service tradingagents-api -- up --build --force-recreate tradingagents-api

# Compose copies the secret files into the container, so deleting the temporary
# override once `up -d` returns is fine: a plain `docker compose stop` / `start`
# (without the override) keeps the copied files and their values. A compose
# command that IS given the override (or an equivalent one), `start` included,
# copies the values again from its own environment: run it through
# $(KEYCHAIN_COMPOSE), or compose takes the key from .env (its interpolation
# file) or, if .env lacks it, fails to start the container. Stop it with
# `docker compose down`, which also deletes the copied secrets. A plain
# `docker compose up` (without this target) sees a changed config and can
# recreate the container without the keyring keys.
# --force-recreate: see docker-run-api.
docker-run-api-detached: ## Run the API server in Docker in the background (port 8000; keys as secrets)
	$(KEYCHAIN_COMPOSE) --secrets --service tradingagents-api -- up -d --build --force-recreate tradingagents-api

# ---------------------------------------------------------------------------
# API server / CLI
# ---------------------------------------------------------------------------

api: ## Start the API server locally (0.0.0.0:8000)
	$(KEYCHAIN_ENV) $(PYTHON) run_server.py --host 0.0.0.0 --port 8000

api-reload: ## Start the API server locally with auto-reload
	$(KEYCHAIN_ENV) $(PYTHON) run_server.py --host 0.0.0.0 --port 8000 --reload

cli: ## Launch the interactive CLI
	$(KEYCHAIN_ENV) $(PYTHON) -m cli.main

# ---------------------------------------------------------------------------
# API keys in the OS keyring (pip install keyring; see scripts/keychain_env.py)
# ---------------------------------------------------------------------------

keychain-add: ## Store an API key in the OS keyring (VAR=ANTHROPIC_API_KEY; hidden prompt)
	@test -n "$(VAR)" || { echo "usage: make keychain-add VAR=ANTHROPIC_API_KEY"; exit 2; }
	@$(KEYCHAIN) add "$(VAR)"

keychain-remove: ## Delete a stored API key (VAR=ANTHROPIC_API_KEY)
	@test -n "$(VAR)" || { echo "usage: make keychain-remove VAR=ANTHROPIC_API_KEY"; exit 2; }
	@$(KEYCHAIN) remove "$(VAR)"

keychain-list: ## Show which API keys are stored in the OS keyring (no values)
	@$(KEYCHAIN) list

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/ -v -m unit

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/ -v -m integration

test-smoke: ## Run smoke tests only
	$(PYTHON) -m pytest tests/ -v -m smoke

test-api: ## Run API tests only
	$(PYTHON) -m pytest tests/ -v -k test_api

# ---------------------------------------------------------------------------
# Lint / format / cleanup
# ---------------------------------------------------------------------------

lint: ## Run ruff linter
	ruff check .

format: ## Run ruff formatter
	ruff format .

clean: ## Remove Python caches and build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
