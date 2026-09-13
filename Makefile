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

PYTHON ?= python3
IMAGE ?= tradingagents
IMAGE_API ?= tradingagents-api
TAG ?= dev

.PHONY: help \
	install install-api install-dev env \
	docker-build docker-build-api docker-run docker-run-ollama docker-run-api \
	api api-reload cli \
	test test-unit test-integration test-smoke test-api \
	lint format clean

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install: ## Install the base package
	pip install -e .

install-api: ## Install with API extras (fastapi + uvicorn)
	pip install -e ".[api]"

install-dev: ## Install with api + dev + bedrock extras
	pip install -e ".[api,dev,bedrock]"

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
	docker compose run --rm tradingagents

docker-run-ollama: ## Run the CLI in Docker with the Ollama profile
	docker compose --profile ollama run --rm tradingagents-ollama

docker-run-api: ## Run the API server in Docker (foreground, port 8000)
	docker compose up tradingagents-api

# ---------------------------------------------------------------------------
# API server / CLI
# ---------------------------------------------------------------------------

api: ## Start the API server locally (0.0.0.0:8000)
	$(PYTHON) run_server.py --host 0.0.0.0 --port 8000

api-reload: ## Start the API server locally with auto-reload
	$(PYTHON) run_server.py --host 0.0.0.0 --port 8000 --reload

cli: ## Launch the interactive CLI
	$(PYTHON) -m cli.main

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
