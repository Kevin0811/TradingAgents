"""Tests for the GET /config endpoint.

Covers:
- Returns the currently-effective config values a caller can override per-request
- Reflects ApiConfig overrides (e.g. an env-var-driven server default)
- Never leaks server-internal paths or anything secret-shaped
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tradingagents.api.app import create_app


def test_get_config_reflects_effective_settings():
    app = create_app(
        overrides={"max_debate_rounds": 3, "deep_think_llm": "test-model"}
    )
    client = TestClient(app)

    resp = client.get("/api/v1/config")

    assert resp.status_code == 200
    body = resp.json()
    assert body["max_debate_rounds"] == 3
    assert body["deep_think_llm"] == "test-model"
    assert "task_max_concurrent" in body


def test_get_config_excludes_internal_paths_and_secrets():
    app = create_app()
    client = TestClient(app)

    body = client.get("/api/v1/config").json()

    for leaked_key in (
        "project_dir",
        "results_dir",
        "data_cache_dir",
        "memory_log_path",
        "api_key",
        "openai_api_key",
    ):
        assert leaked_key not in body
