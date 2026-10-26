"""Tests for tradingagents/api/infrastructure/secret_keys.py.

LLM API keys passed as Docker/Compose secrets are read from
$TRADINGAGENTS_SECRETS_DIR/<VAR> each time a client is built and given to it
as ``api_key=``; they are never put in os.environ. Every test runs against a
temp secrets directory.
"""

from __future__ import annotations

import logging
import os
import threading

import pytest

from tradingagents.api.infrastructure import secret_keys
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph
from tradingagents.llm_clients import factory

pytestmark = pytest.mark.unit

LOGGER = secret_keys.__name__


@pytest.fixture
def secrets(tmp_path, monkeypatch):
    """A temp secrets directory; the on-demand key env vars are unset."""
    directory = tmp_path / "secrets"
    directory.mkdir()
    monkeypatch.setenv(secret_keys.DIR_VAR, str(directory))
    for name in secret_keys.ON_DEMAND_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    return directory


@pytest.fixture
def restore_graph_factory(monkeypatch):
    """Undo install() at teardown: start from the core's own function."""
    monkeypatch.setattr(trading_graph, "create_tier_client", factory.create_tier_client)


def put(directory, name, content):
    path = directory / name
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


def tier_config(**overrides):
    return {**DEFAULT_CONFIG, "llm_provider": "anthropic", "backend_url": None,
            "deep_think_provider": None, "quick_think_provider": None,
            "deep_think_llm": "claude-opus-5-5", "quick_think_llm": "claude-haiku-5-5",
            **overrides}


# ---- names ---------------------------------------------------------------------


def test_api_secret_keys_on_demand_names():
    providers, names = secret_keys.ON_DEMAND_PROVIDERS, secret_keys.ON_DEMAND_KEY_NAMES
    assert providers == {"anthropic", "google"}
    assert names == {"ANTHROPIC_API_KEY", "GOOGLE_API_KEY"}


# ---- read_secret ---------------------------------------------------------------


def test_api_secret_keys_reads_regular_file(secrets):
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-file")
    assert secret_keys.read_secret("ANTHROPIC_API_KEY") == "sk-ant-file"


@pytest.mark.parametrize("content", ["sk-line\n", "sk-line\r\n", "sk-line\n\n\r\n"])
def test_api_secret_keys_trailing_newlines_stripped(secrets, content):
    put(secrets, "ANTHROPIC_API_KEY", content)
    assert secret_keys.read_secret("ANTHROPIC_API_KEY") == "sk-line"


def test_api_secret_keys_inner_and_leading_whitespace_kept(secrets):
    put(secrets, "ANTHROPIC_API_KEY", " a b\n")
    assert secret_keys.read_secret("ANTHROPIC_API_KEY") == " a b"


@pytest.mark.parametrize("content", ["", "\n", "   \n", "\t\r\n"])
def test_api_secret_keys_empty_file_is_none(secrets, content, caplog):
    put(secrets, "ANTHROPIC_API_KEY", content)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert caplog.records == []


def test_api_secret_keys_missing_file_or_dir_is_silent(secrets, monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
        monkeypatch.setenv(secret_keys.DIR_VAR, str(tmp_path / "does-not-exist"))
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert caplog.records == []


def test_api_secret_keys_default_dir_is_run_secrets(monkeypatch):
    monkeypatch.delenv(secret_keys.DIR_VAR, raising=False)
    assert secret_keys.secrets_dir() == "/run/secrets"


@pytest.mark.parametrize("name", ["anthropic_api_key", "1KEY", "_KEY", "A/B", "../KEY", "", "A.B"])
def test_api_secret_keys_invalid_name_rejected(secrets, name):
    with pytest.raises(ValueError, match="not a secret variable name"):
        secret_keys.read_secret(name)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need POSIX")
def test_api_secret_keys_symlink_skipped(secrets, tmp_path, caplog):
    outside = tmp_path / "outside.txt"
    outside.write_text("dummy-outside-value\n")
    (secrets / "ANTHROPIC_API_KEY").symlink_to(outside)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert "ANTHROPIC_API_KEY" in caplog.text and "symlink" in caplog.text
    assert "dummy-outside-value" not in caplog.text


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="needs O_NOFOLLOW")
def test_api_secret_keys_symlink_swapped_in_after_lstat_not_followed(
    secrets, tmp_path, monkeypatch, caplog
):
    # A symlink that replaces the file between the lstat and the open: the
    # open itself refuses it.
    outside = tmp_path / "outside.txt"
    outside.write_text("dummy-outside-value")
    link = secrets / "ANTHROPIC_API_KEY"
    link.symlink_to(outside)
    real_lstat = os.lstat
    monkeypatch.setattr(
        os, "lstat", lambda p, *a, **k: real_lstat(outside) if str(p) == str(link) else real_lstat(p, *a, **k)
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert "Ignoring secret ANTHROPIC_API_KEY: cannot read it" in caplog.text
    assert "dummy-outside-value" not in caplog.text


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_api_secret_keys_fifo_skipped_without_blocking(secrets, caplog):
    os.mkfifo(secrets / "ANTHROPIC_API_KEY")
    result = []
    # On a thread with a timeout, so a regression (an open that blocks) fails
    # the test instead of hanging the run.
    worker = threading.Thread(
        target=lambda: result.append(secret_keys.read_secret("ANTHROPIC_API_KEY")), daemon=True
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        worker.start()
        worker.join(timeout=10)
    assert not worker.is_alive(), "read_secret blocked on a FIFO"
    assert result == [None]
    assert "Ignoring secret ANTHROPIC_API_KEY: not a regular file" in caplog.text


def test_api_secret_keys_directory_entry_skipped(secrets, caplog):
    (secrets / "ANTHROPIC_API_KEY").mkdir()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert "ANTHROPIC_API_KEY" in caplog.text


def test_api_secret_keys_oversized_file_skipped(secrets, caplog):
    put(secrets, "ANTHROPIC_API_KEY", "x" * (64 * 1024 + 1))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert "larger than 65536 bytes" in caplog.text
    assert "x" * 16 not in caplog.text


def test_api_secret_keys_exactly_64k_is_read(secrets):
    put(secrets, "ANTHROPIC_API_KEY", "y" * (64 * 1024))
    assert secret_keys.read_secret("ANTHROPIC_API_KEY") == "y" * (64 * 1024)


def test_api_secret_keys_nul_byte_skipped(secrets, caplog):
    put(secrets, "ANTHROPIC_API_KEY", "sk-a\0b")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert "contains a NUL byte" in caplog.text
    assert "sk-a" not in caplog.text


def test_api_secret_keys_undecodable_file_skipped(secrets, caplog):
    put(secrets, "ANTHROPIC_API_KEY", b"sk-\xff\xfe-raw")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    assert "not valid UTF-8" in caplog.text
    assert "sk-" not in caplog.text


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_api_secret_keys_unreadable_file_skipped(secrets, caplog):
    path = put(secrets, "ANTHROPIC_API_KEY", "sk-unreadable")
    path.chmod(0)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None
    finally:
        path.chmod(0o600)
    assert "cannot read it (PermissionError)" in caplog.text
    assert "sk-unreadable" not in caplog.text


def test_api_secret_keys_never_written_to_environ(secrets):
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-not-in-env")
    put(secrets, "GOOGLE_API_KEY", "g-not-in-env")
    before = dict(os.environ)
    assert secret_keys.api_key_for("anthropic") == "sk-ant-not-in-env"
    assert secret_keys.api_key_for("google") == "g-not-in-env"
    assert dict(os.environ) == before
    assert "sk-ant-not-in-env" not in os.environ.values()


def test_api_secret_keys_reread_every_time(secrets):
    path = put(secrets, "ANTHROPIC_API_KEY", "sk-first")
    assert secret_keys.read_secret("ANTHROPIC_API_KEY") == "sk-first"
    path.write_text("sk-second\n")
    assert secret_keys.read_secret("ANTHROPIC_API_KEY") == "sk-second"
    path.unlink()
    assert secret_keys.read_secret("ANTHROPIC_API_KEY") is None


def test_api_secret_keys_logs_never_contain_values(secrets, tmp_path, caplog):
    values = ["sk-nul\0x", "sk-big-" + "z" * 70000]
    put(secrets, "ANTHROPIC_API_KEY", values[0])
    put(secrets, "GOOGLE_API_KEY", values[1])
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert secret_keys.api_key_for("anthropic") is None
        assert secret_keys.api_key_for("google") is None
    assert len(caplog.records) == 2
    assert "sk-" not in caplog.text


# ---- api_key_for ------------------------------------------------------------------


def test_api_secret_keys_api_key_for_on_demand_providers(secrets):
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant")
    put(secrets, "GOOGLE_API_KEY", "g-key")
    assert secret_keys.api_key_for("anthropic") == "sk-ant"
    assert secret_keys.api_key_for("Anthropic") == "sk-ant"
    assert secret_keys.api_key_for("google") == "g-key"


@pytest.mark.parametrize("provider", ["openai", "ollama", "xai", "openrouter", "azure", "bedrock", "", None])
def test_api_secret_keys_api_key_for_other_providers_is_none(secrets, provider):
    put(secrets, "OPENAI_API_KEY", "sk-openai")
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant")
    assert secret_keys.api_key_for(provider) is None


def test_api_secret_keys_env_value_not_returned(secrets, monkeypatch):
    # No file: None, so the client falls back to its own env lookup.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-dotenv")
    assert secret_keys.api_key_for("anthropic") is None


# ---- the create_tier_client wrapper -----------------------------------------------


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, config, tier, **extra):
        self.calls.append((tier, extra))
        return object()


def test_api_secret_keys_wrapper_adds_key_for_anthropic_tier(secrets):
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-file")
    rec = Recorder()
    secret_keys._wrap(rec)(tier_config(), "deep", callbacks=["cb"])
    assert rec.calls == [("deep", {"callbacks": ["cb"], "api_key": "sk-ant-file"})]


def test_api_secret_keys_wrapper_uses_tier_provider(secrets):
    put(secrets, "GOOGLE_API_KEY", "g-file")
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-file")
    rec = Recorder()
    config = tier_config(llm_provider="ollama", deep_think_provider="google",
                         deep_think_llm="gemini-3.8-flash", quick_think_llm="qwen3:8b")
    wrapped = secret_keys._wrap(rec)
    wrapped(config, "deep")
    wrapped(config, "quick")
    assert rec.calls == [("deep", {"api_key": "g-file"}), ("quick", {})]


def test_api_secret_keys_wrapper_keeps_explicit_api_key(secrets):
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-file")
    rec = Recorder()
    secret_keys._wrap(rec)(tier_config(), "quick", api_key="explicit")
    assert rec.calls == [("quick", {"api_key": "explicit"})]


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_api_secret_keys_wrapper_leaves_other_providers_alone(secrets, provider):
    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-file")
    put(secrets, "OPENAI_API_KEY", "sk-openai-file")
    rec = Recorder()
    secret_keys._wrap(rec)(tier_config(llm_provider=provider), "deep")
    assert rec.calls == [("deep", {})]


def test_api_secret_keys_wrapper_without_secret_adds_nothing(secrets, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-dotenv")
    rec = Recorder()
    secret_keys._wrap(rec)(tier_config(), "deep")
    assert rec.calls == [("deep", {})]


def test_api_secret_keys_wrapper_leaves_bad_config_to_core(secrets):
    rec = Recorder()
    secret_keys._wrap(rec)({}, "deep")
    assert rec.calls == [("deep", {})]


def test_api_secret_keys_install_is_idempotent(restore_graph_factory):
    original = trading_graph.create_tier_client
    secret_keys.install()
    first = trading_graph.create_tier_client
    assert first is not original
    assert getattr(first, secret_keys._WRAPPED_ATTR) is original
    secret_keys.install()
    assert trading_graph.create_tier_client is first


def test_api_secret_keys_install_fails_loudly_without_core_name(monkeypatch):
    monkeypatch.delattr(trading_graph, "create_tier_client")
    with pytest.raises(RuntimeError, match="create_tier_client"):
        secret_keys.install()


def test_api_secret_keys_create_app_installs(restore_graph_factory):
    from tradingagents.api.app import create_app

    create_app()
    assert hasattr(trading_graph.create_tier_client, secret_keys._WRAPPED_ATTR)


def test_api_secret_keys_llm_provider_passes_key(secrets, monkeypatch):
    from unittest.mock import MagicMock

    from tradingagents.api.infrastructure.llm_provider import LLMProviderFactory

    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-file")
    mock = MagicMock()
    monkeypatch.setattr(factory, "create_llm_client", mock)
    LLMProviderFactory(config=tier_config()).create_deep_thinking_llm()
    assert mock.call_args.kwargs["api_key"] == "sk-ant-file"


# ---- guard against upstream drift ---------------------------------------------------


def test_api_secret_keys_real_graph_gets_key_from_file(secrets, tmp_path, restore_graph_factory):
    """A real TradingAgentsGraph builds its anthropic tier with the file's key.

    Fails if the core stops building its tier clients through the module-level
    name ``tradingagents.graph.trading_graph.create_tier_client`` (e.g. calls
    the factory through another import), since install() would then no longer
    reach the analysis graph.
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    put(secrets, "ANTHROPIC_API_KEY", "sk-ant-guard-value\n")
    assert "ANTHROPIC_API_KEY" not in os.environ
    secret_keys.install()
    config = tier_config(
        llm_provider="ollama", quick_think_llm="qwen3:8b",
        deep_think_provider="anthropic", deep_think_llm="claude-opus-5-5",
        results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "memory" / "log.md"),
    )
    graph = TradingAgentsGraph(selected_analysts=["market"], config=config)
    assert graph.deep_thinking_llm.anthropic_api_key.get_secret_value() == "sk-ant-guard-value"
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "sk-ant-guard-value" not in os.environ.values()
