"""Trim a local Ollama model's prompt cache between LLM calls.

Ollama 0.34.x runs MLX models (e.g. ``qwen3.5:4b-mlx``) on an engine that keeps
a prefix-cache snapshot per request under a fixed 8 GiB budget that no setting
lowers (ollama/ollama#18131). The model's resident size grows call by call --
one TWD=X analysis took it from 4.0 to 12.2 GB -- and the host starts swapping.
Unloading the model (``keep_alive: 0``) is the only way to drop that cache; the
next call loads it again in a few seconds.

So after each LLM call of an analysis, :meth:`OllamaCacheTrimmer.check` reads
``GET /api/ps`` and unloads a model of the run once its total ``size`` has grown
more than the budget past its baseline -- its size at the first read after it
was loaded. Ollama defers an unload until the runner's in-flight request has
finished, so an unload never breaks a call; at worst the next call waits for
the reload.

The logic follows fin-insight's trimmer (``backend/internal/gateway/ollama/
cachetrim.go``), minus its ``load_duration`` signal, which the OpenAI-compatible
``/v1`` path the core uses does not report:

* The baseline is per server and normalised model name. A model not loaded at
  a check starts over; a smaller size lowers the baseline.
* The model can be shared with another process (fin-insight calls the same
  Ollama model), which may load it again between our unload and our next
  check. So an unload Ollama confirmed (``done_reason == "unload"``) is checked
  on the next read: within the budget of the old baseline, it is taken as
  reloaded (by us or by that process) and the baseline stays; still over the
  budget, the unload did not take -- a failure, and it is unloaded again.
* After three failures in a row (a failed or refused unload, one that did not
  take, or a failed ``/api/ps`` read), only every tenth due check tries again,
  so a server that cannot unload is not asked on every call.

Every request has a short timeout and every error is logged and swallowed: the
trim must never fail an analysis, nor slow it beyond the check itself. Checks
are serialised by one lock, so concurrent analyses never interleave reads and
unloads of the same model.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# GET /api/ps is a cheap in-memory listing; Ollama answers an unload once it has
# told the runner to stop, before the memory is back. Both are quick, and the
# bounds only keep a hung server from stalling an analysis.
PS_TIMEOUT_SECONDS = 3.0
UNLOAD_TIMEOUT_SECONDS = 10.0

FAILURES_BEFORE_BACKOFF = 3
BACKOFF_CHECKS = 10

_MAX_RESPONSE_BYTES = 1 << 20
_MB = 1 << 20


class OllamaHttp(Protocol):
    """The two HTTP calls the trimmer makes (swapped for a fake in tests)."""

    def get_json(self, url: str, timeout: float) -> Any:
        """GET ``url`` and return the decoded JSON body; raise on any failure."""

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        """POST ``payload`` as JSON to ``url`` and return the decoded JSON body."""


class UrllibOllamaHttp:
    """Stdlib HTTP client: no extra dependency, a timeout on every request."""

    def get_json(self, url: str, timeout: float) -> Any:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 -- configured URL
            return json.loads(resp.read(_MAX_RESPONSE_BYTES))

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read(_MAX_RESPONSE_BYTES))


def normalize_model_name(name: str) -> str:
    """The name with the tag Ollama assumes: ``qwen3.5`` is ``qwen3.5:latest``.

    The tag is a ``:`` after the last ``/``, so a registry path such as
    ``localhost:5000/ns/name`` (whose ``:`` is the host's port) still gets one.
    """
    name = (name or "").strip()
    if not name or ":" in name[name.rfind("/") + 1 :]:
        return name
    return name + ":latest"


def ollama_native_url(backend_url: str | None) -> str:
    """Ollama's native API root for a run: its ``/v1`` base URL without ``/v1``.

    Resolved as the core's OpenAI-compatible client resolves the ``/v1`` URL:
    the configured ``backend_url``, else ``OLLAMA_BASE_URL``, else the
    provider's default (``http://localhost:11434/v1``).
    """
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    spec = OPENAI_COMPATIBLE_PROVIDERS["ollama"]
    env_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
    url = (backend_url or env_url or spec.base_url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url.rstrip("/")


@dataclass
class _ModelState:
    """What is known about one load of one model on one server."""

    baseline: int = 0  # bytes; 0 = not yet measured on this load
    pending: bool = False  # an unload was answered; the next read confirms it
    failures: int = 0  # unloads in a row that failed or did not take
    since_attempt: int = 0  # due checks skipped by the backoff

    def start_over(self) -> None:
        self.baseline, self.pending = 0, False
        self.failures, self.since_attempt = 0, 0


@dataclass
class _ReadState:
    """Failed ``/api/ps`` reads on one server, for the backoff."""

    failures: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class TrimResult:
    """What one check did for one model of the run (for logs and tests)."""

    model: str
    loaded: bool
    size: int = 0
    baseline: int = 0
    trimmed: bool = False
    deferred: bool = False
    error: str | None = None


class OllamaCacheTrimmer:
    """Unloads a run's Ollama models once their prompt cache outgrows a budget.

    One instance is shared by every analysis of the app (``app.state``), so its
    lock serialises all checks.
    """

    def __init__(self, budget_mb: int, http: OllamaHttp | None = None):
        self._budget = max(int(budget_mb), 0) * _MB
        self._http = http or UrllibOllamaHttp()
        self._lock = threading.Lock()
        self._models: dict[str, _ModelState] = {}
        self._reads: dict[str, _ReadState] = {}
        self.trims = 0

    @property
    def budget_mb(self) -> int:
        return self._budget // _MB

    @property
    def enabled(self) -> bool:
        return self._budget > 0

    def check(self, native_url: str, models: Iterable[str]) -> list[TrimResult]:
        """Read ``/api/ps`` once and trim each of ``models`` that outgrew the budget.

        ``models`` are the models of the run (its quick and deep model); other
        loaded models are never touched. Never raises.
        """
        if not self.enabled:
            return []
        wanted = list(dict.fromkeys(normalize_model_name(m) for m in models if m))
        if not wanted:
            return []
        try:
            with self._lock:
                return self._check_locked(native_url, wanted)
        except Exception:  # noqa: BLE001 -- never fail the analysis
            logger.warning("Ollama cache trim check failed", exc_info=True)
            return []

    def _check_locked(self, native_url: str, wanted: list[str]) -> list[TrimResult]:
        read = self._reads.setdefault(native_url, _ReadState())
        if read.failures >= FAILURES_BEFORE_BACKOFF:
            read.skipped += 1
            if read.skipped < BACKOFF_CHECKS:
                return [TrimResult(m, loaded=False, deferred=True) for m in wanted]
            read.skipped = 0
        try:
            body = self._http.get_json(f"{native_url}/api/ps", PS_TIMEOUT_SECONDS)
            running = body.get("models") or []
        except Exception as exc:  # noqa: BLE001
            read.failures += 1
            logger.warning(
                "Could not read the local model's memory to trim its cache (%s/api/ps, "
                "failures=%d): %s",
                native_url,
                read.failures,
                exc,
            )
            return [TrimResult(m, loaded=False, error=str(exc)) for m in wanted]
        read.failures = read.skipped = 0
        return [self._check_model(native_url, model, running) for model in wanted]

    def _check_model(self, native_url: str, model: str, running: list[dict]) -> TrimResult:
        state = self._models.setdefault(f"{native_url} {model}", _ModelState())
        entry = _find_loaded(running, model)
        if entry is None:
            # Unloaded (by us, by the keep-alive or by another app): the load
            # measured is gone; the next one gets a fresh baseline.
            state.start_over()
            return TrimResult(model, loaded=False)
        size = int(entry.get("size") or 0)

        retry = False
        if state.pending:
            state.pending = False
            if size - state.baseline > self._budget:
                # Still over the budget of the old baseline: the unload did
                # not take. (Within it, the model was loaded again -- by our
                # next call or by another app sharing the server.)
                retry = True
                state.failures += 1
                logger.warning(
                    "Ollama model %s was still loaded after its cache trim (%d MB, baseline "
                    "%d MB, failures=%d); unloading it again",
                    model,
                    size // _MB,
                    state.baseline // _MB,
                    state.failures,
                )
            else:
                state.failures = state.since_attempt = 0
        if state.baseline == 0 or size < state.baseline:
            state.baseline = size
        result = TrimResult(model, loaded=True, size=size, baseline=state.baseline)
        if not retry and size - state.baseline <= self._budget:
            return result

        if state.failures >= FAILURES_BEFORE_BACKOFF:
            state.since_attempt += 1
            if state.since_attempt < BACKOFF_CHECKS:
                return replace(result, deferred=True)
        state.since_attempt = 0

        name = entry.get("name") or entry.get("model") or model
        try:
            answer = self._http.post_json(
                f"{native_url}/api/generate",
                {"model": name, "keep_alive": 0},
                UNLOAD_TIMEOUT_SECONDS,
            )
            reason = answer.get("done_reason") if isinstance(answer, dict) else None
            if reason != "unload":
                # A proxy's own 200, or a server that took it for a generation.
                raise RuntimeError(f"done_reason={reason!r}, want 'unload'")
        except Exception as exc:  # noqa: BLE001
            state.failures += 1
            logger.warning(
                "Could not unload Ollama model %s to trim its cache (%d MB, failures=%d): %s",
                model,
                size // _MB,
                state.failures,
                exc,
            )
            return replace(result, error=str(exc))

        state.pending = True
        self.trims += 1
        logger.info(
            "Unloaded Ollama model %s to trim its prompt cache: %d MB, baseline %d MB, "
            "budget %d MB",
            model,
            size // _MB,
            state.baseline // _MB,
            self.budget_mb,
        )
        return replace(result, trimmed=True)


def _find_loaded(running: list[dict], model: str) -> dict | None:
    want = model.lower()
    for entry in running:
        if not isinstance(entry, dict):
            continue
        names = (entry.get("name"), entry.get("model"))
        if any(n and normalize_model_name(n).lower() == want for n in names):
            return entry
    return None
