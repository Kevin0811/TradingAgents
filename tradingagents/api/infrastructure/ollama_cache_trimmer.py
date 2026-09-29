"""Trim a local Ollama model's prompt cache between LLM calls.

Ollama 0.34.x runs MLX models (e.g. ``qwen3.5:4b-mlx``) on an engine that keeps
a prefix-cache snapshot per request under a fixed 8 GiB budget that no setting
lowers (ollama/ollama#18131). The model's resident size grows call by call --
one TWD=X analysis took it from 4.0 to 12.2 GB -- and the host starts swapping.
Unloading the model (``keep_alive: 0``) is the only way to drop that cache; the
next call loads it again in a few seconds.

Only runs whose ``llm_provider`` is ``ollama`` are trimmed. After each of their
LLM calls, :meth:`OllamaCacheTrimmer.check` reads ``GET /api/ps`` and unloads a
model of the run once its total ``size`` has grown more than the budget past its
baseline. Ollama defers an unload until the runner's in-flight request has
finished, so an unload never breaks a call; at worst the next call waits for
the reload.

The logic follows fin-insight's trimmer (``backend/internal/gateway/ollama/
cachetrim.go``), minus its ``load_duration`` signal, which the OpenAI-compatible
``/v1`` path the core uses does not report:

* State is per server and normalised model name. A model not loaded at a check
  starts over; a smaller size lowers the baseline.
* **Baseline.** The size at the first read of a load, capped at the model's
  weights (its ``size`` in ``GET /api/tags``, cached per server and model) plus
  :data:`WEIGHTS_OVERHEAD_MB`. The cap matters when the first read already
  holds a grown cache -- after this container restarted while the host's Ollama
  kept a 12 GB model -- which is then trimmed on that first check. Without
  ``/api/tags`` the first read is the baseline, as before.
* **Context length.** ``/api/ps`` reports the load's ``context_length``. When it
  changes, the model was loaded again with another ``num_ctx`` (fin-insight calls
  the same model through native ``/api/chat`` with its own): the state starts
  over and the fresh size is the baseline, never unloaded at once.
* **After an unload** Ollama confirmed (``done_reason == "unload"``), the next
  read decides, against ``trimmed_at``, the size at the trim. Smaller: the
  memory was released and the model loaded again, by us or by the other app;
  the baseline is re-anchored on the fresh size, even one already over the old
  baseline plus the budget (a reload is never unloaded again at once). Still at
  least ``trimmed_at``: the unload did not take, or the other app reloaded the
  model larger. That is retried once at once (logged at INFO); each further one
  is a WARNING, and from then on only every :data:`NOT_TAKEN_BACKOFF_CHECKS`-th
  due check unloads, so another app's fresh load is not evicted over and over.
  If the trim came from the weights cap alone, a model just as large after it
  is taken as a load that is simply that large (a long context, a preallocated
  cache): the baseline becomes its size and the cap is not used again for it
  until its context length changes.
* **Backoff.** Only failed unload requests (an HTTP error, or an answer whose
  ``done_reason`` is not ``unload``) count as failures. After three in a row,
  only every tenth due check tries again; an answered unload resets them. Three
  failed ``/api/ps`` reads in a row back the reads off the same way.
* A model of the run missing from ``/api/ps`` while other models are loaded is
  logged once per model: its name is most likely not the one Ollama uses.

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

TAGS_TIMEOUT_SECONDS = 3.0

# Failed unload requests (or /api/ps reads) in a row before the backoff, and
# the due checks per attempt once backed off.
FAILURES_BEFORE_BACKOFF = 3
BACKOFF_CHECKS = 10
# An unload Ollama confirmed, but the model still as large at the next read:
# retried this many times at once, then only every NOT_TAKEN_BACKOFF_CHECKS-th
# due check. Higher than BACKOFF_CHECKS: the likeliest cause is another app's
# fresh load with a longer context, which should not be evicted over and over.
NOT_TAKEN_RETRIES = 1
NOT_TAKEN_BACKOFF_CHECKS = 20

# What a fresh load may add to the weights' size before its first read counts
# as already grown. At the 4096-token context the core runs with, the KV cache
# of a small model is well under that -- a 4B Qwen3 (36 layers, 8 KV heads of
# 128 dims, fp16 K and V) takes 144 KiB a token, 576 MiB for 4096 tokens -- and
# the compute buffers add a few hundred MB. A load larger than that (a long
# context) costs one needless unload at most: see "If the trim came from the
# weights cap alone" above.
WEIGHTS_OVERHEAD_MB = 1024

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
    capped: bool = False  # the baseline is the weights reference, below the first read
    context_length: int | None = None  # of the load measured, when /api/ps reports it
    pending: bool = False  # an unload was answered; the next read confirms it
    trimmed_at: int = 0  # bytes: the size at that unload
    reference_trim: bool = False  # that unload was due only through the capped baseline
    failures: int = 0  # unload requests in a row that failed or were refused
    not_taken: int = 0  # answered unloads in a row after which the model was as large
    since_attempt: int = 0  # due checks skipped by the backoff
    use_reference: bool = True  # False once the weights cap proved too low for this model

    def start_over(self) -> None:
        """Forget the load measured (not whether the weights cap suits the model)."""
        self.baseline, self.capped, self.context_length = 0, False, None
        self.pending, self.trimmed_at, self.reference_trim = False, 0, False
        self.failures, self.not_taken, self.since_attempt = 0, 0, 0


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
    lock serialises all checks. The app builds one only when ``llm_provider``
    is ``ollama`` and the budget is above 0.
    """

    def __init__(self, budget_mb: int, http: OllamaHttp | None = None):
        self._budget = max(int(budget_mb), 0) * _MB
        self._http = http or UrllibOllamaHttp()
        self._lock = threading.Lock()
        self._models: dict[str, _ModelState] = {}
        self._reads: dict[str, _ReadState] = {}
        # Per "<server> <model>": the weights' size in bytes from /api/tags, or
        # None when the listing has no such model. A failed read is not kept.
        self._weights: dict[str, int | None] = {}
        self._missing_warned: set[str] = set()
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
        key = f"{native_url} {model}"
        state = self._models.setdefault(key, _ModelState())
        entry = _find_loaded(running, model)
        if entry is None:
            self._warn_if_misnamed(key, model, running)
            # Unloaded (by us, by the keep-alive or by another app): the load
            # measured is gone; the next one gets a fresh baseline.
            state.start_over()
            return TrimResult(model, loaded=False)
        size = int(entry.get("size") or 0)
        context_length = _int_or_none(entry.get("context_length"))

        if (
            context_length is not None
            and state.context_length is not None
            and context_length != state.context_length
        ):
            # Loaded again with another num_ctx (the other app's, or ours):
            # a new load, measured from its fresh size, never unloaded at once.
            logger.info(
                "Ollama model %s was loaded again with context length %d (was %d); "
                "measuring the new load from %d MB",
                model,
                context_length,
                state.context_length,
                size // _MB,
            )
            state.start_over()
            state.use_reference = True
            state.baseline = size
        elif state.pending:
            self._confirm_unload(state, model, size)
        elif state.not_taken and size < state.trimmed_at:
            # Backed off after unloads that did not take, and now smaller
            # than at the last one: released and loaded again after all.
            state.baseline, state.capped, state.not_taken = size, False, 0
        if context_length is not None:
            state.context_length = context_length

        if state.baseline == 0:
            reference = self._reference(native_url, model) if state.use_reference else None
            if reference is not None and size > reference:
                state.baseline, state.capped = reference, True
            else:
                state.baseline, state.capped = size, False
        elif size < state.baseline:
            state.baseline, state.capped = size, False
        result = TrimResult(model, loaded=True, size=size, baseline=state.baseline)
        if size - state.baseline <= self._budget:
            return result

        cadence = None
        if state.failures >= FAILURES_BEFORE_BACKOFF:
            cadence = BACKOFF_CHECKS
        elif state.not_taken > NOT_TAKEN_RETRIES:
            cadence = NOT_TAKEN_BACKOFF_CHECKS
        if cadence is not None:
            state.since_attempt += 1
            if state.since_attempt < cadence:
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

        state.failures = 0
        state.pending, state.trimmed_at, state.reference_trim = True, size, state.capped
        self.trims += 1
        logger.info(
            "Unloaded Ollama model %s to trim its prompt cache: %d MB, baseline %d MB%s, "
            "budget %d MB",
            model,
            size // _MB,
            state.baseline // _MB,
            " (weights + overhead)" if state.capped else "",
            self.budget_mb,
        )
        return replace(result, trimmed=True)

    def _confirm_unload(self, state: _ModelState, model: str, size: int) -> None:
        """Read the first size after an answered unload (see the module docs)."""
        state.pending = False
        if size < state.trimmed_at:
            # The memory was released and the model loaded again -- by our
            # next call or by the other app, possibly with a longer context.
            # Measure that load from here, even past the old baseline + budget.
            state.baseline, state.capped, state.not_taken = size, False, 0
            return
        if state.reference_trim:
            # Unloaded only because the first read was over weights + overhead,
            # and back just as large: this model's loads are simply that big.
            logger.info(
                "Ollama model %s is %d MB right after loading, over its weights' size "
                "plus %d MB; measuring it from there",
                model,
                size // _MB,
                WEIGHTS_OVERHEAD_MB,
            )
            state.baseline, state.capped, state.use_reference = size, False, False
            return
        state.not_taken += 1
        retrying = state.not_taken <= NOT_TAKEN_RETRIES
        logger.log(
            logging.INFO if retrying else logging.WARNING,
            "Ollama model %s was as large after its cache trim (%d MB, %d MB at the trim, "
            "baseline %d MB, not taken %d in a row): the unload did not take, or another "
            "app loaded it again larger; %s",
            model,
            size // _MB,
            state.trimmed_at // _MB,
            state.baseline // _MB,
            state.not_taken,
            "unloading it again"
            if retrying
            else f"retrying every {NOT_TAKEN_BACKOFF_CHECKS} due checks",
        )

    def _reference(self, native_url: str, model: str) -> int | None:
        """The model's weights plus :data:`WEIGHTS_OVERHEAD_MB`, in bytes, if known."""
        key = f"{native_url} {model}"
        if key not in self._weights:
            try:
                body = self._http.get_json(f"{native_url}/api/tags", TAGS_TIMEOUT_SECONDS)
                listing = body.get("models") or []
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not read the local models' sizes (%s/api/tags); measuring %s "
                    "from its first read: %s",
                    native_url,
                    model,
                    exc,
                )
                return None  # not kept: the next fresh load asks again
            entry = _find_loaded(listing, model)
            weights = _int_or_none(entry.get("size")) if entry is not None else None
            self._weights[key] = weights if weights and weights > 0 else None
        weights = self._weights[key]
        return None if weights is None else weights + WEIGHTS_OVERHEAD_MB * _MB

    def _warn_if_misnamed(self, key: str, model: str, running: list) -> None:
        loaded = [e.get("name") or e.get("model") for e in running if isinstance(e, dict)]
        loaded = [n for n in loaded if n]
        if not loaded or key in self._missing_warned:
            return
        self._missing_warned.add(key)
        # A call just ran, so the model should be listed: most likely its name
        # is not the one Ollama reports, and it is never trimmed.
        logger.warning(
            "Ollama model %s is not among the loaded ones (%s); its cache cannot be "
            "trimmed -- check the model name",
            model,
            ", ".join(loaded),
        )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _find_loaded(running: list[dict], model: str) -> dict | None:
    want = model.lower()
    for entry in running:
        if not isinstance(entry, dict):
            continue
        names = (entry.get("name"), entry.get("model"))
        if any(n and normalize_model_name(n).lower() == want for n in names):
            return entry
    return None
