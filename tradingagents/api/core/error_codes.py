"""Stable codes for why an analysis task failed or was cancelled.

``error`` and ``message`` stay free text; a client that wants to show its own
(localised) copy reads ``error_code`` on a failed task and ``cancel_reason`` on
a cancelled one instead. The codes are part of the API: add new ones, never
rename or reuse one.

:func:`classify_error` maps the exception that ended a run to a failure code.
It looks at exception types and HTTP status codes, walking the chain of
``__cause__`` / ``__context__`` (the API raises ``AnalysisError`` *from* the
core's exception, and the core wraps some errors itself; a context hidden by
``raise ... from None`` is not followed). Status codes are read only from LLM
client errors (see :func:`_from_llm_client`), so an unrelated 404 such as
FastAPI's ``HTTPException`` is not a missing model. An exception raised inside a
market data vendor (a traceback frame in ``tradingagents.dataflows`` or
``yfinance``) is ``TA_DATA_VENDOR_FAILED`` even when it is a plain timeout or
connection error, unless an LLM SDK's error is in the chain. Message text is
read only as a last resort, for one case: an Ollama/OpenAI client error that
reports a missing model without a 404 (see :func:`_says_model_not_found`).
"""

from __future__ import annotations

import functools
import importlib
import logging
import re
from collections.abc import Iterator

from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Failure codes (``error_code``; set only when a task ends ``failed``).
TA_LLM_TIMEOUT = "TA_LLM_TIMEOUT"
TA_LLM_UNREACHABLE = "TA_LLM_UNREACHABLE"
TA_LLM_AUTH = "TA_LLM_AUTH"
TA_LLM_MODEL_NOT_FOUND = "TA_LLM_MODEL_NOT_FOUND"
TA_LLM_RATE_LIMITED = "TA_LLM_RATE_LIMITED"
TA_LLM_RESPONSE_INVALID = "TA_LLM_RESPONSE_INVALID"
TA_DATA_VENDOR_FAILED = "TA_DATA_VENDOR_FAILED"
TA_RUN_FAILED = "TA_RUN_FAILED"

ERROR_CODES = (
    TA_LLM_TIMEOUT,
    TA_LLM_UNREACHABLE,
    TA_LLM_AUTH,
    TA_LLM_MODEL_NOT_FOUND,
    TA_LLM_RATE_LIMITED,
    TA_LLM_RESPONSE_INVALID,
    TA_DATA_VENDOR_FAILED,
    TA_RUN_FAILED,
)

# Cancel reasons (``cancel_reason``; set only when a task ends ``cancelled``).
TA_CANCELLED_BEFORE_START = "TA_CANCELLED_BEFORE_START"
TA_CANCELLED_WHILE_RUNNING = "TA_CANCELLED_WHILE_RUNNING"

CANCEL_REASONS = (TA_CANCELLED_BEFORE_START, TA_CANCELLED_WHILE_RUNNING)

# Modules whose exceptions come from a market data vendor. The data layer's own
# taxonomy (``tradingagents.dataflows.errors.VendorError``) lives in the first.
_DATA_VENDOR_MODULES = ("tradingagents.dataflows", "yfinance")

# Modules whose exceptions may carry an LLM provider's HTTP status (any
# ``langchain_*`` package counts too, see :func:`_from_llm_client`).
_LLM_CLIENT_MODULES = ("openai", "anthropic", "google.genai", "httpx", "ollama")

# LLM SDKs proper: one of their errors in the chain means the LLM failed, even
# when the call was made from a vendor frame. (``httpx`` is left out: data
# vendors use it too.)
_LLM_SDK_MODULES = ("openai", "anthropic", "google.genai", "ollama")

# Clients whose "model ... not found" text is trusted when there is no 404.
_MODEL_TEXT_MODULES = ("ollama", "openai", "langchain_ollama", "langchain_openai")

# Exceptions in a chain looked at before giving up (a guard against odd cycles).
_MAX_CHAIN = 32

# Traceback frames looked at per exception when looking for a vendor frame.
_MAX_FRAMES = 512

# Last resort only: Ollama (and servers like it) word a missing model as
# "model 'x' not found" / "model \"x\" not found, try pulling it first".
_MODEL_NOT_FOUND_TEXT = re.compile(r"\bmodel\b.{0,200}?\bnot found\b", re.IGNORECASE | re.DOTALL)


def _optional_types(*names: str) -> tuple[type[BaseException], ...]:
    """The named exception classes (``"module:Class"``) that can be imported.

    Provider SDKs are optional for the API: one that is not installed, or that
    breaks on import, simply contributes nothing (its errors fall back to the
    status-code rules). Called once per process (``_types`` is cached), so each
    failure is logged once, at debug.
    """
    found: list[type[BaseException]] = []
    for name in names:
        module_name, _, attr = name.partition(":")
        try:
            cls = getattr(importlib.import_module(module_name), attr)
        except Exception as exc:  # noqa: BLE001 - a broken SDK must not disable classification
            logger.debug("error_codes: %s unavailable (%s: %s)", name, type(exc).__name__, exc)
            continue
        if isinstance(cls, type) and issubclass(cls, BaseException):
            found.append(cls)
    return tuple(found)


@functools.cache
def _types() -> dict[str, tuple[type[BaseException], ...]]:
    # Imported on first use, not when the API starts.
    from tradingagents.dataflows.errors import VendorError

    return {
        TA_DATA_VENDOR_FAILED: (VendorError,),
        # Timeouts first: the SDKs' APITimeoutError subclasses APIConnectionError.
        TA_LLM_TIMEOUT: (TimeoutError,)  # asyncio.TimeoutError is TimeoutError (3.11+)
        + _optional_types(
            "httpx:TimeoutException",
            "openai:APITimeoutError",
            "anthropic:APITimeoutError",
        ),
        TA_LLM_UNREACHABLE: (ConnectionError,)
        + _optional_types(
            "httpx:ConnectError",
            "openai:APIConnectionError",
            "anthropic:APIConnectionError",
        ),
        TA_LLM_AUTH: _optional_types(
            "openai:AuthenticationError",
            "openai:PermissionDeniedError",
            "anthropic:AuthenticationError",
            "anthropic:PermissionDeniedError",
        ),
        TA_LLM_MODEL_NOT_FOUND: _optional_types("openai:NotFoundError", "anthropic:NotFoundError"),
        TA_LLM_RATE_LIMITED: _optional_types("openai:RateLimitError", "anthropic:RateLimitError"),
        TA_LLM_RESPONSE_INVALID: (ValidationError,)
        + _optional_types(
            "langchain_core.exceptions:OutputParserException",
            "openai:APIResponseValidationError",
            "anthropic:APIResponseValidationError",
        ),
    }


_STATUS_CODES = {
    401: TA_LLM_AUTH,
    403: TA_LLM_AUTH,
    404: TA_LLM_MODEL_NOT_FOUND,
    429: TA_LLM_RATE_LIMITED,
}


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc``, then what it was raised from or while handling, breadth first."""
    seen: set[int] = set()
    queue: list[BaseException] = [exc]
    while queue and len(seen) < _MAX_CHAIN:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        linked_list = [current.__cause__]
        if not current.__suppress_context__:  # ``raise ... from None`` hides the context
            linked_list.append(current.__context__)
        for linked in linked_list:
            if linked is not None and id(linked) not in seen:
                queue.append(linked)


def _module_of(exc: BaseException) -> str:
    module = getattr(type(exc), "__module__", None)
    return module if isinstance(module, str) else ""


def _in_modules(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def _from_llm_client(exc: BaseException) -> bool:
    """Whether ``exc``'s class comes from an LLM client (whose status we trust)."""
    module = _module_of(exc)
    return _in_modules(module, _LLM_CLIENT_MODULES) or module.startswith("langchain_")


def _from_llm_sdk(exc: BaseException) -> bool:
    module = _module_of(exc)
    return _in_modules(module, _LLM_SDK_MODULES) or module.startswith("langchain_")


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status an LLM client error carries, if any.

    ``status_code`` on the OpenAI and Anthropic SDKs' status errors (and the
    Ollama client's ``ResponseError``), ``response.status_code`` on
    ``httpx.HTTPStatusError`` and ``code`` on google-genai's ``APIError``.
    Errors from any other module carry no status here.
    """
    if not _from_llm_client(exc):
        return None
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    module = type(exc).__module__
    if module.startswith("httpx"):
        response_code = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(response_code, int):
            return response_code
    if module.startswith("google.genai"):
        genai_code = getattr(exc, "code", None)
        if isinstance(genai_code, int):
            return genai_code
    return None


def _raised_in_vendor(exc: BaseException) -> bool:
    """Whether a frame of ``exc``'s traceback runs in a data vendor module."""
    tb = exc.__traceback__
    for _ in range(_MAX_FRAMES):
        if tb is None:
            return False
        name = tb.tb_frame.f_globals.get("__name__")
        if isinstance(name, str) and _in_modules(name, _DATA_VENDOR_MODULES):
            return True
        tb = tb.tb_next
    return False


def _from_vendor(exc: BaseException, *, frames: bool = True) -> bool:
    """A vendor's error by type or class module, or (``frames``) by where it was raised."""
    return (
        isinstance(exc, _types()[TA_DATA_VENDOR_FAILED])
        or _in_modules(_module_of(exc), _DATA_VENDOR_MODULES)
        or (frames and _raised_in_vendor(exc))
    )


def _classify_one(exc: BaseException, *, vendor_frames: bool = True) -> str | None:
    """One exception's code by type or status, or None when nothing matches.

    The vendor rule comes first, so a vendor's ``requests`` timeout or 429 is
    not taken for the LLM's.
    """
    if _from_vendor(exc, frames=vendor_frames):
        return TA_DATA_VENDOR_FAILED
    types = _types()
    for code in (TA_LLM_TIMEOUT, TA_LLM_UNREACHABLE):
        if isinstance(exc, types[code]):
            return code
    status = _status_code(exc)
    if status in _STATUS_CODES:
        return _STATUS_CODES[status]
    for code in (TA_LLM_AUTH, TA_LLM_MODEL_NOT_FOUND, TA_LLM_RATE_LIMITED, TA_LLM_RESPONSE_INVALID):
        if isinstance(exc, types[code]):
            return code
    return None


def _says_model_not_found(exc: BaseException) -> bool:
    """Message text, the last resort: a "model ... not found" with no 404.

    Trusted only on an Ollama/OpenAI client's own error, so an unrelated
    ``RuntimeError("Pydantic model 'Decision' field rating not found")`` is no match.
    """
    if not _in_modules(_module_of(exc), _MODEL_TEXT_MODULES):
        return False
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ is just "no match"
        return False
    return bool(_MODEL_NOT_FOUND_TEXT.search(text[:2000]))


def classify_error(exc: BaseException) -> str:
    """The failure code for the exception that ended a run.

    The first exception in the chain (outermost first) that a type or status
    rule matches decides; when none does, a "model ... not found" message on an
    Ollama/OpenAI client error in the chain means ``TA_LLM_MODEL_NOT_FOUND``;
    anything else is ``TA_RUN_FAILED``. A vendor traceback frame counts only
    when no LLM SDK error is in the chain (an LLM call made from a vendor frame
    is still the LLM's failure). Pure: it never raises and has no side effects.
    """
    try:
        chain = list(_chain(exc))
        vendor_frames = not any(_from_llm_sdk(current) for current in chain)
        for current in chain:
            code = _classify_one(current, vendor_frames=vendor_frames)
            if code is not None:
                return code
        if any(_says_model_not_found(current) for current in chain):
            return TA_LLM_MODEL_NOT_FOUND
    except Exception:  # noqa: BLE001 - the worker records the failure regardless
        return TA_RUN_FAILED
    return TA_RUN_FAILED
