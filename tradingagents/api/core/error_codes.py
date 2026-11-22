"""Stable codes for why an analysis task failed or was cancelled.

``error`` and ``message`` stay free text; a client that wants to show its own
(localised) copy reads ``error_code`` on a failed task and ``cancel_reason`` on
a cancelled one instead. The codes are part of the API: add new ones, never
rename or reuse one.

:func:`classify_error` maps the exception that ended a run to a failure code.
It looks at exception types and HTTP status codes, walking the chain of
``__cause__`` / ``__context__`` (the API raises ``AnalysisError`` *from* the
core's exception, and the core wraps some errors itself). Message text is read
only as a last resort, for one case: a provider that reports a missing model
without a 404 (see :func:`_says_model_not_found`).
"""

from __future__ import annotations

import functools
import importlib
import re
from collections.abc import Iterator

from pydantic import ValidationError

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

# Exceptions in a chain looked at before giving up (a guard against odd cycles).
_MAX_CHAIN = 32

# Last resort only: Ollama (and servers like it) word a missing model as
# "model 'x' not found" / "model \"x\" not found, try pulling it first".
_MODEL_NOT_FOUND_TEXT = re.compile(r"\bmodel\b.{0,200}?\bnot found\b", re.IGNORECASE | re.DOTALL)


def _optional_types(*names: str) -> tuple[type[BaseException], ...]:
    """The named exception classes (``"module:Class"``) that can be imported.

    Provider SDKs are optional for the API: one that is not installed simply
    contributes nothing (its errors fall back to the status-code rules).
    """
    found: list[type[BaseException]] = []
    for name in names:
        module_name, _, attr = name.partition(":")
        try:
            cls = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
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
        for linked in (current.__cause__, current.__context__):
            if linked is not None and id(linked) not in seen:
                queue.append(linked)


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status a provider error carries, if any.

    ``status_code`` on the OpenAI and Anthropic SDKs' status errors (and the
    Ollama client's ``ResponseError``), ``response.status_code`` on
    ``httpx.HTTPStatusError`` and ``code`` on google-genai's ``APIError``.
    """
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


def _from_vendor(exc: BaseException) -> bool:
    module = type(exc).__module__
    return isinstance(exc, _types()[TA_DATA_VENDOR_FAILED]) or any(
        module == prefix or module.startswith(prefix + ".") for prefix in _DATA_VENDOR_MODULES
    )


def _classify_one(exc: BaseException) -> str | None:
    """One exception's code by type or status, or None when nothing matches."""
    if _from_vendor(exc):
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
    """Message text, the last resort: a "model ... not found" with no 404."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ is just "no match"
        return False
    return bool(_MODEL_NOT_FOUND_TEXT.search(text[:2000]))


def classify_error(exc: BaseException) -> str:
    """The failure code for the exception that ended a run.

    The first exception in the chain (outermost first) that a type or status
    rule matches decides; when none does, a "model ... not found" message
    anywhere in the chain means ``TA_LLM_MODEL_NOT_FOUND``; anything else is
    ``TA_RUN_FAILED``. Pure: it never raises and has no side effects.
    """
    try:
        chain = list(_chain(exc))
        for current in chain:
            code = _classify_one(current)
            if code is not None:
                return code
        if any(_says_model_not_found(current) for current in chain):
            return TA_LLM_MODEL_NOT_FOUND
    except Exception:  # noqa: BLE001 - the worker records the failure regardless
        return TA_RUN_FAILED
    return TA_RUN_FAILED
