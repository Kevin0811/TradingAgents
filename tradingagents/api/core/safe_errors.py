"""Exception text that is safe to show a client or write to the log.

``str()`` of a pydantic ``ValidationError`` includes the input that failed
(``input_value=...``, the first and last characters of a long value). For an
LLM client that input is the client's keyword arguments, API key included, so
an error raised while a client is built (e.g. ChatGoogleGenerativeAI refusing
a temperature outside [0, 2]) can carry part of the key. The secret-file keys
are passed as ``SecretStr`` and so are masked anyway (see
``tradingagents.api.infrastructure.secret_keys``); this is the second line of
defence, for wherever the API turns such an error into an HTTP detail, a
task's error or a log line.
"""

from __future__ import annotations

from pydantic import ValidationError


def error_text(exc: BaseException) -> str:
    """``str(exc)``, except that a ValidationError omits the input values.

    A ValidationError becomes ``"<n> validation error(s) for <title>: <loc>:
    <msg>; ..."`` from ``errors(include_input=False, include_url=False)``
    (``ctx`` is left out too).
    """
    if not isinstance(exc, ValidationError):
        return str(exc)
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    parts = [
        f"{'.'.join(str(part) for part in error['loc']) or '(model)'}: {error['msg']}"
        for error in errors
    ]
    return f"{len(errors)} validation error(s) for {exc.title}: " + "; ".join(parts)


def traceback_is_safe(exc: BaseException) -> bool:
    """Whether a log line may include ``exc``'s traceback (``exc_info``).

    False when ``exc`` or an exception chained to it (``__cause__``,
    ``__context__``) is a ValidationError: the traceback prints each one's full
    ``str()``. Log ``error_text(exc)`` instead.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ValidationError):
            return False
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return True
