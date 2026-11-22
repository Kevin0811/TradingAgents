"""Custom exception classes for the TradingAgents API."""

from __future__ import annotations

from typing import Any


class TradingAgentsAPIError(Exception):
    """Base exception for TradingAgents API."""

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class AnalysisError(TradingAgentsAPIError):
    """Raised when trading analysis fails.

    ``token_usage`` is the failed run's usage so far (``partial`` true), as the
    ``token_usage`` field reports it; None when the run never got that far.
    """

    def __init__(
        self, message: str, detail: str | None = None, token_usage: dict[str, Any] | None = None
    ):
        super().__init__(message, detail)
        self.token_usage = token_usage


class DataNotFoundError(TradingAgentsAPIError):
    """Raised when requested data is not found."""

    pass


class InvalidRequestError(TradingAgentsAPIError):
    """Raised when the request is invalid."""

    pass


class UnprocessableRequestError(InvalidRequestError):
    """A request value the API refuses (HTTP 422), in the ``{error, detail}`` shape."""

    pass


class StateLogUnreadableError(TradingAgentsAPIError):
    """A saved state log that exists but cannot be read (HTTP 500, or 503 when retryable).

    Not an analysis failure: no run failed and there is no token usage. ``retryable``
    is true for an empty or truncated JSON file, which a run may still be writing.
    """

    def __init__(self, message: str, detail: str | None = None, *, retryable: bool = False):
        super().__init__(message, detail)
        self.retryable = retryable


class ExternalServiceError(TradingAgentsAPIError):
    """Raised when an external service (LLM, Data Provider) fails."""

    pass


class AnalysisCancelled(Exception):
    """Raised inside a running analysis once its task was cancelled.

    Deliberately not a ``TradingAgentsAPIError``: it ends a task as
    ``cancelled``, never as a failure or an HTTP error. As it leaves
    ``AnalysisService``, ``token_usage`` holds the run's usage so far.
    """

    def __init__(self, *args: Any, token_usage: dict[str, Any] | None = None):
        super().__init__(*args)
        self.token_usage = token_usage
