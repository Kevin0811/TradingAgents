"""Custom exception classes for the TradingAgents API."""

from __future__ import annotations


class TradingAgentsAPIError(Exception):
    """Base exception for TradingAgents API."""

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class AnalysisError(TradingAgentsAPIError):
    """Raised when trading analysis fails."""

    pass


class DataNotFoundError(TradingAgentsAPIError):
    """Raised when requested data is not found."""

    pass


class InvalidRequestError(TradingAgentsAPIError):
    """Raised when the request is invalid."""

    pass


class ExternalServiceError(TradingAgentsAPIError):
    """Raised when an external service (LLM, Data Provider) fails."""

    pass