"""Global exception handlers for FastAPI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Request, status
from fastapi.responses import JSONResponse

from tradingagents.api.core.exceptions import (
    AnalysisError,
    DataNotFoundError,
    ExternalServiceError,
    InvalidRequestError,
    TradingAgentsAPIError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


async def trading_agents_api_error_handler(
    request: Request, exc: TradingAgentsAPIError
) -> JSONResponse:
    """Handle TradingAgentsAPIError and subclasses."""
    status_code = status.HTTP_400_BAD_REQUEST
    if isinstance(exc, DataNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, AnalysisError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exc, ExternalServiceError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    logger.error(
        "TradingAgentsAPIError: %s (detail: %s) on %s %s",
        exc.message,
        exc.detail,
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=status_code,
        content={"error": exc.message, "detail": exc.detail},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception("Uncaught exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI app."""
    app.add_exception_handler(TradingAgentsAPIError, trading_agents_api_error_handler)
    app.add_exception_handler(AnalysisError, trading_agents_api_error_handler)
    app.add_exception_handler(DataNotFoundError, trading_agents_api_error_handler)
    app.add_exception_handler(ExternalServiceError, trading_agents_api_error_handler)
    app.add_exception_handler(InvalidRequestError, trading_agents_api_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)