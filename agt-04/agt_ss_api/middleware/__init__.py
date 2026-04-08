"""
agt_ss_api.middleware
----------------------
Request lifecycle middleware:
  - Correlation ID injection (X-Correlation-ID header)
  - Structured access logging (method, path, status, latency)
  - Global exception handler producing consistent JSON error envelopes
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a correlation ID to every request.

    Reads X-Correlation-ID from the incoming request; generates a UUID4 if absent.
    Echoes the ID back in the response header so clients can trace requests.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with method, path, status code, and latency.
    Structured as key=value pairs for CloudWatch Logs Insights parsing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        correlation_id = getattr(request.state, "correlation_id", "-")
        logger.info(
            "method=%s path=%s status=%d latency_ms=%s correlation_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            correlation_id,
        )
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """
    Attach global exception handlers that return consistent JSON error envelopes.
    All unhandled exceptions produce:
      {"error": {"code": "...", "message": "...", "workflow_id": null}}
    """

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.warning("ValueError [%s]: %s", correlation_id, exc)
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INVALID_REQUEST", "message": str(exc)}},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.exception("Unhandled exception [%s]", correlation_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Check server logs.",
                    "correlation_id": correlation_id,
                }
            },
        )


def register_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI application."""
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(AccessLogMiddleware)
    register_exception_handlers(app)
