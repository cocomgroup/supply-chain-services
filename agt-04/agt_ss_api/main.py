# Databricks notebook source
"""
agt_ss_api.main
----------------
FastAPI application factory for the AGT-SS Strategic Sourcing API.

Entrypoint:
    uvicorn agt_ss_api.main:app --host 0.0.0.0 --port 8000

ECS task definition command:
    ["uvicorn", "agt_ss_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

Environment variables (see config.py for full list):
    ANTHROPIC_API_KEY   — Claude API key (from AWS Secrets Manager)
    AWS_REGION          — AWS region (e.g., us-east-1)
    DB_HOST             — Aurora PostgreSQL host (optional; omit for local JSON)
    DB_NAME             — Aurora database name (default: agt_ss)
    DISABLE_AUTH        — Set "true" to bypass API key auth in development
    API_KEYS            — Comma-separated valid API keys
    RUN_WORKFLOWS_ASYNC — "true" (default) for async execution; "false" to block
"""

from __future__ import annotations

import logging
import logging.config
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware import register_middleware
from .routers import analytics, health, workflows

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "format": (
                    "%(asctime)s level=%(levelname)s logger=%(name)s "
                    "message=%(message)s"
                )
            }
        },
        "handlers": {
            "console": {
                "class":     "logging.StreamHandler",
                "stream":    "ext://sys.stdout",
                "formatter": "structured",
            }
        },
        "root": {"level": level.upper(), "handlers": ["console"]},
        # Quiet noisy libraries
        "loggers": {
            "httpx":     {"level": "WARNING"},
            "httpcore":  {"level": "WARNING"},
            "uvicorn":   {"level": "INFO"},
        },
    })


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting %s v%s — environment=%s async=%s",
        settings.app_name,
        settings.app_version,
        settings.environment,
        settings.run_workflows_async,
    )

    if not settings.anthropic_api_key:
        logger.warning(
            "ANTHROPIC_API_KEY is not set. LLM calls will fail unless "
            "Amazon Bedrock fallback is configured."
        )

    if settings.is_production and settings.disable_auth:
        logger.error(
            "SECURITY WARNING: disable_auth=True in production environment. "
            "Set DISABLE_AUTH=false immediately."
        )

    yield

    logger.info("Shutting down %s", settings.app_name)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## AGT-SS Strategic Sourcing & Procurement Agent API

Manages all strategic sourcing and procurement functions through a five-agent
LangGraph pipeline:

- **SpendCategoryAgent** — spend analysis, Kraljic matrix, category strategy, tail spend
- **SupplierMarketAgent** — market intelligence, landed cost models, ASL management
- **SourcingExecutionAgent** — RFI/RFQ/RFP/auction execution, TCO models, bid evaluation
- **ContractSupplierAgent** — negotiation strategy, contract drafting, supplier onboarding
- **AnalyticsGovernanceAgent** — PPV tracking, savings pipeline, maturity roadmap

### Workflow lifecycle

1. `POST /workflows` — start a sourcing workflow (returns immediately in async mode)
2. `GET /workflows/{id}` — poll for progress
3. When `status = awaiting_human` → `GET /workflows/{id}/checkpoint` to see what needs review
4. `POST /workflows/{id}/resume` — approve or reject the checkpoint gate
5. When `status = completed` → `GET /workflows/{id}/final` for the output package

### Authentication

Include `X-API-Key: <key>` in all requests. Contact COCOM Group Technology for API key provisioning.
        """,
        openapi_tags=[
            {
                "name": "Workflows",
                "description": "Create, monitor, and control sourcing workflow execution.",
            },
            {
                "name": "Analytics",
                "description": "Access procurement analytics from completed workflows.",
            },
            {
                "name": "Operations",
                "description": "Health probes and operational metrics.",
            },
        ],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Custom middleware ──────────────────────────────────────────────────
    register_middleware(app)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(workflows.router, prefix="/api/v1")
    app.include_router(analytics.router, prefix="/api/v1")

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (imported by uvicorn)
# ---------------------------------------------------------------------------
app = create_app()