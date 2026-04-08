# Databricks notebook source
"""
agt_ss_api.config
------------------
Environment-driven configuration for the AGT-SS FastAPI service.
All settings can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service identity ──────────────────────────────────────────────────
    app_name: str = "AGT-SS Strategic Sourcing API"
    app_version: str = "1.0.0"
    environment: str = Field("development", description="development | staging | production")
    debug: bool = Field(False, description="Enable debug mode and verbose logging")

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"

    # ── CORS ──────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins. In production, restrict to specific domains.",
    )
    cors_allow_credentials: bool = True

    # ── Auth ──────────────────────────────────────────────────────────────
    api_key_header: str = "X-API-Key"
    api_keys: list[str] = Field(
        default=[],
        description=(
            "Valid API keys. If empty, authentication is disabled (dev mode). "
            "In production, set via AWS Secrets Manager and inject at container start."
        ),
    )
    disable_auth: bool = Field(
        False,
        description="Bypass API key authentication. Never set True in production.",
    )

    # ── AWS ───────────────────────────────────────────────────────────────
    aws_region: str = Field("us-east-1", description="AWS region for all service calls")
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # ── Persistence (Aurora PostgreSQL) ───────────────────────────────────
    db_host: Optional[str] = Field(None, description="Aurora PostgreSQL host")
    db_port: int = 5432
    db_name: str = "agt_ss"
    db_schema: str = "public"
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_pool_size: int = 5
    db_pool_max_overflow: int = 10

    # ── Background execution ──────────────────────────────────────────────
    run_workflows_async: bool = Field(
        True,
        description=(
            "When True, POST /workflows returns immediately with workflow_id "
            "and runs the graph in a background thread. "
            "When False, the request blocks until the workflow completes or parks. "
            "Set False for development/testing."
        ),
    )
    workflow_timeout_seconds: int = Field(
        1800,
        description="Maximum wall-clock seconds for a synchronous workflow run.",
    )

    # ── LLM ───────────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = Field(
        None,
        description="Claude API key. Sourced from AWS Secrets Manager in production.",
    )

    # ── Checkpoint store ──────────────────────────────────────────────────
    checkpoint_dev_dir: str = "/tmp/agt_ss_checkpoints"

    # ── Rate limiting ─────────────────────────────────────────────────────
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 60

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def db_configured(self) -> bool:
        return bool(self.db_host and self.db_user)

    @property
    def auth_enabled(self) -> bool:
        return not self.disable_auth and len(self.api_keys) > 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()