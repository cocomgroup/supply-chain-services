# Databricks notebook source
"""
agt_ss_api.dependencies
------------------------
FastAPI dependency functions injected into route handlers.

Covers:
  - API key authentication
  - Checkpoint store access (Aurora PostgreSQL or local JSON fallback)
  - Workflow runner (sync or background)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

async def verify_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> str:
    """
    Validate the X-API-Key header.

    In dev mode (disable_auth=True or no keys configured) this is a no-op.
    In production, the key must match one of the configured api_keys.
    """
    if not settings.auth_enabled:
        return "dev"

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if x_api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return x_api_key


# Type alias for use in route signatures
AuthDep = Annotated[str, Depends(verify_api_key)]


# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------

class CheckpointStore:
    """
    Thin abstraction over the persistence layer.

    Uses Aurora PostgreSQL when DB_HOST is configured; falls back to the
    local JSON file store (agt_ss/calls/checkpoints/delta.py dev mode).
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._dev_dir  = Path(settings.checkpoint_dev_dir)
        self._dev_dir.mkdir(parents=True, exist_ok=True)
        self._use_db   = settings.db_configured

    # ── Public interface ──────────────────────────────────────────────────

    def load(self, workflow_id: str) -> Optional[dict]:
        """Load a workflow state by ID. Returns None if not found."""
        if self._use_db:
            return self._load_db(workflow_id)
        return self._load_local(workflow_id)

    def save(self, state: dict) -> None:
        """Persist a workflow state (upsert by workflow_id)."""
        if self._use_db:
            self._save_db(state)
        else:
            self._save_local(state)

    def list_all(self) -> list[dict]:
        """Return all persisted workflow states (summaries)."""
        if self._use_db:
            return self._list_db()
        return self._list_local()

    def exists(self, workflow_id: str) -> bool:
        return self.load(workflow_id) is not None

    # ── Aurora PostgreSQL ──────────────────────────────────────────────────

    def _load_db(self, workflow_id: str) -> Optional[dict]:
        """
        SELECT payload FROM agt_ss.workflow_checkpoints WHERE workflow_id = %s
        INTEGRATION POINT: replace with real psycopg2 / SQLAlchemy call.
        """
        logger.warning(
            "[CheckpointStore] Aurora not wired — falling back to local for %s", workflow_id
        )
        return self._load_local(workflow_id)

    def _save_db(self, state: dict) -> None:
        """
        INSERT ... ON CONFLICT (workflow_id) DO UPDATE SET payload = %s
        INTEGRATION POINT: replace with real psycopg2 / SQLAlchemy call.
        """
        logger.warning("[CheckpointStore] Aurora not wired — saving locally")
        self._save_local(state)

    def _list_db(self) -> list[dict]:
        """
        SELECT payload FROM agt_ss.workflow_checkpoints ORDER BY updated_at DESC
        INTEGRATION POINT: replace with real psycopg2 / SQLAlchemy call.
        """
        return self._list_local()

    # ── Local JSON fallback ────────────────────────────────────────────────

    def _save_local(self, state: dict) -> None:
        wid  = state.get("workflow_id")
        if not wid:
            return
        path = self._dev_dir / f"{wid}.json"
        path.write_text(json.dumps(state, indent=2, default=str))

    def _load_local(self, workflow_id: str) -> Optional[dict]:
        path = self._dev_dir / f"{workflow_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.error("[CheckpointStore] Failed to read %s: %s", path, exc)
            return None

    def _list_local(self) -> list[dict]:
        states = []
        for p in sorted(self._dev_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                states.append(json.loads(p.read_text()))
            except Exception:
                pass
        return states


def get_checkpoint_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CheckpointStore:
    return CheckpointStore(settings)


CheckpointStoreDep = Annotated[CheckpointStore, Depends(get_checkpoint_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]