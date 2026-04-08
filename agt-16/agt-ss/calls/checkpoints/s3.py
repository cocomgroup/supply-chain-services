# Databricks notebook source
"""
agt16.calls.checkpoints.s3
----------------------------
Checkpoint and dead-letter persistence backed by S3 (state JSON) + RDS
PostgreSQL (index + dead-letter table).

Falls back to local JSON files when AWS_DEFAULT_REGION is not configured
(dev mode) — identical fallback pattern to agt_ss checkpoints/delta.py.

Environment variables
---------------------
AWS_DEFAULT_REGION      AWS region (us-east-1 default)
AGT16_CHECKPOINT_BUCKET S3 bucket for checkpoint state blobs
AGT16_DB_SECRET_ARN     Secrets Manager ARN for RDS credentials
AGT16_DB_HOST           RDS endpoint (if not using Secrets Manager)
AGT16_DB_NAME           RDS database name (default: agt16)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEV_STORE = Path("/tmp/agt16_checkpoints")
_DEV_STORE.mkdir(exist_ok=True)

_AWS_CONFIGURED = bool(os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION"))
_S3_BUCKET      = os.getenv("AGT16_CHECKPOINT_BUCKET", "agt16-checkpoints")
_S3_PREFIX      = "workflow-state"
_DL_PREFIX      = "dead-letters"


# ---------------------------------------------------------------------------
# Lazy AWS client helpers
# ---------------------------------------------------------------------------

def _s3():
    import boto3
    return boto3.client("s3")


def _get_db_conn():
    """
    Return a psycopg2 connection using RDS credentials from Secrets Manager
    or environment variables (dev fallback).
    """
    import psycopg2

    secret_arn = os.getenv("AGT16_DB_SECRET_ARN")
    if secret_arn:
        import boto3, json as _json
        sm   = boto3.client("secretsmanager")
        sec  = _json.loads(sm.get_secret_value(SecretId=secret_arn)["SecretString"])
        return psycopg2.connect(
            host=sec["host"],
            port=sec.get("port", 5432),
            dbname=sec.get("dbname", "agt16"),
            user=sec["username"],
            password=sec["password"],
        )

    return psycopg2.connect(
        host=os.getenv("AGT16_DB_HOST", "localhost"),
        port=int(os.getenv("AGT16_DB_PORT", "5432")),
        dbname=os.getenv("AGT16_DB_NAME", "agt16"),
        user=os.getenv("AGT16_DB_USER", "agt16"),
        password=os.getenv("AGT16_DB_PASSWORD", ""),
    )


# ---------------------------------------------------------------------------
# Public API  (identical signatures to agt_ss checkpoints/delta.py)
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict) -> None:
    """Persist the full WorkflowState to S3 (and index in RDS)."""
    state = {**state, "updated_at": datetime.utcnow().isoformat()}
    if _AWS_CONFIGURED:
        _save_s3(state)
    else:
        _save_local(state)


def load_checkpoint(workflow_id: str) -> Optional[dict]:
    """Load the latest checkpoint for a given workflow_id."""
    if _AWS_CONFIGURED:
        return _load_s3(workflow_id)
    return _load_local(workflow_id)


def append_dead_letter(record: dict) -> None:
    """Write a failed tool-call record to the dead-letter store."""
    record = {**record, "timestamp": datetime.utcnow().isoformat()}
    if _AWS_CONFIGURED:
        _append_dead_letter_s3(record)
    else:
        path = _DEV_STORE / f"dead_letters_{record.get('agent', 'unknown')}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("Dead-letter written (dev): %s", path)


# ---------------------------------------------------------------------------
# S3 implementations
# ---------------------------------------------------------------------------

def _s3_key(workflow_id: str) -> str:
    return f"{_S3_PREFIX}/{workflow_id}/state.json"


def _save_s3(state: dict) -> None:
    try:
        s3  = _s3()
        key = _s3_key(state["workflow_id"])
        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=json.dumps(state).encode(),
            ContentType="application/json",
        )
        logger.info("Checkpoint saved to S3: s3://%s/%s", _S3_BUCKET, key)

        # Also upsert a lightweight index row in RDS for fast lookup / listing
        _upsert_rds_index(state)

    except Exception as exc:
        logger.warning("S3 checkpoint failed (%s); falling back to local", exc)
        _save_local(state)


def _load_s3(workflow_id: str) -> Optional[dict]:
    try:
        s3  = _s3()
        key = _s3_key(workflow_id)
        obj = s3.get_object(Bucket=_S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        logger.warning("S3 load failed (%s); falling back to local", exc)
        return _load_local(workflow_id)


def _append_dead_letter_s3(record: dict) -> None:
    """Write dead-letter record to S3 as newline-delimited JSON."""
    try:
        s3  = _s3()
        key = f"{_DL_PREFIX}/{record.get('agent', 'unknown')}/{record['timestamp']}.json"
        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=json.dumps(record).encode(),
            ContentType="application/json",
        )
        logger.info("Dead-letter written to S3: s3://%s/%s", _S3_BUCKET, key)

        # Also insert into RDS dead_letters table for operational querying
        _insert_rds_dead_letter(record)

    except Exception as exc:
        logger.warning("S3 dead-letter write failed (%s)", exc)


# ---------------------------------------------------------------------------
# RDS helpers
# ---------------------------------------------------------------------------

def _upsert_rds_index(state: dict) -> None:
    """
    Upsert a row in the workflow_checkpoints table.

    Table DDL (run once via migration):
        CREATE TABLE IF NOT EXISTS workflow_checkpoints (
            workflow_id  TEXT PRIMARY KEY,
            status       TEXT,
            mode         TEXT,
            engagement_id TEXT,
            updated_at   TIMESTAMPTZ,
            s3_key       TEXT
        );
    """
    try:
        conn = _get_db_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO workflow_checkpoints
                        (workflow_id, status, mode, engagement_id, updated_at, s3_key)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (workflow_id) DO UPDATE SET
                        status        = EXCLUDED.status,
                        mode          = EXCLUDED.mode,
                        engagement_id = EXCLUDED.engagement_id,
                        updated_at    = EXCLUDED.updated_at,
                        s3_key        = EXCLUDED.s3_key
                    """,
                    (
                        state["workflow_id"],
                        state.get("status"),
                        state.get("mode"),
                        state.get("engagement_id"),
                        state.get("updated_at"),
                        _s3_key(state["workflow_id"]),
                    ),
                )
        conn.close()
    except Exception as exc:
        logger.warning("RDS checkpoint index upsert failed: %s", exc)


def _insert_rds_dead_letter(record: dict) -> None:
    """
    Insert a row into the dead_letters table.

    Table DDL:
        CREATE TABLE IF NOT EXISTS dead_letters (
            id           BIGSERIAL PRIMARY KEY,
            workflow_id  TEXT,
            agent        TEXT,
            tool_call    TEXT,
            error        TEXT,
            attempt      INT,
            payload      JSONB,
            timestamp    TIMESTAMPTZ
        );
    """
    try:
        conn = _get_db_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dead_letters
                        (workflow_id, agent, tool_call, error, attempt, payload, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.get("workflow_id"),
                        record.get("agent"),
                        record.get("tool_call"),
                        record.get("error"),
                        record.get("attempt"),
                        json.dumps(record.get("payload", {})),
                        record.get("timestamp"),
                    ),
                )
        conn.close()
    except Exception as exc:
        logger.warning("RDS dead-letter insert failed: %s", exc)


# ---------------------------------------------------------------------------
# Local dev fallback  (identical to agt_ss delta.py local fallback)
# ---------------------------------------------------------------------------

def _save_local(state: dict) -> None:
    path = _DEV_STORE / f"{state['workflow_id']}.json"
    path.write_text(json.dumps(state, indent=2))
    logger.debug("Checkpoint saved (dev): %s", path)


def _load_local(workflow_id: str) -> Optional[dict]:
    path = _DEV_STORE / f"{workflow_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
