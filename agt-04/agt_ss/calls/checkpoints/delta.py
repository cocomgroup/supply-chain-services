# Databricks notebook source
"""
agt_ss.checkpoints.delta
-------------------------
Checkpoint and dead-letter persistence backed by Delta Lake on Azure Databricks.
Falls back to local JSON files when DATABRICKS_HOST is not configured (dev mode).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEV_STORE = Path("/tmp/agt_ss_checkpoints")
_DEV_STORE.mkdir(exist_ok=True)

_DATABRICKS_CONFIGURED = bool(os.getenv("DATABRICKS_HOST"))


# ---------------------------------------------------------------------------
# Delta Lake helpers (only imported when running on Databricks)
# ---------------------------------------------------------------------------


def _get_spark():
    """Lazy import of SparkSession — only available inside Databricks."""
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession()
    except ImportError:
        return None


def _checkpoint_table() -> str:
    catalog = os.getenv("UNITY_CATALOG", "agt_ss")
    schema  = os.getenv("UNITY_SCHEMA",  "sourcing")
    return f"{catalog}.{schema}.workflow_checkpoints"


def _dead_letter_table() -> str:
    catalog = os.getenv("UNITY_CATALOG", "agt_ss")
    schema  = os.getenv("UNITY_SCHEMA",  "sourcing")
    return f"{catalog}.{schema}.dead_letters"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(state: dict) -> None:
    """Persist the full WorkflowState to Delta Lake (or local JSON in dev)."""
    state = {**state, "updated_at": datetime.utcnow().isoformat()}

    if _DATABRICKS_CONFIGURED:
        _save_delta(state)
    else:
        _save_local(state)


def load_checkpoint(workflow_id: str) -> Optional[dict]:
    """Load the latest checkpoint for a given workflow_id."""
    if _DATABRICKS_CONFIGURED:
        return _load_delta(workflow_id)
    return _load_local(workflow_id)


def append_dead_letter(record: dict) -> None:
    """Write a failed tool-call record to the dead-letter table."""
    record = {**record, "timestamp": datetime.utcnow().isoformat()}
    if _DATABRICKS_CONFIGURED:
        _append_dead_letter_delta(record)
    else:
        path = _DEV_STORE / f"dead_letters_{record.get('agent', 'unknown')}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("Dead-letter written (dev): %s", path)


# ---------------------------------------------------------------------------
# Delta Lake implementations
# ---------------------------------------------------------------------------


def _save_delta(state: dict) -> None:
    spark = _get_spark()
    if spark is None:
        logger.warning("SparkSession unavailable; falling back to local checkpoint.")
        _save_local(state)
        return

    import pyspark.sql.functions as F
    from pyspark.sql.types import StringType, StructField, StructType

    schema = StructType([
        StructField("workflow_id", StringType(), False),
        StructField("payload",     StringType(), False),
        StructField("updated_at",  StringType(), False),
    ])

    row = [(
        state["workflow_id"],
        json.dumps(state),
        state["updated_at"],
    )]
    df = spark.createDataFrame(row, schema)

    table = _checkpoint_table()
    (
        spark.read.table(table).alias("t")
        .merge(df.alias("s"), "t.workflow_id = s.workflow_id")
        .whenMatchedUpdate(set={"payload": "s.payload", "updated_at": "s.updated_at"})
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info("Checkpoint saved to Delta: %s / %s", table, state["workflow_id"])


def _load_delta(workflow_id: str) -> Optional[dict]:
    spark = _get_spark()
    if spark is None:
        return _load_local(workflow_id)

    table = _checkpoint_table()
    rows = (
        spark.read.table(table)
        .filter(f"workflow_id = '{workflow_id}'")
        .select("payload")
        .collect()
    )
    if not rows:
        return None
    return json.loads(rows[0]["payload"])


def _append_dead_letter_delta(record: dict) -> None:
    spark = _get_spark()
    if spark is None:
        return

    from pyspark.sql.types import StringType, StructField, StructType
    schema = StructType([
        StructField("workflow_id", StringType(), True),
        StructField("agent",       StringType(), True),
        StructField("payload",     StringType(), False),
        StructField("timestamp",   StringType(), False),
    ])
    row = [(
        record.get("workflow_id"),
        record.get("agent"),
        json.dumps(record),
        record["timestamp"],
    )]
    df = spark.createDataFrame(row, schema)
    df.write.mode("append").saveAsTable(_dead_letter_table())


# ---------------------------------------------------------------------------
# Local dev implementations
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