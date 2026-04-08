# Databricks notebook source
"""
agt16.tools.rds.RDSClient
---------------------------
RDS PostgreSQL client for AGT-16.

Covers four sub-clients:
  1. EngagementConfigClient  — read engagement / source-system config
  2. BenchmarkClient         — Gartner / Hackett benchmark library
  3. AnalyticsQueryClient    — run analytical SQL against Gold-layer views
  4. VectorStoreClient       — pgvector upsert + semantic search

Credentials are resolved in priority order:
  1. AGT16_DB_SECRET_ARN  → AWS Secrets Manager (production)
  2. AGT16_DB_*           → environment variables (dev / CI)

Table DDL is in infrastructure/migrations/001_init.sql
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def _get_conn_params() -> dict:
    """Resolve RDS connection parameters from Secrets Manager or env vars."""
    secret_arn = os.getenv("AGT16_DB_SECRET_ARN")
    if secret_arn:
        import boto3
        sm  = boto3.client("secretsmanager",
                           region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        sec = json.loads(sm.get_secret_value(SecretId=secret_arn)["SecretString"])
        return {
            "host":     sec["host"],
            "port":     int(sec.get("port", 5432)),
            "dbname":   sec.get("dbname", "agt16"),
            "user":     sec["username"],
            "password": sec["password"],
        }
    return {
        "host":     os.getenv("AGT16_DB_HOST",     "localhost"),
        "port":     int(os.getenv("AGT16_DB_PORT", "5432")),
        "dbname":   os.getenv("AGT16_DB_NAME",     "agt16"),
        "user":     os.getenv("AGT16_DB_USER",     "agt16"),
        "password": os.getenv("AGT16_DB_PASSWORD", ""),
    }


@contextmanager
def _conn() -> Generator:
    """Context manager: open a psycopg2 connection, commit on exit, always close."""
    import psycopg2
    import psycopg2.extras
    params = _get_conn_params()
    conn   = psycopg2.connect(**params)
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. EngagementConfigClient
# ---------------------------------------------------------------------------

class EngagementConfigClient:
    """
    Read engagement configuration and source-system connectivity from RDS.

    Tables: engagements, source_systems, kpi_targets
    """

    def get_source_systems(self, engagement_id: str) -> list[dict]:
        """
        Return the list of source systems configured for an engagement.

        Schema: source_systems(engagement_id, system_type, system_name,
                               connection_secret_arn, active)
        """
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT system_type, system_name, connection_secret_arn
                    FROM   source_systems
                    WHERE  engagement_id = %s AND active = TRUE
                    ORDER  BY system_type
                    """,
                    (engagement_id,),
                )
                rows = cur.fetchall()

        return [
            {"type": r[0], "name": r[1], "connection_secret_arn": r[2]}
            for r in rows
        ]

    def get_engagement(self, engagement_id: str) -> Optional[dict]:
        """Return engagement metadata row."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT engagement_id, client_id, peer_group, status,
                           start_date, end_date, created_at
                    FROM   engagements
                    WHERE  engagement_id = %s
                    """,
                    (engagement_id,),
                )
                row = cur.fetchone()

        if not row:
            return None
        return {
            "engagement_id": row[0],
            "client_id":     row[1],
            "peer_group":    row[2],
            "status":        row[3],
            "start_date":    str(row[4]),
            "end_date":      str(row[5]) if row[5] else None,
            "created_at":    str(row[6]),
        }

    def get_kpi_targets(self, engagement_id: str) -> dict:
        """Return KPI target dict for an engagement."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT kpi_name, target_value
                    FROM   kpi_targets
                    WHERE  engagement_id = %s
                    """,
                    (engagement_id,),
                )
                rows = cur.fetchall()

        return {r[0]: r[1] for r in rows}

    def upsert_engagement(self, engagement: dict) -> None:
        """Insert or update an engagement record."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO engagements
                        (engagement_id, client_id, peer_group, status, start_date, end_date)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (engagement_id) DO UPDATE SET
                        client_id  = EXCLUDED.client_id,
                        peer_group = EXCLUDED.peer_group,
                        status     = EXCLUDED.status,
                        end_date   = EXCLUDED.end_date
                    """,
                    (
                        engagement["engagement_id"],
                        engagement.get("client_id"),
                        engagement.get("peer_group"),
                        engagement.get("status", "active"),
                        engagement.get("start_date"),
                        engagement.get("end_date"),
                    ),
                )


# ---------------------------------------------------------------------------
# 2. BenchmarkClient
# ---------------------------------------------------------------------------

class BenchmarkClient:
    """
    Read and refresh the firm's supply chain benchmark library.

    Table: benchmarks(peer_group, kpi_name, metric_median, metric_best_in_class,
                      source, survey_year, updated_at)
    """

    def get_benchmarks(self, peer_group: str) -> dict:
        """
        Return all benchmarks for a peer group as a nested dict:
        { kpi_name -> { median, best_in_class, source, survey_year } }
        """
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT kpi_name, metric_median, metric_best_in_class,
                           source, survey_year
                    FROM   benchmarks
                    WHERE  peer_group = %s
                    ORDER  BY kpi_name
                    """,
                    (peer_group,),
                )
                rows = cur.fetchall()

        result = {"peer_group": peer_group, "source": None, "gaps": {}}
        for kpi, median, bic, source, year in rows:
            result["source"] = source   # last row wins (all same source)
            result[kpi] = {"median": median, "best_in_class": bic,
                           "source": source, "survey_year": year}
        return result

    def upsert_benchmark(self, record: dict) -> None:
        """Upsert a single benchmark record (used during benchmark refresh)."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO benchmarks
                        (peer_group, kpi_name, metric_median, metric_best_in_class,
                         source, survey_year, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (peer_group, kpi_name) DO UPDATE SET
                        metric_median       = EXCLUDED.metric_median,
                        metric_best_in_class = EXCLUDED.metric_best_in_class,
                        source              = EXCLUDED.source,
                        survey_year         = EXCLUDED.survey_year,
                        updated_at          = EXCLUDED.updated_at
                    """,
                    (
                        record["peer_group"],
                        record["kpi_name"],
                        record.get("metric_median"),
                        record.get("metric_best_in_class"),
                        record.get("source"),
                        record.get("survey_year"),
                        datetime.utcnow().isoformat(),
                    ),
                )

    def refresh_from_api(self) -> dict:
        """
        Pull the latest Gartner / Hackett benchmark data and upsert into RDS.

        TODO: implement API calls to Gartner Data & Analytics REST API
              and Hackett Group benchmark data service. Requires firm API keys
              stored in Secrets Manager under AGT16_GARTNER_SECRET_ARN and
              AGT16_HACKETT_SECRET_ARN.
        """
        logger.info("[BenchmarkClient] refresh_from_api — TODO: wire Gartner/Hackett APIs")
        return {
            "status":       "not_implemented",
            "refreshed_at": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# 3. AnalyticsQueryClient
# ---------------------------------------------------------------------------

class AnalyticsQueryClient:
    """
    Run analytical SQL against Gold-layer materialized views in RDS.

    These views are populated by the ETL pipeline (DataIntegrationAgent → Gold).
    The Gold layer lands as Parquet on S3; the ETL job also writes summary
    aggregates into RDS for fast query by the analytics agents.

    Tables / views: gold_kpi_summary, gold_cost_to_serve, gold_order_lines,
                    gold_inventory_snapshots, gold_supplier_performance
    """

    def compute_baseline(self, engagement_id: str) -> dict:
        """
        Compute the 5-dimension KPI baseline from Gold-layer RDS views.
        Returns the kpi_baseline dict used by PerformanceBaselineAgent.
        """
        with _conn() as conn:
            with conn.cursor() as cur:

                # Cost dimension
                cur.execute(
                    """
                    SELECT
                        COALESCE(AVG(sc_cost_pct_revenue), 0)    AS sc_cost_pct,
                        COALESCE(AVG(logistics_cost_per_unit), 0) AS log_cost_pu,
                        COALESCE(AVG(inv_carrying_cost_pct), 0)   AS inv_carry_pct
                    FROM gold_kpi_summary
                    WHERE engagement_id = %s
                    """,
                    (engagement_id,),
                )
                cost_row = cur.fetchone() or (0.0, 0.0, 0.0)

                # Service dimension
                cur.execute(
                    """
                    SELECT
                        COALESCE(AVG(otif_pct), 0)          AS otif,
                        COALESCE(AVG(order_cycle_days), 0)  AS cycle_days,
                        COALESCE(AVG(fill_rate_pct), 0)     AS fill_rate
                    FROM gold_kpi_summary
                    WHERE engagement_id = %s
                    """,
                    (engagement_id,),
                )
                svc_row = cur.fetchone() or (0.0, 0.0, 0.0)

                # Inventory dimension
                cur.execute(
                    """
                    SELECT
                        COALESCE(AVG(inventory_turns), 0)   AS turns,
                        COALESCE(AVG(days_on_hand), 0)      AS doh,
                        COALESCE(AVG(eo_pct), 0)            AS eo
                    FROM gold_kpi_summary
                    WHERE engagement_id = %s
                    """,
                    (engagement_id,),
                )
                inv_row = cur.fetchone() or (0.0, 0.0, 0.0)

                # Quality dimension
                cur.execute(
                    """
                    SELECT
                        COALESCE(AVG(defect_rate_ppm), 0)  AS defects,
                        COALESCE(AVG(return_rate_pct), 0)  AS returns
                    FROM gold_kpi_summary
                    WHERE engagement_id = %s
                    """,
                    (engagement_id,),
                )
                qual_row = cur.fetchone() or (0.0, 0.0)

                # Resilience dimension
                cur.execute(
                    """
                    SELECT
                        COALESCE(AVG(supplier_hhi), 0)          AS hhi,
                        COALESCE(AVG(single_source_pct), 0)     AS single_src,
                        COALESCE(AVG(geo_concentration_pct), 0) AS geo_conc
                    FROM gold_kpi_summary
                    WHERE engagement_id = %s
                    """,
                    (engagement_id,),
                )
                res_row = cur.fetchone() or (0.0, 0.0, 0.0)

        return {
            "cost": {
                "total_supply_chain_cost_pct_revenue": float(cost_row[0]),
                "logistics_cost_per_unit":             float(cost_row[1]),
                "inventory_carrying_cost_pct":         float(cost_row[2]),
            },
            "service": {
                "otif_pct":            float(svc_row[0]),
                "order_cycle_time_days": float(svc_row[1]),
                "fill_rate_pct":       float(svc_row[2]),
            },
            "inventory": {
                "inventory_turns":          float(inv_row[0]),
                "days_on_hand":             float(inv_row[1]),
                "excess_and_obsolete_pct":  float(inv_row[2]),
            },
            "quality": {
                "supplier_defect_rate_ppm": float(qual_row[0]),
                "return_rate_pct":          float(qual_row[1]),
            },
            "resilience": {
                "supplier_concentration_hhi":  float(res_row[0]),
                "single_source_pct":           float(res_row[1]),
                "geographic_concentration_pct": float(res_row[2]),
            },
            "computed_at": datetime.utcnow().isoformat(),
        }

    def cost_to_serve(self, engagement_id: str) -> dict:
        """
        Compute landed cost decomposition from gold_cost_to_serve view.
        Returns { by_customer, by_channel, by_product }.
        """
        with _conn() as conn:
            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT customer_segment, AVG(landed_cost) AS avg_cost
                    FROM gold_cost_to_serve
                    WHERE engagement_id = %s
                    GROUP BY customer_segment
                    """,
                    (engagement_id,),
                )
                by_customer = {r[0]: float(r[1]) for r in cur.fetchall()}

                cur.execute(
                    """
                    SELECT channel, AVG(landed_cost) AS avg_cost
                    FROM gold_cost_to_serve
                    WHERE engagement_id = %s
                    GROUP BY channel
                    """,
                    (engagement_id,),
                )
                by_channel = {r[0]: float(r[1]) for r in cur.fetchall()}

                cur.execute(
                    """
                    SELECT sku, AVG(landed_cost) AS avg_cost
                    FROM gold_cost_to_serve
                    WHERE engagement_id = %s
                    GROUP BY sku
                    ORDER BY avg_cost DESC
                    LIMIT 50
                    """,
                    (engagement_id,),
                )
                by_product = {r[0]: float(r[1]) for r in cur.fetchall()}

        return {
            "by_customer": by_customer,
            "by_channel":  by_channel,
            "by_product":  by_product,
            "computed_at": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# 4. VectorStoreClient
# ---------------------------------------------------------------------------

class VectorStoreClient:
    """
    pgvector client for the firm knowledge base.

    Uses the pgvector extension for semantic search over indexed artifacts.

    Table: knowledge_artifacts(id, engagement_id, s3_key, content_text,
                                embedding vector(1536), metadata jsonb,
                                indexed_at timestamptz)

    Embeddings use text-embedding-3-small via the Anthropic / OpenAI client.
    """

    _EMBED_MODEL = "text-embedding-3-small"
    _EMBED_DIM   = 1536

    def __init__(self):
        self._openai = None   # lazy init

    def _get_openai(self):
        if self._openai is None:
            import openai
            self._openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai

    def embed(self, text: str) -> list[float]:
        """Return a 1536-dim embedding vector for the given text."""
        resp = self._get_openai().embeddings.create(
            model=self._EMBED_MODEL,
            input=text[:8000],   # safety truncation
        )
        return resp.data[0].embedding

    def upsert(self, s3_key: str, text: str, metadata: dict,
               engagement_id: str = "") -> None:
        """
        Embed text and upsert into the knowledge_artifacts table.

        On conflict (s3_key), update the embedding and metadata.
        """
        import psycopg2.extras
        vector = self.embed(text)

        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_artifacts
                        (engagement_id, s3_key, content_text, embedding, metadata, indexed_at)
                    VALUES (%s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (s3_key) DO UPDATE SET
                        content_text = EXCLUDED.content_text,
                        embedding    = EXCLUDED.embedding,
                        metadata     = EXCLUDED.metadata,
                        indexed_at   = EXCLUDED.indexed_at
                    """,
                    (
                        engagement_id,
                        s3_key,
                        text[:50_000],
                        str(vector),   # pgvector accepts '[x,y,z,...]' string
                        json.dumps(metadata),
                        datetime.utcnow().isoformat(),
                    ),
                )
        logger.debug("Indexed artifact: %s", s3_key)

    def search(self, query: str, top_k: int = 5,
               engagement_id: Optional[str] = None) -> list[dict]:
        """
        Semantic search over the knowledge base using cosine distance.

        Returns top_k matching artifacts with their s3_key, metadata, and
        similarity score.
        """
        query_vec = self.embed(query)

        filter_clause = "WHERE engagement_id = %s" if engagement_id else ""
        params        = [str(query_vec), top_k]
        if engagement_id:
            params.insert(1, engagement_id)

        sql = f"""
            SELECT s3_key,
                   content_text,
                   metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM   knowledge_artifacts
            {filter_clause}
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
        """
        # Rebuild params with query_vec duplicated for ORDER BY
        if engagement_id:
            params = [str(query_vec), engagement_id, str(query_vec), top_k]
        else:
            params = [str(query_vec), str(query_vec), top_k]

        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            {
                "s3_key":     r[0],
                "content":    r[1][:500],  # snippet
                "metadata":   json.loads(r[2]) if r[2] else {},
                "similarity": float(r[3]),
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# 5. ProposalClient
# ---------------------------------------------------------------------------

class ProposalClient:
    """
    Read and write proposal win/loss records.

    Table: proposals(proposal_id, client_name, deal_size, outcome,
                     loss_reason, competitor, submitted_at, decided_at)
    """

    def get_all_outcomes(self) -> list[dict]:
        """Return all proposal outcomes for win/loss analytics."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT proposal_id, client_name, deal_size, outcome,
                           loss_reason, competitor, submitted_at, decided_at
                    FROM proposals
                    ORDER BY submitted_at DESC
                    """
                )
                rows = cur.fetchall()

        return [
            {
                "proposal_id":  r[0],
                "client_name":  r[1],
                "deal_size":    float(r[2]) if r[2] else 0.0,
                "outcome":      r[3],
                "loss_reason":  r[4],
                "competitor":   r[5],
                "submitted_at": str(r[6]),
                "decided_at":   str(r[7]) if r[7] else None,
            }
            for r in rows
        ]

    def record_outcome(self, record: dict) -> None:
        """Insert a new proposal outcome record."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO proposals
                        (proposal_id, client_name, deal_size, outcome,
                         loss_reason, competitor, submitted_at, decided_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (proposal_id) DO UPDATE SET
                        outcome      = EXCLUDED.outcome,
                        loss_reason  = EXCLUDED.loss_reason,
                        competitor   = EXCLUDED.competitor,
                        decided_at   = EXCLUDED.decided_at
                    """,
                    (
                        record["proposal_id"],
                        record.get("client_name"),
                        record.get("deal_size"),
                        record.get("outcome"),
                        record.get("loss_reason"),
                        record.get("competitor"),
                        record.get("submitted_at"),
                        record.get("decided_at"),
                    ),
                )
