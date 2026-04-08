# Databricks notebook source
"""
agt16.tools.s3.S3DataClient
-----------------------------
S3 client for AGT-16 data lake operations.

Covers three usage patterns:
  1. Data lake I/O  — read/write Parquet across Bronze / Silver / Gold layers
  2. Deliverable I/O — upload PDF, PPTX, JSON, ZIP to the deliverables bucket
  3. Text extraction  — download and decode plain-text artifacts for embedding

Environment variables
---------------------
AGT16_DATA_BUCKET          S3 bucket for Bronze/Silver/Gold data layers
AGT16_DELIVERABLE_BUCKET   S3 bucket for client-facing deliverable artifacts
AWS_DEFAULT_REGION         AWS region (us-east-1 default)
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

_DATA_BUCKET        = os.getenv("AGT16_DATA_BUCKET",        "agt16-data")
_DELIVERABLE_BUCKET = os.getenv("AGT16_DELIVERABLE_BUCKET", "agt16-deliverables")


def _s3():
    import boto3
    return boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))


class S3DataClient:
    """
    Data lake client — Bronze / Silver / Gold layer operations.

    All paths are S3 URIs:  s3://<bucket>/<layer>/<engagement_id>/<source>/
    """

    def __init__(self):
        self._client = _s3()
        self._bucket = _DATA_BUCKET

    # ── Layer path helpers ─────────────────────────────────────────────────

    def bronze_prefix(self, engagement_id: str, source_type: str) -> str:
        return f"bronze/{engagement_id}/{source_type.lower()}/"

    def silver_prefix(self, engagement_id: str, source_type: str) -> str:
        return f"silver/{engagement_id}/{source_type.lower()}/"

    def gold_prefix(self, engagement_id: str, domain: str) -> str:
        return f"gold/{engagement_id}/{domain}/"

    # ── Write helpers ──────────────────────────────────────────────────────

    def put_json(self, key: str, data: Any) -> str:
        """
        Serialize data as JSON and upload to S3.
        Returns the full S3 URI.
        """
        body = json.dumps(data, default=str).encode()
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        uri = f"s3://{self._bucket}/{key}"
        logger.debug("Uploaded JSON to %s", uri)
        return uri

    def put_parquet(self, key: str, df) -> str:
        """
        Write a pandas DataFrame as Parquet to S3.
        Returns the full S3 URI.
        """
        import pandas as pd
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        uri = f"s3://{self._bucket}/{key}"
        logger.debug("Uploaded Parquet to %s (%d rows)", uri, len(df))
        return uri

    # ── Read helpers ───────────────────────────────────────────────────────

    def get_json(self, key: str) -> Any:
        """Download and parse a JSON object from S3."""
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        return json.loads(obj["Body"].read())

    def get_parquet(self, prefix: str):
        """
        Read all Parquet files under a given S3 prefix into a single DataFrame.
        """
        import pandas as pd

        paginator = self._client.get_paginator("list_objects_v2")
        dfs = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".parquet"):
                    continue
                body = self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
                dfs.append(pd.read_parquet(io.BytesIO(body)))

        if not dfs:
            import pandas as pd
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def read_text(self, s3_uri: str) -> str:
        """
        Download a text-based artifact (JSON, TXT, CSV) from an S3 URI
        and return its contents as a string.
        """
        key = s3_uri.replace(f"s3://{self._bucket}/", "")
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read().decode("utf-8", errors="replace")

    def list_keys(self, prefix: str) -> list[str]:
        """Return all object keys under a given prefix."""
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys


class S3DeliverableClient:
    """
    Deliverable client — upload client-facing artifacts to the deliverables bucket.
    """

    def __init__(self):
        self._client = _s3()
        self._bucket = _DELIVERABLE_BUCKET

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to S3. Returns the full S3 URI."""
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        uri = f"s3://{self._bucket}/{key}"
        logger.info("Deliverable uploaded: %s", uri)
        return uri

    def upload_text(self, key: str, text: str, content_type: str = "text/plain") -> str:
        """Upload a UTF-8 string as an S3 object. Returns the full S3 URI."""
        return self.upload_bytes(key, text.encode("utf-8"), content_type)

    def upload_digest(self, engagement_id: str, text: str) -> str:
        """Upload a market intelligence digest. Returns S3 URI."""
        date  = datetime.utcnow().date()
        key   = f"{engagement_id}/market-intelligence/digest_{date}.txt"
        return self.upload_text(key, text, "text/plain")

    def upload_report(self, engagement_id: str, report: dict) -> str:
        """Serialize and upload an engagement report JSON. Returns S3 URI."""
        week = datetime.utcnow().isocalendar().week
        key  = f"{engagement_id}/reports/weekly_status_w{week}.json"
        data = json.dumps(report, default=str).encode()
        return self.upload_bytes(key, data, "application/json")

    def zip_and_upload(self, engagement_id: str, s3_data_client: S3DataClient,
                       artifact_keys: list[str]) -> str:
        """
        Download a list of S3 artifact keys, zip them, and upload the package.
        Returns the S3 URI of the zip file.
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key in artifact_keys:
                try:
                    content = s3_data_client.read_text(key)
                    arcname = PurePosixPath(key).name
                    zf.writestr(arcname, content)
                except Exception as exc:
                    logger.warning("Could not add %s to zip: %s", key, exc)

        date = datetime.utcnow().date()
        key  = f"{engagement_id}/packages/deliverable_package_{date}.zip"
        return self.upload_bytes(buf.getvalue(), key, "application/zip")

    def presign_url(self, s3_uri: str, expiry_seconds: int = 3600) -> str:
        """Generate a pre-signed download URL for a deliverable."""
        key = s3_uri.replace(f"s3://{self._bucket}/", "")
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )
