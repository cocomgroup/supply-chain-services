# Databricks notebook source
"""
agt16.api
----------
Minimal HTTP API for AGT-16.

Routes
------
GET  /health                           liveness check
POST /workflows                        start a new workflow
GET  /workflows/{workflow_id}          retrieve current state (from S3 via RDS index)
POST /workflows/{workflow_id}/resume   resume a checkpoint-parked workflow

Invoked by ECS entry point (CMD in Dockerfile) and by AGT-00 Orchestrator
via REST calls. Uses Python's built-in http.server to keep the container
footprint minimal — no web framework dependency.

For production, swap http.server for FastAPI + uvicorn (same interface).
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from agt16.build_graph.orchestrator import run_workflow, resume_workflow
from agt16.calls.checkpoints.s3 import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "8080"))


class Agt16Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log spam
        logger.info(fmt, *args)

    # ── Routing ────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/health":
            self._json(200, {"status": "ok", "agent": "AGT-16"})

        elif path.startswith("/workflows/"):
            workflow_id = path.split("/workflows/")[1].split("/")[0]
            state = load_checkpoint(workflow_id)
            if state:
                self._json(200, {
                    "workflow_id": state.get("workflow_id"),
                    "status":      state.get("status"),
                    "mode":        state.get("mode"),
                    "engagement_id": state.get("engagement_id"),
                    "next_agent":  state.get("next_agent"),
                    "errors":      state.get("errors", []),
                    "final_output": state.get("final_output"),
                })
            else:
                self._json(404, {"error": f"workflow {workflow_id} not found"})

        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        body = self._read_body()

        if path == "/workflows":
            try:
                goal          = body.get("goal")
                if not goal:
                    self._json(400, {"error": "goal is required"})
                    return
                # Run in background thread for long workflows;
                # for now invoke synchronously (suitable for Fargate tasks)
                final_state = run_workflow(
                    goal=goal,
                    mode=body.get("mode", "engagement"),
                    engagement_id=body.get("engagement_id"),
                    client_id=body.get("client_id"),
                    kpi_targets=body.get("kpi_targets"),
                    context=body.get("context"),
                )
                self._json(200, {
                    "workflow_id":  final_state.get("workflow_id"),
                    "status":       final_state.get("status"),
                    "final_output": final_state.get("final_output"),
                    "errors":       final_state.get("errors", []),
                })
            except Exception as exc:
                logger.exception("Workflow execution failed")
                self._json(500, {"error": str(exc)})

        elif path.endswith("/resume"):
            # /workflows/{workflow_id}/resume
            parts       = path.split("/")
            workflow_id = parts[-2] if len(parts) >= 3 else None
            if not workflow_id:
                self._json(400, {"error": "workflow_id missing from path"})
                return
            try:
                final_state = resume_workflow(workflow_id, body)
                self._json(200, {
                    "workflow_id":  final_state.get("workflow_id"),
                    "status":       final_state.get("status"),
                    "final_output": final_state.get("final_output"),
                    "errors":       final_state.get("errors", []),
                })
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                logger.exception("Resume failed for %s", workflow_id)
                self._json(500, {"error": str(exc)})

        else:
            self._json(404, {"error": "not found"})

    # ── Helpers ────────────────────────────────────────────────────────────

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    logger.info("AGT-16 API starting on port %d", PORT)
    server = HTTPServer(("0.0.0.0", PORT), Agt16Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("AGT-16 API shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
