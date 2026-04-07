"""
agt_ss_api.routers.health
--------------------------
Health, readiness, and operational endpoints.

GET /health        Liveness probe — always 200 if service is up
GET /ready         Readiness probe — checks checkpoint store connectivity
GET /metrics       Operational metrics — active workflows, store stats
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ..background.runner import active_workflow_ids
from ..dependencies import CheckpointStoreDep, SettingsDep
from ..models.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Operations"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 OK if the service process is running. Used by ECS health checks.",
)
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 if the service is ready to serve requests. "
        "Checks that the checkpoint store is reachable. "
        "Returns 503 if the store is unavailable."
    ),
)
async def readiness(
    store: CheckpointStoreDep,
    settings: SettingsDep,
) -> JSONResponse:
    checks: dict[str, str] = {}
    ready = True

    # Checkpoint store: attempt a list (tolerates empty result)
    try:
        store.list_all()
        checks["checkpoint_store"] = "ok"
    except Exception as exc:
        checks["checkpoint_store"] = f"error: {exc}"
        ready = False

    http_status = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=http_status,
        content={
            "status":      "ready" if ready else "not_ready",
            "checks":      checks,
            "environment": settings.environment,
            "version":     settings.app_version,
            "timestamp":   datetime.utcnow().isoformat(),
        },
    )


@router.get(
    "/metrics",
    summary="Operational metrics",
    description=(
        "Returns current operational metrics: active background workflow count, "
        "total stored workflows, and breakdown by status."
    ),
)
async def metrics(
    store: CheckpointStoreDep,
    _settings: SettingsDep,
) -> JSONResponse:
    active_ids   = active_workflow_ids()
    all_states   = store.list_all()

    status_counts: dict[str, int] = {}
    for s in all_states:
        st = s.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    return JSONResponse(content={
        "active_workflows":   len(active_ids),
        "active_workflow_ids": active_ids,
        "total_workflows":    len(all_states),
        "by_status":          status_counts,
        "timestamp":          datetime.utcnow().isoformat(),
    })
