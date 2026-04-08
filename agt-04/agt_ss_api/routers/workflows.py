"""
agt_ss_api.routers.workflows
-----------------------------
All workflow lifecycle endpoints.

POST   /workflows                    Start a new sourcing workflow
GET    /workflows                    List all workflows (summaries)
GET    /workflows/{workflow_id}      Full workflow state and agent outputs
DELETE /workflows/{workflow_id}      Cancel a running workflow (best-effort)

POST   /workflows/{workflow_id}/resume
    Resume a workflow parked at a human checkpoint gate

GET    /workflows/{workflow_id}/checkpoint
    Return the pending checkpoint detail (payload sent to human for review)

GET    /workflows/{workflow_id}/audit
    Return the workflow audit log

GET    /workflows/{workflow_id}/final
    Return only the final_output dict (404 if not yet completed)
"""

from __future__ import annotations

import logging
import sys
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ..background.runner import (
    active_workflow_ids,
    cancel_workflow,
    is_running,
    submit_workflow,
)
from ..dependencies import AuthDep, CheckpointStoreDep, SettingsDep
from ..models.schemas import (
    AuditLogResponse,
    CheckpointDetailResponse,
    CheckpointRecord,
    CheckpointResumeRequest,
    ErrorResponse,
    FinalOutput,
    WorkflowCreateRequest,
    WorkflowDetailResponse,
    WorkflowListResponse,
    WorkflowStatusEnum,
    WorkflowSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ---------------------------------------------------------------------------
# Orchestrator import helper
# ---------------------------------------------------------------------------

def _get_orchestrator():
    """
    Import the orchestrator module.  Resolves the import path whether the
    agt_ss package is on sys.path directly or co-located with this service.
    """
    try:
        # Preferred: agt_ss installed as a package
        from agt_ss.build_graph.orchestrator import run_workflow, resume_workflow
        return run_workflow, resume_workflow
    except ImportError:
        pass

    # Fallback: look for agt_ss next to the api package
    api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent  = os.path.dirname(api_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from agt_ss.build_graph.orchestrator import run_workflow, resume_workflow
    return run_workflow, resume_workflow


# ---------------------------------------------------------------------------
# POST /workflows — start a new workflow
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=WorkflowDetailResponse,
    responses={
        202: {"description": "Workflow accepted and queued for execution"},
        400: {"model": ErrorResponse, "description": "Invalid request body"},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    },
    summary="Start a sourcing workflow",
    description=(
        "Starts a new AGT-SS sourcing workflow. "
        "When `run_workflows_async=true` (default), returns immediately with "
        "`status: pending` and the workflow runs in the background. "
        "Poll `GET /workflows/{workflow_id}` to track progress. "
        "When `run_workflows_async=false`, the request blocks until the workflow "
        "completes or parks at a human checkpoint gate."
    ),
)
async def create_workflow(
    body: WorkflowCreateRequest,
    settings: SettingsDep,
    store: CheckpointStoreDep,
    _auth: AuthDep,
) -> WorkflowDetailResponse:
    run_workflow, _ = _get_orchestrator()

    # Build kwargs for run_workflow
    kwargs = dict(
        goal=body.goal,
        category=body.category,
        budget=body.budget,
        timeline_days=body.timeline_days,
        context=body.context,
    )

    if settings.run_workflows_async:
        # Create a stub initial state so the workflow_id is available immediately
        from agt_ss.state.schema import create_initial_state, WorkflowStatus
        initial = create_initial_state(**kwargs)
        store.save(initial)

        workflow_id = initial["workflow_id"]

        def _on_complete(wid: str, final_state: Optional[dict]) -> None:
            if final_state:
                store.save(final_state)

        submit_workflow(
            workflow_id,
            run_workflow,
            on_complete=_on_complete,
            **kwargs,
        )

        logger.info("[API] Workflow %s submitted to background runner", workflow_id)
        return WorkflowDetailResponse.from_state(initial)

    else:
        # Synchronous — block until done or checkpoint
        final_state = run_workflow(**kwargs)
        store.save(final_state)
        logger.info("[API] Workflow %s completed synchronously — status=%s",
                    final_state.get("workflow_id"), final_state.get("status"))
        return WorkflowDetailResponse.from_state(final_state)


# ---------------------------------------------------------------------------
# GET /workflows — list all workflows
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=WorkflowListResponse,
    summary="List all workflows",
    description="Returns a paginated list of workflow summaries ordered by most recently updated.",
)
async def list_workflows(
    store: CheckpointStoreDep,
    _auth: AuthDep,
    status_filter: Optional[WorkflowStatusEnum] = Query(
        None, alias="status", description="Filter by workflow status"
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> WorkflowListResponse:
    all_states = store.list_all()

    if status_filter:
        all_states = [s for s in all_states if s.get("status") == status_filter.value]

    total   = len(all_states)
    page    = all_states[offset : offset + limit]
    summaries = [WorkflowSummary.from_state(s) for s in page]

    return WorkflowListResponse(items=summaries, total=total)


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id} — full workflow detail
# ---------------------------------------------------------------------------

@router.get(
    "/{workflow_id}",
    response_model=WorkflowDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get workflow detail",
    description=(
        "Returns the full workflow state including all agent outputs, "
        "checkpoint history, audit log, and errors. "
        "Poll this endpoint to track async workflow progress."
    ),
)
async def get_workflow(
    workflow_id: str,
    store: CheckpointStoreDep,
    _auth: AuthDep,
) -> WorkflowDetailResponse:
    state = store.load(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )

    # Annotate if the workflow is actively running in a background thread
    response = WorkflowDetailResponse.from_state(state)
    if is_running(workflow_id) and response.status == WorkflowStatusEnum.PENDING:
        response.status = WorkflowStatusEnum.RUNNING

    return response


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/resume — approve/reject a checkpoint gate
# ---------------------------------------------------------------------------

@router.post(
    "/{workflow_id}/resume",
    response_model=WorkflowDetailResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Workflow not awaiting human input"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
    summary="Resume a workflow at a checkpoint gate",
    description=(
        "Approve or reject a pending human checkpoint gate. "
        "If approved, the workflow resumes execution. "
        "If rejected, the workflow transitions to FAILED. "
        "When `run_workflows_async=true`, returns immediately and resumes in the background."
    ),
)
async def resume_workflow_endpoint(
    workflow_id: str,
    body: CheckpointResumeRequest,
    settings: SettingsDep,
    store: CheckpointStoreDep,
    _auth: AuthDep,
) -> WorkflowDetailResponse:
    # ── Guard checks first — before importing the orchestrator ────────────
    state = store.load(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )

    if state.get("status") != "awaiting_human":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Workflow '{workflow_id}' is not awaiting human input "
                f"(current status: {state.get('status')})"
            ),
        )

    # Validate the gate matches
    pending = state.get("pending_checkpoint")
    if pending and body.gate != pending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Checkpoint gate mismatch: workflow expects '{pending}' "
                f"but request specifies '{body.gate}'"
            ),
        )

    _, resume_workflow = _get_orchestrator()

    checkpoint_decision = {
        "gate":        body.gate,
        "decision":    body.decision.value,
        "approved_by": body.approved_by,
        "notes":       body.notes or "",
    }

    if settings.run_workflows_async:
        # Record the decision in state immediately so GET shows it
        from agt_ss.state.schema import WorkflowStatus
        now = datetime.utcnow().isoformat()
        record = {
            "gate":        body.gate,
            "requested_at": state.get("updated_at", now),
            "approved_at": now,
            "approved_by": body.approved_by,
            "decision":    body.decision.value,
            "notes":       body.notes or "",
            "payload":     {},
        }
        updated_state = {
            **state,
            "pending_checkpoint":  None,
            "status":              WorkflowStatus.RUNNING if body.decision.value != "rejected"
                                   else WorkflowStatus.FAILED,
            "checkpoint_history":  (state.get("checkpoint_history") or []) + [record],
            "updated_at":          now,
        }
        store.save(updated_state)

        if body.decision.value != "rejected":
            def _on_complete(wid: str, final_state: Optional[dict]) -> None:
                if final_state:
                    store.save(final_state)

            submit_workflow(
                workflow_id,
                resume_workflow,
                on_complete=_on_complete,
                workflow_id=workflow_id,
                checkpoint_decision=checkpoint_decision,
            )
            logger.info("[API] Workflow %s resumed in background — gate=%s decision=%s",
                        workflow_id, body.gate, body.decision.value)

        return WorkflowDetailResponse.from_state(updated_state)

    else:
        # Synchronous resume
        final_state = resume_workflow(
            workflow_id=workflow_id,
            checkpoint_decision=checkpoint_decision,
        )
        store.save(final_state)
        return WorkflowDetailResponse.from_state(final_state)


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/checkpoint — pending gate detail
# ---------------------------------------------------------------------------

@router.get(
    "/{workflow_id}/checkpoint",
    response_model=CheckpointDetailResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "No checkpoint is pending"},
    },
    summary="Get pending checkpoint detail",
    description=(
        "Returns the checkpoint payload that needs human review. "
        "Only available when the workflow status is `awaiting_human`."
    ),
)
async def get_checkpoint(
    workflow_id: str,
    store: CheckpointStoreDep,
    _auth: AuthDep,
) -> CheckpointDetailResponse:
    state = store.load(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    pending = state.get("pending_checkpoint")
    if not pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Workflow '{workflow_id}' has no pending checkpoint",
        )

    # Find the most recent checkpoint record in history
    history = state.get("checkpoint_history") or []
    pending_record = next(
        (r for r in reversed(history) if r.get("gate") == pending and r.get("decision") is None),
        {},
    )

    return CheckpointDetailResponse(
        workflow_id=workflow_id,
        gate=pending,
        requested_at=pending_record.get("requested_at", state.get("updated_at", "")),
        payload=pending_record.get("payload", {}),
        history=[
            CheckpointRecord(**{k: v for k, v in r.items()
                                if k in CheckpointRecord.model_fields})
            for r in history
        ],
    )


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/audit — audit log
# ---------------------------------------------------------------------------

@router.get(
    "/{workflow_id}/audit",
    response_model=AuditLogResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get workflow audit log",
    description="Returns the full audit log of all agent actions and orchestrator routing decisions.",
)
async def get_audit_log(
    workflow_id: str,
    store: CheckpointStoreDep,
    _auth: AuthDep,
    agent_filter: Optional[str] = Query(
        None, alias="agent", description="Filter entries by agent name"
    ),
) -> AuditLogResponse:
    state = store.load(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    from ..models.schemas import AuditEntry
    entries = [
        AuditEntry(**{k: v for k, v in e.items() if k in AuditEntry.model_fields})
        for e in (state.get("audit_log") or [])
    ]

    if agent_filter:
        entries = [e for e in entries if e.agent == agent_filter]

    return AuditLogResponse(
        workflow_id=workflow_id,
        entries=entries,
        total=len(entries),
    )


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/final — completed output only
# ---------------------------------------------------------------------------

@router.get(
    "/{workflow_id}/final",
    response_model=FinalOutput,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "Workflow not yet completed"},
    },
    summary="Get final workflow output",
    description=(
        "Returns only the final_output dict. "
        "Returns 404 if the workflow does not exist; "
        "returns 409 if the workflow has not yet reached COMPLETED status."
    ),
)
async def get_final_output(
    workflow_id: str,
    store: CheckpointStoreDep,
    _auth: AuthDep,
) -> FinalOutput:
    state = store.load(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    if state.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Workflow '{workflow_id}' is not completed "
                f"(status: {state.get('status')})"
            ),
        )

    fo = state.get("final_output") or {}
    return FinalOutput(**{k: v for k, v in fo.items() if k in FinalOutput.model_fields})


# ---------------------------------------------------------------------------
# DELETE /workflows/{workflow_id} — cancel
# ---------------------------------------------------------------------------

@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "Workflow cannot be cancelled (already running or done)"},
    },
    summary="Cancel a workflow",
    description=(
        "Cancels a workflow that is queued but not yet executing. "
        "Workflows that are actively running cannot be interrupted mid-execution."
    ),
)
async def cancel_workflow_endpoint(
    workflow_id: str,
    store: CheckpointStoreDep,
    _auth: AuthDep,
) -> None:
    state = store.load(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    cancelled = cancel_workflow(workflow_id)
    if not cancelled and is_running(workflow_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Workflow '{workflow_id}' is actively executing and cannot be cancelled",
        )
