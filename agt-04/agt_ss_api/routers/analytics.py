"""
agt_ss_api.routers.analytics
------------------------------
Read-only analytics endpoints that surface AnalyticsGovernanceAgent outputs.

GET /analytics/dashboard          Aggregate procurement KPI dashboard
GET /analytics/ppv                Purchase price variance report
GET /analytics/savings-pipeline   Savings initiative pipeline
GET /analytics/contract-coverage  Contract coverage heatmap
GET /analytics/maturity           Procurement maturity roadmap
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status

from ..dependencies import AuthDep, CheckpointStoreDep
from ..models.schemas import ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])


def _get_analytics(
    store: CheckpointStoreDep,
    workflow_id: Optional[str],
    key: str,
) -> tuple[dict, str]:
    """
    Resolve analytics data from a specific workflow or the most recently
    completed workflow.

    Returns (data_dict, resolved_workflow_id).
    """
    if workflow_id:
        state = store.load(workflow_id)
        if state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found",
            )
    else:
        # Find the most recently completed workflow with analytics data
        all_states = store.list_all()
        completed = [
            s for s in all_states
            if s.get("status") == "completed" and s.get("analytics_governance")
        ]
        if not completed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "No completed workflow with analytics data found. "
                    "Provide a workflow_id query parameter to target a specific workflow."
                ),
            )
        state = completed[0]

    analytics = state.get("analytics_governance") or {}
    data = analytics.get(key)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Analytics field '{key}' not found in workflow "
                f"'{state.get('workflow_id')}'. "
                "The AnalyticsGovernanceAgent may not have run yet."
            ),
        )
    return data, state.get("workflow_id", "")


# ---------------------------------------------------------------------------
# GET /analytics/dashboard
# ---------------------------------------------------------------------------

@router.get(
    "/dashboard",
    response_model=dict[str, Any],
    responses={404: {"model": ErrorResponse}},
    summary="Monthly procurement dashboard",
    description=(
        "Returns the aggregate procurement KPI dashboard from the most recent "
        "completed workflow, or from a specific workflow if workflow_id is provided. "
        "Includes: total managed spend, savings realized, contract coverage %, "
        "PO compliance %, open sourcing events, and PPV total."
    ),
)
async def get_dashboard(
    store: CheckpointStoreDep,
    _auth: AuthDep,
    workflow_id: Optional[str] = Query(
        None, description="Target a specific workflow. Defaults to most recently completed."
    ),
) -> dict[str, Any]:
    data, wid = _get_analytics(store, workflow_id, "monthly_dashboard")
    return {"workflow_id": wid, **data}


# ---------------------------------------------------------------------------
# GET /analytics/ppv
# ---------------------------------------------------------------------------

@router.get(
    "/ppv",
    response_model=dict[str, Any],
    responses={404: {"model": ErrorResponse}},
    summary="Purchase price variance report",
    description=(
        "Returns the PPV report: actuals vs. budget and prior year by category "
        "and supplier. Variances are sorted by magnitude, largest unfavourable first."
    ),
)
async def get_ppv(
    store: CheckpointStoreDep,
    _auth: AuthDep,
    workflow_id: Optional[str] = Query(None),
    flag: Optional[str] = Query(
        None,
        description="Filter variances by flag: 'favourable' | 'unfavourable' | 'neutral'",
    ),
) -> dict[str, Any]:
    data, wid = _get_analytics(store, workflow_id, "ppv_report")

    if flag:
        variances = [v for v in (data.get("variances") or []) if v.get("flag") == flag]
        data = {**data, "variances": variances, "filtered_by": flag}

    return {"workflow_id": wid, **data}


# ---------------------------------------------------------------------------
# GET /analytics/savings-pipeline
# ---------------------------------------------------------------------------

@router.get(
    "/savings-pipeline",
    response_model=dict[str, Any],
    responses={404: {"model": ErrorResponse}},
    summary="Savings initiative pipeline",
    description=(
        "Returns all savings initiatives with stage, estimated value, confidence %, "
        "owner, and target close date. Initiatives span from identification through "
        "contracted and realised stages."
    ),
)
async def get_savings_pipeline(
    store: CheckpointStoreDep,
    _auth: AuthDep,
    workflow_id: Optional[str] = Query(None),
    stage: Optional[str] = Query(
        None,
        description=(
            "Filter by stage: identification | analysis | negotiation | contracted | realised"
        ),
    ),
) -> dict[str, Any]:
    pipeline, wid = _get_analytics(store, workflow_id, "savings_pipeline")

    items = pipeline if isinstance(pipeline, list) else []
    if stage:
        items = [i for i in items if i.get("stage") == stage]

    total_value = sum(i.get("estimated_savings", 0) for i in items)

    return {
        "workflow_id":  wid,
        "items":        items,
        "total":        len(items),
        "total_value":  round(total_value, 2),
        "filtered_by_stage": stage,
    }


# ---------------------------------------------------------------------------
# GET /analytics/contract-coverage
# ---------------------------------------------------------------------------

@router.get(
    "/contract-coverage",
    response_model=dict[str, Any],
    responses={404: {"model": ErrorResponse}},
    summary="Contract coverage heatmap",
    description=(
        "Returns the contract coverage heatmap by category: coverage %, "
        "covered and uncovered spend, risk classifications, and "
        "the list of high-risk uncovered categories."
    ),
)
async def get_contract_coverage(
    store: CheckpointStoreDep,
    _auth: AuthDep,
    workflow_id: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(
        None,
        description="Filter categories by risk level: 'low' | 'medium' | 'high'",
    ),
) -> dict[str, Any]:
    data, wid = _get_analytics(store, workflow_id, "contract_coverage_heatmap")

    categories = data.get("categories", {})
    if risk_level:
        categories = {
            cat: v for cat, v in categories.items()
            if v.get("risk_level") == risk_level
        }
        data = {**data, "categories": categories, "filtered_by_risk": risk_level}

    return {"workflow_id": wid, **data}


# ---------------------------------------------------------------------------
# GET /analytics/maturity
# ---------------------------------------------------------------------------

@router.get(
    "/maturity",
    response_model=dict[str, Any],
    responses={404: {"model": ErrorResponse}},
    summary="Procurement maturity roadmap",
    description=(
        "Returns the PCMM maturity assessment: current and target level, "
        "dimension scores across 10 procurement capability dimensions, "
        "transformation initiatives with effort and priority, and key gaps."
    ),
)
async def get_maturity_roadmap(
    store: CheckpointStoreDep,
    _auth: AuthDep,
    workflow_id: Optional[str] = Query(None),
    priority: Optional[str] = Query(
        None,
        description="Filter initiatives by priority: 'quick_win' | 'strategic' | 'foundational'",
    ),
) -> dict[str, Any]:
    data, wid = _get_analytics(store, workflow_id, "maturity_roadmap")

    if priority and "initiatives" in data:
        filtered_initiatives = [
            i for i in (data["initiatives"] or [])
            if i.get("priority") == priority
        ]
        data = {**data, "initiatives": filtered_initiatives, "filtered_by_priority": priority}

    return {"workflow_id": wid, **data}
