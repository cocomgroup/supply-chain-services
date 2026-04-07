"""
agt_ss_api.models.schemas
--------------------------
Pydantic v2 request and response models for every AGT-SS API endpoint.

These models are the external contract — they are deliberately decoupled from
WorkflowState (the internal LangGraph dict) so the API shape can evolve
independently from the orchestrator internals.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared enums (mirrors state.schema but as string enums for OpenAPI clarity)
# ---------------------------------------------------------------------------

class WorkflowStatusEnum(str, Enum):
    PENDING        = "pending"
    RUNNING        = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED      = "completed"
    FAILED         = "failed"
    DEAD_LETTER    = "dead_letter"


class CheckpointDecisionEnum(str, Enum):
    APPROVED  = "approved"
    REJECTED  = "rejected"
    MODIFIED  = "modified"


class EventTypeEnum(str, Enum):
    RFI            = "RFI"
    RFQ            = "RFQ"
    RFP            = "RFP"
    REVERSE_AUCTION = "reverse_auction"


# ---------------------------------------------------------------------------
# POST /workflows — start a new sourcing workflow
# ---------------------------------------------------------------------------

class WorkflowCreateRequest(BaseModel):
    """Body for starting a new sourcing workflow."""

    goal: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Natural-language description of the sourcing objective.",
        examples=["Source titanium fasteners for F135 engine overhaul program"],
    )
    category: Optional[str] = Field(
        None,
        max_length=120,
        description="Primary spend category in scope. Defaults to 'all' if omitted.",
        examples=["Titanium Fasteners"],
    )
    budget: Optional[float] = Field(
        None,
        gt=0,
        description="Budget target in USD. Informs negotiation walk-away price.",
        examples=[500000.0],
    )
    timeline_days: Optional[int] = Field(
        None,
        gt=0,
        le=730,
        description="Desired completion timeline in calendar days.",
        examples=[60],
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary additional inputs consumed by sub-agents. "
            "See the context keys reference in the specification document."
        ),
        examples=[{"industry": "aerospace / defense", "annual_volume_units": 10000}],
    )

    @field_validator("goal")
    @classmethod
    def goal_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be blank")
        return v.strip()


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/resume — approve/reject a checkpoint gate
# ---------------------------------------------------------------------------

class CheckpointResumeRequest(BaseModel):
    """Body for resuming a workflow parked at a human checkpoint gate."""

    gate: str = Field(
        ...,
        description=(
            "The checkpoint gate being decided. Must match the gate value in the "
            "workflow's pending_checkpoint field. One of: "
            "category_strategy_approval | negotiation_strategy_approval | contract_award_approval"
        ),
        examples=["category_strategy_approval"],
    )
    decision: CheckpointDecisionEnum = Field(
        ...,
        description="Human decision on the checkpoint.",
    )
    approved_by: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Email address or name of the approver.",
        examples=["evan@cocomgroup.com"],
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional free-text notes or modification instructions.",
    )


# ---------------------------------------------------------------------------
# Shared sub-response models (nested inside WorkflowResponse)
# ---------------------------------------------------------------------------

class CheckpointRecord(BaseModel):
    gate: str
    requested_at: str
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    decision: Optional[str] = None
    notes: Optional[str] = None


class TaskStep(BaseModel):
    step: int
    agent: str
    depends_on: list[str]
    status: str


class AuditEntry(BaseModel):
    timestamp: str
    agent: str
    action: str
    detail: str


class DeadLetterEntry(BaseModel):
    agent: str
    tool_call: str
    error: str
    attempt: int
    timestamp: str


class FinalOutput(BaseModel):
    workflow_id: str
    goal: str
    category: Optional[str] = None
    spend_summary: Optional[dict[str, Any]] = None
    category_strategy: Optional[list[dict[str, Any]]] = None
    award: Optional[dict[str, Any]] = None
    contract_status: Optional[str] = None
    analytics: Optional[dict[str, Any]] = None
    completed_at: str


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id} — full workflow detail
# ---------------------------------------------------------------------------

class WorkflowDetailResponse(BaseModel):
    """Full workflow state returned by GET and after POST operations."""

    workflow_id: str
    workflow_name: str
    status: WorkflowStatusEnum
    current_agent: Optional[str] = None
    pending_checkpoint: Optional[str] = None

    goal: str
    category: Optional[str] = None
    budget: Optional[float] = None
    timeline_days: Optional[int] = None

    created_at: str
    updated_at: str

    task_graph: list[TaskStep] = Field(default_factory=list)
    checkpoint_history: list[CheckpointRecord] = Field(default_factory=list)
    audit_log: list[AuditEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    dead_letters: list[DeadLetterEntry] = Field(default_factory=list)
    escalation_required: bool = False
    iteration_count: int = 0

    # Agent outputs (present when respective agent has run)
    spend_category: dict[str, Any] = Field(default_factory=dict)
    supplier_market: dict[str, Any] = Field(default_factory=dict)
    sourcing_execution: dict[str, Any] = Field(default_factory=dict)
    contract_supplier: dict[str, Any] = Field(default_factory=dict)
    analytics_governance: dict[str, Any] = Field(default_factory=dict)

    final_output: Optional[FinalOutput] = None

    @classmethod
    def from_state(cls, state: dict) -> "WorkflowDetailResponse":
        """Construct from a raw WorkflowState dict."""
        # Coerce task_graph and history lists safely
        task_graph = [
            TaskStep(
                step=t.get("step", 0),
                agent=str(t.get("agent", "")),
                depends_on=[str(d) for d in t.get("depends_on", [])],
                status=t.get("status", "pending"),
            )
            for t in (state.get("task_graph") or [])
        ]
        checkpoint_history = [
            CheckpointRecord(**{k: v for k, v in c.items()
                                if k in CheckpointRecord.model_fields})
            for c in (state.get("checkpoint_history") or [])
        ]
        audit_log = [
            AuditEntry(**{k: v for k, v in e.items()
                          if k in AuditEntry.model_fields})
            for e in (state.get("audit_log") or [])
        ]
        dead_letters = [
            DeadLetterEntry(**{k: v for k, v in d.items()
                               if k in DeadLetterEntry.model_fields})
            for d in (state.get("dead_letters") or [])
        ]
        final_output = None
        if fo := state.get("final_output"):
            final_output = FinalOutput(**{k: v for k, v in fo.items()
                                          if k in FinalOutput.model_fields})

        return cls(
            workflow_id=state.get("workflow_id", ""),
            workflow_name=state.get("workflow_name", ""),
            status=WorkflowStatusEnum(state.get("status", "pending")),
            current_agent=state.get("current_agent"),
            pending_checkpoint=state.get("pending_checkpoint"),
            goal=state.get("goal", ""),
            category=state.get("category"),
            budget=state.get("budget"),
            timeline_days=state.get("timeline_days"),
            created_at=state.get("created_at", ""),
            updated_at=state.get("updated_at", ""),
            task_graph=task_graph,
            checkpoint_history=checkpoint_history,
            audit_log=audit_log,
            errors=state.get("errors") or [],
            dead_letters=dead_letters,
            escalation_required=state.get("escalation_required", False),
            iteration_count=state.get("iteration_count", 0),
            spend_category=state.get("spend_category") or {},
            supplier_market=state.get("supplier_market") or {},
            sourcing_execution=state.get("sourcing_execution") or {},
            contract_supplier=state.get("contract_supplier") or {},
            analytics_governance=state.get("analytics_governance") or {},
            final_output=final_output,
        )


# ---------------------------------------------------------------------------
# GET /workflows — list summary
# ---------------------------------------------------------------------------

class WorkflowSummary(BaseModel):
    """Lightweight summary for list responses."""
    workflow_id: str
    workflow_name: str
    status: WorkflowStatusEnum
    goal: str
    category: Optional[str] = None
    pending_checkpoint: Optional[str] = None
    created_at: str
    updated_at: str

    @classmethod
    def from_state(cls, state: dict) -> "WorkflowSummary":
        return cls(
            workflow_id=state.get("workflow_id", ""),
            workflow_name=state.get("workflow_name", ""),
            status=WorkflowStatusEnum(state.get("status", "pending")),
            goal=state.get("goal", ""),
            category=state.get("category"),
            pending_checkpoint=state.get("pending_checkpoint"),
            created_at=state.get("created_at", ""),
            updated_at=state.get("updated_at", ""),
        )


class WorkflowListResponse(BaseModel):
    items: list[WorkflowSummary]
    total: int


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/checkpoint — pending gate detail
# ---------------------------------------------------------------------------

class CheckpointDetailResponse(BaseModel):
    workflow_id: str
    gate: str
    requested_at: str
    payload: dict[str, Any] = Field(default_factory=dict)
    history: list[CheckpointRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/audit — audit log
# ---------------------------------------------------------------------------

class AuditLogResponse(BaseModel):
    workflow_id: str
    entries: list[AuditEntry]
    total: int


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    workflow_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
