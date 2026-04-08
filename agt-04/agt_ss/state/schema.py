# Databricks notebook source
"""
agt_ss.state.schema
-------------------
Canonical shared state for the AGT-SS Strategic Sourcing multi-agent system.
All agents read from and write to WorkflowState. The orchestrator owns the
top-level fields; agents own their output sub-dicts.

Persisted to Delta Lake via checkpoint.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WorkflowStatus(str, Enum):
    """Lifecycle status of a sourcing workflow."""
    PENDING       = "pending"
    RUNNING       = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED     = "completed"
    FAILED        = "failed"
    DEAD_LETTER   = "dead_letter"


class CheckpointGate(str, Enum):
    """The three mandatory human-in-the-loop gates."""
    CATEGORY_STRATEGY_APPROVAL   = "category_strategy_approval"
    NEGOTIATION_STRATEGY_APPROVAL = "negotiation_strategy_approval"
    CONTRACT_AWARD_APPROVAL       = "contract_award_approval"


class AgentName(str, Enum):
    ORCHESTRATOR          = "orchestrator"
    SPEND_CATEGORY        = "spend_category_intelligence"
    SUPPLIER_MARKET       = "supplier_market_intelligence"
    SOURCING_EXECUTION    = "sourcing_execution"
    CONTRACT_SUPPLIER     = "contract_supplier_management"
    ANALYTICS_GOVERNANCE  = "analytics_governance"


# ---------------------------------------------------------------------------
# Sub-state schemas (one per agent output domain)
# ---------------------------------------------------------------------------


class SpendCategoryOutput(TypedDict, total=False):
    category_strategies: list[dict]        # [{category, quadrant, strategy_doc, ...}]
    spend_classification: dict             # {category: total_spend, ...}
    tail_spend_report: dict
    make_vs_buy_recommendations: list[dict]
    kraljic_matrix: dict                   # {strategic:[...], leverage:[...], ...}


class SupplierMarketOutput(TypedDict, total=False):
    market_analysis_briefs: list[dict]     # [{category, market_structure, ...}]
    landed_cost_models: list[dict]
    approved_supplier_list: list[dict]     # [{supplier_id, name, status, ...}]
    supplier_shortlist: list[dict]         # candidates for sourcing event


class SourcingExecutionOutput(TypedDict, total=False):
    event_type: str                        # RFI | RFQ | RFP | reverse_auction
    event_packages: list[dict]
    bid_evaluation_matrix: dict
    tco_comparison: list[dict]             # [{supplier_id, tco_components, total}]
    award_recommendation: dict             # {supplier_id, rationale, tco_rank}
    supplier_scorecards: list[dict]


class ContractSupplierOutput(TypedDict, total=False):
    negotiation_strategy: dict             # {target_price, batna, concessions, clauses}
    contract_record: dict                  # {contract_id, status, executed_at, ...}
    onboarding_status: dict                # {supplier_id, edi_connected, erp_setup, ...}
    sustainability_ratings: list[dict]


class AnalyticsGovernanceOutput(TypedDict, total=False):
    ppv_report: dict                       # {variances: [...], total_savings_gap}
    monthly_dashboard: dict
    savings_pipeline: list[dict]
    contract_coverage_heatmap: dict
    maturity_roadmap: dict


# ---------------------------------------------------------------------------
# Human checkpoint record
# ---------------------------------------------------------------------------


class HumanCheckpointRecord(TypedDict, total=False):
    gate: str                              # CheckpointGate value
    requested_at: str                      # ISO datetime
    approved_at: Optional[str]
    approved_by: Optional[str]
    decision: Optional[str]               # "approved" | "rejected" | "modified"
    notes: Optional[str]
    payload: dict                          # the artifact sent for review


# ---------------------------------------------------------------------------
# Dead-letter / error record
# ---------------------------------------------------------------------------


class DeadLetterRecord(TypedDict, total=False):
    agent: str
    tool_call: str
    error: str
    attempt: int
    timestamp: str
    payload: dict


# ---------------------------------------------------------------------------
# Master workflow state
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict, total=False):
    # ── Identity ──────────────────────────────────────────────────────────
    workflow_id: str                        # UUID, set at creation
    workflow_name: str                      # human-readable goal description
    created_at: str
    updated_at: str

    # ── Control ───────────────────────────────────────────────────────────
    status: str                             # WorkflowStatus value
    current_agent: str                      # AgentName value
    task_graph: list[dict]                  # [{step, agent, depends_on, status}]
    next_agent: Optional[str]               # routing decision from supervisor
    iteration_count: int                    # guard against infinite loops
    max_iterations: int

    # ── Goal / inputs ─────────────────────────────────────────────────────
    goal: str                               # natural-language sourcing goal
    category: Optional[str]                 # primary spend category in scope
    budget: Optional[float]
    timeline_days: Optional[int]
    context: dict                           # arbitrary extra inputs from caller

    # ── Human checkpoints ─────────────────────────────────────────────────
    pending_checkpoint: Optional[str]       # CheckpointGate value, or None
    checkpoint_history: list[HumanCheckpointRecord]

    # ── Agent outputs ─────────────────────────────────────────────────────
    spend_category: SpendCategoryOutput
    supplier_market: SupplierMarketOutput
    sourcing_execution: SourcingExecutionOutput
    contract_supplier: ContractSupplierOutput
    analytics_governance: AnalyticsGovernanceOutput

    # ── Error handling ────────────────────────────────────────────────────
    errors: list[str]
    dead_letters: list[DeadLetterRecord]
    retry_counts: dict                      # {agent_name: int}

    # ── Audit log ─────────────────────────────────────────────────────────
    audit_log: list[dict]                   # [{timestamp, agent, action, detail}]

    # ── Final outputs ─────────────────────────────────────────────────────
    final_output: Optional[dict]
    escalation_required: bool


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_initial_state(
    goal: str,
    category: Optional[str] = None,
    budget: Optional[float] = None,
    timeline_days: Optional[int] = None,
    context: Optional[dict] = None,
    max_iterations: int = 50,
) -> WorkflowState:
    """Return a fresh WorkflowState ready for orchestrator intake."""
    now = datetime.utcnow().isoformat()
    return WorkflowState(
        workflow_id=str(uuid.uuid4()),
        workflow_name=goal[:80],
        created_at=now,
        updated_at=now,
        status=WorkflowStatus.PENDING,
        current_agent=AgentName.ORCHESTRATOR,
        task_graph=[],
        next_agent=None,
        iteration_count=0,
        max_iterations=max_iterations,
        goal=goal,
        category=category,
        budget=budget,
        timeline_days=timeline_days,
        context=context or {},
        pending_checkpoint=None,
        checkpoint_history=[],
        spend_category={},
        supplier_market={},
        sourcing_execution={},
        contract_supplier={},
        analytics_governance={},
        errors=[],
        dead_letters=[],
        retry_counts={},
        audit_log=[],
        final_output=None,
        escalation_required=False,
    )