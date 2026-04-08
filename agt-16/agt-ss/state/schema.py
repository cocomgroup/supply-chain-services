# Databricks notebook source
"""
agt16.state.schema
-------------------
Canonical shared state for the AGT-16 Supply Chain Analytics & Intelligence
multi-agent system.

All agents read from and write to WorkflowState. The orchestrator owns the
top-level fields; agents own their output sub-dicts.

Persisted to S3 + RDS PostgreSQL via checkpoints/s3.py.
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
    """Lifecycle status of an analytics workflow."""
    PENDING        = "pending"
    RUNNING        = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED      = "completed"
    FAILED         = "failed"
    DEAD_LETTER    = "dead_letter"


class WorkflowMode(str, Enum):
    """Operational mode that triggered this workflow."""
    ENGAGEMENT  = "engagement"   # on-demand client analytics
    MONITORING  = "monitoring"   # scheduled market intelligence
    FIRM        = "firm"         # firm knowledge base / win-loss


class CheckpointGate(str, Enum):
    """Human-in-the-loop gates for high-stakes outputs."""
    BASELINE_APPROVAL          = "baseline_approval"
    FINANCIAL_MODEL_APPROVAL   = "financial_model_approval"
    BOARD_PRESENTATION_APPROVAL = "board_presentation_approval"


class AgentName(str, Enum):
    ORCHESTRATOR           = "orchestrator"
    DATA_INTEGRATION       = "data_integration"
    PERFORMANCE_BASELINE   = "performance_baseline"
    DASHBOARD_VIZ          = "dashboard_visualization"
    COST_ANALYTICS         = "cost_analytics"
    DIAGNOSTICS_RCA        = "diagnostics_rca"
    PREDICTIVE_ANALYTICS   = "predictive_analytics"
    MARKET_INTELLIGENCE    = "market_intelligence"
    ENGAGEMENT_REPORTING   = "engagement_reporting"
    MATURITY_ASSESSMENT    = "maturity_assessment"
    FIRM_KNOWLEDGE         = "firm_knowledge"


# ---------------------------------------------------------------------------
# Sub-state schemas (one per agent output domain)
# ---------------------------------------------------------------------------


class DataIntegrationOutput(TypedDict, total=False):
    ingestion_summary: dict          # {source, rows_loaded, quality_score, loaded_at}
    data_quality_report: dict        # {completeness_pct, freshness_ok, referential_ok}
    schema_map: dict                 # {source_table -> unified_field}
    s3_paths: dict                   # {bronze, silver, gold S3 prefixes}


class PerformanceBaselineOutput(TypedDict, total=False):
    kpi_baseline: dict               # {cost, service, inventory, quality, resilience}
    benchmark_comparison: dict       # {kpi -> {client, peer_median, best_in_class, gap}}
    peer_group: str
    baseline_completeness_pct: float


class DashboardVizOutput(TypedDict, total=False):
    dashboard_urls: list[dict]       # [{name, tool, url, refreshed_at}]
    exhibit_s3_keys: list[str]       # S3 keys for static PDF/PNG exhibits
    presentation_s3_key: str         # board deck S3 key


class CostAnalyticsOutput(TypedDict, total=False):
    cost_to_serve_matrix: dict       # {customer|channel|product -> landed_cost}
    financial_impact_models: list[dict]  # [{recommendation, npv, irr, payback_days}]
    savings_quantification: dict     # {total_identified, by_lever}


class DiagnosticsRCAOutput(TypedDict, total=False):
    performance_gaps: list[dict]     # [{kpi, gap_pct, severity}]
    root_causes: list[dict]          # [{gap, cause, evidence, confidence}]
    hypothesis_test_results: list[dict]


class PredictiveAnalyticsOutput(TypedDict, total=False):
    demand_forecast: dict            # {series, mape, horizon_days}
    risk_scores: list[dict]          # [{node, probability, severity}]
    capacity_forecast: dict
    cost_forecast: dict
    model_metadata: dict             # {model_id, trained_at, auc, mape}


class MarketIntelligenceOutput(TypedDict, total=False):
    commodity_prices: dict           # {commodity -> {price, delta_pct, source}}
    carrier_rates: dict              # {lane -> {rate, trend}}
    labor_cost_index: dict
    disruption_signals: list[dict]   # [{signal, severity, region, detected_at}]
    competitive_intel: list[dict]    # [{company, strategy_summary, source}]
    regulatory_trends: list[dict]    # [{regulation, impact, effective_date}]
    digest_s3_key: str


class EngagementReportingOutput(TypedDict, total=False):
    weekly_status_report: dict       # {milestone_summary, kpi_vs_target, health}
    deliverable_package_s3_key: str  # zipped exhibit package
    benefit_realization_report: dict # {realized_savings, vs_business_case_pct}
    report_s3_key: str


class MaturityAssessmentOutput(TypedDict, total=False):
    maturity_scores: dict            # {domain -> {score, best_practice_score, gap}}
    scorecard_s3_key: str
    improvement_roadmap: list[dict]  # [{domain, initiative, priority, timeline}]


class FirmKnowledgeOutput(TypedDict, total=False):
    knowledge_base_updated: bool
    artifacts_indexed: list[str]     # S3 keys indexed into pgvector
    win_loss_summary: dict           # {total_proposals, win_rate, avg_deal_size}
    benchmark_refresh_status: dict


# ---------------------------------------------------------------------------
# Human checkpoint record
# ---------------------------------------------------------------------------


class HumanCheckpointRecord(TypedDict, total=False):
    gate: str
    requested_at: str
    approved_at: Optional[str]
    approved_by: Optional[str]
    decision: Optional[str]          # "approved" | "rejected" | "modified"
    notes: Optional[str]
    payload: dict


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
    workflow_id: str
    workflow_name: str
    created_at: str
    updated_at: str

    # ── Control ───────────────────────────────────────────────────────────
    status: str                      # WorkflowStatus value
    mode: str                        # WorkflowMode value
    current_agent: str               # AgentName value
    task_graph: list[dict]           # [{step, agent, depends_on, status}]
    next_agent: Optional[str]
    iteration_count: int
    max_iterations: int

    # ── Engagement inputs ─────────────────────────────────────────────────
    goal: str                        # natural-language analytics goal
    engagement_id: Optional[str]
    client_id: Optional[str]
    kpi_targets: dict                # {kpi_name -> target_value}
    context: dict                    # arbitrary extra inputs from AGT-00

    # ── Human checkpoints ─────────────────────────────────────────────────
    pending_checkpoint: Optional[str]
    checkpoint_history: list[HumanCheckpointRecord]

    # ── Agent outputs ─────────────────────────────────────────────────────
    data_integration: DataIntegrationOutput
    performance_baseline: PerformanceBaselineOutput
    dashboard_visualization: DashboardVizOutput
    cost_analytics: CostAnalyticsOutput
    diagnostics_rca: DiagnosticsRCAOutput
    predictive_analytics: PredictiveAnalyticsOutput
    market_intelligence: MarketIntelligenceOutput
    engagement_reporting: EngagementReportingOutput
    maturity_assessment: MaturityAssessmentOutput
    firm_knowledge: FirmKnowledgeOutput

    # ── Error handling ────────────────────────────────────────────────────
    errors: list[str]
    dead_letters: list[DeadLetterRecord]
    retry_counts: dict               # {agent_name: int}

    # ── Audit log ─────────────────────────────────────────────────────────
    audit_log: list[dict]            # [{timestamp, agent, action, detail}]

    # ── Final outputs ─────────────────────────────────────────────────────
    final_output: Optional[dict]
    escalation_required: bool


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_initial_state(
    goal: str,
    mode: str = WorkflowMode.ENGAGEMENT,
    engagement_id: Optional[str] = None,
    client_id: Optional[str] = None,
    kpi_targets: Optional[dict] = None,
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
        mode=mode,
        current_agent=AgentName.ORCHESTRATOR,
        task_graph=[],
        next_agent=None,
        iteration_count=0,
        max_iterations=max_iterations,
        goal=goal,
        engagement_id=engagement_id,
        client_id=client_id,
        kpi_targets=kpi_targets or {},
        context=context or {},
        pending_checkpoint=None,
        checkpoint_history=[],
        data_integration={},
        performance_baseline={},
        dashboard_visualization={},
        cost_analytics={},
        diagnostics_rca={},
        predictive_analytics={},
        market_intelligence={},
        engagement_reporting={},
        maturity_assessment={},
        firm_knowledge={},
        errors=[],
        dead_letters=[],
        retry_counts={},
        audit_log=[],
        final_output=None,
        escalation_required=False,
    )
