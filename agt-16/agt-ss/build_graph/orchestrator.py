# Databricks notebook source
"""
agt16.build_graph.orchestrator
--------------------------------
AGT-16 Orchestrator — the LangGraph StateGraph supervisor for the Supply
Chain Analytics & Intelligence multi-agent system.

Graph topology
--------------

  [intake]
     │
     ▼
  [supervisor]  ◄─────────────────────────────────────────────────────┐
     │                                                                  │
     ├──► [data_integration]      (CM-01 — runs first, no deps)        │
     │         │                                                        │
     ├──► [performance_baseline]  (CM-02 — depends on data_integration)│
     │         │                                                        │
     ├──► [dashboard_viz]         (CM-03 — depends on baseline)        │
     │                                                                  │
     ├──► [cost_analytics]        (CM-04 — depends on baseline)        │
     │                                                                  │
     ├──► [diagnostics_rca]       (CM-05 — depends on baseline)        │
     │                                                                  │
     ├──► [predictive_analytics]  (CM-06 — depends on data + baseline) │
     │                                                                  │
     ├──► [market_intelligence]   (CM-07 — standalone, no deps)        │
     │                                                                  │
     ├──► [engagement_reporting]  (CM-08 — depends on data + baseline) │
     │         │                                                        │
     │   [checkpoint: baseline_approval]                                │
     │   [checkpoint: financial_model_approval]                         │
     │   [checkpoint: board_presentation_approval]                      │
     │                                                                  │
     ├──► [maturity_assessment]   (CM-09 — optional, context-triggered)│
     │                                                                  │
     ├──► [firm_knowledge]        (CM-10 — post-engagement / firm mode)│
     │                                                                  │
     └──► [finalize] ───────────────────────────────────────────────►  │
                │
              [END]

Human checkpoint gates are blocking: the graph parks in AWAITING_HUMAN
and must be resumed with an approved/rejected decision before progressing.

Dead-letter handling: after MAX_RETRIES the failed agent's record is written
to S3 + RDS dead-letter store and the workflow transitions to DEAD_LETTER.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from langgraph.graph import END, StateGraph

from ..agents.base import AgentToolError
from ..agents import (
    DataIntegrationAgent,
    PerformanceBaselineAgent,
    DashboardVizAgent,
    CostAnalyticsAgent,
    DiagnosticsRCAAgent,
    PredictiveAnalyticsAgent,
    MarketIntelligenceAgent,
    EngagementReportingAgent,
    MaturityAssessmentAgent,
    FirmKnowledgeAgent,
)
from ..calls.checkpoints.s3 import append_dead_letter, save_checkpoint, load_checkpoint
from ..state.schema import (
    AgentName,
    CheckpointGate,
    WorkflowState,
    WorkflowStatus,
    create_initial_state,
)

logger = logging.getLogger(__name__)

SUPERVISOR_ITERATION_LIMIT = 50

# ── Instantiate agents ────────────────────────────────────────────────────

_AGENTS = {
    AgentName.DATA_INTEGRATION:     DataIntegrationAgent(),
    AgentName.PERFORMANCE_BASELINE: PerformanceBaselineAgent(),
    AgentName.DASHBOARD_VIZ:        DashboardVizAgent(),
    AgentName.COST_ANALYTICS:       CostAnalyticsAgent(),
    AgentName.DIAGNOSTICS_RCA:      DiagnosticsRCAAgent(),
    AgentName.PREDICTIVE_ANALYTICS: PredictiveAnalyticsAgent(),
    AgentName.MARKET_INTELLIGENCE:  MarketIntelligenceAgent(),
    AgentName.ENGAGEMENT_REPORTING: EngagementReportingAgent(),
    AgentName.MATURITY_ASSESSMENT:  MaturityAssessmentAgent(),
    AgentName.FIRM_KNOWLEDGE:       FirmKnowledgeAgent(),
}


# ============================================================================
# Node functions
# ============================================================================

def node_intake(state: WorkflowState) -> WorkflowState:
    """
    Validate incoming goal, set initial task graph, transition to RUNNING.
    """
    logger.info("[Orchestrator] Intake: %s", state.get("goal", "(no goal)"))

    if not state.get("goal"):
        return {
            **state,
            "status": WorkflowStatus.FAILED,
            "errors": state.get("errors", []) + ["No goal provided to orchestrator."],
        }

    task_graph = _build_task_graph(state)
    updated = {
        **state,
        "status":          WorkflowStatus.RUNNING,
        "task_graph":      task_graph,
        "iteration_count": 0,
        "updated_at":      datetime.utcnow().isoformat(),
    }
    _append_audit(updated, AgentName.ORCHESTRATOR, "intake",
                  f"goal={state.get('goal', '')[:60]}")
    save_checkpoint(updated)
    return updated


def node_supervisor(state: WorkflowState) -> WorkflowState:
    """
    Supervisor node: inspect current state and decide the next agent to invoke.

    Routing priority:
      1. If iteration limit hit → escalate
      2. If pending human checkpoint → park in AWAITING_HUMAN
      3. Run agents in dependency order based on can_run() checks
      4. If all agents done → finalize
    """
    count = state.get("iteration_count", 0) + 1

    if count > SUPERVISOR_ITERATION_LIMIT:
        logger.error("[Supervisor] Iteration limit reached — escalating.")
        updated = {
            **state,
            "iteration_count":     count,
            "status":              WorkflowStatus.FAILED,
            "escalation_required": True,
            "errors":              state.get("errors", []) + ["Iteration limit exceeded."],
        }
        save_checkpoint(updated)
        return updated

    if state.get("pending_checkpoint"):
        logger.info("[Supervisor] Parking — pending checkpoint: %s",
                    state["pending_checkpoint"])
        return {
            **state,
            "iteration_count": count,
            "status":          WorkflowStatus.AWAITING_HUMAN,
        }

    next_agent = _route(state)
    updated = {
        **state,
        "iteration_count": count,
        "next_agent":       next_agent,
        "updated_at":       datetime.utcnow().isoformat(),
    }
    _append_audit(updated, AgentName.ORCHESTRATOR, "route", f"next={next_agent}")
    save_checkpoint(updated)
    return updated


def node_data_integration(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.DATA_INTEGRATION)


def node_performance_baseline(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.PERFORMANCE_BASELINE)


def node_dashboard_viz(state: WorkflowState) -> WorkflowState:
    state = _run_agent(state, AgentName.DASHBOARD_VIZ)
    # Raise board presentation gate after first dashboard build
    if not _gate_approved(state, CheckpointGate.BOARD_PRESENTATION_APPROVAL):
        state = _raise_checkpoint(
            state,
            CheckpointGate.BOARD_PRESENTATION_APPROVAL,
            {"presentation_s3_key": state.get("dashboard_visualization", {})
             .get("presentation_s3_key")},
        )
    return state


def node_cost_analytics(state: WorkflowState) -> WorkflowState:
    state = _run_agent(state, AgentName.COST_ANALYTICS)
    # Raise financial model gate after cost analytics completes
    if not _gate_approved(state, CheckpointGate.FINANCIAL_MODEL_APPROVAL):
        state = _raise_checkpoint(
            state,
            CheckpointGate.FINANCIAL_MODEL_APPROVAL,
            state.get("cost_analytics", {}).get("financial_impact_models", []),
        )
    return state


def node_diagnostics_rca(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.DIAGNOSTICS_RCA)


def node_predictive_analytics(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.PREDICTIVE_ANALYTICS)


def node_market_intelligence(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.MARKET_INTELLIGENCE)


def node_engagement_reporting(state: WorkflowState) -> WorkflowState:
    # Require baseline approval before releasing client-facing reporting package
    if not _gate_approved(state, CheckpointGate.BASELINE_APPROVAL):
        return _raise_checkpoint(
            state,
            CheckpointGate.BASELINE_APPROVAL,
            state.get("performance_baseline", {}).get("kpi_baseline", {}),
        )
    return _run_agent(state, AgentName.ENGAGEMENT_REPORTING)


def node_maturity_assessment(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.MATURITY_ASSESSMENT)


def node_firm_knowledge(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.FIRM_KNOWLEDGE)


def node_finalize(state: WorkflowState) -> WorkflowState:
    """
    Assemble the final output package and mark workflow COMPLETED.
    """
    logger.info("[Orchestrator] Finalizing workflow %s", state.get("workflow_id"))

    final_output = {
        "workflow_id":         state.get("workflow_id"),
        "engagement_id":       state.get("engagement_id"),
        "goal":                state.get("goal"),
        "mode":                state.get("mode"),
        "kpi_baseline":        state.get("performance_baseline", {}).get("kpi_baseline"),
        "benchmark_gaps":      state.get("performance_baseline", {}).get("benchmark_comparison"),
        "savings_identified":  state.get("cost_analytics", {}).get("savings_quantification"),
        "top_risks":           state.get("predictive_analytics", {}).get("risk_scores", [])[:3],
        "disruption_signals":  state.get("market_intelligence", {}).get("disruption_signals"),
        "report_s3_key":       state.get("engagement_reporting", {}).get("report_s3_key"),
        "board_deck_s3_key":   state.get("dashboard_visualization", {}).get("presentation_s3_key"),
        "completed_at":        datetime.utcnow().isoformat(),
    }

    updated = {
        **state,
        "status":       WorkflowStatus.COMPLETED,
        "final_output": final_output,
        "updated_at":   datetime.utcnow().isoformat(),
    }
    _append_audit(updated, AgentName.ORCHESTRATOR, "finalized", "workflow complete")
    save_checkpoint(updated)
    logger.info("[Orchestrator] Workflow COMPLETED: %s", state.get("workflow_id"))
    return updated


# ============================================================================
# Routing edge function
# ============================================================================

def route_from_supervisor(state: WorkflowState) -> Literal[
    "data_integration",
    "performance_baseline",
    "dashboard_viz",
    "cost_analytics",
    "diagnostics_rca",
    "predictive_analytics",
    "market_intelligence",
    "engagement_reporting",
    "maturity_assessment",
    "firm_knowledge",
    "finalize",
    "__end__",
]:
    """
    LangGraph conditional edge: maps state → next node name.
    Called after every supervisor node execution.
    """
    status = state.get("status")

    if status in (WorkflowStatus.FAILED, WorkflowStatus.DEAD_LETTER,
                  WorkflowStatus.AWAITING_HUMAN):
        return "__end__"

    next_agent = state.get("next_agent")

    routing_map = {
        AgentName.DATA_INTEGRATION:     "data_integration",
        AgentName.PERFORMANCE_BASELINE: "performance_baseline",
        AgentName.DASHBOARD_VIZ:        "dashboard_viz",
        AgentName.COST_ANALYTICS:       "cost_analytics",
        AgentName.DIAGNOSTICS_RCA:      "diagnostics_rca",
        AgentName.PREDICTIVE_ANALYTICS: "predictive_analytics",
        AgentName.MARKET_INTELLIGENCE:  "market_intelligence",
        AgentName.ENGAGEMENT_REPORTING: "engagement_reporting",
        AgentName.MATURITY_ASSESSMENT:  "maturity_assessment",
        AgentName.FIRM_KNOWLEDGE:       "firm_knowledge",
        "finalize":                     "finalize",
        None:                           "finalize",
    }
    return routing_map.get(next_agent, "finalize")


# ============================================================================
# Graph builder
# ============================================================================

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for AGT-16.

    Returns the compiled graph, ready for .invoke() or .stream().
    """
    graph = StateGraph(WorkflowState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("intake",               node_intake)
    graph.add_node("supervisor",           node_supervisor)
    graph.add_node("data_integration",     node_data_integration)
    graph.add_node("performance_baseline", node_performance_baseline)
    graph.add_node("dashboard_viz",        node_dashboard_viz)
    graph.add_node("cost_analytics",       node_cost_analytics)
    graph.add_node("diagnostics_rca",      node_diagnostics_rca)
    graph.add_node("predictive_analytics", node_predictive_analytics)
    graph.add_node("market_intelligence",  node_market_intelligence)
    graph.add_node("engagement_reporting", node_engagement_reporting)
    graph.add_node("maturity_assessment",  node_maturity_assessment)
    graph.add_node("firm_knowledge",       node_firm_knowledge)
    graph.add_node("finalize",             node_finalize)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("intake")

    # ── Edges ─────────────────────────────────────────────────────────────
    graph.add_edge("intake", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "data_integration":     "data_integration",
            "performance_baseline": "performance_baseline",
            "dashboard_viz":        "dashboard_viz",
            "cost_analytics":       "cost_analytics",
            "diagnostics_rca":      "diagnostics_rca",
            "predictive_analytics": "predictive_analytics",
            "market_intelligence":  "market_intelligence",
            "engagement_reporting": "engagement_reporting",
            "maturity_assessment":  "maturity_assessment",
            "firm_knowledge":       "firm_knowledge",
            "finalize":             "finalize",
            "__end__":              END,
        },
    )

    # Every agent node feeds back to supervisor
    for node_name in [
        "data_integration",
        "performance_baseline",
        "dashboard_viz",
        "cost_analytics",
        "diagnostics_rca",
        "predictive_analytics",
        "market_intelligence",
        "engagement_reporting",
        "maturity_assessment",
        "firm_knowledge",
    ]:
        graph.add_edge(node_name, "supervisor")

    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================================
# Public entry points  (identical signatures to agt_ss run_workflow / resume_workflow)
# ============================================================================

def run_workflow(
    goal: str,
    mode: str = "engagement",
    engagement_id: str | None = None,
    client_id: str | None = None,
    kpi_targets: dict | None = None,
    context: dict | None = None,
) -> dict:
    """
    Synchronous entry point for an analytics workflow.

    Returns the final WorkflowState dict.
    Caller should inspect state["status"] and state["errors"] on failure.
    """
    initial_state = create_initial_state(
        goal=goal,
        mode=mode,
        engagement_id=engagement_id,
        client_id=client_id,
        kpi_targets=kpi_targets,
        context=context,
    )
    graph = build_graph()
    logger.info("[Orchestrator] Starting workflow: %s", initial_state["workflow_id"])
    return graph.invoke(initial_state)


def resume_workflow(workflow_id: str, checkpoint_decision: dict) -> dict:
    """
    Resume a workflow parked at a human checkpoint gate.

    checkpoint_decision = {
        "gate":        "baseline_approval",
        "decision":    "approved" | "rejected" | "modified",
        "approved_by": "evan@cocomgroup.com",
        "notes":       "...",
    }

    Returns the final WorkflowState dict.
    """
    state = load_checkpoint(workflow_id)
    if state is None:
        raise ValueError(f"No checkpoint found for workflow_id={workflow_id}")

    if state.get("status") != WorkflowStatus.AWAITING_HUMAN:
        raise ValueError(
            f"Workflow {workflow_id} is not awaiting human input "
            f"(status={state.get('status')})"
        )

    gate   = state.get("pending_checkpoint")
    record = {
        "gate":         gate,
        "requested_at": state.get("updated_at"),
        "approved_at":  datetime.utcnow().isoformat(),
        "approved_by":  checkpoint_decision.get("approved_by"),
        "decision":     checkpoint_decision.get("decision", "approved"),
        "notes":        checkpoint_decision.get("notes", ""),
        "payload":      {},
    }
    state = {
        **state,
        "pending_checkpoint":  None,
        "status":              WorkflowStatus.RUNNING,
        "checkpoint_history":  state.get("checkpoint_history", []) + [record],
        "updated_at":          datetime.utcnow().isoformat(),
    }

    if checkpoint_decision.get("decision") == "rejected":
        state = {
            **state,
            "status": WorkflowStatus.FAILED,
            "errors": state.get("errors", []) + [
                f"Checkpoint {gate} rejected by {record['approved_by']}"
            ],
        }
        save_checkpoint(state)
        return state

    save_checkpoint(state)
    graph = build_graph()
    return graph.invoke(state)


# ============================================================================
# Internal helpers  (identical patterns to agt_ss orchestrator)
# ============================================================================

def _run_agent(state: WorkflowState, agent_name: str) -> WorkflowState:
    """Invoke an agent, merge its output into state, handle dead-letter."""
    agent = _AGENTS[agent_name]
    try:
        partial = agent.run(state)
        merged  = _merge_state(state, partial)
        merged  = {
            **merged,
            "current_agent": agent_name,
            "updated_at":    datetime.utcnow().isoformat(),
        }
        _append_audit(merged, agent_name, "success")
        save_checkpoint(merged)
        return merged

    except AgentToolError as exc:
        record = {
            "workflow_id": state.get("workflow_id"),
            "agent":       agent_name,
            "tool_call":   exc.tool_name,
            "error":       exc.detail,
            "attempt":     exc.attempt,
            "timestamp":   datetime.utcnow().isoformat(),
            "payload":     {},
        }
        append_dead_letter(record)
        failed = {
            **state,
            "status":      WorkflowStatus.DEAD_LETTER,
            "errors":      state.get("errors", []) + [str(exc)],
            "dead_letters": state.get("dead_letters", []) + [record],
            "updated_at":  datetime.utcnow().isoformat(),
        }
        save_checkpoint(failed)
        return failed


def _merge_state(base: dict, partial: dict) -> dict:
    """
    Deep-merge a partial agent output dict into the base WorkflowState.
    Top-level list fields are concatenated.
    Sub-dicts are shallow-merged.
    """
    merged = dict(base)
    for key, value in partial.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = merged[key] + value
        else:
            merged[key] = value
    return merged


def _route(state: dict) -> str:
    """
    Determine the next agent to run based on dependency resolution.

    Execution order mirrors the graph topology comment at the top of this file.
    Agents that are already complete (have non-empty output dicts) are skipped.
    """
    execution_order = [
        AgentName.DATA_INTEGRATION,
        AgentName.MARKET_INTELLIGENCE,   # no deps — can run in parallel with data_integration
        AgentName.PERFORMANCE_BASELINE,
        AgentName.COST_ANALYTICS,
        AgentName.DIAGNOSTICS_RCA,
        AgentName.PREDICTIVE_ANALYTICS,
        AgentName.DASHBOARD_VIZ,
        AgentName.ENGAGEMENT_REPORTING,
        AgentName.MATURITY_ASSESSMENT,
        AgentName.FIRM_KNOWLEDGE,
    ]

    completed = _completed_agents(state)

    for agent_name in execution_order:
        if agent_name in completed:
            continue
        agent = _AGENTS[agent_name]
        if agent.can_run(state):
            logger.info("[Supervisor] Routing to: %s", agent_name)
            return agent_name
        else:
            logger.debug("[Supervisor] %s blocked — dependencies not met", agent_name)
            # Don't block here permanently: skip to next candidate that can run
            continue

    return "finalize"


def _completed_agents(state: dict) -> set:
    """Return the set of agent names that have produced non-empty output."""
    completed = set()
    output_map = {
        AgentName.DATA_INTEGRATION:     state.get("data_integration", {}),
        AgentName.PERFORMANCE_BASELINE: state.get("performance_baseline", {}),
        AgentName.DASHBOARD_VIZ:        state.get("dashboard_visualization", {}),
        AgentName.COST_ANALYTICS:       state.get("cost_analytics", {}),
        AgentName.DIAGNOSTICS_RCA:      state.get("diagnostics_rca", {}),
        AgentName.PREDICTIVE_ANALYTICS: state.get("predictive_analytics", {}),
        AgentName.MARKET_INTELLIGENCE:  state.get("market_intelligence", {}),
        AgentName.ENGAGEMENT_REPORTING: state.get("engagement_reporting", {}),
        AgentName.MATURITY_ASSESSMENT:  state.get("maturity_assessment", {}),
        AgentName.FIRM_KNOWLEDGE:       state.get("firm_knowledge", {}),
    }
    for name, output in output_map.items():
        if output:
            completed.add(name)
    return completed


def _build_task_graph(state: dict) -> list[dict]:
    """
    Declarative task graph — mirrors the topology documented at the top of the file.
    """
    return [
        {"step": 1,  "agent": AgentName.DATA_INTEGRATION,     "depends_on": [],                                                           "status": "pending"},
        {"step": 2,  "agent": AgentName.MARKET_INTELLIGENCE,   "depends_on": [],                                                           "status": "pending"},
        {"step": 3,  "agent": AgentName.PERFORMANCE_BASELINE,  "depends_on": [AgentName.DATA_INTEGRATION],                                 "status": "pending"},
        {"step": 4,  "agent": AgentName.COST_ANALYTICS,        "depends_on": [AgentName.PERFORMANCE_BASELINE],                             "status": "pending"},
        {"step": 5,  "agent": AgentName.DIAGNOSTICS_RCA,       "depends_on": [AgentName.PERFORMANCE_BASELINE],                             "status": "pending"},
        {"step": 6,  "agent": AgentName.PREDICTIVE_ANALYTICS,  "depends_on": [AgentName.DATA_INTEGRATION, AgentName.PERFORMANCE_BASELINE], "status": "pending"},
        {"step": 7,  "agent": AgentName.DASHBOARD_VIZ,         "depends_on": [AgentName.PERFORMANCE_BASELINE],                             "status": "pending"},
        {"step": 8,  "agent": AgentName.ENGAGEMENT_REPORTING,  "depends_on": [AgentName.DATA_INTEGRATION, AgentName.PERFORMANCE_BASELINE], "status": "pending"},
        {"step": 9,  "agent": AgentName.MATURITY_ASSESSMENT,   "depends_on": [AgentName.PERFORMANCE_BASELINE],                             "status": "pending"},
        {"step": 10, "agent": AgentName.FIRM_KNOWLEDGE,        "depends_on": [AgentName.ENGAGEMENT_REPORTING],                             "status": "pending"},
    ]


def _raise_checkpoint(state: dict, gate: str, payload: object) -> dict:
    """Set the pending_checkpoint gate and park the workflow."""
    logger.info("[Orchestrator] Raising checkpoint: %s", gate)
    record = {
        "gate":         gate,
        "requested_at": datetime.utcnow().isoformat(),
        "approved_at":  None,
        "approved_by":  None,
        "decision":     None,
        "notes":        None,
        "payload":      payload if isinstance(payload, dict) else {},
    }
    updated = {
        **state,
        "pending_checkpoint":  gate,
        "status":              WorkflowStatus.AWAITING_HUMAN,
        "checkpoint_history":  state.get("checkpoint_history", []) + [record],
        "updated_at":          datetime.utcnow().isoformat(),
    }
    save_checkpoint(updated)
    return updated


def _gate_approved(state: dict, gate: str) -> bool:
    """Return True if the given checkpoint gate has been approved."""
    for rec in state.get("checkpoint_history", []):
        if rec.get("gate") == gate and rec.get("decision") == "approved":
            return True
    return False


def _append_audit(state: dict, agent: str, action: str, detail: str = "") -> None:
    """Append an audit entry directly into state's audit_log list."""
    state.setdefault("audit_log", []).append({
        "timestamp": datetime.utcnow().isoformat(),
        "agent":     agent,
        "action":    action,
        "detail":    detail,
    })
