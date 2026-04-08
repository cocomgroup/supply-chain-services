# Databricks notebook source
"""
agt_ss.orchestrator
--------------------
AGT-SS Orchestrator — the LangGraph StateGraph supervisor for the Strategic
Sourcing & Procurement multi-agent system.

Graph topology
--------------

  [intake]
     │
     ▼
  [supervisor]  ◄──────────────────────────────────────────┐
     │                                                       │
     ├──► [spend_category]                                   │
     │         │                                             │
     ├──► [supplier_market]  (parallel-ready, depends on ^) │
     │         │                                             │
     ├──► [sourcing_execution]  (depends on both above)      │
     │         │                                             │
     │   [checkpoint: category_strategy_approval]            │
     │   [checkpoint: negotiation_strategy_approval]         │
     │         │                                             │
     ├──► [contract_supplier]  (depends on award + gates)   │
     │         │                                             │
     │   [checkpoint: contract_award_approval]               │
     │         │                                             │
     ├──► [analytics_governance]                             │
     │                                                       │
     └──► [finalize] ──────────────────────────────────────►┘
                │
              [END]

Human checkpoint gates are blocking: the graph parks in AWAITING_HUMAN and
must be resumed with an approved/rejected decision before progressing.

Dead-letter handling: after MAX_RETRIES the failed agent's record is written
to the dead-letter table and the workflow transitions to DEAD_LETTER status.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from langgraph.graph import END, StateGraph

from .agents.base import AgentToolError
from .agents.sub_agents import (
    AnalyticsGovernanceAgent,
    ContractSupplierAgent,
    SourcingExecutionAgent,
    SpendCategoryAgent,
    SupplierMarketAgent,
)
from .checkpoints.delta import append_dead_letter, save_checkpoint
from .state.schema import (
    AgentName,
    CheckpointGate,
    WorkflowState,
    WorkflowStatus,
    create_initial_state,
)

logger = logging.getLogger(__name__)

# Max supervisor iterations before forced termination
SUPERVISOR_ITERATION_LIMIT = 50

# ── Instantiate agents ────────────────────────────────────────────────────

_AGENTS = {
    AgentName.SPEND_CATEGORY:       SpendCategoryAgent(),
    AgentName.SUPPLIER_MARKET:      SupplierMarketAgent(),
    AgentName.SOURCING_EXECUTION:   SourcingExecutionAgent(),
    AgentName.CONTRACT_SUPPLIER:    ContractSupplierAgent(),
    AgentName.ANALYTICS_GOVERNANCE: AnalyticsGovernanceAgent(),
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
        "status":         WorkflowStatus.RUNNING,
        "task_graph":     task_graph,
        "iteration_count": 0,
        "updated_at":     datetime.utcnow().isoformat(),
    }
    _append_audit(updated, AgentName.ORCHESTRATOR, "intake", f"goal={state.get('goal', '')[:60]}")
    save_checkpoint(updated)
    return updated


def node_supervisor(state: WorkflowState) -> WorkflowState:
    """
    Supervisor node: inspect current state and decide the next agent to invoke.
    Acts as the central router — called between every agent execution.

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
            "iteration_count":    count,
            "status":             WorkflowStatus.FAILED,
            "escalation_required": True,
            "errors":             state.get("errors", []) + ["Iteration limit exceeded."],
        }
        save_checkpoint(updated)
        return updated

    # Check for a pending human checkpoint gate
    if state.get("pending_checkpoint"):
        logger.info("[Supervisor] Parking — pending checkpoint: %s",
                    state["pending_checkpoint"])
        return {
            **state,
            "iteration_count": count,
            "status":          WorkflowStatus.AWAITING_HUMAN,
        }

    # Determine next agent
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


def node_spend_category(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.SPEND_CATEGORY)


def node_supplier_market(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.SUPPLIER_MARKET)


def node_sourcing_execution(state: WorkflowState) -> WorkflowState:
    state = _run_agent(state, AgentName.SOURCING_EXECUTION)
    # After sourcing execution, raise CATEGORY_STRATEGY_APPROVAL gate if not yet approved
    if not _gate_approved(state, CheckpointGate.CATEGORY_STRATEGY_APPROVAL):
        state = _raise_checkpoint(state, CheckpointGate.CATEGORY_STRATEGY_APPROVAL,
                                  state.get("spend_category", {}).get("category_strategies", []))
    return state


def node_contract_supplier(state: WorkflowState) -> WorkflowState:
    # Before running, ensure NEGOTIATION_STRATEGY_APPROVAL gate is raised
    if not _gate_approved(state, CheckpointGate.NEGOTIATION_STRATEGY_APPROVAL):
        neg_strategy = state.get("sourcing_execution", {}).get("award_recommendation", {})
        return _raise_checkpoint(state, CheckpointGate.NEGOTIATION_STRATEGY_APPROVAL, neg_strategy)

    state = _run_agent(state, AgentName.CONTRACT_SUPPLIER)

    # After contract draft, raise CONTRACT_AWARD_APPROVAL
    if not _gate_approved(state, CheckpointGate.CONTRACT_AWARD_APPROVAL):
        contract = state.get("contract_supplier", {}).get("contract_record", {})
        state = _raise_checkpoint(state, CheckpointGate.CONTRACT_AWARD_APPROVAL, contract)
    return state


def node_analytics_governance(state: WorkflowState) -> WorkflowState:
    return _run_agent(state, AgentName.ANALYTICS_GOVERNANCE)


def node_finalize(state: WorkflowState) -> WorkflowState:
    """
    Assemble the final output package and mark workflow COMPLETED.
    """
    logger.info("[Orchestrator] Finalizing workflow %s", state.get("workflow_id"))

    final_output = {
        "workflow_id":      state.get("workflow_id"),
        "goal":             state.get("goal"),
        "category":         state.get("category"),
        "spend_summary":    state.get("spend_category", {}).get("spend_classification"),
        "category_strategy": state.get("spend_category", {}).get("category_strategies"),
        "award":            state.get("sourcing_execution", {}).get("award_recommendation"),
        "contract_status":  state.get("contract_supplier", {}).get("contract_record", {}).get("status"),
        "analytics":        state.get("analytics_governance", {}).get("monthly_dashboard"),
        "completed_at":     datetime.utcnow().isoformat(),
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
    "spend_category",
    "supplier_market",
    "sourcing_execution",
    "contract_supplier",
    "analytics_governance",
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
        AgentName.SPEND_CATEGORY:       "spend_category",
        AgentName.SUPPLIER_MARKET:      "supplier_market",
        AgentName.SOURCING_EXECUTION:   "sourcing_execution",
        AgentName.CONTRACT_SUPPLIER:    "contract_supplier",
        AgentName.ANALYTICS_GOVERNANCE: "analytics_governance",
        "finalize":                     "finalize",
        None:                           "finalize",
    }
    return routing_map.get(next_agent, "finalize")


# ============================================================================
# Graph builder
# ============================================================================

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for AGT-SS.

    Returns the compiled graph, ready for .invoke() or .stream().
    """
    graph = StateGraph(WorkflowState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("intake",               node_intake)
    graph.add_node("supervisor",           node_supervisor)
    graph.add_node("spend_category",       node_spend_category)
    graph.add_node("supplier_market",      node_supplier_market)
    graph.add_node("sourcing_execution",   node_sourcing_execution)
    graph.add_node("contract_supplier",    node_contract_supplier)
    graph.add_node("analytics_governance", node_analytics_governance)
    graph.add_node("finalize",             node_finalize)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("intake")

    # ── Edges ─────────────────────────────────────────────────────────────
    # intake always flows to supervisor
    graph.add_edge("intake", "supervisor")

    # supervisor routes dynamically
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "spend_category":       "spend_category",
            "supplier_market":      "supplier_market",
            "sourcing_execution":   "sourcing_execution",
            "contract_supplier":    "contract_supplier",
            "analytics_governance": "analytics_governance",
            "finalize":             "finalize",
            "__end__":              END,
        },
    )

    # Every agent node feeds back to supervisor
    for node_name in [
        "spend_category",
        "supplier_market",
        "sourcing_execution",
        "contract_supplier",
        "analytics_governance",
    ]:
        graph.add_edge(node_name, "supervisor")

    # finalize → END
    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================================
# Public entry points
# ============================================================================

def run_workflow(
    goal: str,
    category: str | None = None,
    budget: float | None = None,
    timeline_days: int | None = None,
    context: dict | None = None,
) -> dict:
    """
    Synchronous entry point for a sourcing workflow.

    Returns the final WorkflowState dict.
    Raises if the workflow ends in FAILED or DEAD_LETTER status without a clean
    final_output (caller should inspect state["status"] and state["errors"]).
    """
    initial_state = create_initial_state(
        goal=goal,
        category=category,
        budget=budget,
        timeline_days=timeline_days,
        context=context,
    )
    graph = build_graph()
    logger.info("[Orchestrator] Starting workflow: %s", initial_state["workflow_id"])
    final_state = graph.invoke(initial_state)
    return final_state


def resume_workflow(workflow_id: str, checkpoint_decision: dict) -> dict:
    """
    Resume a workflow parked at a human checkpoint gate.

    checkpoint_decision = {
        "gate":        "contract_award_approval",
        "decision":    "approved" | "rejected" | "modified",
        "approved_by": "evan@cocomgroup.com",
        "notes":       "...",
    }

    Returns the final WorkflowState dict.
    """
    from .checkpoints.delta import load_checkpoint

    state = load_checkpoint(workflow_id)
    if state is None:
        raise ValueError(f"No checkpoint found for workflow_id={workflow_id}")

    if state.get("status") != WorkflowStatus.AWAITING_HUMAN:
        raise ValueError(f"Workflow {workflow_id} is not awaiting human input "
                         f"(status={state.get('status')})")

    # Record the human decision
    gate = state.get("pending_checkpoint")
    record = {
        "gate":        gate,
        "requested_at": state.get("updated_at"),
        "approved_at": datetime.utcnow().isoformat(),
        "approved_by": checkpoint_decision.get("approved_by"),
        "decision":    checkpoint_decision.get("decision", "approved"),
        "notes":       checkpoint_decision.get("notes", ""),
        "payload":     {},
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
            "errors": state.get("errors", []) + [f"Checkpoint {gate} rejected by {record['approved_by']}"],
        }
        save_checkpoint(state)
        return state

    save_checkpoint(state)
    graph = build_graph()
    return graph.invoke(state)


# ============================================================================
# Internal helpers
# ============================================================================

def _run_agent(state: WorkflowState, agent_name: str) -> WorkflowState:
    """
    Invoke an agent, merge its output into state, handle dead-letter.
    """
    agent = _AGENTS[agent_name]
    try:
        partial = agent.run(state)
        merged  = _merge_state(state, partial)
        merged  = {**merged, "current_agent": agent_name, "updated_at": datetime.utcnow().isoformat()}
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
    Top-level list fields (errors, audit_log, etc.) are concatenated.
    Sub-dicts (spend_category, supplier_market, etc.) are shallow-merged.
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
    Returns an AgentName value or "finalize".
    """
    # Ordered execution plan — dependencies enforced by can_run()
    execution_order = [
        AgentName.SPEND_CATEGORY,
        AgentName.SUPPLIER_MARKET,
        AgentName.SOURCING_EXECUTION,
        AgentName.CONTRACT_SUPPLIER,
        AgentName.ANALYTICS_GOVERNANCE,
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
            # Dependency not met — block here until upstream completes
            logger.debug("[Supervisor] %s blocked — dependencies not met", agent_name)
            return agent_name  # will be a no-op if can_run() fails in the node

    # All agents done
    return "finalize"


def _completed_agents(state: dict) -> set:
    """Return the set of agent names that have produced non-empty output."""
    completed = set()
    output_map = {
        AgentName.SPEND_CATEGORY:       state.get("spend_category", {}),
        AgentName.SUPPLIER_MARKET:      state.get("supplier_market", {}),
        AgentName.SOURCING_EXECUTION:   state.get("sourcing_execution", {}),
        AgentName.CONTRACT_SUPPLIER:    state.get("contract_supplier", {}),
        AgentName.ANALYTICS_GOVERNANCE: state.get("analytics_governance", {}),
    }
    for name, output in output_map.items():
        if output:  # non-empty dict = agent has run
            completed.add(name)
    return completed


def _build_task_graph(state: dict) -> list[dict]:
    """
    Build a declarative task graph from the goal context.
    Each step records which agent owns it and its upstream dependencies.
    """
    return [
        {"step": 1, "agent": AgentName.SPEND_CATEGORY,       "depends_on": [],                                "status": "pending"},
        {"step": 2, "agent": AgentName.SUPPLIER_MARKET,       "depends_on": [AgentName.SPEND_CATEGORY],        "status": "pending"},
        {"step": 3, "agent": AgentName.SOURCING_EXECUTION,    "depends_on": [AgentName.SPEND_CATEGORY,
                                                                               AgentName.SUPPLIER_MARKET],       "status": "pending"},
        {"step": 4, "agent": AgentName.CONTRACT_SUPPLIER,     "depends_on": [AgentName.SOURCING_EXECUTION],    "status": "pending"},
        {"step": 5, "agent": AgentName.ANALYTICS_GOVERNANCE,  "depends_on": [],                                "status": "pending"},
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
        "pending_checkpoint": gate,
        "status":             WorkflowStatus.AWAITING_HUMAN,
        "checkpoint_history": state.get("checkpoint_history", []) + [record],
        "updated_at":         datetime.utcnow().isoformat(),
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