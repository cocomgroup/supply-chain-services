# Databricks notebook source
"""
tests/test_orchestrator.py
--------------------------
Unit tests for AGT-SS orchestrator logic.

Tests run without Databricks / SAP / LangGraph installed by mocking
heavy dependencies. Core logic under test:
  - WorkflowState creation and field defaults
  - Supervisor routing decisions
  - Human checkpoint gate raise / approve / reject flow
  - Dead-letter handling and state transitions
  - Agent can_run() dependency checks
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy dependencies so tests run in isolation
# ---------------------------------------------------------------------------

def _stub_module(name: str):
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
    return mod

for _m in ["langgraph", "langgraph.graph", "pyspark", "pyspark.sql",
           "pyspark.sql.functions", "pyspark.sql.types"]:
    _stub_module(_m)

# Stub StateGraph
_lg = sys.modules["langgraph.graph"]
class _FakeStateGraph:
    def __init__(self, *a, **kw): self._nodes = {}; self._edges = []; self._cond = []
    def add_node(self, n, fn): self._nodes[n] = fn
    def set_entry_point(self, n): pass
    def add_edge(self, a, b): self._edges.append((a, b))
    def add_conditional_edges(self, *a, **kw): pass
    def compile(self): return self
    def invoke(self, state): return state

_lg.StateGraph = _FakeStateGraph
_lg.END = "__end__"

# ---------------------------------------------------------------------------
# Now import the modules under test
# ---------------------------------------------------------------------------

sys.path.insert(0, "/home/claude")
from agt_ss.state.schema import (
    AgentName, CheckpointGate, WorkflowStatus, create_initial_state
)
from agt_ss.agents.sub_agents import (
    SpendCategoryAgent, SupplierMarketAgent,
    SourcingExecutionAgent, ContractSupplierAgent,
    AnalyticsGovernanceAgent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    s = create_initial_state(goal="Source titanium fasteners for F135 program",
                              category="fasteners")
    return {**s, **overrides}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateInitialState(unittest.TestCase):

    def test_required_fields_present(self):
        s = create_initial_state(goal="test goal")
        self.assertEqual(s["status"], WorkflowStatus.PENDING)
        self.assertEqual(s["current_agent"], AgentName.ORCHESTRATOR)
        self.assertIsNone(s["pending_checkpoint"])
        self.assertEqual(s["errors"], [])
        self.assertEqual(s["audit_log"], [])

    def test_workflow_id_is_uuid(self):
        import uuid
        s = create_initial_state(goal="test")
        uuid.UUID(s["workflow_id"])  # raises if invalid

    def test_optional_fields(self):
        s = create_initial_state(goal="g", category="MRO", budget=500_000.0,
                                  timeline_days=90, context={"foo": "bar"})
        self.assertEqual(s["category"], "MRO")
        self.assertEqual(s["budget"], 500_000.0)
        self.assertEqual(s["context"]["foo"], "bar")


class TestSpendCategoryCanRun(unittest.TestCase):

    def setUp(self):
        self.agent = SpendCategoryAgent()

    def test_can_run_with_goal(self):
        self.assertTrue(self.agent.can_run(_base_state()))

    def test_cannot_run_without_goal(self):
        s = _base_state()
        s["goal"] = ""
        self.assertFalse(self.agent.can_run(s))


class TestSupplierMarketCanRun(unittest.TestCase):

    def setUp(self):
        self.agent = SupplierMarketAgent()

    def test_blocked_without_kraljic(self):
        s = _base_state()
        self.assertFalse(self.agent.can_run(s))

    def test_unblocked_with_kraljic(self):
        s = _base_state()
        s["spend_category"] = {"kraljic_matrix": {"strategic": ["fasteners"]}}
        self.assertTrue(self.agent.can_run(s))


class TestSourcingExecutionCanRun(unittest.TestCase):

    def setUp(self):
        self.agent = SourcingExecutionAgent()

    def test_blocked_without_shortlist(self):
        s = _base_state()
        s["spend_category"] = {"category_strategies": [{"category": "fasteners"}]}
        self.assertFalse(self.agent.can_run(s))

    def test_blocked_without_strategies(self):
        s = _base_state()
        s["supplier_market"] = {"supplier_shortlist": [{"id": "SUP-001"}]}
        self.assertFalse(self.agent.can_run(s))

    def test_unblocked_with_both(self):
        s = _base_state()
        s["spend_category"]  = {"category_strategies": [{"category": "fasteners"}]}
        s["supplier_market"] = {"supplier_shortlist": [{"id": "SUP-001"}]}
        self.assertTrue(self.agent.can_run(s))


class TestContractSupplierCanRun(unittest.TestCase):

    def setUp(self):
        self.agent = ContractSupplierAgent()

    def test_blocked_without_award(self):
        s = _base_state()
        self.assertFalse(self.agent.can_run(s))

    def test_blocked_without_gate_approval(self):
        s = _base_state()
        s["sourcing_execution"] = {"award_recommendation": {"supplier_id": "SUP-001"}}
        self.assertFalse(self.agent.can_run(s))

    def test_unblocked_with_award_and_gate(self):
        s = _base_state()
        s["sourcing_execution"] = {"award_recommendation": {"supplier_id": "SUP-001"}}
        s["checkpoint_history"] = [{
            "gate": CheckpointGate.CONTRACT_AWARD_APPROVAL,
            "decision": "approved",
        }]
        self.assertTrue(self.agent.can_run(s))


class TestAnalyticsGovernanceCanRun(unittest.TestCase):

    def setUp(self):
        self.agent = AnalyticsGovernanceAgent()

    def test_always_runnable_with_goal(self):
        s = _base_state()
        self.assertTrue(self.agent.can_run(s))


class TestCheckpointHelpers(unittest.TestCase):
    """Test the gate raise / approve helpers via the orchestrator module."""

    def test_gate_not_approved_initially(self):
        from agt_ss.orchestrator import _gate_approved
        s = _base_state()
        self.assertFalse(_gate_approved(s, CheckpointGate.CONTRACT_AWARD_APPROVAL))

    def test_gate_approved_after_record(self):
        from agt_ss.orchestrator import _gate_approved
        s = _base_state()
        s["checkpoint_history"] = [{
            "gate": CheckpointGate.CONTRACT_AWARD_APPROVAL,
            "decision": "approved",
        }]
        self.assertTrue(_gate_approved(s, CheckpointGate.CONTRACT_AWARD_APPROVAL))

    def test_gate_not_approved_if_rejected(self):
        from agt_ss.orchestrator import _gate_approved
        s = _base_state()
        s["checkpoint_history"] = [{
            "gate": CheckpointGate.CONTRACT_AWARD_APPROVAL,
            "decision": "rejected",
        }]
        self.assertFalse(_gate_approved(s, CheckpointGate.CONTRACT_AWARD_APPROVAL))

    @patch("agt_ss.orchestrator.save_checkpoint")
    def test_raise_checkpoint_sets_state(self, mock_save):
        from agt_ss.orchestrator import _raise_checkpoint
        s = _base_state()
        updated = _raise_checkpoint(s, CheckpointGate.CATEGORY_STRATEGY_APPROVAL, {})
        self.assertEqual(updated["status"], WorkflowStatus.AWAITING_HUMAN)
        self.assertEqual(updated["pending_checkpoint"],
                         CheckpointGate.CATEGORY_STRATEGY_APPROVAL)
        self.assertEqual(len(updated["checkpoint_history"]), 1)
        mock_save.assert_called_once()


class TestStateMerge(unittest.TestCase):

    def test_dict_fields_are_shallow_merged(self):
        from agt_ss.orchestrator import _merge_state
        base    = _base_state()
        base["spend_category"] = {"spend_classification": {"fasteners": 100.0}}
        partial = {"spend_category": {"kraljic_matrix": {"strategic": ["fasteners"]}}}
        merged  = _merge_state(base, partial)
        # Both keys should be present
        self.assertIn("spend_classification", merged["spend_category"])
        self.assertIn("kraljic_matrix",       merged["spend_category"])

    def test_list_fields_are_concatenated(self):
        from agt_ss.orchestrator import _merge_state
        base    = _base_state()
        base["errors"] = ["existing error"]
        partial = {"errors": ["new error"]}
        merged  = _merge_state(base, partial)
        self.assertEqual(merged["errors"], ["existing error", "new error"])

    def test_scalar_override(self):
        from agt_ss.orchestrator import _merge_state
        base    = _base_state()
        partial = {"status": WorkflowStatus.COMPLETED}
        merged  = _merge_state(base, partial)
        self.assertEqual(merged["status"], WorkflowStatus.COMPLETED)


class TestRouting(unittest.TestCase):

    def test_routes_to_spend_category_first(self):
        from agt_ss.orchestrator import _route
        s = _base_state()
        # Empty state — SpendCategory.can_run() = True (only needs goal)
        self.assertEqual(_route(s), AgentName.SPEND_CATEGORY)

    def test_routes_to_analytics_when_all_done(self):
        """Analytics has no dependencies so it routes there after all others complete."""
        from agt_ss.orchestrator import _route
        s = _base_state()
        # Simulate all other agents done with non-empty output
        s["spend_category"]    = {"spend_classification": {}, "category_strategies": [{}],
                                   "kraljic_matrix": {"strategic": []}}
        s["supplier_market"]   = {"supplier_shortlist": [{}], "approved_supplier_list": []}
        s["sourcing_execution"] = {"award_recommendation": {"supplier_id": "X"}}
        s["contract_supplier"] = {"contract_record": {"status": "executed"}}
        s["analytics_governance"] = {"ppv_report": {"variances": []}}
        # All completed → should reach finalize
        result = _route(s)
        self.assertEqual(result, "finalize")


class TestIterationGuard(unittest.TestCase):

    @patch("agt_ss.checkpoints.delta.save_checkpoint")
    def test_escalates_at_limit(self, mock_save):
        from agt_ss.orchestrator import node_supervisor
        s = _base_state()
        s["iteration_count"] = 50  # at limit
        result = node_supervisor(s)
        self.assertEqual(result["status"], WorkflowStatus.FAILED)
        self.assertTrue(result["escalation_required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)