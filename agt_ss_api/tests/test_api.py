"""
tests/test_api.py
------------------
FastAPI endpoint tests for AGT-SS API.

All orchestrator calls and checkpoint store reads/writes are mocked so tests
run without the LangGraph runtime, Anthropic API, or Aurora PostgreSQL.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Bootstrap: ensure agt_ss_api is importable ───────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agt_ss_api.main import create_app
from agt_ss_api.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(
    workflow_id: str = None,
    status: str = "completed",
    goal: str = "Source titanium fasteners",
    category: str = "Titanium Fasteners",
    **overrides,
) -> dict:
    now = datetime.utcnow().isoformat()
    wid = workflow_id or str(uuid.uuid4())
    base = {
        "workflow_id":      wid,
        "workflow_name":    goal[:80],
        "status":           status,
        "current_agent":    "orchestrator",
        "pending_checkpoint": None,
        "goal":             goal,
        "category":         category,
        "budget":           500_000.0,
        "timeline_days":    60,
        "created_at":       now,
        "updated_at":       now,
        "task_graph":       [],
        "checkpoint_history": [],
        "audit_log":        [{"timestamp": now, "agent": "orchestrator",
                              "action": "intake", "detail": "goal=Source titanium fasteners"}],
        "errors":           [],
        "dead_letters":     [],
        "escalation_required": False,
        "iteration_count":  3,
        "context":          {},
        "spend_category":   {"spend_classification": {"Titanium Fasteners": {"total_spend": 295000}}},
        "supplier_market":  {"supplier_shortlist": [{"supplier_id": "SUP-A", "name": "AeroFast"}]},
        "sourcing_execution": {
            "event_type":  "RFQ",
            "award_recommendation": {
                "supplier_id":   "SUP-A",
                "tco_per_unit":  9.5,
                "rationale":     "Best TCO",
                "created_at":    now,
            },
        },
        "contract_supplier":   {"contract_record": {"contract_id": "CNT-001", "status": "executed"}},
        "analytics_governance": {
            "monthly_dashboard": {
                "total_managed_spend":   295000,
                "savings_realized":      22000,
                "contract_coverage_pct": 100.0,
                "po_compliance_pct":     95.0,
                "generated_at":          now,
            },
            "ppv_report": {
                "variances":           [{"category": "Titanium Fasteners",
                                         "ppv_vs_budget": -5000, "flag": "favourable"}],
                "total_ppv_usd":       -5000,
                "total_savings_gap":   0,
                "favourable_count":    1,
                "unfavourable_count":  0,
                "top_unfavourable":    [],
                "generated_at":        now,
            },
            "savings_pipeline": [
                {"initiative_id": "SAV-001", "category": "Titanium Fasteners",
                 "stage": "contracted", "estimated_savings": 22000,
                 "confidence_pct": 90, "created_at": now},
            ],
            "contract_coverage_heatmap": {
                "categories":        {"Titanium Fasteners": {"is_covered": True, "risk_level": "low",
                                                              "total_spend": 295000}},
                "coverage_pct":      100.0,
                "uncovered_spend":   0.0,
                "high_risk_uncovered": [],
            },
            "maturity_roadmap": {
                "current_level":       3,
                "current_level_label": "Defined / Structured",
                "target_level":        4,
                "overall_score":       3.2,
                "initiatives":         [{"title": "Spend cube", "priority": "quick_win",
                                          "effort_months": 2}],
                "key_gaps":            ["risk_management"],
                "executive_summary":   "Level 3 maturity.",
            },
        },
        "final_output": {
            "workflow_id":      wid,
            "goal":             goal,
            "category":         category,
            "spend_summary":    {"Titanium Fasteners": {"total_spend": 295000}},
            "category_strategy": [],
            "award":            {"supplier_id": "SUP-A"},
            "contract_status":  "executed",
            "analytics":        {"total_managed_spend": 295000},
            "completed_at":     now,
        },
    }
    base.update(overrides)
    return base


@pytest.fixture
def settings_override():
    """Settings with auth disabled and synchronous execution for tests."""
    return Settings(
        disable_auth=True,
        run_workflows_async=False,
        environment="testing",
        api_keys=[],
    )


@pytest.fixture
def mock_store():
    """In-memory mock checkpoint store."""
    store = MagicMock()
    store.list_all.return_value = []
    store.load.return_value = None
    store.save.return_value = None
    store.exists.return_value = False
    return store


@pytest.fixture
def client(settings_override, mock_store):
    """TestClient with mocked dependencies."""
    app = create_app()

    from agt_ss_api.config import get_settings
    from agt_ss_api.dependencies import get_checkpoint_store

    app.dependency_overrides[get_settings]       = lambda: settings_override
    app.dependency_overrides[get_checkpoint_store] = lambda: mock_store

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, mock_store


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "timestamp" in body

    def test_readiness_returns_200_when_store_ok(self, client):
        c, store = client
        store.list_all.return_value = []
        resp = c.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"

    def test_readiness_returns_503_when_store_fails(self, client):
        c, store = client
        store.list_all.side_effect = Exception("DB connection failed")
        resp = c.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"

    def test_metrics_returns_counts(self, client):
        c, store = client
        store.list_all.return_value = [_make_state(), _make_state(status="running")]
        resp = c.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_workflows"] == 2
        assert "by_status" in body


# ---------------------------------------------------------------------------
# POST /api/v1/workflows
# ---------------------------------------------------------------------------

class TestCreateWorkflow:

    def _mock_run_workflow(self, mock_store, wid: str = None) -> tuple:
        """Configure mock store + patch run_workflow, return (state, wid)."""
        state = _make_state(workflow_id=wid or str(uuid.uuid4()), status="completed")
        mock_store.load.return_value = state
        return state, state["workflow_id"]

    def test_create_workflow_valid(self, client):
        c, store = client
        state = _make_state(status="completed")
        with patch("agt_ss_api.routers.workflows._get_orchestrator") as mock_orch:
            mock_run_workflow = MagicMock(return_value=state)
            mock_orch.return_value = (mock_run_workflow, MagicMock())
            store.load.return_value = state

            resp = c.post("/api/v1/workflows", json={
                "goal": "Source titanium fasteners for F135 program",
                "category": "Titanium Fasteners",
                "budget": 500000,
                "timeline_days": 60,
            })

        assert resp.status_code in (200, 202)
        body = resp.json()
        assert "workflow_id" in body
        assert "status" in body

    def test_create_workflow_missing_goal(self, client):
        c, _ = client
        resp = c.post("/api/v1/workflows", json={"category": "Fasteners"})
        assert resp.status_code == 422

    def test_create_workflow_goal_too_short(self, client):
        c, _ = client
        resp = c.post("/api/v1/workflows", json={"goal": "short"})
        assert resp.status_code == 422

    def test_create_workflow_negative_budget(self, client):
        c, _ = client
        resp = c.post("/api/v1/workflows", json={
            "goal": "Source titanium fasteners for the program",
            "budget": -1000,
        })
        assert resp.status_code == 422

    def test_create_workflow_with_context(self, client):
        c, store = client
        state = _make_state(status="awaiting_human",
                            pending_checkpoint="category_strategy_approval")
        with patch("agt_ss_api.routers.workflows._get_orchestrator") as mock_orch:
            mock_orch.return_value = (MagicMock(return_value=state), MagicMock())
            store.load.return_value = state

            resp = c.post("/api/v1/workflows", json={
                "goal": "Source titanium fasteners for F135 engine overhaul",
                "context": {"industry": "aerospace / defense", "annual_volume_units": 5000},
            })
        assert resp.status_code in (200, 202)

    def test_response_contains_correlation_id(self, client):
        c, store = client
        state = _make_state(status="completed")
        with patch("agt_ss_api.routers.workflows._get_orchestrator") as mock_orch:
            mock_orch.return_value = (MagicMock(return_value=state), MagicMock())
            store.load.return_value = state

            resp = c.post("/api/v1/workflows", json={
                "goal": "Source titanium fasteners for F135 engine program"
            })
        assert "X-Correlation-ID" in resp.headers


# ---------------------------------------------------------------------------
# GET /api/v1/workflows
# ---------------------------------------------------------------------------

class TestListWorkflows:

    def test_list_empty(self, client):
        c, store = client
        store.list_all.return_value = []
        resp = c.get("/api/v1/workflows")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_list_returns_summaries(self, client):
        c, store = client
        store.list_all.return_value = [_make_state(), _make_state()]
        resp = c.get("/api/v1/workflows")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2

    def test_list_filter_by_status(self, client):
        c, store = client
        store.list_all.return_value = [
            _make_state(status="completed"),
            _make_state(status="running"),
            _make_state(status="awaiting_human"),
        ]
        resp = c.get("/api/v1/workflows?status=completed")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["status"] == "completed"

    def test_list_pagination(self, client):
        c, store = client
        store.list_all.return_value = [_make_state() for _ in range(10)]
        resp = c.get("/api/v1/workflows?limit=3&offset=0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 10
        assert len(body["items"]) == 3


# ---------------------------------------------------------------------------
# GET /api/v1/workflows/{id}
# ---------------------------------------------------------------------------

class TestGetWorkflow:

    def test_get_existing_workflow(self, client):
        c, store = client
        state = _make_state()
        wid = state["workflow_id"]
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{wid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["workflow_id"] == wid
        assert body["status"] == "completed"

    def test_get_nonexistent_workflow(self, client):
        c, store = client
        store.load.return_value = None
        resp = c.get(f"/api/v1/workflows/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_get_workflow_includes_agent_outputs(self, client):
        c, store = client
        state = _make_state()
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}")
        body = resp.json()
        assert "spend_category" in body
        assert "sourcing_execution" in body
        assert "analytics_governance" in body

    def test_get_workflow_includes_audit_log(self, client):
        c, store = client
        state = _make_state()
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}")
        body = resp.json()
        assert len(body["audit_log"]) >= 1
        assert body["audit_log"][0]["agent"] == "orchestrator"

    def test_get_awaiting_human_shows_checkpoint(self, client):
        c, store = client
        state = _make_state(status="awaiting_human",
                            pending_checkpoint="category_strategy_approval")
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}")
        body = resp.json()
        assert body["status"] == "awaiting_human"
        assert body["pending_checkpoint"] == "category_strategy_approval"


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/{id}/resume
# ---------------------------------------------------------------------------

class TestResumeWorkflow:

    def test_resume_approved(self, client):
        c, store = client
        wid = str(uuid.uuid4())
        waiting = _make_state(
            workflow_id=wid,
            status="awaiting_human",
            pending_checkpoint="category_strategy_approval",
        )
        completed = _make_state(workflow_id=wid, status="completed")

        store.load.return_value = waiting
        with patch("agt_ss_api.routers.workflows._get_orchestrator") as mock_orch:
            mock_orch.return_value = (MagicMock(), MagicMock(return_value=completed))

            resp = c.post(f"/api/v1/workflows/{wid}/resume", json={
                "gate":        "category_strategy_approval",
                "decision":    "approved",
                "approved_by": "evan@cocomgroup.com",
                "notes":       "Looks good",
            })

        assert resp.status_code in (200, 202)

    def test_resume_rejected_transitions_to_failed(self, client):
        c, store = client
        wid = str(uuid.uuid4())
        waiting = _make_state(
            workflow_id=wid,
            status="awaiting_human",
            pending_checkpoint="contract_award_approval",
        )
        failed_state = _make_state(workflow_id=wid, status="failed")
        store.load.return_value = waiting

        with patch("agt_ss_api.routers.workflows._get_orchestrator") as mock_orch:
            mock_orch.return_value = (MagicMock(), MagicMock(return_value=failed_state))

            resp = c.post(f"/api/v1/workflows/{wid}/resume", json={
                "gate":        "contract_award_approval",
                "decision":    "rejected",
                "approved_by": "evan@cocomgroup.com",
            })
        assert resp.status_code in (200, 202)

    def test_resume_wrong_status_returns_400(self, client):
        c, store = client
        wid = str(uuid.uuid4())
        store.load.return_value = _make_state(workflow_id=wid, status="running")
        resp = c.post(f"/api/v1/workflows/{wid}/resume", json={
            "gate": "contract_award_approval",
            "decision": "approved",
            "approved_by": "evan@cocomgroup.com",
        })
        assert resp.status_code == 400

    def test_resume_gate_mismatch_returns_400(self, client):
        c, store = client
        wid = str(uuid.uuid4())
        store.load.return_value = _make_state(
            workflow_id=wid,
            status="awaiting_human",
            pending_checkpoint="category_strategy_approval",
        )
        resp = c.post(f"/api/v1/workflows/{wid}/resume", json={
            "gate":        "contract_award_approval",   # wrong gate
            "decision":    "approved",
            "approved_by": "evan@cocomgroup.com",
        })
        assert resp.status_code == 400
        assert "mismatch" in resp.json().get("detail", "").lower()

    def test_resume_nonexistent_workflow_returns_404(self, client):
        c, store = client
        store.load.return_value = None
        resp = c.post(f"/api/v1/workflows/{uuid.uuid4()}/resume", json={
            "gate": "category_strategy_approval",
            "decision": "approved",
            "approved_by": "evan@cocomgroup.com",
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/workflows/{id}/checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointEndpoint:

    def test_get_checkpoint_awaiting(self, client):
        c, store = client
        wid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        state = _make_state(
            workflow_id=wid,
            status="awaiting_human",
            pending_checkpoint="negotiation_strategy_approval",
            checkpoint_history=[{
                "gate":         "negotiation_strategy_approval",
                "requested_at": now,
                "approved_at":  None,
                "approved_by":  None,
                "decision":     None,
                "payload":      {"target_price": 9.5},
            }],
        )
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{wid}/checkpoint")
        assert resp.status_code == 200
        body = resp.json()
        assert body["gate"] == "negotiation_strategy_approval"
        assert body["payload"]["target_price"] == 9.5

    def test_get_checkpoint_not_pending_returns_409(self, client):
        c, store = client
        state = _make_state(status="completed", pending_checkpoint=None)
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}/checkpoint")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/v1/workflows/{id}/audit
# ---------------------------------------------------------------------------

class TestAuditLog:

    def test_get_audit_log(self, client):
        c, store = client
        state = _make_state()
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}/audit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert "entries" in body

    def test_audit_log_filter_by_agent(self, client):
        c, store = client
        now = datetime.utcnow().isoformat()
        state = _make_state()
        state["audit_log"] = [
            {"timestamp": now, "agent": "orchestrator", "action": "intake", "detail": ""},
            {"timestamp": now, "agent": "spend_category_intelligence", "action": "success", "detail": ""},
        ]
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}/audit?agent=orchestrator")
        body = resp.json()
        assert all(e["agent"] == "orchestrator" for e in body["entries"])


# ---------------------------------------------------------------------------
# GET /api/v1/workflows/{id}/final
# ---------------------------------------------------------------------------

class TestFinalOutput:

    def test_get_final_output_completed(self, client):
        c, store = client
        state = _make_state(status="completed")
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}/final")
        assert resp.status_code == 200
        body = resp.json()
        assert "workflow_id" in body
        assert "award" in body

    def test_get_final_output_not_completed_returns_409(self, client):
        c, store = client
        state = _make_state(status="running")
        store.load.return_value = state
        resp = c.get(f"/api/v1/workflows/{state['workflow_id']}/final")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------

class TestAnalytics:

    def _completed_state(self, store) -> dict:
        state = _make_state(status="completed")
        store.list_all.return_value = [state]
        store.load.return_value = state
        return state

    def test_get_dashboard(self, client):
        c, store = client
        self._completed_state(store)
        resp = c.get("/api/v1/analytics/dashboard")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_managed_spend" in body

    def test_get_ppv(self, client):
        c, store = client
        self._completed_state(store)
        resp = c.get("/api/v1/analytics/ppv")
        assert resp.status_code == 200
        body = resp.json()
        assert "variances" in body
        assert "total_ppv_usd" in body

    def test_get_ppv_filter_favourable(self, client):
        c, store = client
        self._completed_state(store)
        resp = c.get("/api/v1/analytics/ppv?flag=favourable")
        assert resp.status_code == 200
        body = resp.json()
        for v in body.get("variances", []):
            assert v["flag"] == "favourable"

    def test_get_savings_pipeline(self, client):
        c, store = client
        self._completed_state(store)
        resp = c.get("/api/v1/analytics/savings-pipeline")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "total_value" in body

    def test_get_savings_pipeline_filter_by_stage(self, client):
        c, store = client
        self._completed_state(store)
        resp = c.get("/api/v1/analytics/savings-pipeline?stage=contracted")
        assert resp.status_code == 200
        body = resp.json()
        for item in body["items"]:
            assert item["stage"] == "contracted"

    def test_get_contract_coverage(self, client):
        c, store = client
        self._completed_state(store)
        resp = c.get("/api/v1/analytics/contract-coverage")
        assert resp.status_code == 200
        body = resp.json()
        assert "categories" in body
        assert "coverage_pct" in body

    def test_get_maturity_roadmap(self, client):
        c, store = client
        self._completed_state(store)
        resp = c.get("/api/v1/analytics/maturity")
        assert resp.status_code == 200
        body = resp.json()
        assert "current_level" in body
        assert "initiatives" in body

    def test_analytics_no_completed_workflow_returns_404(self, client):
        c, store = client
        store.list_all.return_value = []
        resp = c.get("/api/v1/analytics/dashboard")
        assert resp.status_code == 404

    def test_analytics_with_workflow_id_param(self, client):
        c, store = client
        state = _make_state(status="completed")
        wid = state["workflow_id"]
        store.load.return_value = state
        store.list_all.return_value = [state]
        resp = c.get(f"/api/v1/analytics/dashboard?workflow_id={wid}")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemas:

    def test_workflow_detail_from_state_roundtrip(self):
        from agt_ss_api.models.schemas import WorkflowDetailResponse
        state = _make_state()
        response = WorkflowDetailResponse.from_state(state)
        assert response.workflow_id == state["workflow_id"]
        assert response.status.value == "completed"
        assert len(response.audit_log) == 1

    def test_workflow_summary_from_state(self):
        from agt_ss_api.models.schemas import WorkflowSummary
        state = _make_state()
        summary = WorkflowSummary.from_state(state)
        assert summary.goal == state["goal"]
        assert summary.category == state["category"]

    def test_checkpoint_resume_request_invalid_decision(self):
        from agt_ss_api.models.schemas import CheckpointResumeRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CheckpointResumeRequest(
                gate="contract_award_approval",
                decision="maybe",   # invalid
                approved_by="evan@cocomgroup.com",
            )

    def test_workflow_create_request_strips_whitespace(self):
        from agt_ss_api.models.schemas import WorkflowCreateRequest
        req = WorkflowCreateRequest(goal="  Source titanium fasteners for program  ")
        assert req.goal == "Source titanium fasteners for program"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
