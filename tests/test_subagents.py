# Databricks notebook source
"""
tests/test_sub_agents.py
-------------------------
Unit tests for all five AGT-SS sub-agent implementations.

LLM calls are mocked via patch to avoid API costs in CI.
Deterministic logic (Kraljic scoring, TCO model, PPV, etc.)
is tested with real inputs to verify correctness.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Stub langgraph to avoid import errors
for _m in ["langgraph", "langgraph.graph", "pyspark", "pyspark.sql",
           "pyspark.sql.functions", "pyspark.sql.types"]:
    mod = types.ModuleType(_m)
    sys.modules.setdefault(_m, mod)

_lg = sys.modules["langgraph.graph"]
_lg.StateGraph = MagicMock()
_lg.END = "__end__"

sys.path.insert(0, "/home/claude")

from agt_ss.state.schema import create_initial_state, CheckpointGate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    s = create_initial_state(
        goal="Source titanium fasteners for F135 engine overhaul",
        category="Titanium Fasteners",
        budget=500_000.0,
        timeline_days=60,
        context={"industry": "aerospace / defense", "annual_volume_units": 10000},
    )
    return {**s, **overrides}


SAMPLE_SPEND_ROWS = [
    {"po_number": "PO-001", "supplier_id": "SUP-A", "supplier_name": "AeroFast Inc",
     "category_raw": "Titanium Fasteners", "category": "Titanium Fasteners",
     "spend_type": "direct_material", "amount_usd": 120_000.0, "po_date": "2024-01-15"},
    {"po_number": "PO-002", "supplier_id": "SUP-B", "supplier_name": "MetalSupply Co",
     "category_raw": "Titanium Fasteners", "category": "Titanium Fasteners",
     "spend_type": "direct_material", "amount_usd": 80_000.0, "po_date": "2024-02-10"},
    {"po_number": "PO-003", "supplier_id": "SUP-C", "supplier_name": "FastFix Ltd",
     "category_raw": "MRO Supplies", "category": "MRO Supplies",
     "spend_type": "mro", "amount_usd": 15_000.0, "po_date": "2024-03-05"},
    {"po_number": "PO-004", "supplier_id": "SUP-A", "supplier_name": "AeroFast Inc",
     "category_raw": "Titanium Fasteners", "category": "Titanium Fasteners",
     "spend_type": "direct_material", "amount_usd": 95_000.0, "po_date": "2024-04-20"},
    {"po_number": "PO-005", "supplier_id": "SUP-D", "supplier_name": "OfficePro",
     "category_raw": "Office Supplies", "category": "Office Supplies",
     "spend_type": "indirect", "amount_usd": 3_500.0, "po_date": "2024-05-01"},
    # Add more to test tail spend
    *[{"po_number": f"PO-0{i:02d}", "supplier_id": f"SUP-TAIL-{i}",
       "supplier_name": f"Tail Supplier {i}", "category": "MRO Supplies",
       "spend_type": "mro", "amount_usd": 800.0, "po_date": "2024-06-01"}
      for i in range(10, 25)],
]


# ===========================================================================
# Test SpendCategoryAgent
# ===========================================================================

class TestSpendCategoryAgent(unittest.TestCase):

    def setUp(self):
        from agt_ss.agents.spend_category import SpendCategoryAgent
        self.agent = SpendCategoryAgent()

    def test_can_run_with_goal(self):
        self.assertTrue(self.agent.can_run(_base_state()))

    def test_can_run_fails_without_goal(self):
        s = _base_state()
        s["goal"] = ""
        self.assertFalse(self.agent.can_run(s))

    def test_aggregate_spend_produces_correct_totals(self):
        agg = self.agent._aggregate_spend(SAMPLE_SPEND_ROWS)
        self.assertIn("Titanium Fasteners", agg)
        tf = agg["Titanium Fasteners"]
        self.assertAlmostEqual(tf["total_spend"], 295_000.0)
        self.assertEqual(tf["transaction_count"], 3)
        self.assertIn("SUP-A", tf["suppliers"])
        self.assertIn("SUP-B", tf["suppliers"])

    def test_aggregate_spend_empty_returns_empty(self):
        self.assertEqual(self.agent._aggregate_spend([]), {})

    def test_spend_share_sums_to_100(self):
        agg = self.agent._aggregate_spend(SAMPLE_SPEND_ROWS)
        total_share = sum(m["spend_share_pct"] for m in agg.values())
        self.assertAlmostEqual(total_share, 100.0, places=1)

    def test_kraljic_assigns_all_categories(self):
        agg = self.agent._aggregate_spend(SAMPLE_SPEND_ROWS)
        with patch.object(self.agent, "_llm_refine_supply_risk", side_effect=lambda s, _: s):
            matrix = self.agent._score_kraljic(agg, _base_state())
        all_cats = (matrix["strategic"] + matrix["leverage"] +
                    matrix["bottleneck"] + matrix["non_critical"])
        assigned = {item["category"] for item in all_cats}
        for cat in agg:
            self.assertIn(cat, assigned)

    def test_kraljic_empty_input_returns_empty_matrix(self):
        with patch.object(self.agent, "_llm_refine_supply_risk", side_effect=lambda s, _: s):
            matrix = self.agent._score_kraljic({}, _base_state())
        self.assertEqual(len(matrix["strategic"]), 0)
        self.assertEqual(len(matrix["leverage"]), 0)

    def test_category_strategies_match_matrix(self):
        agg = self.agent._aggregate_spend(SAMPLE_SPEND_ROWS)
        with patch.object(self.agent, "_llm_refine_supply_risk", side_effect=lambda s, _: s):
            matrix = self.agent._score_kraljic(agg, _base_state())
        with patch.object(self.agent, "_llm_strategy_brief", return_value="mock brief"):
            strategies = self.agent._generate_category_strategies(matrix, _base_state())
        strategy_cats = {s["category"] for s in strategies}
        matrix_cats   = {item["category"]
                         for quad in matrix.values() for item in quad}
        self.assertEqual(strategy_cats, matrix_cats)

    def test_tail_spend_identifies_tail_suppliers(self):
        tail = self.agent._analyse_tail_spend(SAMPLE_SPEND_ROWS,
                                               self.agent._aggregate_spend(SAMPLE_SPEND_ROWS))
        self.assertGreater(tail["tail_supplier_count"], 0)
        self.assertGreaterEqual(tail["tail_spend_usd"], 0)

    def test_tail_spend_empty_input(self):
        tail = self.agent._analyse_tail_spend([], {})
        self.assertEqual(tail["tail_spend_usd"], 0.0)
        self.assertEqual(tail["tail_supplier_count"], 0)

    def test_pcard_candidates_low_avg_transaction(self):
        # Office Supplies: $3,500 total / 1 transaction = $3,500 avg (above $2,500 but 1 tx — won't qualify)
        # Add many small MRO transactions
        agg = self.agent._aggregate_spend(SAMPLE_SPEND_ROWS)
        tail = self.agent._analyse_tail_spend(SAMPLE_SPEND_ROWS, agg)
        # MRO Supplies has many $800 transactions → avg well below $2,500 → P-card candidate
        pcard_cats = [p["category"] for p in tail["pcard_candidates"]]
        self.assertIn("MRO Supplies", pcard_cats)

    @patch("agt_ss.agents.spend_category.extract_json")
    def test_make_vs_buy_uses_llm(self, mock_extract):
        mock_extract.return_value = {
            "recommendation": "buy",
            "cost_advantage": "buy",
            "strategic_factors": ["Not core competency"],
            "risk_factors": ["Single source"],
            "rationale": "External supplier more cost-effective.",
        }
        result = self.agent._make_vs_buy(
            "Titanium Fasteners",
            {"Titanium Fasteners": {"total_spend": 295_000.0}},
            {"internal_cost_usd": 350_000.0, "annual_volume_units": 10000},
        )
        self.assertEqual(result["recommendation"], "buy")
        self.assertIn("cost_delta_usd", result)

    @patch("agt_ss.agents.spend_category.extract_json", side_effect=Exception("API down"))
    def test_make_vs_buy_fallback_on_llm_failure(self, _mock):
        result = self.agent._make_vs_buy(
            "Fasteners",
            {"Fasteners": {"total_spend": 100_000.0}},
            {"internal_cost_usd": 120_000.0},
        )
        # Falls back to cost comparison: external < internal → buy
        # external_cost(100k) < internal_cost(120k): fallback recommends "buy"
        self.assertEqual(result["recommendation"], "buy")


# ===========================================================================
# Test SupplierMarketAgent
# ===========================================================================

class TestSupplierMarketAgent(unittest.TestCase):

    def setUp(self):
        from agt_ss.agents.supplier_market import SupplierMarketAgent
        self.agent = SupplierMarketAgent()

    def test_can_run_requires_kraljic(self):
        s = _base_state()
        self.assertFalse(self.agent.can_run(s))
        s["spend_category"] = {"kraljic_matrix": {"strategic": [{"category": "Titanium"}]}}
        self.assertTrue(self.agent.can_run(s))

    def test_compute_tlc_domestic_cheapest(self):
        from agt_ss.agents.supplier_market import SupplierMarketAgent
        agent = SupplierMarketAgent()
        domestic = agent._compute_tlc(100.0, 1000, "domestic", "direct_material", {})
        asia     = agent._compute_tlc(100.0, 1000, "asia_pacific", "direct_material", {})
        # Domestic should have lower freight and no duty
        self.assertLess(domestic["freight_pct"], asia["freight_pct"])
        self.assertEqual(domestic["duty_pct"], 0.0)

    def test_compute_tlc_components_sum_to_total(self):
        from agt_ss.agents.supplier_market import SupplierMarketAgent
        agent = SupplierMarketAgent()
        result = agent._compute_tlc(100.0, 1000, "europe", "direct_material", {})
        # adjusted_unit_cost × volume + safety_stock ≈ total_landed_cost
        computed = result["adjusted_unit_cost"] * 1000 + result["safety_stock_cost"]
        self.assertAlmostEqual(result["total_landed_cost"], round(computed, 2), places=1)

    def test_qualify_supplier_fallback_on_llm_failure(self):
        with patch("agt_ss.agents.supplier_market.extract_json", side_effect=Exception("LLM down")):
            result = self.agent._qualify_supplier(
                {"supplier_id": "SUP-X", "name": "Test Supplier", "hq_country": "US",
                 "financial_health": "stable", "category": "Titanium"},
                {}
            )
        self.assertEqual(result["status"], "pending_audit")
        self.assertTrue(result["recommended_audit"])

    @patch("agt_ss.agents.supplier_market.extract_json")
    def test_qualify_supplier_approved(self, mock_extract):
        mock_extract.return_value = {
            "qualification_score": 85.0,
            "status": "approved",
            "strengths": ["AS9100 certified"],
            "risks": [],
            "recommended_audit": False,
            "qualification_notes": "Strong aerospace track record.",
        }
        result = self.agent._qualify_supplier(
            {"supplier_id": "SUP-A", "name": "AeroFast", "category": "Fasteners",
             "financial_health": "strong", "hq_country": "US"},
            {}
        )
        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["qualification_score"], 85.0)

    def test_build_shortlist_respects_max_size(self):
        asl = [
            {"supplier_id": f"SUP-{i}", "name": f"Sup {i}",
             "status": "approved", "financial_health": "strong",
             "qualification_score": 90 - i}
            for i in range(10)
        ]
        shortlist = self.agent._build_shortlist(
            asl, [],
            {"strategic": [{"category": "Titanium"}], "leverage": [], "bottleneck": [], "non_critical": []},
            {}
        )
        # Strategic max = 5
        self.assertLessEqual(len(shortlist), 5)

    def test_build_shortlist_excludes_rejected(self):
        asl = [
            {"supplier_id": "SUP-GOOD", "status": "approved", "financial_health": "stable",
             "qualification_score": 80},
            {"supplier_id": "SUP-BAD",  "status": "rejected", "financial_health": "weak",
             "qualification_score": 30},
        ]
        shortlist = self.agent._build_shortlist(
            asl, [], {"leverage": [{"category": "X"}], "strategic": [], "bottleneck": [], "non_critical": []}, {}
        )
        ids = [s["supplier_id"] for s in shortlist]
        self.assertNotIn("SUP-BAD", ids)

    def test_build_shortlist_excludes_weak_financial_for_strategic(self):
        asl = [
            {"supplier_id": "SUP-WEAK", "status": "approved", "financial_health": "weak",
             "qualification_score": 95},
            {"supplier_id": "SUP-OK",   "status": "approved", "financial_health": "stable",
             "qualification_score": 75},
        ]
        shortlist = self.agent._build_shortlist(
            asl, [],
            {"strategic": [{"category": "X"}], "leverage": [], "bottleneck": [], "non_critical": []},
            {}
        )
        ids = [s["supplier_id"] for s in shortlist]
        self.assertNotIn("SUP-WEAK", ids)


# ===========================================================================
# Test SourcingExecutionAgent
# ===========================================================================

class TestSourcingExecutionAgent(unittest.TestCase):

    def setUp(self):
        from agt_ss.agents.sourcing_execution import SourcingExecutionAgent
        self.agent = SourcingExecutionAgent()

    def _state_with_deps(self, **ctx) -> dict:
        s = _base_state(context={
            "industry": "aerospace / defense",
            "annual_volume_units": 10000,
            "target_unit_price": 10.0,
            "contract_years": 3,
            **ctx,
        })
        s["spend_category"] = {
            "category_strategies": [{"category": "Titanium Fasteners",
                                     "quadrant": "leverage", "approach": "competitive_bidding"}]
        }
        s["supplier_market"] = {
            "supplier_shortlist": [
                {"supplier_id": "SUP-A", "name": "AeroFast", "status": "approved",
                 "financial_health": "strong", "qualification_score": 88},
                {"supplier_id": "SUP-B", "name": "MetalSupply", "status": "approved",
                 "financial_health": "stable", "qualification_score": 75},
                {"supplier_id": "SUP-C", "name": "FastFix", "status": "approved",
                 "financial_health": "stable", "qualification_score": 70},
            ]
        }
        return s

    def test_can_run_requires_shortlist_and_strategies(self):
        self.assertFalse(self.agent.can_run(_base_state()))
        self.assertTrue(self.agent.can_run(self._state_with_deps()))

    def test_determine_event_type_by_quadrant(self):
        for quadrant, expected in [("strategic", "RFP"), ("leverage", "RFQ"),
                                    ("bottleneck", "RFI"), ("non_critical", "RFQ")]:
            strategies = [{"quadrant": quadrant, "category": "X"}]
            result = self.agent._determine_event_type(strategies, {})
            self.assertEqual(result, expected)

    def test_event_type_override_respected(self):
        strategies = [{"quadrant": "leverage", "category": "X"}]
        result = self.agent._determine_event_type(strategies, {"event_type_override": "RFP"})
        self.assertEqual(result, "RFP")

    def test_normalize_bids_net45_cheaper_than_net30(self):
        bids = [
            {"supplier_id": "A", "unit_price": 100.0, "payment_terms": "net30", "tooling_cost": 0},
            {"supplier_id": "B", "unit_price": 100.0, "payment_terms": "net45", "tooling_cost": 0},
        ]
        normalized = self.agent._normalize_bids(bids, {"contract_years": 3, "annual_volume_units": 1000})
        price_net30 = next(b["unit_price_normalized"] for b in normalized if b["supplier_id"] == "A")
        price_net45 = next(b["unit_price_normalized"] for b in normalized if b["supplier_id"] == "B")
        self.assertLess(price_net45, price_net30)

    def test_tco_model_lower_defect_ppm_means_lower_quality_cost(self):
        s = self._state_with_deps()
        bids = [
            {"supplier_id": "SUP-A", "supplier_name": "AeroFast",
             "unit_price_normalized": 10.0, "unit_price": 10.0,
             "quality_defect_ppm": 100, "on_time_delivery_pct": 97,
             "lead_time_days": 15, "tooling_per_unit": 0},
            {"supplier_id": "SUP-B", "supplier_name": "MetalSupply",
             "unit_price_normalized": 10.0, "unit_price": 10.0,
             "quality_defect_ppm": 2000, "on_time_delivery_pct": 88,
             "lead_time_days": 30, "tooling_per_unit": 0},
        ]
        shortlist = s["supplier_market"]["supplier_shortlist"]
        tco = self.agent._build_tco_models(bids, shortlist, s)
        tco_a = next(m for m in tco if m["supplier_id"] == "SUP-A")
        tco_b = next(m for m in tco if m["supplier_id"] == "SUP-B")
        self.assertLess(tco_a["tco_components"]["quality_cost"],
                        tco_b["tco_components"]["quality_cost"])

    def test_tco_rank_1_is_lowest_cost(self):
        s = self._state_with_deps()
        bids = [
            {"supplier_id": "SUP-A", "supplier_name": "AeroFast",
             "unit_price_normalized": 9.0, "unit_price": 9.0,
             "quality_defect_ppm": 300, "on_time_delivery_pct": 97,
             "lead_time_days": 14, "tooling_per_unit": 0},
            {"supplier_id": "SUP-B", "supplier_name": "MetalSupply",
             "unit_price_normalized": 12.0, "unit_price": 12.0,
             "quality_defect_ppm": 800, "on_time_delivery_pct": 90,
             "lead_time_days": 30, "tooling_per_unit": 0},
        ]
        tco = self.agent._build_tco_models(bids, s["supplier_market"]["supplier_shortlist"], s)
        self.assertEqual(tco[0]["tco_rank"], 1)
        self.assertLessEqual(tco[0]["tco_per_unit"], tco[1]["tco_per_unit"])

    def test_evaluate_bids_ranks_lower_cost_higher(self):
        s = self._state_with_deps()
        tco = [
            {"supplier_id": "SUP-A", "tco_per_unit": 9.5, "tco_rank": 1,
             "tco_contract_total": 285_000},
            {"supplier_id": "SUP-B", "tco_per_unit": 12.5, "tco_rank": 2,
             "tco_contract_total": 375_000},
        ]
        bids = [
            {"supplier_id": "SUP-A", "supplier_name": "AeroFast",
             "quality_defect_ppm": 300, "on_time_delivery_pct": 97, "technical_score": 8.5},
            {"supplier_id": "SUP-B", "supplier_name": "MetalSupply",
             "quality_defect_ppm": 800, "on_time_delivery_pct": 90, "technical_score": 7.0},
        ]
        evaluation = self.agent._evaluate_bids(bids, tco, "RFQ",
                                                s["supplier_market"]["supplier_shortlist"], s)
        self.assertEqual(evaluation["ranked"][0], "SUP-A")

    @patch("agt_ss.agents.sourcing_execution.reason", return_value="Award SUP-A for lowest TCO.")
    def test_award_recommendation_winner_is_top_ranked(self, _mock_reason):
        s = self._state_with_deps()
        evaluation = {
            "ranked": ["SUP-A", "SUP-B"],
            "scores": {
                "SUP-A": {"supplier_name": "AeroFast", "weighted_total": 82.0},
                "SUP-B": {"supplier_name": "MetalSupply", "weighted_total": 71.0},
            }
        }
        tco = [
            {"supplier_id": "SUP-A", "tco_per_unit": 9.5, "tco_rank": 1, "tco_components": {},
             "tco_contract_total": 285_000},
            {"supplier_id": "SUP-B", "tco_per_unit": 12.5, "tco_rank": 2, "tco_components": {},
             "tco_contract_total": 375_000},
        ]
        award = self.agent._generate_award_recommendation(evaluation, tco, [], s)
        self.assertEqual(award["supplier_id"], "SUP-A")
        self.assertEqual(award["batna_supplier_id"], "SUP-B")

    def test_scorecards_sorted_by_rank(self):
        evaluation = {
            "ranked": ["SUP-A", "SUP-B", "SUP-C"],
            "scores": {
                "SUP-A": {"supplier_name": "A", "weighted_total": 85, "criterion_scores": {}},
                "SUP-B": {"supplier_name": "B", "weighted_total": 75, "criterion_scores": {}},
                "SUP-C": {"supplier_name": "C", "weighted_total": 65, "criterion_scores": {}},
            }
        }
        shortlist = [{"supplier_id": f"SUP-{x}", "name": x} for x in ("A", "B", "C")]
        cards = self.agent._generate_scorecards(evaluation, shortlist)
        ranks = [c["rank"] for c in cards]
        self.assertEqual(ranks, sorted(ranks))


# ===========================================================================
# Test ContractSupplierAgent
# ===========================================================================

class TestContractSupplierAgent(unittest.TestCase):

    def setUp(self):
        from agt_ss.agents.contract_supplier import ContractSupplierAgent
        self.agent = ContractSupplierAgent()

    def _state_with_deps(self) -> dict:
        s = _base_state(context={
            "industry": "aerospace / defense",
            "annual_volume_units": 10000,
            "contract_years": 3,
            "target_unit_price": 9.5,
            "target_lead_time_days": 14,
        })
        s["sourcing_execution"] = {
            "award_recommendation": {
                "supplier_id": "SUP-A",
                "supplier_name": "AeroFast Inc",
                "evaluation_score": 82.0,
                "tco_per_unit": 9.5,
                "tco_contract_total": 285_000.0,
                "tco_rank": 1,
                "batna_supplier_id": "SUP-B",
                "rationale": "Best TCO and quality.",
            },
            "tco_comparison": [
                {"supplier_id": "SUP-A", "tco_per_unit": 9.5, "tco_components": {
                    "unit_price": 9.0, "quality_cost": 0.1, "logistics": 0.3,
                    "tooling_amortized": 0.05, "disruption_risk": 0.05},
                 "tco_contract_total": 285_000.0},
                {"supplier_id": "SUP-B", "tco_per_unit": 12.5, "tco_components": {},
                 "tco_contract_total": 375_000.0},
            ],
            "bid_evaluation_matrix": {
                "scores": {"SUP-A": {"weighted_total": 82}, "SUP-B": {"weighted_total": 71}},
                "ranked": ["SUP-A", "SUP-B"],
            },
        }
        s["checkpoint_history"] = [{
            "gate": CheckpointGate.CONTRACT_AWARD_APPROVAL,
            "decision": "approved",
            "approved_by": "evan@cocomgroup.com",
        }]
        return s

    def test_can_run_blocked_without_gate_approval(self):
        s = self._state_with_deps()
        s["checkpoint_history"] = []
        self.assertFalse(self.agent.can_run(s))

    def test_can_run_unblocked_with_gate_and_award(self):
        self.assertTrue(self.agent.can_run(self._state_with_deps()))

    @patch("agt_ss.agents.contract_supplier.extract_json")
    def test_negotiation_strategy_price_targets(self, mock_extract):
        mock_extract.return_value = {
            "key_leverage_points": ["BATNA availability"],
            "concession_sequence": [],
            "clause_priorities": [],
            "anchoring_strategy": "Cost transparency",
            "tactics": [],
            "risk_mitigations": [],
        }
        s = self._state_with_deps()
        award    = s["sourcing_execution"]["award_recommendation"]
        tco_list = s["sourcing_execution"]["tco_comparison"]
        strategy = self.agent._build_negotiation_strategy(award, tco_list, {}, {}, s["goal"])

        # Target should be below current price (7% reduction target)
        self.assertLess(strategy["target_price"], award["tco_per_unit"])
        # Opening anchor should be below target
        self.assertLess(strategy["opening_position"], strategy["target_price"])
        # Walk-away should be above target
        self.assertGreater(strategy["walk_away_price"], strategy["target_price"])

    def test_contract_draft_contains_required_clauses(self):
        with patch("agt_ss.agents.contract_supplier.extract_json", return_value={}), \
             patch("agt_ss.agents.contract_supplier.reason", return_value="Bespoke clause text."):
            s = self._state_with_deps()
            award    = s["sourcing_execution"]["award_recommendation"]
            strategy = {
                "target_price": 9.0, "walk_away_price": 9.8,
                "clause_priorities": [], "concessions": {},
            }
            contract = self.agent._draft_contract(strategy, award, s)

        self.assertIn("contract_id", contract)
        self.assertIn("pricing",   contract["clauses"])
        self.assertIn("delivery",  contract["clauses"])
        self.assertIn("quality_sla", contract["clauses"])
        self.assertIn("ip_ownership", contract["clauses"])
        self.assertEqual(contract["status"], "draft")

    def test_contract_id_format(self):
        with patch("agt_ss.agents.contract_supplier.extract_json", return_value={}), \
             patch("agt_ss.agents.contract_supplier.reason", return_value=""):
            s = self._state_with_deps()
            contract = self.agent._draft_contract(
                {"target_price": 9.0, "clause_priorities": [], "concessions": {}},
                s["sourcing_execution"]["award_recommendation"],
                s,
            )
        self.assertTrue(contract["contract_id"].startswith("CNT-"))

    def test_onboarding_checklist_completeness(self):
        onboarding = self.agent._onboard_supplier("SUP-A", "AeroFast", {})
        self.assertIn("erp_vendor_created",          onboarding["checklist"])
        self.assertIn("edi_connectivity_tested",     onboarding["checklist"])
        self.assertIn("banking_info_collected",      onboarding["checklist"])
        self.assertEqual(onboarding["items_completed"], 0)

    def test_onboarding_direct_material_extra_steps(self):
        onboarding = self.agent._onboard_supplier(
            "SUP-A", "AeroFast", {"spend_type": "direct_material"})
        self.assertIn("first_article_scheduled",    onboarding["checklist"])
        self.assertIn("ppap_package_submitted",      onboarding["checklist"])

    @patch("agt_ss.agents.contract_supplier.extract_json")
    def test_sustainability_assessment_returns_score(self, mock_extract):
        mock_extract.return_value = {
            "overall_score": 78.0,
            "coc_compliant": True,
            "dimension_scores": {},
            "flags": [],
            "remediation_required": False,
            "remediation_items": [],
            "coc_signed": True,
            "carbon_disclosure_available": True,
        }
        result = self.agent._assess_sustainability("SUP-A", "AeroFast", {})
        self.assertEqual(result["overall_score"], 78.0)
        self.assertTrue(result["coc_compliant"])

    @patch("agt_ss.agents.contract_supplier.extract_json", side_effect=Exception("LLM down"))
    def test_sustainability_fallback(self, _mock):
        result = self.agent._assess_sustainability("SUP-A", "AeroFast", {})
        self.assertFalse(result["coc_compliant"])
        self.assertTrue(result["remediation_required"])

    def test_performance_baseline_has_all_kpis(self):
        s = self._state_with_deps()
        baseline = self.agent._set_performance_baseline(
            s["sourcing_execution"]["award_recommendation"],
            s["sourcing_execution"]["tco_comparison"],
            s["context"],
        )
        kpis = baseline["kpis"]
        for kpi in ("on_time_delivery_pct", "quality_defect_ppm",
                     "purchase_price_variance_pct", "invoice_accuracy_pct"):
            self.assertIn(kpi, kpis)


# ===========================================================================
# Test AnalyticsGovernanceAgent
# ===========================================================================

class TestAnalyticsGovernanceAgent(unittest.TestCase):

    def setUp(self):
        from agt_ss.agents.analytics_governance import AnalyticsGovernanceAgent
        self.agent = AnalyticsGovernanceAgent()

    def test_can_run_always_with_goal(self):
        self.assertTrue(self.agent.can_run(_base_state()))

    def test_ppv_report_favourable_flag(self):
        actuals = [
            {"category": "Titanium Fasteners", "supplier_id": "SUP-A",
             "unit_price": 9.0, "quantity": 10000},
        ]
        budget = {"Titanium Fasteners": {"unit_price": 10.0}}
        s = _base_state(context={"actuals_data": actuals, "budget_data": budget})
        ppv = self.agent._build_ppv_report(s)
        self.assertEqual(len(ppv["variances"]), 1)
        v = ppv["variances"][0]
        self.assertEqual(v["flag"], "favourable")
        self.assertAlmostEqual(v["ppv_vs_budget"], -10_000.0)

    def test_ppv_report_unfavourable_flag(self):
        actuals = [{"category": "MRO", "supplier_id": "SUP-B",
                    "unit_price": 12.0, "quantity": 5000}]
        budget  = {"MRO": {"unit_price": 10.0}}
        s = _base_state(context={"actuals_data": actuals, "budget_data": budget})
        ppv = self.agent._build_ppv_report(s)
        v = ppv["variances"][0]
        self.assertEqual(v["flag"], "unfavourable")
        self.assertAlmostEqual(v["ppv_vs_budget"], 10_000.0)

    def test_ppv_total_is_sum_of_variances(self):
        actuals = [
            {"category": "Cat-A", "supplier_id": "S1", "unit_price": 8.0, "quantity": 1000},
            {"category": "Cat-B", "supplier_id": "S2", "unit_price": 15.0, "quantity": 500},
        ]
        budget = {"Cat-A": {"unit_price": 10.0}, "Cat-B": {"unit_price": 10.0}}
        s = _base_state(context={"actuals_data": actuals, "budget_data": budget})
        ppv = self.agent._build_ppv_report(s)
        # Cat-A: (8-10) × 1000 = -2000 (favourable), Cat-B: (15-10) × 500 = +2500 (unfav)
        self.assertAlmostEqual(ppv["total_ppv_usd"], 500.0)  # net

    def test_ppv_empty_returns_zero(self):
        ppv = self.agent._build_ppv_report(_base_state())
        self.assertEqual(ppv["total_ppv_usd"], 0.0)

    def test_monthly_dashboard_fields_present(self):
        s = _base_state()
        s["spend_category"] = {
            "spend_classification": {"Titanium": {"total_spend": 100_000,
                                                  "spend_share_pct": 100,
                                                  "supplier_count": 2}},
            "tail_spend_report": {"maverick_buying_flags": []},
        }
        ppv = self.agent._build_ppv_report(s)
        dashboard = self.agent._build_monthly_dashboard(s, ppv)
        for field in ("total_managed_spend", "savings_realized", "contract_coverage_pct",
                       "po_compliance_pct", "generated_at"):
            self.assertIn(field, dashboard)

    def test_savings_pipeline_stages(self):
        s = _base_state()
        s["spend_category"] = {"category_strategies": [
            {"category": "Titanium", "quadrant": "leverage",
             "approach": "competitive_bidding", "profit_impact": 8.0}
        ]}
        s["sourcing_execution"] = {"award_recommendation": None, "event_type": None}
        pipeline = self.agent._build_savings_pipeline(s)
        self.assertGreater(len(pipeline), 0)
        valid_stages = {"identification", "analysis", "negotiation", "contracted", "realised"}
        for item in pipeline:
            self.assertIn(item["stage"], valid_stages)

    def test_contract_coverage_identifies_uncovered(self):
        s = _base_state()
        s["spend_category"] = {"spend_classification": {
            "Titanium": {"total_spend": 100_000, "spend_share_pct": 90, "supplier_count": 2},
            "MRO":      {"total_spend": 10_000,  "spend_share_pct": 10, "supplier_count": 5},
        }}
        s["category"] = None  # no contract executed
        coverage = self.agent._build_contract_coverage(s)
        self.assertGreater(coverage["uncovered_spend"], 0)
        self.assertGreater(len(coverage["high_risk_uncovered"]), 0)

    def test_contract_coverage_marks_executed_category(self):
        s = _base_state()
        s["category"] = "Titanium Fasteners"
        s["spend_category"] = {"spend_classification": {
            "Titanium Fasteners": {"total_spend": 100_000, "spend_share_pct": 100,
                                    "supplier_count": 2},
        }}
        s["contract_supplier"] = {"contract_record": {"status": "pending_signature"}}
        coverage = self.agent._build_contract_coverage(s)
        self.assertTrue(coverage["categories"]["Titanium Fasteners"]["is_covered"])
        self.assertAlmostEqual(coverage["coverage_pct"], 100.0)

    @patch("agt_ss.agents.analytics_governance.extract_json")
    def test_maturity_roadmap_structure(self, mock_extract):
        mock_extract.return_value = {
            "dimension_scores":    {"spend_visibility": 3.5},
            "overall_score":       3.0,
            "current_level":       3,
            "current_level_label": "Defined / Structured",
            "target_level":        4,
            "target_level_label":  "Managed / Optimising",
            "initiatives":         [{"title": "Spend cube", "priority": "quick_win",
                                     "effort_months": 2, "timeline_start_months": 1}],
            "key_gaps":            ["risk_management"],
            "executive_summary":   "Procurement is at level 3.",
        }
        s = _base_state()
        s["spend_category"] = {"spend_classification": {"X": {"total_spend": 100_000}},
                                "category_strategies": []}
        dashboard = {"savings_realized": 0, "contract_coverage_pct": 60,
                     "po_compliance_pct": 85}
        roadmap = self.agent._build_maturity_roadmap(s, dashboard)
        self.assertIn("current_level", roadmap)
        self.assertIn("initiatives",   roadmap)
        self.assertIn("maturity_levels", roadmap)

    @patch("agt_ss.agents.analytics_governance.extract_json", side_effect=Exception("LLM down"))
    def test_maturity_roadmap_fallback(self, _mock):
        s = _base_state()
        s["spend_category"] = {"spend_classification": {}, "category_strategies": []}
        dashboard = {"savings_realized": 0, "contract_coverage_pct": 0, "po_compliance_pct": 0}
        roadmap = self.agent._build_maturity_roadmap(s, dashboard)
        self.assertIn("current_level", roadmap)
        self.assertIn("executive_summary", roadmap)
        self.assertGreater(len(roadmap["key_gaps"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)