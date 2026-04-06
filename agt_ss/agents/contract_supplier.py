# Databricks notebook source
"""
agt_ss.agents.contract_supplier
---------------------------------
Agent 4: Contract & Supplier Management

Processes owned (5):
  1. Negotiation strategy — develop target price, BATNA, concession plan,
     and clause priority stack based on TCO models and bid evaluation.
  2. Contract drafting & execution — generate contract from clause library,
     route for approval, execute via DocuSign or equivalent.
  3. Supplier onboarding — ERP/SAP system setup, EDI connectivity,
     performance baseline establishment.
  4. Supplier code of conduct & sustainability requirements — evaluate against
     CoC framework, ESG scoring, and sustainability sourcing criteria.
  5. Supplier performance baseline — set KPI targets for quality (PPM),
     on-time delivery (OTD), cost variance, and responsiveness.

Data flow:
  IN  ← state.sourcing_execution.award_recommendation
        state.sourcing_execution.tco_comparison
        state.sourcing_execution.bid_evaluation_matrix
        state.checkpoint_history (CONTRACT_AWARD_APPROVAL must be approved)
  OUT → state.contract_supplier {negotiation_strategy, contract_record,
                                  onboarding_status, sustainability_ratings,
                                  performance_baseline}
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from .base import AgentToolError, BaseAgent
from .llm import extract_json, reason
from ..state.schema import AgentName, CheckpointGate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contract clause library — production version would pull from Delta Lake
# ---------------------------------------------------------------------------

_CLAUSE_TEMPLATES: dict[str, str] = {
    "pricing": (
        "Unit price shall be {unit_price} USD per unit, firm for the initial "
        "{price_firm_months} months. Thereafter, price adjustments shall be "
        "limited to the lesser of (a) CPI index change or (b) {max_escalation_pct}% "
        "per annum, with 90 days written notice."
    ),
    "delivery": (
        "Supplier shall deliver per agreed schedule with a minimum {otd_target_pct}% "
        "on-time delivery rate. Delivery within ±{delivery_window_days} calendar days "
        "of requested date is acceptable. Expedite fees apply for late delivery."
    ),
    "quality_sla": (
        "Supplier shall maintain a maximum defect rate of {max_defect_ppm} PPM. "
        "Quality escapes shall be remediated within {response_days} business days. "
        "Supplier shall maintain ISO {quality_standard} certification throughout the term."
    ),
    "ip_ownership": (
        "All designs, specifications, and tooling funded by Buyer remain the exclusive "
        "property of Buyer. Supplier grants Buyer a perpetual, royalty-free license "
        "to any improvements made to Buyer-funded tooling."
    ),
    "audit_rights": (
        "Buyer retains the right to audit Supplier's facilities, quality systems, "
        "financial records, and subcontractor relationships upon {audit_notice_days} "
        "days written notice, not more than {max_audits_per_year} times per year."
    ),
    "termination": (
        "Either party may terminate for convenience with {termination_notice_days} days "
        "written notice. Buyer may terminate for cause immediately upon written notice "
        "if Supplier fails to cure a material breach within {cure_period_days} days."
    ),
    "sustainability": (
        "Supplier agrees to comply with Buyer's Supplier Code of Conduct. Supplier shall "
        "achieve a minimum sustainability score of {min_sustainability_score}/100 and "
        "report Scope 1 and Scope 2 emissions annually."
    ),
    "force_majeure": (
        "Neither party shall be liable for delays caused by events beyond reasonable "
        "control, provided the affected party gives prompt notice and uses commercially "
        "reasonable efforts to mitigate the impact."
    ),
}

# ---------------------------------------------------------------------------
# CoC and sustainability assessment dimensions
# ---------------------------------------------------------------------------

_SUSTAINABILITY_DIMENSIONS = [
    "labor_practices",
    "environmental_management",
    "health_and_safety",
    "business_ethics",
    "conflict_minerals",
    "carbon_disclosure",
    "diversity_equity_inclusion",
    "subcontractor_management",
]


class ContractSupplierAgent(BaseAgent):
    """Full implementation of Contract & Supplier Management agent."""

    name = AgentName.CONTRACT_SUPPLIER

    def can_run(self, state: dict) -> bool:
        award    = state.get("sourcing_execution", {}).get("award_recommendation")
        approved = self._is_gate_approved(state, CheckpointGate.CONTRACT_AWARD_APPROVAL)
        return bool(award and award.get("supplier_id")) and approved

    def _is_gate_approved(self, state: dict, gate: str) -> bool:
        for rec in state.get("checkpoint_history", []):
            if rec.get("gate") == gate and rec.get("decision") == "approved":
                return True
        return False

    # =========================================================================
    # Orchestration
    # =========================================================================

    def _execute(self, state: dict) -> dict:
        award      = state["sourcing_execution"]["award_recommendation"]
        tco_list   = state.get("sourcing_execution", {}).get("tco_comparison", [])
        evaluation = state.get("sourcing_execution", {}).get("bid_evaluation_matrix", {})
        context    = state.get("context", {})
        goal       = state.get("goal", "")

        supplier_id   = award["supplier_id"]
        supplier_name = award.get("supplier_name", supplier_id)

        logger.info("[ContractSupplier] Starting — supplier=%s", supplier_id)

        # Step 1 — build negotiation strategy
        neg_strategy = self._build_negotiation_strategy(
            award, tco_list, evaluation, context, goal)

        # Step 2 — draft contract
        contract_draft = self._draft_contract(neg_strategy, award, state)

        # Step 3 — simulate execution (DocuSign routing)
        contract_executed = self._execute_contract(contract_draft, context)

        # Step 4 — sustainability & CoC assessment
        sustainability = self._assess_sustainability(supplier_id, supplier_name, context)

        # Step 5 — supplier onboarding
        onboarding = self._onboard_supplier(supplier_id, supplier_name, context)

        # Step 6 — performance baseline
        baseline = self._set_performance_baseline(award, tco_list, context)

        logger.info("[ContractSupplier] Complete — contract=%s status=%s",
                    contract_executed.get("contract_id"),
                    contract_executed.get("status"))

        return {
            "contract_supplier": {
                "negotiation_strategy":   neg_strategy,
                "contract_record":        contract_executed,
                "onboarding_status":      onboarding,
                "sustainability_ratings": [sustainability],
                "performance_baseline":   baseline,
            }
        }

    # =========================================================================
    # Step 1 — Negotiation strategy
    # =========================================================================

    def _build_negotiation_strategy(self, award: dict, tco_list: list[dict],
                                     evaluation: dict, context: dict,
                                     goal: str) -> dict:
        """
        Build a comprehensive negotiation strategy document:
          - Target price derived from TCO best case
          - Walk-away price derived from budget constraint or 110% of target
          - BATNA = second-ranked supplier's TCO
          - Concession sequence by priority
          - Clause priority stack
          - Anchoring and leverage analysis
        """
        winner_id  = award["supplier_id"]
        winner_tco = next((m for m in tco_list if m["supplier_id"] == winner_id), {})
        batna_id   = award.get("batna_supplier_id")
        batna_tco  = next((m for m in tco_list if m["supplier_id"] == batna_id), {}) if batna_id else {}

        # Derived pricing targets
        current_price    = winner_tco.get("tco_per_unit", 0)
        batna_price      = batna_tco.get("tco_per_unit", current_price * 1.05)
        target_price     = current_price * 0.93     # target: 7% below quoted TCO
        walk_away_price  = batna_price * 0.98       # we walk if > BATNA - 2%
        opening_position = current_price * 0.88     # anchor: 12% below quoted

        # LLM-generated negotiation playbook
        try:
            playbook = extract_json(
                system=(
                    "You are a senior procurement negotiator. Build a negotiation strategy. "
                    "Return JSON:\n"
                    "{\n"
                    "  key_leverage_points: [str],\n"
                    "  concession_sequence: [{round, concession, condition, value_to_us}],\n"
                    "  clause_priorities: [{clause, priority: 'must_have'|'want'|'tradeable', "
                    "    rationale: str}],\n"
                    "  anchoring_strategy: str,\n"
                    "  tactics: [str],\n"
                    "  risk_mitigations: [str]\n"
                    "}"
                ),
                user=(
                    f"Supplier: {winner_id}\n"
                    f"Evaluation score: {award.get('evaluation_score')}/100\n"
                    f"Our target price: ${target_price:,.4f}/unit\n"
                    f"Opening anchor: ${opening_position:,.4f}/unit\n"
                    f"Walk-away: ${walk_away_price:,.4f}/unit\n"
                    f"BATNA supplier: {batna_id} @ ${batna_price:,.4f}/unit\n"
                    f"Key TCO components: {winner_tco.get('tco_components', {})}\n"
                    f"Award rationale: {award.get('rationale', '')[:300]}\n"
                    f"Goal: {goal}\n"
                    "Consider: pricing, lead time, payment terms, quality SLAs, "
                    "tooling ownership, volume flexibility, and contract term."
                ),
                max_tokens=1200,
            )
        except Exception as exc:
            logger.warning("[ContractSupplier] Negotiation playbook LLM failed: %s", exc)
            playbook = {
                "key_leverage_points":  ["BATNA availability", "Volume commitment"],
                "concession_sequence":  [],
                "clause_priorities":    [],
                "anchoring_strategy":   "Open with cost-model transparency",
                "tactics":              ["Present TCO analysis", "Highlight BATNA"],
                "risk_mitigations":     [],
            }

        return {
            "supplier_id":        winner_id,
            "opening_position":   round(opening_position, 4),
            "target_price":       round(target_price, 4),
            "walk_away_price":    round(walk_away_price, 4),
            "batna_supplier_id":  batna_id,
            "batna_price":        round(batna_price, 4),
            "price_gap_pct":      round((walk_away_price - target_price) / target_price * 100, 2),
            "concessions": {
                "price_ceiling":      round(walk_away_price, 4),
                "lead_time_days":     context.get("target_lead_time_days", 21),
                "payment_terms":      "net45",   # give up from net30 if needed
                "volume_flexibility": "±15%",
            },
            **playbook,
            "created_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Step 2 — Contract drafting
    # =========================================================================

    def _draft_contract(self, neg_strategy: dict, award: dict,
                         state: dict) -> dict:
        """
        Generate contract document by populating clause templates with
        negotiated commercial terms.  LLM generates bespoke clauses where
        standard templates are insufficient.
        """
        context     = state.get("context", {})
        supplier_id = award["supplier_id"]
        annual_units= context.get("annual_volume_units", 1000)
        contract_years = context.get("contract_years", 3)
        target_price   = neg_strategy.get("target_price", 0)

        # Populate standard clause templates
        clause_params = {
            "unit_price":           f"{target_price:,.4f}",
            "price_firm_months":    "24",
            "max_escalation_pct":   "3",
            "otd_target_pct":       "95",
            "delivery_window_days": "3",
            "max_defect_ppm":       "500",
            "response_days":        "2",
            "quality_standard":     "9001:2015",
            "audit_notice_days":    "10",
            "max_audits_per_year":  "2",
            "termination_notice_days": "90",
            "cure_period_days":     "30",
            "min_sustainability_score": "70",
        }

        populated_clauses = {
            name: template.format(**{k: v for k, v in clause_params.items()
                                     if f"{{{k}}}" in template})
            for name, template in _CLAUSE_TEMPLATES.items()
        }

        # LLM generates any bespoke clauses flagged as must_have
        bespoke_clauses = {}
        must_have_clauses = [
            cp for cp in neg_strategy.get("clause_priorities", [])
            if cp.get("priority") == "must_have"
            and cp.get("clause") not in populated_clauses
        ]

        for clause_item in must_have_clauses[:3]:  # cap at 3 bespoke clauses
            clause_name = clause_item.get("clause", "")
            try:
                bespoke_text = reason(
                    system=(
                        "You are a procurement contracts attorney. "
                        "Draft a clear, enforceable contract clause in plain English. "
                        "50-100 words. No legalese."
                    ),
                    user=(
                        f"Clause type: {clause_name}\n"
                        f"Rationale: {clause_item.get('rationale', '')}\n"
                        f"Supplier: {supplier_id}\n"
                        f"Category: {state.get('category', '')}"
                    ),
                    max_tokens=200,
                )
                bespoke_clauses[clause_name] = bespoke_text
            except Exception as exc:
                logger.warning("[ContractSupplier] Bespoke clause LLM failed: %s", exc)

        contract_id = f"CNT-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

        return {
            "contract_id":         contract_id,
            "supplier_id":         supplier_id,
            "supplier_name":       award.get("supplier_name", ""),
            "category":            state.get("category", ""),
            "contract_type":       "fixed_price_supply_agreement",
            "effective_date":      datetime.utcnow().isoformat(),
            "expiry_date":         (datetime.utcnow() + timedelta(days=365 * contract_years)).isoformat(),
            "contract_years":      contract_years,
            "unit_price":          target_price,
            "annual_volume":       annual_units,
            "total_value":         round(target_price * annual_units * contract_years, 2),
            "payment_terms":       "net30",
            "incoterms":           "DDP",
            "clauses":             {**populated_clauses, **bespoke_clauses},
            "clause_count":        len(populated_clauses) + len(bespoke_clauses),
            "status":              "draft",
            "drafted_at":          datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Step 3 — Contract execution
    # =========================================================================

    def _execute_contract(self, contract: dict, context: dict) -> dict:
        """
        Route contract for internal approval and initiate DocuSign workflow.

        Production implementation:
            from ..tools.docusign import DocuSignClient
            envelope_id = DocuSignClient().send_for_signature(contract)
            return {**contract, "status": "pending_signature", "envelope_id": envelope_id}
        """
        # INTEGRATION POINT: DocuSign / internal approval workflow
        logger.debug("[ContractSupplier] Contract routing — DocuSign not wired")

        approval_workflow = context.get("skip_docusign", False)
        status = "pending_signature" if not approval_workflow else "executed"

        return {
            **contract,
            "status":          status,
            "envelope_id":     f"ENV-{str(uuid.uuid4())[:12].upper()}",
            "routed_for_approval_at": datetime.utcnow().isoformat(),
            "signatories":     context.get("signatories", []),
        }

    # =========================================================================
    # Step 4 — Sustainability & CoC assessment
    # =========================================================================

    def _assess_sustainability(self, supplier_id: str, supplier_name: str,
                                context: dict) -> dict:
        """
        Evaluate supplier against CoC framework and ESG scoring dimensions.
        Produces an overall sustainability score and flags for remediation.
        """
        industry = context.get("industry", "aerospace / defense")

        try:
            assessment = extract_json(
                system=(
                    f"You are a supply chain sustainability analyst for {industry}. "
                    "Assess a supplier across ESG and CoC dimensions. Return JSON:\n"
                    "{\n"
                    "  overall_score: float (0-100),\n"
                    "  coc_compliant: bool,\n"
                    "  dimension_scores: {"
                    + ", ".join(f"  {d}: float" for d in _SUSTAINABILITY_DIMENSIONS)
                    + "},\n"
                    "  flags: [str],\n"
                    "  remediation_required: bool,\n"
                    "  remediation_items: [str],\n"
                    "  coc_signed: bool,\n"
                    "  carbon_disclosure_available: bool\n"
                    "}"
                ),
                user=(
                    f"Supplier: {supplier_name} (ID: {supplier_id})\n"
                    f"Industry: {industry}\n"
                    f"Known context: {context.get('supplier_esg_context', 'No prior data')}\n"
                    "Assume standard aerospace supplier profile if no specific data available."
                ),
                max_tokens=700,
            )
            return {
                "supplier_id":   supplier_id,
                "supplier_name": supplier_name,
                **assessment,
                "assessed_at":   datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.warning("[ContractSupplier] Sustainability LLM failed: %s", exc)
            return {
                "supplier_id":              supplier_id,
                "supplier_name":            supplier_name,
                "overall_score":            0.0,
                "coc_compliant":            False,
                "coc_signed":               False,
                "dimension_scores":         {d: 0.0 for d in _SUSTAINABILITY_DIMENSIONS},
                "flags":                    ["Assessment unavailable — manual review required"],
                "remediation_required":     True,
                "remediation_items":        ["Complete sustainability questionnaire"],
                "carbon_disclosure_available": False,
                "assessed_at":              datetime.utcnow().isoformat(),
            }

    # =========================================================================
    # Step 5 — Supplier onboarding
    # =========================================================================

    def _onboard_supplier(self, supplier_id: str, supplier_name: str,
                           context: dict) -> dict:
        """
        Initiate supplier onboarding workflow:
          - ERP/SAP master data setup
          - EDI connectivity test
          - Banking / payment information collection
          - Compliance documentation collection
          - First-article inspection scheduling (for direct materials)
        """
        is_direct_material = context.get("spend_type") == "direct_material"

        # INTEGRATION POINT: SAP vendor master API
        # from ..tools.sap import SAPVendorMaster
        # erp_setup = SAPVendorMaster().create_vendor(supplier_id)

        logger.debug("[ContractSupplier] ERP vendor master — SAP not wired for %s", supplier_id)

        checklist = {
            "erp_vendor_created":           False,
            "edi_connectivity_tested":      False,
            "banking_info_collected":       False,
            "w9_or_w8_received":            False,
            "insurance_certificate_received": False,
            "nda_executed":                 False,
            "quality_agreement_signed":     False,
        }

        if is_direct_material:
            checklist["first_article_scheduled"] = False
            checklist["ppap_package_submitted"]   = False
            checklist["production_part_approval"] = False

        total_items    = len(checklist)
        completed_items = sum(1 for v in checklist.values() if v)

        return {
            "supplier_id":        supplier_id,
            "supplier_name":      supplier_name,
            "onboarding_status":  "in_progress",
            "checklist":          checklist,
            "items_total":        total_items,
            "items_completed":    completed_items,
            "completion_pct":     round(completed_items / total_items * 100, 1),
            "is_direct_material": is_direct_material,
            "target_ready_date":  (datetime.utcnow() + timedelta(days=45)).isoformat(),
            "onboarding_start":   datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Step 6 — Performance baseline
    # =========================================================================

    def _set_performance_baseline(self, award: dict, tco_list: list[dict],
                                    context: dict) -> dict:
        """
        Establish KPI targets and measurement cadence for the awarded supplier.
        Targets are derived from the bid commitments captured during evaluation.
        """
        sid      = award["supplier_id"]
        tco      = next((m for m in tco_list if m["supplier_id"] == sid), {})

        # Derive targets from bid commitments or apply industry benchmarks
        return {
            "supplier_id":         sid,
            "kpis": {
                "on_time_delivery_pct": {
                    "target": 95.0,
                    "measurement": "monthly",
                    "source":      "ERP goods receipt vs. PO promise date",
                },
                "quality_defect_ppm": {
                    "target": 500,
                    "measurement": "monthly",
                    "source":      "incoming inspection + field returns",
                },
                "purchase_price_variance_pct": {
                    "target": 0.0,
                    "measurement": "monthly",
                    "source":      "SAP PO vs. invoice",
                },
                "invoice_accuracy_pct": {
                    "target": 99.0,
                    "measurement": "monthly",
                    "source":      "AP 3-way match exception rate",
                },
                "responsiveness_hours": {
                    "target": 4,
                    "measurement": "per_incident",
                    "source":      "issue log response timestamps",
                },
            },
            "review_cadence":      "monthly_scorecard",
            "quarterly_business_review": True,
            "annual_performance_review": True,
            "escalation_threshold_pct":  10.0,  # trigger escalation if any KPI misses by >10%
            "baseline_set_at":     datetime.utcnow().isoformat(),
        }