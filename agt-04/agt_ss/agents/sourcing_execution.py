# Databricks notebook source
"""
agt_ss.agents.sourcing_execution
---------------------------------
Agent 3: Sourcing Execution Agent

Processes owned (5):
  1. Sourcing event execution — end-to-end RFI, RFQ, RFP, and reverse auction
     event management: document generation, supplier distribution, bid collection,
     normalization, and scoring.
  2. Supplier qualification & evaluation — financial health, capability, quality
     history, and compliance scoring for each bidder.
  3. TCO model construction — per-supplier total cost of ownership incorporating
     price, quality cost, logistics, tooling, and supply disruption risk.
  4. Direct materials / BOM cost-down programs — identify cost reduction
     opportunities through design-to-cost, value engineering, and resourcing.
  5. Award recommendation — ranked supplier recommendation with full rationale.

Data flow:
  IN  ← state.spend_category.category_strategies (quadrant → event type)
        state.supplier_market.supplier_shortlist
        state.supplier_market.landed_cost_models
        state.context.bid_responses (optional injected bids for testing)
  OUT → state.sourcing_execution {event_type, event_packages, bid_evaluation_matrix,
                                   tco_comparison, award_recommendation,
                                   supplier_scorecards}
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime
from typing import Any

from .base import AgentToolError, BaseAgent
from .llm import extract_json, reason
from ..state.schema import AgentName

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights by event type
# ---------------------------------------------------------------------------

_SCORING_WEIGHTS: dict[str, dict[str, float]] = {
    "RFP": {
        "total_cost":        0.35,
        "technical_capability": 0.25,
        "quality_history":   0.15,
        "delivery":          0.10,
        "financial_health":  0.10,
        "sustainability":    0.05,
    },
    "RFQ": {
        "total_cost":        0.55,
        "quality_history":   0.15,
        "delivery":          0.15,
        "financial_health":  0.10,
        "sustainability":    0.05,
    },
    "RFI": {
        "technical_capability": 0.40,
        "financial_health":   0.25,
        "quality_history":    0.20,
        "sustainability":     0.15,
    },
    "reverse_auction": {
        "total_cost":        0.70,
        "quality_history":   0.15,
        "delivery":          0.10,
        "financial_health":  0.05,
    },
}

# TCO component weights
_TCO_WEIGHTS = {
    "unit_price":        0.45,
    "quality_cost":      0.20,   # rework, scrap, warranty
    "logistics":         0.15,   # freight, duties, handling
    "tooling_amortized": 0.10,   # tooling / NRE amortized over volume
    "disruption_risk":   0.10,   # risk-adjusted cost of supply failure
}


class SourcingExecutionAgent(BaseAgent):
    """Full implementation of the Sourcing Execution agent."""

    name = AgentName.SOURCING_EXECUTION

    def can_run(self, state: dict) -> bool:
        has_shortlist = bool(state.get("supplier_market", {}).get("supplier_shortlist"))
        has_strategy  = bool(state.get("spend_category", {}).get("category_strategies"))
        return has_shortlist and has_strategy

    # =========================================================================
    # Orchestration
    # =========================================================================

    def _execute(self, state: dict) -> dict:
        category   = state.get("category") or "all"
        strategies = state.get("spend_category", {}).get("category_strategies", [])
        shortlist  = state.get("supplier_market", {}).get("supplier_shortlist", [])
        context    = state.get("context", {})

        logger.info("[SourcingExecution] Starting — %d suppliers, %d strategies",
                    len(shortlist), len(strategies))

        # Step 1 — determine event type from category strategy
        event_type = self._determine_event_type(strategies, context)

        # Step 2 — generate event package documents
        packages = self._generate_event_packages(event_type, category, strategies,
                                                   shortlist, state)

        # Step 3 — collect bids (real or injected)
        bids = self._collect_bids(packages, shortlist, context)

        # Step 4 — normalize bids to common basis
        normalized_bids = self._normalize_bids(bids, context)

        # Step 5 — build TCO models per supplier
        tco_models = self._build_tco_models(normalized_bids, shortlist, state)

        # Step 6 — score and rank bids
        evaluation = self._evaluate_bids(normalized_bids, tco_models, event_type,
                                          shortlist, state)

        # Step 7 — generate award recommendation
        award = self._generate_award_recommendation(evaluation, tco_models,
                                                      strategies, state)

        # Step 8 — generate supplier scorecards
        scorecards = self._generate_scorecards(evaluation, shortlist)

        # Step 9 — BOM cost-down (conditional)
        cost_down = []
        if context.get("bom_cost_down_required"):
            cost_down = self._bom_cost_down(category, tco_models, context)

        logger.info("[SourcingExecution] Complete — event=%s, suppliers=%d, winner=%s",
                    event_type, len(bids), award.get("supplier_id"))

        return {
            "sourcing_execution": {
                "event_type":            event_type,
                "event_packages":        packages,
                "bid_evaluation_matrix": evaluation,
                "tco_comparison":        tco_models,
                "award_recommendation":  award,
                "supplier_scorecards":   scorecards,
                "bom_cost_down":         cost_down,
            }
        }

    # =========================================================================
    # Step 1 — Determine event type
    # =========================================================================

    def _determine_event_type(self, strategies: list[dict], context: dict) -> str:
        """
        Map the dominant category quadrant to the appropriate event type.
        Can be overridden by context.event_type_override.
        """
        if context.get("event_type_override"):
            return context["event_type_override"]

        # Count categories by quadrant
        counts: dict[str, int] = {}
        for s in strategies:
            q = s.get("quadrant", "non_critical")
            counts[q] = counts.get(q, 0) + 1

        dominant = max(counts, key=counts.get) if counts else "leverage"

        mapping = {
            "strategic":    "RFP",
            "leverage":     "RFQ",
            "bottleneck":   "RFI",
            "non_critical": "RFQ",
        }
        event_type = mapping.get(dominant, "RFQ")
        logger.info("[SourcingExecution] Event type: %s (dominant quadrant: %s)",
                    event_type, dominant)
        return event_type

    # =========================================================================
    # Step 2 — Event package generation
    # =========================================================================

    def _generate_event_packages(self, event_type: str, category: str,
                                   strategies: list[dict], shortlist: list[dict],
                                   state: dict) -> list[dict]:
        """
        Generate sourcing event document package for each supplier on the shortlist.
        Package includes: cover letter, requirements specification, evaluation criteria,
        commercial terms template, and submission instructions.
        """
        goal     = state.get("goal", "")
        budget   = state.get("budget")
        context  = state.get("context", {})
        timeline = state.get("timeline_days", 30)

        try:
            package_content = extract_json(
                system=(
                    "You are a senior procurement professional generating sourcing event documents. "
                    "Create a sourcing event package structure. Return JSON:\n"
                    "{\n"
                    "  cover_letter: str,\n"
                    "  scope_of_work: str,\n"
                    "  requirements: [{requirement_id, description, mandatory: bool, weight_pct: int}],\n"
                    "  commercial_template: {pricing_structure: str, payment_terms: str, "
                    "    incoterms: str, warranty_months: int},\n"
                    "  evaluation_criteria: [{criterion, weight_pct, description}],\n"
                    "  submission_instructions: str,\n"
                    "  key_dates: {rfx_issue: str, q_and_a_deadline: str, "
                    "    submission_deadline: str, award_date: str}\n"
                    "}"
                ),
                user=(
                    f"Event type: {event_type}\n"
                    f"Category: {category}\n"
                    f"Sourcing goal: {goal}\n"
                    f"Timeline: {timeline} days\n"
                    + (f"Budget target: ${budget:,.0f}\n" if budget else "")
                    + f"Strategies: {[s.get('approach') for s in strategies[:3]]}\n"
                    f"Industry: {context.get('industry', 'aerospace / defense')}"
                ),
                max_tokens=1500,
            )
        except Exception as exc:
            logger.warning("[SourcingExecution] Event package LLM failed: %s", exc)
            package_content = {
                "cover_letter": f"Invitation to {event_type} for {category}",
                "scope_of_work": f"Supply of {category} per requirements to be specified.",
                "requirements": [],
                "commercial_template": {
                    "pricing_structure": "firm_fixed_price",
                    "payment_terms":     "net30",
                    "incoterms":         "DDP",
                    "warranty_months":   12,
                },
                "evaluation_criteria": [],
                "submission_instructions": "Submit via procurement portal by deadline.",
                "key_dates": {},
            }

        # Create one package record per supplier
        packages = []
        for supplier in shortlist:
            packages.append({
                "supplier_id":  supplier.get("supplier_id"),
                "supplier_name": supplier.get("name"),
                "event_type":   event_type,
                "category":     category,
                "status":       "issued",
                "issued_at":    datetime.utcnow().isoformat(),
                **package_content,
            })

        return packages

    # =========================================================================
    # Step 3 — Collect bids
    # =========================================================================

    def _collect_bids(self, packages: list[dict], shortlist: list[dict],
                       context: dict) -> list[dict]:
        """
        Collect bid responses from suppliers.
        In production: polls supplier portal API or parses email/EDI responses.
        In test mode: uses context.bid_responses injection.
        """
        if context.get("bid_responses"):
            logger.info("[SourcingExecution] Using injected bid responses")
            return context["bid_responses"]

        # INTEGRATION POINT: supplier communication / portal API
        # from ..tools.supplier_portal import SupplierPortal
        # return SupplierPortal().collect_bids(packages)

        logger.warning("[SourcingExecution] Supplier portal not wired — "
                       "generating synthetic bids for demo. "
                       "Inject context.bid_responses or implement tools/supplier_portal.py.")

        # Generate realistic synthetic bids for demo/test purposes
        bids = []
        for i, pkg in enumerate(packages):
            base_price = context.get("target_unit_price", 100.0)
            price_var  = base_price * (0.85 + i * 0.08)  # spread bids realistically
            bids.append({
                "supplier_id":        pkg["supplier_id"],
                "supplier_name":      pkg.get("supplier_name", ""),
                "event_type":         pkg["event_type"],
                "unit_price":         round(price_var, 2),
                "currency":           "USD",
                "lead_time_days":     context.get("target_lead_time_days", 21) + i * 5,
                "minimum_order_qty":  context.get("moq", 100),
                "tooling_cost":       context.get("tooling_cost", 0.0),
                "warranty_months":    12,
                "payment_terms":      "net30",
                "annual_volume_offered": context.get("annual_volume_units", 1000),
                "quality_defect_ppm":  500 - i * 50,  # better suppliers lower PPM
                "on_time_delivery_pct": 92.0 + i * 1.5,
                "submitted_at":       datetime.utcnow().isoformat(),
                "technical_score":    7.5 + i * 0.3,  # mock technical evaluation score
            })
        return bids

    # =========================================================================
    # Step 4 — Normalize bids
    # =========================================================================

    def _normalize_bids(self, bids: list[dict], context: dict) -> list[dict]:
        """
        Normalise bids to a common commercial basis:
          - Convert all currencies to USD
          - Apply incoterms adjustments (e.g., EXW → DDP adds freight)
          - Annualise tooling over contract duration
          - Standardise payment terms to NPV basis
        """
        contract_years = context.get("contract_years", 3)
        annual_volume  = context.get("annual_volume_units", 1000)
        wacc           = context.get("wacc", 0.08)  # discount rate for NPV

        normalized = []
        for bid in bids:
            unit_price   = float(bid.get("unit_price", 0))
            tooling      = float(bid.get("tooling_cost", 0))
            volume       = float(bid.get("annual_volume_offered", annual_volume))

            # Annualise tooling over contract life
            tooling_per_unit = tooling / (volume * contract_years) if volume > 0 else 0

            # Payment terms NPV adjustment (net30=0, net60 saves ~0.4% at 8% WACC)
            terms_map    = {"net30": 0.0, "net45": 0.002, "net60": 0.004, "net90": 0.008}
            terms_adj    = terms_map.get(bid.get("payment_terms", "net30"), 0.0)

            normalized_price = unit_price + tooling_per_unit - (unit_price * terms_adj)

            normalized.append({
                **bid,
                "unit_price_normalized":  round(normalized_price, 4),
                "tooling_per_unit":        round(tooling_per_unit, 4),
                "payment_terms_adj":       round(terms_adj * unit_price, 4),
                "normalization_basis":     f"DDP USD net30, {contract_years}yr tooling amortization",
            })

        return normalized

    # =========================================================================
    # Step 5 — TCO models
    # =========================================================================

    def _build_tco_models(self, bids: list[dict], shortlist: list[dict],
                           state: dict) -> list[dict]:
        """
        Build per-supplier TCO model with five components:
          1. Unit price (normalized)
          2. Quality cost (defect PPM × rework cost)
          3. Logistics (freight + duties — from landed cost models)
          4. Tooling amortized per unit
          5. Supply disruption risk premium
        """
        context        = state.get("context", {})
        annual_units   = context.get("annual_volume_units", 1000)
        contract_years = context.get("contract_years", 3)
        rework_cost    = context.get("rework_cost_per_unit", 15.0)
        disruption_cost= context.get("disruption_cost_per_day", 5000.0)

        # Pull logistics costs from landed cost models if available
        landed_map: dict[str, float] = {}
        for model in state.get("supplier_market", {}).get("landed_cost_models", []):
            for region, data in model.get("regions", {}).items():
                pass  # we'll use per-bid lead time as proxy

        shortlist_map = {s["supplier_id"]: s for s in shortlist}
        tco_models = []

        for bid in bids:
            sid          = bid["supplier_id"]
            sup          = shortlist_map.get(sid, {})
            unit_price   = float(bid.get("unit_price_normalized", bid.get("unit_price", 0)))
            defect_ppm   = float(bid.get("quality_defect_ppm", 1000))
            otd_pct      = float(bid.get("on_time_delivery_pct", 90)) / 100
            lead_days    = int(bid.get("lead_time_days", 30))
            fin_health   = sup.get("financial_health", "stable")

            # Quality cost: defects × rework per unit
            quality_cost_per_unit = (defect_ppm / 1_000_000) * rework_cost

            # Logistics: simplified freight pct based on lead time proxy
            freight_pct   = 0.03 if lead_days < 20 else (0.07 if lead_days < 45 else 0.12)
            logistics_per_unit = unit_price * freight_pct

            # Tooling already in unit_price_normalized
            tooling_per_unit = float(bid.get("tooling_per_unit", 0))

            # Disruption risk: (1 - OTD) × expected lead time cost × safety factor
            disruption_days = lead_days * (1 - otd_pct)
            risk_multiplier = {"strong": 0.5, "stable": 1.0, "weak": 2.0, "unknown": 1.5}.get(fin_health, 1.0)
            disruption_cost_annual = disruption_cost * disruption_days * risk_multiplier
            disruption_per_unit    = disruption_cost_annual / annual_units if annual_units else 0

            # Total TCO per unit
            components = {
                "unit_price":        round(unit_price, 4),
                "quality_cost":      round(quality_cost_per_unit, 4),
                "logistics":         round(logistics_per_unit, 4),
                "tooling_amortized": round(tooling_per_unit, 4),
                "disruption_risk":   round(disruption_per_unit, 4),
            }
            tco_per_unit = sum(components.values())
            tco_total    = tco_per_unit * annual_units * contract_years

            tco_models.append({
                "supplier_id":     sid,
                "supplier_name":   bid.get("supplier_name", ""),
                "tco_components":  components,
                "tco_per_unit":    round(tco_per_unit, 4),
                "tco_annual":      round(tco_per_unit * annual_units, 2),
                "tco_contract_total": round(tco_total, 2),
                "contract_years":  contract_years,
                "annual_units":    annual_units,
                "modelled_at":     datetime.utcnow().isoformat(),
            })

        # Rank by TCO per unit ascending
        tco_models.sort(key=lambda m: m["tco_per_unit"])
        for rank, model in enumerate(tco_models, 1):
            model["tco_rank"] = rank

        return tco_models

    # =========================================================================
    # Step 6 — Score and evaluate bids
    # =========================================================================

    def _evaluate_bids(self, bids: list[dict], tco_models: list[dict],
                        event_type: str, shortlist: list[dict],
                        state: dict) -> dict:
        """
        Score each bid against weighted evaluation criteria.
        Produces a normalised score matrix (0-100 per criterion).
        """
        weights     = _SCORING_WEIGHTS.get(event_type, _SCORING_WEIGHTS["RFQ"])
        tco_map     = {m["supplier_id"]: m for m in tco_models}
        sup_map     = {s["supplier_id"]: s for s in shortlist}

        # Normalise cost scores: lowest TCO = 100, worst = 0
        tco_values  = [m["tco_per_unit"] for m in tco_models if m["tco_per_unit"] > 0]
        min_tco     = min(tco_values) if tco_values else 1.0
        max_tco     = max(tco_values) if tco_values else 1.0
        tco_range   = max_tco - min_tco or 1.0

        scores_by_supplier: dict[str, dict] = {}
        for bid in bids:
            sid  = bid["supplier_id"]
            tco  = tco_map.get(sid, {})
            sup  = sup_map.get(sid, {})

            tco_score  = 100 * (1 - (tco.get("tco_per_unit", max_tco) - min_tco) / tco_range)
            otd_score  = float(bid.get("on_time_delivery_pct", 90))
            qual_score = max(0, 100 - bid.get("quality_defect_ppm", 1000) / 100)
            fin_map    = {"strong": 90, "stable": 70, "weak": 40, "unknown": 55}
            fin_score  = fin_map.get(sup.get("financial_health", "unknown"), 55)
            tech_score = float(bid.get("technical_score", 70))
            sust_score = float(sup.get("sustainability_score", 50)) if "sustainability_score" in sup else 60.0

            raw = {
                "total_cost":          round(tco_score,  2),
                "delivery":            round(otd_score,  2),
                "quality_history":     round(qual_score, 2),
                "financial_health":    round(fin_score,  2),
                "technical_capability": round(tech_score, 2),
                "sustainability":      round(sust_score, 2),
            }

            weighted_total = sum(
                raw.get(criterion, 0) * weight
                for criterion, weight in weights.items()
                if criterion in raw
            )
            scores_by_supplier[sid] = {
                "supplier_name":    bid.get("supplier_name", ""),
                "criterion_scores": raw,
                "weighted_total":   round(weighted_total, 2),
                "tco_rank":         tco.get("tco_rank", 99),
            }

        # Rank by weighted total
        ranked = sorted(
            scores_by_supplier.items(),
            key=lambda x: x[1]["weighted_total"],
            reverse=True,
        )

        return {
            "event_type":   event_type,
            "weights_used": weights,
            "scores":       scores_by_supplier,
            "ranked":       [sid for sid, _ in ranked],
            "evaluated_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Step 7 — Award recommendation
    # =========================================================================

    def _generate_award_recommendation(self, evaluation: dict,
                                         tco_models: list[dict],
                                         strategies: list[dict],
                                         state: dict) -> dict:
        """
        Generate a structured award recommendation with LLM-written rationale.
        """
        ranked      = evaluation.get("ranked", [])
        scores      = evaluation.get("scores", {})
        if not ranked:
            return {"supplier_id": None, "rationale": "No bids received", "tco_rank": None}

        winner_id   = ranked[0]
        winner_data = scores.get(winner_id, {})
        winner_tco  = next((m for m in tco_models if m["supplier_id"] == winner_id), {})

        # BATNA = second-ranked supplier
        batna_id    = ranked[1] if len(ranked) > 1 else None
        batna_data  = scores.get(batna_id, {}) if batna_id else {}
        batna_tco   = next((m for m in tco_models if m["supplier_id"] == batna_id), {})

        try:
            recommendation = reason(
                system=(
                    "You are a sourcing lead presenting an award recommendation to "
                    "procurement leadership. Write a concise recommendation (150-200 words) "
                    "covering: why the recommended supplier wins on a total value basis, "
                    "key differentiators vs. the runner-up, TCO advantage, primary risks "
                    "to monitor, and recommended contract structure. Be direct and quantitative."
                ),
                user=(
                    f"Recommended supplier: {winner_id}\n"
                    f"Evaluation score: {winner_data.get('weighted_total')}/100\n"
                    f"TCO per unit: ${winner_tco.get('tco_per_unit', 0):,.4f}\n"
                    f"TCO breakdown: {winner_tco.get('tco_components', {})}\n\n"
                    f"Runner-up: {batna_id}\n"
                    f"Runner-up score: {batna_data.get('weighted_total')}/100\n"
                    f"Runner-up TCO: ${batna_tco.get('tco_per_unit', 0):,.4f}\n\n"
                    f"Category strategies: {[s.get('approach') for s in strategies[:3]]}\n"
                    f"Goal: {state.get('goal', '')}"
                ),
                max_tokens=400,
            )
        except Exception as exc:
            logger.warning("[SourcingExecution] Award rationale LLM failed: %s", exc)
            recommendation = (
                f"Supplier {winner_id} recommended based on highest weighted evaluation "
                f"score ({winner_data.get('weighted_total')}/100) and lowest TCO "
                f"(${winner_tco.get('tco_per_unit', 0):,.4f}/unit)."
            )

        return {
            "supplier_id":         winner_id,
            "supplier_name":       winner_data.get("supplier_name", ""),
            "evaluation_score":    winner_data.get("weighted_total"),
            "tco_per_unit":        winner_tco.get("tco_per_unit"),
            "tco_contract_total":  winner_tco.get("tco_contract_total"),
            "tco_rank":            winner_tco.get("tco_rank", 1),
            "batna_supplier_id":   batna_id,
            "batna_evaluation_score": batna_data.get("weighted_total"),
            "batna_tco_per_unit":  batna_tco.get("tco_per_unit"),
            "rationale":           recommendation,
            "created_at":          datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Step 8 — Supplier scorecards
    # =========================================================================

    def _generate_scorecards(self, evaluation: dict,
                               shortlist: list[dict]) -> list[dict]:
        """
        Generate a qualification + performance scorecard per supplier.
        """
        scores  = evaluation.get("scores", {})
        sup_map = {s["supplier_id"]: s for s in shortlist}
        cards   = []

        for sid, eval_data in scores.items():
            sup = sup_map.get(sid, {})
            cards.append({
                "supplier_id":      sid,
                "supplier_name":    eval_data.get("supplier_name", ""),
                "overall_score":    eval_data.get("weighted_total"),
                "rank":             evaluation.get("ranked", []).index(sid) + 1
                                    if sid in evaluation.get("ranked", []) else 99,
                "criterion_scores": eval_data.get("criterion_scores", {}),
                "qualification_score": sup.get("qualification_score"),
                "financial_health": sup.get("financial_health"),
                "strengths":        sup.get("strengths", []),
                "risks":            sup.get("risks", []),
                "recommended_audit": sup.get("recommended_audit", False),
                "generated_at":     datetime.utcnow().isoformat(),
            })

        return sorted(cards, key=lambda c: c.get("rank", 99))

    # =========================================================================
    # Step 9 — BOM cost-down
    # =========================================================================

    def _bom_cost_down(self, category: str, tco_models: list[dict],
                        context: dict) -> list[dict]:
        """
        Identify BOM cost-down opportunities through value engineering,
        design-to-cost, material substitution, and resourcing analysis.
        """
        if not tco_models:
            return []

        best_tco    = tco_models[0]  # already sorted by rank
        current_tco = best_tco.get("tco_per_unit", 0)

        try:
            opportunities = extract_json(
                system=(
                    "You are a value engineering specialist for aerospace supply chains. "
                    "Identify BOM cost reduction opportunities and return a JSON array:\n"
                    "[{opportunity_type: str, description: str, "
                    "estimated_savings_pct: float, implementation_effort: 'low'|'medium'|'high', "
                    "timeline_weeks: int, risk: 'low'|'medium'|'high'}]"
                ),
                user=(
                    f"Category: {category}\n"
                    f"Current best TCO per unit: ${current_tco:,.4f}\n"
                    f"TCO components: {best_tco.get('tco_components', {})}\n"
                    f"Annual volume: {context.get('annual_volume_units', 1000)}\n"
                    f"BOM context: {context.get('bom_context', 'Standard aerospace component')}\n"
                    "Identify top 3-5 cost-down opportunities."
                ),
                max_tokens=800,
            )
            return opportunities
        except Exception as exc:
            logger.warning("[SourcingExecution] BOM cost-down LLM failed: %s", exc)
            return []