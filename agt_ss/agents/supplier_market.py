# Databricks notebook source
"""
agt_ss.agents.supplier_market
------------------------------
Agent 2: Supplier Market Intelligence

Processes owned (3):
  1. Supply market analysis — structure, competitive dynamics, pricing trends,
     innovation landscape, and supplier financial health assessment.
  2. Global vs. domestic sourcing trade-offs — total landed cost (TLC) models
     incorporating unit price, freight, tariffs, duties, lead time variability,
     inventory carrying cost, and supply chain risk premium.
  3. Approved Supplier List (ASL) management — qualify new suppliers, flag
     re-qualification triggers, update preferred vendor status.

Data flow:
  IN  ← state.spend_category.kraljic_matrix
        state.spend_category.category_strategies
        state.context.supplier_candidates (optional manual override)
  OUT → state.supplier_market {market_analysis_briefs, landed_cost_models,
                                approved_supplier_list, supplier_shortlist}
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .base import AgentToolError, BaseAgent
from .llm import extract_json, reason
from ..state.schema import AgentName

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landed cost model constants
# ---------------------------------------------------------------------------

# Standard freight cost as % of unit cost by origin region
_FREIGHT_PCT: dict[str, float] = {
    "domestic":      0.03,
    "nearshore":     0.05,   # Mexico, Canada
    "europe":        0.07,
    "asia_pacific":  0.12,
    "south_america": 0.09,
    "middle_east":   0.10,
}

# Average import duty rates by spend_type and region (simplified)
_DUTY_PCT: dict[str, dict[str, float]] = {
    "direct_material": {"domestic": 0.0, "nearshore": 0.0, "europe": 0.035, "asia_pacific": 0.05},
    "mro":             {"domestic": 0.0, "nearshore": 0.0, "europe": 0.02,  "asia_pacific": 0.04},
    "indirect":        {"domestic": 0.0, "nearshore": 0.0, "europe": 0.015, "asia_pacific": 0.03},
    "services":        {"domestic": 0.0, "nearshore": 0.0, "europe": 0.0,   "asia_pacific": 0.0},
}

# Lead time in days by region (average)
_LEAD_TIME_DAYS: dict[str, int] = {
    "domestic":      14,
    "nearshore":     21,
    "europe":        35,
    "asia_pacific":  55,
    "south_america": 40,
    "middle_east":   45,
}

# Annual inventory carrying cost rate
_CARRYING_COST_RATE = 0.22

# Risk premium by region (supply disruption likelihood × impact)
_RISK_PREMIUM_PCT: dict[str, float] = {
    "domestic":      0.01,
    "nearshore":     0.02,
    "europe":        0.03,
    "asia_pacific":  0.05,
    "south_america": 0.06,
    "middle_east":   0.07,
}


class SupplierMarketAgent(BaseAgent):
    """Full implementation of the Supplier Market Intelligence agent."""

    name = AgentName.SUPPLIER_MARKET

    def can_run(self, state: dict) -> bool:
        return bool(state.get("spend_category", {}).get("kraljic_matrix"))

    # =========================================================================
    # Orchestration
    # =========================================================================

    def _execute(self, state: dict) -> dict:
        category   = state.get("category") or "all"
        strategies = state.get("spend_category", {}).get("category_strategies", [])
        kraljic    = state.get("spend_category", {}).get("kraljic_matrix", {})
        context    = state.get("context", {})

        logger.info("[SupplierMarket] Starting — %d category strategies", len(strategies))

        # Step 1 — market analysis per category
        market_briefs = self._analyse_markets(strategies, category, state)

        # Step 2 — total landed cost models
        landed_cost_models = self._build_landed_cost_models(
            strategies, market_briefs, context)

        # Step 3 — update / query ASL
        asl = self._manage_asl(category, market_briefs, context)

        # Step 4 — build shortlist for sourcing event
        shortlist = self._build_shortlist(asl, market_briefs, kraljic, context)

        logger.info("[SupplierMarket] Complete — %d briefs, %d ASL entries, %d shortlisted",
                    len(market_briefs), len(asl), len(shortlist))

        return {
            "supplier_market": {
                "market_analysis_briefs": market_briefs,
                "landed_cost_models":     landed_cost_models,
                "approved_supplier_list": asl,
                "supplier_shortlist":     shortlist,
            }
        }

    # =========================================================================
    # Step 1 — Market analysis
    # =========================================================================

    def _analyse_markets(self, strategies: list[dict], primary_category: str,
                          state: dict) -> list[dict]:
        """
        Build a market intelligence brief for each category strategy.
        Priority order: strategic > leverage > bottleneck > non_critical.
        Strategic and leverage categories get full LLM market analysis.
        Non-critical categories get a summary brief only.
        """
        # Deduplicate categories; process high-priority first
        priority = {"strategic": 0, "leverage": 1, "bottleneck": 2, "non_critical": 3}
        sorted_strategies = sorted(strategies, key=lambda s: priority.get(s.get("quadrant", "non_critical"), 3))

        seen = set()
        briefs = []
        for strat in sorted_strategies:
            cat = strat.get("category", primary_category)
            if cat in seen:
                continue
            seen.add(cat)

            quadrant = strat.get("quadrant", "non_critical")
            full_analysis = quadrant in ("strategic", "leverage", "bottleneck")

            brief = self._build_market_brief(cat, quadrant, full_analysis, state)
            briefs.append(brief)

        return briefs

    def _build_market_brief(self, category: str, quadrant: str,
                             full_analysis: bool, state: dict) -> dict:
        """
        Build one market intelligence brief.
        Full analysis includes LLM-generated competitive landscape, pricing dynamics,
        innovation outlook, and top supplier profiles.
        """
        industry = state.get("context", {}).get("industry", "aerospace / defense")
        goal     = state.get("goal", "")

        if not full_analysis:
            return {
                "category":          category,
                "quadrant":          quadrant,
                "analysis_depth":    "summary",
                "market_structure":  "Not analysed — non-critical category",
                "risk_score":        2.0,
                "analysed_at":       datetime.utcnow().isoformat(),
            }

        try:
            analysis = extract_json(
                system=(
                    f"You are a supply market intelligence analyst for the {industry} industry. "
                    "Return a detailed JSON market brief for a spend category. Schema:\n"
                    "{"
                    "  market_structure: 'oligopoly'|'competitive'|'monopoly'|'fragmented', "
                    "  supplier_count_estimate: int, "
                    "  top_suppliers: [{name, hq_country, market_share_pct, financial_health: 'strong'|'stable'|'weak'}], "
                    "  pricing_trend: 'increasing'|'stable'|'decreasing', "
                    "  pricing_trend_rationale: str, "
                    "  yoy_price_change_pct: float, "
                    "  innovation_landscape: str (1-2 sentences), "
                    "  supply_risk_factors: [str], "
                    "  geopolitical_exposure: 'low'|'medium'|'high', "
                    "  risk_score: float (0-10), "
                    "  recommended_sourcing_regions: [str], "
                    "  key_insights: [str] "
                    "}"
                ),
                user=(
                    f"Category: {category}\n"
                    f"Kraljic quadrant: {quadrant}\n"
                    f"Sourcing goal: {goal}\n"
                    "Provide a current market intelligence brief."
                ),
                max_tokens=1000,
            )
            return {
                "category":       category,
                "quadrant":       quadrant,
                "analysis_depth": "full",
                "analysed_at":    datetime.utcnow().isoformat(),
                **analysis,
            }
        except Exception as exc:
            logger.warning("[SupplierMarket] Market brief LLM failed for %s: %s", category, exc)
            return {
                "category":         category,
                "quadrant":         quadrant,
                "analysis_depth":   "fallback",
                "market_structure": "unknown",
                "risk_score":       5.0,
                "analysed_at":      datetime.utcnow().isoformat(),
            }

    # =========================================================================
    # Step 2 — Total landed cost models
    # =========================================================================

    def _build_landed_cost_models(self, strategies: list[dict],
                                   market_briefs: list[dict],
                                   context: dict) -> list[dict]:
        """
        Build a total landed cost (TLC) comparison for each category with
        meaningful spend (spend_share_pct > 2%).

        Compares: domestic, nearshore, europe, asia_pacific.
        """
        models = []
        brief_map = {b["category"]: b for b in market_briefs}

        for strat in strategies:
            cat          = strat.get("category", "")
            spend_share  = strat.get("profit_impact", 0)  # proxy for relative size
            spend_type   = context.get("spend_type_override") or "direct_material"
            unit_cost    = context.get("unit_cost_usd", {}).get(cat, 100.0)
            annual_units = context.get("annual_volume_units", 1000)
            brief        = brief_map.get(cat, {})

            # Only model categories worth the analysis
            if strat.get("quadrant") == "non_critical":
                continue

            region_models = {}
            for region in ("domestic", "nearshore", "europe", "asia_pacific"):
                region_models[region] = self._compute_tlc(
                    unit_cost, annual_units, region, spend_type, brief)

            best_region = min(region_models, key=lambda r: region_models[r]["total_landed_cost"])

            models.append({
                "category":          cat,
                "annual_units":      annual_units,
                "base_unit_cost":    unit_cost,
                "regions":           region_models,
                "recommended_region": best_region,
                "tlc_vs_domestic_pct": round(
                    (region_models[best_region]["total_landed_cost"] -
                     region_models["domestic"]["total_landed_cost"])
                    / (region_models["domestic"]["total_landed_cost"] or 1) * 100, 2),
                "modelled_at":       datetime.utcnow().isoformat(),
            })

        return models

    def _compute_tlc(self, unit_cost: float, annual_units: int,
                      region: str, spend_type: str, brief: dict) -> dict:
        """
        Compute total landed cost components for a region.

        TLC = unit_cost × (1 + freight + duty + risk_premium)
              + inventory_carrying_cost
        """
        freight_pct  = _FREIGHT_PCT.get(region, 0.08)
        duty_pct     = _DUTY_PCT.get(spend_type, {}).get(region, 0.0)
        risk_pct     = _RISK_PREMIUM_PCT.get(region, 0.05)

        # Adjust risk from market brief if available
        brief_risk   = brief.get("risk_score", 5.0) / 10.0  # normalise to 0-1
        risk_pct     = risk_pct * (1 + brief_risk)

        lead_days    = _LEAD_TIME_DAYS.get(region, 30)
        # Safety stock cost: lead_time_days / 365 × annual_spend × carrying_rate
        annual_spend = unit_cost * annual_units
        safety_stock = (lead_days / 365) * annual_spend * _CARRYING_COST_RATE
        adjusted_cost = unit_cost * (1 + freight_pct + duty_pct + risk_pct)
        total_tlc     = adjusted_cost * annual_units + safety_stock

        return {
            "unit_cost":           round(unit_cost, 2),
            "freight_pct":         round(freight_pct * 100, 2),
            "duty_pct":            round(duty_pct * 100, 2),
            "risk_premium_pct":    round(risk_pct * 100, 2),
            "lead_time_days":      lead_days,
            "safety_stock_cost":   round(safety_stock, 2),
            "adjusted_unit_cost":  round(adjusted_cost, 2),
            "total_landed_cost":   round(total_tlc, 2),
        }

    # =========================================================================
    # Step 3 — ASL management
    # =========================================================================

    def _manage_asl(self, category: str, market_briefs: list[dict],
                     context: dict) -> list[dict]:
        """
        Load ASL from Delta Lake, evaluate qualification status, and merge
        any new candidates from market briefs or context injection.
        """
        # Load existing ASL (Delta Lake / context injection)
        existing_asl = self._load_asl(category, context)

        # Extract candidate suppliers from market brief top_suppliers lists
        candidate_suppliers = []
        for brief in market_briefs:
            for sup in brief.get("top_suppliers", []):
                if sup.get("name"):
                    candidate_suppliers.append({
                        "supplier_id":    f"SUP-{sup['name'][:8].upper().replace(' ', '')}",
                        "name":           sup["name"],
                        "hq_country":     sup.get("hq_country", "Unknown"),
                        "market_share_pct": sup.get("market_share_pct", 0),
                        "financial_health": sup.get("financial_health", "unknown"),
                        "category":       brief.get("category"),
                        "source":         "market_brief",
                    })

        # Add candidates from context (manually specified)
        for s in context.get("supplier_candidates", []):
            candidate_suppliers.append({**s, "source": "manual"})

        # Qualify each candidate not already in ASL
        existing_ids = {s.get("supplier_id") for s in existing_asl}
        qualified = list(existing_asl)

        for candidate in candidate_suppliers:
            if candidate.get("supplier_id") in existing_ids:
                continue
            qualified_entry = self._qualify_supplier(candidate, context)
            qualified.append(qualified_entry)
            existing_ids.add(candidate.get("supplier_id"))

        return qualified

    def _load_asl(self, category: str, context: dict) -> list[dict]:
        """
        Load approved supplier list from Delta Lake.

        Replace with:
            from ..tools.delta import DeltaReader
            return DeltaReader().query_asl(category)
        """
        # Use injected ASL if provided (test / notebook mode)
        if context.get("existing_asl"):
            return context["existing_asl"]

        logger.debug("[SupplierMarket] ASL Delta Lake not wired — using empty list")
        return []

    def _qualify_supplier(self, candidate: dict, context: dict) -> dict:
        """
        Run LLM-assisted qualification assessment on a candidate supplier.
        In production, this would also pull D&B financial data and
        run compliance checks against sanctioned-party lists.
        """
        try:
            qual = extract_json(
                system=(
                    "You are a supplier qualification analyst. "
                    "Assess a candidate supplier and return JSON: "
                    "{"
                    "  qualification_score: float (0-100), "
                    "  status: 'approved'|'conditional'|'rejected'|'pending_audit', "
                    "  strengths: [str], "
                    "  risks: [str], "
                    "  recommended_audit: bool, "
                    "  qualification_notes: str"
                    "}"
                ),
                user=(
                    f"Supplier: {candidate.get('name')}\n"
                    f"HQ country: {candidate.get('hq_country')}\n"
                    f"Financial health: {candidate.get('financial_health', 'unknown')}\n"
                    f"Market share: {candidate.get('market_share_pct', 0)}%\n"
                    f"Category: {candidate.get('category')}\n"
                    f"Industry context: {context.get('industry', 'aerospace / defense')}"
                ),
                max_tokens=500,
            )
            return {
                **candidate,
                **qual,
                "qualified_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.warning("[SupplierMarket] Qualification LLM failed: %s", exc)
            return {
                **candidate,
                "qualification_score": 0.0,
                "status":             "pending_audit",
                "strengths":          [],
                "risks":              ["Qualification assessment unavailable"],
                "recommended_audit":  True,
                "qualified_at":       datetime.utcnow().isoformat(),
            }

    # =========================================================================
    # Step 4 — Build shortlist
    # =========================================================================

    def _build_shortlist(self, asl: list[dict], market_briefs: list[dict],
                          kraljic: dict, context: dict) -> list[dict]:
        """
        Select the optimal supplier shortlist for sourcing events.

        Shortlist size by quadrant:
          strategic:    3-5 (relationship diversity + competition)
          leverage:     5-8 (maximum competition)
          bottleneck:   2-3 (proven incumbents + one alternative)
          non_critical: 2-3 (process simplicity)

        Filters: approved status only, financial_health != 'weak' for strategic.
        """
        target_sizes = {
            "strategic":    (3, 5),
            "leverage":     (5, 8),
            "bottleneck":   (2, 3),
            "non_critical": (2, 3),
        }

        # Determine dominant quadrant
        quadrant_counts = {q: len(items) for q, items in kraljic.items()}
        dominant_quadrant = max(quadrant_counts, key=quadrant_counts.get) \
            if quadrant_counts else "leverage"

        min_size, max_size = target_sizes.get(dominant_quadrant, (3, 5))

        # Filter eligible suppliers
        eligible = [
            s for s in asl
            if s.get("status") in ("approved", "conditional")
        ]

        # For strategic categories, exclude financially weak suppliers
        if dominant_quadrant == "strategic":
            eligible = [s for s in eligible
                        if s.get("financial_health") != "weak"]

        # Sort by qualification score descending
        eligible_sorted = sorted(
            eligible,
            key=lambda s: s.get("qualification_score", 0),
            reverse=True,
        )

        shortlist = eligible_sorted[:max_size]

        # If shortlist is below minimum, flag it
        if len(shortlist) < min_size:
            logger.warning(
                "[SupplierMarket] Shortlist below minimum (%d < %d) — "
                "consider expanding supplier pool or running RFI first.",
                len(shortlist), min_size,
            )

        return [
            {**s, "shortlisted_at": datetime.utcnow().isoformat(),
             "shortlist_rank": i + 1}
            for i, s in enumerate(shortlist)
        ]