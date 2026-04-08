# Databricks notebook source
"""
agt_ss.agents.spend_category
-----------------------------
Agent 1: Spend & Category Intelligence

Processes owned (4):
  1. Spend analysis & classification — cleanse, classify by UNSPSC taxonomy,
     aggregate by category and supplier.
  2. Category strategy development — score each category on supply risk ×
     profit impact (Kraljic), assign quadrant, generate strategy document.
  3. Tail spend management — identify long-tail transactions, maverick buying
     patterns, and P-card / consolidation opportunities.
  4. Make-vs-buy analysis — compare internal production cost against supplier
     TCO for components or services flagged for review.

Data flow:
  IN  ← state.context.spend_data (list of PO rows) or SAP extract
  OUT → state.spend_category {spend_classification, kraljic_matrix,
                               category_strategies, tail_spend_report,
                               make_vs_buy_recommendations}
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any

from .base import AgentToolError, BaseAgent
from .llm import extract_json, reason
from ..state.schema import AgentName

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kraljic scoring constants
# ---------------------------------------------------------------------------

# Profit impact weight factors applied to relative spend share
_PROFIT_WEIGHT = {
    "direct_material": 1.0,
    "indirect":        0.5,
    "services":        0.4,
    "mro":             0.3,
    "capex":           0.6,
    "other":           0.2,
}

# Quadrant thresholds on normalised 0-10 scales
_HIGH_PROFIT_THRESHOLD    = 5.0
_HIGH_SUPPLY_RISK_THRESHOLD = 5.0


class SpendCategoryAgent(BaseAgent):
    """Full implementation of the Spend & Category Intelligence agent."""

    name = AgentName.SPEND_CATEGORY

    def can_run(self, state: dict) -> bool:
        return bool(state.get("goal"))

    # =========================================================================
    # Orchestration
    # =========================================================================

    def _execute(self, state: dict) -> dict:
        category = state.get("category") or "all"
        context  = state.get("context", {})

        logger.info("[SpendCategory] Starting — category=%s", category)

        # Step 1 — acquire and cleanse spend data
        raw_rows = self._acquire_spend_data(state)

        # Step 2 — classify each row to UNSPSC segment
        classified = self._classify_spend(raw_rows, category)

        # Step 3 — aggregate spend metrics
        aggregated = self._aggregate_spend(classified)

        # Step 4 — score Kraljic matrix
        kraljic = self._score_kraljic(aggregated, state)

        # Step 5 — generate category strategies (one per quadrant group)
        strategies = self._generate_category_strategies(kraljic, state)

        # Step 6 — tail spend analysis
        tail = self._analyse_tail_spend(classified, aggregated)

        # Step 7 — make-vs-buy (conditional)
        mvb_results = []
        if context.get("make_vs_buy_required") or context.get("make_vs_buy_categories"):
            mvb_categories = context.get("make_vs_buy_categories", [category])
            for mvb_cat in mvb_categories:
                result = self._make_vs_buy(mvb_cat, aggregated, context)
                mvb_results.append(result)

        logger.info("[SpendCategory] Complete — %d categories, %d strategies",
                    len(aggregated), len(strategies))

        return {
            "spend_category": {
                "spend_classification":        aggregated,
                "kraljic_matrix":              kraljic,
                "category_strategies":         strategies,
                "tail_spend_report":           tail,
                "make_vs_buy_recommendations": mvb_results,
            }
        }

    # =========================================================================
    # Step 1 — Acquire spend data
    # =========================================================================

    def _acquire_spend_data(self, state: dict) -> list[dict]:
        """
        Pull PO rows from context (test/demo) or SAP extract.

        Each row schema:
          {po_number, supplier_id, supplier_name, category_raw, spend_type,
           amount_usd, currency, po_date, buyer, commodity_code}
        """
        context = state.get("context", {})

        # Prefer injected data (useful for testing and notebook invocation)
        if context.get("spend_data"):
            rows = context["spend_data"]
            logger.info("[SpendCategory] Using injected spend data — %d rows", len(rows))
            return rows

        # Otherwise call SAP extract tool
        return self._tool_sap_spend_extract(
            category=state.get("category") or "all",
            months=context.get("lookback_months", 24),
        )

    def _tool_sap_spend_extract(self, category: str, months: int = 24) -> list[dict]:
        """
        Call SAP MCP or REST API to pull PO spend.

        Raises AgentToolError on connectivity or auth failure.
        Replace body with real SAP client invocation:

            from ..tools.sap import SAPSpendClient
            return SAPSpendClient().get_spend_by_category(category, months=months)
        """
        # INTEGRATION POINT: SAP spend extract
        logger.warning("[SpendCategory] SAP tool not wired — returning empty dataset. "
                       "Inject context.spend_data or implement tools/sap.py.")
        return []

    # =========================================================================
    # Step 2 — Classify spend rows
    # =========================================================================

    def _classify_spend(self, rows: list[dict], focus_category: str) -> list[dict]:
        """
        Assign each row a normalised category label and spend_type bucket.
        Uses LLM batch classification when commodity_code is absent.
        """
        if not rows:
            return []

        # Rows that already have a commodity_code skip LLM classification
        needs_llm  = [r for r in rows if not r.get("commodity_code")]
        has_code   = [r for r in rows if     r.get("commodity_code")]

        classified = list(has_code)
        classified_code = {r["po_number"]: r for r in has_code}

        if needs_llm:
            classified += self._llm_classify_rows(needs_llm)

        return classified

    def _llm_classify_rows(self, rows: list[dict]) -> list[dict]:
        """Batch-classify up to 100 rows per call using LLM."""
        BATCH = 80
        results = []
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            descriptions = [
                {"po_number": r.get("po_number", f"row_{i+j}"),
                 "description": r.get("category_raw", ""),
                 "supplier": r.get("supplier_name", "")}
                for j, r in enumerate(batch)
            ]
            try:
                classifications = extract_json(
                    system=(
                        "You are a procurement taxonomy specialist. "
                        "Classify each purchase order line into a normalised "
                        "category (e.g. 'MRO Fasteners', 'IT Software Licenses', "
                        "'Engineering Services', 'Raw Material — Titanium') and "
                        "a spend_type from: direct_material, indirect, services, "
                        "mro, capex, other.\n"
                        "Return a JSON array: [{po_number, category, spend_type}]"
                    ),
                    user=f"Classify these PO lines:\n{descriptions}",
                    max_tokens=2000,
                )
                # Merge classifications back into rows
                class_map = {c["po_number"]: c for c in classifications}
                for row in batch:
                    key = row.get("po_number", "")
                    if key in class_map:
                        row = {**row,
                               "category":   class_map[key].get("category", "Uncategorised"),
                               "spend_type": class_map[key].get("spend_type", "other")}
                    results.append(row)
            except Exception as exc:
                logger.warning("[SpendCategory] LLM classification failed: %s", exc)
                for row in batch:
                    results.append({**row,
                                    "category":   row.get("category_raw", "Uncategorised"),
                                    "spend_type": "other"})
        return results

    # =========================================================================
    # Step 3 — Aggregate spend metrics
    # =========================================================================

    def _aggregate_spend(self, rows: list[dict]) -> dict:
        """
        Aggregate classified rows into per-category metrics.

        Returns:
          {category_label: {total_spend, transaction_count, supplier_count,
                            avg_transaction, spend_type, suppliers: [...]}}
        """
        buckets: dict[str, dict] = defaultdict(lambda: {
            "total_spend":        0.0,
            "transaction_count":  0,
            "suppliers":          set(),
            "spend_type":         "other",
            "amounts":            [],
        })

        for row in rows:
            cat = row.get("category", "Uncategorised")
            amt = float(row.get("amount_usd", 0))
            buckets[cat]["total_spend"]       += amt
            buckets[cat]["transaction_count"] += 1
            buckets[cat]["suppliers"].add(row.get("supplier_id", "unknown"))
            buckets[cat]["spend_type"]         = row.get("spend_type", "other")
            buckets[cat]["amounts"].append(amt)

        total_spend = sum(b["total_spend"] for b in buckets.values()) or 1.0
        result = {}
        for cat, b in buckets.items():
            amounts = b["amounts"]
            result[cat] = {
                "total_spend":        round(b["total_spend"], 2),
                "spend_share_pct":    round(b["total_spend"] / total_spend * 100, 2),
                "transaction_count":  b["transaction_count"],
                "supplier_count":     len(b["suppliers"]),
                "avg_transaction":    round(statistics.mean(amounts), 2) if amounts else 0.0,
                "spend_type":         b["spend_type"],
                "suppliers":          list(b["suppliers"]),
            }
        return result

    # =========================================================================
    # Step 4 — Kraljic matrix scoring
    # =========================================================================

    def _score_kraljic(self, aggregated: dict, state: dict) -> dict:
        """
        Score each category on two axes and assign a Kraljic quadrant.

        Profit Impact (0-10):
          Driven by spend share, spend_type weight, and strategic importance.

        Supply Risk (0-10):
          Driven by supplier concentration (HHI proxy), category criticality
          signals from LLM, and spend_type.

        Returns:
          {strategic: [...], leverage: [...], bottleneck: [...], non_critical: [...]}
          Each item: {category, profit_impact, supply_risk, quadrant, score_rationale}
        """
        if not aggregated:
            return {"strategic": [], "leverage": [], "bottleneck": [], "non_critical": []}

        # ── Compute raw scores ─────────────────────────────────────────────
        scored: list[dict] = []
        total_spend = sum(v["total_spend"] for v in aggregated.values()) or 1.0

        for cat, metrics in aggregated.items():
            spend_share  = metrics["total_spend"] / total_spend
            type_weight  = _PROFIT_WEIGHT.get(metrics["spend_type"], 0.2)
            supplier_cnt = max(metrics["supplier_count"], 1)

            # Profit impact: spend share (0-10) × type weight
            profit_impact = min(10.0, spend_share * 10 * (1 + type_weight))

            # Supply risk: higher when few suppliers and high spend
            # HHI proxy: 1/n where n = supplier count (higher = more concentrated)
            hhi_proxy   = 1.0 / supplier_cnt
            supply_risk = min(10.0, hhi_proxy * 10 * (1 + spend_share))

            scored.append({
                "category":      cat,
                "profit_impact": round(profit_impact, 2),
                "supply_risk":   round(supply_risk, 2),
                "spend_share":   round(spend_share * 100, 2),
                "spend_type":    metrics["spend_type"],
            })

        # ── LLM refinement for supply risk (adds domain knowledge) ────────
        scored = self._llm_refine_supply_risk(scored, state)

        # ── Assign quadrants ───────────────────────────────────────────────
        matrix: dict[str, list] = {"strategic": [], "leverage": [], "bottleneck": [], "non_critical": []}

        for item in scored:
            pi = item["profit_impact"]
            sr = item["supply_risk"]

            if   pi >= _HIGH_PROFIT_THRESHOLD and sr >= _HIGH_SUPPLY_RISK_THRESHOLD:
                quadrant = "strategic"
            elif pi >= _HIGH_PROFIT_THRESHOLD and sr <  _HIGH_SUPPLY_RISK_THRESHOLD:
                quadrant = "leverage"
            elif pi <  _HIGH_PROFIT_THRESHOLD and sr >= _HIGH_SUPPLY_RISK_THRESHOLD:
                quadrant = "bottleneck"
            else:
                quadrant = "non_critical"

            item["quadrant"] = quadrant
            matrix[quadrant].append(item)

        logger.info("[SpendCategory] Kraljic: strategic=%d leverage=%d bottleneck=%d non_critical=%d",
                    len(matrix["strategic"]), len(matrix["leverage"]),
                    len(matrix["bottleneck"]), len(matrix["non_critical"]))
        return matrix

    def _llm_refine_supply_risk(self, scored: list[dict], state: dict) -> list[dict]:
        """
        Ask the LLM to adjust supply_risk scores based on domain knowledge
        (e.g., single-source aerospace components, geopolitical exposure).
        Returns scored list with refined supply_risk values.
        """
        if not scored:
            return scored

        industry  = state.get("context", {}).get("industry", "aerospace / defense")
        goal_ctx  = state.get("goal", "")

        try:
            refinements = extract_json(
                system=(
                    f"You are a procurement risk analyst specialising in {industry}. "
                    "Given a list of spend categories with initial supply_risk scores (0-10), "
                    "adjust each score based on: single-source risk, geopolitical exposure, "
                    "lead time volatility, regulatory constraints, and substitutability. "
                    "Return a JSON array: [{category, supply_risk, risk_rationale}]. "
                    "Only change scores where you have domain knowledge to justify it."
                ),
                user=(
                    f"Sourcing context: {goal_ctx}\n\n"
                    f"Categories to review:\n{scored}"
                ),
                max_tokens=1500,
            )
            risk_map = {r["category"]: r for r in refinements}
            for item in scored:
                if item["category"] in risk_map:
                    ref = risk_map[item["category"]]
                    item["supply_risk"]      = round(float(ref.get("supply_risk", item["supply_risk"])), 2)
                    item["risk_rationale"]   = ref.get("risk_rationale", "")
        except Exception as exc:
            logger.warning("[SpendCategory] Kraljic LLM refinement skipped: %s", exc)

        return scored

    # =========================================================================
    # Step 5 — Category strategy documents
    # =========================================================================

    _STRATEGY_PLAYBOOKS = {
        "strategic":    ("partnership_and_development",
                         "Build long-term supplier partnerships, joint development, "
                         "volume commitments, and shared risk/reward structures."),
        "leverage":     ("competitive_bidding",
                         "Run competitive RFQ/RFP events to capture market pricing. "
                         "Consolidate volume to preferred suppliers for rebates."),
        "bottleneck":   ("supply_assurance",
                         "Qualify alternative sources, build safety stock, negotiate "
                         "supply guarantee clauses and priority allocation agreements."),
        "non_critical": ("process_efficiency",
                         "Automate purchasing via P-card, e-catalogs, or blanket POs. "
                         "Reduce transaction costs through supplier consolidation."),
    }

    def _generate_category_strategies(self, kraljic: dict, state: dict) -> list[dict]:
        """
        For each Kraljic quadrant, generate a strategy document per category.
        High-spend strategic and leverage categories get LLM-generated strategy briefs.
        """
        strategies = []
        goal   = state.get("goal", "")
        budget = state.get("budget")

        for quadrant, items in kraljic.items():
            playbook_name, playbook_desc = self._STRATEGY_PLAYBOOKS.get(
                quadrant, ("standard", "Apply standard procurement practices."))

            for item in items:
                cat           = item["category"]
                spend_share   = item.get("spend_share", 0)
                is_high_value = spend_share > 5.0  # >5% of total spend

                strategy_doc = self._llm_strategy_brief(
                    category=cat,
                    quadrant=quadrant,
                    playbook_desc=playbook_desc,
                    item=item,
                    goal=goal,
                    budget=budget,
                ) if is_high_value else playbook_desc

                strategies.append({
                    "category":        cat,
                    "quadrant":        quadrant,
                    "approach":        playbook_name,
                    "profit_impact":   item.get("profit_impact"),
                    "supply_risk":     item.get("supply_risk"),
                    "strategy_brief":  strategy_doc,
                    "created_at":      datetime.utcnow().isoformat(),
                })

        return strategies

    def _llm_strategy_brief(self, category: str, quadrant: str, playbook_desc: str,
                             item: dict, goal: str, budget: float | None) -> str:
        """Generate a 200-word strategy brief for a high-value category."""
        try:
            return reason(
                system=(
                    "You are a senior strategic sourcing manager. "
                    "Write a concise category strategy brief (150-200 words) covering: "
                    "recommended sourcing approach, key negotiation levers, "
                    "supplier relationship model, risk mitigations, and "
                    "primary savings/value opportunity. Be specific and actionable."
                ),
                user=(
                    f"Category: {category}\n"
                    f"Kraljic quadrant: {quadrant}\n"
                    f"Playbook guidance: {playbook_desc}\n"
                    f"Profit impact score: {item.get('profit_impact')}/10\n"
                    f"Supply risk score: {item.get('supply_risk')}/10\n"
                    f"Risk rationale: {item.get('risk_rationale', 'N/A')}\n"
                    f"Sourcing goal: {goal}\n"
                    + (f"Budget: ${budget:,.0f}\n" if budget else "")
                ),
                max_tokens=350,
            )
        except Exception as exc:
            logger.warning("[SpendCategory] Strategy brief LLM call failed: %s", exc)
            return playbook_desc

    # =========================================================================
    # Step 6 — Tail spend analysis
    # =========================================================================

    def _analyse_tail_spend(self, rows: list[dict], aggregated: dict) -> dict:
        """
        Identify tail spend (bottom 80% of suppliers by spend, top 80% by count),
        maverick buying flags, and consolidation opportunities.

        Returns:
          {tail_spend_usd, tail_supplier_count, consolidation_opportunities: [...],
           maverick_buying_flags: [...], pcard_candidates: [...]}
        """
        if not rows:
            return {
                "tail_spend_usd":               0.0,
                "tail_supplier_count":           0,
                "consolidation_opportunities":   [],
                "maverick_buying_flags":          [],
                "pcard_candidates":              [],
            }

        # Pareto: sort suppliers by spend descending
        supplier_spend: dict[str, float] = defaultdict(float)
        for row in rows:
            supplier_spend[row.get("supplier_id", "unknown")] += float(row.get("amount_usd", 0))

        sorted_suppliers = sorted(supplier_spend.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in sorted_suppliers) or 1.0

        # Find 80% spend threshold (Pareto frontier)
        cumulative = 0.0
        pareto_count = 0
        for _, spend in sorted_suppliers:
            cumulative += spend
            pareto_count += 1
            if cumulative / total >= 0.80:
                break

        tail_suppliers = sorted_suppliers[pareto_count:]
        tail_spend_usd = sum(v for _, v in tail_suppliers)

        # Identify categories with avg transaction < $2,500 as P-card candidates
        pcard_candidates = [
            {"category": cat, "avg_transaction": m["avg_transaction"],
             "transaction_count": m["transaction_count"]}
            for cat, m in aggregated.items()
            if m.get("avg_transaction", 0) < 2500 and m.get("transaction_count", 0) > 5
        ]

        # Maverick: multiple suppliers in same category with no preferred vendor flag
        maverick_flags = [
            {"category": cat,
             "supplier_count": m["supplier_count"],
             "recommendation": "Consolidate to 1-2 preferred suppliers"}
            for cat, m in aggregated.items()
            if m.get("supplier_count", 0) > 4 and m.get("spend_share_pct", 0) < 3.0
        ]

        # Consolidation: categories where top supplier has <60% of category spend
        consolidation_opps = []
        for cat, m in aggregated.items():
            if m.get("supplier_count", 1) > 2 and m.get("spend_share_pct", 0) > 1.0:
                consolidation_opps.append({
                    "category":           cat,
                    "current_suppliers":  m["supplier_count"],
                    "recommended_target": max(1, m["supplier_count"] // 2),
                    "estimated_savings_pct": 8.0,  # industry benchmark for consolidation
                })

        return {
            "tail_spend_usd":               round(tail_spend_usd, 2),
            "tail_supplier_count":           len(tail_suppliers),
            "pareto_supplier_count":         pareto_count,
            "consolidation_opportunities":   consolidation_opps,
            "maverick_buying_flags":          maverick_flags,
            "pcard_candidates":              pcard_candidates,
        }

    # =========================================================================
    # Step 7 — Make-vs-buy analysis
    # =========================================================================

    def _make_vs_buy(self, category: str, aggregated: dict, context: dict) -> dict:
        """
        Compare internal production cost against external supplier cost for a
        given category.  Uses LLM to assess qualitative factors and compute
        a structured recommendation.
        """
        cat_metrics   = aggregated.get(category, {})
        internal_cost = context.get("internal_cost_usd") or 0.0
        external_cost = cat_metrics.get("total_spend", 0.0)
        volume        = context.get("annual_volume_units", 1)

        # LLM qualitative assessment
        try:
            assessment = extract_json(
                system=(
                    "You are a make-vs-buy analyst. Assess the decision and return JSON: "
                    "{recommendation: 'make'|'buy'|'hybrid', "
                    "cost_advantage: 'make'|'buy'|'neutral', "
                    "strategic_factors: [str], "
                    "risk_factors: [str], "
                    "rationale: str (2-3 sentences)}"
                ),
                user=(
                    f"Category: {category}\n"
                    f"Annual internal production cost: ${internal_cost:,.0f}\n"
                    f"Current external spend: ${external_cost:,.0f}\n"
                    f"Annual volume: {volume} units\n"
                    f"Additional context: {context.get('make_vs_buy_context', 'N/A')}\n"
                    "Consider: core competency alignment, capital investment required, "
                    "quality control, IP protection, capacity flexibility, and lead time."
                ),
                max_tokens=600,
            )
        except Exception as exc:
            logger.warning("[SpendCategory] Make-vs-buy LLM failed: %s", exc)
            assessment = {
                "recommendation":    "buy" if external_cost < internal_cost else "make",
                "cost_advantage":    "buy" if external_cost < internal_cost else "make",
                "strategic_factors": [],
                "risk_factors":      [],
                "rationale":         "Assessment unavailable — using cost comparison only.",
            }

        return {
            "category":          category,
            "internal_cost_usd": internal_cost,
            "external_cost_usd": external_cost,
            "cost_delta_usd":    round(external_cost - internal_cost, 2),
            "annual_volume":     volume,
            **assessment,
            "analysed_at":       datetime.utcnow().isoformat(),
        }