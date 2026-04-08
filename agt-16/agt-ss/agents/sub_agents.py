# Databricks notebook source
"""
agt16.agents.sub_agents
------------------------
Concrete sub-agent implementations for AGT-16 Supply Chain Analytics &
Intelligence Agent.

Each class inherits BaseAgent and implements:
  - can_run(state) → dependency check
  - _execute(state) → partial state update

Tool calls are stubbed with clear TODO markers; replace with real
S3 / RDS / Anthropic / market-data API invocations when wiring up the
full system.  See tools/ for client stubs.
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
# CM-01 — Data Integration & Aggregation Engine
# ---------------------------------------------------------------------------

class DataIntegrationAgent(BaseAgent):
    """
    Processes owned (FR-01):
      - Connect to ERP, WMS, TMS, planning, and procurement source systems
      - Run ETL pipelines: Bronze → Silver → Gold S3 layers
      - Enforce data quality gates (completeness, freshness, referential integrity)
      - Produce schema map and ingestion summary

    Preconditions: none — runs first on every engagement workflow.
    """

    name = AgentName.DATA_INTEGRATION

    def can_run(self, state: dict) -> bool:
        return bool(state.get("goal"))

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        logger.info("[DataIntegration] Starting ingestion for engagement %s", engagement_id)

        # ── Tool 1: pull source system inventory from engagement config ──
        source_systems = self._tool_get_source_systems(engagement_id, state)

        # ── Tool 2: run ETL — Bronze layer (raw ingest) ──────────────────
        bronze_paths = self._tool_ingest_bronze(source_systems, engagement_id)

        # ── Tool 3: cleanse and conform — Silver layer ───────────────────
        silver_paths, quality_report = self._tool_promote_silver(bronze_paths)

        # ── Tool 4: aggregate — Gold layer (analytics-ready) ─────────────
        gold_paths = self._tool_promote_gold(silver_paths)

        # ── Tool 5: build schema map ─────────────────────────────────────
        schema_map = self._tool_build_schema_map(silver_paths)

        ingestion_summary = {
            "engagement_id": engagement_id,
            "source_count":  len(source_systems),
            "quality_score": quality_report.get("overall_score", 0.0),
            "loaded_at":     datetime.utcnow().isoformat(),
        }

        return {
            "data_integration": {
                "ingestion_summary":  ingestion_summary,
                "data_quality_report": quality_report,
                "schema_map":         schema_map,
                "s3_paths":           {"bronze": bronze_paths,
                                       "silver": silver_paths,
                                       "gold":   gold_paths},
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_get_source_systems(self, engagement_id: str, state: dict) -> list[dict]:
        """Read engagement config to determine which source systems to connect."""
        # TODO: from ..tools.rds import EngagementConfigClient
        #       return EngagementConfigClient().get_source_systems(engagement_id)
        logger.debug("[DataIntegration] Loading source system config")
        return state.get("context", {}).get("source_systems", [
            {"type": "ERP",       "name": "SAP S/4HANA"},
            {"type": "WMS",       "name": "Manhattan Active WM"},
            {"type": "TMS",       "name": "Oracle TMS"},
            {"type": "Planning",  "name": "Kinaxis RapidResponse"},
            {"type": "Procurement", "name": "Coupa"},
        ])

    def _tool_ingest_bronze(self, source_systems: list, engagement_id: str) -> list[str]:
        """Connect to each source system and land raw data to S3 Bronze prefix."""
        # TODO: from ..tools.s3 import S3DataClient
        #       client = S3DataClient()
        #       for system in source_systems:
        #           client.ingest_source(system, f"s3://{BUCKET}/bronze/{engagement_id}/")
        logger.debug("[DataIntegration] Ingesting %d source systems to Bronze", len(source_systems))
        bucket = "agt16-data"
        return [f"s3://{bucket}/bronze/{engagement_id}/{s['type'].lower()}/" for s in source_systems]

    def _tool_promote_silver(self, bronze_paths: list) -> tuple[list[str], dict]:
        """Cleanse, deduplicate, conform schema; run quality gates; write Silver."""
        # TODO: invoke PySpark / pandas ETL job (ECS task or Step Functions)
        #       quality gates: completeness >= 95%, referential integrity checks
        logger.debug("[DataIntegration] Promoting %d Bronze paths to Silver", len(bronze_paths))
        silver_paths = [p.replace("/bronze/", "/silver/") for p in bronze_paths]
        quality_report = {
            "overall_score":       0.97,
            "completeness_pct":    0.97,
            "freshness_ok":        True,
            "referential_ok":      True,
            "anomalies_flagged":   0,
            "generated_at":        datetime.utcnow().isoformat(),
        }
        return silver_paths, quality_report

    def _tool_promote_gold(self, silver_paths: list) -> list[str]:
        """Aggregate Silver into analytics-ready Gold tables."""
        # TODO: run aggregation SQL / Spark job; write partitioned Parquet to Gold
        logger.debug("[DataIntegration] Building Gold layer")
        return [p.replace("/silver/", "/gold/") for p in silver_paths]

    def _tool_build_schema_map(self, silver_paths: list) -> dict:
        """Infer unified field mappings across source schemas."""
        # TODO: schema inference from Parquet metadata + LLM-assisted field matching
        return {"status": "pending", "silver_paths": silver_paths}


# ---------------------------------------------------------------------------
# CM-02 — Performance Baseline & Benchmarking Module
# ---------------------------------------------------------------------------

class PerformanceBaselineAgent(BaseAgent):
    """
    Processes owned (FR-02, FR-04):
      - Build 5-dimension KPI baseline: cost, service, inventory, quality, resilience
      - Map baseline KPIs against Gartner / Hackett peer-group benchmarks
      - Compute gap-to-best-in-class with statistical significance

    Preconditions: data_integration Gold layer must be ready.
    """

    name = AgentName.PERFORMANCE_BASELINE

    def can_run(self, state: dict) -> bool:
        return bool(state.get("data_integration", {}).get("s3_paths", {}).get("gold"))

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        gold_paths    = state["data_integration"]["s3_paths"]["gold"]
        logger.info("[PerformanceBaseline] Building KPI baseline for %s", engagement_id)

        # ── Tool 1: compute 5-dimension KPI baseline ──────────────────────
        kpi_baseline = self._tool_compute_kpi_baseline(gold_paths, engagement_id)

        # ── Tool 2: pull benchmark data from RDS benchmark library ────────
        benchmarks = self._tool_pull_benchmarks(
            state.get("context", {}).get("peer_group", "industrial_manufacturing")
        )

        # ── Tool 3: compute gaps with statistical tests ───────────────────
        benchmark_comparison = self._tool_compute_gaps(kpi_baseline, benchmarks)

        # ── Tool 4: LLM narrative on most significant gaps ────────────────
        gap_narrative = reason(
            system=(
                "You are a supply chain performance analyst. Given a KPI baseline and "
                "benchmark comparison, identify the top 3 most critical performance gaps "
                "and explain the business impact concisely."
            ),
            user=f"Baseline: {kpi_baseline}\nBenchmark gaps: {benchmark_comparison}",
            max_tokens=600,
        )

        completeness = self._compute_completeness(kpi_baseline)

        return {
            "performance_baseline": {
                "kpi_baseline":               kpi_baseline,
                "benchmark_comparison":       benchmark_comparison,
                "peer_group":                 state.get("context", {}).get("peer_group", ""),
                "baseline_completeness_pct":  completeness,
                "gap_narrative":              gap_narrative,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_compute_kpi_baseline(self, gold_paths: list, engagement_id: str) -> dict:
        """Query Gold layer to compute the 5-dimension KPI baseline."""
        # TODO: from ..tools.rds import AnalyticsQueryClient
        #       return AnalyticsQueryClient().compute_baseline(engagement_id)
        logger.debug("[PerformanceBaseline] Computing KPI baseline from Gold")
        return {
            "cost": {
                "total_supply_chain_cost_pct_revenue": 0.0,
                "logistics_cost_per_unit": 0.0,
                "inventory_carrying_cost_pct": 0.0,
            },
            "service": {
                "otif_pct": 0.0,
                "order_cycle_time_days": 0.0,
                "fill_rate_pct": 0.0,
            },
            "inventory": {
                "inventory_turns": 0.0,
                "days_on_hand": 0.0,
                "excess_and_obsolete_pct": 0.0,
            },
            "quality": {
                "supplier_defect_rate_ppm": 0.0,
                "return_rate_pct": 0.0,
            },
            "resilience": {
                "supplier_concentration_hhi": 0.0,
                "single_source_pct": 0.0,
                "geographic_concentration_pct": 0.0,
            },
            "computed_at": datetime.utcnow().isoformat(),
        }

    def _tool_pull_benchmarks(self, peer_group: str) -> dict:
        """Query RDS benchmark library for Gartner / Hackett peer-group data."""
        # TODO: from ..tools.rds import BenchmarkClient
        #       return BenchmarkClient().get_benchmarks(peer_group)
        logger.debug("[PerformanceBaseline] Pulling benchmarks for peer group: %s", peer_group)
        return {
            "peer_group":    peer_group,
            "source":        "gartner_hackett_2024",
            "otif_pct":      {"median": 0.89, "best_in_class": 0.97},
            "inventory_turns": {"median": 8.2, "best_in_class": 14.1},
            "logistics_cost_pct": {"median": 0.082, "best_in_class": 0.051},
        }

    def _tool_compute_gaps(self, baseline: dict, benchmarks: dict) -> dict:
        """Compute gap = client - best_in_class for each benchmarked KPI."""
        # TODO: iterate over benchmark KPIs; compute gap; run significance test
        logger.debug("[PerformanceBaseline] Computing benchmark gaps")
        return {
            "peer_group":   benchmarks.get("peer_group"),
            "source":       benchmarks.get("source"),
            "gaps":         {},   # populated when real baseline data is available
            "computed_at":  datetime.utcnow().isoformat(),
        }

    def _compute_completeness(self, baseline: dict) -> float:
        """Estimate baseline completeness as fraction of non-zero KPI fields."""
        all_vals = []
        for dim in baseline.values():
            if isinstance(dim, dict):
                all_vals.extend(v for v in dim.values() if isinstance(v, (int, float)))
        if not all_vals:
            return 0.0
        non_zero = sum(1 for v in all_vals if v != 0.0)
        return round(non_zero / len(all_vals), 3)


# ---------------------------------------------------------------------------
# CM-03 — Dashboard & Visualization Engine
# ---------------------------------------------------------------------------

class DashboardVizAgent(BaseAgent):
    """
    Processes owned (FR-03, FR-15):
      - Generate executive supply chain visibility dashboards
      - Produce board/C-suite presentation decks
      - Export static PDF/PNG exhibits to S3

    Preconditions: performance_baseline must be complete.
    """

    name = AgentName.DASHBOARD_VIZ

    def can_run(self, state: dict) -> bool:
        return bool(state.get("performance_baseline", {}).get("kpi_baseline"))

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        logger.info("[DashboardViz] Building dashboards for %s", engagement_id)

        # ── Tool 1: render executive KPI dashboard ────────────────────────
        dashboard_urls = self._tool_render_dashboards(engagement_id, state)

        # ── Tool 2: generate static exhibit package ───────────────────────
        exhibit_keys = self._tool_export_exhibits(engagement_id, state)

        # ── Tool 3: build board presentation deck ─────────────────────────
        deck_s3_key = self._tool_build_board_deck(engagement_id, state)

        return {
            "dashboard_visualization": {
                "dashboard_urls":     dashboard_urls,
                "exhibit_s3_keys":    exhibit_keys,
                "presentation_s3_key": deck_s3_key,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_render_dashboards(self, engagement_id: str, state: dict) -> list[dict]:
        """Publish data to QuickSight / Tableau and return embed URLs."""
        # TODO: from ..tools.s3 import QuickSightClient
        #       return QuickSightClient().publish_dashboard(engagement_id, kpi_baseline)
        logger.debug("[DashboardViz] Rendering KPI dashboards")
        return [
            {"name": "Supply Chain Executive Overview", "tool": "QuickSight", "url": "", "refreshed_at": datetime.utcnow().isoformat()},
            {"name": "Benchmark Comparison",            "tool": "QuickSight", "url": "", "refreshed_at": datetime.utcnow().isoformat()},
            {"name": "Cost-to-Serve Heatmap",           "tool": "QuickSight", "url": "", "refreshed_at": datetime.utcnow().isoformat()},
        ]

    def _tool_export_exhibits(self, engagement_id: str, state: dict) -> list[str]:
        """Render charts as PNG/PDF and upload to S3 exhibits prefix."""
        # TODO: matplotlib / plotly render → S3 upload
        logger.debug("[DashboardViz] Exporting static exhibits")
        bucket = "agt16-deliverables"
        return [
            f"s3://{bucket}/{engagement_id}/exhibits/kpi_baseline.pdf",
            f"s3://{bucket}/{engagement_id}/exhibits/benchmark_gap_chart.pdf",
        ]

    def _tool_build_board_deck(self, engagement_id: str, state: dict) -> str:
        """Generate board-level PowerPoint deck from template + LLM narrative."""
        # TODO: python-pptx template fill + S3 upload
        #       narrative = reason(system=BOARD_DECK_SYSTEM, user=state_summary)
        logger.debug("[DashboardViz] Building board presentation")
        bucket = "agt16-deliverables"
        return f"s3://{bucket}/{engagement_id}/presentations/board_deck.pptx"


# ---------------------------------------------------------------------------
# CM-04 — Cost Analytics Module
# ---------------------------------------------------------------------------

class CostAnalyticsAgent(BaseAgent):
    """
    Processes owned (FR-05, FR-08):
      - Cost-to-serve analysis: landed cost by customer, channel, product
      - Financial impact models: NPV, IRR, payback for all recommendations
      - Savings quantification by improvement lever

    Preconditions: performance_baseline must be complete.
    """

    name = AgentName.COST_ANALYTICS

    def can_run(self, state: dict) -> bool:
        return bool(state.get("performance_baseline", {}).get("kpi_baseline"))

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        logger.info("[CostAnalytics] Running cost-to-serve analysis for %s", engagement_id)

        # ── Tool 1: cost-to-serve decomposition ──────────────────────────
        cts_matrix = self._tool_cost_to_serve(
            state["data_integration"]["s3_paths"]["gold"],
            engagement_id,
        )

        # ── Tool 2: identify improvement levers from baseline gaps ────────
        levers = self._tool_identify_levers(
            state.get("performance_baseline", {}).get("benchmark_comparison", {}),
            state.get("context", {}),
        )

        # ── Tool 3: build financial impact models for each lever ──────────
        financial_models = self._tool_build_financial_models(levers, cts_matrix, state)

        # ── Tool 4: aggregate savings quantification ──────────────────────
        savings = self._tool_aggregate_savings(financial_models)

        return {
            "cost_analytics": {
                "cost_to_serve_matrix":    cts_matrix,
                "financial_impact_models": financial_models,
                "savings_quantification":  savings,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_cost_to_serve(self, gold_paths: list, engagement_id: str) -> dict:
        """Decompose landed cost by customer segment, channel, and product SKU."""
        # TODO: SQL aggregation over order mgmt + logistics cost Gold tables
        logger.debug("[CostAnalytics] Computing cost-to-serve matrix")
        return {
            "by_customer":  {},   # {customer_segment -> avg_landed_cost}
            "by_channel":   {},   # {channel -> avg_landed_cost}
            "by_product":   {},   # {sku -> avg_landed_cost}
            "computed_at":  datetime.utcnow().isoformat(),
        }

    def _tool_identify_levers(self, benchmark_gaps: dict, context: dict) -> list[dict]:
        """Use LLM to map benchmark gaps to specific cost improvement levers."""
        if not benchmark_gaps.get("gaps"):
            return []
        levers_raw = extract_json(
            system=(
                "You are a supply chain cost analyst. Given benchmark gaps, identify "
                "3-5 specific cost improvement levers. Return a JSON array where each "
                "element has: lever_name, description, estimated_savings_pct, timeline_months."
            ),
            user=f"Benchmark gaps: {benchmark_gaps}",
        )
        return levers_raw if isinstance(levers_raw, list) else []

    def _tool_build_financial_models(
        self, levers: list, cts_matrix: dict, state: dict
    ) -> list[dict]:
        """Build NPV / IRR / payback model for each improvement lever."""
        # TODO: numpy_financial NPV/IRR calculation; populate from actual cost data
        logger.debug("[CostAnalytics] Building financial models for %d levers", len(levers))
        models = []
        for lever in levers:
            models.append({
                "recommendation":  lever.get("lever_name", ""),
                "description":     lever.get("description", ""),
                "npv":             0.0,
                "irr":             0.0,
                "payback_days":    0,
                "est_savings_pct": lever.get("estimated_savings_pct", 0.0),
                "modeled_at":      datetime.utcnow().isoformat(),
            })
        return models

    def _tool_aggregate_savings(self, financial_models: list) -> dict:
        """Sum savings across all levers; break out by category."""
        total = sum(m.get("npv", 0.0) for m in financial_models)
        return {
            "total_identified_npv": total,
            "lever_count":          len(financial_models),
            "by_lever":             {m["recommendation"]: m["npv"] for m in financial_models},
        }


# ---------------------------------------------------------------------------
# CM-05 — Diagnostics & Root Cause Analysis Engine
# ---------------------------------------------------------------------------

class DiagnosticsRCAAgent(BaseAgent):
    """
    Processes owned (FR-06):
      - Statistical root cause analysis on supply chain performance gaps
      - Anomaly detection, causal inference, feature importance ranking
      - Hypothesis test results and explainability reports

    Preconditions: performance_baseline + benchmark_comparison must be present.
    """

    name = AgentName.DIAGNOSTICS_RCA

    def can_run(self, state: dict) -> bool:
        return bool(
            state.get("performance_baseline", {}).get("benchmark_comparison")
            and state.get("performance_baseline", {}).get("kpi_baseline")
        )

    def _execute(self, state: dict) -> dict:
        logger.info("[DiagnosticsRCA] Running root cause analysis")

        # ── Tool 1: extract significant performance gaps ──────────────────
        gaps = self._tool_extract_gaps(
            state["performance_baseline"]["benchmark_comparison"],
            state["performance_baseline"]["kpi_baseline"],
        )

        # ── Tool 2: statistical anomaly detection on time-series data ─────
        anomaly_results = self._tool_detect_anomalies(
            state["data_integration"]["s3_paths"]["gold"], gaps
        )

        # ── Tool 3: causal inference / feature importance ─────────────────
        root_causes = self._tool_causal_analysis(gaps, anomaly_results, state)

        # ── Tool 4: hypothesis testing ────────────────────────────────────
        hypothesis_results = self._tool_hypothesis_tests(gaps, state)

        # ── Tool 5: LLM-generated RCA narrative per gap ───────────────────
        root_causes = self._tool_enrich_rca_narrative(root_causes, state)

        return {
            "diagnostics_rca": {
                "performance_gaps":        gaps,
                "root_causes":             root_causes,
                "hypothesis_test_results": hypothesis_results,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_extract_gaps(self, benchmark_comparison: dict, kpi_baseline: dict) -> list[dict]:
        """Identify KPIs where client performance falls below peer median."""
        # TODO: iterate benchmark gaps dict; flag gap_pct > threshold
        logger.debug("[DiagnosticsRCA] Extracting performance gaps")
        gaps = []
        for kpi, benchmarks in benchmark_comparison.get("gaps", {}).items():
            client_val = benchmarks.get("client", 0.0)
            best_val   = benchmarks.get("best_in_class", 0.0)
            if best_val and client_val < best_val:
                gap_pct = round((best_val - client_val) / best_val, 3) if best_val else 0.0
                gaps.append({
                    "kpi":      kpi,
                    "gap_pct":  gap_pct,
                    "severity": "high" if gap_pct > 0.20 else "medium" if gap_pct > 0.10 else "low",
                })
        return gaps

    def _tool_detect_anomalies(self, gold_paths: list, gaps: list) -> dict:
        """Run IsolationForest / statistical outlier detection on KPI time series."""
        # TODO: scikit-learn IsolationForest on Gold Parquet time series
        logger.debug("[DiagnosticsRCA] Detecting anomalies for %d gaps", len(gaps))
        return {"anomalies": [], "detection_method": "isolation_forest"}

    def _tool_causal_analysis(self, gaps: list, anomaly_results: dict, state: dict) -> list[dict]:
        """Run feature importance / causal inference to rank root cause candidates."""
        # TODO: SHAP values over RandomForest model trained on Gold data
        logger.debug("[DiagnosticsRCA] Running causal analysis")
        return [
            {"gap": g["kpi"], "cause": "pending_analysis", "evidence": "", "confidence": 0.0}
            for g in gaps
        ]

    def _tool_hypothesis_tests(self, gaps: list, state: dict) -> list[dict]:
        """Run statistical significance tests on suspected root causes."""
        # TODO: scipy.stats t-test / chi2 / Mann-Whitney U per hypothesis
        return []

    def _tool_enrich_rca_narrative(self, root_causes: list, state: dict) -> list[dict]:
        """Use LLM to generate a clear business narrative for each root cause."""
        if not root_causes:
            return root_causes
        enriched = extract_json(
            system=(
                "You are a supply chain diagnostic expert. For each root cause entry, "
                "write a concise business-language explanation (2-3 sentences) of the cause "
                "and its operational impact. Return the same JSON array with a 'narrative' "
                "field added to each element."
            ),
            user=f"Root causes: {root_causes}",
        )
        return enriched if isinstance(enriched, list) else root_causes


# ---------------------------------------------------------------------------
# CM-06 — Predictive Analytics Module
# ---------------------------------------------------------------------------

class PredictiveAnalyticsAgent(BaseAgent):
    """
    Processes owned (FR-14):
      - Demand signal forecasting (MAPE target <= 15%)
      - Supply risk probability scoring (AUC target >= 0.75)
      - Capacity constraint prediction
      - Cost trend forecasting

    Preconditions: data_integration Gold + performance_baseline must be complete.
    """

    name = AgentName.PREDICTIVE_ANALYTICS

    def can_run(self, state: dict) -> bool:
        return (
            bool(state.get("data_integration", {}).get("s3_paths", {}).get("gold"))
            and bool(state.get("performance_baseline", {}).get("kpi_baseline"))
        )

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        logger.info("[PredictiveAnalytics] Running predictive models for %s", engagement_id)

        # ── Tool 1: demand forecasting ────────────────────────────────────
        demand_forecast = self._tool_demand_forecast(
            state["data_integration"]["s3_paths"]["gold"], engagement_id
        )

        # ── Tool 2: supply risk scoring ───────────────────────────────────
        risk_scores = self._tool_supply_risk_scoring(
            state.get("context", {}).get("supplier_nodes", [])
        )

        # ── Tool 3: capacity constraint prediction ────────────────────────
        capacity_forecast = self._tool_capacity_forecast(
            state["data_integration"]["s3_paths"]["gold"]
        )

        # ── Tool 4: cost trend forecasting ────────────────────────────────
        cost_forecast = self._tool_cost_forecast(
            state.get("market_intelligence", {}).get("commodity_prices", {})
        )

        model_metadata = {
            "engagement_id": engagement_id,
            "models_run":    ["demand_forecast", "risk_score", "capacity_forecast", "cost_forecast"],
            "run_at":        datetime.utcnow().isoformat(),
        }

        return {
            "predictive_analytics": {
                "demand_forecast":   demand_forecast,
                "risk_scores":       risk_scores,
                "capacity_forecast": capacity_forecast,
                "cost_forecast":     cost_forecast,
                "model_metadata":    model_metadata,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_demand_forecast(self, gold_paths: list, engagement_id: str) -> dict:
        """Train / retrieve demand forecast model; return series + MAPE."""
        # TODO: load Gold demand history from S3 Parquet
        #       fit Prophet / XGBoost; register in MLflow; return forecast
        logger.debug("[PredictiveAnalytics] Running demand forecast")
        return {
            "series":       [],    # [{date, forecast, lower, upper}]
            "horizon_days": 90,
            "mape":         None,  # populated after model fit
            "model_id":     None,
        }

    def _tool_supply_risk_scoring(self, supplier_nodes: list) -> list[dict]:
        """Score each supply node on disruption probability + severity."""
        # TODO: XGBoost risk model trained on supplier performance history
        #       features: lead time variance, financial health, geo concentration
        logger.debug("[PredictiveAnalytics] Scoring %d supplier nodes", len(supplier_nodes))
        return [
            {"node": n, "probability": 0.0, "severity": "medium"}
            for n in supplier_nodes
        ]

    def _tool_capacity_forecast(self, gold_paths: list) -> dict:
        """Predict capacity utilization constraints over planning horizon."""
        # TODO: linear regression over historical utilization + demand forecast
        return {"constraints": [], "horizon_days": 90}

    def _tool_cost_forecast(self, commodity_prices: dict) -> dict:
        """Project cost trends based on commodity price movements."""
        # TODO: ARIMA / Prophet on commodity index time series
        return {"projections": [], "horizon_days": 90}


# ---------------------------------------------------------------------------
# CM-07 — Market Intelligence & Monitoring Module
# ---------------------------------------------------------------------------

class MarketIntelligenceAgent(BaseAgent):
    """
    Processes owned (FR-09, FR-13, FR-17):
      - Commodity price monitoring (raw materials, energy)
      - Carrier rate index tracking
      - Labor cost trend monitoring
      - Supply chain disruption signal detection
      - Competitive intelligence on peer strategies
      - Regulatory trend reports

    Preconditions: none — runs standalone in monitoring mode or within engagements.
    """

    name = AgentName.MARKET_INTELLIGENCE

    def can_run(self, state: dict) -> bool:
        return bool(state.get("goal"))

    def _execute(self, state: dict) -> dict:
        logger.info("[MarketIntelligence] Ingesting market intelligence signals")

        # ── Tool 1: commodity prices ──────────────────────────────────────
        commodity_prices = self._tool_pull_commodity_prices()

        # ── Tool 2: carrier rate indices ──────────────────────────────────
        carrier_rates = self._tool_pull_carrier_rates()

        # ── Tool 3: labor cost indices ────────────────────────────────────
        labor_cost_index = self._tool_pull_labor_costs()

        # ── Tool 4: disruption signal detection ───────────────────────────
        disruption_signals = self._tool_detect_disruptions(
            commodity_prices, carrier_rates
        )

        # ── Tool 5: competitive intelligence ─────────────────────────────
        competitive_intel = self._tool_competitive_intel(
            state.get("context", {}).get("peer_companies", [])
        )

        # ── Tool 6: regulatory trend scan ─────────────────────────────────
        regulatory_trends = self._tool_regulatory_scan(
            state.get("context", {}).get("regulatory_domains",
                ["trade_policy", "esg", "customs", "logistics_regulation"])
        )

        # ── Tool 7: synthesize digest and upload to S3 ────────────────────
        digest_s3_key = self._tool_generate_digest({
            "commodity_prices":  commodity_prices,
            "carrier_rates":     carrier_rates,
            "disruption_signals": disruption_signals,
            "competitive_intel": competitive_intel,
            "regulatory_trends": regulatory_trends,
        })

        return {
            "market_intelligence": {
                "commodity_prices":   commodity_prices,
                "carrier_rates":      carrier_rates,
                "labor_cost_index":   labor_cost_index,
                "disruption_signals": disruption_signals,
                "competitive_intel":  competitive_intel,
                "regulatory_trends":  regulatory_trends,
                "digest_s3_key":      digest_s3_key,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_pull_commodity_prices(self) -> dict:
        """Fetch commodity spot and futures prices from market data APIs."""
        # TODO: from ..tools.market import MarketDataClient
        #       return MarketDataClient().get_commodity_prices(["steel", "aluminum", "diesel", "polyethylene"])
        logger.debug("[MarketIntelligence] Pulling commodity prices")
        return {}

    def _tool_pull_carrier_rates(self) -> dict:
        """Fetch ocean, air, and truckload rate indices."""
        # TODO: Freightos API / DAT / Xeneta for carrier rate data
        logger.debug("[MarketIntelligence] Pulling carrier rates")
        return {}

    def _tool_pull_labor_costs(self) -> dict:
        """Fetch regional labor cost indices."""
        # TODO: BLS API / regional labor market data sources
        logger.debug("[MarketIntelligence] Pulling labor cost indices")
        return {}

    def _tool_detect_disruptions(self, commodity_prices: dict, carrier_rates: dict) -> list[dict]:
        """Detect supply chain disruption signals from price movements and news."""
        # TODO: anomaly detection on price time series
        #       + web search for "supply chain disruption" news via Anthropic web_search tool
        logger.debug("[MarketIntelligence] Running disruption detection")
        return []

    def _tool_competitive_intel(self, peer_companies: list) -> list[dict]:
        """Scrape SEC filings, earnings calls, and press releases for supply chain intel."""
        # TODO: AWS Bedrock Knowledge Base query + web search per company
        logger.debug("[MarketIntelligence] Pulling competitive intel for %d companies", len(peer_companies))
        return []

    def _tool_regulatory_scan(self, domains: list) -> list[dict]:
        """Scan regulatory feeds for supply chain-relevant developments."""
        # TODO: Federal Register API + EUR-Lex + custom regulatory RSS feeds
        logger.debug("[MarketIntelligence] Scanning regulatory domains: %s", domains)
        return []

    def _tool_generate_digest(self, intel_bundle: dict) -> str:
        """Use LLM to synthesize a structured weekly intelligence digest; upload to S3."""
        digest_text = reason(
            system=(
                "You are a supply chain market intelligence analyst. Synthesize the provided "
                "market signals into a concise executive intelligence digest. Structure it as: "
                "1) Commodity & Freight Market Pulse, 2) Disruption Watch, "
                "3) Competitive Moves, 4) Regulatory Alerts."
            ),
            user=f"Market data bundle: {intel_bundle}",
            max_tokens=1200,
        )
        # TODO: from ..tools.s3 import S3DeliverableClient
        #       key = S3DeliverableClient().upload_digest(digest_text)
        #       return key
        logger.debug("[MarketIntelligence] Generated digest (%d chars)", len(digest_text))
        bucket = "agt16-deliverables"
        return f"s3://{bucket}/market-intelligence/digest_{datetime.utcnow().date()}.txt"


# ---------------------------------------------------------------------------
# CM-08 — Engagement Reporting Module
# ---------------------------------------------------------------------------

class EngagementReportingAgent(BaseAgent):
    """
    Processes owned (FR-07, FR-10, FR-12):
      - Weekly engagement status reports: milestone tracking, KPI vs. target
      - Client deliverable exhibit packages
      - Post-implementation benefit realization reports

    Preconditions: data_integration + performance_baseline must be present.
    """

    name = AgentName.ENGAGEMENT_REPORTING

    def can_run(self, state: dict) -> bool:
        return (
            bool(state.get("data_integration", {}).get("ingestion_summary"))
            and bool(state.get("performance_baseline", {}).get("kpi_baseline"))
        )

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        logger.info("[EngagementReporting] Generating engagement reports for %s", engagement_id)

        # ── Tool 1: weekly status report ──────────────────────────────────
        weekly_report = self._tool_weekly_status_report(engagement_id, state)

        # ── Tool 2: assemble and zip deliverable exhibit package ──────────
        package_key = self._tool_assemble_deliverable_package(engagement_id, state)

        # ── Tool 3: benefit realization (post-implementation only) ────────
        benefit_report = {}
        if state.get("context", {}).get("post_implementation"):
            benefit_report = self._tool_benefit_realization(engagement_id, state)

        # ── Tool 4: upload status report to S3 ───────────────────────────
        report_key = self._tool_upload_report(engagement_id, weekly_report)

        return {
            "engagement_reporting": {
                "weekly_status_report":        weekly_report,
                "deliverable_package_s3_key":  package_key,
                "benefit_realization_report":  benefit_report,
                "report_s3_key":               report_key,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_weekly_status_report(self, engagement_id: str, state: dict) -> dict:
        """Compile weekly KPI performance vs. targets and milestone status."""
        kpi_baseline = state.get("performance_baseline", {}).get("kpi_baseline", {})
        kpi_targets  = state.get("kpi_targets", {})

        # LLM to synthesize health narrative
        health_narrative = reason(
            system=(
                "You are an engagement manager. Given KPI baseline and targets, write a "
                "2-paragraph weekly status narrative covering: (1) performance vs. targets, "
                "(2) key risks and upcoming milestones."
            ),
            user=(
                f"Engagement: {engagement_id}\n"
                f"KPI Baseline: {kpi_baseline}\n"
                f"KPI Targets: {kpi_targets}"
            ),
            max_tokens=400,
        )

        return {
            "engagement_id":   engagement_id,
            "report_week":     datetime.utcnow().isocalendar().week,
            "milestone_status": [],   # TODO: pull from engagement register in RDS
            "kpi_vs_target":   {},    # TODO: compare baseline dict against kpi_targets
            "health_narrative": health_narrative,
            "generated_at":    datetime.utcnow().isoformat(),
        }

    def _tool_assemble_deliverable_package(self, engagement_id: str, state: dict) -> str:
        """Collect all S3 exhibit keys and zip into a client deliverable package."""
        # TODO: boto3 S3 select + zip assembly
        exhibit_keys = state.get("dashboard_visualization", {}).get("exhibit_s3_keys", [])
        logger.debug("[EngagementReporting] Packaging %d exhibits", len(exhibit_keys))
        bucket = "agt16-deliverables"
        return f"s3://{bucket}/{engagement_id}/packages/deliverable_package_{datetime.utcnow().date()}.zip"

    def _tool_benefit_realization(self, engagement_id: str, state: dict) -> dict:
        """Compare realized benefits to approved business case projections."""
        # TODO: query RDS for business case NPV; compare to actuals from Gold layer
        return {
            "engagement_id":          engagement_id,
            "realized_savings":       0.0,
            "business_case_npv":      0.0,
            "vs_business_case_pct":   0.0,
            "generated_at":           datetime.utcnow().isoformat(),
        }

    def _tool_upload_report(self, engagement_id: str, report: dict) -> str:
        """Serialize report to PDF/JSON and upload to S3."""
        # TODO: from ..tools.s3 import S3DeliverableClient
        #       return S3DeliverableClient().upload_report(engagement_id, report)
        bucket = "agt16-deliverables"
        week   = datetime.utcnow().isocalendar().week
        return f"s3://{bucket}/{engagement_id}/reports/weekly_status_w{week}.json"


# ---------------------------------------------------------------------------
# CM-09 — Maturity Assessment Module
# ---------------------------------------------------------------------------

class MaturityAssessmentAgent(BaseAgent):
    """
    Processes owned (FR-11):
      - Supply chain capability maturity assessment across 6 domains
      - Maturity scorecard with gap-to-best-practice analysis
      - Prioritized improvement roadmap

    Preconditions: triggered by context flag; can run after baseline.
    """

    name = AgentName.MATURITY_ASSESSMENT

    DOMAINS = [
        "planning", "procurement", "manufacturing",
        "logistics", "technology", "risk_management",
    ]

    def can_run(self, state: dict) -> bool:
        # Runs when explicitly requested OR after baseline is complete
        return (
            state.get("context", {}).get("maturity_assessment_required", False)
            or bool(state.get("performance_baseline", {}).get("kpi_baseline"))
        )

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        logger.info("[MaturityAssessment] Running maturity assessment for %s", engagement_id)

        # ── Tool 1: score each capability domain ──────────────────────────
        maturity_scores = self._tool_score_domains(
            state.get("context", {}).get("assessment_responses", {}),
            state.get("performance_baseline", {}).get("kpi_baseline", {}),
        )

        # ── Tool 2: generate improvement roadmap via LLM ──────────────────
        roadmap = self._tool_generate_roadmap(maturity_scores)

        # ── Tool 3: render scorecard and upload to S3 ─────────────────────
        scorecard_key = self._tool_render_scorecard(engagement_id, maturity_scores, roadmap)

        return {
            "maturity_assessment": {
                "maturity_scores":    maturity_scores,
                "scorecard_s3_key":   scorecard_key,
                "improvement_roadmap": roadmap,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_score_domains(self, assessment_responses: dict, kpi_baseline: dict) -> dict:
        """Score each of the 6 capability domains on a 1-5 maturity scale."""
        # TODO: apply scoring rubric from RDS methodology table
        #       cross-reference with KPI baseline for evidence-based scoring
        logger.debug("[MaturityAssessment] Scoring %d domains", len(self.DOMAINS))
        scores = {}
        for domain in self.DOMAINS:
            scores[domain] = {
                "score":              0,    # 1-5
                "best_practice_score": 5,
                "gap":                5,
                "evidence":           [],
            }
        return scores

    def _tool_generate_roadmap(self, maturity_scores: dict) -> list[dict]:
        """Use LLM to generate prioritized improvement initiatives per domain."""
        roadmap_raw = extract_json(
            system=(
                "You are a supply chain transformation expert. Given a maturity scorecard "
                "showing scores for 6 capability domains, generate a prioritized improvement "
                "roadmap. Return a JSON array where each element has: "
                "domain, initiative, priority (1-5), timeline_months, expected_maturity_gain."
            ),
            user=f"Maturity scores: {maturity_scores}",
        )
        return roadmap_raw if isinstance(roadmap_raw, list) else []

    def _tool_render_scorecard(
        self, engagement_id: str, scores: dict, roadmap: list
    ) -> str:
        """Render scorecard as PDF and upload to S3."""
        # TODO: matplotlib radar chart + ReportLab PDF assembly + S3 upload
        logger.debug("[MaturityAssessment] Rendering scorecard PDF")
        bucket = "agt16-deliverables"
        return f"s3://{bucket}/{engagement_id}/maturity/scorecard.pdf"


# ---------------------------------------------------------------------------
# CM-10 — Firm Knowledge Base & Win/Loss Analytics
# ---------------------------------------------------------------------------

class FirmKnowledgeAgent(BaseAgent):
    """
    Processes owned (FR-16, FR-18):
      - Maintain firm knowledge base: client data, benchmarks, models, methodology
      - Index artifacts into RDS pgvector for semantic search
      - Win/loss analytics: proposal outcomes, pricing, competitive positioning

    Preconditions: runs post-engagement or on-demand in firm mode.
    """

    name = AgentName.FIRM_KNOWLEDGE

    def can_run(self, state: dict) -> bool:
        # Runs whenever a completed engagement is ready for archival
        # OR when win/loss analytics are explicitly requested
        return (
            bool(state.get("engagement_reporting", {}).get("report_s3_key"))
            or state.get("mode") == "firm"
            or state.get("context", {}).get("firm_knowledge_update", False)
        )

    def _execute(self, state: dict) -> dict:
        engagement_id = state.get("engagement_id", "adhoc")
        logger.info("[FirmKnowledge] Updating knowledge base for %s", engagement_id)

        # ── Tool 1: collect all S3 artifact keys from this workflow ───────
        artifact_keys = self._tool_collect_artifacts(state)

        # ── Tool 2: anonymize and index artifacts into pgvector ───────────
        indexed_keys = self._tool_index_artifacts(artifact_keys)

        # ── Tool 3: refresh benchmark library from external sources ───────
        benchmark_status = self._tool_refresh_benchmarks()

        # ── Tool 4: win/loss analytics (firm mode) ─────────────────────────
        win_loss = {}
        if state.get("mode") == "firm" or state.get("context", {}).get("win_loss_update"):
            win_loss = self._tool_win_loss_analytics()

        return {
            "firm_knowledge": {
                "knowledge_base_updated":  True,
                "artifacts_indexed":       indexed_keys,
                "benchmark_refresh_status": benchmark_status,
                "win_loss_summary":        win_loss,
            }
        }

    # ── Tool stubs ────────────────────────────────────────────────────────

    def _tool_collect_artifacts(self, state: dict) -> list[str]:
        """Gather all S3 artifact keys produced during this workflow."""
        keys = []
        keys += state.get("dashboard_visualization", {}).get("exhibit_s3_keys", [])
        pres = state.get("dashboard_visualization", {}).get("presentation_s3_key")
        if pres:
            keys.append(pres)
        report = state.get("engagement_reporting", {}).get("report_s3_key")
        if report:
            keys.append(report)
        scorecard = state.get("maturity_assessment", {}).get("scorecard_s3_key")
        if scorecard:
            keys.append(scorecard)
        digest = state.get("market_intelligence", {}).get("digest_s3_key")
        if digest:
            keys.append(digest)
        return keys

    def _tool_index_artifacts(self, artifact_keys: list) -> list[str]:
        """Download artifacts, embed text, and upsert into RDS pgvector."""
        # TODO: from ..tools.rds import VectorStoreClient
        #       for key in artifact_keys:
        #           text = S3Client().read_text(key)
        #           embedding = embed(text)
        #           VectorStoreClient().upsert(key, embedding, metadata)
        logger.debug("[FirmKnowledge] Indexing %d artifacts into pgvector", len(artifact_keys))
        return artifact_keys

    def _tool_refresh_benchmarks(self) -> dict:
        """Pull latest Gartner / Hackett benchmark data into RDS benchmark library."""
        # TODO: from ..tools.rds import BenchmarkClient
        #       BenchmarkClient().refresh_from_api()
        logger.debug("[FirmKnowledge] Refreshing benchmark library")
        return {
            "status":       "ok",
            "sources":      ["gartner", "hackett"],
            "refreshed_at": datetime.utcnow().isoformat(),
        }

    def _tool_win_loss_analytics(self) -> dict:
        """Aggregate proposal win/loss outcomes from RDS; compute key metrics."""
        # TODO: from ..tools.rds import ProposalClient
        #       records = ProposalClient().get_all_outcomes()
        #       compute win_rate, avg_deal_size, top loss reasons via LLM
        logger.debug("[FirmKnowledge] Running win/loss analytics")
        return {
            "total_proposals":    0,
            "win_rate":           0.0,
            "avg_deal_size":      0.0,
            "top_win_themes":     [],
            "top_loss_reasons":   [],
            "computed_at":        datetime.utcnow().isoformat(),
        }
