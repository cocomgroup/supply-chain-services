# Databricks notebook source
"""
agt16.interface_contract.sub_agents
--------------------------------------
Interface contract documentation for AGT-16 sub-agents.

Mirrors the pattern established in agt_ss.interface_contract.sub_agents.

Each agent's contract specifies:
  - can_run() preconditions (state keys that must be non-empty)
  - _execute() input keys consumed
  - _execute() output keys produced (partial state update)
  - Tool calls made (with TODO markers for production wiring)

This file is for reference and onboarding; it is not imported at runtime.

Wiring guide
------------
To replace a stub tool call with a real implementation:

  1. Identify the _tool_*() method in agents/sub_agents.py
  2. Import the appropriate client from tools/s3/, tools/rds/, or tools/market/
  3. Replace the stub return value with a real client call
  4. Add any required environment variables to infrastructure/task-definition.json

Example (DataIntegrationAgent._tool_ingest_bronze):

  Before (stub):
    return [f"s3://{bucket}/bronze/{engagement_id}/{s['type'].lower()}/"
            for s in source_systems]

  After (real):
    from ..tools.s3.S3DataClient import S3DataClient
    client = S3DataClient()
    paths  = []
    for system in source_systems:
        df   = pull_from_erp(system)          # system-specific connector
        key  = client.bronze_prefix(engagement_id, system["type"]) + "data.parquet"
        path = client.put_parquet(key, df)
        paths.append(path)
    return paths
"""

# ---------------------------------------------------------------------------
# Agent contracts (documentation only)
# ---------------------------------------------------------------------------

CONTRACTS = {

    "CM-01 DataIntegrationAgent": {
        "can_run_requires":  ["goal"],
        "consumes":          ["engagement_id", "context.source_systems"],
        "produces":          ["data_integration.ingestion_summary",
                              "data_integration.data_quality_report",
                              "data_integration.schema_map",
                              "data_integration.s3_paths"],
        "tool_calls":        [
            "_tool_get_source_systems   → tools/rds/RDSClient.EngagementConfigClient.get_source_systems()",
            "_tool_ingest_bronze        → system-specific connectors (ERP, WMS, TMS, Planning, Procurement)",
            "_tool_promote_silver       → ETL quality gate pipeline (completeness, referential integrity)",
            "_tool_promote_gold         → aggregation SQL / Spark job",
            "_tool_build_schema_map     → schema inference + LLM field-matching",
        ],
    },

    "CM-02 PerformanceBaselineAgent": {
        "can_run_requires":  ["data_integration.s3_paths.gold"],
        "consumes":          ["data_integration.s3_paths.gold", "context.peer_group"],
        "produces":          ["performance_baseline.kpi_baseline",
                              "performance_baseline.benchmark_comparison",
                              "performance_baseline.peer_group",
                              "performance_baseline.baseline_completeness_pct",
                              "performance_baseline.gap_narrative"],
        "tool_calls":        [
            "_tool_compute_kpi_baseline → tools/rds/RDSClient.AnalyticsQueryClient.compute_baseline()",
            "_tool_pull_benchmarks      → tools/rds/RDSClient.BenchmarkClient.get_benchmarks()",
            "_tool_compute_gaps         → statistical gap computation (scipy)",
            "reason()                   → LLM narrative on top 3 performance gaps",
        ],
    },

    "CM-03 DashboardVizAgent": {
        "can_run_requires":  ["performance_baseline.kpi_baseline"],
        "consumes":          ["performance_baseline", "cost_analytics", "diagnostics_rca"],
        "produces":          ["dashboard_visualization.dashboard_urls",
                              "dashboard_visualization.exhibit_s3_keys",
                              "dashboard_visualization.presentation_s3_key"],
        "tool_calls":        [
            "_tool_render_dashboards    → AWS QuickSight API (publish dataset + dashboard)",
            "_tool_export_exhibits      → matplotlib/plotly render → tools/s3/S3DataClient.S3DeliverableClient.upload_bytes()",
            "_tool_build_board_deck    → python-pptx template fill → S3DeliverableClient.upload_bytes()",
        ],
        "checkpoint_gate":   "board_presentation_approval",
    },

    "CM-04 CostAnalyticsAgent": {
        "can_run_requires":  ["performance_baseline.kpi_baseline"],
        "consumes":          ["data_integration.s3_paths.gold",
                              "performance_baseline.benchmark_comparison",
                              "context"],
        "produces":          ["cost_analytics.cost_to_serve_matrix",
                              "cost_analytics.financial_impact_models",
                              "cost_analytics.savings_quantification"],
        "tool_calls":        [
            "_tool_cost_to_serve        → tools/rds/RDSClient.AnalyticsQueryClient.cost_to_serve()",
            "_tool_identify_levers      → extract_json() LLM lever identification from benchmark gaps",
            "_tool_build_financial_models → numpy_financial NPV/IRR/payback calculation",
            "_tool_aggregate_savings    → summation across levers",
        ],
        "checkpoint_gate":   "financial_model_approval",
    },

    "CM-05 DiagnosticsRCAAgent": {
        "can_run_requires":  ["performance_baseline.benchmark_comparison",
                              "performance_baseline.kpi_baseline"],
        "consumes":          ["performance_baseline", "data_integration.s3_paths.gold"],
        "produces":          ["diagnostics_rca.performance_gaps",
                              "diagnostics_rca.root_causes",
                              "diagnostics_rca.hypothesis_test_results"],
        "tool_calls":        [
            "_tool_extract_gaps         → iterate benchmark_comparison.gaps",
            "_tool_detect_anomalies     → scikit-learn IsolationForest on Gold Parquet",
            "_tool_causal_analysis      → SHAP values over RandomForest (scikit-learn)",
            "_tool_hypothesis_tests     → scipy.stats t-test / Mann-Whitney U",
            "_tool_enrich_rca_narrative → extract_json() LLM narrative per root cause",
        ],
    },

    "CM-06 PredictiveAnalyticsAgent": {
        "can_run_requires":  ["data_integration.s3_paths.gold",
                              "performance_baseline.kpi_baseline"],
        "consumes":          ["data_integration.s3_paths.gold",
                              "market_intelligence.commodity_prices",
                              "context.supplier_nodes"],
        "produces":          ["predictive_analytics.demand_forecast",
                              "predictive_analytics.risk_scores",
                              "predictive_analytics.capacity_forecast",
                              "predictive_analytics.cost_forecast",
                              "predictive_analytics.model_metadata"],
        "tool_calls":        [
            "_tool_demand_forecast      → Prophet / XGBoost on Gold demand history",
            "_tool_supply_risk_scoring  → XGBoost risk model on supplier performance history",
            "_tool_capacity_forecast    → linear regression on utilization + demand forecast",
            "_tool_cost_forecast        → ARIMA / Prophet on commodity index time series",
        ],
    },

    "CM-07 MarketIntelligenceAgent": {
        "can_run_requires":  ["goal"],
        "consumes":          ["context.peer_companies", "context.regulatory_domains"],
        "produces":          ["market_intelligence.commodity_prices",
                              "market_intelligence.carrier_rates",
                              "market_intelligence.labor_cost_index",
                              "market_intelligence.disruption_signals",
                              "market_intelligence.competitive_intel",
                              "market_intelligence.regulatory_trends",
                              "market_intelligence.digest_s3_key"],
        "tool_calls":        [
            "_tool_pull_commodity_prices → tools/market/MarketDataClient.CommodityClient",
            "_tool_pull_carrier_rates    → tools/market/MarketDataClient.CarrierRateClient",
            "_tool_pull_labor_costs      → BLS API",
            "_tool_detect_disruptions   → anomaly detection on price series + web search",
            "_tool_competitive_intel    → web search + SEC EDGAR filings per peer company",
            "_tool_regulatory_scan      → tools/market/MarketDataClient.RegulatoryClient",
            "_tool_generate_digest      → reason() LLM synthesis → S3DeliverableClient.upload_digest()",
        ],
    },

    "CM-08 EngagementReportingAgent": {
        "can_run_requires":  ["data_integration.ingestion_summary",
                              "performance_baseline.kpi_baseline"],
        "consumes":          ["performance_baseline", "kpi_targets",
                              "dashboard_visualization.exhibit_s3_keys",
                              "context.post_implementation"],
        "produces":          ["engagement_reporting.weekly_status_report",
                              "engagement_reporting.deliverable_package_s3_key",
                              "engagement_reporting.benefit_realization_report",
                              "engagement_reporting.report_s3_key"],
        "tool_calls":        [
            "_tool_weekly_status_report  → reason() LLM health narrative + RDS milestone query",
            "_tool_assemble_deliverable_package → S3DeliverableClient.zip_and_upload()",
            "_tool_benefit_realization  → RDS business_cases table + Gold actuals comparison",
            "_tool_upload_report        → S3DeliverableClient.upload_report()",
        ],
        "checkpoint_gate":   "baseline_approval",
    },

    "CM-09 MaturityAssessmentAgent": {
        "can_run_requires":  ["goal"],  # also triggered by context.maturity_assessment_required
        "consumes":          ["context.assessment_responses",
                              "performance_baseline.kpi_baseline"],
        "produces":          ["maturity_assessment.maturity_scores",
                              "maturity_assessment.scorecard_s3_key",
                              "maturity_assessment.improvement_roadmap"],
        "tool_calls":        [
            "_tool_score_domains        → scoring rubric from RDS methodology table",
            "_tool_generate_roadmap    → extract_json() LLM prioritised improvement initiatives",
            "_tool_render_scorecard    → matplotlib radar chart + ReportLab PDF → S3",
        ],
    },

    "CM-10 FirmKnowledgeAgent": {
        "can_run_requires":  ["engagement_reporting.report_s3_key"],  # or mode=firm
        "consumes":          ["dashboard_visualization", "engagement_reporting",
                              "maturity_assessment", "market_intelligence"],
        "produces":          ["firm_knowledge.knowledge_base_updated",
                              "firm_knowledge.artifacts_indexed",
                              "firm_knowledge.benchmark_refresh_status",
                              "firm_knowledge.win_loss_summary"],
        "tool_calls":        [
            "_tool_collect_artifacts    → gather all S3 keys from completed workflow state",
            "_tool_index_artifacts      → tools/rds/RDSClient.VectorStoreClient.upsert() per artifact",
            "_tool_refresh_benchmarks   → tools/rds/RDSClient.BenchmarkClient.refresh_from_api()",
            "_tool_win_loss_analytics   → tools/rds/RDSClient.ProposalClient.get_all_outcomes() + LLM",
        ],
    },
}
