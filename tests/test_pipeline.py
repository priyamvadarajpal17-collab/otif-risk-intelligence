from __future__ import annotations

import json

import pandas as pd

from otif_risk.contracts import PrototypeConfig
from otif_risk.pipeline import _run_directory, bayesian_training_history, run_pipeline


def test_run_pipeline_writes_complete_artifacts(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=7, n_orders=300, output_dir=tmp_path / "artifacts")
    )

    run_directories = list((tmp_path / "artifacts").glob("run-*"))
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "models" / "xgboost_risk.joblib").is_file()
    assert (run_dir / "models" / "bayesian_network.joblib").is_file()
    assert (run_dir / "data" / "scored_orders.csv").is_file()
    assert (run_dir / "data" / "scored_order_lines.csv").is_file()
    assert (run_dir / "data" / "root_causes.csv").is_file()
    assert (run_dir / "data" / "vendor_rollup.csv").is_file()
    assert (run_dir / "data" / "order_type_rollup.csv").is_file()
    assert (run_dir / "data" / "sku_rollup.csv").is_file()
    assert (run_dir / "data" / "simulator_truth.csv").is_file()
    assert (run_dir / "data" / "line_truth.csv").is_file()
    assert (run_dir / "data" / "shocks.csv").is_file()
    assert (run_dir / "data" / "fusion_comparison.csv").is_file()
    assert (run_dir / "planner_feedback.csv").is_file()
    weight_grid = {round(value * 0.1, 1) for value in range(11)}
    assert report["architecture"]["fusion_chosen_weight"] in weight_grid
    assert report["architecture"]["risk_model"] == "xgboost"
    assert 0 <= report["test_metrics"]["pr_auc"] <= 1
    assert 0.10 <= report["data"]["otif_miss_rate"] <= 0.30


def test_fused_threshold_drives_decisions_and_scored_orders_have_sku_evidence(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=11, n_orders=400, output_dir=tmp_path / "artifacts")
    )

    fused = report["model_scores"]["fused"]
    xgb = report["model_scores"]["xgb"]
    bbn = report["model_scores"]["bbn"]
    assert report["threshold"] == fused["threshold"]
    assert report["test_metrics"] == fused["test_metrics"]
    expected_metric_keys = {
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "brier",
        "threshold",
        "flagged_orders",
    }
    for space in (xgb, bbn):
        assert expected_metric_keys <= set(space["test_metrics"])
    assert expected_metric_keys <= set(fused["test_metrics"])

    scored_orders = pd.read_csv(
        next((tmp_path / "artifacts").glob("run-*/data/scored_orders.csv"))
    )
    assert (scored_orders["decision_status"] != "MONITOR").any()
    assert "affected_skus_json" in scored_orders.columns
    assert "affected_sku_count" in scored_orders.columns
    assert "contested_with" in scored_orders.columns


def test_line_evidence_and_cause_fidelity_are_reported(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=13, n_orders=300, output_dir=tmp_path / "artifacts")
    )

    baseline = report["model_scores"]["prevalence_baseline"]
    assert "prevalence" in baseline
    fidelity = report["cause_fidelity"]
    assert 0 <= fidelity["overall_agreement"] <= 1
    assert fidelity["scope"] == "held-out OTIF misses only"
    assert "UNKNOWN" in fidelity["per_cause_recall"]

    line_evidence = report["line_evidence"]
    assert "targeted_evidence" in line_evidence
    assert "naive_all_lines_baseline" in line_evidence
    assert line_evidence["naive_all_lines_baseline"]["recall"] == 1.0
    assert 0 <= fidelity["majority_cause_baseline"] <= 1


def test_canonical_contention_pair_is_actionable_and_contested(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=42, n_orders=400, output_dir=tmp_path / "artifacts")
    )
    scored_path = (
        tmp_path / "artifacts" / report["provenance"]["run_directory"]
        / "data" / "scored_orders.csv"
    )
    scored = pd.read_csv(scored_path).set_index("order_id")

    pair = scored.loc[["O000397", "O000398"]]
    assert set(pair["decision_status"]) == {"RECOMMENDED", "CONTESTED"}
    assert set(pair["resource_id"]) == {"V001"}


def test_bayesian_inference_mode_is_recorded(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=17, n_orders=300, output_dir=tmp_path / "artifacts")
    )

    assert report["architecture"]["bayesian_inference_mode"] in {
        "pgmpy_exact",
        "brute_force_exact",
    }
    assert len(report["architecture"]["bayesian_chain_edges"]) == 11
    assert report["architecture"]["mechanism_nodes"] == ["IN_FULL_FAILURE", "LATE_DELIVERY"]


def test_provenance_and_schema_metadata_are_persisted(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=19, n_orders=300, output_dir=tmp_path / "artifacts")
    )

    provenance = report["provenance"]
    assert provenance["artifact_schema_version"]
    assert provenance["package_version"]
    assert provenance["generated_at_utc"]
    assert provenance["run_directory"]
    assert "scored_orders_columns" in report["schema"]
    assert "decision_status" in report["schema"]["scored_orders_columns"]


def test_rerunning_identical_config_does_not_overwrite_prior_run(tmp_path):
    config = PrototypeConfig(seed=23, n_orders=300, output_dir=tmp_path / "artifacts")

    first_report = run_pipeline(config)
    second_report = run_pipeline(config)

    first_dir = first_report["provenance"]["run_directory"]
    second_dir = second_report["provenance"]["run_directory"]
    assert first_dir != second_dir
    assert (tmp_path / "artifacts" / first_dir / "metrics.json").is_file()
    assert (tmp_path / "artifacts" / second_dir / "metrics.json").is_file()
    with (tmp_path / "artifacts" / first_dir / "metrics.json").open(encoding="utf-8") as handle:
        first_metrics = json.load(handle)
    assert first_metrics["provenance"]["run_directory"] == first_dir


def test_run_directory_appends_distinguishing_suffix_without_deleting(tmp_path):
    config = PrototypeConfig(seed=29, n_orders=300, output_dir=tmp_path / "artifacts")
    first = _run_directory(config)
    first.mkdir(parents=True)
    second = _run_directory(config)

    assert second != first
    assert second.name.startswith(first.name + "-")
    assert first.exists()


def test_bayesian_training_history_excludes_validation_and_test_order_ids():
    causes = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "stage_ORDER_CAPTURE": [1, 0, 1, 0],
            "stage_VENDOR_FAILURE": [0, 1, 0, 1],
            "stage_INVENTORY_SHORTAGE": [0, 0, 0, 0],
            "stage_DC_CAPACITY": [0, 0, 0, 0],
            "stage_WAREHOUSE_OPS": [0, 0, 0, 0],
            "stage_TRANSPORT": [0, 0, 0, 0],
            "stage_CUSTOMER_DELIVERY": [0, 0, 0, 0],
        }
    )
    outcomes = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "otif_miss": [1, 1, 0, 0],
            "on_time": [0, 0, 1, 1],
            "in_full": [1, 0, 1, 1],
        }
    )

    history = bayesian_training_history(causes, outcomes, {"a", "b"})

    assert set(history["order_id"]) == {"a", "b"}
    assert len(history) == 2
    assert {"on_time", "in_full", "otif_miss"} <= set(history.columns)


def test_scored_orders_carry_the_new_causal_intelligence_columns(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=31, n_orders=300, output_dir=tmp_path / "artifacts")
    )
    scored_orders = pd.read_csv(
        next((tmp_path / "artifacts").glob("run-*/data/scored_orders.csv"))
    )
    expected_columns = {
        "causal_attribution_json",
        "intervention_scenarios_json",
        "causal_confidence",
        "evidence_coverage",
        "late_delivery_probability",
        "in_full_failure_probability",
    }
    assert expected_columns <= set(scored_orders.columns)
    assert set(scored_orders["causal_confidence"]) <= {"LOW", "MEDIUM", "HIGH"}
    assert scored_orders["evidence_coverage"].between(0, 1).all()
    assert scored_orders["late_delivery_probability"].between(0, 1).all()
    assert scored_orders["in_full_failure_probability"].between(0, 1).all()
    for value in scored_orders["causal_attribution_json"]:
        assert isinstance(json.loads(value), list)
    for value in scored_orders["intervention_scenarios_json"]:
        assert isinstance(json.loads(value), list)
    assert report["provenance"]["artifact_schema_version"] == "3.0"


def test_metrics_json_reports_mechanism_confidence_and_consistency_diagnostics(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=37, n_orders=300, output_dir=tmp_path / "artifacts")
    )

    mechanism = report["mechanism_metrics"]
    assert {"late_delivery", "in_full_failure"} <= set(mechanism)
    for mechanism_name in ("late_delivery", "in_full_failure"):
        assert 0 <= mechanism[mechanism_name]["brier"] <= 1

    confidence = report["causal_confidence_diagnostics"]
    assert set(confidence["confidence_band_counts"]) == {"LOW", "MEDIUM", "HIGH"}
    assert 0 <= confidence["low_confidence_rate"] <= 1
    assert confidence["total_orders"] == len(
        pd.read_csv(next((tmp_path / "artifacts").glob("run-*/data/scored_orders.csv")))
    )

    consistency = report["causal_consistency"]
    assert "top_attribution_vs_rule_cause" in consistency
    assert "top_intervention_vs_simulator_responsive_cause" in consistency
    assert "validated causal effect" in consistency["note"]
