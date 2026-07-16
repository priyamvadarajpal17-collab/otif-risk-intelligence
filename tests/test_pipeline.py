from __future__ import annotations

import json

import pandas as pd

from otif_risk.contracts import PrototypeConfig
from otif_risk.pipeline import _bayesian_training_history, _run_directory, run_pipeline


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
    assert (run_dir / "data" / "root_causes.csv").is_file()
    assert (run_dir / "data" / "vendor_rollup.csv").is_file()
    assert (run_dir / "data" / "order_type_rollup.csv").is_file()
    assert (run_dir / "data" / "sku_rollup.csv").is_file()
    assert (run_dir / "planner_feedback.csv").is_file()
    assert report["architecture"]["fusion"].startswith("0.70")
    assert report["architecture"]["risk_model"] == "xgboost"
    assert report["threshold_strategy"] == "recall_floor"
    assert 0 <= report["test_metrics"]["pr_auc"] <= 1
    assert 0.10 <= report["data"]["otif_miss_rate"] <= 0.30


def test_fused_threshold_drives_decisions_and_is_not_the_raw_xgb_threshold(tmp_path):
    """Threshold selection and application use the fused score space."""
    report = run_pipeline(
        PrototypeConfig(seed=11, n_orders=400, output_dir=tmp_path / "artifacts")
    )

    fused = report["model_scores"]["fused"]
    xgb = report["model_scores"]["xgb"]
    bbn = report["model_scores"]["bbn"]
    assert report["threshold"] == fused["threshold"]
    assert report["validation_metrics"] == fused["validation_metrics"]
    assert report["test_metrics"] == fused["test_metrics"]
    # Each score space is evaluated/thresholded independently with the same metric set.
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
    for space in (xgb, bbn, fused):
        assert expected_metric_keys <= set(space["test_metrics"])
        assert expected_metric_keys <= set(space["validation_metrics"])

    scored_orders = pd.read_csv(
        next((tmp_path / "artifacts").glob("run-*/data/scored_orders.csv"))
    )
    # The pipeline's fused threshold must produce an actionable work queue.
    assert (scored_orders["decision_status"] != "MONITOR").any()


def test_prevalence_baseline_and_cause_fidelity_are_reported(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=13, n_orders=300, output_dir=tmp_path / "artifacts")
    )

    baseline = report["model_scores"]["prevalence_baseline"]
    assert "prevalence" in baseline
    assert "note" in baseline
    fidelity = report["cause_fidelity"]
    assert 0 <= fidelity["overall_agreement"] <= 1
    assert fidelity["scope"] == "held-out OTIF misses only"
    assert fidelity["evaluated_orders"] == sum(
        report["model_scores"]["fused"]["test_metrics"]["confusion_matrix"][1]
    )
    assert "UNKNOWN" in fidelity["per_cause_recall"]


def test_bayesian_inference_mode_is_recorded(tmp_path):
    report = run_pipeline(
        PrototypeConfig(seed=17, n_orders=300, output_dir=tmp_path / "artifacts")
    )

    assert report["architecture"]["bayesian_inference_mode"] in {
        "pgmpy_exact",
        "empirical_table",
    }


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
    """Canonical reruns must be distinguishable and non-destructive."""
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
    """Bayesian fitting must only see the training split's resolved history."""
    causes = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "cause_ORDER_CAPTURE": [1, 0, 1, 0],
            "cause_VENDOR_FAILURE": [0, 1, 0, 1],
            "cause_INVENTORY_SHORTAGE": [0, 0, 0, 0],
            "cause_DC_CAPACITY": [0, 0, 0, 0],
            "cause_WAREHOUSE_OPS": [0, 0, 0, 0],
            "cause_TRANSPORT": [0, 0, 0, 0],
            "cause_CUSTOMER_DELIVERY": [0, 0, 0, 0],
        }
    )
    outcomes = pd.DataFrame({"order_id": ["a", "b", "c", "d"], "otif_miss": [1, 1, 0, 0]})

    history = _bayesian_training_history(causes, outcomes, {"a", "b"})

    assert set(history["order_id"]) == {"a", "b"}
    assert len(history) == 2
