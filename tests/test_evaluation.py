from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from otif_risk.evaluation import (
    causal_consistency_report,
    confidence_diagnostics,
    mechanism_metrics,
    simulator_responsive_causes,
)


def test_mechanism_metrics_reports_pr_auc_and_brier_for_both_mechanisms():
    rng = np.random.default_rng(0)
    late_truth = rng.integers(0, 2, 200)
    late_probability = np.clip(late_truth * 0.7 + rng.random(200) * 0.3, 0, 1)
    in_full_truth = rng.integers(0, 2, 200)
    in_full_probability = np.clip(in_full_truth * 0.6 + rng.random(200) * 0.4, 0, 1)

    result = mechanism_metrics(late_truth, late_probability, in_full_truth, in_full_probability)

    assert 0 <= result["late_delivery"]["pr_auc"] <= 1
    assert 0 <= result["late_delivery"]["brier"] <= 1
    assert 0 <= result["in_full_failure"]["pr_auc"] <= 1
    assert 0 <= result["in_full_failure"]["brier"] <= 1
    assert result["late_delivery"]["positive_rate"] == pytest.approx(late_truth.mean())


def test_mechanism_metrics_handles_single_class_truth_gracefully():
    result = mechanism_metrics([0, 0, 0], [0.1, 0.2, 0.05], [1, 1, 1], [0.9, 0.8, 0.95])
    assert np.isnan(result["late_delivery"]["pr_auc"])
    assert np.isnan(result["in_full_failure"]["pr_auc"])


def test_simulator_responsive_causes_labels_dominant_stage():
    outcomes = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "on_time": [0, 1, 1, 1],
            "in_full": [1, 0, 1, 1],
        }
    )
    simulator_truth = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "vendor_ready_delay_hours": [2.0, 0.0, 0.0, 0.0],
            "warehouse_delay_hours": [40.0, 0.0, 0.0, 0.0],
            "transit_delay_hours": [1.0, 0.0, 0.0, 0.0],
            "customer_delay_hours": [0.0, 0.0, 0.0, 0.0],
            "unknown_extra_hours": [0.0, 0.0, 0.0, 0.0],
        }
    )

    labels = simulator_responsive_causes(outcomes, simulator_truth)

    assert labels.loc["a"] == "WAREHOUSE_OPS"  # largest delay contributor
    assert labels.loc["b"] == "INVENTORY_SHORTAGE"  # short but on-time
    assert labels.loc["c"] == "ON_TIME"
    assert labels.loc["d"] == "ON_TIME"


def test_simulator_responsive_causes_includes_order_capture_when_orders_supplied():
    """A capture-delay-dominated late order must not be mislabeled as another stage."""
    outcomes = pd.DataFrame(
        {"order_id": ["a", "b"], "on_time": [0, 0], "in_full": [1, 1]}
    )
    simulator_truth = pd.DataFrame(
        {
            "order_id": ["a", "b"],
            "vendor_ready_delay_hours": [2.0, 2.0],
            "warehouse_delay_hours": [1.0, 40.0],
            "transit_delay_hours": [1.0, 1.0],
            "customer_delay_hours": [0.0, 0.0],
            "unknown_extra_hours": [0.0, 0.0],
        }
    )
    orders = pd.DataFrame({"order_id": ["a", "b"], "capture_delay_hours": [50.0, 5.0]})

    without_capture = simulator_responsive_causes(outcomes, simulator_truth)
    with_capture = simulator_responsive_causes(outcomes, simulator_truth, orders)

    # Without capture-delay data, order "a"'s largest recorded delay is a
    # minor 2h vendor delay -- an arbitrary tie-breaking mislabel.
    assert without_capture.loc["a"] == "VENDOR_FAILURE"
    # With it, the genuinely dominant 50h capture delay is correctly surfaced.
    assert with_capture.loc["a"] == "ORDER_CAPTURE"
    assert with_capture.loc["b"] == "WAREHOUSE_OPS"


def test_simulator_responsive_causes_requires_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        simulator_responsive_causes(pd.DataFrame({"order_id": ["a"]}), pd.DataFrame())


def test_causal_consistency_report_computes_agreement_rates():
    comparisons = pd.DataFrame(
        {
            "top_attribution_cause": ["VENDOR_FAILURE", "TRANSPORT", "DC_CAPACITY"],
            "top_intervention_cause": ["VENDOR_FAILURE", "WAREHOUSE_OPS", "DC_CAPACITY"],
            "rule_primary_cause": ["VENDOR_FAILURE", "TRANSPORT", "TRANSPORT"],
            "simulator_responsive_cause": ["VENDOR_FAILURE", "TRANSPORT", "DC_CAPACITY"],
        }
    )

    report = causal_consistency_report(comparisons)

    assert report["evaluated_orders"] == 3
    assert report["top_attribution_vs_rule_cause"] == pytest.approx(2 / 3)
    assert report["top_attribution_vs_simulator_responsive_cause"] == pytest.approx(1.0)
    assert report["top_intervention_vs_rule_cause"] == pytest.approx(1 / 3)
    assert report["top_intervention_vs_simulator_responsive_cause"] == pytest.approx(2 / 3)
    assert "validated causal effect" in report["note"]


def test_causal_consistency_report_requires_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        causal_consistency_report(pd.DataFrame({"top_attribution_cause": ["A"]}))


def test_confidence_diagnostics_summarizes_coverage_and_low_confidence_rate():
    coverage = pd.Series([4 / 7, 5 / 7, 1.0, 1.0])
    confidence = pd.Series(["LOW", "MEDIUM", "HIGH", "HIGH"])

    result = confidence_diagnostics(coverage, confidence)

    assert result["confidence_band_counts"] == {"LOW": 1, "MEDIUM": 1, "HIGH": 2}
    assert result["low_confidence_rate"] == pytest.approx(0.25)
    assert result["total_orders"] == 4
    assert result["evidence_coverage"]["min"] == pytest.approx(4 / 7)
    assert result["evidence_coverage"]["max"] == pytest.approx(1.0)
