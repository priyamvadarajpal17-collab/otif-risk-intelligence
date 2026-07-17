from __future__ import annotations

import numpy as np
import pandas as pd

from otif_risk.drift import evaluate_drift, population_stability_index


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 2000)
    current = rng.normal(0, 1, 2000)
    psi = population_stability_index(baseline, current)
    assert psi < 0.05


def test_psi_is_large_for_a_clearly_shifted_distribution():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 2000)
    current = rng.normal(4, 1, 2000)
    psi = population_stability_index(baseline, current)
    assert psi > 1.0


def test_psi_handles_empty_or_tiny_inputs_gracefully():
    assert population_stability_index(np.array([]), np.array([1, 2, 3])) == 0.0
    assert population_stability_index(np.array([1, 2, 3]), np.array([])) == 0.0


def _feature_frame(rng, n, mean_shift=0.0):
    return pd.DataFrame(
        {
            "vendor_rolling_fault_rate_30d": rng.normal(0.1 + mean_shift, 0.05, n).clip(0, 1),
            "active_leading_signal_count": rng.integers(0, 3, n),
            "days_to_promised_delivery": rng.normal(3 + mean_shift * 5, 1, n),
        }
    )


def test_evaluate_drift_does_not_trigger_for_stable_inputs():
    rng = np.random.default_rng(1)
    baseline_features = _feature_frame(rng, 500)
    current_features = _feature_frame(rng, 200)
    report = evaluate_drift(
        baseline_features=baseline_features,
        current_features=current_features,
        baseline_scores=rng.uniform(0, 0.3, 500),
        current_scores=rng.uniform(0, 0.3, 200),
        baseline_missingness=0.3,
        current_missingness=0.31,
        baseline_otif_rate=0.18,
        current_otif_rate=0.19,
        recent_otif_observations=100,
    )
    assert report.triggered is False
    assert report.reasons == []


def test_evaluate_drift_triggers_on_a_large_recent_otif_rate_shift():
    rng = np.random.default_rng(2)
    baseline_features = _feature_frame(rng, 500)
    current_features = _feature_frame(rng, 200)
    report = evaluate_drift(
        baseline_features=baseline_features,
        current_features=current_features,
        baseline_scores=rng.uniform(0, 0.3, 500),
        current_scores=rng.uniform(0, 0.3, 200),
        baseline_missingness=0.3,
        current_missingness=0.3,
        baseline_otif_rate=0.15,
        current_otif_rate=0.45,
        recent_otif_observations=100,
    )
    assert report.triggered is True
    assert any("recent OTIF rate shift" in reason for reason in report.reasons)


def test_evaluate_drift_report_serializes_to_dict():
    rng = np.random.default_rng(3)
    baseline_features = _feature_frame(rng, 100)
    current_features = _feature_frame(rng, 50)
    report = evaluate_drift(
        baseline_features=baseline_features,
        current_features=current_features,
        baseline_scores=rng.uniform(0, 1, 100),
        current_scores=rng.uniform(0, 1, 50),
        baseline_missingness=0.2,
        current_missingness=0.2,
        baseline_otif_rate=0.2,
        current_otif_rate=0.2,
        recent_otif_observations=100,
    )
    payload = report.to_dict()
    assert set(payload) == {
        "feature_psi",
        "score_mean_shift",
        "missingness_rate_shift",
        "recent_otif_rate_shift",
        "recent_otif_observations",
        "otif_rate_trigger_eligible",
        "triggered",
        "reasons",
    }


def test_recent_otif_rate_does_not_trigger_on_tiny_sample() -> None:
    rng = np.random.default_rng(4)
    report = evaluate_drift(
        baseline_features=_feature_frame(rng, 500),
        current_features=_feature_frame(rng, 200),
        baseline_scores=np.full(500, 0.2),
        current_scores=np.full(200, 0.2),
        baseline_missingness=0.3,
        current_missingness=0.3,
        baseline_otif_rate=0.15,
        current_otif_rate=0.60,
        recent_otif_observations=5,
    )
    assert report.otif_rate_trigger_eligible is False
    assert report.triggered is False
