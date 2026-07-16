from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from otif_risk.fusion import (
    WEIGHT_GRID,
    FusionBundle,
    alert_rate,
    expected_calibration_error,
    fuse_scores,
    lift_at_capacity,
    select_fusion_weight,
)


def test_fuse_scores_applies_fixed_transparent_weights_by_order_id():
    risk = pd.DataFrame(
        {"order_id": ["a", "b"], "risk_model_score": [0.8, 0.2]}
    )
    bayesian = pd.DataFrame(
        {"order_id": ["b", "a"], "bbn_risk_score": [0.6, 0.4]}
    )

    fused = fuse_scores(risk, bayesian)

    assert fused["order_id"].tolist() == ["a", "b"]
    assert fused["fused_risk_score"].tolist() == pytest.approx(
        [0.7 * 0.8 + 0.3 * 0.4, 0.7 * 0.2 + 0.3 * 0.6]
    )
    assert set(fused["endpoint"]) == {"OTIF_MISS"}


def test_fuse_scores_requires_matching_orders():
    risk = pd.DataFrame({"order_id": ["a"], "risk_model_score": [0.8]})
    bayesian = pd.DataFrame({"order_id": ["b"], "bbn_risk_score": [0.4]})

    with pytest.raises(ValueError, match="same order_id"):
        fuse_scores(risk, bayesian)


def test_fuse_scores_rejects_invalid_probability():
    risk = pd.DataFrame({"order_id": ["a"], "risk_model_score": [1.2]})
    bayesian = pd.DataFrame({"order_id": ["a"], "bbn_risk_score": [0.4]})

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuse_scores(risk, bayesian)


def test_fusion_bundle_requires_same_otif_endpoint():
    risk_bundle = SimpleNamespace(endpoint="OTIF_MISS")
    other_endpoint_bundle = SimpleNamespace(endpoint="LATE_ONLY")

    with pytest.raises(ValueError, match="same endpoint"):
        FusionBundle(risk_bundle, other_endpoint_bundle)


def _synthetic_scores(seed: int, n: int = 400, xgb_quality: float = 0.7, bbn_quality: float = 0.2):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.2).astype(int)
    xgb = np.clip(labels * xgb_quality + rng.normal(0, 0.12, n) + 0.12, 0, 1)
    bbn = np.clip(labels * bbn_quality + rng.normal(0, 0.2, n) + 0.2, 0, 1)
    return labels, xgb, bbn


def test_select_fusion_weight_prefers_the_clearly_better_standalone_model():
    labels, xgb, bbn = _synthetic_scores(0, xgb_quality=0.75, bbn_quality=0.05)

    selection = select_fusion_weight(labels, xgb, bbn)

    assert selection.chosen_weight == 1.0
    assert selection.chosen_label == "xgb_only"
    assert set(selection.comparison["xgb_weight"]) == set(WEIGHT_GRID)
    assert "brier" in selection.comparison.columns
    assert "capacity_recall" in selection.comparison.columns
    assert selection.comparison["threshold"].notna().sum() == 1
    assert "No stacking model" in selection.rationale


def test_select_fusion_weight_guardrail_uses_capacity_recall_not_per_candidate_threshold():
    """A miscalibrated candidate with an inflated per-threshold recall must not
    win purely because its own recall-floor search fell back to an extreme
    threshold; the guardrail compares recall at a fixed capacity cutoff."""
    labels, xgb, bbn = _synthetic_scores(1, xgb_quality=0.8, bbn_quality=0.0)

    selection = select_fusion_weight(labels, xgb, bbn, capacity_fraction=0.2)

    # With bbn carrying no real signal, xgb-leaning weights must dominate.
    assert selection.chosen_weight >= 0.5


def test_select_fusion_weight_never_fits_a_stacking_model_and_covers_full_grid():
    labels, xgb, bbn = _synthetic_scores(2)
    selection = select_fusion_weight(labels, xgb, bbn)
    assert len(selection.comparison) == len(WEIGHT_GRID)
    assert selection.comparison["xgb_weight"].tolist() == sorted(selection.comparison["xgb_weight"])


def test_expected_calibration_error_is_zero_for_a_perfectly_calibrated_score():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    probabilities = np.full(10, 0.6)
    assert expected_calibration_error(labels, probabilities) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_penalizes_overconfidence():
    labels = np.array([0] * 8 + [1] * 2)
    overconfident = np.full(10, 0.95)
    assert expected_calibration_error(labels, overconfident) > 0.5


def test_lift_at_capacity_exceeds_one_for_a_good_ranker():
    rng = np.random.default_rng(3)
    n = 500
    labels = (rng.random(n) < 0.2).astype(int)
    scores = np.clip(labels * 0.8 + rng.normal(0, 0.1, n), 0, 1)
    lift = lift_at_capacity(labels, scores, 0.2)
    assert lift > 1.0


def test_alert_rate_matches_flagged_fraction():
    scores = np.array([0.1, 0.9, 0.4, 0.8, 0.2])
    assert alert_rate(scores, 0.5) == pytest.approx(0.4)
