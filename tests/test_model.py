from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

import otif_pdf.explain as explain_module
from otif_pdf.model import (
    capacity_threshold,
    select_threshold,
    train_risk_model,
)


def _split(seed: int, size: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inventory = rng.integers(0, 2, size)
    transport = rng.integers(0, 2, size)
    backlog = rng.normal(0, 1, size)
    region = rng.choice(["east", "west", "central"], size)
    latent = 1.4 * inventory + 1.1 * transport + 0.6 * backlog + (region == "west")
    target = (latent + rng.normal(0, 0.8, size) > 1.6).astype(int)
    return pd.DataFrame(
        {
            "order_id": [f"{seed}-{index}" for index in range(size)],
            "prediction_timestamp": pd.date_range("2025-01-01", periods=size, freq="h"),
            "leading_signal_INVENTORY_SHORTAGE": inventory,
            "leading_signal_TRANSPORT": transport,
            "backlog_zscore": backlog,
            "region": region,
            "otif_miss": target,
        }
    )


def test_train_risk_model_scores_metrics_and_round_trips(tmp_path):
    train, validation, test = _split(1, 80), _split(2), _split(3)

    result = train_risk_model(train, validation, test, planner_capacity_fraction=0.25)
    probabilities = result.bundle.predict_proba(test)
    path = tmp_path / "risk.joblib"
    joblib.dump(result.bundle, path)
    restored = joblib.load(path)

    assert result.bundle.model_kind == "xgboost"
    assert probabilities.shape == (len(test),)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    assert restored.predict_proba(test) == pytest.approx(probabilities)
    assert set(result.test_metrics) == {
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
    assert np.asarray(result.test_metrics["confusion_matrix"]).shape == (2, 2)


def test_train_risk_model_requires_xgboost(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)

    with pytest.raises(RuntimeError, match="XGBoost is required"):
        train_risk_model(_split(4, 64), _split(5), _split(6))


def test_capacity_threshold_selects_last_score_within_capacity():
    probabilities = np.array([0.05, 0.9, 0.4, 0.8, 0.2])

    threshold = capacity_threshold(probabilities, 0.4)

    assert threshold == pytest.approx(0.8)
    assert int((probabilities >= threshold).sum()) == 2


def test_recall_floor_threshold_prefers_higher_cutoff_when_recall_is_met():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    probabilities = np.array([0.95, 0.85, 0.75, 0.70, 0.55, 0.40, 0.20, 0.05])

    selection = select_threshold(
        labels,
        probabilities,
        strategy="recall_floor",
        target_recall=0.66,
        min_precision=0.30,
    )

    assert selection.strategy == "recall_floor"
    assert selection.validation_metrics["recall"] >= 0.66
    assert selection.threshold >= 0.75


def test_explanations_fall_back_to_non_causal_local_associations(monkeypatch):
    train, validation, test = _split(8, 64), _split(9), _split(10)
    bundle = train_risk_model(train, validation, test).bundle

    def incompatible_shap(*args, **kwargs):
        raise RuntimeError("unsupported backend")

    monkeypatch.setattr(explain_module, "_shap_factors", incompatible_shap)
    explained = explain_module.explain_predictions(
        bundle, test.iloc[:3], background=train, top_n=3
    )
    factors = json.loads(explained.loc[0, "top_factors_json"])

    assert list(explained.columns) == ["order_id", "top_factors_json"]
    assert len(factors) == 3
    assert all(item["interpretation"] == "association_not_causation" for item in factors)
    assert all(item["method"] == "local_perturbation_association" for item in factors)
