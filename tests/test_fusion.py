from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from otif_pdf.fusion import FusionBundle, fuse_scores


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
