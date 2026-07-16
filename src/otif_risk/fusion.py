"""Transparent fixed-weight fusion for the shared OTIF-miss endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from otif_risk.bayesian import BayesianBundle
from otif_risk.model import ENDPOINT, RiskBundle

RISK_MODEL_WEIGHT = 0.7
BBN_WEIGHT = 0.3


@dataclass
class FusionBundle:
    risk_bundle: RiskBundle
    bayesian_bundle: BayesianBundle

    def __post_init__(self) -> None:
        _validate_endpoints(self.risk_bundle.endpoint, self.bayesian_bundle.endpoint)

    @property
    def endpoint(self) -> str:
        return self.risk_bundle.endpoint

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        return fuse_scores(
            self.risk_bundle.score(frame),
            self.bayesian_bundle.score(frame),
            risk_endpoint=self.risk_bundle.endpoint,
            bbn_endpoint=self.bayesian_bundle.endpoint,
        )


def fuse_scores(
    risk_scores: pd.DataFrame,
    bbn_scores: pd.DataFrame,
    *,
    risk_endpoint: str = ENDPOINT,
    bbn_endpoint: str = ENDPOINT,
) -> pd.DataFrame:
    """Combine matched order risks as 70% model plus 30% Bayesian score."""
    _validate_endpoints(risk_endpoint, bbn_endpoint)
    _validate_score_frame(risk_scores, "risk_model_score", "risk_scores")
    _validate_score_frame(bbn_scores, "bbn_risk_score", "bbn_scores")
    if set(risk_scores["order_id"]) != set(bbn_scores["order_id"]):
        raise ValueError("risk and Bayesian scores must contain the same order_id values")

    merged = risk_scores[["order_id", "risk_model_score"]].merge(
        bbn_scores[["order_id", "bbn_risk_score"]],
        on="order_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    model_values = _validated_probabilities(merged["risk_model_score"], "risk_model_score")
    bbn_values = _validated_probabilities(merged["bbn_risk_score"], "bbn_risk_score")
    merged["fused_risk_score"] = RISK_MODEL_WEIGHT * model_values + BBN_WEIGHT * bbn_values
    merged["endpoint"] = risk_endpoint
    return merged[
        ["order_id", "risk_model_score", "bbn_risk_score", "fused_risk_score", "endpoint"]
    ]


def fuse_risk_scores(
    risk_scores: pd.DataFrame,
    bbn_scores: pd.DataFrame,
    *,
    risk_endpoint: str = ENDPOINT,
    bbn_endpoint: str = ENDPOINT,
) -> pd.DataFrame:
    """Compatibility alias for score fusion."""
    return fuse_scores(
        risk_scores,
        bbn_scores,
        risk_endpoint=risk_endpoint,
        bbn_endpoint=bbn_endpoint,
    )


def _validate_endpoints(risk_endpoint: str, bbn_endpoint: str) -> None:
    if risk_endpoint != bbn_endpoint:
        raise ValueError("risk model and Bayesian network must predict the same endpoint")
    if risk_endpoint != ENDPOINT:
        raise ValueError(f"fusion supports only the {ENDPOINT} endpoint")


def _validate_score_frame(frame: pd.DataFrame, score_column: str, name: str) -> None:
    missing = sorted({"order_id", score_column} - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame["order_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate order_id values")


def _validated_probabilities(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return values
