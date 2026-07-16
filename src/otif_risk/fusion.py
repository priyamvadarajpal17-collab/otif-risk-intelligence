"""Evidence-based, explainable score fusion (no stacking model).

Compares XGBoost-only, Bayesian-only, the historical fixed 70/30 blend, and
every other simple convex weight in 10% increments -- all points on one
``fused = w * risk_model_score + (1 - w) * bbn_risk_score`` line -- and
selects the operating weight on **validation only**, using Brier score with a
guardrail that the choice must not materially reduce recall relative to the
best individual candidate. No stacking model is trained on top of the two
scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from otif_risk.bayesian import BayesianBundle
from otif_risk.evaluation import score_space_metrics
from otif_risk.model import (
    ENDPOINT,
    RiskBundle,
    ThresholdStrategy,
    capacity_threshold,
    evaluate_predictions,
)

#: The historical fixed blend, retained as one point on the comparison grid.
RISK_MODEL_WEIGHT = 0.7
BBN_WEIGHT = 0.3
#: Simple, explainable 10% increments for the validation-selected weight search.
WEIGHT_GRID = tuple(round(value, 1) for value in np.arange(0.0, 1.0001, 0.1))
#: A candidate must not fall more than this far below the best individual
#: candidate's recall to be eligible for Brier-based selection.
DEFAULT_RECALL_TOLERANCE = 0.03
#: Treat validation Brier differences within this narrow band as practically
#: equivalent, then prefer the blend with more causal-model contribution.
DEFAULT_BRIER_TOLERANCE = 0.002


@dataclass
class FusionBundle:
    risk_bundle: RiskBundle
    bayesian_bundle: BayesianBundle
    xgb_weight: float = RISK_MODEL_WEIGHT

    def __post_init__(self) -> None:
        _validate_endpoints(self.risk_bundle.endpoint, self.bayesian_bundle.endpoint)
        if not 0 <= self.xgb_weight <= 1:
            raise ValueError("xgb_weight must be in [0, 1]")

    @property
    def endpoint(self) -> str:
        return self.risk_bundle.endpoint

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        return fuse_scores(
            self.risk_bundle.score(frame),
            self.bayesian_bundle.score(frame),
            xgb_weight=self.xgb_weight,
            risk_endpoint=self.risk_bundle.endpoint,
            bbn_endpoint=self.bayesian_bundle.endpoint,
        )


def fuse_scores(
    risk_scores: pd.DataFrame,
    bbn_scores: pd.DataFrame,
    *,
    xgb_weight: float = RISK_MODEL_WEIGHT,
    risk_endpoint: str = ENDPOINT,
    bbn_endpoint: str = ENDPOINT,
) -> pd.DataFrame:
    """Combine matched order risks as a transparent convex blend."""
    _validate_endpoints(risk_endpoint, bbn_endpoint)
    if not 0 <= xgb_weight <= 1:
        raise ValueError("xgb_weight must be in [0, 1]")
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
    merged["fused_risk_score"] = xgb_weight * model_values + (1 - xgb_weight) * bbn_values
    merged["endpoint"] = risk_endpoint
    return merged[
        ["order_id", "risk_model_score", "bbn_risk_score", "fused_risk_score", "endpoint"]
    ]


def fuse_risk_scores(
    risk_scores: pd.DataFrame,
    bbn_scores: pd.DataFrame,
    *,
    xgb_weight: float = RISK_MODEL_WEIGHT,
    risk_endpoint: str = ENDPOINT,
    bbn_endpoint: str = ENDPOINT,
) -> pd.DataFrame:
    """Convenience alias for score fusion."""
    return fuse_scores(
        risk_scores,
        bbn_scores,
        xgb_weight=xgb_weight,
        risk_endpoint=risk_endpoint,
        bbn_endpoint=bbn_endpoint,
    )


@dataclass(frozen=True)
class FusionSelection:
    chosen_weight: float
    chosen_label: str
    comparison: pd.DataFrame
    rationale: str


def select_fusion_weight(
    labels: np.ndarray,
    xgb_scores: np.ndarray,
    bbn_scores: np.ndarray,
    *,
    threshold_strategy: ThresholdStrategy = "recall_floor",
    capacity_fraction: float = 0.15,
    target_recall: float = 0.55,
    min_precision: float = 0.35,
    recall_tolerance: float = DEFAULT_RECALL_TOLERANCE,
    brier_tolerance: float = DEFAULT_BRIER_TOLERANCE,
) -> FusionSelection:
    """Select validation-only fusion under Brier and recall tolerances.

    The guardrail compares recall at a *fixed, comparable* operating point --
    the top ``capacity_fraction`` of orders by score -- rather than each
    weight's own independently re-tuned threshold. Using each candidate's own
    ``threshold_strategy`` threshold for the guardrail is not a fair
    comparison: a poorly calibrated candidate whose recall/precision search
    falls back to an F1-maximizing threshold can show an inflated recall that
    has nothing to do with overall score quality. The final *operating*
    threshold (used for real decisions) is computed separately, once, for
    whichever weight is chosen, using ``threshold_strategy``.
    """
    labels = np.asarray(labels, dtype=int)
    xgb_scores = np.asarray(xgb_scores, dtype=float)
    bbn_scores = np.asarray(bbn_scores, dtype=float)
    if len(labels) != len(xgb_scores) or len(labels) != len(bbn_scores):
        raise ValueError("labels, xgb_scores, and bbn_scores must be the same length")
    if brier_tolerance < 0:
        raise ValueError("brier_tolerance must be non-negative")

    rows: list[dict[str, float]] = []
    for weight in WEIGHT_GRID:
        fused = weight * xgb_scores + (1 - weight) * bbn_scores
        capacity_cutoff = capacity_threshold(fused, capacity_fraction)
        capacity_metrics = evaluate_predictions(labels, fused, capacity_cutoff)
        rows.append(
            {
                "xgb_weight": weight,
                "bbn_weight": round(1 - weight, 1),
                "pr_auc": capacity_metrics["pr_auc"],
                "roc_auc": capacity_metrics["roc_auc"],
                "brier": capacity_metrics["brier"],
                "capacity_recall": capacity_metrics["recall"],
                "capacity_precision": capacity_metrics["precision"],
                "calibration_error": expected_calibration_error(labels, fused),
            }
        )
    comparison = pd.DataFrame(rows)

    best_recall = float(comparison["capacity_recall"].max())
    eligible = comparison.loc[comparison["capacity_recall"] >= best_recall - recall_tolerance]
    if eligible.empty:  # pragma: no cover - defensive, cannot happen given max() membership
        eligible = comparison
    best_brier = float(eligible["brier"].min())
    near_best = eligible.loc[eligible["brier"] <= best_brier + brier_tolerance]
    ranked = near_best.sort_values(["xgb_weight", "brier"], ascending=[True, True])
    chosen = ranked.iloc[0]
    chosen_weight = float(chosen["xgb_weight"])

    if chosen_weight == 1.0:
        label = "xgb_only"
    elif chosen_weight == 0.0:
        label = "bbn_only"
    elif chosen_weight == RISK_MODEL_WEIGHT:
        label = "fixed_70_30"
    else:
        label = "validation_selected"

    # The final decision threshold is computed once, for the chosen weight,
    # using the configured (recall/precision/capacity) strategy.
    chosen_fused = chosen_weight * xgb_scores + (1 - chosen_weight) * bbn_scores
    final_selection = score_space_metrics(
        labels,
        chosen_fused,
        strategy=threshold_strategy,
        capacity_fraction=capacity_fraction,
        target_recall=target_recall,
        min_precision=min_precision,
    )
    comparison["threshold"] = np.nan
    chosen_mask = comparison["xgb_weight"] == chosen_weight
    comparison.loc[chosen_mask, "threshold"] = final_selection["threshold"]
    for metric_name in ("precision", "recall", "f1"):
        comparison[metric_name] = np.nan
        comparison.loc[chosen_mask, metric_name] = final_selection["metrics"][metric_name]

    rationale = (
        f"Selected xgb_weight={chosen_weight:.1f} ({label}) on validation: Brier "
        f"({float(chosen['brier']):.4f}) within {brier_tolerance:.3f} of the best "
        f"eligible Brier ({best_brier:.4f}), preferring causal-model contribution among "
        f"practically equivalent candidates whose top-{capacity_fraction:.0%}-"
        f"capacity recall is within {recall_tolerance:.2f} of the best candidate's capacity "
        f"recall ({best_recall:.3f}). The operating threshold "
        f"({final_selection['threshold']:.3f}) is then tuned separately via "
        f"'{threshold_strategy}' on the chosen weight's fused validation scores. No "
        "stacking model was fit; this is a 10%-increment line search over one "
        "interpretable convex weight."
    )
    return FusionSelection(
        chosen_weight=chosen_weight,
        chosen_label=label,
        comparison=comparison,
        rationale=rationale,
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Simple equal-width-bin expected calibration error."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if labels.size == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(probabilities, bin_edges[1:-1], right=True), 0, n_bins - 1)
    total_error = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        bin_confidence = float(probabilities[mask].mean())
        bin_accuracy = float(labels[mask].mean())
        total_error += (mask.sum() / labels.size) * abs(bin_confidence - bin_accuracy)
    return round(total_error, 6)


def lift_at_capacity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    capacity_fraction: float,
) -> float:
    """Miss rate among the top ``capacity_fraction`` scored orders vs. overall rate."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    if labels.size == 0:
        return float("nan")
    overall_rate = float(labels.mean())
    if overall_rate <= 0:
        return float("nan")
    capacity = max(1, int(np.ceil(labels.size * capacity_fraction)))
    top_indices = np.argsort(probabilities)[::-1][:capacity]
    top_rate = float(labels[top_indices].mean())
    return round(top_rate / overall_rate, 4)


def alert_rate(probabilities: np.ndarray, threshold: float) -> float:
    """Fraction of scored orders that would be flagged at ``threshold``."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0:
        return float("nan")
    return round(float((probabilities >= threshold).mean()), 4)


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
