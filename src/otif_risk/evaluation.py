"""Held-out evaluation shared across the risk model, Bayesian network, and fusion.

Item 4 of the remediation plan requires XGBoost, Bayesian, and fused probabilities
to be evaluated independently on the *same* held-out labels with the *same* metric
set, a simple prevalence baseline for context, and a cause/pathway fidelity check
against the generator's ground truth. This module intentionally contains no
model-fitting logic; it only scores already-produced probability columns so the
same functions can be reused for validation and test splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from otif_risk.contracts import CAUSE_CATEGORIES
from otif_risk.model import ThresholdStrategy, evaluate_predictions, select_threshold


def score_space_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    *,
    strategy: ThresholdStrategy,
    capacity_fraction: float,
    target_recall: float,
    min_precision: float,
) -> dict[str, Any]:
    """Select a threshold on ``probabilities`` and return validation metrics.

    Used identically for the XGBoost, Bayesian, and fused score spaces so that
    each is evaluated, and thresholded, entirely within its own probability
    space rather than a threshold selected on one score being silently applied
    to another (the defect this remediation corrects for the fused score).
    """
    selection = select_threshold(
        labels,
        probabilities,
        strategy=strategy,
        capacity_fraction=capacity_fraction,
        target_recall=target_recall,
        min_precision=min_precision,
    )
    return {
        "threshold": selection.threshold,
        "strategy": selection.strategy,
        "metrics": selection.validation_metrics,
    }


def evaluate_at_threshold(
    labels: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one score space at an already-selected operating threshold."""
    return evaluate_predictions(labels, probabilities, threshold)


def prevalence_baseline_metrics(labels: pd.Series | np.ndarray) -> dict[str, Any]:
    """Score a trivial baseline that predicts the sample prevalence for every order.

    This baseline has no discriminative power by construction (every order gets
    the same score), so it exists purely as a floor for comparison: any of the
    three real score spaces should clearly separate from it. Because every
    prediction is tied, PR-AUC/ROC-AUC are not meaningfully defined (scikit-learn
    still returns a number driven entirely by tie-breaking); they are reported
    for transparency but flagged as uninformative for this reason.
    """
    y_true = np.asarray(labels, dtype=int)
    prevalence = float(y_true.mean()) if y_true.size else 0.0
    probabilities = np.full(y_true.shape, prevalence, dtype=float)
    metrics = evaluate_predictions(y_true, probabilities, threshold=prevalence)
    metrics["prevalence"] = prevalence
    metrics["note"] = (
        "Constant-probability baseline (always predicts the sample prevalence). "
        "PR-AUC/ROC-AUC are not meaningful here because the baseline assigns every "
        "order an identical score and cannot rank them; use it only as a floor, "
        "not as a ranking comparison."
    )
    return metrics


def cause_fidelity_report(
    predicted_cause: pd.Series,
    truth_cause: pd.Series,
) -> dict[str, Any]:
    """Compare the recovered primary cause with ground truth for OTIF misses.

    ``predicted_cause`` is the pipeline's evidence-derived primary cause (from
    observed leading signals and Bayesian cause lift ranking); ``truth_cause`` is
    the retrospective root-cause label computed directly from full event history
    in ``root_causes.py``. Agreement is a pathway *fidelity* diagnostic, not a
    causality claim: both series are order_id-aligned label vectors over the
    same held-out missed orders. Successful orders are intentionally excluded:
    the open-order pathway scorer returns ``UNKNOWN`` when it sees no leading
    evidence, while retrospective truth correctly labels those rows ``ON_TIME``.
    Including them would measure outcome classification rather than cause fidelity.
    """
    if len(predicted_cause) != len(truth_cause):
        raise ValueError("predicted_cause and truth_cause must be the same length")
    aligned = pd.DataFrame(
        {
            "predicted": pd.Series(predicted_cause).astype(str).reset_index(drop=True),
            "truth": pd.Series(truth_cause).astype(str).reset_index(drop=True),
        }
    )
    overall_agreement = (
        float((aligned["predicted"] == aligned["truth"]).mean()) if len(aligned) else float("nan")
    )
    per_cause: dict[str, dict[str, Any]] = {}
    for cause in (*CAUSE_CATEGORIES, "UNKNOWN"):
        mask = aligned["truth"] == cause
        support = int(mask.sum())
        per_cause[cause] = {
            "support": support,
            "recall": (
                float((aligned.loc[mask, "predicted"] == cause).mean()) if support else float("nan")
            ),
        }
    return {
        "evaluated_orders": int(len(aligned)),
        "scope": "held-out OTIF misses only",
        "overall_agreement": overall_agreement,
        "per_cause_recall": per_cause,
        "note": (
            "Agreement compares the evidence-derived primary cause against the "
            "generator's retrospective ground-truth label on held-out OTIF misses. "
            "Successful ON_TIME orders are excluded because they have no failure "
            "cause to recover. This is a fidelity/association diagnostic, not proof "
            "that the Bayesian pathway is causally correct."
        ),
    }
