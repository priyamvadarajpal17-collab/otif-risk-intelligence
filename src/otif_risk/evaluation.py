"""Held-out evaluation shared across the risk model, Bayesian network, and fusion.

XGBoost, Bayesian, and fused probabilities are evaluated independently on the
same held-out labels with the same metrics. This module also provides a prevalence
baseline and cause/pathway fidelity diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from otif_risk.contracts import CAUSE_CATEGORIES
from otif_risk.model import ThresholdStrategy, evaluate_predictions, select_threshold

#: Maps each simulator-truth delay column to the operational cause node it
#: represents, for the evaluation-only "simulator responsive cause" proxy
#: (see `simulator_responsive_causes`). Never used as a model feature.
STAGE_DELAY_TO_CAUSE: dict[str, str] = {
    "vendor_ready_delay_hours": "VENDOR_FAILURE",
    "warehouse_delay_hours": "WAREHOUSE_OPS",
    "transit_delay_hours": "TRANSPORT",
    "customer_delay_hours": "CUSTOMER_DELIVERY",
    "unknown_extra_hours": "UNKNOWN",
}


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

    Used identically for the XGBoost, Bayesian, and fused score spaces so each
    is evaluated and thresholded within its own probability space.
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
    majority_cause_baseline = (
        float(aligned["truth"].value_counts(normalize=True).max())
        if len(aligned)
        else float("nan")
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
        "majority_cause_baseline": majority_cause_baseline,
        "per_cause_recall": per_cause,
        "note": (
            "Agreement compares the evidence-derived primary cause against the "
            "retrospective rule-derived primary cause on held-out OTIF misses. "
            "Successful ON_TIME orders are excluded because they have no failure "
            "cause to recover. The reference shares operational evidence with the "
            "point-in-time scorer, so this measures derivation consistency rather "
            "than latent simulator-truth recovery or causal correctness."
        ),
    }


def _safe_auc(metric: Any, labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(metric(labels, probabilities))


def mechanism_metrics(
    late_delivery_truth: pd.Series | np.ndarray,
    late_delivery_probability: pd.Series | np.ndarray,
    in_full_failure_truth: pd.Series | np.ndarray,
    in_full_failure_probability: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """PR-AUC/Brier for the two mechanism nodes against their own held-out truth.

    ``LATE_DELIVERY`` truth is ``1 - on_time``; ``IN_FULL_FAILURE`` truth is
    ``1 - in_full``. This evaluates each intermediate mechanism node directly
    against the half of the OTIF definition it represents, independent of the
    fused XGBoost decision.
    """
    late_truth = np.asarray(late_delivery_truth, dtype=int)
    late_probability = np.asarray(late_delivery_probability, dtype=float)
    in_full_truth = np.asarray(in_full_failure_truth, dtype=int)
    in_full_probability = np.asarray(in_full_failure_probability, dtype=float)
    return {
        "late_delivery": {
            "pr_auc": _safe_auc(average_precision_score, late_truth, late_probability),
            "brier": float(brier_score_loss(late_truth, late_probability)),
            "positive_rate": float(late_truth.mean()) if late_truth.size else float("nan"),
        },
        "in_full_failure": {
            "pr_auc": _safe_auc(average_precision_score, in_full_truth, in_full_probability),
            "brier": float(brier_score_loss(in_full_truth, in_full_probability)),
            "positive_rate": float(in_full_truth.mean()) if in_full_truth.size else float("nan"),
        },
        "note": (
            "Mechanism metrics evaluate each intermediate node's probability "
            "(P(LATE_DELIVERY), P(IN_FULL_FAILURE)) against its own held-out "
            "ground truth (1 - on_time, 1 - in_full) -- a genuine predictive "
            "check of the split OTIF definition, independent of the fused "
            "XGBoost decision."
        ),
    }


def simulator_responsive_causes(
    outcomes: pd.DataFrame,
    simulator_truth: pd.DataFrame,
    orders: pd.DataFrame | None = None,
) -> pd.Series:
    """Evaluation-only "which stage plausibly drove this miss" reference label.

    Derived purely from the simulator's per-stage delay-hours and
    shortfall-ratio columns (never used as a model feature): a late order is
    labeled with whichever timing stage carries the largest recorded delay
    (or "UNKNOWN", for the organic unexplained-miss noise); a
    short-but-on-time order is labeled ``INVENTORY_SHORTAGE``; an on-time,
    in-full order is labeled ``ON_TIME``. This is a simulator-derived
    *reference* label for a consistency diagnostic, not a proven causal
    ground truth -- real orders can have multiple contributing stages, and
    this picks a single dominant one deterministically.

    When ``orders`` is supplied, its ``capture_delay_hours`` column (the same
    field ``root_causes.py`` uses for the ``ORDER_CAPTURE`` rule -- available
    at prediction time, not a leak) is included as a fifth timing-delay
    candidate mapped to ``ORDER_CAPTURE``, so a capture-delay-dominated late
    order is not silently mislabeled as whichever of the other four stages
    happens to be largest. ``orders`` is optional only for backward
    compatibility with callers that do not have it on hand.
    """
    required_outcomes = {"order_id", "on_time", "in_full"}
    if missing := sorted(required_outcomes - set(outcomes.columns)):
        raise ValueError(f"outcomes is missing columns: {missing}")
    required_truth = {"order_id", *STAGE_DELAY_TO_CAUSE}
    if missing := sorted(required_truth - set(simulator_truth.columns)):
        raise ValueError(f"simulator_truth is missing columns: {missing}")

    merged = outcomes[["order_id", "on_time", "in_full"]].merge(
        simulator_truth[["order_id", *STAGE_DELAY_TO_CAUSE]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    delay_columns = list(STAGE_DELAY_TO_CAUSE)
    delay_to_cause = dict(STAGE_DELAY_TO_CAUSE)
    if orders is not None:
        required_orders = {"order_id", "capture_delay_hours"}
        if missing := sorted(required_orders - set(orders.columns)):
            raise ValueError(f"orders is missing columns: {missing}")
        merged = merged.merge(
            orders[["order_id", "capture_delay_hours"]],
            on="order_id",
            how="left",
            validate="one_to_one",
        )
        delay_columns = [*delay_columns, "capture_delay_hours"]
        delay_to_cause = {**delay_to_cause, "capture_delay_hours": "ORDER_CAPTURE"}

    delays = merged[delay_columns].fillna(0.0)
    dominant_delay_cause = delays.idxmax(axis=1).map(delay_to_cause)

    late = merged["on_time"].astype(int) == 0
    short = merged["in_full"].astype(int) == 0
    labels = pd.Series("ON_TIME", index=merged.index, dtype=object)
    labels = labels.mask(short & ~late, "INVENTORY_SHORTAGE")
    labels = labels.mask(late, dominant_delay_cause)
    return pd.Series(
        labels.to_numpy(), index=merged["order_id"].to_numpy(), name="simulator_responsive_cause"
    )


def causal_consistency_report(comparisons: pd.DataFrame) -> dict[str, Any]:
    """Agreement between the model's top-ranked causal node and two reference labels.

    ``comparisons`` must have one row per evaluated order with columns
    ``top_attribution_cause`` (the evidence-attribution node with the largest
    absolute contribution), ``top_intervention_cause`` (the single-node
    mitigation scenario with the largest absolute risk reduction),
    ``rule_primary_cause`` (the retrospective rule-derived primary cause), and
    ``simulator_responsive_cause`` (see `simulator_responsive_causes`). This
    reports plain agreement rates -- a *consistency* diagnostic showing the
    model usually ranks the same node an independent rule/simulator would
    flag -- not evidence that the modeled risk reduction is a validated causal
    effect.
    """
    required = {
        "top_attribution_cause",
        "top_intervention_cause",
        "rule_primary_cause",
        "simulator_responsive_cause",
    }
    if missing := sorted(required - set(comparisons.columns)):
        raise ValueError(f"comparisons is missing columns: {missing}")

    evaluated = comparisons.dropna(subset=["top_attribution_cause", "top_intervention_cause"])

    def _agreement(predicted_column: str, reference_column: str) -> float:
        aligned = evaluated.dropna(subset=[predicted_column, reference_column])
        if aligned.empty:
            return float("nan")
        return float((aligned[predicted_column] == aligned[reference_column]).mean())

    return {
        "evaluated_orders": int(len(evaluated)),
        "scope": "held-out OTIF misses with at least one active evidence node",
        "top_attribution_vs_rule_cause": _agreement("top_attribution_cause", "rule_primary_cause"),
        "top_attribution_vs_simulator_responsive_cause": _agreement(
            "top_attribution_cause", "simulator_responsive_cause"
        ),
        "top_intervention_vs_rule_cause": _agreement(
            "top_intervention_cause", "rule_primary_cause"
        ),
        "top_intervention_vs_simulator_responsive_cause": _agreement(
            "top_intervention_cause", "simulator_responsive_cause"
        ),
        "note": (
            "These are agreement/consistency rates between the model's top-ranked "
            "evidence-attribution or intervention-mitigation node and two "
            "independently derived reference labels for held-out OTIF misses "
            "(the retrospective rule-derived primary cause and a simulator-derived "
            "'largest recorded stage delay/"
            "shortfall' proxy). They measure whether the model's ranking usually "
            "points at the same node an independent rule/simulator would flag -- "
            "NOT that the modeled risk reduction is a validated causal effect. "
            "Structural interventions remain fixed-structure scenario analysis "
            "under this model's assumptions."
        ),
    }


def confidence_diagnostics(
    evidence_coverage: pd.Series | np.ndarray, causal_confidence: pd.Series
) -> dict[str, Any]:
    """Evidence-coverage distribution and the LOW-confidence (abstention-adjacent) rate.

    This prototype never abstains -- every order is always scored -- so
    ``low_confidence_rate`` reports how often that score should be read with
    the LOW-confidence caveat rather than an actual withheld prediction.
    """
    coverage = pd.to_numeric(pd.Series(evidence_coverage), errors="coerce").dropna()
    confidence = pd.Series(causal_confidence).astype(str)
    total = int(len(confidence))
    band_counts = confidence.value_counts().to_dict()
    low_count = int(band_counts.get("LOW", 0))
    return {
        "evidence_coverage": {
            "mean": float(coverage.mean()) if len(coverage) else float("nan"),
            "median": float(coverage.median()) if len(coverage) else float("nan"),
            "min": float(coverage.min()) if len(coverage) else float("nan"),
            "max": float(coverage.max()) if len(coverage) else float("nan"),
        },
        "confidence_band_counts": {
            band: int(band_counts.get(band, 0)) for band in ("LOW", "MEDIUM", "HIGH")
        },
        "low_confidence_rate": (low_count / total) if total else float("nan"),
        "total_orders": total,
        "note": (
            "Evidence coverage is observed cause nodes / 7. LOW-confidence orders "
            "are scored with no gated stage (vendor-ready/shipped/transit) observed "
            "yet; this prototype never abstains -- they are still scored -- but the "
            "causal_pathway and intervention scenarios should be read with that "
            "caveat."
        ),
    }
