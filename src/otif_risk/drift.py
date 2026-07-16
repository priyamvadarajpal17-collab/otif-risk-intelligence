"""Simple, explainable drift diagnostics for the daily operations replay.

Every measure here is intentionally a well-known, hand-computable statistic
(no external drift-detection library): population stability index (PSI) on
selected features, a mean-shift score-distribution check, a missingness-rate
change, and a recent-OTIF-rate change. ``evaluate_drift`` combines them into
one explicit trigger decision so the replay can log *why* it retrained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Conventional PSI interpretation thresholds (see e.g. credit-risk model
#: monitoring literature): >0.25 is "significant" distribution shift. These
#: are intentionally generous for a small-batch daily replay (a few hundred
#: open orders/day), where naive small-sample noise would otherwise trip a
#: tighter threshold on almost every day.
PSI_WARNING_THRESHOLD = 0.25
PSI_TRIGGER_THRESHOLD = 0.60
SCORE_MEAN_SHIFT_TRIGGER = 0.15
MISSINGNESS_SHIFT_TRIGGER = 0.20
OTIF_RATE_SHIFT_TRIGGER = 0.10
MIN_RECENT_OTIF_OBSERVATIONS = 30


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """PSI of ``current`` against ``baseline``'s own quantile-bin edges."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if baseline.size < bins or current.size == 0:
        return 0.0
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    baseline_fractions = np.clip(baseline_counts / baseline_counts.sum(), 1e-6, None)
    current_fractions = np.clip(current_counts / max(current_counts.sum(), 1), 1e-6, None)
    ratio = current_fractions / baseline_fractions
    return float(np.sum((current_fractions - baseline_fractions) * np.log(ratio)))


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    score_mean_shift: float
    missingness_rate_shift: float
    recent_otif_rate_shift: float
    recent_otif_observations: int
    otif_rate_trigger_eligible: bool
    triggered: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_psi": self.feature_psi,
            "score_mean_shift": self.score_mean_shift,
            "missingness_rate_shift": self.missingness_rate_shift,
            "recent_otif_rate_shift": self.recent_otif_rate_shift,
            "recent_otif_observations": self.recent_otif_observations,
            "otif_rate_trigger_eligible": self.otif_rate_trigger_eligible,
            "triggered": self.triggered,
            "reasons": self.reasons,
        }


def evaluate_drift(
    *,
    baseline_features: pd.DataFrame,
    current_features: pd.DataFrame,
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    baseline_missingness: float,
    current_missingness: float,
    baseline_otif_rate: float,
    current_otif_rate: float,
    recent_otif_observations: int,
    monitored_columns: tuple[str, ...] = (
        "vendor_rolling_fault_rate_30d",
        "active_leading_signal_count",
        "days_to_promised_delivery",
    ),
) -> DriftReport:
    """Compute all drift measures and one explicit trigger decision."""
    feature_psi = {
        column: round(
            population_stability_index(
                baseline_features[column].to_numpy(dtype=float),
                current_features[column].to_numpy(dtype=float),
            ),
            4,
        )
        for column in monitored_columns
        if column in baseline_features.columns and column in current_features.columns
    }
    score_mean_shift = float(
        abs(np.mean(current_scores) - np.mean(baseline_scores))
    ) if len(current_scores) and len(baseline_scores) else 0.0
    missingness_rate_shift = round(abs(current_missingness - baseline_missingness), 4)
    recent_otif_rate_shift = round(abs(current_otif_rate - baseline_otif_rate), 4)

    reasons: list[str] = []
    max_psi = max(feature_psi.values()) if feature_psi else 0.0
    if max_psi >= PSI_TRIGGER_THRESHOLD:
        reasons.append(f"feature PSI {max_psi:.3f} >= {PSI_TRIGGER_THRESHOLD}")
    if score_mean_shift >= SCORE_MEAN_SHIFT_TRIGGER:
        reasons.append(f"score mean shift {score_mean_shift:.3f} >= {SCORE_MEAN_SHIFT_TRIGGER}")
    if missingness_rate_shift >= MISSINGNESS_SHIFT_TRIGGER:
        reasons.append(
            f"missingness rate shift {missingness_rate_shift:.3f} >= {MISSINGNESS_SHIFT_TRIGGER}"
        )
    otif_rate_trigger_eligible = (
        recent_otif_observations >= MIN_RECENT_OTIF_OBSERVATIONS
    )
    if otif_rate_trigger_eligible and recent_otif_rate_shift >= OTIF_RATE_SHIFT_TRIGGER:
        reasons.append(
            f"recent OTIF rate shift {recent_otif_rate_shift:.3f} >= {OTIF_RATE_SHIFT_TRIGGER}"
        )
    return DriftReport(
        feature_psi=feature_psi,
        score_mean_shift=round(score_mean_shift, 4),
        missingness_rate_shift=missingness_rate_shift,
        recent_otif_rate_shift=recent_otif_rate_shift,
        recent_otif_observations=recent_otif_observations,
        otif_rate_trigger_eligible=otif_rate_trigger_eligible,
        triggered=bool(reasons),
        reasons=reasons,
    )
