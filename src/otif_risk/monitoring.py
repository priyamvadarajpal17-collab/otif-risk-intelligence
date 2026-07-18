"""Rolling operations monitoring and SLO reporting (Stage 2 governance).

Computes rolling-origin realized prediction/policy quality, regime
(normal vs. scripted-drift) quality, time-to-detection, feature
freshness, measured local runtime, and soft data-quality metrics over an
**already-completed** operations replay -- everything here is measured
from persisted artifacts (the decision ledger, the order master data, the
daily replay log, and an assembled source-table snapshot), never
recomputed against the simulator's own potential outcomes. The exact,
common-random-number potential-outcome policy value stays in
``policy_evaluation.py``/``artifacts/policy_benchmark.json``; this module
answers a different, narrower question: "how is the deployed policy's
*realized*, observational quality trending over time, and is it inside
its own transparent operating targets?"

Every runtime number here is explicitly labeled ``measured_local_runtime``
-- this prototype makes no production-latency claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from otif_risk.data import DRIFT_WINDOW_FRACTION
from otif_risk.fusion import expected_calibration_error
from otif_risk.model import evaluate_predictions

MONITORING_SCHEMA_VERSION = "1.0"

#: A rolling window/regime cohort needs at least this many matured
#: decisions before its quality metrics are reported as numbers rather
#: than withheld.
MIN_ROLLING_SAMPLE = 30
DEFAULT_WINDOW_DAYS = 30


@dataclass(frozen=True)
class SloTarget:
    metric: str
    target: float
    comparison: str  # "gte" or "lte"
    scope: str


#: Fixed, transparent operating targets. ``scope`` documents what kind of
#: claim each target is (prediction quality measured on this twin, or
#: local runtime/freshness), so a reader never mistakes a locally-measured
#: number for a production SLA.
SLO_TARGETS: tuple[SloTarget, ...] = (
    SloTarget("rolling_pr_auc", 0.55, "gte", "prediction_quality_on_this_twin"),
    SloTarget("rolling_calibration_error", 0.15, "lte", "prediction_quality_on_this_twin"),
    SloTarget("rolling_alert_rate", 0.45, "lte", "operational_load_on_this_twin"),
    SloTarget("contract_failure_count", 0, "lte", "data_quality"),
    SloTarget("feature_freshness_days", 1.0, "lte", "measured_local_runtime"),
)


def _slo_entry(value: float | None, target: SloTarget) -> dict[str, Any]:
    if value is None:
        return {
            "value": None,
            "target": target.target,
            "comparison": target.comparison,
            "scope": target.scope,
            "passed": None,
        }
    passed = value >= target.target if target.comparison == "gte" else value <= target.target
    return {
        "value": round(float(value), 4),
        "target": target.target,
        "comparison": target.comparison,
        "scope": target.scope,
        "passed": bool(passed),
    }


def rolling_prediction_quality(
    ledger: pd.DataFrame,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_sample: int = MIN_ROLLING_SAMPLE,
) -> list[dict[str, Any]]:
    """Rolling-origin realized PR-AUC/precision/recall/calibration/alert-rate,
    computed over fixed, chronologically-ordered ``window_days``-wide windows
    of *matured* ledger decisions only.

    A window with fewer than ``min_sample`` matured decisions still appears
    in the output (so gaps are visible), but its metrics are withheld
    (``sufficient_sample: False``, no metric keys) rather than reported on
    too few observations.
    """
    if ledger.empty:
        return []
    matured = ledger.loc[ledger["matured"].astype(bool)].copy()
    if matured.empty:
        return []
    matured["decision_timestamp"] = pd.to_datetime(matured["decision_timestamp"])
    matured = matured.sort_values("decision_timestamp")
    start = matured["decision_timestamp"].min()
    end = matured["decision_timestamp"].max()

    windows: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        window_end = cursor + pd.Timedelta(days=window_days)
        subset = matured.loc[
            (matured["decision_timestamp"] >= cursor) & (matured["decision_timestamp"] < window_end)
        ]
        n = int(len(subset))
        entry: dict[str, Any] = {
            "window_start": cursor.isoformat(),
            "window_end": window_end.isoformat(),
            "n_matured_decisions": n,
            "sufficient_sample": n >= min_sample,
        }
        if n >= min_sample:
            labels = subset["matured_otif_miss"].astype(int).to_numpy()
            scores = subset["risk_score"].astype(float).to_numpy()
            threshold = float(subset["threshold"].astype(float).median())
            evaluated = evaluate_predictions(labels, scores, threshold)
            entry.update(
                {
                    "pr_auc": evaluated["pr_auc"],
                    "precision": evaluated["precision"],
                    "recall": evaluated["recall"],
                    "calibration_error": expected_calibration_error(labels, scores),
                    "alert_rate": round(
                        float((subset["planner_decision"] != "MONITORED").mean()), 4
                    ),
                }
            )
        windows.append(entry)
        cursor = window_end
    return windows


def regime_quality(
    ledger: pd.DataFrame,
    order_dates: pd.Series,
    *,
    min_sample: int = MIN_ROLLING_SAMPLE,
) -> dict[str, Any]:
    """Normal-vs-scripted-drift realized quality (see ``data.DRIFT_WINDOW_FRACTION``).

    ``order_dates`` must be a ``pd.Series`` indexed by ``order_id`` giving
    each order's own capture date (``orders.csv``'s ``order_date``) -- the
    same regime boundary ``data.generate_dataset``/``policy_evaluation``
    already script, applied here to the ledger's matured, realized outcomes.
    """
    if ledger.empty:
        return {}
    matured = ledger.loc[ledger["matured"].astype(bool)].copy()
    if matured.empty:
        return {}
    matured["order_date"] = matured["order_id"].map(order_dates)
    matured = matured.dropna(subset=["order_date"])
    if matured.empty:
        return {}

    horizon_start = order_dates.min()
    horizon_end = order_dates.max()
    drift_start = horizon_end - (horizon_end - horizon_start) * DRIFT_WINDOW_FRACTION
    matured["regime"] = np.where(matured["order_date"] >= drift_start, "drift", "normal")

    result: dict[str, Any] = {}
    for regime, group in matured.groupby("regime"):
        n = int(len(group))
        if n < min_sample:
            result[regime] = {"n_matured_decisions": n, "sufficient_sample": False}
            continue
        labels = group["matured_otif_miss"].astype(int).to_numpy()
        scores = group["risk_score"].astype(float).to_numpy()
        threshold = float(group["threshold"].astype(float).median())
        evaluated = evaluate_predictions(labels, scores, threshold)
        accepted = group.loc[group["planner_decision"] == "ACCEPTED"]
        result[regime] = {
            "n_matured_decisions": n,
            "sufficient_sample": True,
            "pr_auc": evaluated["pr_auc"],
            "precision": evaluated["precision"],
            "recall": evaluated["recall"],
            "mean_realized_penalty_accepted": (
                round(float(accepted["realized_penalty"].mean()), 2) if len(accepted) else None
            ),
        }
    return result


def time_to_detection(
    ledger: pd.DataFrame, orders: pd.DataFrame, *, min_sample: int = MIN_ROLLING_SAMPLE
) -> dict[str, Any]:
    """Median lead time between an order first being flagged (accepted or
    capacity-rejected, i.e. scored above threshold) and its promised
    delivery date, restricted to matured orders that went on to miss OTIF.

    A larger lead time means the model/policy is catching risk earlier
    relative to the delivery promise, giving operations more time to act.
    """
    if ledger.empty:
        return {"n": 0, "sufficient_sample": False}
    flagged = ledger.loc[
        ledger["planner_decision"].isin(["ACCEPTED", "REJECTED"])
        & ledger["matured"].astype(bool)
        & (pd.to_numeric(ledger["matured_otif_miss"], errors="coerce") == 1)
    ]
    if flagged.empty:
        return {"n": 0, "sufficient_sample": False}
    first_flagged_at = pd.to_datetime(flagged["decision_timestamp"]).groupby(
        flagged["order_id"]
    ).min()
    promised = orders.set_index("order_id")["promised_delivery_date"]
    merged = pd.DataFrame(
        {"first_flagged_at": first_flagged_at}
    ).join(promised, how="inner")
    n = int(len(merged))
    if n < min_sample:
        return {"n": n, "sufficient_sample": False}
    lead_hours = (
        pd.to_datetime(merged["promised_delivery_date"])
        - pd.to_datetime(merged["first_flagged_at"])
    ).dt.total_seconds() / 3600.0
    return {
        "n": n,
        "sufficient_sample": True,
        "median_lead_time_hours_before_promised_delivery": round(float(lead_hours.median()), 2),
        "min_lead_time_hours_before_promised_delivery": round(float(lead_hours.min()), 2),
    }


def feature_freshness(daily_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Scoring cadence/freshness of the replay's as-of feature snapshots.

    This prototype's daily replay always scores every still-open order at
    an as-of timestamp exactly one calendar day fresh, by construction --
    there is no queuing/ingestion delay to measure, so the cadence is
    reported as a structural fact, not an invented SLA.
    """
    return {
        "scoring_cadence_days": 1.0,
        "days_replayed": len(daily_log),
        "note": (
            "This prototype's daily operations replay scores every open order "
            "at a single, shared as-of snapshot exactly one calendar day after "
            "the prior snapshot -- there is no batch/queue delay in this local "
            "harness to measure."
        ),
    }


def runtime_metrics(
    scoring_seconds: list[float] | None = None,
    retrain_seconds: list[float] | None = None,
) -> dict[str, Any]:
    """Measured *local* wall-clock scoring/retrain runtime.

    Explicitly scoped as ``measured_local_runtime`` -- never a production
    latency claim (no network, database, or queuing layer exists in this
    local prototype to characterize).
    """

    def _stats(values: list[float]) -> dict[str, Any] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "mean_seconds": round(float(array.mean()), 4),
            "max_seconds": round(float(array.max()), 4),
        }

    return {
        "scope": "measured_local_runtime_only_not_a_production_latency_claim",
        "scoring_runtime": _stats(scoring_seconds or []),
        "retrain_runtime": _stats(retrain_seconds or []),
    }


def build_monitoring_report(
    *,
    ledger: pd.DataFrame,
    orders: pd.DataFrame,
    order_dates: pd.Series,
    daily_log: list[dict[str, Any]],
    data_quality: dict[str, Any],
    scoring_seconds: list[float] | None = None,
    retrain_seconds: list[float] | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_sample: int = MIN_ROLLING_SAMPLE,
) -> dict[str, Any]:
    """Aggregate every monitoring dimension into one persisted report + SLO status."""
    rolling = rolling_prediction_quality(ledger, window_days=window_days, min_sample=min_sample)
    regime = regime_quality(ledger, order_dates, min_sample=min_sample)
    detection = time_to_detection(ledger, orders, min_sample=min_sample)
    freshness = feature_freshness(daily_log)
    runtime = runtime_metrics(scoring_seconds, retrain_seconds)
    contract_failure_count = int(data_quality.get("contract_failure_count", 0))

    latest_sufficient = next(
        (window for window in reversed(rolling) if window["sufficient_sample"]), None
    )
    slo_status = {
        "rolling_pr_auc": _slo_entry(
            latest_sufficient.get("pr_auc") if latest_sufficient else None, SLO_TARGETS[0]
        ),
        "rolling_calibration_error": _slo_entry(
            latest_sufficient.get("calibration_error") if latest_sufficient else None,
            SLO_TARGETS[1],
        ),
        "rolling_alert_rate": _slo_entry(
            latest_sufficient.get("alert_rate") if latest_sufficient else None, SLO_TARGETS[2]
        ),
        "contract_failure_count": _slo_entry(contract_failure_count, SLO_TARGETS[3]),
        "feature_freshness_days": _slo_entry(
            freshness["scoring_cadence_days"], SLO_TARGETS[4]
        ),
    }

    return {
        "monitoring_schema_version": MONITORING_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "min_sample_guard": min_sample,
        "window_days": window_days,
        "rolling_prediction_quality_windows": rolling,
        "regime_quality": regime,
        "time_to_detection": detection,
        "feature_freshness": freshness,
        "runtime": runtime,
        "data_quality": data_quality,
        "slo_status": slo_status,
        "slo_targets": [asdict(target) for target in SLO_TARGETS],
        "qualification": (
            "Rolling/regime quality is measured from the decision ledger's own "
            "matured, realized outcomes (observational), never recomputed from "
            "the simulator's potential outcomes -- see policy_evaluation.py for "
            "the exact, common-random-number measured policy value, kept "
            "separate. Runtime figures are this local machine's measured "
            "wall-clock time only, not a production latency claim."
        ),
    }


def write_monitoring_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
