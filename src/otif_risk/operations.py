"""Local daily operations replay: scoring, closures, drift, and retraining.

Simulates a control-tower operating loop entirely offline against one
generated digital-twin dataset (no external services, no live scheduler):

1. Train an initial model bundle on a historical window.
2. For each subsequent simulated day: score every still-open order as of
   that day (`features.build_feature_table(..., as_of_timestamp=today)`),
   allocate today's resource capacities, persist the day's queue, close any
   orders whose outcome has now resolved, derive their actual cause, and
   append feedback.
3. Track drift (PSI, score-distribution shift, missingness, recent OTIF
   rate) against the last training baseline.
4. Retrain on a documented cadence or when drift triggers and enough new
   matured labels exist, incrementing a versioned model registry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from otif_risk.contracts import PrototypeConfig, PrototypeDataset
from otif_risk.data import generate_dataset
from otif_risk.decisions import RECOMMENDATION_TABLE
from otif_risk.drift import evaluate_drift
from otif_risk.features import attach_line_evidence_features, build_feature_table, temporal_split
from otif_risk.feedback import append_feedback
from otif_risk.pipeline import TrainedBundle, score_orders, train_full_bundle
from otif_risk.resources import allocate_interventions, default_daily_capacities
from otif_risk.root_causes import calculate_outcomes, derive_root_causes
from otif_risk.validation import validate_dataset

#: Default replay/training-window sizing. Kept modest so a canonical replay
#: completes in practical local runtime.
DEFAULT_INITIAL_TRAINING_FRACTION = 0.45
DEFAULT_REPLAY_DAYS = 90
DEFAULT_RETRAIN_CADENCE_DAYS = 15
DEFAULT_MIN_NEW_LABELS_FOR_RETRAIN = 30
DEFAULT_DRIFT_LOOKBACK_DAYS = 10


@dataclass
class OperationsConfig:
    data_config: PrototypeConfig
    output_dir: Path = Path("artifacts")
    initial_training_fraction: float = DEFAULT_INITIAL_TRAINING_FRACTION
    replay_days: int = DEFAULT_REPLAY_DAYS
    retrain_cadence_days: int = DEFAULT_RETRAIN_CADENCE_DAYS
    min_new_labels_for_retrain: int = DEFAULT_MIN_NEW_LABELS_FOR_RETRAIN
    drift_lookback_days: int = DEFAULT_DRIFT_LOOKBACK_DAYS
    #: Minimum simulated days between any two retrains (even drift-triggered
    #: ones), so persistent drift cannot thrash the model every single day.
    min_days_between_retrains: int = 10


@dataclass
class ModelRegistryEntry:
    version: int
    trained_at_simulated_day: str
    training_window_start: str
    training_window_end: str
    n_training_orders: int
    validation_metrics: dict[str, Any]
    threshold: float
    fusion_weight: float
    fusion_label: str
    trigger: str
    trigger_reasons: list[str]
    artifact_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DailyRecord:
    simulated_day: str
    open_orders: int
    newly_closed_orders: int
    recommended: int
    contested: int
    monitor: int
    model_version: int
    drift: dict[str, Any] = field(default_factory=dict)
    retrained: bool = False
    retrain_trigger: str | None = None


def _daily_candidate_frame(
    scored: pd.DataFrame,
    risk_threshold: float,
) -> pd.DataFrame:
    """Attach the recommendation-table policy and priority score for one day."""
    result = scored.copy()
    policies = result["primary_cause"].astype(str).str.upper().map(RECOMMENDATION_TABLE)
    fallback = {
        "action": "Review the order exception and confirm a recovery plan",
        "owner": "OTIF control tower",
        "resource_type": "dc",
    }
    policies = policies.map(lambda value: value if isinstance(value, dict) else fallback)
    result["recommended_action"] = policies.map(lambda value: value["action"])
    result["action_owner"] = policies.map(lambda value: value["owner"])

    risk = pd.to_numeric(result["combined_risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
    value = pd.to_numeric(result.get("order_value", 0.0), errors="coerce").fillna(0.0).clip(lower=0)
    value_scale = max(float(value.quantile(0.95)), 1.0)
    value_weight = (value / value_scale).clip(upper=1.0)
    quantity = pd.to_numeric(result.get("total_order_qty", 0.0), errors="coerce").fillna(0.0)
    result["priority_score"] = (100 * risk * (0.7 + 0.3 * value_weight)).round(2)
    result["quantity_at_risk"] = (quantity * risk * 0.5).round(2)
    result["is_candidate"] = risk >= risk_threshold
    return result


def _run_operations_directory(config: OperationsConfig) -> Path:
    import hashlib

    values = {**asdict(config.data_config), "output_dir": str(config.data_config.output_dir)}
    values["ops"] = {
        "initial_training_fraction": config.initial_training_fraction,
        "replay_days": config.replay_days,
        "retrain_cadence_days": config.retrain_cadence_days,
        "min_new_labels_for_retrain": config.min_new_labels_for_retrain,
    }
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:10]
    base = config.output_dir / f"ops-{digest}"
    if not base.exists():
        return base
    suffix = 2
    while (config.output_dir / f"ops-{digest}-{suffix}").exists():
        suffix += 1
    return config.output_dir / f"ops-{digest}-{suffix}"


def _retrain(
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    matured_order_ids: pd.Index,
    data_config: PrototypeConfig,
) -> tuple[TrainedBundle, pd.DataFrame]:
    """Train a fresh bundle on every order that has matured by `today`."""
    feature_table = build_feature_table(
        dataset, outcomes, causes, order_ids=matured_order_ids
    )
    feature_table = attach_line_evidence_features(dataset, feature_table)
    split = temporal_split(feature_table)
    trained = train_full_bundle(dataset, outcomes, causes, split, data_config)
    return trained, feature_table


def run_operations_replay(config: OperationsConfig) -> dict[str, Any]:
    """Run the full daily replay and persist registry/queue/drift artifacts."""
    dataset = generate_dataset(config.data_config)
    validate_dataset(dataset)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)

    horizon_end = dataset.orders["order_date"].max()
    horizon_start = dataset.orders["order_date"].min()
    total_days = (horizon_end - horizon_start).days
    initial_cutoff = horizon_start + pd.Timedelta(
        days=int(total_days * config.initial_training_fraction)
    )

    matured_at_start = outcomes.loc[
        (outcomes["prediction_timestamp"] <= initial_cutoff)
        & (outcomes["outcome_timestamp"] <= initial_cutoff),
        "order_id",
    ]
    if len(matured_at_start) < 50:
        raise ValueError("initial training window has too few matured orders to fit a model")

    trained, baseline_features = _retrain(
        dataset, outcomes, causes, pd.Index(matured_at_start), config.data_config
    )
    registry: list[ModelRegistryEntry] = [
        ModelRegistryEntry(
            version=1,
            trained_at_simulated_day=initial_cutoff.isoformat(),
            training_window_start=horizon_start.isoformat(),
            training_window_end=initial_cutoff.isoformat(),
            n_training_orders=len(matured_at_start),
            validation_metrics=trained.validation_metrics,
            threshold=trained.fused_threshold,
            fusion_weight=trained.fusion_selection.chosen_weight,
            fusion_label=trained.fusion_selection.chosen_label,
            trigger="initial",
            trigger_reasons=[],
            artifact_path="models/v1",
        )
    ]

    run_dir = _run_operations_directory(config)
    queue_dir = run_dir / "daily_queues"
    model_dir = run_dir / "models"
    queue_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained.risk_training.bundle, model_dir / "v1_xgboost_risk.joblib")
    joblib.dump(trained.bayesian_bundle, model_dir / "v1_bayesian_network.joblib")

    feedback_path = run_dir / "planner_feedback.csv"
    baseline_scores = trained.risk_training.bundle.predict_proba(baseline_features)
    baseline_missingness = float(baseline_features["missing_event_stage_count"].mean() / 3)
    baseline_otif_rate = float(
        outcomes.loc[outcomes["order_id"].isin(matured_at_start), "otif_miss"].mean()
    )

    daily_records: list[DailyRecord] = []
    labels_since_retrain = 0
    last_retrain_day = initial_cutoff
    max_available_days = max(1, total_days - int(total_days * config.initial_training_fraction))
    replay_days = min(config.replay_days, max_available_days)

    for offset in range(1, replay_days + 1):
        today = initial_cutoff + pd.Timedelta(days=offset)
        if today > horizon_end:
            break

        open_mask = (dataset.orders["order_date"] <= today) & (
            outcomes.set_index("order_id")["outcome_timestamp"].reindex(dataset.orders["order_id"]).to_numpy()
            > np.datetime64(today)
        )
        open_order_ids = pd.Index(dataset.orders.loc[open_mask.to_numpy(), "order_id"])

        newly_closed_mask = (
            (outcomes["outcome_timestamp"] > today - pd.Timedelta(days=1))
            & (outcomes["outcome_timestamp"] <= today)
        )
        newly_closed_ids = outcomes.loc[newly_closed_mask, "order_id"]
        for order_id in newly_closed_ids:
            outcome_row = outcomes.loc[outcomes["order_id"] == order_id].iloc[0]
            cause_row = causes.loc[causes["order_id"] == order_id].iloc[0]
            append_feedback(
                feedback_path,
                order_id=order_id,
                feedback_action="ACCEPT",
                original_status="CLOSED",
                original_recommendation=(
                    f"actual_otif_miss={int(outcome_row['otif_miss'])} "
                    f"actual_cause={cause_row['primary_cause']}"
                ),
                actor="operations-replay",
                timestamp=today.to_pydatetime(),
            )
        labels_since_retrain += len(newly_closed_ids)

        record = DailyRecord(
            simulated_day=today.isoformat(),
            open_orders=len(open_order_ids),
            newly_closed_orders=len(newly_closed_ids),
            recommended=0,
            contested=0,
            monitor=0,
            model_version=registry[-1].version,
        )

        if len(open_order_ids) > 0:
            features_today = build_feature_table(
                dataset, outcomes, causes, as_of_timestamp=today, order_ids=open_order_ids
            )
            features_today = attach_line_evidence_features(dataset, features_today)
            scored_today = score_orders(
                dataset,
                features_today,
                trained.risk_training.bundle,
                trained.bayesian_bundle,
                trained.fusion_selection.chosen_weight,
                background=baseline_features,
            )
            candidates = _daily_candidate_frame(scored_today, trained.fused_threshold)
            actionable = candidates.loc[candidates["is_candidate"]].copy()
            capacities = default_daily_capacities(dataset)
            if len(actionable):
                allocated, _remaining = allocate_interventions(actionable, capacities)
                allocated_columns = [
                    "order_id",
                    "decision_status",
                    "resource_type",
                    "resource_id",
                    "contested_with",
                ]
                candidates = candidates.merge(
                    allocated[allocated_columns], on="order_id", how="left"
                )
            else:
                candidates["decision_status"] = "MONITOR"
                candidates["resource_type"] = ""
                candidates["resource_id"] = ""
                candidates["contested_with"] = ""
            candidates["decision_status"] = candidates["decision_status"].fillna("MONITOR")
            candidates.to_csv(queue_dir / f"{today.date().isoformat()}.csv", index=False)

            record.recommended = int((candidates["decision_status"] == "RECOMMENDED").sum())
            record.contested = int((candidates["decision_status"] == "CONTESTED").sum())
            record.monitor = int((candidates["decision_status"] == "MONITOR").sum())

            # Drift comparison population: restrict to orders that have
            # reached (roughly) their normal scoring horizon, so "today's"
            # population is comparable to the training baseline (which was
            # built from orders scored at their own prediction_timestamp).
            # Without this, a batch of freshly-captured open orders (0-1
            # days old) would look artificially "different" purely because
            # they haven't had time to accumulate observable events yet.
            horizon_days = config.data_config.prediction_horizon_days
            matured_enough = features_today["days_since_order"] >= horizon_days
            drift_features = features_today.loc[matured_enough]
            drift_scores = candidates.loc[
                matured_enough.to_numpy(), "combined_risk_score"
            ].to_numpy()
            lookback_start = today - pd.Timedelta(days=config.drift_lookback_days)
            recent_otif = outcomes.loc[
                outcomes["outcome_timestamp"].between(lookback_start, today),
                "otif_miss",
            ]
            if len(drift_features) >= 5:
                current_missingness = float(
                    drift_features["missing_event_stage_count"].mean() / 3
                )
                current_otif_rate = (
                    float(recent_otif.mean()) if len(recent_otif) else baseline_otif_rate
                )
                drift_report = evaluate_drift(
                    baseline_features=baseline_features,
                    current_features=drift_features,
                    baseline_scores=baseline_scores,
                    current_scores=drift_scores,
                    baseline_missingness=baseline_missingness,
                    current_missingness=current_missingness,
                    baseline_otif_rate=baseline_otif_rate,
                    current_otif_rate=current_otif_rate,
                    recent_otif_observations=len(recent_otif),
                )
            else:
                drift_report = None
            record.drift = drift_report.to_dict() if drift_report is not None else {}
        else:
            drift_report = None

        days_since_retrain = (today - last_retrain_day).days
        cadence_due = days_since_retrain >= config.retrain_cadence_days
        drift_triggered = bool(drift_report and drift_report.triggered)
        enough_labels = labels_since_retrain >= config.min_new_labels_for_retrain
        cooldown_elapsed = days_since_retrain >= config.min_days_between_retrains
        if (cadence_due or drift_triggered) and enough_labels and cooldown_elapsed:
            trigger = "scheduled" if cadence_due else "drift"
            trigger_reasons = list(drift_report.reasons) if drift_report else []
            if cadence_due:
                trigger_reasons.append(
                    f"scheduled cadence {config.retrain_cadence_days} days reached"
                )
            matured_ids = outcomes.loc[
                (outcomes["prediction_timestamp"] <= today)
                & (outcomes["outcome_timestamp"] <= today),
                "order_id",
            ]
            trained, baseline_features = _retrain(
                dataset, outcomes, causes, pd.Index(matured_ids), config.data_config
            )
            version = registry[-1].version + 1
            joblib.dump(
                trained.risk_training.bundle, model_dir / f"v{version}_xgboost_risk.joblib"
            )
            joblib.dump(
                trained.bayesian_bundle, model_dir / f"v{version}_bayesian_network.joblib"
            )
            registry.append(
                ModelRegistryEntry(
                    version=version,
                    trained_at_simulated_day=today.isoformat(),
                    training_window_start=horizon_start.isoformat(),
                    training_window_end=today.isoformat(),
                    n_training_orders=len(matured_ids),
                    validation_metrics=trained.validation_metrics,
                    threshold=trained.fused_threshold,
                    fusion_weight=trained.fusion_selection.chosen_weight,
                    fusion_label=trained.fusion_selection.chosen_label,
                    trigger=trigger,
                    trigger_reasons=trigger_reasons,
                    artifact_path=f"models/v{version}",
                )
            )
            baseline_scores = trained.risk_training.bundle.predict_proba(baseline_features)
            baseline_missingness = float(
                baseline_features["missing_event_stage_count"].mean() / 3
            )
            baseline_otif_rate = float(
                outcomes.loc[outcomes["order_id"].isin(matured_ids), "otif_miss"].mean()
            )
            last_retrain_day = today
            labels_since_retrain = 0
            record.retrained = True
            record.retrain_trigger = trigger
            record.model_version = version

        daily_records.append(record)

    registry_payload = [entry.to_dict() for entry in registry]
    (run_dir / "model_registry.json").write_text(
        json.dumps(registry_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(registry_payload).to_csv(run_dir / "model_registry.csv", index=False)

    daily_payload = [asdict(record) for record in daily_records]
    (run_dir / "daily_log.json").write_text(
        json.dumps(daily_payload, indent=2, default=str), encoding="utf-8"
    )

    summary = {
        "run_directory": run_dir.name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "initial_cutoff": initial_cutoff.isoformat(),
        "replay_days_completed": len(daily_records),
        "model_versions_trained": len(registry),
        "retrain_events": [
            {
                "day": entry.trained_at_simulated_day,
                "trigger": entry.trigger,
                "reasons": entry.trigger_reasons,
            }
            for entry in registry[1:]
        ],
        "drift_warning_days": [
            record.simulated_day for record in daily_records if record.drift.get("triggered")
        ],
        "final_model_version": registry[-1].version,
        "final_threshold": registry[-1].threshold,
        "final_fusion_weight": registry[-1].fusion_weight,
    }
    (run_dir / "operations_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "summary": summary,
        "registry": registry_payload,
        "daily_log": daily_payload,
        "run_dir": run_dir,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--replay-days", type=int, default=DEFAULT_REPLAY_DAYS)
    parser.add_argument("--retrain-cadence-days", type=int, default=DEFAULT_RETRAIN_CADENCE_DAYS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ops_config = OperationsConfig(
        data_config=PrototypeConfig(
            seed=args.seed, n_orders=args.orders, output_dir=args.output_dir
        ),
        output_dir=args.output_dir,
        replay_days=args.replay_days,
        retrain_cadence_days=args.retrain_cadence_days,
    )
    result = run_operations_replay(ops_config)
    summary = result["summary"]
    print(
        f"replay_days={summary['replay_days_completed']} "
        f"model_versions={summary['model_versions_trained']} "
        f"retrain_events={len(summary['retrain_events'])} "
        f"drift_warning_days={len(summary['drift_warning_days'])} "
        f"run_dir={summary['run_directory']}"
    )


if __name__ == "__main__":
    main()
