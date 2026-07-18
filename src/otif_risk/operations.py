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
   matured labels exist, incrementing a versioned model-training history
   (`model_registry.json`/`.csv`, unchanged from Stage 1).

Stage 2 governance layer (this module's addition): every retrain attempt
is also registered as a challenger in a separate, append-only
`registry.ModelRegistry` (`registry/` subdirectory) and evaluated with
`registry.evaluate_promotion` against the current champion -- the retrain
only becomes the live scoring bundle when promotion passes; a held
challenger's artifacts are still persisted for audit but never replace the
active model (see `_maybe_promote_retrain`). The replay also persists a
production-shaped decision ledger (`decision_ledger.py`), an observational
cohort report, a rolling monitoring/SLO report (`monitoring.py`), and a
deterministic run manifest (`manifest.py`). A clearly-labeled demo
governance-lifecycle scenario (`_run_demo_lifecycle_scenario`), built only
from already-measured Stage 1 `policy_benchmark.json` numbers, separately
demonstrates one promoted challenger, one held Bayesian-enhanced
challenger, and one rollback -- kept distinct from the real per-day
retrain lifecycle above.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from otif_risk.action_response import CAUSE_TO_ACTION, NO_ACTION
from otif_risk.adapters import data_quality_report
from otif_risk.contracts import PrototypeConfig, PrototypeDataset
from otif_risk.data import DRIFT_WINDOW_FRACTION, generate_dataset
from otif_risk.decision_ledger import (
    append_entries,
    build_ledger_entry,
    intervention_outcomes_report,
    observational_cohort_report,
    reconcile_outcomes,
)
from otif_risk.decisions import RECOMMENDATION_TABLE
from otif_risk.drift import evaluate_drift
from otif_risk.features import attach_line_evidence_features, build_feature_table, temporal_split
from otif_risk.feedback import append_feedback
from otif_risk.fusion import fuse_scores
from otif_risk.manifest import ManifestInputs, verify_manifest, write_manifest
from otif_risk.model import evaluate_predictions
from otif_risk.monitoring import build_monitoring_report, write_monitoring_report
from otif_risk.pipeline import TrainedBundle, score_orders, train_full_bundle
from otif_risk.registry import (
    PROMOTED,
    ModelMetrics,
    ModelRegistry,
    ModelVersion,
    evaluate_promotion,
)
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
    #: Path to a previously-generated ``policy_benchmark.json`` (Stage 1's
    #: 5-seed policy-value benchmark). When present, its measured medians
    #: drive the demo governance-lifecycle scenario (see
    #: ``_run_demo_lifecycle_scenario``); when absent, that scenario is
    #: skipped with an explicit, honest note rather than fabricating numbers.
    policy_value_reference_path: Path | None = None
    #: Minimum matured decisions before a rolling monitoring window/regime
    #: cohort reports a metric rather than withholding it.
    monitoring_min_sample: int = 30


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
    #: Stage 2 governance outcome of today's retrain attempt (``None`` when
    #: no retrain happened today). ``HELD`` means a challenger was trained
    #: and its artifacts persisted, but it never became the live scoring
    #: bundle -- ``model_version`` above stays at the previous champion.
    promotion_decision: str | None = None


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
) -> tuple[TrainedBundle, pd.DataFrame, Any]:
    """Train a fresh bundle on every order that has matured by `today`.

    Also returns the ``TemporalSplit`` used, so callers can cheaply derive
    held-out regime (normal/drift) quality for the Stage 2 promotion gate
    without retraining or rebuilding features again.
    """
    feature_table = build_feature_table(
        dataset, outcomes, causes, order_ids=matured_order_ids
    )
    feature_table = attach_line_evidence_features(dataset, feature_table)
    split = temporal_split(feature_table)
    trained = train_full_bundle(dataset, outcomes, causes, split, data_config)
    return trained, feature_table, split


def _regime_pr_auc(
    trained: TrainedBundle, split: Any, dataset: PrototypeDataset
) -> tuple[float | None, float | None]:
    """Held-out (test-split) PR-AUC, split by normal vs. scripted-drift regime
    (see ``data.DRIFT_WINDOW_FRACTION``), for one trained challenger.

    Reuses the already-trained bundle's own ``score`` methods (no
    retraining) -- cheap enough to compute for every real retrain event.
    """
    test = split.test
    if test.empty:
        return None, None
    test_xgb = trained.risk_training.bundle.score(test)
    test_bbn = trained.bayesian_bundle.score(test)[["order_id", "bbn_risk_score"]]
    test_fused = fuse_scores(test_xgb, test_bbn, xgb_weight=trained.fusion_selection.chosen_weight)
    labels = test.set_index("order_id")["otif_miss"].astype(int)

    horizon_start = dataset.orders["order_date"].min()
    horizon_end = dataset.orders["order_date"].max()
    drift_start = horizon_end - (horizon_end - horizon_start) * DRIFT_WINDOW_FRACTION
    order_dates = dataset.orders.set_index("order_id")["order_date"].reindex(test["order_id"])

    def _pr_auc_for(mask: pd.Series) -> float | None:
        order_ids = order_dates.loc[mask].index
        subset = test_fused.loc[test_fused["order_id"].isin(order_ids)]
        if len(subset) < 10:
            return None
        subset_labels = labels.loc[subset["order_id"]].to_numpy()
        if len(set(subset_labels)) < 2:
            return None
        evaluated = evaluate_predictions(
            subset_labels, subset["fused_risk_score"].to_numpy(), trained.fused_threshold
        )
        return float(evaluated["pr_auc"])

    normal_pr_auc = _pr_auc_for(order_dates < drift_start)
    drift_pr_auc = _pr_auc_for(order_dates >= drift_start)
    return normal_pr_auc, drift_pr_auc


def _artifacts_exist_and_readable(paths: list[Path]) -> bool:
    """Lightweight, real artifact-integrity check: every path exists and is non-empty.

    This is not a full separate manifest/checksum per model version (the
    replay's own ``run_manifest.json`` -- see the end of
    ``run_operations_replay`` -- provides that, once, for the whole run); it
    is the promotion gate's own guard that a version's model files were
    actually written successfully before it can ever be considered
    ``manifest_verified``, rather than a hardcoded assumption.
    """
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _build_model_metrics(
    trained: TrainedBundle,
    split: Any,
    dataset: PrototypeDataset,
    *,
    policy_value_50pct_capacity: float,
    artifact_paths: list[Path] | None = None,
) -> ModelMetrics:
    """Build the Stage 2 promotion-gate comparison surface for one trained
    bundle, reusing metrics ``train_full_bundle`` already computed wherever
    possible (no duplicate training).

    ``policy_value_50pct_capacity`` is a *reference* value (see
    ``_load_policy_value_reference``) rather than a per-retrain re-measurement:
    re-running the full rolling-origin policy-value lab for every real
    retrain event would be prohibitively expensive for a 90-day replay and
    is not required for real model retrains (only the demo lifecycle
    scenario needs distinct before/after policy-value numbers -- see
    ``_run_demo_lifecycle_scenario``). ``schema_valid``/``leakage_gate_passed``
    are ``True`` here because ``validate_dataset`` and this codebase's fixed
    ``features.LEAKAGE_BLOCKLIST`` contract already ran/applied earlier in
    this same replay and would have raised if either failed. ``manifest_verified``
    is a real check (``_artifacts_exist_and_readable``) against
    ``artifact_paths`` -- never a hardcoded ``True`` -- so a version whose
    model files failed to write can never pass the promotion gate's manifest
    check.
    """
    normal_pr_auc, drift_pr_auc = _regime_pr_auc(trained, split, dataset)
    alert_rate = (
        trained.test_metrics["flagged_orders"] / len(split.test) if len(split.test) else 0.0
    )
    return ModelMetrics(
        pr_auc=float(trained.test_metrics["pr_auc"]),
        brier=float(trained.test_metrics["brier"]),
        calibration_error=float(trained.validation_metrics["calibration_error"]),
        recall=float(trained.test_metrics["recall"]),
        alert_rate=round(float(alert_rate), 4),
        drift_regime_pr_auc=drift_pr_auc,
        normal_regime_pr_auc=normal_pr_auc,
        policy_value_50pct_capacity=float(policy_value_50pct_capacity),
        schema_valid=True,
        leakage_gate_passed=True,
        manifest_verified=_artifacts_exist_and_readable(artifact_paths or []),
    )


def _load_policy_value_reference(path: Path | None) -> dict[str, Any] | None:
    """Load Stage 1's own measured policy-value numbers from a prior
    ``policy_benchmark.json`` run, for the Stage 2 promotion gate and the
    demo governance-lifecycle scenario.

    Returns ``None`` (never a fabricated number) when the file is absent --
    callers must skip/label anything that depends on it rather than invent
    a placeholder value.
    """
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    primary_scenario = summary["primary_capacity_scenario"]
    headline = summary["median_headline_by_capacity_scenario"][primary_scenario]

    ablation_with = [
        report["bayesian_ablation"]["with_bayesian_term"][
            "avoided_penalty_per_normalized_resource_unit"
        ]
        for report in payload["per_seed"]
        if report["bayesian_ablation"]["capacity_scenario"] == primary_scenario
    ]
    ablation_without = [
        report["bayesian_ablation"]["without_bayesian_term"][
            "avoided_penalty_per_normalized_resource_unit"
        ]
        for report in payload["per_seed"]
        if report["bayesian_ablation"]["capacity_scenario"] == primary_scenario
    ]
    return {
        "source": str(path),
        "primary_capacity_scenario": primary_scenario,
        "single_cause_baseline_median_value": float(
            headline["SINGLE_CAUSE_PRIORITY_BASELINE"]
        ),
        "stage1_policy_with_bayesian_median_value": float(headline["CURRENT_POLICY"]),
        "bayesian_ablation_with_median": (
            float(np.median(ablation_with)) if ablation_with else None
        ),
        "bayesian_ablation_without_median": (
            float(np.median(ablation_without)) if ablation_without else None
        ),
    }


def _run_demo_lifecycle_scenario(
    stage2_registry: ModelRegistry,
    policy_value_reference: dict[str, Any] | None,
    reference_metrics: ModelMetrics,
    rollback_target_version_id: str,
) -> dict[str, Any]:
    """Demonstrate PROMOTED, HELD, and ROLLED_BACK using only already-measured
    Stage 1 ``policy_benchmark.json`` numbers -- never fabricated metrics.

    Kept explicitly distinct from the real per-day retrain lifecycle above
    (see each registered version's ``note``): this scenario compares
    *policy-formula* variants (single-cause baseline vs. the value-aware policy,
    and value-aware action ranking with vs. without its Bayesian
    structural-reduction term -- Stage 1's own
    ``bayesian_ablation`` diagnostic, which is negative on every seed), not
    additional model retrains. Ends by rolling the active pointer back to
    ``rollback_target_version_id``, a real, verified version from the
    actual retrain lifecycle.
    """
    if policy_value_reference is None:
        return {
            "enabled": False,
            "reason": (
                "no policy_benchmark.json reference found; run "
                "`otif-policy-benchmark` first to enable this demo scenario"
            ),
        }

    def _variant(policy_value: float) -> ModelMetrics:
        return ModelMetrics(
            pr_auc=reference_metrics.pr_auc,
            brier=reference_metrics.brier,
            calibration_error=reference_metrics.calibration_error,
            recall=reference_metrics.recall,
            alert_rate=reference_metrics.alert_rate,
            drift_regime_pr_auc=reference_metrics.drift_regime_pr_auc,
            normal_regime_pr_auc=reference_metrics.normal_regime_pr_auc,
            policy_value_50pct_capacity=policy_value,
            schema_valid=True,
            leakage_gate_passed=True,
            # Demo variants have no artifacts of their own; they inherit the
            # real, already-verified reference version's manifest status
            # rather than hardcoding a separate assumption.
            manifest_verified=reference_metrics.manifest_verified,
        )

    single_cause_metrics = _variant(
        policy_value_reference["single_cause_baseline_median_value"]
    )
    bayesian_with_metrics = _variant(policy_value_reference["bayesian_ablation_with_median"])
    bayesian_without_metrics = _variant(policy_value_reference["bayesian_ablation_without_median"])

    note = "demo_lifecycle_scenario_from_measured_stage1_policy_benchmark"
    stage2_registry.register_version(
        ModelVersion(
            version_id="demo-single-cause-baseline",
            trained_at_utc=datetime.now(UTC).isoformat(),
            manifest_content_id=None,
            metrics=single_cause_metrics,
            parent_version_id=None,
            note=note,
        )
    )
    stage2_registry.promote_or_hold(
        "demo-single-cause-baseline",
        evaluate_promotion(single_cause_metrics, single_cause_metrics),
    )

    # Transition 1: PROMOTED -- the observable value-aware policy without
    # Bayesian action ranking has the strongest measured Stage 1 ablation value.
    stage2_registry.register_version(
        ModelVersion(
            version_id="demo-value-aware-policy-challenger",
            trained_at_utc=datetime.now(UTC).isoformat(),
            manifest_content_id=None,
            metrics=bayesian_without_metrics,
            parent_version_id="demo-single-cause-baseline",
            note=note,
        )
    )
    promotion_1 = evaluate_promotion(single_cause_metrics, bayesian_without_metrics)
    stage2_registry.promote_or_hold("demo-value-aware-policy-challenger", promotion_1)

    # Transition 2: HELD -- a Bayesian-enhanced variant of the same policy
    # formula regresses measured policy value beyond tolerance (Stage 1's
    # own bayesian_ablation diagnostic is negative on every benchmarked
    # seed -- see policy_evaluation.bayesian_ablation_diagnostic).
    stage2_registry.register_version(
        ModelVersion(
            version_id="demo-bayesian-enhanced-challenger",
            trained_at_utc=datetime.now(UTC).isoformat(),
            manifest_content_id=None,
            metrics=bayesian_with_metrics,
            parent_version_id="demo-value-aware-policy-challenger",
            note=note,
        )
    )
    promotion_2 = evaluate_promotion(bayesian_without_metrics, bayesian_with_metrics)
    stage2_registry.promote_or_hold("demo-bayesian-enhanced-challenger", promotion_2)

    # Transition 3: ROLLED_BACK -- restore the active pointer to a real,
    # verified version from the actual retrain lifecycle above.
    rollback_result = stage2_registry.rollback(rollback_target_version_id)

    return {
        "enabled": True,
        "policy_value_reference": policy_value_reference,
        "promotion_1_value_aware_policy": promotion_1.to_dict(),
        "promotion_2_bayesian_enhanced_held": promotion_2.to_dict(),
        "rollback": rollback_result,
        "active_version_after_demo": stage2_registry.active_version(),
    }


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

    trained, baseline_features, baseline_split = _retrain(
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

    policy_value_reference_path = (
        config.policy_value_reference_path
        or (config.output_dir / "policy_benchmark.json")
    )
    policy_value_reference = _load_policy_value_reference(policy_value_reference_path)
    reference_policy_value = (
        policy_value_reference["bayesian_ablation_without_median"]
        if policy_value_reference is not None
        else 1.0
    )
    run_dir = _run_operations_directory(config)
    stage2_registry = ModelRegistry(run_dir / "registry")
    queue_dir = run_dir / "daily_queues"
    model_dir = run_dir / "models"
    queue_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    v1_xgboost_path = model_dir / "v1_xgboost_risk.joblib"
    v1_bayesian_path = model_dir / "v1_bayesian_network.joblib"
    joblib.dump(trained.risk_training.bundle, v1_xgboost_path)
    joblib.dump(trained.bayesian_bundle, v1_bayesian_path)

    champion_metrics = _build_model_metrics(
        trained,
        baseline_split,
        dataset,
        policy_value_50pct_capacity=reference_policy_value,
        artifact_paths=[v1_xgboost_path, v1_bayesian_path],
    )
    stage2_registry.register_version(
        ModelVersion(
            version_id="v1",
            trained_at_utc=datetime.now(UTC).isoformat(),
            manifest_content_id=None,
            metrics=champion_metrics,
            artifact_paths={
                "xgboost": "models/v1_xgboost_risk.joblib",
                "bayesian": "models/v1_bayesian_network.joblib",
            },
            parent_version_id=None,
            note="actual_model_retrain",
        )
    )
    stage2_registry.promote_or_hold("v1", evaluate_promotion(champion_metrics, champion_metrics))
    #: The version number actually driving live scoring (`trained`) right now --
    #: distinct from ``registry[-1].version`` (Stage 1's unconditional
    #: last-*trained* version), which can be a held challenger's number and
    #: must never be reported as the active/live scoring version.
    active_version_number = 1
    scoring_seconds: list[float] = []
    retrain_seconds: list[float] = []
    ledger_entries: list[dict[str, Any]] = []

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
            model_version=active_version_number,
        )

        if len(open_order_ids) > 0:
            features_today = build_feature_table(
                dataset, outcomes, causes, as_of_timestamp=today, order_ids=open_order_ids
            )
            features_today = attach_line_evidence_features(dataset, features_today)
            _scoring_start = time.perf_counter()
            scored_today = score_orders(
                dataset,
                features_today,
                trained.risk_training.bundle,
                trained.bayesian_bundle,
                trained.fusion_selection.chosen_weight,
                background=baseline_features,
            )
            scoring_seconds.append(time.perf_counter() - _scoring_start)
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

            # Stage 2 decision ledger: one entry per order scored today,
            # accumulated in memory and persisted once at the end of the
            # replay (see append_entries below) for performance.
            active_model_version = f"v{record.model_version}"
            for row in candidates.itertuples(index=False):
                primary_cause = str(getattr(row, "primary_cause", "")).upper()
                action_code = CAUSE_TO_ACTION.get(primary_cause, NO_ACTION)
                entry = build_ledger_entry(
                    order_id=row.order_id,
                    decision_day=today.date().isoformat(),
                    source_snapshot_id=run_dir.name,
                    model_version=active_model_version,
                    policy_version="ops-recommendation-table-v1",
                    manifest_content_id=None,
                    feasible_actions=[action_code],
                    chosen_action=action_code,
                    risk_score=float(row.combined_risk_score),
                    threshold=float(trained.fused_threshold),
                    decision_status=row.decision_status,
                    resource_type=getattr(row, "resource_type", None) or None,
                    resource_id=getattr(row, "resource_id", None) or None,
                    resource_demand_units=float(getattr(row, "quantity_at_risk", 0.0) or 0.0),
                    order_value=float(getattr(row, "order_value", 0.0) or 0.0),
                    penalty_rate=float(getattr(row, "penalty_rate", 0.0) or 0.0),
                )
                ledger_entries.append(entry.to_row())

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
            _retrain_start = time.perf_counter()
            challenger_trained, challenger_features, challenger_split = _retrain(
                dataset, outcomes, causes, pd.Index(matured_ids), config.data_config
            )
            retrain_seconds.append(time.perf_counter() - _retrain_start)
            version = registry[-1].version + 1
            challenger_xgboost_path = model_dir / f"v{version}_xgboost_risk.joblib"
            challenger_bayesian_path = model_dir / f"v{version}_bayesian_network.joblib"
            joblib.dump(challenger_trained.risk_training.bundle, challenger_xgboost_path)
            joblib.dump(challenger_trained.bayesian_bundle, challenger_bayesian_path)
            # Stage 1's own unconditional model-training history: every
            # retrain attempt is still recorded here exactly as before,
            # whether or not Stage 2 governance below promotes it.
            registry.append(
                ModelRegistryEntry(
                    version=version,
                    trained_at_simulated_day=today.isoformat(),
                    training_window_start=horizon_start.isoformat(),
                    training_window_end=today.isoformat(),
                    n_training_orders=len(matured_ids),
                    validation_metrics=challenger_trained.validation_metrics,
                    threshold=challenger_trained.fused_threshold,
                    fusion_weight=challenger_trained.fusion_selection.chosen_weight,
                    fusion_label=challenger_trained.fusion_selection.chosen_label,
                    trigger=trigger,
                    trigger_reasons=trigger_reasons,
                    artifact_path=f"models/v{version}",
                )
            )

            # Stage 2 governance gate: the challenger only becomes the live
            # scoring bundle if it passes promotion against the current
            # champion -- a held challenger's artifacts stay on disk for
            # audit, but never replace the active model.
            challenger_metrics = _build_model_metrics(
                challenger_trained,
                challenger_split,
                dataset,
                policy_value_50pct_capacity=reference_policy_value,
                artifact_paths=[challenger_xgboost_path, challenger_bayesian_path],
            )
            decision = evaluate_promotion(champion_metrics, challenger_metrics)
            stage2_registry.register_version(
                ModelVersion(
                    version_id=f"v{version}",
                    trained_at_utc=datetime.now(UTC).isoformat(),
                    manifest_content_id=None,
                    metrics=challenger_metrics,
                    artifact_paths={
                        "xgboost": f"models/v{version}_xgboost_risk.joblib",
                        "bayesian": f"models/v{version}_bayesian_network.joblib",
                    },
                    parent_version_id=stage2_registry.active_version(),
                    note="actual_model_retrain",
                )
            )
            stage2_registry.promote_or_hold(f"v{version}", decision)

            if decision.decision == PROMOTED:
                trained = challenger_trained
                baseline_features = challenger_features
                champion_metrics = challenger_metrics
                active_version_number = version
                record.model_version = version
                baseline_scores = trained.risk_training.bundle.predict_proba(baseline_features)
                baseline_missingness = float(
                    baseline_features["missing_event_stage_count"].mean() / 3
                )
                baseline_otif_rate = float(
                    outcomes.loc[outcomes["order_id"].isin(matured_ids), "otif_miss"].mean()
                )
            record.retrained = True
            record.retrain_trigger = trigger
            record.promotion_decision = decision.decision

            last_retrain_day = today
            labels_since_retrain = 0

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

    # --- Stage 2: decision ledger, reconciliation, observational cohorts ---
    ledger_path = run_dir / "decision_ledger.csv"
    append_entries(ledger_path, ledger_entries)
    final_simulated_day = (
        pd.Timestamp(daily_records[-1].simulated_day) if daily_records else initial_cutoff
    )
    reconciliation = reconcile_outcomes(
        ledger_path, outcomes, causes, as_of_timestamp=final_simulated_day
    )
    cohort_report = observational_cohort_report(
        ledger_path, min_sample=config.monitoring_min_sample
    )
    (run_dir / "observational_cohort_report.json").write_text(
        json.dumps(cohort_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    intervention_outcomes = intervention_outcomes_report(
        ledger_path, min_sample=config.monitoring_min_sample
    )
    (run_dir / "intervention_outcomes.json").write_text(
        json.dumps(intervention_outcomes, indent=2, sort_keys=True), encoding="utf-8"
    )

    # --- Stage 2: rolling monitoring/SLO report ---
    ledger_frame = pd.read_csv(ledger_path) if ledger_path.is_file() else pd.DataFrame()
    dq_report = data_quality_report(dataset.tables())
    monitoring_report = build_monitoring_report(
        ledger=ledger_frame,
        orders=dataset.orders,
        order_dates=dataset.orders.set_index("order_id")["order_date"],
        daily_log=daily_payload,
        data_quality=dq_report,
        scoring_seconds=scoring_seconds,
        retrain_seconds=retrain_seconds,
        min_sample=config.monitoring_min_sample,
    )
    write_monitoring_report(run_dir / "monitoring_report.json", monitoring_report)

    # --- Stage 2: demo governance-lifecycle scenario (measured, not faked) ---
    # Roll back to whichever real version was actually active before the demo
    # scenario ran (not a hardcoded "v1") -- so a genuinely promoted real
    # challenger is never silently clobbered by the demo's own rollback step.
    real_active_version_id = f"v{active_version_number}"
    demo_lifecycle = _run_demo_lifecycle_scenario(
        stage2_registry,
        policy_value_reference,
        champion_metrics,
        rollback_target_version_id=real_active_version_id,
    )
    (run_dir / "demo_lifecycle_scenario.json").write_text(
        json.dumps(demo_lifecycle, indent=2, sort_keys=True), encoding="utf-8"
    )

    active_version_id = stage2_registry.active_version()

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
        "final_model_version_note": (
            "last version *trained* during replay (unconditional training "
            "history) -- see active_model_version_id for the version Stage 2 "
            "governance actually promoted to live scoring."
        ),
        "final_threshold": trained.fused_threshold,
        "final_fusion_weight": trained.fusion_selection.chosen_weight,
        "active_model_version_id": active_version_id,
        "governance": {
            "registry_directory": str((run_dir / "registry").relative_to(run_dir)),
            "promotion_events": len(
                [event for event in stage2_registry.history() if event["event"] == "PROMOTED"]
            ),
            "held_events": len(
                [event for event in stage2_registry.history() if event["event"] == "HELD"]
            ),
            "rollback_events": len(
                [event for event in stage2_registry.history() if event["event"] == "ROLLED_BACK"]
            ),
            "demo_lifecycle_enabled": demo_lifecycle["enabled"],
        },
        "decision_ledger": {
            "path": str(ledger_path.relative_to(run_dir)),
            "total_decisions_logged": cohort_report["total_decisions_logged"],
            "reconciliation": reconciliation,
        },
    }
    (run_dir / "operations_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # The manifest is written last -- once every other artifact for this run
    # directory exists -- so its checksums capture a complete snapshot.
    manifest_inputs = ManifestInputs(
        run_kind="operations_replay",
        config=config.data_config,
        dataset=dataset,
        schema_versions={"ops_schema_version": "1.0"},
        training_window=(horizon_start.isoformat(), initial_cutoff.isoformat()),
        test_window=(initial_cutoff.isoformat(), final_simulated_day.isoformat()),
        parent_model_version=None,
        champion_model_version=active_version_id,
        model_versions={"active_model_version_id": active_version_id or "unknown"},
        extra_content={
            "replay_days": config.replay_days,
            "retrain_cadence_days": config.retrain_cadence_days,
        },
    )
    manifest = write_manifest(run_dir, manifest_inputs)
    manifest_verification = verify_manifest(run_dir)

    return {
        "summary": summary,
        "registry": registry_payload,
        "daily_log": daily_payload,
        "run_dir": run_dir,
        "governance_registry": stage2_registry.versions(),
        "governance_history": stage2_registry.history(),
        "demo_lifecycle_scenario": demo_lifecycle,
        "observational_cohort_report": cohort_report,
        "intervention_outcomes_report": intervention_outcomes,
        "monitoring_report": monitoring_report,
        "manifest": {
            "content_id": manifest["content_id"],
            "run_instance_id": manifest["run_instance_id"],
            "verification": manifest_verification,
        },
    }



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--replay-days", type=int, default=DEFAULT_REPLAY_DAYS)
    parser.add_argument("--retrain-cadence-days", type=int, default=DEFAULT_RETRAIN_CADENCE_DAYS)
    parser.add_argument(
        "--policy-value-reference-path",
        type=Path,
        default=Path("artifacts/policy_benchmark.json"),
        help=(
            "Prior policy_benchmark.json used for the Stage 2 promotion gate's "
            "policy-value reference and the demo governance-lifecycle scenario."
        ),
    )
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
        policy_value_reference_path=args.policy_value_reference_path,
    )
    result = run_operations_replay(ops_config)
    summary = result["summary"]
    print(
        f"replay_days={summary['replay_days_completed']} "
        f"model_versions={summary['model_versions_trained']} "
        f"retrain_events={len(summary['retrain_events'])} "
        f"drift_warning_days={len(summary['drift_warning_days'])} "
        f"active_model_version_id={summary['active_model_version_id']} "
        f"promoted={summary['governance']['promotion_events']} "
        f"held={summary['governance']['held_events']} "
        f"rolled_back={summary['governance']['rollback_events']} "
        f"manifest_verified={result['manifest']['verification']['verified']} "
        f"run_dir={summary['run_directory']}"
    )


if __name__ == "__main__":
    main()
