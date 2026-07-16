"""End-to-end OTIF risk intelligence pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from otif_risk.bayesian import CHAIN_PARENTS, MECHANISM_NODES, BayesianBundle, fit_bayesian_network
from otif_risk.contracts import CAUSE_CATEGORIES, PrototypeConfig, PrototypeDataset
from otif_risk.data import generate_dataset
from otif_risk.decisions import (
    build_rollups,
    recommend_orders,
    service_impact_summary,
)
from otif_risk.evaluation import (
    causal_consistency_report,
    cause_fidelity_report,
    confidence_diagnostics,
    evaluate_at_threshold,
    mechanism_metrics,
    prevalence_baseline_metrics,
    score_space_metrics,
    simulator_responsive_causes,
)
from otif_risk.explain import explain_predictions
from otif_risk.features import (
    TemporalSplit,
    attach_line_evidence_features,
    build_feature_table,
    temporal_split,
)
from otif_risk.feedback import FEEDBACK_COLUMNS
from otif_risk.fusion import FusionBundle, FusionSelection, fuse_scores, select_fusion_weight
from otif_risk.line_evidence import (
    affected_sku_summary,
    build_line_evidence,
    evaluate_line_evidence,
)
from otif_risk.model import RiskBundle, TrainingResult, train_risk_model
from otif_risk.narratives import order_narrative
from otif_risk.root_causes import calculate_outcomes, derive_root_causes
from otif_risk.validation import validate_dataset

#: Increment when persisted artifact columns or semantics change.
ARTIFACT_SCHEMA_VERSION = "3.0"


def _package_version() -> str:
    try:
        return version("otif-risk-intelligence")
    except PackageNotFoundError:  # pragma: no cover - editable/local checkouts
        return "0.0.0+local"


def _run_directory(config: PrototypeConfig) -> Path:
    """Return a run directory for ``config`` without overwriting a prior run."""
    values = asdict(config)
    values["output_dir"] = str(values["output_dir"])
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:10]
    base = config.output_dir / f"run-{digest}"
    if not base.exists():
        return base
    suffix = 2
    while (config.output_dir / f"run-{digest}-{suffix}").exists():
        suffix += 1
    return config.output_dir / f"run-{digest}-{suffix}"


def _probable_cause(row: pd.Series) -> str:
    active = [
        category
        for category in CAUSE_CATEGORIES
        if int(row.get(f"leading_signal_{category}", 0)) == 1
    ]
    if not active:
        return "UNKNOWN"
    return active[0]


def _top_attribution_cause(value: Any) -> str | None:
    """Parse `causal_attribution_json` and return its top |contribution| node."""
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    top = parsed[0]
    return str(top["node"]) if isinstance(top, dict) and "node" in top else None


def _top_intervention_cause(value: Any) -> str | None:
    """Parse `intervention_scenarios_json` and return the single-node scenario

    with the largest absolute risk reduction (the combined-mitigation scenario
    is excluded, since it is not attributable to one node).
    """
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    single_node = [
        item
        for item in parsed
        if isinstance(item, dict) and item.get("type") == "single_node_mitigation"
    ]
    if not single_node:
        return None
    best = max(single_node, key=lambda item: item.get("absolute_risk_reduction", 0.0))
    nodes = best.get("intervened_nodes") or []
    return str(nodes[0]) if nodes else None


def _enrich_business_context(scored: pd.DataFrame, dataset: PrototypeDataset) -> pd.DataFrame:
    lines_with_value = dataset.order_lines.merge(
        dataset.skus[["sku_id", "base_unit_value"]],
        on="sku_id",
        how="left",
        validate="many_to_one",
    )
    requested_qty = lines_with_value["requested_qty"].astype(float)
    unit_value = lines_with_value["base_unit_value"].fillna(50.0)
    lines_with_value["line_value"] = requested_qty * unit_value
    line_context = lines_with_value.groupby("order_id", as_index=False).agg(
        order_value=("line_value", "sum"),
        representative_sku=("sku_id", "first"),
    )
    enriched = scored.merge(line_context, on="order_id", how="left", validate="one_to_one")
    customer_number = enriched["customer_id"].astype(str).str.extract(r"(\d+)", expand=False)
    customer_number = pd.to_numeric(customer_number, errors="coerce").fillna(0).astype(int)
    enriched["customer_tier"] = customer_number.mod(4).map(
        {0: "PLATINUM", 1: "GOLD", 2: "SILVER", 3: "BRONZE"}
    )
    enriched["penalty_rate"] = enriched["customer_tier"].map(
        {"PLATINUM": 0.05, "GOLD": 0.03, "SILVER": 0.02, "BRONZE": 0.01}
    )
    return enriched


def bayesian_training_history(
    causes: pd.DataFrame,
    outcomes: pd.DataFrame,
    train_order_ids: set[str],
) -> pd.DataFrame:
    """Restrict Bayesian fitting evidence to the training split's order IDs only.

    Carries ``on_time``/``in_full`` alongside ``otif_miss`` so the mechanism
    nodes (``IN_FULL_FAILURE = 1 - in_full``, ``LATE_DELIVERY = 1 - on_time``)
    are fit directly from this split's own resolved outcomes, not inferred
    from the seven-category failure-only cause labels.
    """
    history = causes[
        ["order_id", *(f"stage_{category}" for category in CAUSE_CATEGORIES)]
    ].merge(
        outcomes[["order_id", "otif_miss", "on_time", "in_full"]],
        on="order_id",
        validate="one_to_one",
    )
    return history.loc[history["order_id"].isin(train_order_ids)].reset_index(drop=True)


#: Kept as a private alias for backward compatibility with earlier internal callers/tests.
_bayesian_training_history = bayesian_training_history


@dataclass
class TrainedBundle:
    """Everything produced by one end-to-end training cycle."""

    risk_training: TrainingResult
    bayesian_bundle: BayesianBundle
    fusion_selection: FusionSelection
    fused_threshold: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    xgb_test_metrics: dict[str, Any]
    bbn_test_metrics: dict[str, Any]
    prevalence_metrics: dict[str, Any]
    cause_fidelity: dict[str, Any]
    mechanism_metrics: dict[str, Any]
    confidence_diagnostics: dict[str, Any]
    causal_consistency: dict[str, Any]


def train_full_bundle(
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    split: TemporalSplit,
    config: PrototypeConfig,
) -> TrainedBundle:
    """Train the XGBoost model + Bayesian chain and select the fusion weight.

    Shared by the canonical single pipeline run and the operations replay's
    versioned retraining, so both use identical evaluation/selection logic.
    """
    risk_training = train_risk_model(
        split.train,
        split.validation,
        split.test,
        planner_capacity_fraction=config.planner_capacity_fraction,
        threshold_strategy=config.threshold_strategy,  # type: ignore[arg-type]
        target_recall=config.target_recall,
        min_precision=config.min_precision,
        random_state=config.seed,
    )

    train_order_ids = set(split.train["order_id"])
    bayesian_history = bayesian_training_history(causes, outcomes, train_order_ids)
    bayesian_bundle = fit_bayesian_network(bayesian_history)
    # Validates both bundles target the same endpoint (raises if not) even
    # though scores below are combined directly via `fuse_scores`.
    FusionBundle(risk_training.bundle, bayesian_bundle)

    validation_labels = split.validation.set_index("order_id")["otif_miss"].astype(int)
    test_labels = split.test.set_index("order_id")["otif_miss"].astype(int)

    validation_xgb = risk_training.bundle.score(split.validation)
    validation_bbn = bayesian_bundle.score(split.validation)[["order_id", "bbn_risk_score"]]
    test_xgb = risk_training.bundle.score(split.test)
    test_bbn_full = bayesian_bundle.score(split.test)
    test_bbn = test_bbn_full[["order_id", "bbn_risk_score"]]

    def _labels_for(frame: pd.DataFrame, labels: pd.Series) -> np.ndarray:
        return labels.loc[frame["order_id"]].to_numpy()

    validation_labels_array = _labels_for(validation_xgb, validation_labels)
    threshold_kwargs = {
        "strategy": config.threshold_strategy,
        "capacity_fraction": config.planner_capacity_fraction,
        "target_recall": config.target_recall,
        "min_precision": config.min_precision,
    }
    fusion_selection = select_fusion_weight(
        validation_labels_array,
        validation_xgb["risk_model_score"].to_numpy(),
        validation_bbn["bbn_risk_score"].to_numpy(),
        threshold_strategy=config.threshold_strategy,  # type: ignore[arg-type]
        capacity_fraction=config.planner_capacity_fraction,
        target_recall=config.target_recall,
        min_precision=config.min_precision,
    )
    chosen_row = fusion_selection.comparison.loc[
        fusion_selection.comparison["xgb_weight"] == fusion_selection.chosen_weight
    ].iloc[0]
    fused_threshold = float(chosen_row["threshold"])

    test_fused = fuse_scores(test_xgb, test_bbn, xgb_weight=fusion_selection.chosen_weight)

    fused_validation_metrics = {
        "pr_auc": float(chosen_row["pr_auc"]),
        "roc_auc": float(chosen_row["roc_auc"]),
        "precision": float(chosen_row["precision"]),
        "recall": float(chosen_row["recall"]),
        "f1": float(chosen_row["f1"]),
        "brier": float(chosen_row["brier"]),
        "threshold": fused_threshold,
        "calibration_error": float(chosen_row["calibration_error"]),
    }
    fused_test_metrics = evaluate_at_threshold(
        _labels_for(test_fused, test_labels), test_fused["fused_risk_score"], fused_threshold
    )

    xgb_selection = score_space_metrics(
        validation_labels_array, validation_xgb["risk_model_score"], **threshold_kwargs
    )
    bbn_selection = score_space_metrics(
        validation_labels_array, validation_bbn["bbn_risk_score"], **threshold_kwargs
    )
    xgb_test_metrics = evaluate_at_threshold(
        _labels_for(test_xgb, test_labels), test_xgb["risk_model_score"], xgb_selection["threshold"]
    )
    bbn_test_metrics = evaluate_at_threshold(
        _labels_for(test_bbn, test_labels), test_bbn["bbn_risk_score"], bbn_selection["threshold"]
    )
    prevalence_metrics = prevalence_baseline_metrics(_labels_for(test_fused, test_labels))

    test_truth_causes = causes.set_index("order_id").loc[test_fused["order_id"], "primary_cause"]
    missed_order_mask = test_labels.loc[test_fused["order_id"]].to_numpy() == 1
    predicted_cause = test_fused[["order_id"]].merge(
        split.test[["order_id", *(f"leading_signal_{c}" for c in CAUSE_CATEGORIES)]],
        on="order_id",
    )
    predicted_cause["primary_cause"] = predicted_cause.apply(_probable_cause, axis=1)
    cause_fidelity = cause_fidelity_report(
        predicted_cause.loc[missed_order_mask, "primary_cause"],
        test_truth_causes.loc[missed_order_mask],
    )

    test_bbn_indexed = test_bbn_full.set_index("order_id").reindex(test_fused["order_id"])
    test_outcomes = outcomes.set_index("order_id").reindex(test_fused["order_id"])
    mechanism = mechanism_metrics(
        late_delivery_truth=1 - test_outcomes["on_time"].astype(int).to_numpy(),
        late_delivery_probability=test_bbn_indexed["late_delivery_probability"].to_numpy(),
        in_full_failure_truth=1 - test_outcomes["in_full"].astype(int).to_numpy(),
        in_full_failure_probability=test_bbn_indexed["in_full_failure_probability"].to_numpy(),
    )
    confidence = confidence_diagnostics(
        test_bbn_indexed["evidence_coverage"], test_bbn_indexed["causal_confidence"]
    )
    responsive_causes = simulator_responsive_causes(
        outcomes, dataset.simulator_truth, dataset.orders
    ).reindex(test_fused["order_id"])
    comparisons = pd.DataFrame(
        {
            "top_attribution_cause": test_bbn_indexed["causal_attribution_json"].map(
                _top_attribution_cause
            ),
            "top_intervention_cause": test_bbn_indexed["intervention_scenarios_json"].map(
                _top_intervention_cause
            ),
            "rule_primary_cause": test_truth_causes.reindex(test_fused["order_id"]).to_numpy(),
            "simulator_responsive_cause": responsive_causes.to_numpy(),
        }
    )
    causal_consistency = causal_consistency_report(
        comparisons.loc[missed_order_mask].reset_index(drop=True)
    )

    return TrainedBundle(
        risk_training=risk_training,
        bayesian_bundle=bayesian_bundle,
        fusion_selection=fusion_selection,
        fused_threshold=fused_threshold,
        validation_metrics=fused_validation_metrics,
        test_metrics=fused_test_metrics,
        xgb_test_metrics=xgb_test_metrics,
        bbn_test_metrics=bbn_test_metrics,
        prevalence_metrics=prevalence_metrics,
        cause_fidelity=cause_fidelity,
        mechanism_metrics=mechanism,
        confidence_diagnostics=confidence,
        causal_consistency=causal_consistency,
    )


def score_orders(
    dataset: PrototypeDataset,
    features: pd.DataFrame,
    risk_bundle: RiskBundle,
    bayesian_bundle: BayesianBundle,
    xgb_weight: float,
    *,
    background: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score any feature-table slice with both models, fuse, and explain."""
    xgb_scores = risk_bundle.score(features)
    bbn_scores = bayesian_bundle.score(features)
    fused = fuse_scores(
        xgb_scores, bbn_scores[["order_id", "bbn_risk_score"]], xgb_weight=xgb_weight
    )
    explain_background = background if background is not None else features
    explanations = explain_predictions(
        risk_bundle, features, background=explain_background, top_n=4
    )
    bbn_extra_columns = [
        "order_id",
        "causal_pathway",
        "causal_attribution_json",
        "intervention_scenarios_json",
        "causal_confidence",
        "evidence_coverage",
        "late_delivery_probability",
        "in_full_failure_probability",
    ]
    scored = (
        features.drop(columns=["otif_miss"], errors="ignore")
        .merge(bbn_scores[bbn_extra_columns], on="order_id", validate="one_to_one")
        .merge(fused, on="order_id", validate="one_to_one")
        .merge(explanations, on="order_id", validate="one_to_one")
    )
    scored = scored.rename(
        columns={"risk_model_score": "xgb_risk_score", "fused_risk_score": "combined_risk_score"}
    )
    scored["primary_cause"] = scored.apply(_probable_cause, axis=1)
    scored = _enrich_business_context(scored, dataset)

    line_evidence = build_line_evidence(dataset, features)
    sku_summary = affected_sku_summary(line_evidence)
    scored = scored.merge(sku_summary, on="order_id", how="left", validate="one_to_one")
    scored["affected_sku_count"] = scored["affected_sku_count"].fillna(0).astype(int)
    scored["affected_skus_json"] = scored["affected_skus_json"].fillna("[]")
    return scored


def run_pipeline(config: PrototypeConfig) -> dict[str, Any]:
    """Generate data, train both risk layers, fuse scores, and write demo artifacts."""

    dataset = generate_dataset(config)
    validate_dataset(dataset)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    feature_table = build_feature_table(dataset, outcomes, causes)
    feature_table = attach_line_evidence_features(dataset, feature_table)
    split = temporal_split(feature_table)

    trained = train_full_bundle(dataset, outcomes, causes, split, config)

    scored = score_orders(
        dataset,
        split.test,
        trained.risk_training.bundle,
        trained.bayesian_bundle,
        trained.fusion_selection.chosen_weight,
        background=split.train,
    )

    line_evidence = build_line_evidence(dataset, split.test)
    test_order_ids = set(split.test["order_id"])
    scored_order_lines = line_evidence.loc[line_evidence["order_id"].isin(test_order_ids)]
    line_truth_test = dataset.line_truth.loc[dataset.line_truth["order_id"].isin(test_order_ids)]
    scored_order_lines_with_truth = scored_order_lines.merge(
        line_truth_test[["order_line_id", "truly_affected", "shortfall_ratio"]].rename(
            columns={
                "truly_affected": "truth_truly_affected",
                "shortfall_ratio": "truth_shortfall_ratio",
            }
        ),
        on="order_line_id",
        how="left",
    )
    line_evidence_eval = evaluate_line_evidence(scored_order_lines, line_truth_test)

    decisions = recommend_orders(scored, risk_threshold=trained.fused_threshold)
    decisions["narrative"] = decisions.apply(lambda row: order_narrative(row.to_dict()), axis=1)
    rollups = build_rollups(decisions, order_lines=dataset.order_lines)
    impact = service_impact_summary(decisions)

    run_dir = _run_directory(config)
    data_dir = run_dir / "data"
    model_dir = run_dir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, table in dataset.tables().items():
        table.to_csv(data_dir / f"{name}.csv", index=False)
    for name, table in dataset.truth_tables().items():
        table.to_csv(data_dir / f"{name}.csv", index=False)
    outcomes.to_csv(data_dir / "outcomes.csv", index=False)
    causes.to_csv(data_dir / "root_causes.csv", index=False)
    feature_table.to_csv(data_dir / "feature_table.csv", index=False)
    decisions.to_csv(data_dir / "scored_orders.csv", index=False)
    scored_order_lines_with_truth.to_csv(data_dir / "scored_order_lines.csv", index=False)
    for name, rollup in rollups.items():
        rollup.to_csv(data_dir / f"{name}_rollup.csv", index=False)
    trained.fusion_selection.comparison.to_csv(data_dir / "fusion_comparison.csv", index=False)
    pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(run_dir / "planner_feedback.csv", index=False)
    joblib.dump(trained.risk_training.bundle, model_dir / "xgboost_risk.joblib")
    joblib.dump(trained.bayesian_bundle, model_dir / "bayesian_network.joblib")

    model_scores = {
        "xgb": {"test_metrics": trained.xgb_test_metrics},
        "bbn": {"test_metrics": trained.bbn_test_metrics},
        "fused": {
            "validation_metrics": trained.validation_metrics,
            "test_metrics": trained.test_metrics,
            "threshold": trained.fused_threshold,
            "threshold_strategy": config.threshold_strategy,
        },
        "prevalence_baseline": trained.prevalence_metrics,
        "note": (
            "xgb/bbn test metrics use their own independently-selected validation "
            "threshold for standalone comparison only; only the fused threshold "
            "(chosen via the validation-selected fusion weight) drives decisions/UI."
        ),
    }

    report: dict[str, Any] = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "architecture": {
            "risk_model": trained.risk_training.bundle.model_kind,
            "risk_endpoint": trained.risk_training.bundle.endpoint,
            "bayesian_chain_edges": [
                f"{parent}->{node}" for node, parents in CHAIN_PARENTS.items() for parent in parents
            ],
            "bayesian_inference_mode": trained.bayesian_bundle.inference_mode,
            "bayesian_engine_build_error": trained.bayesian_bundle.engine_build_error,
            "mechanism_nodes": list(MECHANISM_NODES),
            "fusion": trained.fusion_selection.rationale,
            "fusion_chosen_weight": trained.fusion_selection.chosen_weight,
            "fusion_chosen_label": trained.fusion_selection.chosen_label,
            "explanation": "SHAP with local perturbation fallback",
            "endpoint_design_note": (
                "The predictive endpoint is binary OTIF miss risk. The 10-node "
                "Bayesian mechanism graph splits it into IN_FULL_FAILURE (quantity) "
                "and LATE_DELIVERY (timing) sub-mechanisms feeding OTIF_MISS, "
                "matching the actual OTIF definition; seven-category root-cause and "
                "mechanism-route pathway outputs are retained and evaluated "
                "separately (see cause_fidelity, mechanism_metrics)."
            ),
            "intervention_note": (
                "intervention_scenarios_json on every scored order reports exact "
                "structural do(node=0) scenarios under the fixed Bayesian network -- "
                "fixed-structure scenario analysis, not proven treatment effects. "
                "They are decision-support diagnostics only and never feed the "
                "XGBoost score or the operational recommend/contest decision."
            ),
            "vendor_fairness_note": (
                "vendor_rolling_fault_rate_* is conditioned on vendor_fault (only "
                "misses where the vendor was among the matched root causes), so a "
                "vendor is not penalized in its own rolling score for misses "
                "caused elsewhere (DC capacity, transport, customer scheduling)."
            ),
        },
        "data": {
            "orders": len(dataset.orders),
            "order_lines": len(dataset.order_lines),
            "events": len(dataset.events),
            "otif_miss_rate": float(outcomes["otif_miss"].mean()),
            "synthetic_data_note": (
                "This is a noisy, partially-observable digital twin: outcomes fall "
                "out of accumulated per-stage delay/shortfall (never pre-selected), "
                "with stable entity/SKU heterogeneity, seasonality, correlated "
                "vendor/DC/lane shocks, missing events, and measurement noise. See "
                "model_scores.prevalence_baseline and cause_fidelity for context."
            ),
        },
        "model_scores": model_scores,
        "fusion_comparison": trained.fusion_selection.comparison.to_dict(orient="records"),
        "cause_fidelity": trained.cause_fidelity,
        "mechanism_metrics": trained.mechanism_metrics,
        "causal_confidence_diagnostics": trained.confidence_diagnostics,
        "causal_consistency": trained.causal_consistency,
        "line_evidence": line_evidence_eval,
        "validation_metrics": trained.validation_metrics,
        "test_metrics": trained.test_metrics,
        "threshold": trained.fused_threshold,
        "threshold_strategy": config.threshold_strategy,
        "impact": impact,
        "provenance": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "package_version": _package_version(),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_directory": run_dir.name,
        },
        "schema": {
            "scored_orders_columns": list(decisions.columns),
            "feature_table_columns": list(feature_table.columns),
            "root_causes_columns": list(causes.columns),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_pipeline(
        PrototypeConfig(
            seed=args.seed,
            n_orders=args.orders,
            output_dir=args.output_dir,
        )
    )
    metrics = report["test_metrics"]
    print(
        f"model={report['architecture']['risk_model']} "
        f"fusion_weight={report['architecture']['fusion_chosen_weight']:.1f} "
        f"PR-AUC={metrics['pr_auc']:.3f} "
        f"recall={metrics['recall']:.3f} "
        f"precision={metrics['precision']:.3f} "
        f"threshold={report['threshold']:.3f}"
    )


if __name__ == "__main__":
    main()
