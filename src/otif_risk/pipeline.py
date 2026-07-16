"""End-to-end OTIF risk intelligence pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from otif_risk.bayesian import fit_bayesian_network
from otif_risk.contracts import CAUSE_CATEGORIES, PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.decisions import (
    build_rollups,
    recommend_orders,
    service_impact_summary,
)
from otif_risk.evaluation import (
    cause_fidelity_report,
    evaluate_at_threshold,
    prevalence_baseline_metrics,
    score_space_metrics,
)
from otif_risk.explain import explain_predictions
from otif_risk.features import build_feature_table, temporal_split
from otif_risk.feedback import FEEDBACK_COLUMNS
from otif_risk.fusion import FusionBundle, fuse_scores
from otif_risk.model import train_risk_model
from otif_risk.narratives import order_narrative
from otif_risk.root_causes import calculate_outcomes, derive_root_causes
from otif_risk.validation import validate_dataset

#: Bumped whenever a change alters the shape/semantics of persisted artifacts
#: (columns added/removed/renamed, decision/threshold semantics changed, etc.),
#: so downstream readers (and this README) can tell reruns apart from the
#: pre-remediation artifact generation.
ARTIFACT_SCHEMA_VERSION = "2.0"


def _package_version() -> str:
    try:
        return version("otif-risk-intelligence")
    except PackageNotFoundError:  # pragma: no cover - editable/local checkouts
        return "0.0.0+local"


def _run_directory(config: PrototypeConfig) -> Path:
    """Return a run directory for ``config`` without overwriting a prior run.

    The directory name is content-addressed on the configuration, so identical
    configurations are easy to recognize at a glance. If a run with that exact
    configuration already exists on disk (for example, regenerating the
    canonical run), a monotonically increasing numeric suffix is appended so
    both runs remain present and individually distinguishable; this function
    never deletes or overwrites existing artifacts.
    """
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


def _probable_cause(row: pd.Series, cause_lifts: dict[str, float]) -> str:
    active = [
        category
        for category in CAUSE_CATEGORIES
        if int(row.get(f"leading_signal_{category}", 0)) == 1
    ]
    if not active:
        return "UNKNOWN"
    return max(
        active,
        key=lambda category: cause_lifts.get(f"cause_{category}", 0.0),
    )


def _enrich_business_context(
    scored: pd.DataFrame,
    order_lines: pd.DataFrame,
) -> pd.DataFrame:
    line_context = (
        order_lines.assign(
            line_value=order_lines["requested_qty"].astype(float) * 100.0
        )
        .groupby("order_id", as_index=False)
        .agg(
            order_value=("line_value", "sum"),
            representative_sku=("sku_id", "first"),
        )
    )
    enriched = scored.merge(
        line_context, on="order_id", how="left", validate="one_to_one"
    )
    customer_number = (
        enriched["customer_id"].astype(str).str.extract(r"(\d+)", expand=False)
    )
    customer_number = pd.to_numeric(customer_number, errors="coerce").fillna(0).astype(int)
    enriched["customer_tier"] = customer_number.mod(4).map(
        {0: "PLATINUM", 1: "GOLD", 2: "SILVER", 3: "BRONZE"}
    )
    enriched["penalty_rate"] = enriched["customer_tier"].map(
        {"PLATINUM": 0.05, "GOLD": 0.03, "SILVER": 0.02, "BRONZE": 0.01}
    )
    return enriched


def _bayesian_training_history(
    causes: pd.DataFrame,
    outcomes: pd.DataFrame,
    train_order_ids: set[str],
) -> pd.DataFrame:
    """Restrict Bayesian fitting evidence to the training split's order IDs only.

    Fitting on the full dataset (including validation/test outcomes) would let
    the Bayesian network see resolved future-relative-to-scoring-time outcomes,
    breaking the same chronological boundary already enforced for the risk
    model. This mirrors that boundary for the Bayesian network (item 3's fix).
    """
    history = causes[
        ["order_id", *(f"cause_{category}" for category in CAUSE_CATEGORIES)]
    ].merge(outcomes[["order_id", "otif_miss"]], on="order_id", validate="one_to_one")
    return history.loc[history["order_id"].isin(train_order_ids)].reset_index(drop=True)


def run_pipeline(config: PrototypeConfig) -> dict[str, Any]:
    """Generate data, train both risk layers, fuse scores, and write demo artifacts."""

    dataset = generate_dataset(config)
    validate_dataset(dataset)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    feature_table = build_feature_table(dataset, outcomes, causes)
    split = temporal_split(feature_table)

    training = train_risk_model(
        split.train,
        split.validation,
        split.test,
        planner_capacity_fraction=config.planner_capacity_fraction,
        threshold_strategy=config.threshold_strategy,  # type: ignore[arg-type]
        target_recall=config.target_recall,
        min_precision=config.min_precision,
        random_state=config.seed,
    )

    # Bayesian fitting uses only the training split's resolved history — not
    # validation/test outcomes — matching the chronological boundary already
    # enforced for the risk model (item 3's fix).
    train_order_ids = set(split.train["order_id"])
    bayesian_history = _bayesian_training_history(causes, outcomes, train_order_ids)
    bayesian_bundle = fit_bayesian_network(bayesian_history)
    # Validates the risk model and Bayesian bundle target the same endpoint
    # even though scores below are combined directly with `fuse_scores` to
    # avoid redundant re-scoring through `FusionBundle.score`.
    FusionBundle(training.bundle, bayesian_bundle)

    # Score every split with all three score spaces (XGB, BBN, fused) so each
    # can be evaluated — and thresholded — entirely within its own probability
    # space. This is the fix for the threshold-selected-on-XGB-but-applied-to
    # -fused defect: the fused threshold below is selected on fused validation
    # scores and is the only threshold used for decisions/UI.
    validation_labels = split.validation.set_index("order_id")["otif_miss"].astype(int)
    test_labels = split.test.set_index("order_id")["otif_miss"].astype(int)

    validation_xgb = training.bundle.score(split.validation)
    validation_bbn = bayesian_bundle.score(split.validation)[["order_id", "bbn_risk_score"]]
    validation_fused = fuse_scores(validation_xgb, validation_bbn)

    test_xgb = training.bundle.score(split.test)
    test_bbn_full = bayesian_bundle.score(split.test)
    test_bbn = test_bbn_full[["order_id", "bbn_risk_score"]]
    test_fused = fuse_scores(test_xgb, test_bbn)

    def _labels_for(frame: pd.DataFrame, labels: pd.Series) -> Any:
        return labels.loc[frame["order_id"]].to_numpy()

    threshold_kwargs = {
        "strategy": config.threshold_strategy,
        "capacity_fraction": config.planner_capacity_fraction,
        "target_recall": config.target_recall,
        "min_precision": config.min_precision,
    }
    xgb_selection = score_space_metrics(
        _labels_for(validation_xgb, validation_labels),
        validation_xgb["risk_model_score"],
        **threshold_kwargs,
    )
    bbn_selection = score_space_metrics(
        _labels_for(validation_bbn, validation_labels),
        validation_bbn["bbn_risk_score"],
        **threshold_kwargs,
    )
    fused_selection = score_space_metrics(
        _labels_for(validation_fused, validation_labels),
        validation_fused["fused_risk_score"],
        **threshold_kwargs,
    )

    xgb_test_metrics = evaluate_at_threshold(
        _labels_for(test_xgb, test_labels), test_xgb["risk_model_score"], xgb_selection["threshold"]
    )
    bbn_test_metrics = evaluate_at_threshold(
        _labels_for(test_bbn, test_labels), test_bbn["bbn_risk_score"], bbn_selection["threshold"]
    )
    fused_test_metrics = evaluate_at_threshold(
        _labels_for(test_fused, test_labels),
        test_fused["fused_risk_score"],
        fused_selection["threshold"],
    )
    prevalence_metrics = prevalence_baseline_metrics(_labels_for(test_fused, test_labels))

    explanations = explain_predictions(
        training.bundle,
        split.test,
        background=split.train,
        top_n=4,
    )
    scored = (
        split.test.drop(columns=["otif_miss"])
        .merge(
            test_bbn_full[["order_id", "causal_pathway"]],
            on="order_id",
            validate="one_to_one",
        )
        .merge(test_fused, on="order_id", validate="one_to_one")
        .merge(explanations, on="order_id", validate="one_to_one")
    )
    scored = scored.rename(
        columns={
            "risk_model_score": "xgb_risk_score",
            "fused_risk_score": "combined_risk_score",
        }
    )
    scored["primary_cause"] = scored.apply(
        _probable_cause,
        axis=1,
        cause_lifts=bayesian_bundle.cause_lifts,
    )
    scored = _enrich_business_context(scored, dataset.order_lines)

    test_truth_causes = causes.set_index("order_id").loc[scored["order_id"], "primary_cause"]
    missed_order_mask = test_labels.loc[scored["order_id"]].to_numpy() == 1
    cause_fidelity = cause_fidelity_report(
        scored.loc[missed_order_mask, "primary_cause"],
        test_truth_causes.loc[missed_order_mask],
    )

    decisions = recommend_orders(
        scored,
        risk_threshold=fused_selection["threshold"],
    )
    decisions["narrative"] = decisions.apply(
        lambda row: order_narrative(row.to_dict()), axis=1
    )
    rollups = build_rollups(decisions, order_lines=dataset.order_lines)
    impact = service_impact_summary(decisions)

    run_dir = _run_directory(config)
    data_dir = run_dir / "data"
    model_dir = run_dir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, table in dataset.tables().items():
        table.to_csv(data_dir / f"{name}.csv", index=False)
    outcomes.to_csv(data_dir / "outcomes.csv", index=False)
    causes.to_csv(data_dir / "root_causes.csv", index=False)
    feature_table.to_csv(data_dir / "feature_table.csv", index=False)
    decisions.to_csv(data_dir / "scored_orders.csv", index=False)
    for name, rollup in rollups.items():
        rollup.to_csv(data_dir / f"{name}_rollup.csv", index=False)
    pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(
        run_dir / "planner_feedback.csv", index=False
    )
    joblib.dump(training.bundle, model_dir / "xgboost_risk.joblib")
    joblib.dump(bayesian_bundle, model_dir / "bayesian_network.joblib")

    model_scores = {
        "xgb": {
            "validation_metrics": xgb_selection["metrics"],
            "test_metrics": xgb_test_metrics,
            "threshold": xgb_selection["threshold"],
            "threshold_strategy": xgb_selection["strategy"],
        },
        "bbn": {
            "validation_metrics": bbn_selection["metrics"],
            "test_metrics": bbn_test_metrics,
            "threshold": bbn_selection["threshold"],
            "threshold_strategy": bbn_selection["strategy"],
        },
        "fused": {
            "validation_metrics": fused_selection["metrics"],
            "test_metrics": fused_test_metrics,
            "threshold": fused_selection["threshold"],
            "threshold_strategy": fused_selection["strategy"],
        },
        "prevalence_baseline": prevalence_metrics,
        "note": (
            "xgb/bbn/fused are each evaluated and thresholded independently in "
            "their own probability space using the same metric set. Only the "
            "fused threshold is used for decisions/UI; xgb/bbn thresholds here "
            "are reported for standalone-model comparison only."
        ),
    }

    report: dict[str, Any] = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "architecture": {
            "risk_model": training.bundle.model_kind,
            "risk_endpoint": training.bundle.endpoint,
            "bayesian_nodes": list(CAUSE_CATEGORIES),
            "bayesian_inference_mode": bayesian_bundle.inference_mode,
            "bayesian_engine_build_error": bayesian_bundle.engine_build_error,
            "fusion": "0.70 risk model + 0.30 Bayesian network",
            "explanation": "SHAP with local perturbation fallback",
            "endpoint_design_note": (
                "The predictive endpoint is binary OTIF miss risk; seven-category "
                "root-cause and pathway outputs are retained and evaluated "
                "separately (see cause_fidelity)."
            ),
            "vendor_fairness_note": (
                "vendor_rolling_fault_rate is conditioned on vendor_fault (only "
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
                "leading_signal_* columns are derived from point-in-time-observable "
                "operational fields/events (not the generator's latent cause), which "
                "removes future/generator-to-feature leakage. However, this synthetic "
                "generator still assigns each cause's own operational fields with "
                "deterministic, low-noise thresholds (e.g. capture delay > 24h), so "
                "some causes remain highly separable once their evidence has posted. "
                "Strong held-out metrics on this dataset are a leakage/separability "
                "diagnostic for this specific synthetic generator, not evidence of "
                "production-grade predictive skill; see model_scores.prevalence_baseline "
                "and cause_fidelity for additional context."
            ),
        },
        "model_scores": model_scores,
        "cause_fidelity": cause_fidelity,
        # Backward-compatible top-level aliases: these now describe the FUSED
        # score space (the one actually used for decisions), not the XGB-only
        # values a prior version of this pipeline reported here.
        "validation_metrics": fused_selection["metrics"],
        "test_metrics": fused_test_metrics,
        "threshold": fused_selection["threshold"],
        "threshold_strategy": fused_selection["strategy"],
        "capacity_baseline_metrics": training.capacity_baseline_metrics,
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
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
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
        f"strategy={report['threshold_strategy']} "
        f"PR-AUC={metrics['pr_auc']:.3f} "
        f"recall={metrics['recall']:.3f} "
        f"precision={metrics['precision']:.3f} "
        f"threshold={report['threshold']:.3f}"
    )


if __name__ == "__main__":
    main()
