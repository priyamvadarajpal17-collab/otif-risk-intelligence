"""Multi-seed benchmark: median/range, not one favorable seed.

Runs the canonical single-pipeline (``pipeline.run_pipeline``) across several
fixed seeds and reports the median and range of the metrics that matter for
judging this prototype's realism, plus explicit pass/fail against the
acceptance gates in the iteration plan. The target ranges are diagnostics,
not a held-out-test-set tuning objective: if a seed falls outside range, it
is reported as-is.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from otif_risk.contracts import PrototypeConfig
from otif_risk.pipeline import run_pipeline

#: At least five seeds by default, per the iteration plan; callers (e.g.
#: tests) may pass fewer/smaller fixtures.
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_N_ORDERS = 2_500

ACCEPTANCE_GATES: dict[str, tuple[float, float]] = {
    "otif_miss_rate": (0.15, 0.25),
    "fused_pr_auc": (0.65, 0.80),
    "fused_recall": (0.65, 0.80),
}


def _extract_seed_metrics(report: dict[str, Any]) -> dict[str, Any]:
    test_metrics = report["test_metrics"]
    line_evidence = report["line_evidence"]
    mechanism = report["mechanism_metrics"]
    confidence = report["causal_confidence_diagnostics"]
    consistency = report["causal_consistency"]
    return {
        "seed": report["config"]["seed"],
        "otif_miss_rate": report["data"]["otif_miss_rate"],
        "fused_pr_auc": test_metrics["pr_auc"],
        "fused_roc_auc": test_metrics["roc_auc"],
        "fused_recall": test_metrics["recall"],
        "fused_precision": test_metrics["precision"],
        "fused_brier": test_metrics["brier"],
        "xgb_pr_auc": report["model_scores"]["xgb"]["test_metrics"]["pr_auc"],
        "bbn_pr_auc": report["model_scores"]["bbn"]["test_metrics"]["pr_auc"],
        "prevalence_pr_auc": report["model_scores"]["prevalence_baseline"]["pr_auc"],
        "fusion_chosen_weight": report["architecture"]["fusion_chosen_weight"],
        "fusion_chosen_label": report["architecture"]["fusion_chosen_label"],
        "cause_fidelity_overall_agreement": report["cause_fidelity"]["overall_agreement"],
        "cause_fidelity_majority_baseline": report["cause_fidelity"][
            "majority_cause_baseline"
        ],
        "line_evidence_precision": line_evidence["targeted_evidence"]["precision"],
        "line_evidence_recall": line_evidence["targeted_evidence"]["recall"],
        "naive_line_precision": line_evidence["naive_all_lines_baseline"]["precision"],
        "alert_rate": test_metrics["flagged_orders"] / max(report["config"]["n_orders"] * 0.2, 1),
        "threshold": report["threshold"],
        "run_directory": report["provenance"]["run_directory"],
        "late_delivery_pr_auc": mechanism["late_delivery"]["pr_auc"],
        "late_delivery_brier": mechanism["late_delivery"]["brier"],
        "in_full_failure_pr_auc": mechanism["in_full_failure"]["pr_auc"],
        "in_full_failure_brier": mechanism["in_full_failure"]["brier"],
        "evidence_coverage_mean": confidence["evidence_coverage"]["mean"],
        "low_confidence_rate": confidence["low_confidence_rate"],
        "top_attribution_vs_rule_cause": consistency["top_attribution_vs_rule_cause"],
        "top_intervention_vs_rule_cause": consistency["top_intervention_vs_rule_cause"],
        "top_attribution_vs_simulator_cause": consistency[
            "top_attribution_vs_simulator_responsive_cause"
        ],
        "top_intervention_vs_simulator_cause": consistency[
            "top_intervention_vs_simulator_responsive_cause"
        ],
    }


def _summarize(values: list[float]) -> dict[str, float]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "median": round(statistics.median(clean), 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
    }


def run_benchmark(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_orders: int = DEFAULT_N_ORDERS,
    output_dir: Path = Path("artifacts"),
    **config_overrides: Any,
) -> dict[str, Any]:
    """Run the canonical pipeline across ``seeds`` and summarize the results."""
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        config = PrototypeConfig(
            seed=seed, n_orders=n_orders, output_dir=output_dir, **config_overrides
        )
        report = run_pipeline(config)
        per_seed.append(_extract_seed_metrics(report))

    summary = {
        metric: _summarize([row[metric] for row in per_seed])
        for metric in (
            "otif_miss_rate",
            "fused_pr_auc",
            "fused_roc_auc",
            "fused_recall",
            "fused_precision",
            "fused_brier",
            "xgb_pr_auc",
            "bbn_pr_auc",
            "cause_fidelity_overall_agreement",
            "cause_fidelity_majority_baseline",
            "line_evidence_precision",
            "line_evidence_recall",
            "alert_rate",
            "late_delivery_pr_auc",
            "late_delivery_brier",
            "in_full_failure_pr_auc",
            "in_full_failure_brier",
            "evidence_coverage_mean",
            "low_confidence_rate",
            "top_attribution_vs_rule_cause",
            "top_intervention_vs_rule_cause",
            "top_attribution_vs_simulator_cause",
            "top_intervention_vs_simulator_cause",
        )
    }

    gates: dict[str, Any] = {}
    for metric, (low, high) in ACCEPTANCE_GATES.items():
        median = summary[metric]["median"]
        gates[metric] = {
            "target_range": [low, high],
            "median": median,
            "within_range": bool(low <= median <= high),
        }
    gates["fused_beats_prevalence"] = bool(
        summary["fused_pr_auc"]["median"]
        > statistics.median(row["prevalence_pr_auc"] for row in per_seed)
    )
    gates["fused_calibration_reasonable"] = bool(summary["fused_brier"]["median"] <= 0.25)
    gates["line_evidence_beats_naive"] = bool(
        summary["line_evidence_precision"]["median"]
        > statistics.median(row["naive_line_precision"] for row in per_seed)
    )
    gates["cause_fidelity_beats_majority_baseline"] = bool(
        summary["cause_fidelity_overall_agreement"]["median"]
        > summary["cause_fidelity_majority_baseline"]["median"]
    )

    payload = {
        "seeds": list(seeds),
        "n_orders": n_orders,
        "per_seed": per_seed,
        "summary": summary,
        "acceptance_gates": gates,
        "note": (
            "Target ranges are diagnostics describing plausible honest-simulation "
            "behavior, not a held-out-test-set tuning objective. Seeds outside "
            "range are reported, not adjusted away."
        ),
    }
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--orders", type=int, default=DEFAULT_N_ORDERS)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--benchmark-path", type=Path, default=Path("artifacts/benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_benchmark(tuple(args.seeds), args.orders, args.output_dir)
    args.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    summary = payload["summary"]
    print(
        f"seeds={payload['seeds']} "
        f"miss_rate_median={summary['otif_miss_rate']['median']} "
        f"fused_pr_auc_median={summary['fused_pr_auc']['median']} "
        f"fused_recall_median={summary['fused_recall']['median']} "
        f"wrote {args.benchmark_path}"
    )


if __name__ == "__main__":
    main()
