"""Multi-seed decision-policy benchmark CLI (Stage 1 of the decision-value plan).

Runs ``policy_evaluation.run_seed_evaluation`` across several fixed seeds,
aggregates medians via ``policy_evaluation.summarize_multi_seed``, and writes
the full per-seed/per-policy report plus acceptance gates to
``artifacts/policy_benchmark.json``.

This is deliberately a separate CLI/module from ``benchmark.py`` (which
benchmarks prediction/causal-fidelity quality, not decision value): the two
answer different questions ("is the risk model good?" vs "does acting on it
create more value than simpler baselines, under real capacity limits?") and
are reported side by side, not merged into one number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from otif_risk.contracts import PrototypeConfig
from otif_risk.manifest import ManifestInputs, verify_manifest, write_manifest
from otif_risk.policy_evaluation import (
    POLICY_EVALUATION_VERSION,
    run_seed_evaluation,
    summarize_multi_seed,
)

#: At least five seeds by default, per the Decision Value + Governance plan.
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_N_ORDERS = 2_500


def run_policy_benchmark(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_orders: int = DEFAULT_N_ORDERS,
    **config_overrides: Any,
) -> dict[str, Any]:
    """Run ``run_seed_evaluation`` across ``seeds`` and summarize the results."""
    per_seed = [
        run_seed_evaluation(PrototypeConfig(seed=seed, n_orders=n_orders, **config_overrides))
        for seed in seeds
    ]
    summary = summarize_multi_seed(per_seed)
    return {
        "seeds": list(seeds),
        "n_orders": n_orders,
        "per_seed": per_seed,
        "summary": summary,
        "note": (
            "Policy value is measured via a probabilistic action-response digital "
            "twin re-simulating each feasible intervention under common random "
            "numbers, not assumed via a fixed effectiveness fraction. Every policy "
            "is evaluated at three pre-specified capacity-stress scenarios (25%/"
            "50%/100% of default capacity, applied uniformly to every resource "
            "pool for every policy -- see summary.capacity_scenarios); the "
            "acceptance gates are measured at the 50%-capacity scenario "
            "(summary.primary_capacity_scenario), not the 100% baseline, because "
            "this twin's default capacities are rarely binding at 100%. The oracle "
            "policy is an evaluation-only, unattainable ceiling used solely for "
            "regret -- never a deployable recommendation. Seeds outside any "
            "informal expectation are reported as measured, not adjusted away."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--orders", type=int, default=DEFAULT_N_ORDERS)
    parser.add_argument(
        "--benchmark-path", type=Path, default=Path("artifacts/policy_benchmark.json")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_policy_benchmark(tuple(args.seeds), args.orders)
    args.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # Written last, in the benchmark's own shared output directory, and
    # scoped (via only_paths) to exactly the file this run produced -- the
    # directory may also hold unrelated sibling artifacts (other runs,
    # benchmark.json, ...), which must never be described as this run's own.
    manifest_inputs = ManifestInputs(
        run_kind="policy_evaluation",
        config=PrototypeConfig(seed=args.seeds[0], n_orders=args.orders),
        schema_versions={"policy_evaluation_version": POLICY_EVALUATION_VERSION},
        extra_content={
            "seeds": list(args.seeds),
            "primary_capacity_scenario": payload["summary"]["primary_capacity_scenario"],
        },
    )
    manifest_dir = args.benchmark_path.parent
    write_manifest(
        manifest_dir,
        manifest_inputs,
        filename=f"{args.benchmark_path.stem}_manifest.json",
        only_paths=[args.benchmark_path],
    )
    verification = verify_manifest(
        manifest_dir, filename=f"{args.benchmark_path.stem}_manifest.json"
    )

    gates = payload["summary"]["acceptance_gates"]
    primary_scenario = payload["summary"]["primary_capacity_scenario"]
    headline = payload["summary"]["median_headline_by_capacity_scenario"][primary_scenario]
    print(
        f"seeds={payload['seeds']} "
        f"primary_capacity_scenario={primary_scenario} "
        f"current_policy_headline_median={headline['CURRENT_POLICY']} "
        f"beats_random={gates['current_beats_random_at_primary_capacity']} "
        f"beats_highest_risk={gates['current_beats_highest_risk_at_primary_capacity']} "
        f"primary_gate_passed={gates['primary_gate_passed']} "
        f"no_action_identity={gates['no_action_identity']} "
        f"manifest_verified={verification['verified']} "
        f"wrote {args.benchmark_path}"
    )


if __name__ == "__main__":
    main()
