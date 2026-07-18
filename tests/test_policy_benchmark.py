from __future__ import annotations

import json

from otif_risk.policy_benchmark import run_policy_benchmark


def test_run_policy_benchmark_schema_and_gates():
    payload = run_policy_benchmark(seeds=(3, 5), n_orders=250)
    assert payload["seeds"] == [3, 5]
    assert len(payload["per_seed"]) == 2
    for report in payload["per_seed"]:
        assert report["no_action_identity_gate"]["passed"] is True

    summary = payload["summary"]
    assert "acceptance_gates" in summary
    assert summary["primary_capacity_scenario"] == "SCARCE_50_PERCENT"
    assert "median_headline_by_capacity_scenario" in summary
    assert set(summary["median_headline_by_capacity_scenario"]) == {
        "SCARCE_25_PERCENT",
        "SCARCE_50_PERCENT",
        "BASE_100_PERCENT",
    }
    # Must be JSON-serializable end to end (this is what the CLI writes).
    json.dumps(payload)


def test_run_policy_benchmark_is_reproducible_for_a_fixed_seed():
    first = run_policy_benchmark(seeds=(9,), n_orders=250)
    second = run_policy_benchmark(seeds=(9,), n_orders=250)
    first_policies = first["per_seed"][0]["policy_evaluation"]["policies"]
    second_policies = second["per_seed"][0]["policy_evaluation"]["policies"]
    for policy, metrics in first_policies.items():
        assert metrics["total_avoided_penalty"] == second_policies[policy]["total_avoided_penalty"]

    first_scenarios = first["per_seed"][0]["capacity_sensitivity"]["scenarios"]
    second_scenarios = second["per_seed"][0]["capacity_sensitivity"]["scenarios"]
    for scenario in first_scenarios:
        for policy, metrics in first_scenarios[scenario]["policies"].items():
            assert (
                metrics["total_avoided_penalty"]
                == second_scenarios[scenario]["policies"][policy]["total_avoided_penalty"]
            )
