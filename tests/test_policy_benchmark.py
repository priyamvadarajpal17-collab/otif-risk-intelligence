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


def test_main_cli_writes_scoped_manifest_alongside_benchmark(tmp_path, monkeypatch):
    from otif_risk.manifest import verify_manifest
    from otif_risk.policy_benchmark import main

    benchmark_path = tmp_path / "artifacts" / "policy_benchmark.json"
    (tmp_path / "artifacts").mkdir(parents=True)
    # An unrelated sibling artifact must never be described by this run's manifest.
    (tmp_path / "artifacts" / "unrelated.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "otif-policy-benchmark",
            "--seeds",
            "3",
            "--orders",
            "250",
            "--benchmark-path",
            str(benchmark_path),
        ],
    )
    main()

    manifest_path = tmp_path / "artifacts" / "policy_benchmark_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["artifact_checksums"]) == {"policy_benchmark.json"}

    verification = verify_manifest(
        tmp_path / "artifacts", filename="policy_benchmark_manifest.json"
    )
    assert verification["verified"] is True
