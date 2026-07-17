from __future__ import annotations

from pathlib import Path

from otif_risk.benchmark import run_benchmark


def test_run_benchmark_produces_per_seed_and_summary(tmp_path: Path) -> None:
    payload = run_benchmark(seeds=(1, 2, 3), n_orders=300, output_dir=tmp_path / "artifacts")

    assert len(payload["per_seed"]) == 3
    assert {row["seed"] for row in payload["per_seed"]} == {1, 2, 3}
    for metric in ("otif_miss_rate", "fused_pr_auc", "fused_recall"):
        assert {"median", "min", "max"} <= set(payload["summary"][metric])
    assert "acceptance_gates" in payload
    assert "note" in payload


def test_acceptance_gates_report_target_ranges_and_within_range_flag(tmp_path: Path) -> None:
    payload = run_benchmark(seeds=(4, 5), n_orders=300, output_dir=tmp_path / "artifacts")

    for gate_name in ("otif_miss_rate", "fused_pr_auc", "fused_recall"):
        gate = payload["acceptance_gates"][gate_name]
        assert "target_range" in gate
        assert "within_range" in gate
        assert isinstance(gate["within_range"], bool)


def test_benchmark_reports_fused_vs_prevalence_and_naive_line_baselines(tmp_path: Path) -> None:
    payload = run_benchmark(seeds=(6,), n_orders=300, output_dir=tmp_path / "artifacts")

    gates = payload["acceptance_gates"]
    assert isinstance(gates["fused_beats_prevalence"], bool)
    assert isinstance(gates["line_evidence_beats_naive"], bool)


def test_benchmark_is_reproducible_across_repeated_runs(tmp_path: Path) -> None:
    first = run_benchmark(seeds=(7,), n_orders=300, output_dir=tmp_path / "artifacts")
    second = run_benchmark(seeds=(7,), n_orders=300, output_dir=tmp_path / "artifacts")

    assert first["per_seed"][0]["otif_miss_rate"] == second["per_seed"][0]["otif_miss_rate"]
    assert first["per_seed"][0]["fused_pr_auc"] == second["per_seed"][0]["fused_pr_auc"]
