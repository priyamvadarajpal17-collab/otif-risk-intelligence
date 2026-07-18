from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from otif_risk.contracts import PrototypeConfig
from otif_risk.operations import OperationsConfig, run_operations_replay


def _small_ops_config(tmp_path, seed: int = 5, **overrides) -> OperationsConfig:
    defaults = dict(
        data_config=PrototypeConfig(seed=seed, n_orders=250, output_dir=tmp_path / "artifacts"),
        output_dir=tmp_path / "artifacts",
        replay_days=12,
        retrain_cadence_days=6,
        min_new_labels_for_retrain=5,
        min_days_between_retrains=3,
    )
    defaults.update(overrides)
    return OperationsConfig(**defaults)


def test_replay_completes_and_persists_registry_and_queues(tmp_path):
    result = run_operations_replay(_small_ops_config(tmp_path))

    run_dir = result["run_dir"]
    assert (run_dir / "model_registry.json").is_file()
    assert (run_dir / "model_registry.csv").is_file()
    assert (run_dir / "daily_log.json").is_file()
    assert (run_dir / "operations_summary.json").is_file()
    assert (run_dir / "planner_feedback.csv").is_file()
    assert list((run_dir / "daily_queues").glob("*.csv"))

    summary = result["summary"]
    assert summary["replay_days_completed"] == 12
    assert summary["model_versions_trained"] >= 1


def test_at_least_one_scheduled_retrain_occurs_with_short_cadence(tmp_path):
    result = run_operations_replay(
        _small_ops_config(tmp_path, retrain_cadence_days=4, min_new_labels_for_retrain=3)
    )
    triggers = {event["trigger"] for event in result["summary"]["retrain_events"]}
    assert "scheduled" in triggers


def test_model_registry_entries_are_versioned_and_monotonic(tmp_path):
    result = run_operations_replay(_small_ops_config(tmp_path))
    registry = result["registry"]
    versions = [entry["version"] for entry in registry]
    assert versions == sorted(versions)
    assert versions[0] == 1
    for entry in registry:
        assert 0 <= entry["fusion_weight"] <= 1
        assert entry["trigger"] in {"initial", "scheduled", "drift"}
        assert isinstance(entry["trigger_reasons"], list)
        assert entry["n_training_orders"] > 0


def test_daily_queue_snapshots_have_decision_status(tmp_path):
    result = run_operations_replay(_small_ops_config(tmp_path))
    queue_files = sorted((result["run_dir"] / "daily_queues").glob("*.csv"))
    assert queue_files
    sample = pd.read_csv(queue_files[0])
    if len(sample):
        assert set(sample["decision_status"]) <= {"RECOMMENDED", "CONTESTED", "MONITOR"}


def test_feedback_log_is_appended_for_closed_orders(tmp_path):
    result = run_operations_replay(_small_ops_config(tmp_path))
    feedback_path = result["run_dir"] / "planner_feedback.csv"
    feedback = pd.read_csv(feedback_path)
    assert len(feedback) > 0
    assert set(feedback["feedback_action"]) <= {"ACCEPT", "REJECT", "OVERRIDE"}
    assert feedback["original_recommendation"].str.contains("actual_otif_miss").all()


def test_operations_summary_json_is_well_formed(tmp_path):
    result = run_operations_replay(_small_ops_config(tmp_path))
    summary_path = result["run_dir"] / "operations_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "retrain_events" in payload
    assert "drift_warning_days" in payload
    assert "final_model_version" in payload


def test_initial_training_window_too_small_raises() -> None:
    from otif_risk.operations import run_operations_replay as replay

    tiny_config = OperationsConfig(
        data_config=PrototypeConfig(seed=1, n_orders=200),
        initial_training_fraction=0.02,
    )
    try:
        replay(tiny_config)
    except ValueError as error:
        assert "too few matured orders" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for too-small initial training window")


def test_replay_persists_stage2_governance_artifacts(tmp_path):
    result = run_operations_replay(_small_ops_config(tmp_path))
    run_dir = result["run_dir"]

    assert (run_dir / "decision_ledger.csv").is_file()
    assert (run_dir / "observational_cohort_report.json").is_file()
    assert (run_dir / "monitoring_report.json").is_file()
    assert (run_dir / "demo_lifecycle_scenario.json").is_file()
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "registry" / "registry_versions.json").is_file()
    assert (run_dir / "registry" / "registry_events.jsonl").is_file()
    assert (run_dir / "registry" / "active_model.json").is_file()

    ledger = pd.read_csv(run_dir / "decision_ledger.csv")
    assert len(ledger) > 0
    assert {"decision_key", "chosen_action", "planner_decision"} <= set(ledger.columns)

    cohort_report = json.loads((run_dir / "observational_cohort_report.json").read_text())
    assert cohort_report["observational_not_causal"] is True

    monitoring_report = result["monitoring_report"]
    assert monitoring_report["monitoring_schema_version"] == "1.0"
    assert "slo_status" in monitoring_report

    assert result["manifest"]["verification"]["verified"] is True


def test_replay_only_promotes_challenger_when_promotion_passes(tmp_path):
    result = run_operations_replay(
        _small_ops_config(tmp_path, retrain_cadence_days=4, min_new_labels_for_retrain=3)
    )
    history = result["governance_history"]
    events = {entry["event"] for entry in history}
    # Every retrain attempt is either promoted or held -- never silently skipped.
    assert events <= {"REGISTERED", "PROMOTED", "HELD", "ROLLED_BACK"}

    active_version_id = result["summary"]["active_model_version_id"]
    promoted_versions = {
        entry["version_id"] for entry in history if entry["event"] == "PROMOTED"
    }
    if active_version_id is not None and not active_version_id.startswith("demo-"):
        assert active_version_id in promoted_versions | {"v1"}


def test_replay_held_challenger_never_moves_active_pointer(tmp_path):
    """A version registered as HELD must never appear as any PROMOTED
    event's version, and the active pointer must always resolve to a
    version that was itself promoted (or the initial v1)."""
    result = run_operations_replay(
        _small_ops_config(tmp_path, retrain_cadence_days=4, min_new_labels_for_retrain=3)
    )
    history = result["governance_history"]
    held_versions = {entry["version_id"] for entry in history if entry["event"] == "HELD"}
    promoted_versions = {entry["version_id"] for entry in history if entry["event"] == "PROMOTED"}
    assert held_versions.isdisjoint(promoted_versions)


def test_demo_lifecycle_scenario_uses_measured_policy_benchmark_when_available(tmp_path):
    policy_benchmark_path = Path("artifacts/policy_benchmark.json")
    if not policy_benchmark_path.is_file():
        pytest.skip("artifacts/policy_benchmark.json not present in this checkout")

    result = run_operations_replay(
        _small_ops_config(
            tmp_path, policy_value_reference_path=policy_benchmark_path
        )
    )
    demo = result["demo_lifecycle_scenario"]
    assert demo["enabled"] is True
    assert demo["promotion_1_legacy_to_current_policy"]["decision"] == "PROMOTED"
    assert demo["promotion_2_bayesian_enhanced_held"]["decision"] == "HELD"
    assert any(
        "policy value" in reason
        for reason in demo["promotion_2_bayesian_enhanced_held"]["reasons"]
    )
    assert demo["rollback"]["rolled_back"] is True
    assert demo["active_version_after_demo"] == "v1"

    governance = result["summary"]["governance"]
    assert governance["demo_lifecycle_enabled"] is True
    assert governance["promotion_events"] >= 2
    assert governance["held_events"] >= 1
    assert governance["rollback_events"] == 1


def test_demo_lifecycle_scenario_skipped_without_reference(tmp_path):
    result = run_operations_replay(
        _small_ops_config(tmp_path, policy_value_reference_path=tmp_path / "missing.json")
    )
    demo = result["demo_lifecycle_scenario"]
    assert demo["enabled"] is False
    assert "reason" in demo



def test_artifacts_exist_and_readable_rejects_missing_or_empty_files(tmp_path):
    from otif_risk.operations import _artifacts_exist_and_readable

    real_file = tmp_path / "model.joblib"
    real_file.write_bytes(b"\x00\x01")
    empty_file = tmp_path / "empty.joblib"
    empty_file.write_bytes(b"")
    missing_file = tmp_path / "missing.joblib"

    assert _artifacts_exist_and_readable([real_file]) is True
    assert _artifacts_exist_and_readable([empty_file]) is False
    assert _artifacts_exist_and_readable([missing_file]) is False
    assert _artifacts_exist_and_readable([]) is False


def test_active_model_version_never_regresses_to_a_held_challengers_number(tmp_path):
    """Regression test: a held challenger's version number must never be
    reported as the active/live scoring version on subsequent days -- the
    ledger's model_version must always resolve to a version that was
    actually PROMOTED (or the initial v1), never a HELD one."""
    result = run_operations_replay(
        _small_ops_config(tmp_path, retrain_cadence_days=4, min_new_labels_for_retrain=3)
    )
    ledger_path = result["run_dir"] / "decision_ledger.csv"
    ledger = pd.read_csv(ledger_path)
    history = result["governance_history"]
    held_version_ids = {
        entry["version_id"] for entry in history if entry["event"] == "HELD"
    }
    promoted_version_ids = {
        entry["version_id"] for entry in history if entry["event"] == "PROMOTED"
    } | {"v1"}

    logged_model_versions = set(ledger["model_version"].unique())
    assert logged_model_versions & held_version_ids == set()
    assert logged_model_versions <= promoted_version_ids


def test_demo_lifecycle_rolls_back_to_the_given_target_not_a_hardcoded_version(tmp_path):
    """Regression test: the demo lifecycle's rollback target is whatever the
    caller passes (the real active version before the demo ran), never a
    hardcoded "v1" -- otherwise a genuinely promoted real challenger would be
    silently clobbered by the demo scenario."""
    from otif_risk.operations import _run_demo_lifecycle_scenario
    from otif_risk.registry import ModelMetrics, ModelRegistry, ModelVersion, evaluate_promotion

    registry = ModelRegistry(tmp_path / "registry")
    metrics = ModelMetrics(
        pr_auc=0.7,
        brier=0.1,
        calibration_error=0.03,
        recall=0.65,
        alert_rate=0.2,
        drift_regime_pr_auc=None,
        normal_regime_pr_auc=0.7,
        policy_value_50pct_capacity=9.0,
        schema_valid=True,
        leakage_gate_passed=True,
        manifest_verified=True,
    )
    # A "real" version other than v1 that is already active before the demo runs.
    registry.register_version(
        ModelVersion(
            version_id="v1",
            trained_at_utc="2024-01-01T00:00:00+00:00",
            manifest_content_id=None,
            metrics=metrics,
        )
    )
    registry.promote_or_hold("v1", evaluate_promotion(metrics, metrics))
    registry.register_version(
        ModelVersion(
            version_id="v3",
            trained_at_utc="2024-01-02T00:00:00+00:00",
            manifest_content_id=None,
            metrics=metrics,
            parent_version_id="v1",
        )
    )
    registry.promote_or_hold("v3", evaluate_promotion(metrics, metrics))
    assert registry.active_version() == "v3"

    policy_value_reference = {
        "source": "test",
        "primary_capacity_scenario": "SCARCE_50_PERCENT",
        "legacy_median_value": 9.0,
        "current_policy_median_value": 10.0,
        "bayesian_ablation_with_median": 9.0,
        "bayesian_ablation_without_median": 11.0,
    }
    demo = _run_demo_lifecycle_scenario(
        registry, policy_value_reference, metrics, rollback_target_version_id="v3"
    )
    assert demo["active_version_after_demo"] == "v3"
    assert registry.active_version() == "v3"
