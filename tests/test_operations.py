from __future__ import annotations

import json

import pandas as pd

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
