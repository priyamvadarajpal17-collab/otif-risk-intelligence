from __future__ import annotations

import json
import time
from pathlib import Path

from otif_risk.contracts import PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.manifest import (
    ManifestInputs,
    artifact_checksums,
    build_manifest,
    checksum_file,
    content_id,
    dataset_fingerprints,
    dependency_versions,
    feature_schema_hash,
    git_info,
    table_fingerprint,
    verify_manifest,
    write_manifest,
)


def test_git_info_never_raises_and_reports_sha_when_available():
    info = git_info()
    assert "sha" in info and "dirty" in info and "status" in info
    # This repository is a git checkout, so a SHA should resolve.
    assert info["sha"] is None or isinstance(info["sha"], str)


def test_dependency_versions_reports_known_and_missing_packages():
    versions = dependency_versions(("pandas", "definitely-not-a-real-package"))
    assert versions["pandas"] is not None
    assert versions["definitely-not-a-real-package"] is None


def test_table_fingerprint_reports_row_count_schema_and_date_range():
    dataset = generate_dataset(PrototypeConfig(seed=1, n_orders=210))
    fingerprint = table_fingerprint(dataset.orders, timestamp_column="order_date")
    assert fingerprint["row_count"] == len(dataset.orders)
    assert fingerprint["columns"] == list(dataset.orders.columns)
    assert fingerprint["date_range"] is not None
    assert len(fingerprint["content_hash"]) == 64


def test_table_fingerprint_without_timestamp_column_has_no_date_range():
    dataset = generate_dataset(PrototypeConfig(seed=1, n_orders=210))
    fingerprint = table_fingerprint(dataset.vendors, timestamp_column=None)
    assert fingerprint["date_range"] is None


def test_dataset_fingerprints_excludes_truth_tables():
    dataset = generate_dataset(PrototypeConfig(seed=1, n_orders=210))
    fingerprints = dataset_fingerprints(dataset)
    assert set(fingerprints) == set(dataset.tables())
    assert "simulator_truth" not in fingerprints
    assert "line_truth" not in fingerprints
    assert "shocks" not in fingerprints


def test_feature_schema_hash_is_stable_for_identical_columns():
    dataset = generate_dataset(PrototypeConfig(seed=1, n_orders=210))
    hash_a = feature_schema_hash(dataset.orders)
    hash_b = feature_schema_hash(dataset.orders.copy())
    assert hash_a == hash_b
    assert feature_schema_hash(dataset.orders.drop(columns=["order_id"])) != hash_a


def test_checksum_file_detects_content_change(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    first = checksum_file(path)
    path.write_text("hello!", encoding="utf-8")
    second = checksum_file(path)
    assert first != second


def test_artifact_checksums_excludes_manifest_file(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    checksums = artifact_checksums(tmp_path)
    assert "data.csv" in checksums
    assert "run_manifest.json" not in checksums


def _minimal_inputs(seed: int = 1, n_orders: int = 210) -> ManifestInputs:
    config = PrototypeConfig(seed=seed, n_orders=n_orders)
    dataset = generate_dataset(config)
    return ManifestInputs(
        run_kind="pipeline",
        config=config,
        dataset=dataset,
        feature_table=dataset.orders,
        schema_versions={"artifact_schema_version": "3.0"},
        training_window=("2024-01-01", "2024-06-01"),
        validation_window=("2024-06-01", "2024-08-01"),
        test_window=("2024-08-01", "2024-10-01"),
        model_versions={"xgboost": 1, "bayesian": 1},
        extra_content={"fused_threshold": 0.42},
    )


def test_identical_seed_config_code_produces_identical_content_id(tmp_path):
    manifest_a = build_manifest(_minimal_inputs(), tmp_path / "run-a")
    time.sleep(0.01)
    manifest_b = build_manifest(_minimal_inputs(), tmp_path / "run-b")

    assert manifest_a["content_id"] == manifest_b["content_id"]
    # Run-instance metadata legitimately differs and is excluded from content_id.
    assert manifest_a["run_instance_id"] != manifest_b["run_instance_id"]
    assert manifest_a["run_directory"] != manifest_b["run_directory"]


def test_content_id_ignores_output_dir_and_timestamp(tmp_path):
    config_a = PrototypeConfig(seed=2, n_orders=210, output_dir=Path("artifacts"))
    config_b = PrototypeConfig(seed=2, n_orders=210, output_dir=Path("other_output"))
    dataset = generate_dataset(config_a)
    inputs_a = ManifestInputs(run_kind="pipeline", config=config_a, dataset=dataset)
    inputs_b = ManifestInputs(run_kind="pipeline", config=config_b, dataset=dataset)

    manifest_a = build_manifest(inputs_a, tmp_path / "run-1")
    manifest_b = build_manifest(inputs_b, tmp_path / "run-2-different-name")
    assert manifest_a["content_id"] == manifest_b["content_id"]


def test_content_id_changes_when_config_changes(tmp_path):
    manifest_a = build_manifest(_minimal_inputs(seed=1), tmp_path / "run-a")
    manifest_b = build_manifest(_minimal_inputs(seed=2), tmp_path / "run-b")
    assert manifest_a["content_id"] != manifest_b["content_id"]


def test_content_id_pure_function_of_payload():
    payload = {"a": 1, "b": [1, 2, 3]}
    assert content_id(payload) == content_id(dict(payload))


def test_write_manifest_then_verify_passes(tmp_path):
    run_dir = tmp_path / "run-x"
    run_dir.mkdir()
    (run_dir / "data").mkdir()
    scored_path = run_dir / "data" / "scored_orders.csv"
    scored_path.write_text("order_id,risk\nO-1,0.5\n", encoding="utf-8")
    (run_dir / "models").mkdir()
    (run_dir / "models" / "model.bin").write_bytes(b"\x00\x01\x02")

    write_manifest(run_dir, _minimal_inputs())
    report = verify_manifest(run_dir)

    assert report["verified"] is True
    assert report["files_missing"] == []
    assert report["files_mismatched"] == []
    assert report["files_verified"] >= 2


def test_verify_manifest_detects_tampering(tmp_path):
    run_dir = tmp_path / "run-y"
    run_dir.mkdir()
    (run_dir / "data").mkdir()
    scored_path = run_dir / "data" / "scored_orders.csv"
    scored_path.write_text("order_id,risk\nO-1,0.5\n", encoding="utf-8")

    write_manifest(run_dir, _minimal_inputs())

    # Tamper with an already-checksummed artifact after the manifest was written.
    scored_path.write_text("order_id,risk\nO-1,0.99\n", encoding="utf-8")

    report = verify_manifest(run_dir)
    assert report["verified"] is False
    assert "data/scored_orders.csv" in report["files_mismatched"]


def test_verify_manifest_detects_missing_file(tmp_path):
    run_dir = tmp_path / "run-z"
    run_dir.mkdir()
    (run_dir / "data").mkdir()
    scored_path = run_dir / "data" / "scored_orders.csv"
    scored_path.write_text("order_id,risk\nO-1,0.5\n", encoding="utf-8")

    write_manifest(run_dir, _minimal_inputs())
    scored_path.unlink()

    report = verify_manifest(run_dir)
    assert report["verified"] is False
    assert "data/scored_orders.csv" in report["files_missing"]


def test_verify_manifest_missing_manifest_file(tmp_path):
    report = verify_manifest(tmp_path)
    assert report["verified"] is False
    assert "not found" in report["reason"]


def test_write_manifest_persists_valid_json(tmp_path):
    run_dir = tmp_path / "run-w"
    run_dir.mkdir()
    manifest = write_manifest(run_dir, _minimal_inputs())
    persisted = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert persisted["content_id"] == manifest["content_id"]
    assert persisted["manifest_schema_version"] == "1.0"
