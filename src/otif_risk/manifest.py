"""Deterministic run lineage manifests (Stage 2 governance).

Persists ``run_manifest.json`` alongside every pipeline run, policy
benchmark/evaluation, and operations-replay artifact directory. The core
design separates two things that are easy to conflate:

- a **deterministic content ID** (``content_id``) -- a stable hash of
  everything that determines the run's *model-facing content*: seed,
  normalized config, code version (git SHA), package/dependency versions,
  input-table fingerprints, feature schema, and policy/model schema
  versions. Two runs built from identical seed/config/code/model-facing
  artifacts produce the *same* ``content_id``, even if they ran at
  different times or wrote to different output directories.
- **run-instance metadata** (``run_instance_id``, ``generated_at_utc``,
  ``run_directory``) that legitimately differs between two runs of
  otherwise-identical inputs and is deliberately excluded from
  ``content_id``.

``artifact_checksums`` records the SHA-256 of every other file already
written into the run directory at manifest-build time (the manifest is
always the *last* file written for a run) -- this is run-instance
provenance (proves what these particular output bytes were), not part of
``content_id`` (output bytes can carry incidental floating point/ordering
noise across platforms that content identity should not depend on).

``verify_manifest`` recomputes every listed checksum and reports any
missing or mismatched file -- the tamper-evidence contract judges can
run before trusting a decision.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from otif_risk.contracts import PrototypeConfig, PrototypeDataset

#: Bump when the manifest's own field set/semantics change.
MANIFEST_SCHEMA_VERSION = "1.0"

#: Key runtime dependencies whose versions are recorded verbatim (best
#: effort -- an editable/local checkout without metadata is reported as
#: ``None`` rather than raising).
KEY_DEPENDENCIES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "pgmpy",
    "shap",
    "streamlit",
    "joblib",
)

#: Model-facing source tables fingerprinted for lineage. Deliberately
#: excludes ``PrototypeDataset.truth_tables()`` (``simulator_truth``,
#: ``line_truth``, ``shocks``): those are evaluation-only ground truth and
#: must never be described as a model input.
SOURCE_TABLE_TIMESTAMP_COLUMNS: dict[str, str] = {
    "orders": "order_date",
    "events": "event_timestamp",
    "capacity_snapshots": "snapshot_date",
}

#: Files never included in ``artifact_checksums`` -- the manifest cannot
#: checksum itself (that would be circular self-reference), and cache
#: files are not meaningful run artifacts.
ARTIFACT_CHECKSUM_EXCLUDE_NAMES = frozenset({"run_manifest.json", ".DS_Store"})
ARTIFACT_CHECKSUM_EXCLUDE_SUFFIXES = frozenset({".pyc"})

MANIFEST_FILENAME = "run_manifest.json"


def _run_git(args: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Best-effort git commit SHA + dirty/clean working tree state.

    Never raises: if ``git`` is unavailable or ``repo_root`` is not inside a
    git checkout (e.g. an extracted release tarball), every field is
    ``None``/``"unknown"`` rather than failing the run.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    sha = _run_git(["rev-parse", "HEAD"], cwd=root)
    if sha is None:
        return {"sha": None, "dirty": None, "status": "unavailable"}
    porcelain = _run_git(["status", "--porcelain"], cwd=root)
    dirty = bool(porcelain) if porcelain is not None else None
    return {"sha": sha, "dirty": dirty, "status": "clean" if dirty is False else "dirty"}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def dependency_versions(names: tuple[str, ...] = KEY_DEPENDENCIES) -> dict[str, str | None]:
    """Best-effort installed version of each key dependency."""
    return {name: _package_version(name) for name in names}


def _content_hash_frame(frame: pd.DataFrame) -> str:
    """Deterministic content hash of a DataFrame's values.

    Hashes the CSV serialization of ``frame`` in its existing column/row
    order. Two frames built from identical (seed, config, generation code)
    inputs always serialize identically; this is never used to fingerprint
    incidental artifact bytes (see the module docstring's ``content_id`` vs
    ``artifact_checksums`` split).
    """
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def table_fingerprint(
    frame: pd.DataFrame, *, timestamp_column: str | None = None
) -> dict[str, Any]:
    """Row count, schema, date range (if applicable), and content hash for one table."""
    date_range: list[str] | None = None
    if timestamp_column and timestamp_column in frame.columns and len(frame):
        parsed = pd.to_datetime(frame[timestamp_column])
        date_range = [parsed.min().isoformat(), parsed.max().isoformat()]
    return {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "date_range": date_range,
        "content_hash": _content_hash_frame(frame),
    }


def dataset_fingerprints(dataset: PrototypeDataset) -> dict[str, dict[str, Any]]:
    """Fingerprint every model-facing source table (never the truth tables)."""
    return {
        name: table_fingerprint(
            table, timestamp_column=SOURCE_TABLE_TIMESTAMP_COLUMNS.get(name)
        )
        for name, table in dataset.tables().items()
    }


def feature_schema_hash(feature_table: pd.DataFrame) -> str:
    """Stable hash of the feature table's column set (schema, not values)."""
    payload = json.dumps(sorted(feature_table.columns), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_checksum(path: Path) -> bool:
    return path.name not in ARTIFACT_CHECKSUM_EXCLUDE_NAMES and (
        path.suffix not in ARTIFACT_CHECKSUM_EXCLUDE_SUFFIXES
    )


def artifact_checksums(
    run_dir: Path, *, only_paths: Sequence[Path] | None = None
) -> dict[str, str]:
    """SHA-256 of every regular file already written under ``run_dir``.

    Excludes ``run_manifest.json`` itself (see
    ``ARTIFACT_CHECKSUM_EXCLUDE_NAMES``) -- callers must build/write the
    manifest only after every other artifact for the run has been written,
    so this captures a complete, self-consistent snapshot without ever
    checksumming the manifest file it is embedded in.

    ``only_paths`` restricts checksumming to an explicit file list --
    needed when ``run_dir`` is a shared directory that also holds
    unrelated artifacts from other runs (e.g. the flat ``artifacts/``
    directory a multi-seed benchmark CLI writes into), so this run's
    manifest never describes files it did not itself produce.
    """
    if only_paths is not None:
        checksums: dict[str, str] = {}
        for path in only_paths:
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(run_dir).as_posix()
            except ValueError:
                relative = path.name
            checksums[relative] = checksum_file(path)
        return checksums

    checksums = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or not _should_checksum(path):
            continue
        relative = path.relative_to(run_dir).as_posix()
        checksums[relative] = checksum_file(path)
    return checksums


def _normalized_config(config: PrototypeConfig) -> dict[str, Any]:
    """``PrototypeConfig`` as a plain dict with the output path stringified.

    ``output_dir`` is deliberately kept in the returned dict for
    provenance display, but is stripped before hashing into ``content_id``
    (see ``_content_identity_payload``) -- an identical run written to a
    different output directory must still produce the same content ID.
    """
    values = asdict(config)
    values["output_dir"] = str(values["output_dir"])
    return values


@dataclass
class ManifestInputs:
    """Everything one run needs to build a complete ``run_manifest.json``.

    ``run_kind`` identifies which caller produced the manifest
    (``"pipeline"``, ``"policy_evaluation"``, or ``"operations_replay"``);
    every other field is optional so each caller only supplies what it has.
    """

    run_kind: str
    config: PrototypeConfig
    dataset: PrototypeDataset | None = None
    feature_table: pd.DataFrame | None = None
    schema_versions: dict[str, str] = field(default_factory=dict)
    training_window: tuple[str, str] | None = None
    validation_window: tuple[str, str] | None = None
    test_window: tuple[str, str] | None = None
    parent_model_version: str | None = None
    champion_model_version: str | None = None
    model_versions: dict[str, Any] = field(default_factory=dict)
    #: Any other content that should participate in ``content_id`` (e.g. a
    #: policy-evaluation version string, a threshold, a chosen fusion
    #: weight) -- never timestamps/paths/instance IDs.
    extra_content: dict[str, Any] = field(default_factory=dict)


def _content_identity_payload(
    inputs: ManifestInputs,
    *,
    git_sha: str | None,
    package_version_str: str,
    deps: dict[str, str | None],
    dataset_fp: dict[str, Any] | None,
    feature_hash: str | None,
) -> dict[str, Any]:
    config_for_hash = _normalized_config(inputs.config)
    config_for_hash.pop("output_dir", None)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_kind": inputs.run_kind,
        "git_sha": git_sha,
        "package_version": package_version_str,
        "dependency_versions": deps,
        "config": config_for_hash,
        "schema_versions": dict(sorted(inputs.schema_versions.items())),
        "dataset_fingerprints": dataset_fp,
        "feature_schema_hash": feature_hash,
        "training_window": inputs.training_window,
        "validation_window": inputs.validation_window,
        "test_window": inputs.test_window,
        "parent_model_version": inputs.parent_model_version,
        "model_versions": dict(sorted(inputs.model_versions.items())),
        "extra_content": dict(sorted(inputs.extra_content.items())),
    }


def content_id(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _package_version_self() -> str:
    try:
        return version("otif-risk-intelligence")
    except PackageNotFoundError:  # pragma: no cover - editable/local checkouts
        return "0.0.0+local"


def build_manifest(inputs: ManifestInputs, run_dir: Path) -> dict[str, Any]:
    """Build the full manifest payload (without artifact checksums).

    Checksums require every other artifact to already be written, so
    ``build_manifest``/``write_manifest`` are split: build first (does not
    touch disk beyond fingerprinting in-memory tables), checksum the
    already-written run directory, then write ``run_manifest.json`` last.
    """
    git = git_info()
    deps = dependency_versions()
    package_version_str = _package_version_self()
    dataset_fp = dataset_fingerprints(inputs.dataset) if inputs.dataset is not None else None
    feature_hash = (
        feature_schema_hash(inputs.feature_table) if inputs.feature_table is not None else None
    )
    identity_payload = _content_identity_payload(
        inputs,
        git_sha=git["sha"],
        package_version_str=package_version_str,
        deps=deps,
        dataset_fp=dataset_fp,
        feature_hash=feature_hash,
    )
    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_kind": inputs.run_kind,
        "run_instance_id": uuid.uuid4().hex,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_directory": run_dir.name,
        "git": git,
        "package_version": package_version_str,
        "dependency_versions": deps,
        "config": _normalized_config(inputs.config),
        "schema_versions": dict(sorted(inputs.schema_versions.items())),
        "dataset_fingerprints": dataset_fp,
        "dataset_fingerprint_note": (
            "Fingerprints cover PrototypeDataset.tables() only -- the "
            "model-facing source tables. Evaluation-only truth tables "
            "(simulator_truth/line_truth/shocks) are intentionally excluded: "
            "they are never a model input and must never be described as one."
        ),
        "feature_schema_hash": feature_hash,
        "training_window": inputs.training_window,
        "validation_window": inputs.validation_window,
        "test_window": inputs.test_window,
        "parent_model_version": inputs.parent_model_version,
        "champion_model_version": inputs.champion_model_version,
        "model_versions": dict(sorted(inputs.model_versions.items())),
        "extra_content": dict(sorted(inputs.extra_content.items())),
        "content_id": content_id(identity_payload),
        "content_id_note": (
            "sha256 over code/config/data/schema/model-version identity only -- "
            "excludes generated_at_utc, run_instance_id, and run_directory/output_dir "
            "paths. Identical seed/config/code/model-facing artifacts always produce "
            "the same content_id even when run_instance_id differs."
        ),
        "artifact_checksums": {},
        "artifact_checksum_note": (
            "SHA-256 of every other file already written to this run directory "
            "at manifest-build time (never this manifest file itself -- see "
            "ARTIFACT_CHECKSUM_EXCLUDE_NAMES). Verify with manifest.verify_manifest()."
        ),
    }
    return manifest


def write_manifest(
    run_dir: Path,
    inputs: ManifestInputs,
    *,
    filename: str = MANIFEST_FILENAME,
    only_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Build the manifest, checksum every already-written artifact, and persist it.

    Must be called only after every other artifact for ``run_dir`` has been
    written -- the checksums are a snapshot of "everything else in this
    run directory right now" (or, when ``only_paths`` is given, of exactly
    those files -- see ``artifact_checksums``).
    """
    manifest = build_manifest(inputs, run_dir)
    manifest["artifact_checksums"] = artifact_checksums(run_dir, only_paths=only_paths)
    manifest_path = run_dir / filename
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_manifest(run_dir: Path, *, filename: str = MANIFEST_FILENAME) -> dict[str, Any]:
    """Recompute checksums for every artifact listed in the run's manifest file.

    Returns a verification report with per-file status plus an overall
    ``verified`` boolean -- ``False`` if any listed file is missing, any
    file's checksum no longer matches (tamper detection), or the manifest
    itself is absent/unreadable.
    """
    manifest_path = run_dir / filename
    if not manifest_path.is_file():
        return {
            "verified": False,
            "reason": f"{filename} not found",
            "run_directory": run_dir.name,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "verified": False,
            "reason": f"{filename} is not valid JSON",
            "run_directory": run_dir.name,
        }

    recorded = manifest.get("artifact_checksums", {})
    missing: list[str] = []
    mismatched: list[str] = []
    matched: list[str] = []
    for relative_path, expected_hash in recorded.items():
        candidate = run_dir / relative_path
        if not candidate.is_file():
            missing.append(relative_path)
            continue
        actual_hash = checksum_file(candidate)
        if actual_hash != expected_hash:
            mismatched.append(relative_path)
        else:
            matched.append(relative_path)

    verified = not missing and not mismatched
    return {
        "verified": verified,
        "run_directory": run_dir.name,
        "content_id": manifest.get("content_id"),
        "files_verified": len(matched),
        "files_missing": missing,
        "files_mismatched": mismatched,
        "checked_at_utc": datetime.now(UTC).isoformat(),
    }
