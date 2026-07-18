"""Champion/challenger model registry and promotion gate (Stage 2 governance).

Every trained model version is registered once, immutably (see
``ModelRegistry.register_version`` -- re-registering an existing version ID
raises rather than silently overwriting it), and every lifecycle transition
(``PROMOTED``/``HELD``/``ROLLED_BACK``) is appended to an append-only event
log that is never rewritten or truncated. ``active_model.json`` is the one
auditable pointer a serving process would read to know which version is
live; it is written atomically (temp file + ``os.replace``) so a reader
never observes a half-written pointer.

``evaluate_promotion`` is the promotion gate itself: a challenger is
promoted only if it does not regress the champion beyond an explicit,
fixed tolerance on PR-AUC, Brier/calibration, recall, alert rate,
drift-regime quality, and simulated policy value at the 50%-capacity
scenario (``policy_evaluation.PRIMARY_CAPACITY_SCENARIO``) -- and only if
its own schema/leakage/manifest-checksum gates already passed. Any failed
check holds the challenger (with an explicit, honest reason) and leaves
the active pointer untouched.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = "1.0"

PROMOTED = "PROMOTED"
HELD = "HELD"
ROLLED_BACK = "ROLLED_BACK"
REGISTERED = "REGISTERED"

REGISTRY_VERSIONS_FILENAME = "registry_versions.json"
REGISTRY_EVENTS_FILENAME = "registry_events.jsonl"
ACTIVE_MODEL_FILENAME = "active_model.json"


@dataclass(frozen=True)
class ModelMetrics:
    """The champion/challenger comparison surface for one model version."""

    pr_auc: float
    brier: float
    calibration_error: float
    recall: float
    alert_rate: float
    #: Realized PR-AUC restricted to the scripted-drift regime (see
    #: ``drift.py``/``data.DRIFT_WINDOW_FRACTION``); ``None`` when not
    #: computed for this version.
    drift_regime_pr_auc: float | None
    normal_regime_pr_auc: float | None
    #: ``CURRENT_POLICY``'s ``avoided_penalty_per_normalized_resource_unit``
    #: at ``policy_evaluation.PRIMARY_CAPACITY_SCENARIO`` (50% capacity) --
    #: the plan's headline decision-value number.
    policy_value_50pct_capacity: float
    schema_valid: bool
    leakage_gate_passed: bool
    manifest_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionTolerances:
    """Fixed, documented regression tolerances -- never retuned per-decision."""

    max_pr_auc_regression: float = 0.02
    max_brier_regression: float = 0.02
    max_calibration_error_regression: float = 0.05
    max_recall_regression: float = 0.05
    max_alert_rate_increase: float = 0.05
    max_drift_regime_pr_auc_regression: float = 0.05
    #: Relative regression tolerance vs. the champion's own policy value
    #: (e.g. 0.05 == challenger may be up to 5% below champion).
    max_policy_value_regression_fraction: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionDecision:
    decision: str  # PROMOTED | HELD
    reasons: list[str]
    gate_results: dict[str, GateResult]
    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    tolerances: PromotionTolerances

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": self.reasons,
            "gate_results": {
                name: {"passed": result.passed, "detail": result.detail}
                for name, result in self.gate_results.items()
            },
            "champion_metrics": self.champion_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "tolerances": self.tolerances.to_dict(),
        }


#: Default tolerances singleton (dataclass is frozen/immutable, so sharing
#: one instance across calls is safe).
DEFAULT_PROMOTION_TOLERANCES = PromotionTolerances()


def evaluate_promotion(
    champion: ModelMetrics,
    challenger: ModelMetrics,
    tolerances: PromotionTolerances | None = None,
) -> PromotionDecision:
    """Evaluate a challenger against the current champion under fixed tolerances.

    Every check is independent and all must pass for ``PROMOTED``; any
    single failed check produces ``HELD`` with that check's reason
    appended (multiple failures all appear in ``reasons``, never just the
    first).
    """
    tolerances = tolerances or DEFAULT_PROMOTION_TOLERANCES
    gate_results: dict[str, GateResult] = {}

    gate_results["schema_valid"] = GateResult(
        challenger.schema_valid, "challenger schema validation failed"
    )
    gate_results["leakage_gate"] = GateResult(
        challenger.leakage_gate_passed, "challenger leakage gate failed"
    )
    gate_results["manifest_verified"] = GateResult(
        challenger.manifest_verified, "challenger manifest checksum verification failed"
    )
    gate_results["pr_auc"] = GateResult(
        challenger.pr_auc >= champion.pr_auc - tolerances.max_pr_auc_regression,
        f"PR-AUC regressed {champion.pr_auc:.4f} -> {challenger.pr_auc:.4f} "
        f"(tolerance {tolerances.max_pr_auc_regression})",
    )
    gate_results["brier"] = GateResult(
        challenger.brier <= champion.brier + tolerances.max_brier_regression,
        f"Brier regressed {champion.brier:.4f} -> {challenger.brier:.4f} "
        f"(tolerance {tolerances.max_brier_regression})",
    )
    gate_results["calibration"] = GateResult(
        challenger.calibration_error
        <= champion.calibration_error + tolerances.max_calibration_error_regression,
        f"calibration error regressed {champion.calibration_error:.4f} -> "
        f"{challenger.calibration_error:.4f} "
        f"(tolerance {tolerances.max_calibration_error_regression})",
    )
    gate_results["recall"] = GateResult(
        challenger.recall >= champion.recall - tolerances.max_recall_regression,
        f"recall regressed {champion.recall:.4f} -> {challenger.recall:.4f} "
        f"(tolerance {tolerances.max_recall_regression})",
    )
    gate_results["alert_rate"] = GateResult(
        challenger.alert_rate <= champion.alert_rate + tolerances.max_alert_rate_increase,
        f"alert rate increased {champion.alert_rate:.4f} -> {challenger.alert_rate:.4f} "
        f"(tolerance {tolerances.max_alert_rate_increase})",
    )
    if champion.drift_regime_pr_auc is not None and challenger.drift_regime_pr_auc is not None:
        gate_results["drift_regime_quality"] = GateResult(
            challenger.drift_regime_pr_auc
            >= champion.drift_regime_pr_auc - tolerances.max_drift_regime_pr_auc_regression,
            f"drift-regime PR-AUC regressed {champion.drift_regime_pr_auc:.4f} -> "
            f"{challenger.drift_regime_pr_auc:.4f} "
            f"(tolerance {tolerances.max_drift_regime_pr_auc_regression})",
        )
    else:
        # Never silently absent: recorded as a passing, transparently-labeled
        # gate rather than omitted, so a reader can see this dimension was
        # not evaluated (insufficient matured drift-regime sample on one or
        # both sides) rather than mistaking its absence for an oversight.
        gate_results["drift_regime_quality"] = GateResult(
            True,
            "insufficient drift-regime sample on champion or challenger "
            "(fewer than the minimum matured observations); not evaluated, "
            "does not block promotion",
        )
    policy_floor = champion.policy_value_50pct_capacity * (
        1 - tolerances.max_policy_value_regression_fraction
    )
    gate_results["policy_value_50pct_capacity"] = GateResult(
        challenger.policy_value_50pct_capacity >= policy_floor,
        f"policy value at 50% capacity regressed {champion.policy_value_50pct_capacity:.4f} -> "
        f"{challenger.policy_value_50pct_capacity:.4f} "
        f"(floor {policy_floor:.4f}, tolerance fraction "
        f"{tolerances.max_policy_value_regression_fraction})",
    )

    reasons = [result.detail for result in gate_results.values() if not result.passed]
    decision = PROMOTED if not reasons else HELD
    return PromotionDecision(
        decision=decision,
        reasons=reasons,
        gate_results=gate_results,
        champion_metrics=champion,
        challenger_metrics=challenger,
        tolerances=tolerances,
    )


@dataclass(frozen=True)
class ModelVersion:
    """One immutable, registered model version."""

    version_id: str
    trained_at_utc: str
    manifest_content_id: str | None
    metrics: ModelMetrics
    artifact_paths: dict[str, str] = field(default_factory=dict)
    parent_version_id: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


class ModelRegistry:
    """Append-only version/event registry plus an atomic active-model pointer."""

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.versions_path = self.registry_dir / REGISTRY_VERSIONS_FILENAME
        self.events_path = self.registry_dir / REGISTRY_EVENTS_FILENAME
        self.active_pointer_path = self.registry_dir / ACTIVE_MODEL_FILENAME

    def _load_versions(self) -> dict[str, dict[str, Any]]:
        if not self.versions_path.is_file():
            return {}
        return json.loads(self.versions_path.read_text(encoding="utf-8"))

    def _save_versions(self, versions: dict[str, dict[str, Any]]) -> None:
        _atomic_write_json(self.versions_path, versions)

    def _append_event(self, event: dict[str, Any]) -> None:
        """Append one event line -- the log file is only ever appended to,
        never rewritten or truncated."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    def register_version(self, version: ModelVersion) -> None:
        """Register a new immutable version. Raises if the ID already exists."""
        versions = self._load_versions()
        if version.version_id in versions:
            raise ValueError(f"model version already registered: {version.version_id}")
        versions[version.version_id] = version.to_dict()
        self._save_versions(versions)
        self._append_event(
            {
                "event": REGISTERED,
                "version_id": version.version_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
        )

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        return self._load_versions().get(version_id)

    def promote_or_hold(
        self, challenger_version_id: str, decision: PromotionDecision
    ) -> dict[str, Any]:
        """Apply a ``PromotionDecision``: activate the pointer on ``PROMOTED``,
        leave it untouched on ``HELD``. Always appends an event either way."""
        versions = self._load_versions()
        if challenger_version_id not in versions:
            raise ValueError(f"unknown challenger version: {challenger_version_id}")

        event: dict[str, Any] = {
            "event": decision.decision,
            "version_id": challenger_version_id,
            "reasons": decision.reasons,
            "gate_results": decision.to_dict()["gate_results"],
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        if decision.decision == PROMOTED:
            self._set_active_pointer(challenger_version_id, event="PROMOTED")
        self._append_event(event)
        return event

    def _set_active_pointer(self, version_id: str, *, event: str) -> None:
        previous = self.active_version()
        pointer_payload = {
            "active_version_id": version_id,
            "previous_version_id": previous,
            "set_by_event": event,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(self.active_pointer_path, pointer_payload)

    def active_version(self) -> str | None:
        if not self.active_pointer_path.is_file():
            return None
        payload = json.loads(self.active_pointer_path.read_text(encoding="utf-8"))
        return payload.get("active_version_id")

    def rollback(self, target_version_id: str) -> dict[str, Any]:
        """Roll the active pointer back to ``target_version_id``.

        Only permitted when the target is an already-registered version
        whose manifest checksum verification passed
        (``ModelMetrics.manifest_verified``) -- rollback must never restore
        an unverified or unknown artifact. Always appends an event; on
        failure the event records ``rolled_back=False`` and the reason,
        and the active pointer is left untouched.
        """
        versions = self._load_versions()
        target = versions.get(target_version_id)
        timestamp = datetime.now(UTC).isoformat()
        if target is None:
            event = {
                "event": ROLLED_BACK,
                "version_id": target_version_id,
                "rolled_back": False,
                "reason": f"unknown version: {target_version_id}",
                "timestamp_utc": timestamp,
            }
            self._append_event(event)
            return event
        if not target["metrics"]["manifest_verified"]:
            event = {
                "event": ROLLED_BACK,
                "version_id": target_version_id,
                "rolled_back": False,
                "reason": "target version failed manifest checksum verification",
                "timestamp_utc": timestamp,
            }
            self._append_event(event)
            return event

        self._set_active_pointer(target_version_id, event="ROLLED_BACK")
        event = {
            "event": ROLLED_BACK,
            "version_id": target_version_id,
            "rolled_back": True,
            "reason": "manual rollback to verified version",
            "timestamp_utc": timestamp,
        }
        self._append_event(event)
        return event

    def history(self) -> list[dict[str, Any]]:
        """Every append-only lifecycle event, in the order they were written."""
        if not self.events_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def versions(self) -> dict[str, dict[str, Any]]:
        return self._load_versions()
