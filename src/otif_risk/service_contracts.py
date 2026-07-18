"""Typed scoring service contracts (Stage 2 governance).

Defines the request/response/sink boundary a deployed scoring service would
expose -- without adding a web framework, database, or message broker (see
the plan's explicit non-goals). ``score_via_service`` is the one function
that plays the role of "the service": given a validated ``ScoreRequest``
and an already-trained model bundle, it runs the *exact* same
``features.build_feature_table`` -> ``pipeline.score_orders`` ->
``decisions.recommend_orders`` path the canonical offline pipeline uses,
and returns one ``ScoreResponse`` per requested order. This is what
``test_adapters.py``'s offline/batch parity test compares against a
direct, in-process (offline) call of the same functions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from otif_risk.adapters import assemble_prototype_dataset, default_adapters
from otif_risk.bayesian import BayesianBundle
from otif_risk.contracts import PrototypeDataset
from otif_risk.decisions import DEFAULT_RISK_THRESHOLD, recommend_orders
from otif_risk.features import attach_line_evidence_features, build_feature_table
from otif_risk.model import RiskBundle
from otif_risk.pipeline import score_orders


class ContractError(ValueError):
    """Raised when a ``ScoreRequest``/``ScoreResponse`` fails contract validation."""


@dataclass(frozen=True)
class ScoreRequest:
    """A single batch scoring request against one point-in-time snapshot.

    ``source_snapshot_id`` identifies which adapter-assembled snapshot the
    request was scored against (e.g. a manifest content ID or a snapshot
    hash) -- the traceability link between a decision and the exact input
    data behind it. ``idempotency_key`` lets a retried request (same order,
    same as-of, same snapshot) be safely re-submitted without creating a
    duplicate decision (see ``DecisionSink``).
    """

    as_of_timestamp: pd.Timestamp
    order_ids: tuple[str, ...]
    source_snapshot_id: str
    idempotency_key: str

    def validate(self) -> None:
        if not self.order_ids:
            raise ContractError("ScoreRequest.order_ids must be non-empty")
        if len(set(self.order_ids)) != len(self.order_ids):
            raise ContractError("ScoreRequest.order_ids must not contain duplicates")
        if not self.source_snapshot_id:
            raise ContractError("ScoreRequest.source_snapshot_id must be non-empty")
        if not self.idempotency_key:
            raise ContractError("ScoreRequest.idempotency_key must be non-empty")
        if pd.isna(pd.Timestamp(self.as_of_timestamp)):
            raise ContractError("ScoreRequest.as_of_timestamp must be a valid timestamp")


@dataclass(frozen=True)
class ScoreResponse:
    """One order's scoring result, as a deployed service would return it."""

    idempotency_key: str
    order_id: str
    as_of_timestamp: str
    source_snapshot_id: str
    model_version: str
    policy_version: str
    manifest_content_id: str | None
    risk_score: float
    threshold: float
    confidence: str
    explanation: list[dict[str, Any]]
    decision_status: str
    recommended_action: str | None
    resource_type: str | None
    resource_status: str

    def decision_key(self) -> str:
        """Idempotency-stable identity for upsert-by-decision-key sinks."""
        return f"{self.order_id}:{self.idempotency_key}"

    def validate(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ContractError(f"risk_score out of [0, 1]: {self.risk_score}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ContractError(f"threshold out of [0, 1]: {self.threshold}")
        if self.decision_status not in {"RECOMMENDED", "CONTESTED", "MONITOR"}:
            raise ContractError(f"unknown decision_status: {self.decision_status}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["explanation"] = json.dumps(self.explanation)
        return payload


def score_via_service(
    request: ScoreRequest,
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    risk_bundle: RiskBundle,
    bayesian_bundle: BayesianBundle,
    xgb_weight: float,
    *,
    model_version: str,
    policy_version: str,
    manifest_content_id: str | None,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    background: pd.DataFrame | None = None,
) -> list[ScoreResponse]:
    """Score ``request.order_ids`` through the canonical feature/scoring path.

    Deliberately reuses ``features.build_feature_table`` and
    ``pipeline.score_orders`` unmodified -- the same functions the offline
    canonical pipeline calls -- so a service boundary built on top of this
    function cannot silently diverge from offline behavior. See
    ``test_adapters.py``'s parity test.
    """
    request.validate()
    order_ids = pd.Index(request.order_ids)
    features = build_feature_table(
        dataset, outcomes, causes, as_of_timestamp=request.as_of_timestamp, order_ids=order_ids
    )
    features = attach_line_evidence_features(dataset, features)
    scored = score_orders(
        dataset, features, risk_bundle, bayesian_bundle, xgb_weight, background=background
    )
    decisions = recommend_orders(scored, risk_threshold=risk_threshold)

    responses: list[ScoreResponse] = []
    for _, row in decisions.iterrows():
        explanation = []
        top_factors = row.get("top_factors_json")
        if isinstance(top_factors, str):
            try:
                explanation = json.loads(top_factors)
            except json.JSONDecodeError:
                explanation = []
        response = ScoreResponse(
            idempotency_key=request.idempotency_key,
            order_id=str(row["order_id"]),
            as_of_timestamp=pd.Timestamp(request.as_of_timestamp).isoformat(),
            source_snapshot_id=request.source_snapshot_id,
            model_version=model_version,
            policy_version=policy_version,
            manifest_content_id=manifest_content_id,
            risk_score=float(row["combined_risk_score"]),
            threshold=float(risk_threshold),
            confidence=str(row.get("causal_confidence", "UNKNOWN")),
            explanation=explanation,
            decision_status=str(row["decision_status"]),
            recommended_action=row.get("recommended_action"),
            resource_type=row.get("resource_type"),
            resource_status=str(row["decision_status"]),
        )
        response.validate()
        responses.append(response)
    return responses


@runtime_checkable
class DecisionSink(Protocol):
    """A durable, idempotent write target for scored decisions."""

    def write(self, responses: list[ScoreResponse]) -> int:
        """Upsert ``responses`` by decision key; return the number of new keys written."""
        ...

    def read_all(self) -> list[dict[str, Any]]:
        """Return every persisted decision (for reconciliation/reporting)."""
        ...


@dataclass
class JsonlDecisionSink:
    """Idempotent JSONL sink: one line per decision key, upserted on write.

    A retried ``write`` with the same ``(order_id, idempotency_key)`` never
    creates a duplicate line -- the existing line is overwritten in place
    (file is fully rewritten, sorted by decision key, so output bytes are
    deterministic given an identical decision set).
    """

    path: Path
    _index: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._index[record["decision_key"]] = record

    def write(self, responses: list[ScoreResponse]) -> int:
        new_keys = 0
        for response in responses:
            key = response.decision_key()
            if key not in self._index:
                new_keys += 1
            self._index[key] = {"decision_key": key, **response.to_dict()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for key in sorted(self._index):
                handle.write(json.dumps(self._index[key], sort_keys=True))
                handle.write("\n")
        return new_keys

    def read_all(self) -> list[dict[str, Any]]:
        return [self._index[key] for key in sorted(self._index)]


@dataclass
class CsvDecisionSink:
    """Idempotent CSV sink with the same upsert-by-decision-key semantics as
    :class:`JsonlDecisionSink`, for callers that prefer a tabular artifact."""

    path: Path
    _index: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _fieldnames: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.is_file():
            with self.path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self._fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    self._index[row["decision_key"]] = row

    def write(self, responses: list[ScoreResponse]) -> int:
        new_keys = 0
        for response in responses:
            key = response.decision_key()
            record = {"decision_key": key, **response.to_dict()}
            if not self._fieldnames:
                self._fieldnames = list(record.keys())
            if key not in self._index:
                new_keys += 1
            self._index[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            for key in sorted(self._index):
                writer.writerow(self._index[key])
        return new_keys

    def read_all(self) -> list[dict[str, Any]]:
        return [self._index[key] for key in sorted(self._index)]


def run_offline_batch_parity_check(
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    risk_bundle: RiskBundle,
    bayesian_bundle: BayesianBundle,
    xgb_weight: float,
    *,
    data_dir: Path,
    as_of_timestamp: pd.Timestamp,
    order_ids: pd.Index,
    background: pd.DataFrame,
) -> dict[str, Any]:
    """Persisted evidence for the offline/batch parity claim (see
    ``test_adapters.py``): scores ``order_ids`` both directly against the
    in-memory ``dataset`` (offline) and via ``adapters``-sourced,
    CSV-round-tripped tables read back from ``data_dir`` (service
    boundary), then compares. Returns a small, JSON-persistable report --
    intended to be written once per canonical run and loaded as-is by the
    Governance UI (never recomputed at UI render time).
    """
    offline_features = build_feature_table(
        dataset, outcomes, causes, as_of_timestamp=as_of_timestamp, order_ids=order_ids
    )
    offline_features = attach_line_evidence_features(dataset, offline_features)
    offline_scored = score_orders(
        dataset, offline_features, risk_bundle, bayesian_bundle, xgb_weight, background=background
    )

    assembled = assemble_prototype_dataset(
        default_adapters(data_dir), as_of_timestamp, truth_tables=dataset.truth_tables()
    )
    request = ScoreRequest(
        as_of_timestamp=as_of_timestamp,
        order_ids=tuple(order_ids),
        source_snapshot_id=data_dir.parent.name,
        idempotency_key="offline-batch-parity-check",
    )
    responses = score_via_service(
        request,
        assembled,
        outcomes,
        causes,
        risk_bundle,
        bayesian_bundle,
        xgb_weight,
        model_version="parity-check",
        policy_version="parity-check",
        manifest_content_id=None,
        background=background,
    )

    offline_scores = (
        offline_scored.set_index("order_id")["combined_risk_score"].astype(float).to_dict()
    )
    service_scores = {response.order_id: response.risk_score for response in responses}
    mismatched = [
        order_id
        for order_id, offline_score in offline_scores.items()
        if order_id not in service_scores
        or abs(service_scores[order_id] - offline_score) > 1e-9
    ]
    return {
        "n_orders_checked": len(offline_scores),
        "mismatched_order_ids": mismatched,
        "passed": not mismatched,
        "as_of_timestamp": pd.Timestamp(as_of_timestamp).isoformat(),
        "qualification": (
            "Compares the same order/as-of snapshot through the canonical "
            "in-process feature builder (offline) against adapters-sourced, "
            "CSV round-tripped tables through the service boundary "
            "(score_via_service). Persisted once per canonical run; the "
            "Governance UI only ever loads this file, never recomputes it."
        ),
    }
