from __future__ import annotations

import pandas as pd
import pytest

from otif_risk.adapters import (
    LocalCsvERPAdapter,
    LocalCsvSRMAdapter,
    LocalCsvTMSAdapter,
    LocalCsvWMSAdapter,
    assemble_prototype_dataset,
    assemble_source_tables,
    data_quality_report,
    default_adapters,
)
from otif_risk.contracts import PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.features import attach_line_evidence_features, build_feature_table, temporal_split
from otif_risk.pipeline import score_orders, train_full_bundle
from otif_risk.root_causes import calculate_outcomes, derive_root_causes
from otif_risk.service_contracts import (
    ContractError,
    CsvDecisionSink,
    JsonlDecisionSink,
    ScoreRequest,
    score_via_service,
)


def _write_dataset_csvs(dataset, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, table in dataset.tables().items():
        table.to_csv(data_dir / f"{name}.csv", index=False)
    return data_dir


@pytest.fixture(scope="module")
def small_trained_context(tmp_path_factory):
    """Train one small bundle + persist its dataset CSVs once for adapter tests."""
    config = PrototypeConfig(seed=11, n_orders=260)
    dataset = generate_dataset(config)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    feature_table = build_feature_table(dataset, outcomes, causes)
    feature_table = attach_line_evidence_features(dataset, feature_table)
    split = temporal_split(feature_table)
    trained = train_full_bundle(dataset, outcomes, causes, split, config)

    data_dir = _write_dataset_csvs(dataset, tmp_path_factory.mktemp("adapter_data"))
    return {
        "config": config,
        "dataset": dataset,
        "outcomes": outcomes,
        "causes": causes,
        "trained": trained,
        "data_dir": data_dir,
    }


def test_erp_adapter_filters_orders_not_yet_captured(small_trained_context):
    dataset = small_trained_context["dataset"]
    data_dir = small_trained_context["data_dir"]
    adapter = LocalCsvERPAdapter(
        data_dir / "orders.csv", data_dir / "order_lines.csv", data_dir / "customers.csv"
    )
    cutoff = pd.Timestamp(dataset.orders["order_date"].quantile(0.5, interpolation="lower"))
    loaded = adapter.load(cutoff)
    assert (loaded["orders"]["order_date"] <= cutoff).all()
    assert loaded["order_lines"]["order_id"].isin(loaded["orders"]["order_id"]).all()
    assert set(loaded["orders"]["order_id"]) <= set(dataset.orders["order_id"])
    assert len(loaded["orders"]) < len(dataset.orders)


def test_wms_adapter_redacts_future_event_timestamps(small_trained_context):
    dataset = small_trained_context["dataset"]
    data_dir = small_trained_context["data_dir"]
    adapter = LocalCsvWMSAdapter(
        data_dir / "dcs.csv", data_dir / "capacity_snapshots.csv", data_dir / "events.csv"
    )
    as_of = pd.Timestamp(dataset.orders["order_date"].min()) + pd.Timedelta(days=5)
    loaded = adapter.load(as_of)
    assert set(loaded["events"]["event_type"].unique()) <= {"SHIPPED"}
    occurred = loaded["events"]["event_timestamp"].dropna()
    assert (occurred <= as_of).all()


def test_tms_and_srm_adapters_own_disjoint_event_types(small_trained_context):
    data_dir = small_trained_context["data_dir"]
    dataset = small_trained_context["dataset"]
    as_of = pd.Timestamp(dataset.orders["order_date"].max())
    tms = LocalCsvTMSAdapter(data_dir / "lanes.csv", data_dir / "events.csv")
    srm = LocalCsvSRMAdapter(
        data_dir / "vendors.csv", data_dir / "skus.csv", data_dir / "events.csv"
    )
    tms_events = tms.load(as_of)["events"]
    srm_events = srm.load(as_of)["events"]
    assert set(tms_events["event_type"].unique()) <= {"IN_TRANSIT", "DELIVERED"}
    assert set(srm_events["event_type"].unique()) <= {"VENDOR_READY"}


def test_assemble_source_tables_merges_events_and_requires_all_tables(small_trained_context):
    data_dir = small_trained_context["data_dir"]
    dataset = small_trained_context["dataset"]
    as_of = pd.Timestamp(dataset.orders["order_date"].max())
    adapters = default_adapters(data_dir)
    tables = assemble_source_tables(adapters, as_of)
    assert set(tables) == {
        "orders",
        "order_lines",
        "events",
        "vendors",
        "dcs",
        "lanes",
        "customers",
        "skus",
        "capacity_snapshots",
    }
    assert set(tables["events"]["event_type"].unique()) == {
        "VENDOR_READY",
        "SHIPPED",
        "IN_TRANSIT",
        "DELIVERED",
    }


def test_assemble_source_tables_raises_on_duplicate_table():
    class DuplicateAdapter:
        source_name = "dup"

        def load(self, as_of_timestamp):
            return {"orders": pd.DataFrame({"order_id": ["O-1"]})}

    with pytest.raises(ValueError, match="duplicate source table"):
        assemble_source_tables([DuplicateAdapter(), DuplicateAdapter()], pd.Timestamp("2024-01-01"))


def test_assemble_prototype_dataset_uses_empty_truth_placeholders_by_default(
    small_trained_context,
):
    dataset = small_trained_context["dataset"]
    data_dir = small_trained_context["data_dir"]
    as_of = pd.Timestamp(dataset.orders["order_date"].max())
    assembled = assemble_prototype_dataset(default_adapters(data_dir), as_of)
    assert assembled.simulator_truth.empty
    assert assembled.line_truth.empty
    assert assembled.shocks.empty


def test_data_quality_report_flags_duplicate_primary_key(small_trained_context):
    data_dir = small_trained_context["data_dir"]
    dataset = small_trained_context["dataset"]
    as_of = pd.Timestamp(dataset.orders["order_date"].max())
    tables = assemble_source_tables(default_adapters(data_dir), as_of)
    report = data_quality_report(tables)
    assert report["passed"] is True
    assert report["contract_failure_count"] == 0

    tampered = dict(tables)
    tampered["orders"] = pd.concat(
        [tampered["orders"], tampered["orders"].iloc[[0]]], ignore_index=True
    )
    tampered_report = data_quality_report(tampered)
    assert tampered_report["passed"] is False
    assert tampered_report["uniqueness_by_table"]["orders"]["duplicate_keys"] == 1


def test_score_request_validation_rejects_bad_input():
    with pytest.raises(ContractError):
        ScoreRequest(
            as_of_timestamp=pd.Timestamp("2024-01-01"),
            order_ids=(),
            source_snapshot_id="snap-1",
            idempotency_key="key-1",
        ).validate()
    with pytest.raises(ContractError):
        ScoreRequest(
            as_of_timestamp=pd.Timestamp("2024-01-01"),
            order_ids=("O-1", "O-1"),
            source_snapshot_id="snap-1",
            idempotency_key="key-1",
        ).validate()
    with pytest.raises(ContractError):
        ScoreRequest(
            as_of_timestamp=pd.Timestamp("2024-01-01"),
            order_ids=("O-1",),
            source_snapshot_id="",
            idempotency_key="key-1",
        ).validate()


def test_offline_and_service_boundary_produce_identical_features_and_scores(
    small_trained_context,
):
    """Explicit offline/batch parity: the same order/as-of snapshot through the
    canonical feature builder (in-process dataset) and through the service
    boundary (adapter-sourced, CSV round-tripped dataset) must produce an
    identical feature vector and an identical score."""
    dataset = small_trained_context["dataset"]
    outcomes = small_trained_context["outcomes"]
    causes = small_trained_context["causes"]
    trained = small_trained_context["trained"]
    data_dir = small_trained_context["data_dir"]

    as_of = pd.Timestamp(dataset.orders["order_date"].max())
    order_ids = pd.Index(dataset.orders["order_id"].iloc[:15])

    # Offline path: build_feature_table called directly on the in-memory dataset.
    offline_features = build_feature_table(
        dataset, outcomes, causes, as_of_timestamp=as_of, order_ids=order_ids
    )
    offline_features = attach_line_evidence_features(dataset, offline_features)
    offline_scored = score_orders(
        dataset,
        offline_features,
        trained.risk_training.bundle,
        trained.bayesian_bundle,
        trained.fusion_selection.chosen_weight,
        background=offline_features,
    )

    # Service-boundary path: adapters re-read the same tables from persisted CSVs.
    assembled = assemble_prototype_dataset(
        default_adapters(data_dir), as_of, truth_tables=dataset.truth_tables()
    )
    request = ScoreRequest(
        as_of_timestamp=as_of,
        order_ids=tuple(order_ids),
        source_snapshot_id="snapshot-parity-test",
        idempotency_key="parity-key-1",
    )
    responses = score_via_service(
        request,
        assembled,
        outcomes,
        causes,
        trained.risk_training.bundle,
        trained.bayesian_bundle,
        trained.fusion_selection.chosen_weight,
        model_version="v-test",
        policy_version="p-test",
        manifest_content_id="content-test",
        background=offline_features,
    )

    service_scores = {response.order_id: response.risk_score for response in responses}
    offline_scores = (
        offline_scored.set_index("order_id")["combined_risk_score"].astype(float).to_dict()
    )
    assert set(service_scores) == set(offline_scores)
    for order_id, offline_score in offline_scores.items():
        assert service_scores[order_id] == pytest.approx(offline_score, abs=1e-9)

    # Feature-vector parity: rebuild the service-side feature table directly
    # (not just via the response) and compare numeric columns cell-for-cell.
    service_features = build_feature_table(
        assembled, outcomes, causes, as_of_timestamp=as_of, order_ids=order_ids
    )
    common_columns = [
        c for c in offline_features.columns if c in service_features.columns and c != "order_id"
    ]
    left = offline_features.set_index("order_id")[common_columns].sort_index()
    right = service_features.set_index("order_id")[common_columns].sort_index()
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_jsonl_decision_sink_upsert_is_idempotent_on_retry(tmp_path):
    from otif_risk.service_contracts import ScoreResponse

    sink_path = tmp_path / "decisions.jsonl"
    response = ScoreResponse(
        idempotency_key="key-1",
        order_id="O-1",
        as_of_timestamp="2024-01-01T00:00:00",
        source_snapshot_id="snap-1",
        model_version="v1",
        policy_version="p1",
        manifest_content_id="c1",
        risk_score=0.42,
        threshold=0.5,
        confidence="HIGH",
        explanation=[],
        decision_status="MONITOR",
        recommended_action=None,
        resource_type=None,
        resource_status="MONITOR",
    )

    sink = JsonlDecisionSink(sink_path)
    written_first = sink.write([response])
    written_retry = sink.write([response])

    assert written_first == 1
    assert written_retry == 0
    lines = sink_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    # A brand-new sink instance re-reading the same file also stays idempotent.
    reloaded = JsonlDecisionSink(sink_path)
    reloaded.write([response])
    assert len(sink_path.read_text(encoding="utf-8").splitlines()) == 1
    assert reloaded.read_all()[0]["order_id"] == "O-1"


def test_csv_decision_sink_upsert_is_idempotent_on_retry(tmp_path):
    from otif_risk.service_contracts import ScoreResponse

    sink_path = tmp_path / "decisions.csv"
    response = ScoreResponse(
        idempotency_key="key-1",
        order_id="O-1",
        as_of_timestamp="2024-01-01T00:00:00",
        source_snapshot_id="snap-1",
        model_version="v1",
        policy_version="p1",
        manifest_content_id="c1",
        risk_score=0.42,
        threshold=0.5,
        confidence="HIGH",
        explanation=[],
        decision_status="MONITOR",
        recommended_action=None,
        resource_type=None,
        resource_status="MONITOR",
    )
    updated_response = ScoreResponse(**{**response.__dict__, "risk_score": 0.91})

    sink = CsvDecisionSink(sink_path)
    sink.write([response])
    sink.write([updated_response])

    rows = sink.read_all()
    assert len(rows) == 1
    assert float(rows[0]["risk_score"]) == pytest.approx(0.91)

    reloaded = CsvDecisionSink(sink_path)
    assert len(reloaded.read_all()) == 1


def test_decision_sink_rejects_nothing_on_empty_write(tmp_path):
    sink = JsonlDecisionSink(tmp_path / "empty.jsonl")
    assert sink.write([]) == 0
    assert sink.read_all() == []
