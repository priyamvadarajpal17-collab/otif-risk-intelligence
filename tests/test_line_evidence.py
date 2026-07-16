from __future__ import annotations

from otif_risk.contracts import PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.features import build_feature_table
from otif_risk.line_evidence import (
    affected_sku_summary,
    build_line_evidence,
    evaluate_line_evidence,
    order_line_aggregates,
)
from otif_risk.root_causes import calculate_outcomes, derive_root_causes


def _inputs(seed: int = 4, n_orders: int = 400):
    dataset = generate_dataset(PrototypeConfig(seed=seed, n_orders=n_orders))
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    features = build_feature_table(dataset, outcomes, causes)
    return dataset, features


def test_line_evidence_uses_only_capture_time_safe_fields():
    dataset, features = _inputs()
    evidence = build_line_evidence(dataset, features)

    assert set(evidence["order_id"]) <= set(dataset.orders["order_id"])
    assert {
        "order_line_id",
        "sku_id",
        "criticality_tier",
        "allocation_gap_ratio",
        "inventory_coverage_ratio",
        "evidence_strength",
        "likely_affected",
    } <= set(evidence.columns)
    assert evidence["evidence_strength"].between(0, 1).all()
    assert len(evidence) == len(dataset.order_lines)
    # shipped_qty (the retrospective truth) must never appear as a feature here.
    assert "shipped_qty" not in evidence.columns


def test_order_line_aggregates_are_safe_and_bounded():
    dataset, features = _inputs(seed=8)
    evidence = build_line_evidence(dataset, features)
    aggregates = order_line_aggregates(evidence)

    assert set(aggregates["order_id"]) <= set(dataset.orders["order_id"])
    assert aggregates["worst_line_shortage_ratio"].between(0, 1).all()
    assert aggregates["critical_sku_share"].between(0, 1).all()
    assert aggregates["line_qty_concentration"].between(0, 1.0001).all()
    assert (aggregates["affected_line_count"] >= 0).all()


def test_affected_sku_summary_only_lists_likely_affected_skus():
    dataset, features = _inputs(seed=12)
    evidence = build_line_evidence(dataset, features)
    summary = affected_sku_summary(evidence, top_n=2)

    assert {"order_id", "affected_sku_count", "affected_skus_json"} <= set(summary.columns)
    import json

    for row in summary.itertuples(index=False):
        parsed = json.loads(row.affected_skus_json)
        assert len(parsed) <= 2
        for item in parsed:
            assert item["evidence_strength"] >= 0


def test_evaluate_line_evidence_beats_the_naive_all_lines_baseline():
    dataset, features = _inputs(seed=21, n_orders=600)
    evidence = build_line_evidence(dataset, features)

    report = evaluate_line_evidence(evidence, dataset.line_truth)

    assert report["naive_all_lines_baseline"]["recall"] == 1.0
    assert (
        report["targeted_evidence"]["precision"]
        > report["naive_all_lines_baseline"]["precision"]
    )
    assert report["evaluated_lines"] == len(evidence)


def test_build_line_evidence_requires_leading_signal_columns():
    dataset, features = _inputs(seed=1)
    stripped = features.drop(columns=["leading_signal_VENDOR_FAILURE"])
    try:
        build_line_evidence(dataset, stripped)
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for missing leading_signal columns")
