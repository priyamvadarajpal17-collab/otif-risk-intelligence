from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import otif_pdf.app
from otif_pdf.app import latest_run_directory, load_run_artifacts


def _write_fake_run(root: Path) -> Path:
    run = root / "run-20260715-120000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps({"model_name": "prototype ensemble", "roc_auc": 0.82}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "order_id": ["O-1", "O-2"],
            "combined_risk_score": [0.82, 0.31],
            "xgb_risk_score": [0.80, 0.35],
            "bbn_risk_score": [0.85, 0.27],
            "primary_cause": ["TRANSPORT", "INVENTORY_SHORTAGE"],
            "causal_pathway": ["lane → delay", "stock → short shipment"],
            "top_factors_json": ['["carrier_delay"]', '["inventory_gap"]'],
            "vendor_id": ["V-1", "V-2"],
            "dc_id": ["D-1", "D-2"],
            "lane_id": ["L-1", "L-2"],
            "customer_id": ["C-1", "C-2"],
            "order_value": [10_000, 4_000],
            "total_order_qty": [100, 40],
            "customer_tier": ["GOLD", "SILVER"],
            "penalty_rate": [0.05, 0.02],
        }
    ).to_csv(run / "scored_orders.csv", index=False)
    return run


def _write_fake_run_with_persisted_decisions(root: Path, *, fused_threshold: float) -> Path:
    """A run whose scored_orders.csv already carries the pipeline's own decisions.

    The persisted `decision_status` deliberately disagrees with what
    `recommend_orders(scored_orders)` would compute at the default 0.5
    threshold, so a test can prove the UI reused the persisted value instead
    of silently recomputing a different policy.
    """
    run = root / "run-persisted-decisions"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps({"model_name": "prototype ensemble", "threshold": fused_threshold}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "order_id": ["O-1", "O-2"],
            # O-1's risk (0.55) is above the naive 0.5 default but the pipeline's
            # own persisted decision intentionally marks it MONITOR (e.g. it was
            # contested and downgraded); a silent recompute would flip it.
            "combined_risk_score": [0.55, 0.10],
            "primary_cause": ["TRANSPORT", "INVENTORY_SHORTAGE"],
            "causal_pathway": ["lane → delay", "stock → short shipment"],
            "top_factors_json": ['["carrier_delay"]', '["inventory_gap"]'],
            "vendor_id": ["V-1", "V-2"],
            "dc_id": ["D-1", "D-2"],
            "lane_id": ["L-1", "L-2"],
            "customer_id": ["C-1", "C-2"],
            "order_value": [10_000, 4_000],
            "total_order_qty": [100, 40],
            "customer_tier": ["GOLD", "SILVER"],
            "penalty_rate": [0.05, 0.02],
            "decision_status": ["MONITOR", "MONITOR"],
            "recommended_action": ["Held for review", "Held for review"],
            "action_owner": ["OTIF control tower", "OTIF control tower"],
            "resource_type": ["lane", "dc"],
            "resource_id": ["L-1", "D-2"],
            "priority_score": [10.0, 1.0],
            "estimated_penalty_exposure": [0.0, 0.0],
            "estimated_avoidable_penalty": [0.0, 0.0],
            "quantity_at_risk": [0.0, 0.0],
        }
    ).to_csv(run / "scored_orders.csv", index=False)
    return run


def test_latest_run_requires_metrics_and_selects_newest(tmp_path) -> None:
    old = _write_fake_run(tmp_path)
    incomplete = tmp_path / "run-99999999-999999"
    incomplete.mkdir()

    assert latest_run_directory(tmp_path) == old


def test_load_run_artifacts_reuses_persisted_decisions_without_recomputing(tmp_path) -> None:
    """Item 5: the UI must not silently override a persisted pipeline decision."""
    _write_fake_run_with_persisted_decisions(tmp_path, fused_threshold=0.90)

    _, _, decisions = load_run_artifacts(str(tmp_path.resolve()))

    # recommend_orders at the default 0.5 threshold would flag O-1 (risk 0.55)
    # as RECOMMENDED; the persisted MONITOR must be preserved unchanged.
    assert decisions.set_index("order_id").loc["O-1", "decision_status"] == "MONITOR"
    assert decisions.set_index("order_id").loc["O-1", "recommended_action"] == "Held for review"


def test_load_run_artifacts_computes_from_persisted_fused_threshold_when_absent(tmp_path) -> None:
    """When decisions are absent, the fallback must use metrics.json's fused threshold."""
    run = _write_fake_run(tmp_path)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["threshold"] = 0.90
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    _, _, decisions = load_run_artifacts(str(tmp_path.resolve()))

    # Both orders (0.82 and 0.31) are below the persisted 0.90 threshold.
    assert set(decisions["decision_status"]) == {"MONITOR"}


def test_streamlit_three_view_smoke(tmp_path, monkeypatch) -> None:
    _write_fake_run(tmp_path)
    monkeypatch.setenv("OTIF_ARTIFACTS_DIR", str(tmp_path))
    app_path = Path(otif_pdf.app.__file__)

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "OTIF Intervention Desk"
    assert app.header[0].value == "Order lookup"

    navigation = next(radio for radio in app.radio if "Ranked portfolio" in radio.options)
    navigation.set_value("Ranked portfolio").run()
    assert not app.exception
    assert any(header.value == "Ranked portfolio" for header in app.header)

    navigation = next(radio for radio in app.radio if "Hotspots + impact" in radio.options)
    navigation.set_value("Hotspots + impact").run()
    assert not app.exception
    assert any(header.value == "Hotspots + impact" for header in app.header)

