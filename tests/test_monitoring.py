from __future__ import annotations

import numpy as np
import pandas as pd

from otif_risk.decision_ledger import LEDGER_COLUMNS
from otif_risk.monitoring import (
    MIN_ROLLING_SAMPLE,
    build_monitoring_report,
    feature_freshness,
    regime_quality,
    rolling_prediction_quality,
    runtime_metrics,
    time_to_detection,
    write_monitoring_report,
)


def _synthetic_ledger(n: int, *, start="2024-01-01", miss_rate_seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(miss_rate_seed)
    days = pd.date_range(start, periods=n, freq="D")
    scores = rng.uniform(0, 1, size=n)
    labels = (rng.uniform(0, 1, size=n) < scores * 0.5).astype(int)
    planner = np.where(scores >= 0.5, "ACCEPTED", "MONITORED")
    frame = pd.DataFrame(
        {
            "decision_key": [f"key-{i}" for i in range(n)],
            "order_id": [f"O-{i}" for i in range(n)],
            "decision_timestamp": [day.isoformat() for day in days],
            "risk_score": scores,
            "threshold": [0.5] * n,
            "planner_decision": planner,
            "matured": [True] * n,
            "matured_otif_miss": labels,
            "realized_penalty": [100.0 if miss else 0.0 for miss in labels],
        }
    )
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame


def test_rolling_prediction_quality_withholds_small_windows():
    ledger = _synthetic_ledger(10)
    windows = rolling_prediction_quality(ledger, window_days=30, min_sample=MIN_ROLLING_SAMPLE)
    assert len(windows) == 1
    assert windows[0]["sufficient_sample"] is False
    assert "pr_auc" not in windows[0]


def test_rolling_prediction_quality_reports_metrics_when_sample_sufficient():
    ledger = _synthetic_ledger(60)
    windows = rolling_prediction_quality(ledger, window_days=90, min_sample=MIN_ROLLING_SAMPLE)
    assert len(windows) == 1
    assert windows[0]["sufficient_sample"] is True
    assert 0.0 <= windows[0]["pr_auc"] <= 1.0
    assert 0.0 <= windows[0]["calibration_error"] <= 1.0
    assert 0.0 <= windows[0]["alert_rate"] <= 1.0


def test_rolling_prediction_quality_empty_ledger():
    assert rolling_prediction_quality(pd.DataFrame()) == []


def test_regime_quality_splits_normal_and_drift_by_order_date():
    ledger = _synthetic_ledger(80)
    order_dates = pd.Series(
        pd.date_range("2024-01-01", periods=80, freq="D"),
        index=[f"O-{i}" for i in range(80)],
    )
    result = regime_quality(ledger, order_dates, min_sample=10)
    assert set(result) <= {"normal", "drift"}
    for regime_result in result.values():
        assert "n_matured_decisions" in regime_result


def test_regime_quality_withholds_small_regime_cohort():
    ledger = _synthetic_ledger(15)
    order_dates = pd.Series(
        pd.date_range("2024-01-01", periods=15, freq="D"),
        index=[f"O-{i}" for i in range(15)],
    )
    result = regime_quality(ledger, order_dates, min_sample=MIN_ROLLING_SAMPLE)
    for regime_result in result.values():
        assert regime_result["sufficient_sample"] is False


def test_time_to_detection_computes_lead_time_for_missed_flagged_orders():
    ledger = pd.DataFrame(
        {
            "order_id": [f"O-{i}" for i in range(25)],
            "decision_timestamp": ["2024-01-01"] * 25,
            "planner_decision": ["ACCEPTED"] * 25,
            "matured": [True] * 25,
            "matured_otif_miss": [1] * 25,
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [f"O-{i}" for i in range(25)],
            "promised_delivery_date": ["2024-01-04"] * 25,
        }
    )
    report = time_to_detection(ledger, orders, min_sample=20)
    assert report["sufficient_sample"] is True
    assert report["median_lead_time_hours_before_promised_delivery"] == 72.0


def test_time_to_detection_withholds_below_min_sample():
    ledger = pd.DataFrame(
        {
            "order_id": ["O-1"],
            "decision_timestamp": ["2024-01-01"],
            "planner_decision": ["ACCEPTED"],
            "matured": [True],
            "matured_otif_miss": [1],
        }
    )
    orders = pd.DataFrame({"order_id": ["O-1"], "promised_delivery_date": ["2024-01-04"]})
    report = time_to_detection(ledger, orders, min_sample=20)
    assert report["sufficient_sample"] is False


def test_feature_freshness_reports_structural_cadence():
    report = feature_freshness([{"simulated_day": "2024-01-01"}] * 12)
    assert report["scoring_cadence_days"] == 1.0
    assert report["days_replayed"] == 12


def test_runtime_metrics_labeled_local_only():
    report = runtime_metrics([0.1, 0.2, 0.15], [2.0, 3.0])
    assert report["scope"] == "measured_local_runtime_only_not_a_production_latency_claim"
    assert report["scoring_runtime"]["n"] == 3
    assert report["retrain_runtime"]["mean_seconds"] == 2.5


def test_runtime_metrics_handles_missing_samples():
    report = runtime_metrics(None, None)
    assert report["scoring_runtime"] is None
    assert report["retrain_runtime"] is None


def test_build_monitoring_report_aggregates_and_persists(tmp_path):
    ledger = _synthetic_ledger(60)
    order_dates = pd.Series(
        pd.date_range("2024-01-01", periods=60, freq="D"),
        index=[f"O-{i}" for i in range(60)],
    )
    orders = pd.DataFrame(
        {
            "order_id": [f"O-{i}" for i in range(60)],
            "promised_delivery_date": pd.date_range("2024-01-05", periods=60, freq="D"),
        }
    )
    daily_log = [{"simulated_day": str(day)} for day in range(60)]
    data_quality = {"contract_failure_count": 0, "passed": True}

    report = build_monitoring_report(
        ledger=ledger,
        orders=orders,
        order_dates=order_dates,
        daily_log=daily_log,
        data_quality=data_quality,
        scoring_seconds=[0.5, 0.6],
        retrain_seconds=[5.0],
        window_days=90,
        min_sample=30,
    )
    assert report["monitoring_schema_version"] == "1.0"
    assert "slo_status" in report
    assert report["slo_status"]["contract_failure_count"]["passed"] is True
    runtime_scope = "measured_local_runtime_only_not_a_production_latency_claim"
    assert report["runtime"]["scope"] == runtime_scope

    output_path = write_monitoring_report(tmp_path / "monitoring_report.json", report)
    assert output_path.is_file()
    import json

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["monitoring_schema_version"] == "1.0"


def test_slo_status_fails_when_contract_failures_present(tmp_path):
    ledger = _synthetic_ledger(5)
    order_dates = pd.Series(
        pd.date_range("2024-01-01", periods=5, freq="D"), index=[f"O-{i}" for i in range(5)]
    )
    orders = pd.DataFrame(
        {
            "order_id": [f"O-{i}" for i in range(5)],
            "promised_delivery_date": pd.date_range("2024-01-05", periods=5, freq="D"),
        }
    )
    report = build_monitoring_report(
        ledger=ledger,
        orders=orders,
        order_dates=order_dates,
        daily_log=[],
        data_quality={"contract_failure_count": 3, "passed": False},
    )
    assert report["slo_status"]["contract_failure_count"]["passed"] is False
    assert report["slo_status"]["contract_failure_count"]["value"] == 3
