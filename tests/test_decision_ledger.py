from __future__ import annotations

import pandas as pd

from otif_risk.decision_ledger import (
    LEDGER_COLUMNS,
    append_entries,
    build_ledger_entry,
    derive_decision_id,
    ledger_decision_key,
    observational_cohort_report,
    reconcile_outcomes,
)


def _entry(order_id="O-1", day="2024-02-01", status="RECOMMENDED", model_version="v1"):
    return build_ledger_entry(
        order_id=order_id,
        decision_day=day,
        source_snapshot_id="snap-1",
        model_version=model_version,
        policy_version="policy-v1",
        manifest_content_id="content-1",
        feasible_actions=["VENDOR_ESCALATION", "WAREHOUSE_EXPEDITE"],
        chosen_action="VENDOR_ESCALATION",
        risk_score=0.7,
        threshold=0.5,
        decision_status=status,
        resource_type="vendor",
        resource_id="V-1",
        resource_demand_units=1.0,
        resource_capacity_before=1.0,
        resource_capacity_after=0.0,
        selection_mode="EXPLOIT",
        assignment_probability=1.0,
    )


def test_derive_decision_id_is_deterministic_for_same_key():
    key = ledger_decision_key("O-1", "2024-02-01", "snap-1", "v1")
    assert derive_decision_id(key) == derive_decision_id(key)
    other_key = ledger_decision_key("O-2", "2024-02-01", "snap-1", "v1")
    assert derive_decision_id(key) != derive_decision_id(other_key)


def test_build_ledger_entry_maps_status_to_planner_and_execution():
    recommended = _entry(status="RECOMMENDED")
    assert recommended.planner_decision == "ACCEPTED"
    assert recommended.execution_status == "EXECUTED"
    assert recommended.chosen_action == "VENDOR_ESCALATION"
    assert recommended.rejected_actions == ["WAREHOUSE_EXPEDITE"]

    contested = _entry(order_id="O-2", status="CONTESTED")
    assert contested.planner_decision == "REJECTED"
    assert contested.execution_status == "NOT_EXECUTED"
    assert contested.chosen_action is None

    monitored = _entry(order_id="O-3", status="MONITOR")
    assert monitored.planner_decision == "MONITORED"
    assert monitored.execution_status == "NOT_EXECUTED"


def test_append_entries_is_idempotent_on_retry(tmp_path):
    path = tmp_path / "ledger.csv"
    entry = _entry()

    first_write = append_entries(path, [entry])
    retry_write = append_entries(path, [entry])

    assert first_write == 1
    assert retry_write == 0
    ledger = pd.read_csv(path)
    assert len(ledger) == 1
    assert list(ledger.columns) == list(LEDGER_COLUMNS)


def test_append_entries_overwrites_mutable_fields_on_same_key(tmp_path):
    path = tmp_path / "ledger.csv"
    entry = _entry()
    append_entries(path, [entry])

    entry.matured = True
    entry.matured_otif_miss = 1
    entry.realized_penalty = 42.0
    append_entries(path, [entry])

    ledger = pd.read_csv(path)
    assert len(ledger) == 1
    assert bool(ledger.iloc[0]["matured"]) is True
    assert ledger.iloc[0]["realized_penalty"] == 42.0


def test_append_entries_multiple_orders_and_days_all_persist(tmp_path):
    path = tmp_path / "ledger.csv"
    entries = [
        _entry(order_id="O-1", day="2024-02-01"),
        _entry(order_id="O-2", day="2024-02-01", status="CONTESTED"),
        _entry(order_id="O-1", day="2024-02-02"),  # same order, later day: new key
    ]
    written = append_entries(path, entries)
    assert written == 3
    ledger = pd.read_csv(path)
    assert len(ledger) == 3


def _outcomes_causes_for(order_ids, otif_miss, outcome_day="2024-02-05"):
    outcomes = pd.DataFrame(
        {
            "order_id": order_ids,
            "outcome_timestamp": [pd.Timestamp(outcome_day)] * len(order_ids),
            "otif_miss": otif_miss,
            "penalty_rate": [0.05] * len(order_ids),
            "order_value": [1000.0] * len(order_ids),
        }
    )
    causes = pd.DataFrame(
        {"order_id": order_ids, "primary_cause": ["VENDOR_FAILURE"] * len(order_ids)}
    )
    return outcomes, causes


def test_reconcile_outcomes_updates_only_matured_orders(tmp_path):
    path = tmp_path / "ledger.csv"
    append_entries(path, [_entry(order_id="O-1", day="2024-02-01")])
    append_entries(path, [_entry(order_id="O-2", day="2024-02-01")])

    outcomes, causes = _outcomes_causes_for(["O-1", "O-2"], [1, 0], outcome_day="2024-02-03")
    # O-2's outcome matures later than our as_of cutoff.
    outcomes.loc[outcomes["order_id"] == "O-2", "outcome_timestamp"] = pd.Timestamp("2024-02-10")

    result = reconcile_outcomes(path, outcomes, causes, as_of_timestamp=pd.Timestamp("2024-02-05"))
    assert result["newly_matured"] == 1
    assert result["still_open"] == 1

    ledger = pd.read_csv(path).set_index("order_id")
    assert bool(ledger.loc["O-1", "matured"]) is True
    assert ledger.loc["O-1", "matured_otif_miss"] == 1
    assert ledger.loc["O-1", "realized_penalty"] == 50.0
    assert bool(ledger.loc["O-2", "matured"]) is False


def test_reconcile_outcomes_is_idempotent(tmp_path):
    path = tmp_path / "ledger.csv"
    append_entries(path, [_entry(order_id="O-1", day="2024-02-01")])
    outcomes, causes = _outcomes_causes_for(["O-1"], [1], outcome_day="2024-02-03")

    first = reconcile_outcomes(path, outcomes, causes, as_of_timestamp=pd.Timestamp("2024-02-05"))
    second = reconcile_outcomes(path, outcomes, causes, as_of_timestamp=pd.Timestamp("2024-02-05"))

    assert first["newly_matured"] == 1
    assert second["newly_matured"] == 0
    assert second["already_matured"] == 1


def test_observational_cohort_report_withholds_small_cohorts(tmp_path):
    path = tmp_path / "ledger.csv"
    entries = [_entry(order_id=f"O-{i}", status="RECOMMENDED") for i in range(5)]
    append_entries(path, entries)
    outcomes, causes = _outcomes_causes_for(
        [f"O-{i}" for i in range(5)], [1, 0, 1, 0, 0], outcome_day="2024-01-01"
    )
    reconcile_outcomes(path, outcomes, causes, as_of_timestamp=pd.Timestamp("2024-03-01"))

    report = observational_cohort_report(path, min_sample=20)
    assert report["observational_not_causal"] is True
    assert report["cohorts"]["ACCEPTED"]["sufficient_sample"] is False
    assert report["cohorts"]["ACCEPTED"]["miss_rate"] is None
    assert report["cohorts"]["ACCEPTED"]["n_matured_decisions"] == 5


def test_observational_cohort_report_reports_rate_when_sample_sufficient(tmp_path):
    path = tmp_path / "ledger.csv"
    order_ids = [f"O-{i}" for i in range(25)]
    entries = [_entry(order_id=oid, status="RECOMMENDED") for oid in order_ids]
    append_entries(path, entries)
    misses = [1 if i % 2 == 0 else 0 for i in range(25)]
    outcomes, causes = _outcomes_causes_for(order_ids, misses, outcome_day="2024-01-01")
    reconcile_outcomes(path, outcomes, causes, as_of_timestamp=pd.Timestamp("2024-03-01"))

    report = observational_cohort_report(path, min_sample=20)
    accepted = report["cohorts"]["ACCEPTED"]
    assert accepted["sufficient_sample"] is True
    assert accepted["miss_rate"] == pytest_approx(13 / 25)
    assert accepted["total_realized_penalty"] is not None
    qualification = report["qualification"].lower()
    assert "non-causal" in qualification or "causal" in qualification


def pytest_approx(value):
    import pytest

    return pytest.approx(round(value, 4), abs=1e-4)


def test_observational_cohort_report_empty_ledger(tmp_path):
    report = observational_cohort_report(tmp_path / "missing.csv")
    assert report["total_decisions_logged"] == 0
    assert report["observational_not_causal"] is True
