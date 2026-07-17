from __future__ import annotations

import pandas as pd
import pytest

from otif_risk.contracts import PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.resources import (
    ResourceCapacities,
    allocate_interventions,
    default_daily_capacities,
    demand_units_for,
)


def test_default_daily_capacities_uses_real_dc_throughput_and_flat_assumptions():
    dataset = generate_dataset(PrototypeConfig(seed=1, n_orders=250))
    capacities = default_daily_capacities(dataset)

    dc_row = dataset.dcs.iloc[0]
    expected_dc_units = pytest.approx(dc_row["daily_capacity_units"] * 0.20)
    assert capacities.dc_units[dc_row["dc_id"]] == expected_dc_units
    assert set(capacities.lane_units) == set(dataset.lanes["lane_id"])
    assert set(capacities.vendor_slots) == set(dataset.vendors["vendor_id"])
    assert set(capacities.customer_slots) == set(dataset.customers["customer_id"])


def test_demand_units_are_quantity_based_for_dc_and_lane_slot_based_otherwise():
    assert demand_units_for("dc", 42.0) == 42.0
    assert demand_units_for("lane", 10.0) == 10.0
    assert demand_units_for("vendor", 999.0) == 1.0
    assert demand_units_for("customer", 999.0) == 1.0


def test_allocate_interventions_marks_overflow_contested_with_competitors():
    capacities = ResourceCapacities(vendor_slots={"V001": 1.0})
    candidates = pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3"],
            "primary_cause": ["VENDOR_FAILURE"] * 3,
            "priority_score": [90, 80, 70],
            "quantity_at_risk": [5, 5, 5],
            "vendor_id": ["V001"] * 3,
        }
    )

    result, remaining = allocate_interventions(candidates, capacities)

    assert result.set_index("order_id").loc["O1", "decision_status"] == "RECOMMENDED"
    assert result.set_index("order_id").loc["O2", "decision_status"] == "CONTESTED"
    assert result.set_index("order_id").loc["O3", "decision_status"] == "CONTESTED"
    contested_with = result.set_index("order_id").loc["O2", "contested_with"]
    assert "O1" in contested_with
    assert "O3" in contested_with
    assert remaining.vendor_slots["V001"] == 0.0


def test_allocate_interventions_dc_pool_is_quantity_denominated():
    capacities = ResourceCapacities(dc_units={"D1": 100.0})
    candidates = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "primary_cause": ["INVENTORY_SHORTAGE", "INVENTORY_SHORTAGE"],
            "priority_score": [90, 80],
            "quantity_at_risk": [60.0, 60.0],
            "dc_id": ["D1", "D1"],
        }
    )

    result, remaining = allocate_interventions(candidates, capacities)

    assert result.set_index("order_id").loc["O1", "decision_status"] == "RECOMMENDED"
    assert result.set_index("order_id").loc["O2", "decision_status"] == "CONTESTED"
    assert remaining.dc_units["D1"] == pytest.approx(40.0)


def test_allocate_interventions_requires_expected_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        allocate_interventions(pd.DataFrame({"order_id": ["O1"]}), ResourceCapacities())


def test_missing_resource_id_falls_back_to_zero_capacity_and_contests():
    capacities = ResourceCapacities()
    candidates = pd.DataFrame(
        {
            "order_id": ["O1"],
            "primary_cause": ["VENDOR_FAILURE"],
            "priority_score": [50],
            "quantity_at_risk": [1.0],
        }
    )
    result, _remaining = allocate_interventions(candidates, capacities)
    assert result.loc[0, "resource_id"] == "UNASSIGNED"
    assert result.loc[0, "decision_status"] == "CONTESTED"
