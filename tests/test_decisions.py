from __future__ import annotations

import pandas as pd
import pytest

from otif_pdf.decisions import (
    CONTESTED,
    MONITOR,
    RECOMMENDED,
    build_rollups,
    recommend_orders,
    service_impact_summary,
)


def scored_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O-1", "O-2", "O-3", "O-4"],
            "combined_risk_score": [0.90, 0.80, 0.70, 0.20],
            "primary_cause": ["TRANSPORT", "TRANSPORT", "TRANSPORT", "VENDOR_FAILURE"],
            "lane_id": ["L-1", "L-1", "L-1", "L-2"],
            "vendor_id": ["V-1", "V-1", "V-2", "V-2"],
            "dc_id": ["D-1", "D-1", "D-2", "D-2"],
            "customer_id": ["C-1", "C-2", "C-3", "C-4"],
            "customer_tier": ["GOLD", "SILVER", "SILVER", "PLATINUM"],
            "order_value": [10_000, 8_000, 6_000, 5_000],
            "total_order_qty": [100, 80, 60, 50],
            "penalty_rate": [0.05, 0.05, 0.05, 0.05],
            "order_priority": ["EXPEDITE", "STANDARD", "STANDARD", "STANDARD"],
        }
    )


def order_lines_for(scored: pd.DataFrame) -> pd.DataFrame:
    """Two SKU lines per order so the SKU rollup exercises exploded logic."""
    rows = []
    for order_id in scored["order_id"]:
        rows.append({"order_id": order_id, "sku_id": "SKU-A"})
        rows.append({"order_id": order_id, "sku_id": f"SKU-{order_id}"})
    return pd.DataFrame(rows)


def test_recommendations_and_shared_resource_conflicts_are_deterministic() -> None:
    decisions = recommend_orders(scored_orders(), resource_limits={"lane": 1})

    assert decisions["decision_status"].tolist() == [
        RECOMMENDED,
        CONTESTED,
        CONTESTED,
        MONITOR,
    ]
    assert decisions.loc[0, "recommended_action"].startswith("Secure alternate")
    assert decisions.loc[0, "priority_score"] > decisions.loc[1, "priority_score"]
    assert decisions.loc[0, "estimated_penalty_exposure"] == pytest.approx(450.0)
    assert decisions.loc[3, "estimated_avoidable_penalty"] == 0


def test_dc_conflicts_are_capacity_and_quantity_aware() -> None:
    """DC conflicts must be driven by real capacity/quantity, not a headcount.

    Four orders share one DC with a small real daily capacity. Their combined
    `quantity_at_risk` deliberately exceeds the DC's recovery allowance
    (20% of daily capacity by default) once more than one order is admitted,
    so at least one must be CONTESTED even though the count-based limit
    (`resource_limits["dc"]`) is left generous enough that a naive headcount
    rule would recommend all four.
    """
    orders = pd.DataFrame(
        {
            "order_id": ["O-1", "O-2", "O-3", "O-4"],
            "combined_risk_score": [0.95, 0.90, 0.85, 0.80],
            "primary_cause": ["DC_CAPACITY"] * 4,
            "dc_id": ["D-1"] * 4,
            "dc_daily_capacity_units": [200] * 4,
            "total_order_qty": [80, 80, 80, 80],
            "order_value": [1_000, 1_000, 1_000, 1_000],
        }
    )
    # quantity_at_risk = qty * risk * 0.5, so each order contributes 32-38 units;
    # the DC's recovery allowance is 200 * 0.20 = 40 units. The top-priority
    # order (38) fits alone, but any second order pushes the cumulative total
    # past 40, guaranteeing a genuine capacity conflict.
    decisions = recommend_orders(orders, resource_limits={"dc": 10}, risk_threshold=0.5)

    assert decisions["decision_status"].tolist()[0] == RECOMMENDED
    assert CONTESTED in decisions["decision_status"].tolist()
    # The highest-priority order must still clear before capacity is exhausted.
    assert decisions.loc[decisions["decision_status"] == RECOMMENDED, "order_id"].tolist() == [
        "O-1"
    ]


def test_dc_conflicts_fall_back_to_count_limit_without_capacity_data() -> None:
    """Without `dc_daily_capacity_units`, DC keeps the documented count-based limit."""
    orders = pd.DataFrame(
        {
            "order_id": ["O-1", "O-2", "O-3", "O-4"],
            "combined_risk_score": [0.90, 0.80, 0.70, 0.20],
            "primary_cause": ["DC_CAPACITY", "DC_CAPACITY", "WAREHOUSE_OPS", "VENDOR_FAILURE"],
            "dc_id": ["D-1", "D-1", "D-2", "D-2"],
        }
    )
    decisions = recommend_orders(orders, resource_limits={"dc": 1})

    dc_1_orders = decisions.loc[decisions["dc_id"] == "D-1"]
    assert dc_1_orders["decision_status"].tolist() == [RECOMMENDED, CONTESTED]


def test_rollups_and_impact_include_auditable_assumptions() -> None:
    decisions = recommend_orders(scored_orders(), resource_limits={"lane": 2})

    rollups = build_rollups(decisions, order_lines=order_lines_for(scored_orders()))
    impact = service_impact_summary(decisions)

    assert set(rollups) == {"vendor", "dc", "lane", "customer", "order_type", "sku"}
    assert int(rollups["lane"].loc[0, "actionable_orders"]) == 3
    for entity in ("vendor", "dc", "lane", "customer", "order_type"):
        assert {
            "order_count",
            "actionable_orders",
            "pct_at_risk",
            "average_risk",
            "penalty_exposure",
            "value_at_risk",
            "quantity_at_risk",
            "dominant_cause",
        } <= set(rollups[entity].columns)
    assert rollups["lane"]["pct_at_risk"].between(0, 1).all()
    # SKU-A appears on every order, so its rollup row must aggregate all 4 orders.
    sku_a = rollups["sku"].loc[rollups["sku"]["sku_id"] == "SKU-A"].iloc[0]
    assert int(sku_a["order_count"]) == 4
    assert impact["recommended_orders"] == 2
    assert impact["contested_orders"] == 1
    assert "order value" in impact["assumptions"]["penalty"]
    assert "Quantity at risk" in impact["assumptions"]["service_impact"]


def test_missing_required_scored_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="primary_cause"):
        recommend_orders(scored_orders().drop(columns="primary_cause"))

