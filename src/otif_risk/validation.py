"""Fail-fast integrity checks for prototype datasets."""

from __future__ import annotations

import pandas as pd

from .contracts import PrototypeDataset

REQUIRED_COLUMNS = {
    "orders": {
        "order_id",
        "order_date",
        "order_capture_timestamp",
        "prediction_timestamp",
        "requested_delivery_date",
        "promised_delivery_date",
        "vendor_id",
        "dc_id",
        "lane_id",
        "customer_id",
        "total_order_qty",
        # `leading_signal_*` columns are intentionally NOT part of the raw orders
        # table: they are derived point-in-time in features.py from operational
        # fields/events, not generated directly from the latent disruption cause.
    },
    "order_lines": {
        "order_line_id",
        "order_id",
        "requested_qty",
        "allocated_qty",
        "shipped_qty",
        "stockout_flag",
    },
    "events": {
        "order_id",
        "event_type",
        "planned_timestamp",
        "event_timestamp",
        "exception_code",
    },
    "vendors": {"vendor_id"},
    "dcs": {"dc_id", "daily_capacity_units"},
    "lanes": {"lane_id", "origin_dc_id", "planned_transit_days"},
    "customers": {"customer_id"},
    "skus": {"sku_id", "criticality_tier", "base_unit_value"},
    "capacity_snapshots": {
        "dc_id",
        "snapshot_date",
        "available_capacity_units",
        "planned_units",
        "utilization",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_dataset(dataset: PrototypeDataset) -> None:
    """Validate schema, keys, references, quantities, and core temporal invariants."""
    for table_name, frame in dataset.tables().items():
        _require(isinstance(frame, pd.DataFrame), f"{table_name} must be a pandas DataFrame")
        _require(not frame.empty, f"{table_name} must not be empty")
        missing = REQUIRED_COLUMNS[table_name] - set(frame.columns)
        _require(not missing, f"{table_name} missing required columns: {sorted(missing)}")

    unique_keys = {
        "orders": "order_id",
        "order_lines": "order_line_id",
        "vendors": "vendor_id",
        "dcs": "dc_id",
        "lanes": "lane_id",
        "customers": "customer_id",
        "skus": "sku_id",
    }
    for table_name, key in unique_keys.items():
        frame = getattr(dataset, table_name)
        _require(frame[key].notna().all(), f"{table_name}.{key} contains nulls")
        _require(frame[key].is_unique, f"{table_name}.{key} must be unique")

    order_ids = set(dataset.orders["order_id"])
    _require(
        set(dataset.order_lines["order_id"]).issubset(order_ids),
        "order_lines contains unknown order_id references",
    )
    _require(
        set(dataset.events["order_id"]).issubset(order_ids),
        "events contains unknown order_id references",
    )
    _require(
        set(dataset.orders["vendor_id"]).issubset(set(dataset.vendors["vendor_id"])),
        "orders contains unknown vendor_id references",
    )
    _require(
        set(dataset.orders["dc_id"]).issubset(set(dataset.dcs["dc_id"])),
        "orders contains unknown dc_id references",
    )
    _require(
        set(dataset.orders["lane_id"]).issubset(set(dataset.lanes["lane_id"])),
        "orders contains unknown lane_id references",
    )
    _require(
        set(dataset.orders["customer_id"]).issubset(set(dataset.customers["customer_id"])),
        "orders contains unknown customer_id references",
    )
    _require(
        set(dataset.capacity_snapshots["dc_id"]).issubset(set(dataset.dcs["dc_id"])),
        "capacity_snapshots contains unknown dc_id references",
    )
    _require(
        set(dataset.lanes["origin_dc_id"]).issubset(set(dataset.dcs["dc_id"])),
        "lanes contains unknown origin_dc_id references",
    )
    _require(
        set(dataset.order_lines["sku_id"]).issubset(set(dataset.skus["sku_id"])),
        "order_lines contains unknown sku_id references",
    )

    lines = dataset.order_lines
    _require((lines["requested_qty"] > 0).all(), "requested_qty must be positive")
    _require((lines["allocated_qty"] >= 0).all(), "allocated_qty must be non-negative")
    _require(
        (lines["allocated_qty"] <= lines["requested_qty"]).all(),
        "allocated_qty cannot exceed requested_qty",
    )
    _require(
        (lines["shipped_qty"] <= lines["requested_qty"]).all(),
        "shipped_qty cannot exceed requested_qty",
    )
    summed = lines.groupby("order_id")["requested_qty"].sum()
    declared = dataset.orders.set_index("order_id")["total_order_qty"]
    aligned = summed.reindex(declared.index)
    _require(
        aligned.notna().all() and aligned.eq(declared).all(),
        "order total_order_qty does not equal line requested quantities",
    )

    orders = dataset.orders
    _require(
        (orders["order_date"] <= orders["prediction_timestamp"]).all(),
        "prediction_timestamp precedes order_date",
    )
    _require(
        (orders["prediction_timestamp"] < orders["promised_delivery_date"]).all(),
        "prediction_timestamp must precede promised delivery",
    )
    _require(
        (orders["promised_delivery_date"] >= orders["requested_delivery_date"]).all(),
        "promised delivery cannot precede requested delivery",
    )
    _require(
        dataset.events["event_type"]
        .isin({"VENDOR_READY", "SHIPPED", "IN_TRANSIT", "DELIVERED"})
        .all(),
        "events contains unsupported event_type",
    )
    delivered_counts = (
        dataset.events.loc[dataset.events["event_type"] == "DELIVERED"]
        .groupby("order_id")
        .size()
        .reindex(orders["order_id"], fill_value=0)
    )
    _require((delivered_counts == 1).all(), "every order must have exactly one DELIVERED event")
    _require(
        dataset.capacity_snapshots[["dc_id", "snapshot_date"]].duplicated().sum() == 0,
        "capacity snapshot keys must be unique",
    )
    _require(
        (dataset.capacity_snapshots["available_capacity_units"] > 0).all(),
        "available capacity must be positive",
    )
