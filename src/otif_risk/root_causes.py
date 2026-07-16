"""Transparent OTIF outcome and root-cause rules."""

from __future__ import annotations

import json

import pandas as pd

from .contracts import CAUSE_CATEGORIES, PrototypeDataset
from .validation import validate_dataset

# Rules are evaluated independently, then this upstream-to-downstream order resolves primary cause.
CAUSE_PRIORITY = CAUSE_CATEGORIES


def calculate_outcomes(dataset: PrototypeDataset) -> pd.DataFrame:
    """Return one row per order with the OTIF target and its observable components."""
    validate_dataset(dataset)
    quantities = dataset.order_lines.groupby("order_id").agg(
        requested_qty=("requested_qty", "sum"),
        delivered_qty=("shipped_qty", "sum"),
    )
    delivered = (
        dataset.events.loc[dataset.events["event_type"] == "DELIVERED"]
        .set_index("order_id")["event_timestamp"]
        .rename("delivered_timestamp")
    )
    outcomes = (
        dataset.orders[["order_id", "prediction_timestamp", "promised_delivery_date"]]
        .set_index("order_id")
        .join(quantities)
        .join(delivered)
        .reset_index()
    )
    outcomes["on_time"] = (
        outcomes["delivered_timestamp"] <= outcomes["promised_delivery_date"]
    ).astype(int)
    outcomes["in_full"] = (outcomes["delivered_qty"] >= outcomes["requested_qty"]).astype(int)
    outcomes["otif_miss"] = ((outcomes["on_time"] == 0) | (outcomes["in_full"] == 0)).astype(int)
    outcomes["outcome_timestamp"] = outcomes["delivered_timestamp"]
    return outcomes


def _event_evidence(dataset: PrototypeDataset, event_type: str) -> pd.DataFrame:
    frame = dataset.events.loc[dataset.events["event_type"] == event_type].copy()
    frame["event_delay_hours"] = (
        frame["event_timestamp"] - frame["planned_timestamp"]
    ).dt.total_seconds() / 3600
    return frame.set_index("order_id")


def derive_root_causes(
    dataset: PrototypeDataset, outcomes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Evaluate every cause rule and assign an upstream-priority primary cause."""
    if outcomes is None:
        outcomes = calculate_outcomes(dataset)
    required = {"order_id", "otif_miss"}
    if missing := required - set(outcomes.columns):
        raise ValueError(f"outcomes missing required columns: {sorted(missing)}")

    orders = dataset.orders.set_index("order_id")
    lines = dataset.order_lines.groupby("order_id").agg(
        requested_qty=("requested_qty", "sum"),
        allocated_qty=("allocated_qty", "sum"),
        shipped_qty=("shipped_qty", "sum"),
        stockout_flag=("stockout_flag", "max"),
    )
    vendor = _event_evidence(dataset, "VENDOR_READY")
    shipped = _event_evidence(dataset, "SHIPPED")
    transit = _event_evidence(dataset, "IN_TRANSIT")
    delivered = _event_evidence(dataset, "DELIVERED")
    capacity = dataset.capacity_snapshots.set_index(["dc_id", "snapshot_date"])

    rows: list[dict[str, object]] = []
    for outcome in outcomes.itertuples(index=False):
        order_id = outcome.order_id
        order = orders.loc[order_id]
        line = lines.loc[order_id]
        snapshot_key = (order["dc_id"], order["order_date"].normalize())
        utilization = float(capacity.loc[snapshot_key, "utilization"])
        evidence = {
            "ORDER_CAPTURE": float(order["capture_delay_hours"]) > 24,
            "VENDOR_FAILURE": float(vendor.loc[order_id, "event_delay_hours"]) > 24
            or vendor.loc[order_id, "exception_code"] == "SUPPLIER_LATE",
            "INVENTORY_SHORTAGE": bool(line["stockout_flag"])
            or float(line["allocated_qty"]) < float(line["requested_qty"]),
            "DC_CAPACITY": utilization > 1.0,
            "WAREHOUSE_OPS": shipped.loc[order_id, "exception_code"] == "PICK_PACK_DELAY",
            "TRANSPORT": transit.loc[order_id, "exception_code"] == "CARRIER_DELAY",
            "CUSTOMER_DELIVERY": (
                delivered.loc[order_id, "exception_code"] == "CUSTOMER_APPOINTMENT"
            ),
        }
        matched = [
            cause for cause in CAUSE_PRIORITY if evidence[cause] and int(outcome.otif_miss) == 1
        ]
        primary = matched[0] if matched else ("UNKNOWN" if outcome.otif_miss else "ON_TIME")
        secondary = matched[1:]
        details = {
            "ORDER_CAPTURE": f"capture delayed {float(order['capture_delay_hours']):.0f}h",
            "VENDOR_FAILURE": (
                f"vendor ready delayed {float(vendor.loc[order_id, 'event_delay_hours']):.0f}h"
            ),
            "INVENTORY_SHORTAGE": (
                f"allocated {int(line['allocated_qty'])}/{int(line['requested_qty'])} units"
            ),
            "DC_CAPACITY": f"DC utilization {utilization:.0%}",
            "WAREHOUSE_OPS": (f"warehouse exception {shipped.loc[order_id, 'exception_code']}"),
            "TRANSPORT": f"transport exception {transit.loc[order_id, 'exception_code']}",
            "CUSTOMER_DELIVERY": (
                f"delivery exception {delivered.loc[order_id, 'exception_code']}"
            ),
            "UNKNOWN": "OTIF miss with no observable rule evidence",
            "ON_TIME": "order delivered on time and in full",
        }
        row: dict[str, object] = {
            "order_id": order_id,
            "primary_cause": primary,
            "detail": details[primary],
            "secondary_causes": json.dumps(secondary),
            "confidence": 0.95 if matched else (0.25 if primary == "UNKNOWN" else 1.0),
            "vendor_fault": int("VENDOR_FAILURE" in matched),
        }
        row.update({f"cause_{cause}": int(cause in matched) for cause in CAUSE_CATEGORIES})
        rows.append(row)
    return pd.DataFrame(rows)
