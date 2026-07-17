"""Transparent OTIF outcome and root-cause rules."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .contracts import CAUSE_CATEGORIES, PrototypeDataset
from .validation import validate_dataset

# Rules are evaluated independently, then this upstream-to-downstream order resolves primary cause.
CAUSE_PRIORITY = CAUSE_CATEGORIES


def compute_service_outcome(
    delivered_timestamp: pd.Series,
    delivered_qty: pd.Series,
    requested_qty: pd.Series,
    promised_delivery_date: pd.Series,
) -> dict[str, pd.Series]:
    """Apply the twin's one OTIF definition to any (delivered, requested) pair.

    This is the single source of truth for ``on_time``/``in_full``/``otif_miss``:
    ``calculate_outcomes`` uses it for the twin's original simulated lifecycle,
    and ``action_response.py`` calls it again on a *potential* (evaluation-only)
    delivered timestamp/quantity after a candidate intervention, so both use
    identical service-outcome logic -- never a re-derived approximation.
    """
    on_time = (delivered_timestamp <= promised_delivery_date).astype(int)
    in_full = (delivered_qty >= requested_qty).astype(int)
    otif_miss = ((on_time == 0) | (in_full == 0)).astype(int)
    return {"on_time": on_time, "in_full": in_full, "otif_miss": otif_miss}


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
    service_outcome = compute_service_outcome(
        outcomes["delivered_timestamp"],
        outcomes["delivered_qty"],
        outcomes["requested_qty"],
        outcomes["promised_delivery_date"],
    )
    outcomes["on_time"] = service_outcome["on_time"]
    outcomes["in_full"] = service_outcome["in_full"]
    outcomes["otif_miss"] = service_outcome["otif_miss"]
    outcomes["outcome_timestamp"] = outcomes["delivered_timestamp"]
    return outcomes


def _event_evidence(
    dataset: PrototypeDataset, event_type: str, order_ids: pd.Index
) -> pd.DataFrame:
    """Return per-order event evidence, reindexed so *missing* events are NaN/None.

    Some events are never logged at all (partial observability), so this must
    not raise on absence -- it must report the absence as "no evidence"
    (``event_delay_hours`` NaN, ``exception_code`` None), which the caller
    treats as the rule simply not matching.
    """
    frame = dataset.events.loc[dataset.events["event_type"] == event_type].copy()
    frame["event_delay_hours"] = (
        frame["event_timestamp"] - frame["planned_timestamp"]
    ).dt.total_seconds() / 3600
    indexed = frame.drop_duplicates("order_id").set_index("order_id")
    return indexed.reindex(order_ids)[["event_delay_hours", "exception_code"]]


def derive_root_causes(
    dataset: PrototypeDataset, outcomes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Evaluate every cause rule and assign an upstream-priority primary cause."""
    if outcomes is None:
        outcomes = calculate_outcomes(dataset)
    required = {"order_id", "otif_miss"}
    if missing := required - set(outcomes.columns):
        raise ValueError(f"outcomes missing required columns: {sorted(missing)}")

    order_ids = pd.Index(outcomes["order_id"])
    orders = dataset.orders.set_index("order_id").reindex(order_ids)
    lines = (
        dataset.order_lines.groupby("order_id")
        .agg(
            requested_qty=("requested_qty", "sum"),
            allocated_qty=("allocated_qty", "sum"),
            shipped_qty=("shipped_qty", "sum"),
            stockout_flag=("stockout_flag", "max"),
        )
        .reindex(order_ids)
    )
    vendor = _event_evidence(dataset, "VENDOR_READY", order_ids)
    shipped = _event_evidence(dataset, "SHIPPED", order_ids)
    transit = _event_evidence(dataset, "IN_TRANSIT", order_ids)
    delivered = _event_evidence(dataset, "DELIVERED", order_ids)

    snapshot_keys = list(
        zip(orders["dc_id"].to_numpy(), orders["order_date"].dt.normalize().to_numpy(), strict=True)
    )
    capacity = dataset.capacity_snapshots.set_index(["dc_id", "snapshot_date"])["utilization"]
    utilization = np.array([float(capacity.get(key, 0.5)) for key in snapshot_keys])

    otif_miss = outcomes.set_index("order_id")["otif_miss"].reindex(order_ids).to_numpy()

    evidence = {
        "ORDER_CAPTURE": (orders["capture_delay_hours"].astype(float) > 24).to_numpy(),
        "VENDOR_FAILURE": (
            (vendor["event_delay_hours"].astype(float) > 24)
            | (vendor["exception_code"] == "SUPPLIER_LATE")
        ).to_numpy(),
        "INVENTORY_SHORTAGE": (
            lines["shipped_qty"].astype(float) < lines["requested_qty"].astype(float)
        ).to_numpy(),
        "DC_CAPACITY": utilization > 0.90,
        "WAREHOUSE_OPS": (shipped["exception_code"] == "PICK_PACK_DELAY").to_numpy(),
        "TRANSPORT": (transit["exception_code"] == "CARRIER_DELAY").to_numpy(),
        "CUSTOMER_DELIVERY": (delivered["exception_code"] == "CUSTOMER_APPOINTMENT").to_numpy(),
    }

    rows: list[dict[str, object]] = []
    for row_index, order_id in enumerate(order_ids):
        matched = [
            cause
            for cause in CAUSE_PRIORITY
            if evidence[cause][row_index] and int(otif_miss[row_index]) == 1
        ]
        primary = matched[0] if matched else ("UNKNOWN" if otif_miss[row_index] else "ON_TIME")
        secondary = matched[1:]
        vendor_delay = vendor["event_delay_hours"].iloc[row_index]
        capture_delay_hours = float(orders["capture_delay_hours"].iloc[row_index])
        details = {
            "ORDER_CAPTURE": f"capture delayed {capture_delay_hours:.0f}h",
            "VENDOR_FAILURE": (
                f"vendor ready delayed {float(vendor_delay):.0f}h"
                if pd.notna(vendor_delay)
                else "vendor ready event not observed"
            ),
            "INVENTORY_SHORTAGE": (
                f"shipped {int(lines['shipped_qty'].iloc[row_index])}/"
                f"{int(lines['requested_qty'].iloc[row_index])} units"
            ),
            "DC_CAPACITY": f"DC utilization {utilization[row_index]:.0%}",
            "WAREHOUSE_OPS": f"warehouse exception {shipped['exception_code'].iloc[row_index]}",
            "TRANSPORT": f"transport exception {transit['exception_code'].iloc[row_index]}",
            "CUSTOMER_DELIVERY": (
                f"delivery exception {delivered['exception_code'].iloc[row_index]}"
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
        row.update(
            {f"stage_{cause}": int(evidence[cause][row_index]) for cause in CAUSE_CATEGORIES}
        )
        row.update({f"cause_{cause}": int(cause in matched) for cause in CAUSE_CATEGORIES})
        rows.append(row)
    return pd.DataFrame(rows)
