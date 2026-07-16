"""Deterministic, normalized synthetic data for the standalone OTIF prototype."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import CAUSE_CATEGORIES, PrototypeConfig, PrototypeDataset
from .validation import validate_dataset


def _ids(prefix: str, size: int) -> list[str]:
    return [f"{prefix}{index:03d}" for index in range(1, size + 1)]


def generate_dataset(config: PrototypeConfig) -> PrototypeDataset:
    """Generate an understandable synthetic dataset with roughly 20% OTIF misses."""
    rng = np.random.default_rng(config.seed)
    n = config.n_orders
    vendor_ids, dc_ids = _ids("V", 12), _ids("DC", 5)
    lane_ids, customer_ids = _ids("L", 15), _ids("C", 40)

    vendors = pd.DataFrame(
        {
            "vendor_id": vendor_ids,
            "vendor_name": [f"Vendor {i}" for i in range(1, 13)],
            "country": rng.choice(["US", "MX", "CA"], 12, p=[0.55, 0.3, 0.15]),
            "contract_lead_days": rng.integers(1, 5, 12),
            "reliability_score": rng.uniform(0.82, 0.98, 12).round(3),
        }
    )
    dcs = pd.DataFrame(
        {
            "dc_id": dc_ids,
            "dc_name": [f"Distribution Center {i}" for i in range(1, 6)],
            "region": ["WEST", "CENTRAL", "SOUTH", "EAST", "NORTH"],
            "daily_capacity_units": rng.integers(850, 1250, 5),
        }
    )
    lanes = pd.DataFrame(
        {
            "lane_id": lane_ids,
            "origin_dc_id": rng.choice(dc_ids, 15),
            "destination_region": rng.choice(["WEST", "CENTRAL", "SOUTH", "EAST"], 15),
            "carrier": rng.choice(["NorthStar", "RoadRunner", "BlueLine"], 15),
            "planned_transit_days": rng.integers(1, 5, 15),
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_name": [f"Customer {i}" for i in range(1, 41)],
            "region": rng.choice(["WEST", "CENTRAL", "SOUTH", "EAST"], 40),
            "appointment_required": rng.random(40) < 0.42,
        }
    )

    order_dates = pd.Timestamp(config.start_date) + pd.to_timedelta(
        rng.integers(0, max(90, n // 10), n), unit="D"
    )
    order_ids = [f"O{i:06d}" for i in range(1, n + 1)]
    selected_vendors = rng.choice(vendor_ids, n)
    selected_dcs = rng.choice(dc_ids, n)
    selected_lanes = rng.choice(lane_ids, n)
    selected_customers = rng.choice(customer_ids, n)
    requested = order_dates + pd.to_timedelta(rng.integers(5, 11, n), unit="D")
    promised = requested.copy()
    prediction = order_dates + pd.to_timedelta(config.prediction_horizon_days, unit="D")
    prediction = pd.Series(
        np.minimum(prediction.to_numpy(), (promised - pd.Timedelta(hours=1)).to_numpy())
    )

    # Exactly 20% of orders are disrupted. Most have an observable cause; a small
    # subset intentionally has no matching evidence to exercise UNKNOWN handling.
    disrupted = rng.choice(n, size=round(n * 0.20), replace=False)
    latent_primary = np.full(n, "", dtype=object)
    observable = disrupted[: round(len(disrupted) * 0.95)]
    latent_primary[observable] = rng.choice(CAUSE_CATEGORIES, len(observable))
    latent_primary[disrupted[len(observable) :]] = "UNKNOWN"
    latent_secondary = np.full(n, "", dtype=object)
    multi = rng.choice(observable, size=max(1, round(len(observable) * 0.18)), replace=False)
    for index in multi:
        alternatives = [cause for cause in CAUSE_CATEGORIES if cause != latent_primary[index]]
        latent_secondary[index] = rng.choice(alternatives)

    def has(cause: str) -> np.ndarray:
        return (latent_primary == cause) | (latent_secondary == cause)

    line_counts = rng.integers(1, 4, n)
    line_rows: list[dict[str, object]] = []
    total_qty = np.zeros(n, dtype=int)
    for index, line_count in enumerate(line_counts):
        quantities = rng.integers(5, 60, line_count)
        total_qty[index] = int(quantities.sum())
        shortage = has("INVENTORY_SHORTAGE")[index]
        for line_number, quantity in enumerate(quantities, start=1):
            short_qty = int(max(1, round(quantity * 0.25))) if shortage and line_number == 1 else 0
            line_rows.append(
                {
                    "order_line_id": f"{order_ids[index]}-{line_number}",
                    "order_id": order_ids[index],
                    "line_number": line_number,
                    "sku_id": f"SKU{rng.integers(1, 121):04d}",
                    "requested_qty": int(quantity),
                    "allocated_qty": int(quantity - short_qty),
                    "shipped_qty": int(quantity - short_qty),
                    "inventory_available_at_order": int(quantity - short_qty),
                    "stockout_flag": bool(short_qty),
                }
            )
    order_lines = pd.DataFrame(line_rows)

    capture_delay = np.where(has("ORDER_CAPTURE"), rng.integers(25, 49, n), rng.integers(0, 9, n))
    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "order_date": order_dates,
            "order_capture_timestamp": order_dates + pd.to_timedelta(capture_delay, unit="h"),
            "prediction_timestamp": prediction,
            "requested_delivery_date": requested,
            "promised_delivery_date": promised,
            "vendor_id": selected_vendors,
            "dc_id": selected_dcs,
            "lane_id": selected_lanes,
            "customer_id": selected_customers,
            "order_priority": rng.choice(["STANDARD", "EXPEDITE"], n, p=[0.88, 0.12]),
            "total_order_qty": total_qty,
            "capture_delay_hours": capture_delay,
        }
    )
    # NOTE: `leading_signal_*` columns are intentionally NOT generated here. Earlier
    # prototype code produced them as a noisy function of the *latent* disruption
    # cause (`has(cause)`), which any order could see regardless of whether that
    # cause had actually become observable by `prediction_timestamp`. That directly
    # leaked the label-adjacent generator state into model features (see the
    # near-perfect historical AUCs this caused). Leading signals are now derived
    # in `features.py`, strictly from operational fields/events already filtered to
    # `event_timestamp <= prediction_timestamp`, so a cause can only contribute a
    # signal once it is genuinely knowable.

    planned_vendor = order_dates + pd.to_timedelta(2, unit="D")
    vendor_ready = planned_vendor + pd.to_timedelta(np.where(has("VENDOR_FAILURE"), 2, 0), unit="D")
    planned_ship = order_dates + pd.to_timedelta(3, unit="D")
    actual_ship = planned_ship + pd.to_timedelta(
        np.where(has("WAREHOUSE_OPS"), 2, 0)
        + np.where(has("VENDOR_FAILURE") | has("DC_CAPACITY"), 1, 0),
        unit="D",
    )
    transit_delay = np.where(has("TRANSPORT"), 2, 0)
    customer_delay = np.where(has("CUSTOMER_DELIVERY"), 2, 0)
    delivered = promised + pd.to_timedelta(
        np.where(np.isin(np.arange(n), disrupted), 2, -1), unit="D"
    )
    event_rows: list[dict[str, object]] = []
    for index, order_id in enumerate(order_ids):
        event_rows.extend(
            [
                {
                    "order_id": order_id,
                    "event_type": "VENDOR_READY",
                    "planned_timestamp": planned_vendor[index],
                    "event_timestamp": vendor_ready[index],
                    "exception_code": ("SUPPLIER_LATE" if has("VENDOR_FAILURE")[index] else None),
                },
                {
                    "order_id": order_id,
                    "event_type": "SHIPPED",
                    "planned_timestamp": planned_ship[index],
                    "event_timestamp": actual_ship[index],
                    "exception_code": ("PICK_PACK_DELAY" if has("WAREHOUSE_OPS")[index] else None),
                },
                {
                    "order_id": order_id,
                    "event_type": "IN_TRANSIT",
                    "planned_timestamp": actual_ship[index] + pd.Timedelta(days=1),
                    "event_timestamp": actual_ship[index]
                    + pd.Timedelta(days=1 + transit_delay[index]),
                    "exception_code": "CARRIER_DELAY" if transit_delay[index] else None,
                },
                {
                    "order_id": order_id,
                    "event_type": "DELIVERED",
                    "planned_timestamp": promised[index],
                    "event_timestamp": delivered[index],
                    "exception_code": (
                        "CUSTOMER_APPOINTMENT"
                        if customer_delay[index]
                        else ("UNEXPLAINED" if latent_primary[index] == "UNKNOWN" else None)
                    ),
                },
            ]
        )
    events = pd.DataFrame(event_rows).sort_values(["order_id", "event_timestamp"])

    all_days = pd.date_range(order_dates.min().normalize(), delivered.max().normalize(), freq="D")
    capacity_rows: list[dict[str, object]] = []
    capacity_orders = orders.assign(snapshot_date=orders["order_date"].dt.normalize())
    for dc_id in dc_ids:
        base = int(dcs.loc[dcs["dc_id"] == dc_id, "daily_capacity_units"].iloc[0])
        daily_load = (
            capacity_orders.loc[capacity_orders["dc_id"] == dc_id]
            .groupby("snapshot_date")["total_order_qty"]
            .sum()
        )
        affected_dates = set(
            capacity_orders.loc[
                (capacity_orders["dc_id"] == dc_id) & has("DC_CAPACITY"), "snapshot_date"
            ]
        )
        for day in all_days:
            planned_units = int(daily_load.get(day, rng.integers(base // 3, 3 * base // 4)))
            if day in affected_dates:
                planned_units = max(planned_units, int(base * 1.15))
            capacity_rows.append(
                {
                    "dc_id": dc_id,
                    "snapshot_date": day,
                    "available_capacity_units": base,
                    "planned_units": planned_units,
                    "utilization": planned_units / base,
                }
            )
    capacity_snapshots = pd.DataFrame(capacity_rows)

    dataset = PrototypeDataset(
        orders=orders,
        order_lines=order_lines,
        events=events,
        vendors=vendors,
        dcs=dcs,
        lanes=lanes,
        customers=customers,
        capacity_snapshots=capacity_snapshots,
    )
    validate_dataset(dataset)
    return dataset
