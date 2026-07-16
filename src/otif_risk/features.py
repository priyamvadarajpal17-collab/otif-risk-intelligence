"""Point-in-time-safe features and chronological model splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import CAUSE_CATEGORIES, PrototypeDataset

# These fields are known only after prediction (or directly encode an explanation/target).
LEAKAGE_BLOCKLIST = frozenset(
    {
        "delivered_timestamp",
        "delivered_qty",
        "on_time",
        "in_full",
        "outcome_timestamp",
        "primary_cause",
        "detail",
        "secondary_causes",
        "confidence",
        "vendor_fault",
        *(f"cause_{cause}" for cause in CAUSE_CATEGORIES),
    }
)

ORDER_CONTEXT_COLUMNS = (
    "order_date",
    "order_capture_timestamp",
    "requested_delivery_date",
    "promised_delivery_date",
    "capture_delay_hours",
)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def build_feature_table(
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
) -> pd.DataFrame:
    """Build one order row using signals available at each prediction timestamp."""
    outcome_required = {"order_id", "prediction_timestamp", "outcome_timestamp", "otif_miss"}
    if missing := outcome_required - set(outcomes.columns):
        raise ValueError(f"outcomes missing required columns: {sorted(missing)}")
    if "order_id" not in causes:
        raise ValueError("causes missing required column: order_id")
    order_ids = set(dataset.orders["order_id"])
    if set(outcomes["order_id"]) != order_ids or set(causes["order_id"]) != order_ids:
        raise ValueError("orders, outcomes, and causes must contain the same order_id values")

    base_columns = [
        "order_id",
        "vendor_id",
        "dc_id",
        "lane_id",
        "customer_id",
        "prediction_timestamp",
        "order_priority",
        "total_order_qty",
        *ORDER_CONTEXT_COLUMNS,
    ]
    features = dataset.orders[base_columns].merge(
        outcomes[["order_id", "outcome_timestamp", "otif_miss"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    features = (
        features.merge(_order_line_features(dataset.order_lines), on="order_id", how="left")
        .merge(_dimension_features(dataset), on="order_id", how="left")
        .merge(
            _capacity_features(dataset.orders, dataset.capacity_snapshots),
            on="order_id",
            how="left",
        )
        .merge(_event_features(dataset.orders, dataset.events), on="order_id", how="left")
    )
    features = _add_timing_features(features)
    features = _add_leading_signals(features)
    features = _add_signal_density(features)
    features = _add_entity_rolling_features(features, outcomes, causes, dataset.orders)

    features = features.drop(
        columns=[
            "outcome_timestamp",
            *ORDER_CONTEXT_COLUMNS,
            "order_capture_timestamp",
            "requested_delivery_date",
            "promised_delivery_date",
        ],
        errors="ignore",
    )
    leaked = LEAKAGE_BLOCKLIST.intersection(features.columns) - {"otif_miss"}
    if leaked:
        raise AssertionError(f"leakage-prone columns entered feature table: {sorted(leaked)}")
    if features["order_id"].duplicated().any():
        raise AssertionError("feature table must contain one row per order")
    return features.sort_values(["prediction_timestamp", "order_id"]).reset_index(drop=True)


def _order_line_features(order_lines: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        order_lines.groupby("order_id", as_index=False)
        .agg(
            line_count=("order_line_id", "count"),
            distinct_sku_count=("sku_id", "nunique"),
            total_requested_qty=("requested_qty", "sum"),
            total_allocated_qty=("allocated_qty", "sum"),
            stockout_line_count=("stockout_flag", "sum"),
        )
        .assign(
            allocation_ratio=lambda frame: frame["total_allocated_qty"]
            / frame["total_requested_qty"].clip(lower=1),
            has_stockout_flag=lambda frame: (frame["stockout_line_count"] > 0).astype(int),
        )
    )
    return grouped.drop(columns=["total_requested_qty", "total_allocated_qty"])


def _dimension_features(dataset: PrototypeDataset) -> pd.DataFrame:
    orders = dataset.orders[
        ["order_id", "vendor_id", "dc_id", "lane_id", "customer_id"]
    ].copy()
    vendors = dataset.vendors.rename(
        columns={
            "reliability_score": "vendor_reliability_score",
            "contract_lead_days": "vendor_contract_lead_days",
            "country": "vendor_country",
        }
    )
    dcs = dataset.dcs.rename(
        columns={
            "daily_capacity_units": "dc_daily_capacity_units",
            "region": "dc_region",
        }
    )
    lanes = dataset.lanes.rename(
        columns={
            "destination_region": "lane_destination_region",
            "carrier": "lane_carrier",
            "planned_transit_days": "lane_planned_transit_days",
        }
    )
    customers = dataset.customers.rename(
        columns={
            "region": "customer_region",
            "appointment_required": "customer_appointment_required",
        }
    )
    enriched = (
        orders.merge(vendors, on="vendor_id", how="left", validate="many_to_one")
        .merge(dcs, on="dc_id", how="left", validate="many_to_one")
        .merge(lanes, on="lane_id", how="left", validate="many_to_one")
        .merge(customers, on="customer_id", how="left", validate="many_to_one")
    )
    attribute_columns = [
        "order_id",
        "vendor_reliability_score",
        "vendor_contract_lead_days",
        "vendor_country",
        "dc_daily_capacity_units",
        "dc_region",
        "lane_destination_region",
        "lane_carrier",
        "lane_planned_transit_days",
        "customer_region",
        "customer_appointment_required",
    ]
    return enriched[attribute_columns]


def _capacity_features(
    orders: pd.DataFrame,
    capacity_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_lookup = capacity_snapshots.rename(columns={"snapshot_date": "capacity_snapshot_date"})
    joined = orders[["order_id", "dc_id", "prediction_timestamp"]].assign(
        capacity_snapshot_date=orders["prediction_timestamp"].dt.normalize()
    )
    joined = joined.merge(
        snapshot_lookup,
        on=["dc_id", "capacity_snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    return joined[
        [
            "order_id",
            "available_capacity_units",
            "planned_units",
            "utilization",
        ]
    ].rename(
        columns={
            "available_capacity_units": "dc_available_capacity_units",
            "planned_units": "dc_planned_units_at_prediction",
            "utilization": "dc_utilization_at_prediction",
        }
    ).assign(
        dc_capacity_headroom=lambda frame: (
            frame["dc_available_capacity_units"] - frame["dc_planned_units_at_prediction"]
        ),
        dc_over_capacity_flag=lambda frame: (
            frame["dc_utilization_at_prediction"] > 1.0
        ).astype(int),
    )


def _event_features(orders: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Summarize only events observed by ``prediction_timestamp`` (point-in-time safe).

    Alongside delay magnitudes this also exposes per-stage "observed" and
    "exception" flags. Those flags are the raw ingredients for the
    ``leading_signal_*`` columns built in :func:`_add_leading_signals`: a stage's
    exception can only contribute a signal once its event has actually posted
    before the prediction timestamp, so a cause that has not yet become
    observable correctly contributes no signal (rather than a leaked one).
    """
    rows: list[dict[str, float | int | str]] = []
    for row in orders.itertuples(index=False):
        observed = events.loc[
            (events["order_id"] == row.order_id)
            & (events["event_timestamp"] <= row.prediction_timestamp)
        ]
        vendor_ready = observed.loc[observed["event_type"] == "VENDOR_READY"]
        shipped = observed.loc[observed["event_type"] == "SHIPPED"]
        in_transit = observed.loc[observed["event_type"] == "IN_TRANSIT"]
        rows.append(
            {
                "order_id": row.order_id,
                "observed_event_count": int(len(observed)),
                "has_pre_prediction_exception": int(observed["exception_code"].notna().any()),
                "vendor_ready_observed": int(len(vendor_ready) > 0),
                "vendor_ready_delay_hours": _hours_delta(
                    vendor_ready["planned_timestamp"],
                    vendor_ready["event_timestamp"],
                ),
                "vendor_ready_exception_supplier_late": int(
                    (vendor_ready["exception_code"] == "SUPPLIER_LATE").any()
                ),
                "shipped_observed": int(len(shipped) > 0),
                "ship_delay_hours": _hours_delta(
                    shipped["planned_timestamp"],
                    shipped["event_timestamp"],
                ),
                "shipped_exception_pick_pack_delay": int(
                    (shipped["exception_code"] == "PICK_PACK_DELAY").any()
                ),
                "transit_observed": int(len(in_transit) > 0),
                "transit_delay_hours": _hours_delta(
                    in_transit["planned_timestamp"],
                    in_transit["event_timestamp"],
                ),
                "transit_exception_carrier_delay": int(
                    (in_transit["exception_code"] == "CARRIER_DELAY").any()
                ),
            }
        )
    return pd.DataFrame(rows)


def _hours_delta(planned: pd.Series, actual: pd.Series) -> float:
    if planned.empty or actual.empty:
        return 0.0
    delta = actual.iloc[0] - planned.iloc[0]
    return float(max(delta.total_seconds() / 3600.0, 0.0))


def _add_timing_features(features: pd.DataFrame) -> pd.DataFrame:
    enriched = features.copy()
    enriched["days_since_order"] = (
        enriched["prediction_timestamp"] - enriched["order_date"]
    ).dt.total_seconds() / 86_400.0
    enriched["days_to_promised_delivery"] = (
        enriched["promised_delivery_date"] - enriched["prediction_timestamp"]
    ).dt.total_seconds() / 86_400.0
    enriched["is_expedite"] = (enriched["order_priority"] == "EXPEDITE").astype(int)
    return enriched


def _add_leading_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Derive one point-in-time-observable leading signal per cause category.

    Each signal is a deterministic function of operational fields/events that are
    already filtered to ``event_timestamp <= prediction_timestamp`` (or, for
    ``ORDER_CAPTURE``/``INVENTORY_SHORTAGE``/``CUSTOMER_DELIVERY``, of fields known
    at order capture or from customer master data). None of these signals read the
    generator's latent disruption cause directly, so a cause only lights up a
    signal once real, observable evidence exists as of prediction time; a cause
    whose evidence has not yet posted (for example a vendor-ready event still
    pending) correctly contributes no signal rather than a leaked one.
    """
    enriched = features.copy()
    enriched["leading_signal_ORDER_CAPTURE"] = (
        enriched["capture_delay_hours"].astype(float) > 24
    ).astype(int)
    enriched["leading_signal_VENDOR_FAILURE"] = (
        (enriched["vendor_ready_delay_hours"].astype(float) > 24)
        | (enriched["vendor_ready_exception_supplier_late"] == 1)
    ).astype(int)
    enriched["leading_signal_INVENTORY_SHORTAGE"] = (
        (enriched["has_stockout_flag"] == 1) | (enriched["allocation_ratio"] < 1.0)
    ).astype(int)
    enriched["leading_signal_DC_CAPACITY"] = (
        enriched["dc_utilization_at_prediction"].astype(float) > 1.0
    ).astype(int)
    enriched["leading_signal_WAREHOUSE_OPS"] = (
        enriched["shipped_exception_pick_pack_delay"] == 1
    ).astype(int)
    enriched["leading_signal_TRANSPORT"] = (
        enriched["transit_exception_carrier_delay"] == 1
    ).astype(int)
    # The DELIVERED event (and any customer-appointment exception on it) only
    # posts after the promised delivery date, which is always after
    # prediction_timestamp — so it can never be used as a leading signal here.
    # The best point-in-time proxy is the customer master's appointment
    # requirement, which is known well in advance and is a genuine forward
    # looking risk factor for appointment-driven delivery delay.
    enriched["leading_signal_CUSTOMER_DELIVERY"] = (
        enriched["customer_appointment_required"].astype(bool)
    ).astype(int)
    return enriched


def _add_signal_density(features: pd.DataFrame) -> pd.DataFrame:
    signal_columns = [f"leading_signal_{cause}" for cause in CAUSE_CATEGORIES]
    enriched = features.copy()
    enriched["active_leading_signal_count"] = enriched[signal_columns].sum(axis=1).astype(int)
    enriched["has_any_leading_signal"] = (enriched["active_leading_signal_count"] > 0).astype(int)
    return enriched


def _add_entity_rolling_features(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Add matured, time-windowed rolling reliability rates per entity.

    Vendor fairness correction: a vendor's rolling rate is conditioned on
    ``vendor_fault`` (whether the *vendor* was among the matched root causes for
    that historical order) rather than the raw ``otif_miss`` outcome. Without
    this, a vendor would be penalized in its own rolling score for misses it did
    not cause (for example a DC capacity overload or a customer's own scheduling
    exception), preserving fair vendor attribution.
    DC/lane/customer rolling rates keep the raw OTIF-miss rate because this
    prototype does not generate a symmetric per-stage "fault" attribution for
    those dimensions.
    """
    historical = (
        outcomes[["order_id", "prediction_timestamp", "outcome_timestamp", "otif_miss"]]
        .merge(
            orders[["order_id", "vendor_id", "dc_id", "lane_id", "customer_id"]],
            on="order_id",
            validate="one_to_one",
        )
        .merge(causes[["order_id", "vendor_fault"]], on="order_id", validate="one_to_one")
        .copy()
    )
    enriched = features.copy()
    for entity in ("vendor", "dc", "lane", "customer"):
        key = f"{entity}_id"
        outcome_column = "vendor_fault" if entity == "vendor" else "otif_miss"
        rate_name = "rolling_fault_rate" if entity == "vendor" else "rolling_otif_miss_rate"
        counts: list[int] = []
        rates: list[float] = []
        for row in enriched.itertuples(index=False):
            prior = historical.loc[
                (historical[key] == getattr(row, key))
                & (historical["prediction_timestamp"] < row.prediction_timestamp)
                & (historical["outcome_timestamp"] < row.prediction_timestamp)
            ]
            counts.append(len(prior))
            rates.append(float(prior[outcome_column].mean()) if len(prior) else 0.0)
        enriched[f"{entity}_prior_matured_orders"] = counts
        enriched[f"{entity}_{rate_name}"] = rates
    return enriched


def temporal_split(feature_table: pd.DataFrame) -> TemporalSplit:
    """Chronologically split rows 60/20/20 without shuffling."""
    if "prediction_timestamp" not in feature_table or "order_id" not in feature_table:
        raise ValueError("feature_table requires prediction_timestamp and order_id")
    if len(feature_table) < 5:
        raise ValueError("feature_table requires at least five rows")
    ordered = feature_table.sort_values(["prediction_timestamp", "order_id"]).reset_index(drop=True)
    train_end = int(np.floor(len(ordered) * 0.60))
    validation_end = int(np.floor(len(ordered) * 0.80))
    return TemporalSplit(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        validation=ordered.iloc[train_end:validation_end].reset_index(drop=True),
        test=ordered.iloc[validation_end:].reset_index(drop=True),
    )
