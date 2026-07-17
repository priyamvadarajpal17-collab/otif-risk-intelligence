"""Point-in-time-safe features with an explicit ``as_of_timestamp`` contract.

Every feature in this module is a function of data knowable strictly at (or
before) a declared ``as_of_timestamp`` -- never the order's own future
outcome, and never the simulator's latent-truth columns. The same builder is
shared by two callers:

- historical/backtest snapshots, where each order is scored at its own
  baked-in ``prediction_timestamp`` (the default, ``as_of_timestamp=None``);
- ``operations.py``'s daily replay, which scores every still-open order as of
  one shared, explicit "today" (``as_of_timestamp=<simulated day>``).

Both paths flow through the same event/rolling-history filtering logic, so
there is exactly one point-in-time contract to audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import CAUSE_CATEGORIES, PrototypeDataset
from .line_evidence import build_line_evidence, order_line_aggregates

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

#: Rolling-history windows evaluated for every entity dimension, in addition
#: to "all matured history". ``None`` means unbounded (all matured history).
ROLLING_WINDOWS_DAYS: tuple[int | None, ...] = (30, 90, None)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def build_feature_table(
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    *,
    as_of_timestamp: pd.Timestamp | None = None,
    order_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """Build one row per order using signals available as of ``as_of_timestamp``.

    ``as_of_timestamp`` is the explicit feature contract: when omitted, each
    order is scored at its own ``prediction_timestamp`` (matches historical
    training/backtest behavior). When provided, every selected order is
    scored as of that single shared timestamp, which is how the daily
    operations replay scores still-open orders. Orders whose own
    ``order_date`` is after the effective as-of are rejected -- scoring an
    order before it exists is a caller bug, not a feature.
    """
    outcome_required = {"order_id", "prediction_timestamp", "outcome_timestamp", "otif_miss"}
    if missing := outcome_required - set(outcomes.columns):
        raise ValueError(f"outcomes missing required columns: {sorted(missing)}")
    if "order_id" not in causes:
        raise ValueError("causes missing required column: order_id")

    selected_ids = order_ids if order_ids is not None else dataset.orders["order_id"]
    selected_ids = pd.Index(selected_ids)
    if not selected_ids.isin(dataset.orders["order_id"]).all():
        raise ValueError("order_ids must be a subset of dataset.orders order_id")

    base_columns = [
        "order_id",
        "vendor_id",
        "dc_id",
        "lane_id",
        "customer_id",
        "order_priority",
        "total_order_qty",
        *ORDER_CONTEXT_COLUMNS,
    ]
    orders_slice = (
        dataset.orders.loc[dataset.orders["order_id"].isin(selected_ids), base_columns]
        .merge(
            outcomes[["order_id", "prediction_timestamp", "outcome_timestamp", "otif_miss"]],
            on="order_id",
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    if as_of_timestamp is not None:
        as_of_timestamp = pd.Timestamp(as_of_timestamp)
        if (orders_slice["order_date"] > as_of_timestamp).any():
            raise ValueError("as_of_timestamp precedes the order_date of a selected order")
        orders_slice["as_of_timestamp"] = as_of_timestamp
    else:
        orders_slice["as_of_timestamp"] = orders_slice["prediction_timestamp"]

    features = (
        orders_slice.merge(_order_line_features(dataset.order_lines), on="order_id", how="left")
        .merge(_dimension_features(dataset), on="order_id", how="left")
        .merge(
            _capacity_features(orders_slice, dataset.capacity_snapshots),
            on="order_id",
            how="left",
        )
        .merge(_event_features(orders_slice, dataset.events), on="order_id", how="left")
    )
    features = _add_timing_features(features)
    features = _add_leading_signals(features)
    features = _add_signal_density(features)
    features = _add_entity_rolling_features(features, outcomes, causes, dataset)

    features = features.drop(
        columns=[
            "outcome_timestamp",
            "prediction_timestamp",
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
    return features.sort_values(["as_of_timestamp", "order_id"]).reset_index(drop=True)


def attach_line_evidence_features(
    dataset: PrototypeDataset, features: pd.DataFrame
) -> pd.DataFrame:
    """Merge safe, order-level line-evidence aggregates onto the feature table.

    Kept as a separate pass (rather than inlined in ``build_feature_table``)
    because line evidence itself depends on this table's own
    ``leading_signal_*`` columns -- computing it as a second stage avoids a
    circular dependency while still keeping "one call per caller" simple.
    """
    line_evidence = build_line_evidence(dataset, features)
    aggregates = order_line_aggregates(line_evidence)
    merged = features.merge(aggregates, on="order_id", how="left", validate="one_to_one")
    fill_defaults = {
        "worst_line_shortage_ratio": 0.0,
        "affected_line_count": 0,
        "max_line_risk_evidence": 0.0,
        "critical_sku_share": 0.0,
        "line_qty_concentration": 1.0,
    }
    for column, default in fill_defaults.items():
        if column in merged:
            merged[column] = merged[column].fillna(default)
    return merged


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
    orders_slice: pd.DataFrame,
    capacity_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_lookup = capacity_snapshots.rename(columns={"snapshot_date": "capacity_snapshot_date"})
    joined = orders_slice[["order_id", "dc_id", "as_of_timestamp"]].assign(
        capacity_snapshot_date=orders_slice["as_of_timestamp"].dt.normalize()
    )
    joined = joined.merge(
        snapshot_lookup,
        on=["dc_id", "capacity_snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    trend_lookup = snapshot_lookup.rename(
        columns={
            "capacity_snapshot_date": "trend_snapshot_date",
            "utilization": "utilization_7d_ago",
        }
    )[["dc_id", "trend_snapshot_date", "utilization_7d_ago"]]
    joined = joined.assign(
        trend_snapshot_date=joined["capacity_snapshot_date"] - pd.Timedelta(days=7)
    )
    joined = joined.merge(trend_lookup, on=["dc_id", "trend_snapshot_date"], how="left")

    result = joined[
        [
            "order_id",
            "available_capacity_units",
            "planned_units",
            "utilization",
            "utilization_7d_ago",
        ]
    ].rename(
        columns={
            "available_capacity_units": "dc_available_capacity_units",
            "planned_units": "dc_planned_units_at_prediction",
            "utilization": "dc_utilization_at_prediction",
        }
    )
    result["dc_capacity_headroom"] = (
        result["dc_available_capacity_units"] - result["dc_planned_units_at_prediction"]
    )
    result["dc_over_capacity_flag"] = (result["dc_utilization_at_prediction"] > 0.90).astype(int)
    result["dc_utilization_trend_7d"] = (
        result["dc_utilization_at_prediction"] - result["utilization_7d_ago"].fillna(
            result["dc_utilization_at_prediction"]
        )
    ).round(4)
    return result.drop(columns=["utilization_7d_ago"])


def _event_features(orders_slice: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Summarize only events observed by ``as_of_timestamp`` (point-in-time safe).

    Alongside delay magnitudes this also exposes per-stage "observed" and
    "exception" flags, plus freshness/missingness diagnostics:
    ``hours_since_last_observed_event`` and ``missing_event_stage_count``.
    Those flags are the raw ingredients for the ``leading_signal_*`` columns
    built in :func:`_add_leading_signals`: a stage's exception can only
    contribute a signal once its event has actually posted before the as-of
    timestamp, so a cause that has not yet become observable correctly
    contributes no signal (rather than a leaked one).
    """
    rows: list[dict[str, float | int | str]] = []
    events_by_order = {
        order_id: group for order_id, group in events.groupby("order_id", sort=False)
    }
    for row in orders_slice.itertuples(index=False):
        order_events = events_by_order.get(row.order_id)
        if order_events is None:
            observed = events.iloc[0:0]
        else:
            observed = order_events.loc[order_events["event_timestamp"] <= row.as_of_timestamp]
        vendor_ready = observed.loc[observed["event_type"] == "VENDOR_READY"]
        shipped = observed.loc[observed["event_type"] == "SHIPPED"]
        in_transit = observed.loc[observed["event_type"] == "IN_TRANSIT"]
        stage_observed_count = int(
            (len(vendor_ready) > 0) + (len(shipped) > 0) + (len(in_transit) > 0)
        )
        last_event_timestamp = observed["event_timestamp"].max() if len(observed) else pd.NaT
        if pd.isna(last_event_timestamp):
            hours_since_last_event = (
                row.as_of_timestamp - row.order_date
            ).total_seconds() / 3600.0
        else:
            hours_since_last_event = (
                row.as_of_timestamp - last_event_timestamp
            ).total_seconds() / 3600.0
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
                "hours_since_last_observed_event": round(max(hours_since_last_event, 0.0), 2),
                "missing_event_stage_count": 3 - stage_observed_count,
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
        enriched["as_of_timestamp"] - enriched["order_date"]
    ).dt.total_seconds() / 86_400.0
    enriched["days_to_promised_delivery"] = (
        enriched["promised_delivery_date"] - enriched["as_of_timestamp"]
    ).dt.total_seconds() / 86_400.0
    enriched["remaining_slack_hours"] = (
        enriched["promised_delivery_date"] - enriched["as_of_timestamp"]
    ).dt.total_seconds() / 3600.0
    enriched["is_expedite"] = (enriched["order_priority"] == "EXPEDITE").astype(int)
    return enriched


def _add_leading_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Derive one point-in-time-observable leading signal per cause category.

    Each signal is a deterministic function of operational fields/events that
    are already filtered to ``event_timestamp <= as_of_timestamp`` (or, for
    ``ORDER_CAPTURE``/``INVENTORY_SHORTAGE``/``CUSTOMER_DELIVERY``, of fields
    known at order capture or from customer master data). None of these
    signals read the generator's latent disruption cause directly, so a
    cause only lights up a signal once real, observable evidence exists as of
    the as-of timestamp; a cause whose evidence has not yet posted (for
    example a vendor-ready event still pending) correctly contributes no
    signal rather than a leaked one.
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
        enriched["dc_utilization_at_prediction"].astype(float) > 0.90
    ).astype(int)
    enriched["leading_signal_WAREHOUSE_OPS"] = (
        enriched["shipped_exception_pick_pack_delay"] == 1
    ).astype(int)
    enriched["leading_signal_TRANSPORT"] = (
        enriched["transit_exception_carrier_delay"] == 1
    ).astype(int)
    # The DELIVERED event (and any customer-appointment exception on it) only
    # posts after the promised delivery date, which is always after
    # as_of_timestamp -- so it can never be used as a leading signal here.
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


def _rolling_window_feature(
    scoring: pd.DataFrame,
    history: pd.DataFrame,
    *,
    entity_col: str,
    rate_col: str,
    window_days: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized as-of rolling rate: matured prior history only, per entity.

    "Matured" means both ``prediction_timestamp`` and ``outcome_timestamp``
    precede the scoring row's ``as_of_timestamp``; ``window_days`` further
    restricts to history maturing within the trailing window. Uses a sorted
    cumulative-sum + ``searchsorted`` per entity group instead of an O(n^2)
    row-by-row scan, which matters once this runs 30+ times per replay day.
    """
    rates = np.zeros(len(scoring), dtype=float)
    counts = np.zeros(len(scoring), dtype=int)
    scoring_as_of = scoring["as_of_timestamp"].to_numpy()
    scoring_entities = scoring[entity_col].to_numpy()

    for entity_value in pd.unique(scoring_entities):
        hist = history.loc[history[entity_col] == entity_value].sort_values("outcome_timestamp")
        hist_times = hist["outcome_timestamp"].to_numpy()
        hist_values = hist[rate_col].to_numpy(dtype=float)
        cumulative_sum = np.concatenate(([0.0], np.cumsum(hist_values)))
        cumulative_count = np.arange(len(hist_values) + 1)

        row_mask = scoring_entities == entity_value
        as_of_values = scoring_as_of[row_mask]
        upper_idx = np.searchsorted(hist_times, as_of_values, side="left")
        if window_days is not None:
            lower_bound = as_of_values - np.timedelta64(window_days, "D")
            lower_idx = np.searchsorted(hist_times, lower_bound, side="left")
        else:
            lower_idx = np.zeros_like(upper_idx)

        window_sum = cumulative_sum[upper_idx] - cumulative_sum[lower_idx]
        window_count = cumulative_count[upper_idx] - cumulative_count[lower_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            window_rate = np.where(window_count > 0, window_sum / np.maximum(window_count, 1), 0.0)
        rates[row_mask] = window_rate
        counts[row_mask] = window_count
    return rates, counts


def _sku_history(dataset: PrototypeDataset, outcomes: pd.DataFrame) -> pd.DataFrame:
    timing = outcomes[["order_id", "prediction_timestamp", "outcome_timestamp"]]
    observed_lines = dataset.order_lines[
        ["order_id", "sku_id", "requested_qty", "shipped_qty"]
    ].copy()
    observed_lines["observed_short_shipment"] = (
        observed_lines["shipped_qty"] < observed_lines["requested_qty"]
    ).astype(int)
    return observed_lines[["order_id", "sku_id", "observed_short_shipment"]].merge(
        timing, on="order_id", how="left", validate="many_to_one"
    )


def _add_entity_rolling_features(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    dataset: PrototypeDataset,
) -> pd.DataFrame:
    """Add matured, 30/90-day and all-time rolling reliability rates per entity.

    Vendor fairness: a vendor's rolling rate is conditioned on
    ``vendor_fault`` (whether the *vendor* was among the matched root causes
    for that historical order) rather than the raw ``otif_miss`` outcome.
    Without this, a vendor would be penalized in its own rolling score for
    misses it did not cause (for example a DC capacity overload or a
    customer's own scheduling exception), preserving fair vendor attribution.
    DC/lane/customer rolling rates keep the raw OTIF-miss rate. SKU rolling
    rates use observed short shipments from matured order lines.
    """
    orders = dataset.orders
    entity_history = (
        outcomes[["order_id", "prediction_timestamp", "outcome_timestamp", "otif_miss"]]
        .merge(
            orders[["order_id", "vendor_id", "dc_id", "lane_id", "customer_id"]],
            on="order_id",
            validate="one_to_one",
        )
        .merge(causes[["order_id", "vendor_fault"]], on="order_id", validate="one_to_one")
        .copy()
    )
    sku_history = _sku_history(dataset, outcomes)

    enriched = features.copy()
    for entity in ("vendor", "dc", "lane", "customer"):
        key = f"{entity}_id"
        outcome_column = "vendor_fault" if entity == "vendor" else "otif_miss"
        rate_name = "rolling_fault_rate" if entity == "vendor" else "rolling_otif_miss_rate"
        for window in ROLLING_WINDOWS_DAYS:
            rates, counts = _rolling_window_feature(
                enriched,
                entity_history,
                entity_col=key,
                rate_col=outcome_column,
                window_days=window,
            )
            suffix = f"{window}d" if window is not None else "all_time"
            enriched[f"{entity}_{rate_name}_{suffix}"] = rates
            enriched[f"{entity}_prior_matured_orders_{suffix}"] = counts
        # Back-compat convenience aliases pointing at the all-time window.
        enriched[f"{entity}_prior_matured_orders"] = enriched[
            f"{entity}_prior_matured_orders_all_time"
        ]
        enriched[f"{entity}_{rate_name}"] = enriched[f"{entity}_{rate_name}_all_time"]

    lines_scoring = dataset.order_lines[["order_id", "sku_id"]].merge(
        enriched[["order_id", "as_of_timestamp"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    for window in ROLLING_WINDOWS_DAYS:
        rates, _counts = _rolling_window_feature(
            lines_scoring,
            sku_history,
            entity_col="sku_id",
            rate_col="observed_short_shipment",
            window_days=window,
        )
        suffix = f"{window}d" if window is not None else "all_time"
        lines_scoring[f"_sku_rate_{suffix}"] = rates
        order_level = lines_scoring.groupby("order_id")[f"_sku_rate_{suffix}"].agg(["max", "mean"])
        enriched[f"sku_rolling_shortfall_rate_max_{suffix}"] = (
            enriched["order_id"].map(order_level["max"]).fillna(0.0)
        )
        enriched[f"sku_rolling_shortfall_rate_mean_{suffix}"] = (
            enriched["order_id"].map(order_level["mean"]).fillna(0.0)
        )
    return enriched


def temporal_split(feature_table: pd.DataFrame) -> TemporalSplit:
    """Chronologically split rows 60/20/20 by *timestamp group*, not row count.

    Splitting strictly by row count can cut a group of orders sharing the
    exact same ``as_of_timestamp`` across a train/validation/test boundary.
    Instead, unique timestamps are ordered chronologically and assigned to a
    split as whole groups, so identical timestamps never straddle a
    boundary.
    """
    time_column = (
        "as_of_timestamp" if "as_of_timestamp" in feature_table else "prediction_timestamp"
    )
    if time_column not in feature_table or "order_id" not in feature_table:
        raise ValueError("feature_table requires as_of_timestamp and order_id")
    if len(feature_table) < 5:
        raise ValueError("feature_table requires at least five rows")
    ordered = feature_table.sort_values([time_column, "order_id"]).reset_index(drop=True)

    unique_times = ordered[time_column].drop_duplicates().sort_values().to_numpy()
    counts_per_time = ordered.groupby(time_column).size().reindex(unique_times).to_numpy()
    cumulative = np.cumsum(counts_per_time)
    total = len(ordered)
    train_end_idx = int(np.searchsorted(cumulative, total * 0.60, side="left"))
    validation_end_idx = int(np.searchsorted(cumulative, total * 0.80, side="left"))
    train_end_idx = min(max(train_end_idx, 0), len(unique_times) - 1)
    validation_end_idx = min(max(validation_end_idx, train_end_idx), len(unique_times) - 1)

    train_boundary = unique_times[train_end_idx]
    validation_boundary = unique_times[validation_end_idx]
    train_mask = ordered[time_column] <= train_boundary
    validation_mask = (ordered[time_column] > train_boundary) & (
        ordered[time_column] <= validation_boundary
    )
    test_mask = ordered[time_column] > validation_boundary
    return TemporalSplit(
        train=ordered.loc[train_mask].reset_index(drop=True),
        validation=ordered.loc[validation_mask].reset_index(drop=True),
        test=ordered.loc[test_mask].reset_index(drop=True),
    )
