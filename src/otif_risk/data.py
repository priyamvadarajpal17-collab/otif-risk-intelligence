"""Noisy, partially-observable synthetic supply-chain digital twin.

Unlike a generator that first selects which orders will miss OTIF and then
forces matching events, this module simulates each order's lifecycle stage by
stage (capture, vendor readiness, line allocation, warehouse, transport,
customer delivery) using persistent entity/SKU traits, seasonal demand,
correlated disruption shocks, and independent operational noise at every
stage. The OTIF outcome then falls out of the *accumulated* delay and
quantity shortfall actually simulated -- it is never chosen up front.

Ground truth needed only for evaluation (which shock drove which line/order,
the latent primary cause, accumulated delay/shortfall, and whether an
intervention would plausibly have helped) is kept in separate
``simulator_truth`` / ``line_truth`` / ``shocks`` tables. These are never
merged into the model-facing feature table (see ``features.py``).

A small, fixed slice of orders (``config.scenario_order_count``, default 5) is
deterministically scripted into named demonstration scenarios so the demo is
guaranteed to contain: multi-cause propagation, two orders contesting the same
recovery capacity, a line-level (not all-lines) stockout, and an unexplained
("UNKNOWN") miss. Concept drift is not a single scripted order: it is a
date-conditioned regime shift (elevated disruption density) in the final
slice of the simulated calendar, so ``operations.py``'s daily replay can
detect it without any single order being hand-picked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import PrototypeConfig, PrototypeDataset
from .validation import validate_dataset

N_VENDORS = 12
N_DCS = 5
N_LANES = 15
N_CUSTOMERS = 40
N_SKUS = 120

#: Fraction of the calendar (from the end) treated as the "drift window":
#: disruption density is deliberately elevated here so a replay that scores
#: through this window sees a genuine, explainable performance/rate shift.
DRIFT_WINDOW_FRACTION = 0.22
DRIFT_VENDOR_COUNT = 2

SCENARIO_TAGS = (
    "multi_cause_propagation",
    "resource_contention_a",
    "resource_contention_b",
    "line_level_stockout",
    "uncertain_unknown_cause",
)


def _ids(prefix: str, size: int) -> list[str]:
    return [f"{prefix}{index:03d}" for index in range(1, size + 1)]


def _seasonal_multiplier(day_of_year: np.ndarray) -> np.ndarray:
    """A smooth demand/congestion seasonality curve, peaking near day ~300 (Q4)."""
    return 1.0 + 0.28 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0)


@dataclass
class _Entities:
    vendors: pd.DataFrame
    dcs: pd.DataFrame
    lanes: pd.DataFrame
    customers: pd.DataFrame
    skus: pd.DataFrame


def _build_entities(rng: np.random.Generator) -> _Entities:
    vendor_ids, dc_ids = _ids("V", N_VENDORS), _ids("DC", N_DCS)
    lane_ids, customer_ids, sku_ids = (
        _ids("L", N_LANES),
        _ids("C", N_CUSTOMERS),
        _ids("SKU", N_SKUS),
    )
    reliability = rng.uniform(0.80, 0.985, N_VENDORS).round(3)
    vendors = pd.DataFrame(
        {
            "vendor_id": vendor_ids,
            "vendor_name": [f"Vendor {i}" for i in range(1, N_VENDORS + 1)],
            "country": rng.choice(["US", "MX", "CA"], N_VENDORS, p=[0.55, 0.3, 0.15]),
            "contract_lead_days": rng.integers(1, 5, N_VENDORS),
            "reliability_score": reliability,
            # Stable heterogeneity: less reliable vendors have wider (noisier)
            # lead-time variance, not just a lower mean.
            "delay_scale_hours": ((1 - reliability) * 22 + 3).round(2),
        }
    )
    dcs = pd.DataFrame(
        {
            "dc_id": dc_ids,
            "dc_name": [f"Distribution Center {i}" for i in range(1, N_DCS + 1)],
            "region": ["WEST", "CENTRAL", "SOUTH", "EAST", "NORTH"],
            "daily_capacity_units": rng.integers(850, 1250, N_DCS),
            "capacity_variability": rng.uniform(0.08, 0.22, N_DCS).round(3),
        }
    )
    dcs.loc[dcs["dc_id"] == dc_ids[0], "daily_capacity_units"] = 1_200
    lanes = pd.DataFrame(
        {
            "lane_id": lane_ids,
            "origin_dc_id": rng.choice(dc_ids, N_LANES),
            "destination_region": rng.choice(["WEST", "CENTRAL", "SOUTH", "EAST"], N_LANES),
            "carrier": rng.choice(["NorthStar", "RoadRunner", "BlueLine"], N_LANES),
            "planned_transit_days": rng.integers(1, 5, N_LANES),
            "transit_variability_days": rng.uniform(0.3, 1.4, N_LANES).round(2),
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_name": [f"Customer {i}" for i in range(1, N_CUSTOMERS + 1)],
            "region": rng.choice(["WEST", "CENTRAL", "SOUTH", "EAST"], N_CUSTOMERS),
            "appointment_required": rng.random(N_CUSTOMERS) < 0.42,
            "reschedule_trait": rng.beta(1.2, 6.0, N_CUSTOMERS).round(3),
        }
    )
    criticality = rng.choice(
        ["CRITICAL", "STANDARD", "LOW"], N_SKUS, p=[0.15, 0.65, 0.20]
    )
    skus = pd.DataFrame(
        {
            "sku_id": sku_ids,
            "criticality_tier": criticality,
            "base_unit_value": rng.lognormal(4.0, 0.6, N_SKUS).round(2),
            # Stable per-SKU baseline shortfall propensity, independent of shocks.
            "scarcity_trait": rng.beta(1.3, 28.0, N_SKUS).round(4),
            "demand_volatility": rng.uniform(0.05, 0.35, N_SKUS).round(3),
        }
    )
    return _Entities(vendors, dcs, lanes, customers, skus)


def _build_shocks(
    rng: np.random.Generator,
    entities: _Entities,
    horizon_days: int,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Generate correlated disruption shocks for vendors, DCs, and lanes.

    Each shock is a contiguous date window with a severity multiplier. A
    concept-drift epoch near the end of the horizon deliberately elevates
    vendor-shock density (``DRIFT_VENDOR_COUNT`` additional forced shocks) so
    a daily replay that scores through that window observes a genuine,
    explainable regime shift rather than a hand-picked single order.
    """
    rows: list[dict[str, object]] = []
    shock_id = 0

    def _add(
        entity_type: str, entity_id: str, start_day: int, duration: int, severity: float
    ) -> None:
        nonlocal shock_id
        shock_id += 1
        start = start_date + pd.Timedelta(days=int(start_day))
        rows.append(
            {
                "shock_id": f"SHK{shock_id:04d}",
                "shock_type": f"{entity_type.upper()}_DISRUPTION",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "start_date": start,
                "end_date": start + pd.Timedelta(days=int(duration)),
                "severity": round(float(severity), 3),
            }
        )

    for vendor_id in entities.vendors["vendor_id"]:
        if rng.random() < 0.45:
            start_day = int(rng.integers(0, max(1, horizon_days - 25)))
            _add("vendor", vendor_id, start_day, int(rng.integers(10, 25)), rng.uniform(1.8, 3.2))
    for dc_id in entities.dcs["dc_id"]:
        if rng.random() < 0.5:
            start_day = int(rng.integers(0, max(1, horizon_days - 20)))
            _add("dc", dc_id, start_day, int(rng.integers(9, 22)), rng.uniform(1.8, 3.0))
    for lane_id in entities.lanes["lane_id"]:
        if rng.random() < 0.4:
            start_day = int(rng.integers(0, max(1, horizon_days - 15)))
            _add("lane", lane_id, start_day, int(rng.integers(6, 16)), rng.uniform(1.7, 2.8))

    # Concept-drift epoch: force extra vendor disruption density late in the
    # horizon regardless of the random draws above.
    drift_start_day = int(horizon_days * (1 - DRIFT_WINDOW_FRACTION))
    drift_vendors = entities.vendors["vendor_id"].iloc[:DRIFT_VENDOR_COUNT]
    for vendor_id in drift_vendors:
        _add(
            "vendor",
            vendor_id,
            drift_start_day + int(rng.integers(0, 5)),
            int(rng.integers(12, 20)),
            rng.uniform(2.4, 3.4),
        )
    return pd.DataFrame(rows)


def _shock_active(
    shocks: pd.DataFrame,
    entity_type: str,
    entity_ids: np.ndarray,
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (active mask, severity) for each order against one entity type's shocks."""
    active = np.zeros(len(entity_ids), dtype=bool)
    severity = np.ones(len(entity_ids), dtype=float)
    subset = shocks.loc[shocks["entity_type"] == entity_type]
    date_values = dates.to_numpy()
    for shock in subset.itertuples(index=False):
        mask = (
            (entity_ids == shock.entity_id)
            & (date_values >= np.datetime64(shock.start_date))
            & (date_values <= np.datetime64(shock.end_date))
        )
        active |= mask
        severity = np.where(mask, np.maximum(severity, shock.severity), severity)
    return active, severity


def _force_line_shortfall_for_scenarios(
    order_lines: pd.DataFrame,
    shortfall_order_ids: list[str],
    clean_order_ids: list[str],
) -> pd.DataFrame:
    """Deterministically guarantee scenario line outcomes (not left to chance).

    ``shortfall_order_ids`` get their first line's ``shipped_qty`` cut to ~65%
    of what was requested (a real, evidenced shortfall); ``clean_order_ids``
    get every line fully shipped, so the "uncertain/unknown cause" scenario
    genuinely has zero inventory evidence.
    """
    result = order_lines.copy()
    for order_id in shortfall_order_ids:
        mask = (result["order_id"] == order_id) & (result["line_number"] == 1)
        requested = result.loc[mask, "requested_qty"]
        shipped = (requested * 0.65).round().astype(int).clip(lower=1)
        result.loc[mask, "shipped_qty"] = shipped
        result.loc[mask, "allocated_qty"] = np.maximum(
            result.loc[mask, "allocated_qty"], shipped
        )
        result.loc[mask, "stockout_flag"] = True
        result.loc[mask, "shortfall_ratio"] = round(1 - 0.65, 4)
    for order_id in clean_order_ids:
        mask = result["order_id"] == order_id
        result.loc[mask, "shipped_qty"] = result.loc[mask, "requested_qty"]
        result.loc[mask, "allocated_qty"] = result.loc[mask, "requested_qty"]
        result.loc[mask, "stockout_flag"] = False
        result.loc[mask, "shortfall_ratio"] = 0.0
    return result


def _sync_line_truth_with_forced_lines(
    line_truth: pd.DataFrame,
    order_lines: pd.DataFrame,
    shortfall_order_ids: list[str],
    clean_order_ids: list[str],
) -> pd.DataFrame:
    """Keep evaluation-only ``line_truth`` consistent with forced scenario lines."""
    result = line_truth.merge(
        order_lines[["order_line_id", "shortfall_ratio"]],
        on="order_line_id",
        how="left",
        suffixes=("", "_forced"),
    )
    forced_mask = result["order_id"].isin(shortfall_order_ids) | result["order_id"].isin(
        clean_order_ids
    )
    result.loc[forced_mask, "shortfall_ratio"] = result.loc[forced_mask, "shortfall_ratio_forced"]
    result.loc[forced_mask, "truly_affected"] = result.loc[forced_mask, "shortfall_ratio"] > 0.05
    result.loc[forced_mask, "vendor_stage_shortfall"] = (
        result.loc[forced_mask, "shortfall_ratio"] > 0.05
    )
    return result.drop(columns=["shortfall_ratio_forced"])


def _force_dc_utilization_for_scenarios(
    capacity_snapshots: pd.DataFrame,
    keys: list[tuple[str, pd.Timestamp]],
    *,
    low_keys: list[tuple[str, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Guarantee DC_CAPACITY evidence for the scripted contention/multi-cause orders.

    ``low_keys`` does the opposite: force utilization comfortably *below* the
    DC_CAPACITY threshold for a scenario (e.g. the uncertain/unknown-cause
    order), so ambient organic noise from other orders sharing that DC/day
    cannot accidentally hand it an explainable cause.
    """
    result = capacity_snapshots.copy()
    for dc_id, snapshot_date in keys:
        mask = (result["dc_id"] == dc_id) & (result["snapshot_date"] == snapshot_date)
        base = result.loc[mask, "available_capacity_units"]
        forced_units = (base * 1.05).round().astype(int)
        result.loc[mask, "planned_units"] = forced_units
        result.loc[mask, "utilization"] = 1.05
    for dc_id, snapshot_date in low_keys or []:
        mask = (result["dc_id"] == dc_id) & (result["snapshot_date"] == snapshot_date)
        base = result.loc[mask, "available_capacity_units"]
        forced_units = (base * 0.35).round().astype(int)
        result.loc[mask, "planned_units"] = forced_units
        result.loc[mask, "utilization"] = 0.35
    return result


def generate_dataset(config: PrototypeConfig) -> PrototypeDataset:
    """Simulate a noisy, partially-observable order lifecycle end to end."""
    rng = np.random.default_rng(config.seed)
    n = config.n_orders
    horizon_days = max(200, round(n / 8))
    start_date = pd.Timestamp(config.start_date)

    entities = _build_entities(rng)
    shocks = _build_shocks(rng, entities, horizon_days, start_date)

    vendor_ids = entities.vendors["vendor_id"].to_numpy()
    dc_ids = entities.dcs["dc_id"].to_numpy()
    lane_ids = entities.lanes["lane_id"].to_numpy()
    customer_ids = entities.customers["customer_id"].to_numpy()

    # --- Order skeleton: seasonality-weighted arrival dates -------------------
    day_pool = np.arange(horizon_days)
    seasonal_weight = _seasonal_multiplier(day_pool)
    order_day_offsets = rng.choice(day_pool, size=n, p=seasonal_weight / seasonal_weight.sum())
    order_dates = (start_date + pd.to_timedelta(order_day_offsets, unit="D")).to_numpy().copy()
    order_ids = [f"O{i:06d}" for i in range(1, n + 1)]

    selected_vendors = rng.choice(vendor_ids, n)
    selected_dcs = rng.choice(dc_ids, n)
    selected_lanes = rng.choice(lane_ids, n)
    selected_customers = rng.choice(customer_ids, n)
    order_priority = rng.choice(["STANDARD", "EXPEDITE"], n, p=[0.88, 0.12])

    # --- Scenario reservation (last K orders get scripted context) -----------
    n_scenarios = min(config.scenario_order_count, len(SCENARIO_TAGS))
    scenario_tag = np.full(n, "", dtype=object)
    scenario_index: dict[str, int] = {
        tag: n - n_scenarios + i for i, tag in enumerate(SCENARIO_TAGS[:n_scenarios])
    }
    for tag, index in scenario_index.items():
        scenario_tag[index] = tag

    scenario_day_offsets = {
        "multi_cause_propagation": int(horizon_days * 0.88),
        "resource_contention_a": int(horizon_days * 0.90),
        "resource_contention_b": int(horizon_days * 0.90),
        "line_level_stockout": int(horizon_days * 0.92),
        "uncertain_unknown_cause": int(horizon_days * 0.94),
    }
    for tag, index in scenario_index.items():
        order_dates[index] = np.datetime64(
            start_date + pd.Timedelta(days=scenario_day_offsets[tag])
        )

    idx_a = scenario_index.get("resource_contention_a")
    idx_b = scenario_index.get("resource_contention_b")
    idx_multi = scenario_index.get("multi_cause_propagation")
    if idx_a is not None and idx_b is not None:
        shared_vendor, shared_dc = vendor_ids[0], dc_ids[0]
        for idx in (idx_a, idx_b):
            selected_vendors[idx] = shared_vendor
            selected_dcs[idx] = shared_dc
            order_priority[idx] = "EXPEDITE"
        selected_customers[idx_a] = customer_ids[3]  # C004 maps to PLATINUM priority.
        selected_customers[idx_b] = customer_ids[0]  # C001 maps to GOLD priority.
        if idx_multi is not None:
            selected_vendors[idx_multi] = shared_vendor
            selected_dcs[idx_multi] = shared_dc
            order_priority[idx_multi] = "EXPEDITE"

    orders_frame_dates = pd.Series(pd.to_datetime(order_dates))
    requested = orders_frame_dates + pd.to_timedelta(rng.integers(8, 13, n), unit="D")
    promised = requested.copy()
    prediction = orders_frame_dates + pd.to_timedelta(config.prediction_horizon_days, unit="D")
    prediction = pd.Series(
        np.minimum(prediction.to_numpy(), (promised - pd.Timedelta(hours=1)).to_numpy())
    )

    # --- Shock membership per order (vendor / DC / lane) ----------------------
    vendor_shock_active, vendor_shock_severity = _shock_active(
        shocks, "vendor", selected_vendors, orders_frame_dates
    )
    dc_shock_active, dc_shock_severity = _shock_active(
        shocks, "dc", selected_dcs, orders_frame_dates
    )
    lane_shock_active, lane_shock_severity = _shock_active(
        shocks, "lane", selected_lanes, orders_frame_dates
    )

    # Force scenario shock membership so the qualitative story is guaranteed
    # while the numeric noise draws below remain random.
    if idx_multi is not None:
        vendor_shock_active[idx_multi] = True
        vendor_shock_severity[idx_multi] = max(vendor_shock_severity[idx_multi], 3.0)
        dc_shock_active[idx_multi] = True
        dc_shock_severity[idx_multi] = max(dc_shock_severity[idx_multi], 2.0)
    for idx in (idx_a, idx_b):
        if idx is not None:
            vendor_shock_active[idx] = True
            vendor_shock_severity[idx] = max(vendor_shock_severity[idx], 2.8)
            dc_shock_active[idx] = True
            dc_shock_severity[idx] = max(dc_shock_severity[idx], 1.8)
    idx_unknown = scenario_index.get("uncertain_unknown_cause")
    if idx_unknown is not None:
        vendor_shock_active[idx_unknown] = False
        dc_shock_active[idx_unknown] = False
        lane_shock_active[idx_unknown] = False

    # --- Order capture --------------------------------------------------------
    capture_baseline = rng.gamma(1.5, 4.0, n)
    capture_exception_roll = rng.random(n) < 0.06
    capture_extra = np.where(capture_exception_roll, rng.uniform(20, 50, n), 0.0)
    capture_delay_hours = capture_baseline + capture_extra
    if idx_unknown is not None:
        capture_delay_hours[idx_unknown] = float(rng.uniform(1, 6))

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "order_date": orders_frame_dates,
            "order_capture_timestamp": orders_frame_dates
            + pd.to_timedelta(capture_delay_hours, unit="h"),
            "prediction_timestamp": prediction,
            "requested_delivery_date": requested,
            "promised_delivery_date": promised,
            "vendor_id": selected_vendors,
            "dc_id": selected_dcs,
            "lane_id": selected_lanes,
            "customer_id": selected_customers,
            "order_priority": order_priority,
            "capture_delay_hours": capture_delay_hours.round(2),
            "scenario_tag": scenario_tag,
        }
    )

    # --- Line allocation (accumulated shortfall, not pre-selected) ------------
    order_lines, line_truth = _build_lines(
        rng, orders, entities.skus, vendor_shock_active, vendor_shock_severity, scenario_index
    )
    forced_shortfall_ids = [order_ids[idx] for idx in (idx_multi, idx_a, idx_b) if idx is not None]
    forced_clean_ids = (
        [order_ids[scenario_index["uncertain_unknown_cause"]]]
        if "uncertain_unknown_cause" in scenario_index
        else []
    )
    order_lines = _force_line_shortfall_for_scenarios(
        order_lines, forced_shortfall_ids, forced_clean_ids
    )
    line_truth = _sync_line_truth_with_forced_lines(
        line_truth, order_lines, forced_shortfall_ids, forced_clean_ids
    )
    line_totals = order_lines.groupby("order_id", as_index=False).agg(
        total_order_qty=("requested_qty", "sum"),
        shortfall_ratio=("shortfall_ratio", "mean"),
    )
    orders = orders.merge(
        line_totals[["order_id", "total_order_qty"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # --- Vendor readiness -------------------------------------------------------
    vendor_scale = entities.vendors.set_index("vendor_id")["delay_scale_hours"]
    contract_lead = entities.vendors.set_index("vendor_id")["contract_lead_days"]
    order_vendor_scale = pd.Series(selected_vendors).map(vendor_scale).to_numpy()
    order_contract_lead = pd.Series(selected_vendors).map(contract_lead).to_numpy()

    vendor_base_delay = rng.gamma(1.0, order_vendor_scale)
    vendor_shock_extra = np.where(
        vendor_shock_active, rng.gamma(2.0, 8.0, n) * vendor_shock_severity, 0.0
    )
    vendor_ready_delay_hours = vendor_base_delay + vendor_shock_extra
    vendor_observed = rng.random(n) < 0.97  # ~3% of vendor-ready events go unlogged entirely
    vendor_exception_logged = (vendor_ready_delay_hours > 24) & (rng.random(n) < 0.85)
    for idx in (idx_multi, idx_a, idx_b):
        if idx is not None:
            vendor_ready_delay_hours[idx] = 54.0
            vendor_exception_logged[idx] = True
            vendor_observed[idx] = True
    if idx_unknown is not None:
        vendor_ready_delay_hours[idx_unknown] = float(rng.uniform(1, 8))
        vendor_exception_logged[idx_unknown] = False
        vendor_observed[idx_unknown] = True

    planned_vendor = orders_frame_dates + pd.to_timedelta(order_contract_lead, unit="D")
    actual_vendor_ready = planned_vendor + pd.to_timedelta(vendor_ready_delay_hours, unit="h")

    # --- Warehouse / DC stage ----------------------------------------------------
    capacity_snapshots = _build_capacity_snapshots(
        rng, entities.dcs, orders, shocks, start_date, horizon_days
    )
    capacity_snapshots = _force_dc_utilization_for_scenarios(
        capacity_snapshots,
        [
            (selected_dcs[idx], orders_frame_dates.iloc[idx].normalize())
            for idx in (idx_multi, idx_a, idx_b)
            if idx is not None
        ],
        low_keys=(
            [(selected_dcs[idx_unknown], orders_frame_dates.iloc[idx_unknown].normalize())]
            if idx_unknown is not None
            else []
        ),
    )
    snapshot_lookup = capacity_snapshots.set_index(["dc_id", "snapshot_date"])["utilization"]
    order_snapshot_keys = list(zip(selected_dcs, orders_frame_dates.dt.normalize(), strict=True))
    utilization = np.array([float(snapshot_lookup.get(key, 0.5)) for key in order_snapshot_keys])
    warehouse_base_delay = rng.gamma(1.1, 2.2, n)
    warehouse_util_extra = np.clip(utilization - 0.80, 0, None) * rng.uniform(25, 55, n)
    shortfall_by_order = (
        line_totals.set_index("order_id")["shortfall_ratio"]
        .reindex(orders["order_id"])
        .fillna(0.0)
        .to_numpy()
    )
    warehouse_shortfall_extra = shortfall_by_order * rng.uniform(8, 20, n)
    warehouse_delay_hours = warehouse_base_delay + warehouse_util_extra + warehouse_shortfall_extra
    warehouse_exception_logged = (warehouse_delay_hours > 12) & (rng.random(n) < 0.85)
    warehouse_observed = rng.random(n) < 0.97
    for idx in (idx_multi, idx_a, idx_b):
        if idx is not None:
            warehouse_delay_hours[idx] = max(warehouse_delay_hours[idx], 26.0)
            warehouse_exception_logged[idx] = True
            warehouse_observed[idx] = True
    if idx_unknown is not None:
        warehouse_delay_hours[idx_unknown] = float(rng.uniform(1, 5))
        warehouse_exception_logged[idx_unknown] = False
        warehouse_observed[idx_unknown] = True

    planned_ship = orders_frame_dates + pd.Timedelta(days=3)
    ship_start = pd.Series(np.maximum(planned_ship.to_numpy(), actual_vendor_ready.to_numpy()))
    actual_ship = ship_start + pd.to_timedelta(warehouse_delay_hours, unit="h")

    # --- Transport stage -----------------------------------------------------------
    lane_variability = entities.lanes.set_index("lane_id")["transit_variability_days"]
    planned_transit_days = entities.lanes.set_index("lane_id")["planned_transit_days"]
    order_lane_variability = pd.Series(selected_lanes).map(lane_variability).to_numpy()
    order_planned_transit_days = pd.Series(selected_lanes).map(planned_transit_days).to_numpy()

    transit_base_delay = rng.gamma(1.0, order_lane_variability * 6.0)
    transit_shock_extra = np.where(
        lane_shock_active, rng.gamma(2.0, 10.0, n) * lane_shock_severity, 0.0
    )
    transit_delay_hours = transit_base_delay + transit_shock_extra
    transit_exception_logged = (transit_delay_hours > 15) & (rng.random(n) < 0.85)
    transit_observed = rng.random(n) < 0.97
    if idx_multi is not None:
        transit_delay_hours[idx_multi] = max(transit_delay_hours[idx_multi], 22.0)
        transit_exception_logged[idx_multi] = True
        transit_observed[idx_multi] = True
    if idx_unknown is not None:
        transit_delay_hours[idx_unknown] = float(rng.uniform(1, 5))
        transit_exception_logged[idx_unknown] = False
        transit_observed[idx_unknown] = True

    planned_transit_start = actual_ship + pd.Timedelta(days=1)
    actual_transit_arrival = (
        actual_ship
        + pd.to_timedelta(order_planned_transit_days, unit="D")
        + pd.to_timedelta(transit_delay_hours, unit="h")
    )

    # --- Customer delivery stage -----------------------------------------------------
    appointment_required = entities.customers.set_index("customer_id")["appointment_required"]
    reschedule_trait = entities.customers.set_index("customer_id")["reschedule_trait"]
    order_appointment_required = pd.Series(selected_customers).map(appointment_required).to_numpy()
    order_reschedule_trait = pd.Series(selected_customers).map(reschedule_trait).to_numpy()

    reschedule_roll = rng.random(n) < order_reschedule_trait
    customer_delay_active = order_appointment_required & reschedule_roll
    customer_delay_hours = np.where(customer_delay_active, rng.uniform(24, 72, n), 0.0)
    customer_exception_logged = customer_delay_active & (rng.random(n) < 0.90)

    # Small organic unexplained-miss noise, independent of any observable cause.
    unknown_roll = rng.random(n) < 0.015
    unknown_extra_hours = np.where(unknown_roll, rng.uniform(24, 60, n), 0.0)
    if idx_unknown is not None:
        unknown_extra_hours[idx_unknown] = float(rng.uniform(90, 140))
        unknown_roll[idx_unknown] = True
        customer_delay_active[idx_unknown] = False
        customer_exception_logged[idx_unknown] = False

    delivered = actual_transit_arrival + pd.to_timedelta(
        customer_delay_hours + unknown_extra_hours, unit="h"
    )
    if idx_unknown is not None:
        # Direct, deterministic guarantee: this order misses its promised
        # date by a comfortable margin regardless of the random requested
        # -delivery-window draw, with zero corroborating stage evidence.
        delivered.iloc[idx_unknown] = promised.iloc[idx_unknown] + pd.Timedelta(hours=48)

    # --- Events -----------------------------------------------------------------------
    events = _build_events(
        order_ids,
        planned_vendor,
        actual_vendor_ready,
        vendor_observed,
        vendor_exception_logged,
        planned_ship,
        actual_ship,
        warehouse_observed,
        warehouse_exception_logged,
        planned_transit_start,
        actual_transit_arrival,
        transit_observed,
        transit_exception_logged,
        promised,
        delivered,
        customer_exception_logged,
        unknown_roll,
    )

    simulator_truth = pd.DataFrame(
        {
            "order_id": order_ids,
            "vendor_shock_active": vendor_shock_active,
            "vendor_shock_severity": np.where(vendor_shock_active, vendor_shock_severity, 0.0),
            "dc_shock_active": dc_shock_active,
            "dc_shock_severity": np.where(dc_shock_active, dc_shock_severity, 0.0),
            "lane_shock_active": lane_shock_active,
            "lane_shock_severity": np.where(lane_shock_active, lane_shock_severity, 0.0),
            "vendor_ready_delay_hours": vendor_ready_delay_hours.round(2),
            "warehouse_delay_hours": warehouse_delay_hours.round(2),
            "transit_delay_hours": transit_delay_hours.round(2),
            "customer_delay_hours": customer_delay_hours.round(2),
            "unknown_extra_hours": unknown_extra_hours.round(2),
            "accumulated_delay_hours": (
                vendor_ready_delay_hours
                + warehouse_delay_hours
                + transit_delay_hours
                + customer_delay_hours
                + unknown_extra_hours
            ).round(2),
            "shortfall_ratio": shortfall_by_order.round(4),
            "scenario_tag": scenario_tag,
        }
    )

    dataset = PrototypeDataset(
        orders=orders,
        order_lines=order_lines.drop(columns=["shortfall_ratio"], errors="ignore"),
        events=events,
        vendors=entities.vendors,
        dcs=entities.dcs,
        lanes=entities.lanes,
        customers=entities.customers,
        skus=entities.skus,
        capacity_snapshots=capacity_snapshots,
        simulator_truth=simulator_truth,
        line_truth=line_truth,
        shocks=shocks,
    )
    validate_dataset(dataset)
    return dataset


def _build_lines(
    rng: np.random.Generator,
    orders: pd.DataFrame,
    skus: pd.DataFrame,
    vendor_shock_active: np.ndarray,
    vendor_shock_severity: np.ndarray,
    scenario_index: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build order lines with accumulated (not pre-selected) shortfall.

    Two independent shortfall mechanisms are modeled: a small stable
    ``scarcity_trait``-driven baseline (known essentially at capture time,
    via ``allocated_qty``) and a vendor-shock-driven *secondary* shortfall
    that is only realized once the vendor-ready stage resolves
    (``shipped_qty < allocated_qty``). The second mechanism is what makes
    some misses show no early evidence: the initial allocation can look
    fully healthy while a later vendor failure still cuts what actually
    ships.
    """
    scarcity = skus.set_index("sku_id")["scarcity_trait"]
    criticality = skus.set_index("sku_id")["criticality_tier"]
    sku_ids = skus["sku_id"].to_numpy()
    n = len(orders)
    line_counts = rng.integers(1, 4, n)

    line_level_idx = scenario_index.get("line_level_stockout")
    if line_level_idx is not None:
        line_counts[line_level_idx] = 3
    contention_indices = {
        index
        for tag in ("resource_contention_a", "resource_contention_b")
        if (index := scenario_index.get(tag)) is not None
    }
    for index in contention_indices:
        line_counts[index] = 2

    line_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    for order_index, order in enumerate(orders.itertuples(index=False)):
        order_id = order.order_id
        count = int(line_counts[order_index])
        quantities = rng.integers(5, 60, count)
        if order_index in contention_indices:
            quantities = np.full(count, 240)
        chosen_skus = rng.choice(sku_ids, count)
        if order_index == line_level_idx:
            forced_critical_sku = str(
                skus.loc[skus["criticality_tier"] == "CRITICAL", "sku_id"].iloc[0]
            )
            chosen_skus = chosen_skus.copy()
            chosen_skus[0] = forced_critical_sku

        for line_number in range(1, count + 1):
            sku_id = str(chosen_skus[line_number - 1])
            requested_qty = int(quantities[line_number - 1])
            base_shortfall_prob = float(scarcity.get(sku_id, 0.05))
            immediate_roll = rng.random() < base_shortfall_prob
            immediate_shortfall_fraction = rng.uniform(0.1, 0.4) if immediate_roll else 0.0

            vendor_active = bool(vendor_shock_active[order_index])
            secondary_prob = 0.028
            if vendor_active:
                secondary_prob = min(
                    0.9, base_shortfall_prob + 0.22 * vendor_shock_severity[order_index]
                )
            secondary_roll = rng.random() < secondary_prob
            secondary_shortfall_fraction = rng.uniform(0.15, 0.7) if secondary_roll else 0.0

            if order_index == line_level_idx:
                if line_number == 1:
                    secondary_roll = True
                    secondary_shortfall_fraction = rng.uniform(0.5, 0.75)
                    immediate_roll = False
                    immediate_shortfall_fraction = 0.0
                else:
                    secondary_roll = False
                    secondary_shortfall_fraction = 0.0
                    immediate_roll = False
                    immediate_shortfall_fraction = 0.0

            allocated_qty = max(0, round(requested_qty * (1 - immediate_shortfall_fraction)))
            # Point-in-time-safe capture flag: known immediately from the ATP
            # allocation, independent of what happens at the vendor stage.
            capture_time_stockout = allocated_qty < requested_qty
            # Some capture-time shortfalls get backfilled by an expedited
            # replenishment before shipment -- this decouples the knowable
            # capture-time signal from the ultimate (only-later-observable)
            # shipped outcome, so an early stockout flag is a genuine risk
            # factor rather than a certainty.
            recovered_by_shipment = capture_time_stockout and (rng.random() < 0.32)
            effective_qty_for_shipping = requested_qty if recovered_by_shipment else allocated_qty
            shipped_qty = max(
                0, round(effective_qty_for_shipping * (1 - secondary_shortfall_fraction))
            )
            inventory_noise = rng.normal(0, 0.05)
            inventory_available_at_order = max(0, round(allocated_qty * (1 + inventory_noise)))
            shortfall_ratio = 1 - (shipped_qty / requested_qty if requested_qty else 1)

            line_rows.append(
                {
                    "order_line_id": f"{order_id}-{line_number}",
                    "order_id": order_id,
                    "line_number": line_number,
                    "sku_id": sku_id,
                    "requested_qty": requested_qty,
                    "allocated_qty": allocated_qty,
                    "shipped_qty": shipped_qty,
                    "inventory_available_at_order": inventory_available_at_order,
                    "stockout_flag": bool(capture_time_stockout),
                    "shortfall_ratio": round(shortfall_ratio, 4),
                }
            )
            truth_rows.append(
                {
                    "order_line_id": f"{order_id}-{line_number}",
                    "order_id": order_id,
                    "sku_id": sku_id,
                    "criticality_tier": str(criticality.get(sku_id, "STANDARD")),
                    "immediate_shortfall": immediate_roll,
                    "vendor_stage_shortfall": secondary_roll,
                    "truly_affected": bool(shortfall_ratio > 0.05),
                    "shortfall_ratio": round(shortfall_ratio, 4),
                }
            )
    return pd.DataFrame(line_rows), pd.DataFrame(truth_rows)


def _build_capacity_snapshots(
    rng: np.random.Generator,
    dcs: pd.DataFrame,
    orders: pd.DataFrame,
    shocks: pd.DataFrame,
    start_date: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    all_days = pd.date_range(start_date, periods=horizon_days + 15, freq="D")
    day_of_year = all_days.dayofyear.to_numpy()
    seasonal = _seasonal_multiplier(day_of_year)
    capacity_orders = orders.assign(snapshot_date=orders["order_date"].dt.normalize())
    dc_shocks = shocks.loc[shocks["entity_type"] == "dc"]

    rows: list[dict[str, object]] = []
    for dc in dcs.itertuples(index=False):
        base = int(dc.daily_capacity_units)
        variability = float(dc.capacity_variability)
        daily_load = (
            capacity_orders.loc[capacity_orders["dc_id"] == dc.dc_id]
            .groupby("snapshot_date")["total_order_qty"]
            .sum()
        )
        # Day-level shock severity for *this* DC only, active strictly within
        # each shock's own date window (never bleeding across the full horizon).
        day_severity = np.ones(len(all_days))
        for shock in dc_shocks.loc[dc_shocks["entity_id"] == dc.dc_id].itertuples(index=False):
            mask = (all_days >= shock.start_date) & (all_days <= shock.end_date)
            day_severity[mask] = np.maximum(day_severity[mask], shock.severity)
        organic_baseline = rng.integers(int(base * 0.35), int(base * 0.60), len(all_days))
        noise = rng.normal(1.0, variability * 0.5, len(all_days))
        for position, day in enumerate(all_days):
            order_load = int(daily_load.get(day, 0))
            planned_units = max(
                0,
                int(
                    (organic_baseline[position] + order_load)
                    * day_severity[position]
                    * seasonal[position]
                    * noise[position]
                ),
            )
            rows.append(
                {
                    "dc_id": dc.dc_id,
                    "snapshot_date": day,
                    "available_capacity_units": base,
                    "planned_units": planned_units,
                    "utilization": round(planned_units / base, 4),
                }
            )
    return pd.DataFrame(rows)


def _build_events(
    order_ids: list[str],
    planned_vendor: pd.Series,
    actual_vendor_ready: pd.Series,
    vendor_observed: np.ndarray,
    vendor_exception_logged: np.ndarray,
    planned_ship: pd.Series,
    actual_ship: pd.Series,
    warehouse_observed: np.ndarray,
    warehouse_exception_logged: np.ndarray,
    planned_transit_start: pd.Series,
    actual_transit_arrival: pd.Series,
    transit_observed: np.ndarray,
    transit_exception_logged: np.ndarray,
    promised: pd.Series,
    delivered: pd.Series,
    customer_exception_logged: np.ndarray,
    unexplained: np.ndarray,
) -> pd.DataFrame:
    """Assemble the events table, honoring partial observability (missing events)."""
    rows: list[dict[str, object]] = []
    for index, order_id in enumerate(order_ids):
        if vendor_observed[index]:
            rows.append(
                {
                    "order_id": order_id,
                    "event_type": "VENDOR_READY",
                    "planned_timestamp": planned_vendor.iloc[index],
                    "event_timestamp": actual_vendor_ready.iloc[index],
                    "exception_code": (
                        "SUPPLIER_LATE" if vendor_exception_logged[index] else None
                    ),
                }
            )
        if warehouse_observed[index]:
            rows.append(
                {
                    "order_id": order_id,
                    "event_type": "SHIPPED",
                    "planned_timestamp": planned_ship.iloc[index],
                    "event_timestamp": actual_ship.iloc[index],
                    "exception_code": (
                        "PICK_PACK_DELAY" if warehouse_exception_logged[index] else None
                    ),
                }
            )
        if transit_observed[index]:
            rows.append(
                {
                    "order_id": order_id,
                    "event_type": "IN_TRANSIT",
                    "planned_timestamp": planned_transit_start.iloc[index],
                    "event_timestamp": actual_transit_arrival.iloc[index],
                    "exception_code": (
                        "CARRIER_DELAY" if transit_exception_logged[index] else None
                    ),
                }
            )
        rows.append(
            {
                "order_id": order_id,
                "event_type": "DELIVERED",
                "planned_timestamp": promised.iloc[index],
                "event_timestamp": delivered.iloc[index],
                "exception_code": (
                    "CUSTOMER_APPOINTMENT"
                    if customer_exception_logged[index]
                    else ("UNEXPLAINED" if unexplained[index] else None)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["order_id", "event_timestamp"])
