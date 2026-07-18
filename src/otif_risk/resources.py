"""Generic, transparent daily resource capacities for interventions.

Each mitigation type maps to exactly one resource pool and a transparent
demand unit (see ``RESOURCE_TYPE_FOR_CAUSE`` / ``demand_units_for``).
Allocation is a deterministic greedy priority policy -- not a MILP optimizer:
within a resource pool, the highest-priority candidates are accepted up to
that pool's capacity for the day; everything past capacity is marked
``CONTESTED`` and annotated with the other order IDs it is competing with.

This module is used by the daily operations replay (``operations.py``), where
capacities genuinely reset each simulated day. The one-shot canonical
pipeline (``pipeline.py``) uses ``decisions.recommend_orders``, which already
implements a compatible (DC quantity/capacity-aware, count-based elsewhere)
version of the same policy for a single scored snapshot.

At their default (100%) values, this twin's daily capacities are sized
generously relative to typical daily eligible-order volume, so they rarely
bind in practice -- a deliberately conservative production default, but one
that gives a capacity-priority *ranking* very little to be discriminative
about. ``CAPACITY_SCENARIOS``/``build_capacity_schedule`` let the Decision
Value Lab (``policy_evaluation.py``) re-run every policy under uniformly
scaled-down capacity to answer the sharper question: does the deployed
ranking still create more value than simpler baselines when capacity is
genuinely scarce?
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from otif_risk.contracts import PrototypeDataset
from otif_risk.decisions import CONTESTED, RECOMMENDATION_TABLE, RECOMMENDED

#: Default fraction of a DC's daily throughput assumed available for
#: concurrent expedited-recovery actions (INVENTORY_SHORTAGE/DC_CAPACITY/WAREHOUSE_OPS).
DEFAULT_DC_RECOVERY_FRACTION = 0.20
#: Transparent, documented assumption: no per-lane capacity field exists in
#: this prototype's data, so alternate-transport capacity is a flat unit pool.
DEFAULT_LANE_ALTERNATE_CAPACITY_UNITS = 140.0
#: Vendor escalation slots/day and customer appointment slots/day: small
#: integer pools, since these are headcount-style constraints, not quantity ones.
DEFAULT_VENDOR_ESCALATION_SLOTS = 1.0
DEFAULT_CUSTOMER_APPOINTMENT_SLOTS = 2.0

RESOURCE_POOL_FOR_TYPE = {
    "dc": "dc_units",
    "lane": "lane_units",
    "vendor": "vendor_slots",
    "customer": "customer_slots",
}

#: Pre-specified capacity-stress sensitivity scenarios: every policy under
#: evaluation (including the evaluation-only oracle) is scored against the
#: *same* multiplier applied uniformly to every resource pool -- see
#: ``build_capacity_schedule``. Keys are the canonical scenario names used
#: throughout reporting; values are the multiplier applied to every pool's
#: base daily capacity. Ordered ascending (scarcest first) for readable
#: reports. ``policy_evaluation.PRIMARY_CAPACITY_SCENARIO`` designates
#: ``SCARCE_50_PERCENT`` as the Stage 1 headline: at the unscaled 100%
#: baseline this twin's default capacities are rarely binding (see this
#: module's docstring), which is not discriminative between policies, so it
#: cannot answer "does this ranking create more value than simpler
#: baselines under real scarcity" -- the business question Stage 1 exists
#: to answer.
CAPACITY_SCENARIOS: dict[str, float] = {
    "SCARCE_25_PERCENT": 0.25,
    "SCARCE_50_PERCENT": 0.5,
    "BASE_100_PERCENT": 1.0,
}


@dataclass
class ResourceCapacities:
    """One day's remaining capacity for each of the four resource pools."""

    dc_units: dict[str, float] = field(default_factory=dict)
    lane_units: dict[str, float] = field(default_factory=dict)
    vendor_slots: dict[str, float] = field(default_factory=dict)
    customer_slots: dict[str, float] = field(default_factory=dict)

    def pool(self, resource_type: str) -> dict[str, float]:
        attribute = RESOURCE_POOL_FOR_TYPE.get(resource_type)
        if attribute is None:
            raise ValueError(f"unknown resource_type: {resource_type}")
        return getattr(self, attribute)


def default_daily_capacities(
    dataset: PrototypeDataset,
    *,
    dc_recovery_fraction: float = DEFAULT_DC_RECOVERY_FRACTION,
    lane_alternate_capacity_units: float = DEFAULT_LANE_ALTERNATE_CAPACITY_UNITS,
    vendor_escalation_slots: float = DEFAULT_VENDOR_ESCALATION_SLOTS,
    customer_appointment_slots: float = DEFAULT_CUSTOMER_APPOINTMENT_SLOTS,
) -> ResourceCapacities:
    """Build one day's fresh capacities from the dataset's real DC throughput
    plus transparent, documented flat assumptions for lane/vendor/customer
    pools (this prototype has no equivalent numeric fields for those).
    """
    if not 0 < dc_recovery_fraction <= 1:
        raise ValueError("dc_recovery_fraction must be in (0, 1]")
    dc_units = {
        row.dc_id: float(row.daily_capacity_units) * dc_recovery_fraction
        for row in dataset.dcs.itertuples(index=False)
    }
    lane_units = {
        lane_id: lane_alternate_capacity_units for lane_id in dataset.lanes["lane_id"]
    }
    vendor_slots = {
        vendor_id: vendor_escalation_slots for vendor_id in dataset.vendors["vendor_id"]
    }
    customer_slots = {
        customer_id: customer_appointment_slots for customer_id in dataset.customers["customer_id"]
    }
    return ResourceCapacities(
        dc_units=dc_units,
        lane_units=lane_units,
        vendor_slots=vendor_slots,
        customer_slots=customer_slots,
    )


def _day_key(day: Any) -> str:
    return day.isoformat() if hasattr(day, "isoformat") else str(day)


def build_capacity_schedule(
    dataset: PrototypeDataset,
    days: Sequence[Any],
    multiplier: float,
    **capacity_kwargs: float,
) -> dict[str, ResourceCapacities]:
    """Precompute one capacity-stress-scaled ``ResourceCapacities`` snapshot
    per calendar day in ``days``, for a single ``multiplier`` applied
    uniformly to every resource pool. The returned schedule is built once
    and is meant to be shared, unmodified, across every policy evaluated
    against it -- capacity is a property of the *scenario*, never of which
    policy is consuming it.

    **Continuous pools** (``dc``/``lane``, quantity-denominated -- see
    ``demand_units_for``) scale directly: ``base_capacity * multiplier``
    every day, with no rounding, since a fractional throughput unit is
    already a meaningful quantity.

    **Discrete pools** (``vendor``/``customer``, always exactly 1 demand
    unit/order) are small integer slot counts (1 vendor escalation slot, 2
    customer appointment slots/day by default -- see
    ``DEFAULT_VENDOR_ESCALATION_SLOTS``/``DEFAULT_CUSTOMER_APPOINTMENT_SLOTS``),
    so ``base_capacity * multiplier`` is frequently a fraction smaller than
    one whole slot (e.g. a 1-slot pool at 25%/50%). Flooring that fraction
    to zero every day would silently erase the pool for the entire
    scenario, and drawing a fresh random 0/1 outcome per day would let
    different policies see different realized capacity purely by chance.
    Instead this uses a deterministic whole-slot schedule -- an
    error-diffusion ("Bresenham line") accumulator keyed by
    ``(resource_type, resource_id)`` and walked once across ``days`` in
    chronological order -- that assigns each day either
    ``floor(running_target)`` or one more whole slot, with the exact
    long-run average across ``days`` equal to ``base_capacity *
    multiplier``: a 1-slot pool at 50% realizes exactly 0, 1, 0, 1, ...; at
    25% it realizes a slot on exactly one day out of every four. The
    schedule depends only on the day sequence, ``multiplier``, and the
    dataset's base capacities -- never on seed, priority order, or which
    policy is asking -- so every policy allocating against this schedule
    sees byte-identical realized capacity on every day, even under scarce,
    sub-one-slot scenarios. At ``multiplier == 1.0`` the accumulator never
    accrues a fractional remainder (each day's target is already a whole
    number equal to the base capacity), so this schedule reduces exactly
    to ``default_daily_capacities`` repeated every day -- no scheduling
    artifact at the unscaled baseline.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    base = default_daily_capacities(dataset, **capacity_kwargs)
    day_keys = [_day_key(day) for day in days]
    if len(set(day_keys)) != len(day_keys):
        raise ValueError("days must be unique")

    schedule: dict[str, ResourceCapacities] = {key: ResourceCapacities() for key in day_keys}

    # Continuous pools: same directly-scaled value every day, no rounding.
    for key in day_keys:
        schedule[key].dc_units = {
            resource_id: capacity * multiplier for resource_id, capacity in base.dc_units.items()
        }
        schedule[key].lane_units = {
            resource_id: capacity * multiplier for resource_id, capacity in base.lane_units.items()
        }

    # Discrete pools: deterministic whole-slot accumulator per resource,
    # walked once across the full chronological day sequence so every
    # policy shares the exact same realized per-day capacity.
    discrete_pools = (
        ("vendor", base.vendor_slots),
        ("customer", base.customer_slots),
    )
    for resource_type, base_pool in discrete_pools:
        for resource_id, base_capacity in base_pool.items():
            accumulator = 0.0
            for key in day_keys:
                accumulator += base_capacity * multiplier
                day_capacity = math.floor(accumulator + 1e-9)
                accumulator -= day_capacity
                target_pool = (
                    schedule[key].vendor_slots
                    if resource_type == "vendor"
                    else schedule[key].customer_slots
                )
                target_pool[resource_id] = float(day_capacity)
    return schedule


def demand_units_for(resource_type: str, quantity_at_risk: float) -> float:
    """Transparent per-order demand for the resource its mitigation consumes.

    DC/lane pools are quantity-denominated (real recovery capacity is a
    throughput constraint); vendor/customer pools are slot-denominated (one
    order consumes exactly one escalation/appointment slot, since these are
    headcount-style constraints, not quantity ones).
    """
    if resource_type in ("dc", "lane"):
        return max(float(quantity_at_risk), 0.0)
    return 1.0


def allocate_under_capacity(
    candidates: pd.DataFrame,
    capacities: ResourceCapacities,
) -> tuple[pd.DataFrame, ResourceCapacities]:
    """Greedily allocate already-classified interventions by priority under capacity.

    Unlike :func:`allocate_interventions`, this does **not** derive
    ``resource_type``/``resource_id``/``demand_units`` from ``primary_cause`` --
    the caller must already have attached those three columns (plus
    ``order_id`` and ``priority_score``). This is the shared capacity-fairness
    core used both by ``allocate_interventions`` (production
    cause-to-resource-type mapping) and by ``policy_evaluation`` (which needs
    to allocate capacity for policies whose chosen *action* -- and therefore
    resource pool -- is not always the one ``primary_cause`` would imply, e.g.
    the evaluation-only oracle policy).
    """
    required = {"order_id", "resource_type", "resource_id", "priority_score", "demand_units"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing required columns: {sorted(missing)}")

    result = candidates.copy()
    result["decision_status"] = RECOMMENDED
    result["contested_with"] = ""

    remaining = ResourceCapacities(
        dc_units=dict(capacities.dc_units),
        lane_units=dict(capacities.lane_units),
        vendor_slots=dict(capacities.vendor_slots),
        customer_slots=dict(capacities.customer_slots),
    )
    ranked = result.sort_values(
        ["priority_score", "order_id"], ascending=[False, True], kind="stable"
    )
    for (resource_type, resource_id), group in ranked.groupby(
        ["resource_type", "resource_id"], sort=False
    ):
        pool = remaining.pool(resource_type)
        capacity = pool.get(resource_id, 0.0)
        cumulative = group["demand_units"].cumsum()
        overflow_mask = cumulative > capacity
        overflow_index = group.index[overflow_mask]
        accepted_index = group.index[~overflow_mask]
        accepted_demand = float(group.loc[accepted_index, "demand_units"].sum())
        pool[resource_id] = max(0.0, capacity - accepted_demand)
        if len(overflow_index):
            all_ids = group["order_id"].tolist()
            result.loc[overflow_index, "decision_status"] = CONTESTED
            for index in overflow_index:
                competitors = [
                    order_id for order_id in all_ids if order_id != result.at[index, "order_id"]
                ]
                result.at[index, "contested_with"] = ",".join(competitors)
    return result, remaining


def allocate_interventions(
    candidates: pd.DataFrame,
    capacities: ResourceCapacities,
) -> tuple[pd.DataFrame, ResourceCapacities]:
    """Greedily allocate interventions by priority under daily capacity.

    ``candidates`` must contain: ``order_id``, ``primary_cause``,
    ``priority_score``, ``quantity_at_risk``. Returns a copy with
    ``decision_status`` (RECOMMENDED/CONTESTED), ``resource_type``,
    ``resource_id``, ``demand_units``, and ``contested_with`` (a comma-joined
    list of the other order IDs competing for the same exhausted pool),
    plus the capacities remaining after today's allocation.
    """
    required = {"order_id", "primary_cause", "priority_score", "quantity_at_risk"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing required columns: {sorted(missing)}")

    result = candidates.copy()
    policies = result["primary_cause"].astype(str).str.upper().map(RECOMMENDATION_TABLE)
    result["resource_type"] = policies.map(
        lambda value: value["resource_type"] if isinstance(value, dict) else "dc"
    )
    resource_key_column = {
        "dc": "dc_id",
        "lane": "lane_id",
        "vendor": "vendor_id",
        "customer": "customer_id",
    }
    result["resource_id"] = [
        str(result.at[index, resource_key_column[kind]])
        if resource_key_column[kind] in result.columns
        and pd.notna(result.at[index, resource_key_column[kind]])
        else "UNASSIGNED"
        for index, kind in zip(result.index, result["resource_type"], strict=True)
    ]
    result["demand_units"] = [
        demand_units_for(kind, qty)
        for kind, qty in zip(result["resource_type"], result["quantity_at_risk"], strict=True)
    ]
    return allocate_under_capacity(result, capacities)
