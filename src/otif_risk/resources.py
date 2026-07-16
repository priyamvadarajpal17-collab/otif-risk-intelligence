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
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
