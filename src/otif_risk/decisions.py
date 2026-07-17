"""Deterministic decision policy for the standalone OTIF prototype."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from .contracts import CAUSE_CATEGORIES

if TYPE_CHECKING:
    from .contracts import PrototypeDataset

RECOMMENDED = "RECOMMENDED"
CONTESTED = "CONTESTED"
MONITOR = "MONITOR"

DEFAULT_RISK_THRESHOLD = 0.50
# Count-based fallback limits. These remain genuinely arbitrary for vendor/lane
# /customer because this prototype has no numeric recovery-capacity field for
# those dimensions (no "how many escalations can Supplier Management run today"
# data exists). DC conflicts instead use real quantity/capacity data below
# whenever `dc_daily_capacity_units` is present on the scored frame; this count
# is only the fallback for DC when that column is absent.
DEFAULT_RESOURCE_LIMITS = {"vendor": 1, "dc": 3, "lane": 2, "customer": 2}
#: Fraction of a DC's daily throughput capacity assumed available for
#: concurrent expedited-recovery actions (the remainder runs normal operations).
DEFAULT_DC_RECOVERY_CAPACITY_FRACTION = 0.20

# The prototype deliberately uses an auditable policy table rather than an optimizer.
RECOMMENDATION_TABLE: dict[str, dict[str, str]] = {
    "ORDER_CAPTURE": {
        "action": "Validate order details and release any capture hold",
        "owner": "Customer operations",
        "resource_type": "customer",
    },
    "VENDOR_FAILURE": {
        "action": "Escalate supplier recovery and confirm replacement supply",
        "owner": "Supplier management",
        "resource_type": "vendor",
    },
    "INVENTORY_SHORTAGE": {
        "action": "Reallocate available inventory and protect critical demand",
        "owner": "Inventory planning",
        "resource_type": "dc",
    },
    "DC_CAPACITY": {
        "action": "Reserve DC capacity and reprioritize the outbound wave",
        "owner": "DC operations",
        "resource_type": "dc",
    },
    "WAREHOUSE_OPS": {
        "action": "Expedite pick-pack and resolve the warehouse exception",
        "owner": "Warehouse operations",
        "resource_type": "dc",
    },
    "TRANSPORT": {
        "action": "Secure alternate transport capacity for the affected lane",
        "owner": "Transportation",
        "resource_type": "lane",
    },
    "CUSTOMER_DELIVERY": {
        "action": "Coordinate a revised delivery appointment with the customer",
        "owner": "Customer operations",
        "resource_type": "customer",
    },
}

FALLBACK_RECOMMENDATION = {
    "action": "Review the order exception and confirm a recovery plan",
    "owner": "OTIF control tower",
    "resource_type": "dc",
}


def primary_cause_from_signals(row: Mapping[str, Any]) -> str:
    """Return the first active ``leading_signal_{cause}`` in upstream-priority order.

    This is the scoring-time (predicted, not ground-truth) cause used to pick
    a recommended action: it reads only observable-by-prediction-time
    ``leading_signal_*`` feature flags, never the retrospective
    ``root_causes.derive_root_causes`` rule evaluation used for training
    labels/evaluation. Shared by ``pipeline.score_orders`` and
    ``action_response``/``policy_evaluation`` so every consumer of a scored
    order's ``primary_cause`` agrees on how it was chosen.
    """
    active = [
        category
        for category in CAUSE_CATEGORIES
        if int(row.get(f"leading_signal_{category}", 0)) == 1
    ]
    if not active:
        return "UNKNOWN"
    return active[0]


def attach_business_context(scored: pd.DataFrame, dataset: PrototypeDataset) -> pd.DataFrame:
    """Attach ``order_value``/``customer_tier``/``penalty_rate`` to a scored frame.

    ``order_value`` sums each line's requested quantity times its SKU's base
    unit value (missing SKU prices default to $50/unit). ``customer_tier`` is
    a deterministic function of the customer ID (not random), so the same
    customer always gets the same tier/penalty rate across every run and
    every consumer (``pipeline.score_orders`` for the deployed decision
    frame, ``action_response`` for evaluation-only realized penalty).
    """
    lines_with_value = dataset.order_lines.merge(
        dataset.skus[["sku_id", "base_unit_value"]],
        on="sku_id",
        how="left",
        validate="many_to_one",
    )
    requested_qty = lines_with_value["requested_qty"].astype(float)
    unit_value = lines_with_value["base_unit_value"].fillna(50.0)
    lines_with_value["line_value"] = requested_qty * unit_value
    line_context = lines_with_value.groupby("order_id", as_index=False).agg(
        order_value=("line_value", "sum"),
        representative_sku=("sku_id", "first"),
    )
    enriched = scored.merge(line_context, on="order_id", how="left", validate="one_to_one")
    customer_number = enriched["customer_id"].astype(str).str.extract(r"(\d+)", expand=False)
    customer_number = pd.to_numeric(customer_number, errors="coerce").fillna(0).astype(int)
    enriched["customer_tier"] = customer_number.mod(4).map(
        {0: "PLATINUM", 1: "GOLD", 2: "SILVER", 3: "BRONZE"}
    )
    enriched["penalty_rate"] = enriched["customer_tier"].map(
        {"PLATINUM": 0.05, "GOLD": 0.03, "SILVER": 0.02, "BRONZE": 0.01}
    )
    return enriched


@dataclass(frozen=True)
class ImpactAssumptions:
    """Transparent monetary and service assumptions used by this prototype."""

    avoided_risk_fraction: float = 0.60
    quantity_at_risk_fraction: float = 0.50
    default_penalty_rate: float = 0.02
    default_customer_tier_weight: float = 1.0


DEFAULT_IMPACT_ASSUMPTIONS = ImpactAssumptions()


def _numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


def _tier_weight(series: pd.Series) -> pd.Series:
    weights = {"PLATINUM": 1.30, "GOLD": 1.15, "SILVER": 1.00, "BRONZE": 0.90}
    return series.astype(str).str.upper().map(weights).fillna(1.0)


def _validate_orders(orders: pd.DataFrame) -> None:
    required = {"order_id", "combined_risk_score", "primary_cause"}
    missing = sorted(required - set(orders.columns))
    if missing:
        raise ValueError(f"Missing required scored-order columns: {', '.join(missing)}")


def _resource_overflow(
    group: pd.DataFrame,
    kind: str,
    limits: Mapping[str, int],
    dc_capacity_recovery_fraction: float,
) -> pd.Index:
    """Return the (already priority-sorted) indices that exceed this resource's capacity.

    For ``dc`` groups with a usable ``dc_daily_capacity_units`` value, capacity is
    quantity-based: candidates are accepted in priority order while their
    cumulative ``quantity_at_risk`` stays within the DC's recovery allowance, and
    every candidate once that allowance is exceeded is contested — including a
    single order whose own ``quantity_at_risk`` alone exceeds the allowance, since
    it genuinely cannot be resourced within capacity. All other resource types
    (and DC when capacity data is unavailable) keep the count-based limit.
    """
    if kind == "dc" and "dc_daily_capacity_units" in group.columns:
        capacity_values = pd.to_numeric(group["dc_daily_capacity_units"], errors="coerce").dropna()
        if not capacity_values.empty and capacity_values.iloc[0] > 0:
            capacity_units = float(capacity_values.iloc[0]) * dc_capacity_recovery_fraction
            quantity_at_risk = pd.to_numeric(
                group["quantity_at_risk"], errors="coerce"
            ).fillna(0.0)
            cumulative = quantity_at_risk.cumsum()
            return group.index[cumulative > capacity_units]
    return group.index[limits.get(kind, 1) :]


def recommend_orders(
    orders: pd.DataFrame,
    *,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    resource_limits: Mapping[str, int] | None = None,
    assumptions: ImpactAssumptions = DEFAULT_IMPACT_ASSUMPTIONS,
    dc_capacity_recovery_fraction: float = DEFAULT_DC_RECOVERY_CAPACITY_FRACTION,
) -> pd.DataFrame:
    """Apply lookup recommendations and a deterministic shared-resource capacity check.

    Candidate orders are ranked by priority. For each resource, the highest
    priority interventions are recommended up to that resource's capacity;
    remaining candidates are marked contested. This is intentionally not a MILP.

    DC conflicts are quantity/capacity aware: when the scored frame carries
    ``dc_daily_capacity_units`` (the DC's real daily throughput capacity), a DC's
    allowed recovery load is ``dc_daily_capacity_units * dc_capacity_recovery_fraction``
    units, and candidates are accepted, in priority order, until their cumulative
    ``quantity_at_risk`` would exceed that allowance — a genuine capacity
    argument rather than an arbitrary headcount. When that column is absent (or
    non-positive), DC falls back to the same count-based limit used for vendor,
    lane, and customer, which remain count-based because this prototype has no
    equivalent numeric recovery-capacity field for those dimensions.
    """

    _validate_orders(orders)
    if not 0 <= risk_threshold <= 1:
        raise ValueError("risk_threshold must be in [0, 1]")
    if not 0 < dc_capacity_recovery_fraction <= 1:
        raise ValueError("dc_capacity_recovery_fraction must be in (0, 1]")

    limits = dict(DEFAULT_RESOURCE_LIMITS)
    if resource_limits is not None:
        limits.update(resource_limits)
    if any(not isinstance(value, int) or value < 1 for value in limits.values()):
        raise ValueError("resource limits must be positive integers")

    result = orders.copy()
    policies = result["primary_cause"].astype(str).str.upper().map(RECOMMENDATION_TABLE)
    policies = policies.map(
        lambda value: value if isinstance(value, dict) else FALLBACK_RECOMMENDATION
    )
    result["recommended_action"] = policies.map(lambda value: value["action"])
    result["action_owner"] = policies.map(lambda value: value["owner"])
    result["resource_type"] = policies.map(lambda value: value["resource_type"])

    risk = _numeric(result["combined_risk_score"]).clip(0, 1)
    value = _numeric(result.get("order_value", pd.Series(0.0, index=result.index))).clip(lower=0)
    quantity = _numeric(
        result.get("total_order_qty", pd.Series(0.0, index=result.index))
    ).clip(lower=0)
    penalty_rate = _numeric(
        result.get(
            "penalty_rate",
            pd.Series(assumptions.default_penalty_rate, index=result.index),
        ),
        assumptions.default_penalty_rate,
    ).clip(lower=0)
    tier_weight = _tier_weight(
        result.get("customer_tier", pd.Series("SILVER", index=result.index))
    )

    result["estimated_penalty_exposure"] = (value * penalty_rate * risk).round(2)
    result["quantity_at_risk"] = (quantity * risk * assumptions.quantity_at_risk_fraction).round(
        2
    )
    # Priority balances probability, customer criticality, and normalized financial exposure.
    value_scale = max(float(value.quantile(0.95)), 1.0)
    value_weight = (value / value_scale).clip(upper=1.0)
    result["priority_score"] = (100 * risk * tier_weight * (0.7 + 0.3 * value_weight)).round(2)
    result["decision_status"] = MONITOR

    candidates = result.index[risk >= risk_threshold]
    result.loc[candidates, "decision_status"] = RECOMMENDED
    result["resource_id"] = [
        str(result.at[index, f"{kind}_id"])
        if f"{kind}_id" in result.columns and pd.notna(result.at[index, f"{kind}_id"])
        else "UNASSIGNED"
        for index, kind in zip(result.index, result["resource_type"], strict=True)
    ]

    ranked_candidates = result.loc[candidates].sort_values(
        ["priority_score", "combined_risk_score", "order_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    result["contested_with"] = ""
    for (kind, _resource_id), group in ranked_candidates.groupby(
        ["resource_type", "resource_id"], sort=False
    ):
        overflow = _resource_overflow(
            group, kind, limits, dc_capacity_recovery_fraction
        )
        result.loc[overflow, "decision_status"] = CONTESTED
        if len(overflow):
            all_ids = group["order_id"].tolist()
            for index in overflow:
                competitors = [
                    order_id for order_id in all_ids if order_id != result.at[index, "order_id"]
                ]
                result.at[index, "contested_with"] = ",".join(competitors)

    result["estimated_avoidable_penalty"] = (
        result["estimated_penalty_exposure"] * assumptions.avoided_risk_fraction
    ).round(2)
    result.loc[result["decision_status"] == MONITOR, "estimated_avoidable_penalty"] = 0.0
    return result


#: Maps each rollup name to the column it groups by. This prototype represents
#: ``order_type`` using the modeled order priority (STANDARD/EXPEDITE) rather
#: than inventing an additional taxonomy.
ROLLUP_ENTITY_COLUMNS = {
    "vendor": "vendor_id",
    "dc": "dc_id",
    "lane": "lane_id",
    "customer": "customer_id",
    "order_type": "order_priority",
}


def _dominant_cause(causes: pd.Series) -> str:
    non_null = causes.dropna()
    if non_null.empty:
        return "UNKNOWN"
    return str(non_null.mode(dropna=True).iloc[0])


def _rollup_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            order_count=("order_id", "nunique"),
            actionable_orders=("_is_actionable", "sum"),
            average_risk=("combined_risk_score", "mean"),
            penalty_exposure=("estimated_penalty_exposure", "sum"),
            quantity_at_risk=("quantity_at_risk", "sum"),
            value_at_risk=("_value_at_risk", "sum"),
            dominant_cause=("primary_cause", _dominant_cause),
        )
        .reset_index()
        .sort_values(["penalty_exposure", "average_risk"], ascending=False, kind="stable")
    )
    grouped["pct_at_risk"] = (
        (grouped["actionable_orders"] / grouped["order_count"].clip(lower=1)).round(4)
    )
    grouped["average_risk"] = grouped["average_risk"].round(3)
    grouped["penalty_exposure"] = grouped["penalty_exposure"].round(2)
    grouped["quantity_at_risk"] = grouped["quantity_at_risk"].round(2)
    grouped["value_at_risk"] = grouped["value_at_risk"].round(2)
    ordered_columns = [
        column,
        "order_count",
        "actionable_orders",
        "pct_at_risk",
        "average_risk",
        "penalty_exposure",
        "value_at_risk",
        "quantity_at_risk",
        "dominant_cause",
    ]
    return grouped[ordered_columns]


def build_rollups(
    decisions: pd.DataFrame,
    order_lines: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build vendor, DC, lane, customer, order-type, and SKU hotspot summaries.

    Every rollup reports: order count, actionable orders, percentage of orders
    at risk (``pct_at_risk``), average risk, penalty exposure, ``value_at_risk``
    (order value weighted by risk — a broader revenue-exposure figure than the
    penalty-rate-scaled ``penalty_exposure``), quantity at risk, and the
    dominant (most frequent) primary cause.

    SKU representation: the order-level scored table carries only a single
    ``representative_sku`` per order (the first line's SKU) because an order
    can span multiple SKUs and the order-level risk score is not itself
    SKU-specific — attributing one risk score across several SKUs on the same
    order would overstate precision. The SKU rollup instead uses *exploded*
    order-line logic: every order line is joined to its order's decision, so a
    multi-SKU order contributes to every SKU it actually touches. This is the
    defensible choice documented in the README for this order-level
    prototype.
    """

    rollups: dict[str, pd.DataFrame] = {}
    frame = decisions.assign(
        _is_actionable=decisions["decision_status"].isin([RECOMMENDED, CONTESTED]),
        _value_at_risk=(
            _numeric(decisions.get("order_value", pd.Series(0.0, index=decisions.index)))
            * _numeric(decisions["combined_risk_score"]).clip(0, 1)
        ),
    )
    for entity, column in ROLLUP_ENTITY_COLUMNS.items():
        if column not in decisions.columns:
            rollups[entity] = pd.DataFrame()
            continue
        rollups[entity] = _rollup_by(frame, column)

    if order_lines is not None and {"order_id", "sku_id"} <= set(order_lines.columns):
        order_skus = order_lines[["order_id", "sku_id"]].drop_duplicates()
        exploded = order_skus.merge(
            frame, on="order_id", how="inner", validate="many_to_one"
        )
        rollups["sku"] = _rollup_by(exploded, "sku_id")
    else:
        rollups["sku"] = pd.DataFrame()
    return rollups


def service_impact_summary(
    decisions: pd.DataFrame,
    assumptions: ImpactAssumptions = DEFAULT_IMPACT_ASSUMPTIONS,
) -> dict[str, Any]:
    """Return aggregate impact together with the assumptions that produced it."""

    required = {
        "decision_status",
        "estimated_penalty_exposure",
        "estimated_avoidable_penalty",
        "quantity_at_risk",
    }
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"Missing decision columns: {', '.join(missing)}")
    actionable = decisions["decision_status"].isin([RECOMMENDED, CONTESTED])
    return {
        "orders_reviewed": int(len(decisions)),
        "recommended_orders": int((decisions["decision_status"] == RECOMMENDED).sum()),
        "contested_orders": int((decisions["decision_status"] == CONTESTED).sum()),
        "monitor_orders": int((decisions["decision_status"] == MONITOR).sum()),
        "penalty_exposure": round(
            float(decisions.loc[actionable, "estimated_penalty_exposure"].sum()), 2
        ),
        "estimated_avoidable_penalty": round(
            float(decisions.loc[actionable, "estimated_avoidable_penalty"].sum()), 2
        ),
        "quantity_at_risk": round(float(decisions.loc[actionable, "quantity_at_risk"].sum()), 2),
        "assumptions": {
            "penalty": (
                "Exposure = order value × order penalty rate × combined risk score; "
                f"missing penalty rates default to {assumptions.default_penalty_rate:.1%}."
            ),
            "service_impact": (
                "Quantity at risk = total quantity × combined risk score × "
                f"{assumptions.quantity_at_risk_fraction:.0%}; intervention is assumed to avoid "
                f"{assumptions.avoided_risk_fraction:.0%} of penalty exposure."
            ),
        },
    }


def assumptions_json(assumptions: ImpactAssumptions = DEFAULT_IMPACT_ASSUMPTIONS) -> str:
    """Serialize assumptions for artifact metadata or display."""

    return json.dumps(service_impact_summary(pd.DataFrame({
        "decision_status": [],
        "estimated_penalty_exposure": [],
        "estimated_avoidable_penalty": [],
        "quantity_at_risk": [],
    }), assumptions)["assumptions"], sort_keys=True)
