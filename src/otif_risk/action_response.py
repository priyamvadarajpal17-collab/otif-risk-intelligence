"""Heterogeneous, probabilistic action-response digital twin (evaluation-only).

For every ``(order, feasible action)`` pair this module draws a **common
random number** keyed by ``(seed, order_id, action_code)`` -- a stable
SHA-256-derived seed, never the row's position/index -- so every policy under
evaluation (random, highest-risk, current policy, oracle, ...) that is
offered the same action on the same order sees the *exact same* potential
outcome. This is what makes the policy comparisons in ``policy_evaluation.py``
fair: differences in measured value come only from *which* orders/actions a
policy chooses, never from re-rolled luck.

Each action targets exactly one lifecycle mechanism already produced by
``data.generate_dataset`` (a per-stage delay in hours, or the order's
quantity shortfall). A successful response reduces only that mechanism and
recomputes the delivered timestamp/quantity -- and, from it, on-time,
in-full, OTIF miss, and a realized penalty -- through the *same* shared
lifecycle/service-outcome helpers the twin itself uses
(``data.recompute_lifecycle_timestamps`` and
``root_causes.compute_service_outcome``). Nothing here is a second, separate
outcome model: it is the twin's own equations, replayed with one input
changed.

Response probability is a fully transparent, documented weighted sum of five
``[0, 1]`` components (weights sum to 1.0, see ``_response_probability``):

- ``match`` (0.40): does the action's mechanism match the order's actual
  primary/secondary cause?
- ``timing`` (0.15): how much slack remains between the prediction timestamp
  (the decision point) and the promised delivery date?
- ``severity`` (0.15): how large is the existing delay/shortfall already
  accumulated in the targeted mechanism (bigger holes are harder to fully
  close)?
- ``flexibility`` (0.15): a stable per-resource trait (vendor reliability,
  SKU scarcity/criticality, DC predictability, lane variability, customer
  reschedule propensity).
- ``availability`` (0.15): a stable, order-date-observable resource-slack
  signal (DC utilization snapshot, vendor contract lead time, ...).

A failed attempt can still be *adverse* (a small, documented chance the
targeted mechanism gets slightly worse -- wasted rush, disrupted kitting,
etc.), which is more likely the worse the action/cause mismatch. Potential
outcomes therefore include genuine no-benefit and harmful cases; not every
action is made to succeed, and ``ORDER_CAPTURE_CORRECTION`` is a documented
structural no-op (see below).

Every value here is evaluation-only: it is used by ``policy_evaluation.py``
to score policies against shared counterfactuals, and must never be joined
into ``features.build_feature_table`` or any model-facing table.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import PrototypeDataset
from .data import recompute_lifecycle_timestamps
from .root_causes import compute_service_outcome

#: Bumped whenever the response-probability formula, magnitude formulas, or
#: the action/cause mapping change in a way that would alter potential
#: outcomes for an already-evaluated (seed, order_id, action_code) triple.
ACTION_RESPONSE_VERSION = "action-response-v1"

NO_ACTION = "NO_ACTION"

#: One action per feasible operational mitigation named in the plan. Aligned
#: 1:1 with ``decisions.RECOMMENDATION_TABLE`` action text/owners, except
#: DC_CAPACITY and WAREHOUSE_OPS share one action because the twin already
#: adds both effects into the same ``warehouse_delay_hours`` mechanism (see
#: ``data.py``), so "reserve DC capacity" and "expedite pick-pack" are the
#: same lever in this simulator.
ACTIONS: tuple[str, ...] = (
    "ORDER_CAPTURE_CORRECTION",
    "VENDOR_ESCALATION",
    "INVENTORY_REALLOCATION",
    "WAREHOUSE_EXPEDITE",
    "ALTERNATE_TRANSPORT",
    "APPOINTMENT_COORDINATION",
)

#: Cause categories (see ``contracts.CAUSE_CATEGORIES``) each action's
#: mechanism plausibly addresses. Used only for the ``match`` probability
#: component -- an action may still be attempted (and rarely help) outside
#: this set, reflecting real operational uncertainty about the true cause.
ACTION_TARGET_CAUSES: dict[str, tuple[str, ...]] = {
    "ORDER_CAPTURE_CORRECTION": ("ORDER_CAPTURE",),
    "VENDOR_ESCALATION": ("VENDOR_FAILURE",),
    "INVENTORY_REALLOCATION": ("INVENTORY_SHORTAGE",),
    "WAREHOUSE_EXPEDITE": ("DC_CAPACITY", "WAREHOUSE_OPS"),
    "ALTERNATE_TRANSPORT": ("TRANSPORT",),
    "APPOINTMENT_COORDINATION": ("CUSTOMER_DELIVERY",),
}

#: Resource pool each action consumes, matching
#: ``decisions.RECOMMENDATION_TABLE``/``resources.RESOURCE_POOL_FOR_TYPE`` so
#: policy evaluation consumes the exact same capacity pools as production.
ACTION_RESOURCE_TYPE: dict[str, str] = {
    "ORDER_CAPTURE_CORRECTION": "customer",
    "VENDOR_ESCALATION": "vendor",
    "INVENTORY_REALLOCATION": "dc",
    "WAREHOUSE_EXPEDITE": "dc",
    "ALTERNATE_TRANSPORT": "lane",
    "APPOINTMENT_COORDINATION": "customer",
}

#: Inverse of ``ACTION_TARGET_CAUSES``: the single action the deployed
#: recommendation table (``decisions.RECOMMENDATION_TABLE``) would assign to
#: an order with this ``primary_cause``. ``UNKNOWN``/``ON_TIME`` have no
#: mapped action (no observable mechanism to intervene on).
CAUSE_TO_ACTION: dict[str, str] = {
    cause: action for action, causes in ACTION_TARGET_CAUSES.items() for cause in causes
}

#: Column on the order-context frame holding the resource's ID for each
#: action's pool.
RESOURCE_ID_COLUMN = {
    "dc": "dc_id",
    "lane": "lane_id",
    "vendor": "vendor_id",
    "customer": "customer_id",
}

#: The simulator_truth delay-hours column each action targets. Missing entry
#: (INVENTORY_REALLOCATION) is handled specially via ``shortfall_fraction``.
_STAGE_DELAY_COLUMN = {
    "ORDER_CAPTURE_CORRECTION": "capture_delay_hours",
    "VENDOR_ESCALATION": "vendor_ready_delay_hours",
    "WAREHOUSE_EXPEDITE": "warehouse_delay_hours",
    "ALTERNATE_TRANSPORT": "transit_delay_hours",
    "APPOINTMENT_COORDINATION": "customer_delay_hours",
}

#: Existing accumulated delay above which further reduction gets structurally
#: harder to fully realize (documented severity-scaling cap, in hours).
SEVERITY_CAP_HOURS = 72.0
#: Hours of prediction-to-promise slack treated as "full" timing headroom
#: (the default prediction horizon is 7 days).
TIMING_NORMALIZATION_HOURS = 168.0
#: Response-probability floor/ceiling and component weights. Weights sum to
#: 1.0 so every unit of probability is attributable to a named, documented
#: driver.
PROBABILITY_FLOOR = 0.02
PROBABILITY_CEILING = 0.97
WEIGHT_BASE = 0.05
WEIGHT_MATCH = 0.40
WEIGHT_TIMING = 0.15
WEIGHT_SEVERITY = 0.15
WEIGHT_FLEXIBILITY = 0.15
WEIGHT_AVAILABILITY = 0.10
#: Match-quality score by whether the action's mechanism is the order's
#: primary cause, a secondary (contributing) cause, or unrelated.
MATCH_PRIMARY = 1.0
MATCH_SECONDARY = 0.4
MATCH_NONE = 0.1
#: Reduction fraction (of the targeted mechanism) on a successful response is
#: `REDUCTION_BASE + REDUCTION_SPAN * flexibility_score`, i.e. 0.50-0.90.
REDUCTION_BASE = 0.50
REDUCTION_SPAN = 0.40
#: A failed, mismatched attempt has a higher chance of a small adverse bump.
ADVERSE_PROB_BASE = 0.05
ADVERSE_PROB_MISMATCH_SPAN = 0.20
ADVERSE_BUMP_MIN = 0.02
ADVERSE_BUMP_MAX = 0.15

#: Realized-penalty severity caps (documented, transparent business
#: assumptions distinct from ``decisions.ImpactAssumptions``, which is a
#: risk-*weighted* expected exposure, not a realized-outcome penalty).
LATE_PENALTY_FULL_HOURS = 48.0


def deterministic_uniforms(
    seed: int, order_id: str, action_code: str, count: int = 3
) -> np.ndarray:
    """Return ``count`` uniform draws, a pure function of ``(seed, order_id, action_code)``.

    Uses SHA-256 over the exact key string to build the RNG seed, so the draw
    is identical regardless of row iteration order, DataFrame index, process,
    or platform -- the common-random-number contract every policy relies on.
    """
    key = f"{seed}|{order_id}|{action_code}".encode()
    digest = hashlib.sha256(key).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed_int)
    return rng.random(count)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(max(value, low), high))


def build_order_context(
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble one row per order with every observable/latent input the
    response-probability and magnitude formulas need.

    Deliberately draws on ``dataset.simulator_truth`` (evaluation-only ground
    truth) for per-stage delay hours: this context frame is consumed only by
    ``simulate_action_response``/``policy_evaluation`` and must never be
    joined into a model feature table.
    """
    orders = dataset.orders[
        [
            "order_id",
            "order_date",
            "promised_delivery_date",
            "prediction_timestamp",
            "vendor_id",
            "dc_id",
            "lane_id",
            "customer_id",
            "capture_delay_hours",
        ]
    ].copy()
    truth = dataset.simulator_truth[
        [
            "order_id",
            "vendor_ready_delay_hours",
            "warehouse_delay_hours",
            "transit_delay_hours",
            "customer_delay_hours",
            "unknown_extra_hours",
        ]
    ]
    context = orders.merge(truth, on="order_id", validate="one_to_one")
    context = context.merge(
        outcomes[
            [
                "order_id",
                "requested_qty",
                "delivered_qty",
                "delivered_timestamp",
                "on_time",
                "in_full",
                "otif_miss",
            ]
        ],
        on="order_id",
        validate="one_to_one",
    )
    context = context.merge(
        causes[["order_id", "primary_cause", "secondary_causes"]],
        on="order_id",
        validate="one_to_one",
    )

    vendors = dataset.vendors.set_index("vendor_id")
    context["vendor_reliability"] = context["vendor_id"].map(vendors["reliability_score"])
    context["vendor_contract_lead_days"] = context["vendor_id"].map(vendors["contract_lead_days"])

    dcs = dataset.dcs.set_index("dc_id")
    context["dc_capacity_variability"] = context["dc_id"].map(dcs["capacity_variability"])
    snapshot = dataset.capacity_snapshots.set_index(["dc_id", "snapshot_date"])["utilization"]
    keys = list(zip(context["dc_id"], context["order_date"].dt.normalize(), strict=True))
    context["dc_utilization"] = [float(snapshot.get(key, 0.5)) for key in keys]

    lanes = dataset.lanes.set_index("lane_id")
    context["lane_planned_transit_days"] = context["lane_id"].map(lanes["planned_transit_days"])
    context["lane_transit_variability_days"] = context["lane_id"].map(
        lanes["transit_variability_days"]
    )

    customers = dataset.customers.set_index("customer_id")
    context["customer_appointment_required"] = (
        context["customer_id"].map(customers["appointment_required"]).astype(bool)
    )
    context["customer_reschedule_trait"] = context["customer_id"].map(
        customers["reschedule_trait"]
    )

    lines = dataset.order_lines.merge(
        dataset.skus[["sku_id", "criticality_tier", "scarcity_trait"]], on="sku_id", how="left"
    )
    qty = lines["requested_qty"].astype(float).clip(lower=1e-6)
    weighted_scarcity = (lines["scarcity_trait"].astype(float) * qty).groupby(
        lines["order_id"]
    ).sum() / qty.groupby(lines["order_id"]).sum()
    has_critical = (lines["criticality_tier"] == "CRITICAL").groupby(lines["order_id"]).any()
    sku_agg = pd.DataFrame(
        {
            "sku_scarcity_trait": weighted_scarcity,
            "sku_has_critical": has_critical,
        }
    ).reset_index().rename(columns={"index": "order_id"})
    context = context.merge(sku_agg, on="order_id", how="left")
    context["sku_scarcity_trait"] = context["sku_scarcity_trait"].fillna(
        float(dataset.skus["scarcity_trait"].mean())
    )
    context["sku_has_critical"] = context["sku_has_critical"].fillna(False)

    slack_hours = (
        context["promised_delivery_date"] - context["prediction_timestamp"]
    ).dt.total_seconds() / 3600.0
    context["slack_hours"] = slack_hours.clip(lower=0.0)
    context["shortfall_fraction"] = (
        1.0
        - context["delivered_qty"].astype(float)
        / context["requested_qty"].astype(float).clip(lower=1e-6)
    ).clip(lower=0.0, upper=1.0)
    return context


@dataclass(frozen=True)
class _ResponseComponents:
    match: float
    timing: float
    severity: float
    flexibility: float
    availability: float

    @property
    def probability(self) -> float:
        raw = (
            WEIGHT_BASE
            + WEIGHT_MATCH * self.match
            + WEIGHT_TIMING * self.timing
            + WEIGHT_SEVERITY * self.severity
            + WEIGHT_FLEXIBILITY * self.flexibility
            + WEIGHT_AVAILABILITY * self.availability
        )
        return _clip(raw, PROBABILITY_FLOOR, PROBABILITY_CEILING)


def _match_score(action: str, primary_cause: str, secondary_causes: list[str]) -> float:
    targets = ACTION_TARGET_CAUSES[action]
    if primary_cause in targets:
        return MATCH_PRIMARY
    if any(cause in targets for cause in secondary_causes):
        return MATCH_SECONDARY
    return MATCH_NONE


def _timing_score(row: pd.Series) -> float:
    return _clip(float(row["slack_hours"]) / TIMING_NORMALIZATION_HOURS)


def _severity_score(action: str, row: pd.Series) -> float:
    if action == "INVENTORY_REALLOCATION":
        value = float(row["shortfall_fraction"])
        return _clip(1.0 - value)
    column = _STAGE_DELAY_COLUMN[action]
    value = float(row[column])
    return _clip(1.0 - value / SEVERITY_CAP_HOURS)


def _flexibility_score(action: str, row: pd.Series) -> float:
    if action == "VENDOR_ESCALATION":
        return _clip(float(row["vendor_reliability"]))
    if action == "INVENTORY_REALLOCATION":
        base = _clip(1.0 - 3.0 * float(row["sku_scarcity_trait"]))
        bonus = 1.15 if bool(row["sku_has_critical"]) else 1.0
        return _clip(base * bonus)
    if action == "WAREHOUSE_EXPEDITE":
        return _clip(1.0 - float(row["dc_capacity_variability"]) / 0.22)
    if action == "ALTERNATE_TRANSPORT":
        return _clip(1.0 - float(row["lane_transit_variability_days"]) / 1.4)
    if action == "APPOINTMENT_COORDINATION":
        return _clip(1.0 - 2.0 * float(row["customer_reschedule_trait"]))
    return 0.8  # ORDER_CAPTURE_CORRECTION: no differentiating stable trait available.


def _availability_score(action: str, row: pd.Series) -> float:
    if action == "VENDOR_ESCALATION":
        return _clip(1.0 - (float(row["vendor_contract_lead_days"]) - 1.0) / 4.0)
    if action in ("INVENTORY_REALLOCATION", "WAREHOUSE_EXPEDITE"):
        return _clip(1.0 - float(row["dc_utilization"]))
    if action == "ALTERNATE_TRANSPORT":
        # No per-lane daily-capacity signal exists beyond the flat pool
        # already assumed in resources.py; use a documented neutral value.
        return 0.6
    if action == "APPOINTMENT_COORDINATION":
        return 0.5 if bool(row["customer_appointment_required"]) else 0.9
    return 0.8  # ORDER_CAPTURE_CORRECTION: internal desk fix, no external pool.


def response_components(action: str, row: pd.Series) -> _ResponseComponents:
    secondary_causes = row.get("_secondary_causes_parsed", [])
    return _ResponseComponents(
        match=_match_score(action, row["primary_cause"], secondary_causes),
        timing=_timing_score(row),
        severity=_severity_score(action, row),
        flexibility=_flexibility_score(action, row),
        availability=_availability_score(action, row),
    )


def _no_action_row(row: pd.Series, seed: int) -> dict:
    """The no-action potential outcome: identical to the order's real outcome."""
    return {
        "seed": seed,
        "order_id": row["order_id"],
        "action_code": NO_ACTION,
        "eligible": True,
        "resource_type": None,
        "resource_id": None,
        "response_probability": 1.0,
        "success": True,
        "adverse": False,
        "requested_qty": float(row["requested_qty"]),
        "promised_delivery_date": row["promised_delivery_date"],
        "delivered_timestamp": row["delivered_timestamp"],
        "delivered_qty": float(row["delivered_qty"]),
        "on_time": int(row["on_time"]),
        "in_full": int(row["in_full"]),
        "otif_miss": int(row["otif_miss"]),
        "match_score": 0.0,
        "timing_score": 0.0,
        "severity_score": 0.0,
        "flexibility_score": 0.0,
        "availability_score": 0.0,
        "mechanism_note": "no_action: identity outcome, never perturbed.",
    }


def _action_row(row: pd.Series, action: str, seed: int) -> dict:
    resource_type = ACTION_RESOURCE_TYPE[action]
    resource_id_column = RESOURCE_ID_COLUMN[resource_type]
    resource_id = row.get(resource_id_column)
    eligible = pd.notna(resource_id)

    components = response_components(action, row)
    probability = components.probability

    draws = deterministic_uniforms(seed, row["order_id"], action)
    success = bool(draws[0] < probability)
    adverse = False

    vendor_ready = float(row["vendor_ready_delay_hours"])
    warehouse = float(row["warehouse_delay_hours"])
    transit = float(row["transit_delay_hours"])
    customer_delay = float(row["customer_delay_hours"])
    unknown = float(row["unknown_extra_hours"])
    shortfall_fraction = float(row["shortfall_fraction"])
    requested_qty = float(row["requested_qty"])

    if success:
        reduction_fraction = _clip(
            REDUCTION_BASE + REDUCTION_SPAN * components.flexibility + (draws[1] - 0.5) * 0.1,
            0.05,
            0.95,
        )
        if action == "VENDOR_ESCALATION":
            vendor_ready *= 1.0 - reduction_fraction
        elif action == "WAREHOUSE_EXPEDITE":
            warehouse *= 1.0 - reduction_fraction
        elif action == "ALTERNATE_TRANSPORT":
            transit *= 1.0 - reduction_fraction
        elif action == "APPOINTMENT_COORDINATION":
            customer_delay *= 1.0 - reduction_fraction
        elif action == "INVENTORY_REALLOCATION":
            shortfall_fraction *= 1.0 - reduction_fraction
        # ORDER_CAPTURE_CORRECTION: capture_delay_hours is not a lifecycle
        # input (see recompute_lifecycle_timestamps docstring) -- success has
        # no mechanical effect on delivered timestamp/quantity by design.
    else:
        adverse_probability = _clip(
            ADVERSE_PROB_BASE + ADVERSE_PROB_MISMATCH_SPAN * (1.0 - components.match)
        )
        adverse = bool(draws[2] < adverse_probability)
        if adverse:
            bump = ADVERSE_BUMP_MIN + (ADVERSE_BUMP_MAX - ADVERSE_BUMP_MIN) * float(draws[1])
            if action == "VENDOR_ESCALATION":
                vendor_ready *= 1.0 + bump
            elif action == "WAREHOUSE_EXPEDITE":
                warehouse *= 1.0 + bump
            elif action == "ALTERNATE_TRANSPORT":
                transit *= 1.0 + bump
            elif action == "APPOINTMENT_COORDINATION":
                customer_delay *= 1.0 + bump
            elif action == "INVENTORY_REALLOCATION":
                shortfall_fraction = _clip(shortfall_fraction * (1.0 + bump))
            # ORDER_CAPTURE_CORRECTION: no downstream mechanism to worsen.

    lifecycle = recompute_lifecycle_timestamps(
        pd.Series([row["order_date"]]),
        [row["vendor_contract_lead_days"]],
        [vendor_ready],
        [warehouse],
        [row["lane_planned_transit_days"]],
        [transit],
        [customer_delay],
        [unknown],
    )
    delivered_timestamp = lifecycle["delivered_timestamp"].iloc[0]
    delivered_qty = requested_qty * (1.0 - shortfall_fraction)
    delivered_qty = min(max(delivered_qty, 0.0), requested_qty)

    outcome = compute_service_outcome(
        pd.Series([delivered_timestamp]),
        pd.Series([delivered_qty]),
        pd.Series([requested_qty]),
        pd.Series([row["promised_delivery_date"]]),
    )
    return {
        "seed": seed,
        "order_id": row["order_id"],
        "action_code": action,
        "eligible": bool(eligible),
        "resource_type": resource_type,
        "resource_id": str(resource_id) if eligible else None,
        "response_probability": round(probability, 6),
        "success": success,
        "adverse": adverse,
        "requested_qty": float(requested_qty),
        "promised_delivery_date": row["promised_delivery_date"],
        "delivered_timestamp": delivered_timestamp,
        "delivered_qty": float(delivered_qty),
        "on_time": int(outcome["on_time"].iloc[0]),
        "in_full": int(outcome["in_full"].iloc[0]),
        "otif_miss": int(outcome["otif_miss"].iloc[0]),
        "match_score": components.match,
        "timing_score": components.timing,
        "severity_score": components.severity,
        "flexibility_score": components.flexibility,
        "availability_score": components.availability,
        "mechanism_note": (
            f"match={components.match:.2f} timing={components.timing:.2f} "
            f"severity={components.severity:.2f} flexibility={components.flexibility:.2f} "
            f"availability={components.availability:.2f} success={success} adverse={adverse}"
        ),
    }


def realized_penalty(
    otif_miss: pd.Series,
    on_time: pd.Series,
    delivered_timestamp: pd.Series,
    promised_delivery_date: pd.Series,
    delivered_qty: pd.Series,
    requested_qty: pd.Series,
    order_value: pd.Series,
    penalty_rate: pd.Series,
) -> pd.Series:
    """Evaluation-only realized penalty for one already-resolved simulated outcome.

    Distinct from ``decisions.ImpactAssumptions``' ``estimated_penalty_exposure``
    (a *risk-weighted expected* exposure computed before the outcome is known):
    this is charged only when ``otif_miss`` is actually 1, scaled by how late
    (capped at ``LATE_PENALTY_FULL_HOURS``) or how short the delivery actually
    was -- a transparent, documented severity scaling, not a uniform
    assumption.
    """
    hours_late = (
        (delivered_timestamp - promised_delivery_date).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)
    late_fraction = (hours_late / LATE_PENALTY_FULL_HOURS).clip(upper=1.0)
    late_fraction = late_fraction.where(on_time == 0, 0.0)
    shortfall_fraction = (
        1.0 - delivered_qty.astype(float) / requested_qty.astype(float).clip(lower=1e-6)
    ).clip(lower=0.0, upper=1.0)
    severity = pd.concat([late_fraction, shortfall_fraction], axis=1).max(axis=1)
    penalty = order_value.astype(float) * penalty_rate.astype(float) * severity
    return penalty.where(otif_miss == 1, 0.0).round(4)


def simulate_action_response(
    dataset: PrototypeDataset,
    outcomes: pd.DataFrame,
    causes: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Return one row per ``(order_id, action_code)`` potential outcome.

    Includes ``NO_ACTION`` (identity) plus every action in ``ACTIONS`` for
    every order -- eligibility/probability, not a hard feasibility gate,
    determines whether an action plausibly helps. Rows are independent of
    iteration order (see :func:`deterministic_uniforms`), so this may be run
    once per seed and reused by every policy in ``policy_evaluation.py``.

    Attaches ``realized_penalty`` (evaluation-only; see :func:`realized_penalty`)
    using the same ``order_value``/``penalty_rate`` business context
    ``pipeline.score_orders`` attaches to the deployed decision frame
    (``decisions.attach_business_context``), so avoided-penalty comparisons
    use one consistent financial-exposure definition everywhere.
    """
    import json

    from .decisions import attach_business_context

    context = build_order_context(dataset, outcomes, causes)
    context["_secondary_causes_parsed"] = context["secondary_causes"].map(
        lambda value: json.loads(value) if isinstance(value, str) and value else []
    )

    rows: list[dict] = []
    for _, row in context.iterrows():
        rows.append(_no_action_row(row, seed))
        for action in ACTIONS:
            rows.append(_action_row(row, action, seed))
    responses = pd.DataFrame(rows)
    responses["seed"] = seed

    business = attach_business_context(dataset.orders[["order_id", "customer_id"]], dataset)
    responses = responses.merge(
        business[["order_id", "order_value", "penalty_rate", "customer_tier"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    responses["realized_penalty"] = realized_penalty(
        responses["otif_miss"],
        responses["on_time"],
        responses["delivered_timestamp"],
        responses["promised_delivery_date"],
        responses["delivered_qty"],
        responses["requested_qty"],
        responses["order_value"],
        responses["penalty_rate"],
    )
    return with_avoided_penalty(responses)


def with_avoided_penalty(responses: pd.DataFrame) -> pd.DataFrame:
    """Attach ``no_action_penalty``/``avoided_penalty`` to a responses frame.

    ``avoided_penalty`` is ``no_action_penalty - realized_penalty`` for each
    ``(order_id, action_code)`` row: positive means the action reduced
    realized penalty versus doing nothing, negative means it made the
    simulated outcome worse (an adverse response), zero means no benefit.
    Idempotent: re-running on an already-annotated frame recomputes the same
    columns.
    """
    result = responses.drop(columns=["no_action_penalty", "avoided_penalty"], errors="ignore")
    no_action_penalty = result.loc[
        result["action_code"] == NO_ACTION, ["order_id", "realized_penalty"]
    ].rename(columns={"realized_penalty": "no_action_penalty"})
    result = result.merge(no_action_penalty, on="order_id", how="left", validate="many_to_one")
    result["avoided_penalty"] = (result["no_action_penalty"] - result["realized_penalty"]).round(4)
    return result
