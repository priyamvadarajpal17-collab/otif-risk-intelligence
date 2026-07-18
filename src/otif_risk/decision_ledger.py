"""Production-shaped decision/action/outcome ledger (Stage 2 governance).

Persists one auditable row per (order, decision-day) candidate the daily
operations replay actually considered -- whether it was executed,
capacity-contested, or simply monitored below the eligibility threshold --
and later reconciles each row with the order's own matured OTIF outcome as
it becomes available.

This ledger is deliberately **not** the same thing as
``policy_evaluation.py``'s exact, common-random-number potential-outcome
policy value (``artifacts/policy_benchmark.json``): that lab answers "how
much value would this policy create if every feasible action were replayed
through the twin's own mechanics" using evaluation-only counterfactuals.
This ledger instead answers a narrower, production-shaped question: "what
did the deployed policy actually decide, and what actually happened next,
for orders it did/did not act on" -- a purely **observational** comparison
(see ``observational_cohort_report``). Because this prototype's daily
operations replay closes every order using its pre-generated outcome
(action does not yet feed back into the replayed lifecycle -- only
``action_response.py``'s evaluation-only twin does that), any accepted-vs-
rejected miss-rate difference here reflects *which orders the policy chose
to act on*, not a measured causal effect of acting. Every report this
module produces is therefore explicitly labeled ``observational_not_causal``
and must never be read as a treatment effect.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

LEDGER_SCHEMA_VERSION = "1.0"

PLANNER_ACCEPTED = "ACCEPTED"
PLANNER_REJECTED = "REJECTED"
PLANNER_MONITORED = "MONITORED"
PLANNER_OVERRIDDEN = "OVERRIDDEN"

EXECUTION_EXECUTED = "EXECUTED"
EXECUTION_NOT_EXECUTED = "NOT_EXECUTED"

#: Decision statuses (see ``operations._daily_candidate_frame``/
#: ``resources.allocate_interventions``) mapped to this ledger's planner
#: disposition and execution status.
_STATUS_TO_PLANNER = {
    "RECOMMENDED": (PLANNER_ACCEPTED, EXECUTION_EXECUTED),
    "CONTESTED": (PLANNER_REJECTED, EXECUTION_NOT_EXECUTED),
    "MONITOR": (PLANNER_MONITORED, EXECUTION_NOT_EXECUTED),
}

#: A cohort needs at least this many decisions before its miss rate/penalty
#: is reported as a number rather than withheld -- guards against a single
#: cohort's rate looking meaningful off of a handful of orders.
MIN_COHORT_SAMPLE = 20

LEDGER_COLUMNS: tuple[str, ...] = (
    "decision_id",
    "decision_key",
    "idempotency_key",
    "order_id",
    "decision_timestamp",
    "source_snapshot_id",
    "model_version",
    "policy_version",
    "manifest_content_id",
    "feasible_actions",
    "chosen_action",
    "rejected_actions",
    "risk_score",
    "threshold",
    "selection_mode",
    "assignment_probability",
    "resource_type",
    "resource_id",
    "resource_demand_units",
    "resource_capacity_before",
    "resource_capacity_after",
    "order_value",
    "penalty_rate",
    "planner_decision",
    "execution_status",
    "execution_timestamp",
    "matured",
    "matured_otif_miss",
    "matured_cause",
    "realized_penalty",
)


def derive_decision_id(decision_key: str) -> str:
    """Deterministic decision ID derived from the idempotency-stable decision key.

    Using a hash of the key (rather than a random UUID minted at write time)
    guarantees a retried write of the same decision always resolves to the
    same ``decision_id`` without any extra bookkeeping.
    """
    return hashlib.sha256(decision_key.encode("utf-8")).hexdigest()[:16]


def ledger_decision_key(
    order_id: str, decision_day: str, source_snapshot_id: str, model_version: str
) -> str:
    """Stable idempotency key for one (order, day, snapshot, model) decision."""
    payload = f"{order_id}|{decision_day}|{source_snapshot_id}|{model_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass
class LedgerEntry:
    decision_id: str
    decision_key: str
    idempotency_key: str
    order_id: str
    decision_timestamp: str
    source_snapshot_id: str
    model_version: str
    policy_version: str
    manifest_content_id: str | None
    feasible_actions: list[str]
    chosen_action: str | None
    rejected_actions: list[str]
    risk_score: float
    threshold: float
    selection_mode: str
    assignment_probability: float | None
    resource_type: str | None
    resource_id: str | None
    resource_demand_units: float
    resource_capacity_before: float | None
    resource_capacity_after: float | None
    #: Captured at decision time from the same scored/business-context frame
    #: the daily replay already builds (``decisions.attach_business_context``)
    #: -- ``outcomes``/``causes`` never carry these fields, so reconciliation
    #: must never try to read them from there (see ``reconcile_outcomes``).
    order_value: float
    penalty_rate: float
    planner_decision: str
    execution_status: str
    execution_timestamp: str | None
    matured: bool = False
    matured_otif_miss: int | None = None
    matured_cause: str | None = None
    realized_penalty: float | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["feasible_actions"] = json.dumps(self.feasible_actions)
        row["rejected_actions"] = json.dumps(self.rejected_actions)
        return row


def build_ledger_entry(
    *,
    order_id: str,
    decision_day: str,
    source_snapshot_id: str,
    model_version: str,
    policy_version: str,
    manifest_content_id: str | None,
    feasible_actions: list[str],
    chosen_action: str | None,
    risk_score: float,
    threshold: float,
    decision_status: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    resource_demand_units: float = 0.0,
    resource_capacity_before: float | None = None,
    resource_capacity_after: float | None = None,
    selection_mode: str | None = None,
    assignment_probability: float | None = None,
    order_value: float = 0.0,
    penalty_rate: float = 0.0,
) -> LedgerEntry:
    """Build one ledger entry from a daily candidate's scoring/allocation result."""
    key = ledger_decision_key(order_id, decision_day, source_snapshot_id, model_version)
    planner_decision, execution_status = _STATUS_TO_PLANNER.get(
        decision_status, (PLANNER_MONITORED, EXECUTION_NOT_EXECUTED)
    )
    rejected = [action for action in feasible_actions if action != chosen_action]
    return LedgerEntry(
        decision_id=derive_decision_id(key),
        decision_key=key,
        idempotency_key=key,
        order_id=order_id,
        decision_timestamp=decision_day,
        source_snapshot_id=source_snapshot_id,
        model_version=model_version,
        policy_version=policy_version,
        manifest_content_id=manifest_content_id,
        feasible_actions=feasible_actions,
        chosen_action=chosen_action if execution_status == EXECUTION_EXECUTED else None,
        rejected_actions=rejected,
        risk_score=float(risk_score),
        threshold=float(threshold),
        selection_mode=selection_mode or decision_status,
        assignment_probability=assignment_probability,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_demand_units=float(resource_demand_units),
        resource_capacity_before=resource_capacity_before,
        resource_capacity_after=resource_capacity_after,
        order_value=float(order_value),
        penalty_rate=float(penalty_rate),
        planner_decision=planner_decision,
        execution_status=execution_status,
        execution_timestamp=decision_day if execution_status == EXECUTION_EXECUTED else None,
    )


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(path)


def append_entries(path: Path, entries: list[LedgerEntry] | list[dict[str, Any]]) -> int:
    """Idempotently upsert ``entries`` into the ledger CSV at ``path``.

    A retried write of a decision that shares an already-persisted
    ``decision_key`` overwrites that row in place (last write wins for
    mutable fields such as reconciliation results) rather than appending a
    duplicate. Returns the number of genuinely new decision keys written.
    """
    rows = [entry.to_row() if isinstance(entry, LedgerEntry) else dict(entry) for entry in entries]
    if not rows:
        return 0
    new_frame = pd.DataFrame(rows)
    existing = _load_ledger(path)
    new_keys = (
        int((~new_frame["decision_key"].isin(existing["decision_key"])).sum())
        if not existing.empty
        else len(new_frame)
    )
    combined = pd.concat([existing, new_frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=["decision_key"], keep="last")
    combined = combined.sort_values(["decision_timestamp", "order_id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return new_keys


def reconcile_outcomes(
    path: Path, outcomes: pd.DataFrame, causes: pd.DataFrame, as_of_timestamp: pd.Timestamp
) -> dict[str, Any]:
    """Update every not-yet-matured ledger row whose order has matured by ``as_of_timestamp``.

    Idempotent: re-running with the same ``as_of_timestamp`` (or an earlier
    one) leaves already-matured rows untouched and never re-derives a
    different realized outcome for the same order.
    """
    ledger = _load_ledger(path)
    if ledger.empty:
        return {"newly_matured": 0, "already_matured": 0, "still_open": 0}

    as_of_timestamp = pd.Timestamp(as_of_timestamp)
    outcomes_indexed = outcomes.set_index("order_id")
    causes_indexed = causes.set_index("order_id")

    matured_mask = ledger["matured"].astype(bool)
    already_matured = int(matured_mask.sum())
    newly_matured = 0

    # Freshly-written CSVs with an all-empty ``matured_cause``/etc. column
    # infer a numeric dtype from the NaNs; widen to ``object`` before
    # assigning strings/ints so pandas never silently upcast-rejects them.
    for column in ("matured_cause", "matured_otif_miss", "realized_penalty"):
        ledger[column] = ledger[column].astype(object)

    for index in ledger.index[~matured_mask]:
        order_id = ledger.at[index, "order_id"]
        if order_id not in outcomes_indexed.index:
            continue
        outcome_row = outcomes_indexed.loc[order_id]
        if pd.Timestamp(outcome_row["outcome_timestamp"]) > as_of_timestamp:
            continue
        ledger.at[index, "matured"] = True
        ledger.at[index, "matured_otif_miss"] = int(outcome_row["otif_miss"])
        if order_id in causes_indexed.index:
            ledger.at[index, "matured_cause"] = str(causes_indexed.loc[order_id, "primary_cause"])
        # ``order_value``/``penalty_rate`` are captured on the ledger row itself
        # at decision time (see ``build_ledger_entry``) -- ``outcomes`` never
        # carries either field, so reading them from there would silently
        # always default to 0.0 rather than the order's real business context.
        penalty_rate = float(ledger.at[index, "penalty_rate"] or 0.0)
        order_value = float(ledger.at[index, "order_value"] or 0.0)
        ledger.at[index, "realized_penalty"] = (
            penalty_rate * order_value * int(outcome_row["otif_miss"])
        )
        newly_matured += 1

    ledger.to_csv(path, index=False)
    return {
        "newly_matured": newly_matured,
        "already_matured": already_matured,
        "still_open": int(len(ledger) - already_matured - newly_matured),
        "reconciled_at_utc": datetime.now(UTC).isoformat(),
    }


def observational_cohort_report(
    path: Path, *, min_sample: int = MIN_COHORT_SAMPLE
) -> dict[str, Any]:
    """Accepted/executed vs rejected vs monitored cohort miss-rate/penalty report.

    Every number here is a plain observational comparison over whichever
    orders the deployed policy actually accepted, capacity-rejected, or
    left monitored -- **never** a causal treatment-effect estimate (no
    randomized assignment/propensity adjustment is applied here; the
    ``policy_evaluation`` module's common-random-number replay is the only
    place this codebase measures actual policy value). A cohort smaller
    than ``min_sample`` has its rate/penalty withheld (``None``) rather than
    reported on too few observations.
    """
    ledger = _load_ledger(path)
    matured = ledger.loc[ledger["matured"].astype(bool)] if not ledger.empty else ledger

    cohorts: dict[str, Any] = {}
    for planner_decision in (PLANNER_ACCEPTED, PLANNER_REJECTED, PLANNER_MONITORED):
        subset = matured.loc[matured["planner_decision"] == planner_decision]
        n = int(len(subset))
        sufficient = n >= min_sample
        cohorts[planner_decision] = {
            "n_matured_decisions": n,
            "sufficient_sample": sufficient,
            "miss_rate": (
                round(float(subset["matured_otif_miss"].mean()), 4) if sufficient else None
            ),
            "total_realized_penalty": (
                round(float(subset["realized_penalty"].sum()), 2) if sufficient else None
            ),
            "mean_realized_penalty": (
                round(float(subset["realized_penalty"].mean()), 2) if sufficient else None
            ),
        }

    return {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "min_sample_guard": min_sample,
        "total_decisions_logged": int(len(ledger)),
        "total_matured_decisions": int(len(matured)),
        "cohorts": cohorts,
        "observational_not_causal": True,
        "qualification": (
            "Observational cohort comparison only -- accepted/rejected/monitored "
            "orders are not randomly assigned and no propensity or confounder "
            "adjustment is applied here. This prototype's daily operations replay "
            "closes every order using its own pre-generated outcome regardless of "
            "the decision made, so any miss-rate difference between cohorts "
            "reflects which orders the policy chose to prioritize, not a measured "
            "causal effect of acting. The only causally-interpretable (exact, "
            "common-random-number potential-outcome) policy value in this codebase "
            "is `policy_evaluation.py` / `artifacts/policy_benchmark.json` -- kept "
            "fully separate from this observational report."
        ),
    }


def intervention_outcomes_report(
    path: Path, *, min_sample: int = MIN_COHORT_SAMPLE
) -> dict[str, Any]:
    """Realized outcome by *intervention type* (``chosen_action``), plus a
    no-intervention baseline -- the per-action complement to
    ``observational_cohort_report``'s per-planner-decision cohorts.

    Every executed decision's ``order_value``/``penalty_rate`` are captured
    on the ledger row at decision time (see ``build_ledger_entry``), so
    ``realized_penalty`` reflects the order's real business context, not a
    fallback zero. Like ``observational_cohort_report``, this is a plain
    **observational** comparison -- action assignment is not randomized here
    -- and is guarded by ``min_sample`` the same way.
    """
    ledger = _load_ledger(path)
    matured = ledger.loc[ledger["matured"].astype(bool)] if not ledger.empty else ledger

    def _stats(subset: pd.DataFrame) -> dict[str, Any]:
        n = int(len(subset))
        sufficient = n >= min_sample
        return {
            "n_matured_decisions": n,
            "sufficient_sample": sufficient,
            "miss_rate": (
                round(float(subset["matured_otif_miss"].mean()), 4) if sufficient else None
            ),
            "total_realized_penalty": (
                round(float(subset["realized_penalty"].sum()), 2) if sufficient else None
            ),
            "mean_realized_penalty": (
                round(float(subset["realized_penalty"].mean()), 2) if sufficient else None
            ),
            "mean_order_value": (
                round(float(subset["order_value"].mean()), 2) if sufficient else None
            ),
        }

    executed = matured.loc[matured["execution_status"] == EXECUTION_EXECUTED]
    by_action: dict[str, Any] = {}
    if not executed.empty:
        for action, subset in executed.groupby("chosen_action"):
            by_action[str(action)] = _stats(subset)

    no_intervention = matured.loc[matured["execution_status"] == EXECUTION_NOT_EXECUTED]

    return {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "min_sample_guard": min_sample,
        "total_decisions_logged": int(len(ledger)),
        "total_matured_decisions": int(len(matured)),
        "outcomes_by_intervention_type": by_action,
        "no_intervention_baseline": _stats(no_intervention),
        "observational_not_causal": True,
        "qualification": (
            "Realized-outcome breakdown by intervention type (chosen_action), "
            "observational only -- interventions are not randomly assigned across "
            "orders, so a difference between action types or versus the "
            "no-intervention baseline reflects which orders/actions the policy "
            "chose to prioritize, not a measured causal effect. This prototype's "
            "daily operations replay closes every order using its own "
            "pre-generated outcome regardless of the decision made (action does "
            "not yet feed back into the replayed lifecycle). The only "
            "causally-interpretable (exact, common-random-number potential-"
            "outcome) policy value in this codebase is `policy_evaluation.py` / "
            "`artifacts/policy_benchmark.json` -- kept fully separate from this "
            "report."
        ),
    }
