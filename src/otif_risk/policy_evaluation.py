"""Capacity-constrained decision-policy evaluation (Stage 1: prove decision value).

Compares eight policies against the *same* out-of-sample order population (scored via
rolling-origin, chronological cross-fitting -- see ``build_seed_context`` -- so no
evaluated order's own row was ever used to train or select the model that scored it),
the *same* common-random-number potential outcomes from ``action_response.py``, and, at
every one of three pre-specified capacity-stress scenarios (``resources.CAPACITY_SCENARIOS``:
25%/50%/100% of default capacity, applied uniformly to every resource pool for every
policy -- see ``evaluate_policies_across_capacity_scenarios``), the *same* daily resource
capacities (``resources.build_capacity_schedule`` / ``resources.allocate_under_capacity``
-- the exact engine ``operations.py``'s daily replay already uses in production, not a
new optimizer):

1. ``NO_ACTION`` -- nobody is treated; the baseline every avoided-penalty
   figure is measured against.
2. ``RANDOM_AT_CAPACITY`` -- eligible orders accepted in a seeded-random
   order, at the same capacity as every other policy.
3. ``HIGHEST_RISK_AT_CAPACITY`` -- ranked by ``combined_risk_score`` alone.
4. ``HIGHEST_FINANCIAL_AT_CAPACITY`` -- ranked by ``estimated_penalty_exposure``.
5. ``STRONGEST_SIGNAL_HEURISTIC`` -- ranked by upstream cause priority and
   corroborating leading-signal count, independent of the fused ML score.
6. ``SINGLE_CAUSE_PRIORITY_BASELINE`` -- the transparent ``decisions.recommend_orders``
   ranking (risk x tier x value), with the single ``action_response.CAUSE_TO_ACTION``
   action implied by ``primary_cause``. Retained as a deployable baseline so
   any improvement from ``CURRENT_POLICY`` below is directly attributable, not just an
   artifact of a different eligible pool or evaluation.
7. ``CURRENT_POLICY`` -- the **value-aware** deployable policy (see "Value-aware
   CURRENT_POLICY formula" below): for every eligible order it considers every
   point-in-time-feasible action candidate (not just the one ``primary_cause``
   implies), ranks order-action pairs by an explainable **expected avoided penalty per
   normalized resource capacity consumed**, and resources the highest-density
   candidates first, with the same 10% seeded-exploration capacity carve-out (see
   ``EXPLORATION_FRACTION``) and a full per-decision log as before.
8. ``ORACLE_EVALUATION_ONLY`` -- an evaluation-only ceiling that may choose
   *any* action (not just the one ``primary_cause`` implies) and ranks by the
   simulator's own realized avoided penalty. It is never a deployable
   recommendation; it exists only to compute regret for the other seven.

Every policy except ``NO_ACTION``/``ORACLE_EVALUATION_ONLY`` uses the *same*
eligible pool (``combined_risk_score >= fused_threshold`` **and** ``primary_cause``
mapped by ``action_response.CAUSE_TO_ACTION``) -- an apples-to-apples
prioritization/action-choice comparison, never a comparison of different eligibility
rules.

Value-aware ``CURRENT_POLICY`` formula
---------------------------------------
Every input is observable/model-derived at decision time -- never the retrospective
``root_causes`` rule evaluation, the simulator's own response draw, or any potential-
outcome/oracle field (see ``test_policy_evaluation.py``'s leakage-invariance tests).

**Candidate actions** (``_value_aware_candidate_actions``): for each eligible order, one
candidate per *active* ``leading_signal_{cause}`` that maps to a feasible action
(``action_response.CAUSE_TO_ACTION``). If a persisted Bayesian structural
``do(node=0)`` intervention scenario (``intervention_scenarios_json``, produced by
``bayesian.BayesianBundle.score`` from evidence observed strictly at/before decision
time) exists for that cause, its ``relative_risk_reduction`` is used as the candidate's
*structural reduction* term; otherwise the candidate falls back to a fixed, documented
``avoided_risk_fraction`` (``decisions.ImpactAssumptions``, the same 60% constant the
deployed UI already uses) scaled by whether the cause is the order's primary
(confidence 1.0) or a corroborating active secondary signal (confidence 0.4). If, in the
rare case no active signal maps to a feasible action, the eligible order's own
``primary_cause`` mapping is used as a last-resort single candidate (documented
``primary_cause_fallback`` source) -- this mirrors the single-cause baseline's mapping
exactly, so the value-aware policy is never worse off than "no candidate" for an
eligible order.

**Expected value density** (``_expected_value_density``): for candidate
``(order, action)``,

    expected_benefit = estimated_penalty_exposure x structural_reduction x execution_feasibility
    value_density     = expected_benefit / normalized_resource_fraction

- ``estimated_penalty_exposure`` (dollars) is the existing fused-risk-weighted exposure
  (``decisions.recommend_orders``): order value x penalty rate x combined (fused) risk.
  It already encodes *how much risk-weighted value is at stake*.
- ``structural_reduction`` (the term above) is *what share of that risk this specific
  action removes* -- a bounded ``[0, 1]`` fraction, never re-deriving or double-counting
  the risk term itself.
- ``execution_feasibility`` (``_execution_feasibility``, bounded ``[0, 1]``) is a fixed
  weighted sum of three observable point-in-time proxies: remaining promise slack
  (0.4, ``remaining_slack_hours`` normalized over a 7-day horizon), an action-specific
  resource trait (0.4: vendor reliability for ``VENDOR_ESCALATION``, DC utilization
  headroom for ``WAREHOUSE_EXPEDITE``, SKU scarcity for ``INVENTORY_REALLOCATION``, lane
  transit variability for ``ALTERNATE_TRANSPORT``, customer appointment context for
  ``APPOINTMENT_COORDINATION``), and Bayesian evidence coverage (0.2, how much of the
  10-node network's evidence is actually observed for this order -- a causal-confidence
  proxy, not the model's own risk score, avoiding double counting the risk term again).
- ``normalized_resource_fraction`` divides the action's ``resources.demand_units_for``
  demand by that resource pool's *scenario-independent default* daily capacity
  (``resources.default_daily_capacities``) -- so the ranking used to prioritize
  candidates is identical across the 25%/50%/100% capacity-stress scenarios; only the
  *acceptance* cutoff (via ``resources.build_capacity_schedule``) varies by scenario.

The deployed action for each eligible order is the candidate with the highest positive
``value_density``; the same score is the priority used to resource-rank across orders
(``_allocate_day_with_exploration``, unchanged), with the same ~10% seeded-exploration
capacity carve-out as before. Every term, weight, and fallback above is fixed before any
benchmark run and is never retuned after seeing results.

At this twin's *default* (100%) capacity, resource pools are rarely binding, so a
priority ranking has little to be discriminative about (see
``resources``'s module docstring). ``PRIMARY_CAPACITY_SCENARIO`` (50% of default
capacity) is therefore Stage 1's headline scenario, not the unscaled baseline; 25%/100%
are still fully evaluated and reported (``evaluate_policies_across_capacity_scenarios``,
``summarize_multi_seed``) as sensitivity/diagnostic context, including on any
seed/scenario where ``CURRENT_POLICY`` does not win.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .action_response import (
    ACTION_RESOURCE_TYPE,
    ACTIONS,
    CAUSE_TO_ACTION,
    NO_ACTION,
    RESOURCE_ID_COLUMN,
    deterministic_uniforms,
    simulate_action_response,
)
from .contracts import CAUSE_CATEGORIES, PrototypeConfig, PrototypeDataset
from .data import DRIFT_WINDOW_FRACTION, generate_dataset
from .decisions import (
    DEFAULT_IMPACT_ASSUMPTIONS,
    attach_business_context,
    primary_cause_from_signals,
    recommend_orders,
)
from .features import TemporalSplit, attach_line_evidence_features, build_feature_table
from .fusion import fuse_scores
from .pipeline import TrainedBundle, train_full_bundle
from .resources import (
    CAPACITY_SCENARIOS,
    ResourceCapacities,
    allocate_under_capacity,
    build_capacity_schedule,
    default_daily_capacities,
    demand_units_for,
)
from .root_causes import calculate_outcomes, derive_root_causes

#: Bumped whenever policy definitions, priority formulas, capacity mapping,
#: or exploration mechanics change in a way that would alter a past
#: evaluation's numbers for identical (seed, config) inputs.
POLICY_EVALUATION_VERSION = "policy-eval-v4"

POLICY_NO_ACTION = "NO_ACTION"
POLICY_RANDOM = "RANDOM_AT_CAPACITY"
POLICY_HIGHEST_RISK = "HIGHEST_RISK_AT_CAPACITY"
POLICY_HIGHEST_FINANCIAL = "HIGHEST_FINANCIAL_AT_CAPACITY"
POLICY_STRONGEST_SIGNAL = "STRONGEST_SIGNAL_HEURISTIC"
#: The transparent ``decisions.recommend_orders`` risk x tier x value ranking,
#: with one ``CAUSE_TO_ACTION``-implied action, retained as a deployable baseline.
POLICY_LEGACY = "SINGLE_CAUSE_PRIORITY_BASELINE"
#: The value-aware deployable policy under evaluation -- see the module docstring's
#: "Value-aware CURRENT_POLICY formula" section for the exact formula.
POLICY_CURRENT = "CURRENT_POLICY"
POLICY_ORACLE = "ORACLE_EVALUATION_ONLY"

#: Evaluation order matters only for report readability.
POLICIES: tuple[str, ...] = (
    POLICY_NO_ACTION,
    POLICY_RANDOM,
    POLICY_HIGHEST_RISK,
    POLICY_HIGHEST_FINANCIAL,
    POLICY_STRONGEST_SIGNAL,
    POLICY_LEGACY,
    POLICY_CURRENT,
    POLICY_ORACLE,
)

#: Value-aware ``CURRENT_POLICY`` formula constants -- every weight/term here is fixed
#: and documented before any benchmark run and never retuned after seeing gate results.
#: Execution-feasibility component weights (sum to 1.0): remaining promise slack, an
#: action-specific resource trait, and Bayesian evidence coverage (causal confidence).
FEASIBILITY_WEIGHT_SLACK = 0.4
FEASIBILITY_WEIGHT_RESOURCE_TRAIT = 0.4
FEASIBILITY_WEIGHT_CAUSAL_CONFIDENCE = 0.2
#: Hours of prediction-to-promise slack treated as "full" timing headroom (matches the
#: default 7-day prediction horizon; independently defined here rather than imported
#: from ``action_response`` so the policy's own formula stays self-contained and is
#: never coupled to changes in the simulator's response-probability tuning).
FEASIBILITY_SLACK_NORMALIZATION_HOURS = 168.0
#: Documented normalization ceiling for the lane transit-variability master-data trait
#: (``dataset.lanes.transit_variability_days``, generated in the 0.3-1.4 day range).
FEASIBILITY_LANE_VARIABILITY_NORMALIZATION_DAYS = 1.4
#: Documented scaling factor turning the SKU scarcity master-data trait (small, beta
#: distributed) into a meaningful [0, 1] feasibility penalty.
FEASIBILITY_SKU_SCARCITY_SCALE = 3.0
#: Structural-reduction fallback when no persisted Bayesian scenario exists for a
#: candidate's cause: reuses the existing, documented 60% deployed-UI effectiveness
#: assumption (``decisions.ImpactAssumptions.avoided_risk_fraction``), scaled by a
#: cause-match confidence (see below) -- never a newly invented number.
FALLBACK_STRUCTURAL_REDUCTION = DEFAULT_IMPACT_ASSUMPTIONS.avoided_risk_fraction
#: Cause-match confidence for the fallback structural-reduction term: full confidence
#: when the candidate's cause is the order's own primary cause, partial confidence when
#: it is a corroborating active (but non-primary) leading signal.
CAUSE_MATCH_PRIMARY_CONFIDENCE = 1.0
CAUSE_MATCH_SECONDARY_CONFIDENCE = 0.4
#: Divide-by-zero guard only (a normalized resource fraction of exactly 0.0 would
#: otherwise make value density undefined) -- not a tuning parameter.
VALUE_DENSITY_MIN_RESOURCE_FRACTION = 1e-6

#: Stage 1's headline capacity-stress scenario (see
#: ``resources.CAPACITY_SCENARIOS``): the business question this lab
#: answers is "does the deployed ranking create more value than simpler
#: baselines under real, scarce capacity", not "...under this twin's
#: generously-sized default capacity" -- so 50% of default capacity, not
#: 100%, is the primary number every acceptance gate is measured against.
#: 25%/100% are still evaluated and reported in full (see
#: ``evaluate_policies_across_capacity_scenarios``) as sensitivity/
#: diagnostic context -- never hidden, including on scenarios/seeds where
#: ``CURRENT_POLICY`` loses.
PRIMARY_CAPACITY_SCENARIO = "SCARCE_50_PERCENT"
#: Retained for continuity with pre-sensitivity-analysis reports: the
#: unscaled 100%-capacity scenario is still computed and reported every
#: run, but only as a diagnostic, never as the acceptance-gate headline.
DIAGNOSTIC_CAPACITY_SCENARIO = "BASE_100_PERCENT"
#: Two measured headline numbers within this tolerance of each other are
#: treated as a tie, not a win/loss, when computing paired per-seed
#: win/tie/loss counts -- deliberately small (this is a rounding-noise
#: guard, not a manufactured significance threshold) since medians are
#: already rounded to 4 decimal places by ``_summarize``.
CAPACITY_SCENARIO_TIE_TOLERANCE = 1e-6

#: Target fraction of every resource pool's daily capacity reserved for
#: seeded exploration among near-equal eligible orders under
#: ``POLICY_CURRENT`` (see the plan's "10% of intervention capacity"
#: requirement). Two distinct, capacity-preserving designs implement this
#: target depending on the pool's demand granularity (see
#: ``resources.demand_units_for``) -- a single fractional
#: ``exploit_capacity = capacity * (1 - EXPLORATION_FRACTION)`` threshold
#: cannot be shared between them without silently dropping capacity:
#:
#: - **Discrete pools** (``vendor``/``customer``, always exactly 1 demand
#:   unit per order): a 1-slot pool's fractional "explore capacity" would be
#:   0.1 units -- too small to ever fit a 1-unit order, and 0.9 units is
#:   also too small, so *neither* the exploit nor the explore slice could
#:   ever accept anyone, silently discarding that slot's capacity every
#:   single day. ``_discrete_explore_counts``/``_discrete_propensities``
#:   instead split *whole-slot counts* via seeded stochastic rounding, so
#:   the pool's full ``min(capacity, n_candidates)`` slots are always
#:   filled, ~``EXPLORATION_FRACTION`` of the time via the explore branch
#:   (exact in expectation, deterministic given the seed). This still
#:   applies correctly when the pool's *daily* capacity itself is already
#:   whole-slot-scheduled to 0 or 1 by ``resources.build_capacity_schedule``
#:   under a scarce capacity scenario -- exploration always operates on
#:   whatever that day's realized capacity is, never the unscaled base.
#: - **Continuous pools** (``dc``/``lane``, quantity-denominated demand):
#:   the explore stage's fill budget is the pool's *actual* remaining
#:   capacity after exploit acceptance, not a fixed
#:   ``capacity * EXPLORATION_FRACTION`` slice, so a large order that
#:   overshoots the nominal 10% slice but still fits in the true leftover
#:   capacity is never dropped.
EXPLORATION_FRACTION = 0.10

#: Resource types whose demand is always exactly 1 unit per order (see
#: ``resources.demand_units_for``) -- headcount-style slot pools where whole
#: units must be preserved exactly.
DISCRETE_RESOURCE_TYPES = frozenset({"vendor", "customer"})
#: Resource types whose demand is a real-valued quantity (see
#: ``resources.demand_units_for``).
CONTINUOUS_RESOURCE_TYPES = frozenset({"dc", "lane"})

SELECTION_EXPLOIT = "EXPLOIT"
SELECTION_EXPLORE = "EXPLORE"
SELECTION_CONTESTED = "CONTESTED"
SELECTION_NOT_ELIGIBLE = "NOT_ELIGIBLE"
SELECTION_NO_ACTION = "NO_ACTION"


def decision_key(
    seed: int,
    order_id: str,
    policy: str,
    day: str,
    capacity_scenario: str = DIAGNOSTIC_CAPACITY_SCENARIO,
) -> str:
    """Stable idempotency key for one (seed, order, policy, day, capacity
    scenario) decision.

    ``capacity_scenario`` is included (defaulting to the single, unscaled
    ``DIAGNOSTIC_CAPACITY_SCENARIO`` a production deployment always runs at)
    because the *same* (seed, order, policy, day) can realize a genuinely
    different decision -- a different chosen action, or ``CONTESTED``
    instead of accepted -- at a different capacity-stress scenario (see
    ``evaluate_policies_across_capacity_scenarios``); omitting it would let
    two different scenarios' decisions silently collide on one key.
    """
    payload = f"{POLICY_EVALUATION_VERSION}|{seed}|{order_id}|{policy}|{day}|{capacity_scenario}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def content_fingerprint(*parts: Any) -> str:
    """Deterministic evaluation ID: a stable hash of arbitrary JSON-able parts.

    Used to identify one (config, seed, code-version) evaluation run without
    depending on wall-clock time or output directory naming -- the minimal
    reproducibility contract Stage 1 needs (full lineage/registry is Stage 2).
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _score_population(
    dataset: PrototypeDataset,
    features: pd.DataFrame,
    trained: TrainedBundle,
) -> pd.DataFrame:
    """Score every order in ``features`` without SHAP (not needed for policy value).

    Mirrors ``pipeline.score_orders`` (fused score, ``primary_cause``,
    business context) but skips ``explain_predictions``, which dominates
    runtime and is irrelevant to policy-value metrics.
    """
    xgb_scores = trained.risk_training.bundle.score(features)
    bbn_scores = trained.bayesian_bundle.score(features)
    fused = fuse_scores(
        xgb_scores,
        bbn_scores[["order_id", "bbn_risk_score"]],
        xgb_weight=trained.fusion_selection.chosen_weight,
    )
    scored = features.drop(columns=["otif_miss"], errors="ignore").merge(
        fused, on="order_id", validate="one_to_one"
    )
    scored = scored.rename(
        columns={"risk_model_score": "xgb_risk_score", "fused_risk_score": "combined_risk_score"}
    )
    scored["primary_cause"] = scored.apply(primary_cause_from_signals, axis=1)
    scored = attach_business_context(scored, dataset)
    bbn_extra = bbn_scores[["order_id", "intervention_scenarios_json", "evidence_coverage"]]
    scored = scored.merge(bbn_extra, on="order_id", validate="one_to_one")
    scored = _attach_value_aware_master_data(scored, dataset)
    return scored


def _attach_value_aware_master_data(
    scored: pd.DataFrame, dataset: PrototypeDataset
) -> pd.DataFrame:
    """Attach the two static, point-in-time-observable master-data traits the
    value-aware ``CURRENT_POLICY``'s execution-feasibility term needs (lane transit
    variability, SKU scarcity) that ``features.build_feature_table`` does not already
    carry for model training.

    Both are fixed per-entity generator traits (``data.py``'s ``lanes``/``skus``
    tables) -- stable business master data, never a per-order realized outcome -- so
    attaching them here (policy-evaluation-layer only, never fed back into
    ``build_feature_table``) does not leak any latent/realized information into either
    the model or the policy's decision.
    """
    enriched = scored.merge(
        dataset.lanes[["lane_id", "transit_variability_days"]].rename(
            columns={"transit_variability_days": "lane_transit_variability_days"}
        ),
        on="lane_id",
        how="left",
        validate="many_to_one",
    )
    sku_scarcity = dataset.skus.set_index("sku_id")["scarcity_trait"]
    enriched["sku_scarcity_trait"] = enriched["representative_sku"].map(sku_scarcity)
    enriched["sku_scarcity_trait"] = enriched["sku_scarcity_trait"].fillna(
        float(dataset.skus["scarcity_trait"].mean())
    )
    return enriched


@dataclass
class SeedContext:
    """Everything one seed's policy evaluation needs, built once and reused."""

    config: PrototypeConfig
    dataset: PrototypeDataset
    outcomes: pd.DataFrame
    causes: pd.DataFrame
    trained: TrainedBundle  # last (most-recent-history) fold's bundle
    decisions: pd.DataFrame  # out-of-sample scored + recommend_orders enrichment
    responses: pd.DataFrame  # action_response potential outcomes, all orders x actions
    coverage: dict[str, Any]  # rolling-origin cross-fitting coverage/fold diagnostics


#: Rolling-origin cross-fitting: the calendar is split into this many equal
#: chronological folds (by unique as-of-timestamp groups, the same
#: boundary-safe grouping ``features.temporal_split`` uses). The first fold
#: is pure warm-up history -- never scored, since there is no prior data to
#: fit a model on yet -- and every later fold is scored only by a model
#: trained (and its fusion weight/threshold selected) on history strictly
#: before it, so every evaluated order is genuinely out-of-sample and no
#: fold's own evaluation rows ever influence its model. See
#: ``build_seed_context``.
ROLLING_ORIGIN_FOLDS = 5
#: Fraction of each fold's growing training history held out as that fold's
#: own internal validation split (for fusion-weight/threshold selection) --
#: never overlapping with the fold's evaluation rows.
ROLLING_ORIGIN_TRAIN_FRACTION = 0.75


def _chronological_split_by_fraction(
    frame: pd.DataFrame, boundaries: tuple[float, ...]
) -> list[pd.DataFrame]:
    """Split ``frame`` chronologically into ``len(boundaries) + 1`` parts.

    ``boundaries`` are cumulative fractions in (0, 1), e.g. ``(0.6, 0.8)``
    for a 60/20/20 split (what ``features.temporal_split`` used to compute
    directly) or ``(0.2, 0.4, 0.6, 0.8)`` for 5 equal rolling-origin folds.
    Splits are made on whole groups of identical ``as_of_timestamp``/
    ``prediction_timestamp`` values, never mid-group -- the same
    boundary-safe grouping ``features.temporal_split`` uses, generalized to
    an arbitrary number of cut points.
    """
    time_column = "as_of_timestamp" if "as_of_timestamp" in frame else "prediction_timestamp"
    if time_column not in frame or "order_id" not in frame:
        raise ValueError("frame requires as_of_timestamp/prediction_timestamp and order_id")
    ordered = frame.sort_values([time_column, "order_id"]).reset_index(drop=True)

    unique_times = ordered[time_column].drop_duplicates().sort_values().to_numpy()
    counts_per_time = ordered.groupby(time_column).size().reindex(unique_times).to_numpy()
    cumulative = np.cumsum(counts_per_time)
    total = len(ordered)

    cut_times: list[Any] = []
    last_idx = 0
    for fraction in boundaries:
        idx = int(np.searchsorted(cumulative, total * fraction, side="left"))
        idx = min(max(idx, last_idx), len(unique_times) - 1)
        cut_times.append(unique_times[idx])
        last_idx = idx

    parts: list[pd.DataFrame] = []
    lower = None
    for cut_time in cut_times:
        mask = ordered[time_column] <= cut_time
        if lower is not None:
            mask &= ordered[time_column] > lower
        parts.append(ordered.loc[mask].reset_index(drop=True))
        lower = cut_time
    parts.append(ordered.loc[ordered[time_column] > lower].reset_index(drop=True))
    return parts


def build_seed_context(config: PrototypeConfig) -> SeedContext:
    """Generate data and score the full calendar via rolling-origin,
    chronological cross-fitting, so every evaluated order is genuinely
    out-of-sample.

    The calendar is split into ``ROLLING_ORIGIN_FOLDS`` equal chronological
    folds; the first fold is warm-up history (excluded from evaluation
    entirely -- there is no prior data to train a model on yet, reported
    honestly via ``coverage`` rather than silently trained on); each later
    fold is scored by its own model, trained via the same
    ``pipeline.train_full_bundle`` used everywhere else on an internal
    75/25 train/validation split of only the history strictly before that
    fold (so the fusion weight/threshold is also selected without ever
    seeing that fold's evaluation rows), then that fold's own held-out rows
    are scored by that fold-specific model. The full evaluated set is the
    union of all non-warm-up folds -- most of the calendar, including the
    scripted drift window (see ``DRIFT_WINDOW_FRACTION``), all genuinely
    out-of-sample.
    """
    dataset = generate_dataset(config)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    feature_table = build_feature_table(dataset, outcomes, causes)
    feature_table = attach_line_evidence_features(dataset, feature_table)

    n_folds = ROLLING_ORIGIN_FOLDS
    fold_boundaries = tuple(i / n_folds for i in range(1, n_folds))
    folds = _chronological_split_by_fraction(feature_table, fold_boundaries)

    decision_parts: list[pd.DataFrame] = []
    fold_diagnostics: list[dict[str, Any]] = []
    last_trained: TrainedBundle | None = None

    history = folds[0]
    for fold_index in range(1, n_folds):
        evaluation_fold = folds[fold_index]
        if evaluation_fold.empty:
            continue
        train_part, validation_part = _chronological_split_by_fraction(
            history, (ROLLING_ORIGIN_TRAIN_FRACTION,)
        )
        fold_split = TemporalSplit(
            train=train_part, validation=validation_part, test=evaluation_fold
        )
        fold_trained = train_full_bundle(dataset, outcomes, causes, fold_split, config)
        fold_scored = _score_population(dataset, evaluation_fold, fold_trained)
        fold_decisions = recommend_orders(fold_scored, risk_threshold=fold_trained.fused_threshold)
        fold_decisions["cross_fit_fold"] = fold_index
        decision_parts.append(fold_decisions)
        fold_diagnostics.append(
            {
                "fold": fold_index,
                "history_orders": int(len(history)),
                "train_orders": int(len(train_part)),
                "validation_orders": int(len(validation_part)),
                "evaluation_orders": int(len(evaluation_fold)),
                "fused_threshold": fold_trained.fused_threshold,
            }
        )
        last_trained = fold_trained
        # This fold's now-scored rows join history for every later fold --
        # never the reverse -- keeping the walk strictly forward in time.
        history = pd.concat([history, evaluation_fold], ignore_index=True)

    if last_trained is None:
        raise ValueError(
            "rolling-origin cross-fitting produced no evaluable folds -- "
            "increase n_orders or reduce ROLLING_ORIGIN_FOLDS"
        )

    decisions_all = pd.concat(decision_parts, ignore_index=True)
    decisions_all = decisions_all.merge(
        dataset.orders[["order_id", "order_date"]], on="order_id", validate="one_to_one"
    )
    responses = simulate_action_response(dataset, outcomes, causes, seed=config.seed)
    coverage = {
        "design": "rolling_origin_chronological_cross_fitting",
        "n_folds": n_folds,
        "orders_total": int(len(feature_table)),
        "orders_evaluated_out_of_sample": int(len(decisions_all)),
        "coverage_fraction": round(len(decisions_all) / len(feature_table), 4),
        "warm_up_orders_excluded": int(len(folds[0])),
        "folds": fold_diagnostics,
        "note": (
            "The first chronological fold is warm-up history with no prior "
            "data available to train a model on, so it is excluded from "
            "evaluation entirely (never scored, in-sample or otherwise); "
            "every other order is scored by a model trained -- and its "
            "fusion weight/threshold selected -- strictly on history before "
            "it. No evaluated order's own row was ever used to train or "
            "select the model that scored it."
        ),
    }
    return SeedContext(
        config=config,
        dataset=dataset,
        outcomes=outcomes,
        causes=causes,
        trained=last_trained,
        decisions=decisions_all,
        responses=responses,
        coverage=coverage,
    )


def _regime(order_date: pd.Series, dataset: PrototypeDataset) -> pd.Series:
    """Tag each order ``normal``/``drift`` using the same window ``data.py`` scripts."""
    horizon_start = dataset.orders["order_date"].min()
    horizon_end = dataset.orders["order_date"].max()
    drift_start = horizon_end - (horizon_end - horizon_start) * DRIFT_WINDOW_FRACTION
    return pd.Series(
        ["drift" if value >= drift_start else "normal" for value in order_date],
        index=order_date.index,
    )


def _priority_for_policy(policy: str, candidates: pd.DataFrame, seed: int) -> pd.Series:
    if policy == POLICY_RANDOM:
        return candidates["order_id"].map(
            lambda order_id: float(
                deterministic_uniforms(seed, order_id, f"policy::{POLICY_RANDOM}")[0]
            )
        )
    if policy == POLICY_HIGHEST_RISK:
        return candidates["combined_risk_score"].astype(float)
    if policy == POLICY_HIGHEST_FINANCIAL:
        return candidates["estimated_penalty_exposure"].astype(float)
    if policy == POLICY_STRONGEST_SIGNAL:
        cause_rank = candidates["primary_cause"].map(
            lambda cause: len(CAUSE_CATEGORIES) - list(CAUSE_CATEGORIES).index(cause)
            if cause in CAUSE_CATEGORIES
            else 0
        )
        signal_count = pd.to_numeric(
            candidates.get("active_leading_signal_count", 0), errors="coerce"
        ).fillna(0)
        return cause_rank * 10.0 + signal_count
    if policy == POLICY_LEGACY:
        return candidates["priority_score"].astype(float)
    if policy == POLICY_CURRENT:
        raise ValueError(
            "POLICY_CURRENT's priority comes from _value_aware_candidate_frame, "
            "not this single-priority-column dispatcher"
        )
    raise ValueError(f"no priority rule for policy {policy!r}")


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(max(value, low), high))


def _resource_id_for(row: pd.Series, resource_type: str) -> str:
    column = RESOURCE_ID_COLUMN[resource_type]
    value = row.get(column)
    return str(value) if pd.notna(value) else "UNASSIGNED"


def _single_node_scenarios(intervention_scenarios_json: Any) -> dict[str, dict[str, Any]]:
    """Parse ``intervention_scenarios_json`` into ``{intervened_node: scenario}`` for
    every persisted single-node ``do(node=0)`` scenario.

    The combined-mitigation scenario (multiple nodes intervened at once), when
    present, is excluded -- it is not attributable to one action.
    """
    try:
        parsed = json.loads(intervention_scenarios_json)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict) or item.get("type") != "single_node_mitigation":
            continue
        nodes = item.get("intervened_nodes") or []
        if len(nodes) == 1:
            result[str(nodes[0])] = item
    return result


def _value_aware_candidate_actions(
    row: pd.Series, *, use_bayesian: bool = True
) -> list[dict[str, Any]]:
    """Every feasible ``(action, structural_reduction, source)`` candidate this
    order's point-in-time evidence supports, before ranking/selection (see the module
    docstring's "Value-aware CURRENT_POLICY formula" -> "Candidate actions").

    Reads only ``leading_signal_*`` (active-at-decision-time flags) and, when
    ``use_bayesian`` (disabled only by the Bayesian-ablation diagnostic), the
    persisted Bayesian ``intervention_scenarios_json`` -- never ``root_causes``'
    retrospective rule evaluation, the simulator's own response draw, or any
    potential-outcome field.
    """
    scenarios = (
        _single_node_scenarios(row.get("intervention_scenarios_json")) if use_bayesian else {}
    )
    active_causes = [
        cause for cause in CAUSE_CATEGORIES if int(row.get(f"leading_signal_{cause}", 0)) == 1
    ]
    primary_cause = row.get("primary_cause")
    candidates: dict[str, dict[str, Any]] = {}
    for cause in active_causes:
        action = CAUSE_TO_ACTION.get(cause)
        if action is None:
            continue
        scenario = scenarios.get(cause)
        if scenario is not None:
            structural_reduction = _clip(float(scenario.get("relative_risk_reduction", 0.0)))
            source = "bayesian_structural_scenario"
        else:
            confidence = (
                CAUSE_MATCH_PRIMARY_CONFIDENCE
                if cause == primary_cause
                else CAUSE_MATCH_SECONDARY_CONFIDENCE
            )
            structural_reduction = _clip(confidence * FALLBACK_STRUCTURAL_REDUCTION)
            source = "leading_signal_fallback"
        existing = candidates.get(action)
        # An action may be reachable via more than one active cause (e.g.
        # WAREHOUSE_EXPEDITE from both DC_CAPACITY and WAREHOUSE_OPS) -- keep
        # whichever active cause gives this action the larger, better-evidenced
        # structural reduction; never sum them (no double counting).
        if existing is None or structural_reduction > existing["structural_reduction"]:
            candidates[action] = {
                "action_code": action,
                "cause": cause,
                "structural_reduction": structural_reduction,
                "structural_reduction_source": source,
            }
    if not candidates and primary_cause in CAUSE_TO_ACTION:
        # No active leading signal maps to a feasible action (should not occur given
        # this policy's eligible pool already requires primary_cause in
        # CAUSE_TO_ACTION, but handled defensively): fall back to the existing
        # deployed single-primary-cause mapping, explicitly documented.
        action = CAUSE_TO_ACTION[primary_cause]
        candidates[action] = {
            "action_code": action,
            "cause": primary_cause,
            "structural_reduction": _clip(FALLBACK_STRUCTURAL_REDUCTION),
            "structural_reduction_source": "primary_cause_fallback",
        }
    return list(candidates.values())


def _resource_trait_score(action: str, row: pd.Series) -> float:
    """Point-in-time-observable execution-feasibility trait for ``action``'s resource
    pool -- stable business master data (vendor/DC/lane/customer/SKU attributes),
    never simulator truth or a realized outcome.
    """
    if action == "VENDOR_ESCALATION":
        return _clip(float(row.get("vendor_reliability_score", 0.85)))
    if action == "INVENTORY_REALLOCATION":
        scarcity = float(row.get("sku_scarcity_trait", 0.05))
        return _clip(1.0 - FEASIBILITY_SKU_SCARCITY_SCALE * scarcity)
    if action == "WAREHOUSE_EXPEDITE":
        return _clip(1.0 - float(row.get("dc_utilization_at_prediction", 0.5)))
    if action == "ALTERNATE_TRANSPORT":
        variability = float(row.get("lane_transit_variability_days", 0.7))
        return _clip(1.0 - variability / FEASIBILITY_LANE_VARIABILITY_NORMALIZATION_DAYS)
    if action == "APPOINTMENT_COORDINATION":
        return 0.4 if bool(row.get("customer_appointment_required", False)) else 0.9
    return 0.7  # ORDER_CAPTURE_CORRECTION: no differentiating point-in-time trait available.


def _slack_score(row: pd.Series) -> float:
    slack_hours = float(row.get("remaining_slack_hours", 0.0))
    return _clip(slack_hours / FEASIBILITY_SLACK_NORMALIZATION_HOURS)


def _execution_feasibility(action: str, row: pd.Series) -> float:
    """Bounded ``[0, 1]`` execution-feasibility proxy: fixed weighted sum of remaining
    promise slack, an action-specific resource trait, and Bayesian evidence coverage
    (causal confidence) -- see the module docstring.
    """
    causal_confidence = _clip(float(row.get("evidence_coverage", 1.0)))
    return _clip(
        FEASIBILITY_WEIGHT_SLACK * _slack_score(row)
        + FEASIBILITY_WEIGHT_RESOURCE_TRAIT * _resource_trait_score(action, row)
        + FEASIBILITY_WEIGHT_CAUSAL_CONFIDENCE * causal_confidence
    )


def _expected_value_density(
    row: pd.Series,
    action: str,
    structural_reduction: float,
    *,
    base_capacity: float,
) -> tuple[float, float]:
    """Return ``(expected_benefit, value_density)`` for candidate ``(row, action)``.

    ``expected_benefit = estimated_penalty_exposure * structural_reduction *
    execution_feasibility``; ``value_density = expected_benefit /
    normalized_resource_fraction`` (see the module docstring for the full
    derivation/justification against double counting).
    """
    exposure = float(row.get("estimated_penalty_exposure", 0.0))
    feasibility = _execution_feasibility(action, row)
    expected_benefit = exposure * structural_reduction * feasibility
    resource_type = ACTION_RESOURCE_TYPE[action]
    demand = _demand_units(resource_type, float(row.get("quantity_at_risk", 0.0)))
    normalized_fraction = demand / base_capacity if base_capacity > 0 else demand
    density = expected_benefit / max(normalized_fraction, VALUE_DENSITY_MIN_RESOURCE_FRACTION)
    return expected_benefit, density


def _value_aware_candidate_frame(
    decisions: pd.DataFrame,
    base_capacities: ResourceCapacities,
    *,
    use_bayesian: bool = True,
) -> pd.DataFrame:
    """Build ``POLICY_CURRENT``'s one-row-per-order candidate frame: the single
    highest positive-value-density feasible action among every candidate this
    order's point-in-time evidence supports (see the module docstring's
    "Value-aware CURRENT_POLICY formula").

    Uses the *same* eligible pool as every other non-oracle/non-no-action policy
    (``decision_status != MONITOR`` and ``primary_cause`` mapped by
    ``CAUSE_TO_ACTION``) so improvement is attributable to the ranking/action-choice,
    never a different eligibility rule.
    """
    eligible = decisions.loc[decisions["decision_status"] != "MONITOR"].copy()
    eligible = eligible.loc[eligible["primary_cause"].isin(CAUSE_TO_ACTION)].copy()

    rows: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        action_candidates = _value_aware_candidate_actions(row, use_bayesian=use_bayesian)
        best: dict[str, Any] | None = None
        evaluated: list[dict[str, Any]] = []
        for candidate in action_candidates:
            action = candidate["action_code"]
            resource_type = ACTION_RESOURCE_TYPE[action]
            resource_id = _resource_id_for(row, resource_type)
            base_capacity = base_capacities.pool(resource_type).get(resource_id, 0.0)
            expected_benefit, density = _expected_value_density(
                row, action, candidate["structural_reduction"], base_capacity=base_capacity
            )
            scored_candidate = {
                **candidate,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "expected_benefit": expected_benefit,
                "value_density": density,
            }
            evaluated.append(scored_candidate)
            if best is None or density > best["value_density"]:
                best = scored_candidate
        if best is None:
            continue
        row_out = row.to_dict()
        row_out.update(
            {
                "action_code": best["action_code"],
                "resource_type": best["resource_type"],
                "resource_id": best["resource_id"],
                "priority_score": best["value_density"],
                "expected_benefit": best["expected_benefit"],
                "structural_reduction": best["structural_reduction"],
                "structural_reduction_source": best["structural_reduction_source"],
                "candidate_action_count": len(evaluated),
                "candidate_actions_json": json.dumps(
                    sorted({c["action_code"] for c in evaluated}), separators=(",", ":")
                ),
            }
        )
        rows.append(row_out)
    return pd.DataFrame(rows)


def _candidate_frame(
    policy: str,
    decisions: pd.DataFrame,
    responses: pd.DataFrame,
    seed: int,
    *,
    base_capacities: ResourceCapacities | None = None,
    use_bayesian: bool = True,
) -> pd.DataFrame:
    """Return one row per order this policy would consider acting on.

    Non-oracle, non-``CURRENT_POLICY`` policies share the same eligible pool
    (``combined_risk_score >= fused_threshold``, already reflected in
    ``decisions["decision_status"] != MONITOR``) and the same single
    cause-to-action mapping as production. ``CURRENT_POLICY`` shares the identical
    eligible pool but may choose *any* point-in-time-feasible action per order (see
    ``_value_aware_candidate_frame``); only ``ORACLE_EVALUATION_ONLY`` may pick a
    different action per order based on the twin's own potential outcomes.
    """
    if policy == POLICY_ORACLE:
        action_rows = responses.loc[responses["action_code"] != NO_ACTION].copy()
        best_idx = action_rows.groupby("order_id")["avoided_penalty"].idxmax()
        best = action_rows.loc[best_idx].set_index("order_id")
        best = best.loc[best["avoided_penalty"] > 0]
        # decisions already carries its own primary_cause-implied resource_type
        # /resource_id (from recommend_orders); the oracle may choose a
        # different action/resource, so drop those before merging in its own
        # mapping to avoid a silent _x/_y column collision.
        decisions_base = decisions.drop(columns=["resource_type", "resource_id"], errors="ignore")
        candidates = decisions_base.merge(
            best[["action_code", "resource_type", "resource_id", "avoided_penalty"]],
            on="order_id",
            how="inner",
            validate="one_to_one",
        )
        candidates["priority_score"] = candidates["avoided_penalty"]
        return candidates

    if policy == POLICY_CURRENT:
        if base_capacities is None:
            raise ValueError("POLICY_CURRENT's candidate frame requires base_capacities")
        return _value_aware_candidate_frame(decisions, base_capacities, use_bayesian=use_bayesian)

    eligible = decisions.loc[decisions["decision_status"] != "MONITOR"].copy()
    eligible = eligible.loc[eligible["primary_cause"].isin(CAUSE_TO_ACTION)].copy()
    eligible["action_code"] = eligible["primary_cause"].map(CAUSE_TO_ACTION)
    eligible["resource_type"] = eligible["action_code"].map(ACTION_RESOURCE_TYPE)
    eligible["resource_id"] = eligible.apply(
        lambda row: _resource_id_for(row, row["resource_type"]), axis=1
    )
    eligible["priority_score"] = _priority_for_policy(policy, eligible, seed)
    return eligible


def _demand_units(resource_type: str, quantity_at_risk: float) -> float:
    """Thin alias for :func:`resources.demand_units_for` (imported name kept
    local so call sites read naturally alongside this module's own helpers).
    """
    return demand_units_for(resource_type, quantity_at_risk)


def _allocate_day(
    day_candidates: pd.DataFrame, capacities: ResourceCapacities, day: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if day_candidates.empty:
        return day_candidates.assign(decision_status=pd.Series(dtype=str)), []
    frame = day_candidates.copy()
    frame["demand_units"] = [
        _demand_units(resource_type, qty)
        for resource_type, qty in zip(
            frame["resource_type"], frame["quantity_at_risk"], strict=True
        )
    ]
    result, remaining = allocate_under_capacity(frame, capacities)
    ledger: list[dict[str, Any]] = []
    for resource_type, resource_id in set(
        zip(frame["resource_type"], frame["resource_id"], strict=True)
    ):
        before = capacities.pool(resource_type).get(resource_id, 0.0)
        after = remaining.pool(resource_type).get(resource_id, 0.0)
        ledger.append(
            {
                "day": day,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "capacity_before": before,
                "capacity_after": after,
                "consumed": before - after,
            }
        )
    return result, ledger


def _discrete_explore_counts(
    k_total: int,
    capacity: float,
    seed: int,
    resource_type: str,
    resource_id: str,
    day: str,
) -> tuple[int, int, float, int]:
    """Split ``k_total`` accepted discrete slots into ``(k_exploit, k_explore)``.

    Targets ``EXPLORATION_FRACTION`` of the pool's *capacity* (not merely of
    however many happen to be accepted) via stochastic rounding: a single
    seeded Bernoulli draw, with probability equal to the fractional
    remainder of ``capacity * EXPLORATION_FRACTION``, converts a non-integer
    target into a whole slot, unbiased in expectation
    (``E[k_explore] == capacity * EXPLORATION_FRACTION``) and fully
    reproducible given ``seed``. Returns ``(k_exploit, k_explore, residual,
    floor_k)`` -- the latter two are also needed by
    :func:`_discrete_propensities` to compute each candidate's exact
    marginal selection probability, marginalized over that one Bernoulli
    draw.
    """
    target = capacity * EXPLORATION_FRACTION
    floor_k = int(target)
    residual = target - floor_k
    draw_key = f"explore_extra::{resource_type}::{resource_id}::{day}"
    extra = 1 if deterministic_uniforms(seed, draw_key, "explore_schedule")[0] < residual else 0
    k_explore = min(k_total, floor_k + extra)
    k_exploit = k_total - k_explore
    return k_exploit, k_explore, residual, floor_k


def _discrete_propensities(n: int, k_total: int, floor_k: int, residual: float) -> list[float]:
    """Exact marginal per-rank selection probability for the discrete scheme.

    Rank 0 is the highest-priority candidate in the pool's priority-ranked
    order. Marginalizes over the single Bernoulli draw in
    :func:`_discrete_explore_counts` (which of the two possible
    exploit/explore boundaries was realized) and, within whichever branch's
    randomly shuffled remainder pool a rank falls into, the exact
    without-replacement inclusion probability (``k_explore_branch /
    remainder_size_branch``) -- identical for every member of that pool
    since the shuffle draw is i.i.d. uniform, regardless of the realized
    outcome.
    """

    def _branch(k_exploit_branch: int) -> list[float]:
        k_exploit_branch = max(0, min(k_total, k_exploit_branch))
        k_explore_branch = k_total - k_exploit_branch
        remainder_size = n - k_exploit_branch
        remainder_prob = k_explore_branch / remainder_size if remainder_size > 0 else 0.0
        return [1.0 if rank < k_exploit_branch else remainder_prob for rank in range(n)]

    branch_extra_0 = _branch(k_total - floor_k)
    branch_extra_1 = _branch(k_total - floor_k - 1)
    return [
        (1.0 - residual) * p0 + residual * p1
        for p0, p1 in zip(branch_extra_0, branch_extra_1, strict=True)
    ]


def _allocate_day_with_exploration(
    day: str,
    day_candidates: pd.DataFrame,
    capacities: ResourceCapacities,
    seed: int,
    capacity_scenario: str = DIAGNOSTIC_CAPACITY_SCENARIO,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Same greedy allocation, but ~``EXPLORATION_FRACTION`` of every pool's
    daily capacity is reserved for a seeded-random draw among eligible
    candidates that did not make the priority-ranked ("exploit") cut, rather
    than always going to the next-highest-priority order.

    Uses one of two capacity-preserving designs per pool (see
    ``EXPLORATION_FRACTION``'s docstring for why a single fractional
    threshold cannot be shared between them without dropping capacity):

    - **Discrete** (``vendor``/``customer``) pools split *whole-slot counts*
      (:func:`_discrete_explore_counts`), so ``min(capacity, n_candidates)``
      slots are always filled -- a 1-slot pool never goes unfilled -- and
      every candidate's exact marginal propensity
      (:func:`_discrete_propensities`) is logged as
      ``assignment_probability``.
    - **Continuous** (``dc``/``lane``) pools fill the explore stage up to
      the pool's *actual* remaining capacity after exploit acceptance
      (``capacity - accepted_demand``), not a fixed
      ``capacity * EXPLORATION_FRACTION`` slice, so real leftover capacity
      is never dropped just because an order's size didn't match the
      nominal slice. Because order sizes differ, a sequential
      fits-if-it-fits random draw's exact per-order inclusion probability
      has no simple closed form; rather than mislabel an approximation as
      exact, every candidate in that pool's random remainder gets
      ``assignment_probability = None`` and a separately named
      ``pool_reservation_ratio`` (the *pool's* intended explore share, not
      a per-order probability) instead.
    """
    if day_candidates.empty:
        return day_candidates.assign(decision_status=pd.Series(dtype=str)), [], []

    frame = day_candidates.copy()
    frame["demand_units"] = [
        _demand_units(resource_type, qty)
        for resource_type, qty in zip(
            frame["resource_type"], frame["quantity_at_risk"], strict=True
        )
    ]
    frame["decision_status"] = "CONTESTED"
    frame["selection_mode"] = SELECTION_CONTESTED
    frame["assignment_probability"] = 0.0
    frame["pool_reservation_ratio"] = None

    remaining = ResourceCapacities(
        dc_units=dict(capacities.dc_units),
        lane_units=dict(capacities.lane_units),
        vendor_slots=dict(capacities.vendor_slots),
        customer_slots=dict(capacities.customer_slots),
    )
    ledger: list[dict[str, Any]] = []
    decision_log: list[dict[str, Any]] = []

    ranked = frame.sort_values(
        ["priority_score", "order_id"], ascending=[False, True], kind="stable"
    )
    for (resource_type, resource_id), group in ranked.groupby(
        ["resource_type", "resource_id"], sort=False
    ):
        pool = remaining.pool(resource_type)
        capacity = pool.get(resource_id, 0.0)
        n = len(group)
        explore_index: list[Any] = []

        if resource_type in DISCRETE_RESOURCE_TYPES:
            k_total = min(int(capacity + 1e-9), n)
            k_exploit, k_explore, residual, floor_k = _discrete_explore_counts(
                k_total, capacity, seed, resource_type, resource_id, day
            )
            exploit_index = group.index[:k_exploit]
            remainder = group.iloc[k_exploit:]

            frame.loc[exploit_index, "decision_status"] = "RECOMMENDED"
            frame.loc[exploit_index, "selection_mode"] = SELECTION_EXPLOIT
            accepted_demand = float(k_exploit)

            if not remainder.empty and k_explore > 0:
                order_keys = remainder["order_id"].tolist()
                draw_key = f"explore::{resource_type}::{resource_id}::{day}"
                random_keys = {
                    order_id: deterministic_uniforms(seed, order_id, draw_key)[0]
                    for order_id in order_keys
                }
                shuffled = remainder.assign(
                    _explore_draw=remainder["order_id"].map(random_keys)
                ).sort_values("_explore_draw", kind="stable")
                explore_index = shuffled.index[:k_explore].tolist()
                frame.loc[explore_index, "decision_status"] = "RECOMMENDED"
                frame.loc[explore_index, "selection_mode"] = SELECTION_EXPLORE
                accepted_demand += float(k_explore)

            # Exact marginal propensity for every candidate in the pool
            # (selected or not), marginalized over the Bernoulli draw above
            # -- overwrites the frame-level default/exploit placeholder.
            propensities = _discrete_propensities(n, k_total, floor_k, residual)
            for rank, index in enumerate(group.index):
                frame.loc[index, "assignment_probability"] = propensities[rank]

            pool[resource_id] = max(0.0, capacity - accepted_demand)
        else:
            exploit_capacity = capacity * (1.0 - EXPLORATION_FRACTION)
            cumulative = group["demand_units"].cumsum()
            exploit_mask = cumulative <= exploit_capacity
            exploit_index = group.index[exploit_mask]
            remainder_index = group.index[~exploit_mask]

            frame.loc[exploit_index, "decision_status"] = "RECOMMENDED"
            frame.loc[exploit_index, "selection_mode"] = SELECTION_EXPLOIT
            frame.loc[exploit_index, "assignment_probability"] = 1.0
            accepted_demand = float(group.loc[exploit_index, "demand_units"].sum())

            remainder = group.loc[remainder_index]
            # Use the pool's *actual* remaining capacity, not a fixed
            # capacity * EXPLORATION_FRACTION slice, so an order that
            # doesn't fit in the nominal exploit slice but still fits in the
            # true leftover capacity is never dropped.
            explore_capacity = max(0.0, capacity - accepted_demand)
            if not remainder.empty and explore_capacity > 0:
                order_keys = remainder["order_id"].tolist()
                draw_key = f"explore::{resource_type}::{resource_id}::{day}"
                random_keys = {
                    order_id: deterministic_uniforms(seed, order_id, draw_key)[0]
                    for order_id in order_keys
                }
                shuffled = remainder.assign(
                    _explore_draw=remainder["order_id"].map(random_keys)
                ).sort_values("_explore_draw", kind="stable")
                explore_cumulative = shuffled["demand_units"].cumsum()
                explore_mask = explore_cumulative <= explore_capacity
                explore_index = shuffled.index[explore_mask].tolist()
                frame.loc[explore_index, "decision_status"] = "RECOMMENDED"
                frame.loc[explore_index, "selection_mode"] = SELECTION_EXPLORE
                accepted_demand += float(shuffled.loc[explore_index, "demand_units"].sum())
            # Heterogeneous order sizes mean the exact per-order inclusion
            # probability of the sequential fits-if-it-fits random draw has
            # no closed form for *any* member of this remainder pool (not
            # just the ones actually selected); log the honest pool-level
            # reservation share instead of a mislabeled point estimate.
            if not remainder.empty:
                frame.loc[remainder.index, "assignment_probability"] = None
                frame.loc[remainder.index, "pool_reservation_ratio"] = EXPLORATION_FRACTION

            pool[resource_id] = max(0.0, capacity - accepted_demand)

        ledger.append(
            {
                "day": day,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "capacity_before": capacity,
                "capacity_after": pool[resource_id],
                "consumed": capacity - pool[resource_id],
            }
        )
        accepted_ids = set(frame.loc[exploit_index, "order_id"]) | set(
            frame.loc[explore_index, "order_id"] if explore_index else []
        )
        for _, row in group.iterrows():
            probability = frame.loc[row.name, "assignment_probability"]
            reservation_ratio = frame.loc[row.name, "pool_reservation_ratio"]
            decision_log.append(
                {
                    "day": day,
                    "order_id": row["order_id"],
                    "policy": POLICY_CURRENT,
                    "policy_version": POLICY_EVALUATION_VERSION,
                    "capacity_scenario": capacity_scenario,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "capacity_before": capacity,
                    "capacity_after": pool[resource_id],
                    "chosen_action": (
                        row["action_code"] if row["order_id"] in accepted_ids else None
                    ),
                    "rejected_feasible_actions": json.dumps(
                        [action for action in ACTIONS if action != row["action_code"]]
                    ),
                    "selection_mode": frame.loc[row.name, "selection_mode"],
                    "assignment_probability": (
                        None if pd.isna(probability) else float(probability)
                    ),
                    "pool_reservation_ratio": (
                        None if pd.isna(reservation_ratio) else float(reservation_ratio)
                    ),
                    "decision_key": decision_key(
                        seed, row["order_id"], POLICY_CURRENT, day, capacity_scenario
                    ),
                }
            )
    return frame, ledger, decision_log


def _treatment_table(
    policy: str,
    decisions: pd.DataFrame,
    responses_indexed: pd.DataFrame,
    responses_flat: pd.DataFrame,
    dataset: PrototypeDataset,
    capacity_schedule: dict[str, ResourceCapacities],
    *,
    seed: int,
    capacity_scenario: str = DIAGNOSTIC_CAPACITY_SCENARIO,
    base_capacities: ResourceCapacities | None = None,
    use_bayesian: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Return (per-order treatment outcomes, resource ledger, per-candidate
    allocation log, decision log) for one policy at one capacity scenario.

    ``capacity_schedule`` (see ``resources.build_capacity_schedule``) maps
    each calendar day (ISO date string) to that day's already-scaled
    ``ResourceCapacities`` for the scenario under evaluation -- built once
    and shared, unmodified, across every policy, so no policy ever sees a
    different realized capacity than any other for the same (day,
    scenario). ``base_capacities`` (the scenario-*independent* default daily
    capacities, see ``resources.default_daily_capacities``) is only used by
    ``POLICY_CURRENT``'s value-density ranking (never for acceptance), so the
    ranking itself is identical across capacity-stress scenarios. ``use_bayesian``
    is only consumed by ``POLICY_CURRENT`` (see ``bayesian_ablation_diagnostic``).
    """
    base = decisions[
        [
            "order_id",
            "order_date",
            "primary_cause",
            "combined_risk_score",
            "customer_tier",
            "vendor_id",
            "dc_id",
            "lane_id",
            "customer_id",
            "representative_sku",
            "order_value",
            "quantity_at_risk",
        ]
    ].copy()
    base["order_date"] = base["order_date"].dt.normalize()
    base["regime"] = _regime(base["order_date"], dataset)
    sku_criticality = dataset.skus.set_index("sku_id")["criticality_tier"]
    base["sku_criticality_tier"] = (
        base["representative_sku"].map(sku_criticality).fillna("STANDARD")
    )

    no_action = responses_indexed.xs(NO_ACTION, level="action_code")
    base = base.join(
        no_action[["realized_penalty"]].rename(columns={"realized_penalty": "no_action_penalty"}),
        on="order_id",
    )
    base["assigned_action"] = NO_ACTION
    base["treated"] = False
    base["realized_penalty"] = base["no_action_penalty"]
    base["avoided_penalty"] = 0.0
    base["demand_units"] = 0.0
    base["resource_type"] = None
    base["resource_id"] = None
    base = base.set_index("order_id")

    ledger_rows: list[dict[str, Any]] = []
    decision_log: list[dict[str, Any]] = []
    candidate_log_parts: list[pd.DataFrame] = []
    candidate_log_columns = ["order_id", "resource_type", "resource_id", "decision_status"]

    if policy == POLICY_NO_ACTION:
        return (
            base.reset_index(),
            pd.DataFrame(ledger_rows),
            pd.DataFrame(columns=candidate_log_columns),
            decision_log,
        )

    candidates = _candidate_frame(
        policy,
        decisions,
        responses_flat,
        seed,
        base_capacities=base_capacities,
        use_bayesian=use_bayesian,
    )
    if "quantity_at_risk" not in candidates.columns:
        candidates = candidates.merge(
            decisions[["order_id", "quantity_at_risk"]],
            on="order_id",
            how="left",
            validate="many_to_one",
        )
    candidates["order_date"] = candidates["order_date"].dt.normalize()

    # Always log action_code/priority_score (useful for every policy's chosen-action
    # mix); POLICY_CURRENT additionally carries candidate-coverage/Bayesian-evidence
    # diagnostics that no other policy's candidate frame has.
    candidate_log_columns = ["order_id", "resource_type", "resource_id", "decision_status"]
    for extra_column in ("action_code", "priority_score"):
        if extra_column in candidates.columns:
            candidate_log_columns.append(extra_column)
    for extra_column in ("structural_reduction_source", "candidate_action_count"):
        if extra_column in candidates.columns:
            candidate_log_columns.append(extra_column)

    for day, day_candidates in candidates.groupby("order_date", sort=True):
        day_str = day.isoformat()
        capacities = capacity_schedule[day_str]
        if policy == POLICY_CURRENT:
            allocated, ledger, log_rows = _allocate_day_with_exploration(
                day_str, day_candidates, capacities, seed, capacity_scenario
            )
            decision_log.extend(log_rows)
        else:
            allocated, ledger = _allocate_day(day_candidates, capacities, day_str)
        ledger_rows.extend(ledger)
        candidate_log_parts.append(allocated[candidate_log_columns])

        accepted = allocated.loc[allocated["decision_status"] == "RECOMMENDED"]
        for _, row in accepted.iterrows():
            key = (row["order_id"], row["action_code"])
            if key not in responses_indexed.index:
                continue
            outcome = responses_indexed.loc[key]
            base.loc[row["order_id"], "assigned_action"] = row["action_code"]
            base.loc[row["order_id"], "treated"] = True
            base.loc[row["order_id"], "realized_penalty"] = outcome["realized_penalty"]
            base.loc[row["order_id"], "avoided_penalty"] = outcome["avoided_penalty"]
            base.loc[row["order_id"], "demand_units"] = row["demand_units"]
            base.loc[row["order_id"], "resource_type"] = row["resource_type"]
            base.loc[row["order_id"], "resource_id"] = row["resource_id"]

    candidate_log = (
        pd.concat(candidate_log_parts, ignore_index=True)
        if candidate_log_parts
        else pd.DataFrame(columns=candidate_log_columns)
    )
    return base.reset_index(), pd.DataFrame(ledger_rows), candidate_log, decision_log


def _summarize(
    table: pd.DataFrame, ledger: pd.DataFrame, candidate_log: pd.DataFrame
) -> dict[str, Any]:
    treated = table.loc[table["treated"]]
    total_avoided = float(table["avoided_penalty"].sum())
    n_orders = len(table)
    n_treated = len(treated)

    if not ledger.empty:
        normalized_rows = []
        for _, row in ledger.iterrows():
            capacity = row["capacity_before"]
            normalized_rows.append(row["consumed"] / capacity if capacity > 0 else 0.0)
        total_normalized_units = float(sum(normalized_rows))
        by_resource: dict[str, Any] = {}
        for resource_type, group in ledger.groupby("resource_type"):
            consumed = float(group["consumed"].sum())
            normalized = float(
                sum(
                    row["consumed"] / row["capacity_before"] if row["capacity_before"] > 0 else 0.0
                    for _, row in group.iterrows()
                )
            )
            by_resource[resource_type] = {
                "units_consumed": consumed,
                "normalized_units": normalized,
            }
    else:
        total_normalized_units = 0.0
        by_resource = {}

    avoided_per_normalized_unit = (
        total_avoided / total_normalized_units if total_normalized_units > 0 else 0.0
    )
    avoided_misses_per_100 = (
        100.0 * float((treated["avoided_penalty"] > 0).sum()) / n_treated if n_treated else 0.0
    )
    action_precision = (
        float((treated["avoided_penalty"] > 0).sum()) / n_treated if n_treated else 0.0
    )
    waste_rate = float((treated["avoided_penalty"] == 0).sum()) / n_treated if n_treated else 0.0
    adverse_rate = float((treated["avoided_penalty"] < 0).sum()) / n_treated if n_treated else 0.0
    protected_value = float(treated.loc[treated["avoided_penalty"] > 0, "order_value"].sum())

    def _grouped_avoided(column: str) -> dict[str, float]:
        if column not in table.columns:
            return {}
        return {
            str(key): round(float(value), 4)
            for key, value in table.groupby(column)["avoided_penalty"].sum().items()
        }

    # Discriminativeness diagnostics: does capacity actually constrain this
    # policy's allocation, or does it accept every eligible candidate
    # regardless of priority (in which case the ranking cannot possibly be
    # measured against another ranking at this capacity level)? Reported at
    # every capacity scenario -- see requirement #3 / the module docstring.
    n_eligible_candidates = int(len(candidate_log))
    n_contested = (
        int((candidate_log["decision_status"] == "CONTESTED").sum())
        if n_eligible_candidates
        else 0
    )
    contested_rate = n_contested / n_eligible_candidates if n_eligible_candidates else 0.0

    if not ledger.empty:
        pool_day_binding = ledger["consumed"] >= (ledger["capacity_before"] - 1e-9)
        capacity_binding_rate = float(pool_day_binding.mean())
        n_pool_days = int(len(ledger))
    else:
        capacity_binding_rate = 0.0
        n_pool_days = 0

    # Value-aware-policy-only diagnostics (only populated when candidate_log carries
    # these columns, i.e. for POLICY_CURRENT -- see _treatment_table): candidate/
    # chosen action mix and Bayesian-vs-fallback candidate-action coverage.
    if "action_code" in candidate_log.columns and n_eligible_candidates:
        candidate_action_mix = {
            str(key): round(float(value), 4)
            for key, value in candidate_log["action_code"].value_counts(normalize=True).items()
        }
        recommended = candidate_log.loc[candidate_log["decision_status"] == "RECOMMENDED"]
        chosen_action_mix = (
            {
                str(key): round(float(value), 4)
                for key, value in recommended["action_code"].value_counts(normalize=True).items()
            }
            if len(recommended)
            else {}
        )
    else:
        candidate_action_mix = {}
        chosen_action_mix = {}

    if "structural_reduction_source" in candidate_log.columns and n_eligible_candidates:
        source = candidate_log["structural_reduction_source"]
        candidate_action_coverage = round(float((source != "primary_cause_fallback").mean()), 4)
        bayesian_evidence_rate = round(
            float((source == "bayesian_structural_scenario").mean()), 4
        )
    else:
        candidate_action_coverage = None
        bayesian_evidence_rate = None

    mean_candidate_actions_per_order = (
        round(float(candidate_log["candidate_action_count"].mean()), 4)
        if "candidate_action_count" in candidate_log.columns and n_eligible_candidates
        else None
    )

    return {
        "orders_total": n_orders,
        "orders_treated": n_treated,
        "total_avoided_penalty": round(total_avoided, 4),
        "median_avoided_penalty_per_order": round(float(table["avoided_penalty"].median()), 4),
        "avoided_misses_per_100_actions": round(avoided_misses_per_100, 4),
        "protected_value": round(protected_value, 4),
        "action_precision": round(action_precision, 4),
        "normalized_resource_units_consumed": round(total_normalized_units, 6),
        "avoided_penalty_per_normalized_resource_unit": round(avoided_per_normalized_unit, 4),
        "waste_no_benefit_rate": round(waste_rate, 4),
        "adverse_response_rate": round(adverse_rate, 4),
        "eligible_candidates": n_eligible_candidates,
        "contested_candidates": n_contested,
        "contested_rate": round(contested_rate, 4),
        "pool_days_observed": n_pool_days,
        "capacity_binding_rate": round(capacity_binding_rate, 4),
        "resource_breakdown": by_resource,
        "value_by_action": _grouped_avoided("assigned_action"),
        "value_by_cause": _grouped_avoided("primary_cause"),
        "value_by_customer_tier": _grouped_avoided("customer_tier"),
        "value_by_vendor": _grouped_avoided("vendor_id"),
        "value_by_sku_criticality": _grouped_avoided("sku_criticality_tier"),
        "value_by_regime": _grouped_avoided("regime"),
        "candidate_action_mix": candidate_action_mix,
        "chosen_action_mix": chosen_action_mix,
        "candidate_action_coverage": candidate_action_coverage,
        "bayesian_evidence_rate": bayesian_evidence_rate,
        "mean_candidate_actions_per_order": mean_candidate_actions_per_order,
    }


def evaluation_calendar_days(context: SeedContext) -> pd.DatetimeIndex:
    """The full, deduplicated, chronologically-sorted set of order dates in
    this seed's evaluated population -- the day sequence every capacity
    scenario's schedule is built over (see ``resources.build_capacity_schedule``),
    so every policy shares the exact same day->capacity mapping regardless
    of which subset of days that policy's own candidates happen to touch.
    """
    return pd.DatetimeIndex(
        context.decisions["order_date"].dt.normalize().unique()
    ).sort_values()


def evaluate_policies(
    context: SeedContext,
    *,
    capacity_multiplier: float = 1.0,
    scenario_name: str = DIAGNOSTIC_CAPACITY_SCENARIO,
) -> dict[str, Any]:
    """Evaluate all eight policies for one seed's ``SeedContext`` at a single
    capacity-stress scenario.

    ``capacity_multiplier`` is applied uniformly to every resource pool via
    one shared ``resources.build_capacity_schedule`` call, then handed
    unmodified to every policy's own allocation -- no policy ever sees a
    different realized capacity than any other for the same scenario (see
    ``test_policy_evaluation.py``'s capacity-scenario-equality tests).
    ``POLICY_CURRENT``'s value-density ranking additionally uses the
    scenario-*independent* ``resources.default_daily_capacities`` (see
    ``_treatment_table``), so its ranking is identical across scenarios; only the
    acceptance cutoff varies.

    Returns a dict keyed by policy name with metrics from ``_summarize``,
    plus ``avoidable_miss_coverage``/``regret_vs_oracle`` computed relative
    to the oracle, and the current policy's decision log. Use
    ``evaluate_policies_across_capacity_scenarios`` to run every
    pre-specified scenario in ``resources.CAPACITY_SCENARIOS`` at once.
    """
    calendar_days = evaluation_calendar_days(context)
    capacity_schedule = build_capacity_schedule(context.dataset, calendar_days, capacity_multiplier)
    base_capacities = default_daily_capacities(context.dataset)

    responses_indexed = context.responses.set_index(["order_id", "action_code"])
    results: dict[str, Any] = {}
    tables: dict[str, pd.DataFrame] = {}
    decision_log_rows: list[dict[str, Any]] = []

    for policy in POLICIES:
        table, ledger, candidate_log, log_rows = _treatment_table(
            policy,
            context.decisions,
            responses_indexed,
            context.responses,
            context.dataset,
            capacity_schedule,
            seed=context.config.seed,
            capacity_scenario=scenario_name,
            base_capacities=base_capacities,
        )
        tables[policy] = table
        results[policy] = _summarize(table, ledger, candidate_log)
        if policy == POLICY_CURRENT:
            decision_log_rows = log_rows

    oracle_total = results[POLICY_ORACLE]["total_avoided_penalty"]
    oracle_reachable = tables[POLICY_ORACLE]
    avoidable_orders = set(
        oracle_reachable.loc[oracle_reachable["avoided_penalty"] > 0, "order_id"]
    )
    for policy in POLICIES:
        this_avoided = set(
            tables[policy].loc[tables[policy]["avoided_penalty"] > 0, "order_id"]
        )
        coverage = (
            len(this_avoided & avoidable_orders) / len(avoidable_orders)
            if avoidable_orders
            else 0.0
        )
        results[policy]["avoidable_miss_coverage"] = round(coverage, 4)
        results[policy]["regret_vs_oracle"] = round(
            oracle_total - results[policy]["total_avoided_penalty"], 4
        )

    return {
        "seed": context.config.seed,
        "capacity_scenario": scenario_name,
        "capacity_multiplier": capacity_multiplier,
        "policy_evaluation_version": POLICY_EVALUATION_VERSION,
        "fused_threshold": context.trained.fused_threshold,
        "fused_threshold_note": (
            "The last (most-recent-history) rolling-origin fold's threshold; "
            "see 'scoring_coverage'.folds for every fold's own threshold -- "
            "each fold is scored by its own model/threshold, never a single "
            "shared one."
        ),
        "scoring_coverage": context.coverage,
        "policies": results,
        "current_policy_decision_log_sample": decision_log_rows[:200],
        "current_policy_decision_log_count": len(decision_log_rows),
    }


def evaluate_policies_across_capacity_scenarios(context: SeedContext) -> dict[str, Any]:
    """Evaluate all eight policies at every pre-specified capacity-stress
    scenario in ``resources.CAPACITY_SCENARIOS`` (25%/50%/100% of default
    capacity): the same eligible pool, the same priorities, the same
    common-random-number potential outcomes, and the same *multiplier*
    applied uniformly to every resource pool for every policy -- only the
    realized capacity differs by scenario, never by policy.

    ``PRIMARY_CAPACITY_SCENARIO`` (50% capacity) is Stage 1's headline; the
    25%/100% scenarios are always computed and reported too, as
    sensitivity/diagnostic context that is never hidden, including on any
    scenario/seed where ``CURRENT_POLICY`` does not win.
    """
    scenarios: dict[str, Any] = {
        scenario_name: evaluate_policies(
            context, capacity_multiplier=multiplier, scenario_name=scenario_name
        )
        for scenario_name, multiplier in CAPACITY_SCENARIOS.items()
    }
    return {
        "primary_capacity_scenario": PRIMARY_CAPACITY_SCENARIO,
        "diagnostic_capacity_scenario": DIAGNOSTIC_CAPACITY_SCENARIO,
        "capacity_scenarios": dict(CAPACITY_SCENARIOS),
        "scenarios": scenarios,
    }


def bayesian_ablation_diagnostic(
    context: SeedContext, *, capacity_scenario: str = PRIMARY_CAPACITY_SCENARIO
) -> dict[str, Any]:
    """Compare ``POLICY_CURRENT`` with vs. without its Bayesian structural-reduction
    term, holding every other formula component, the eligible pool, and the capacity
    scenario fixed -- an honest ablation of whether the persisted Bayesian
    intervention scenarios add measurable value over the leading-signal-only
    fallback. A negative/zero delta is reported as-is, never hidden or re-tuned away.
    """
    multiplier = CAPACITY_SCENARIOS[capacity_scenario]
    calendar_days = evaluation_calendar_days(context)
    capacity_schedule = build_capacity_schedule(context.dataset, calendar_days, multiplier)
    base_capacities = default_daily_capacities(context.dataset)
    responses_indexed = context.responses.set_index(["order_id", "action_code"])

    def _run(use_bayesian: bool) -> dict[str, Any]:
        table, ledger, candidate_log, _ = _treatment_table(
            POLICY_CURRENT,
            context.decisions,
            responses_indexed,
            context.responses,
            context.dataset,
            capacity_schedule,
            seed=context.config.seed,
            capacity_scenario=capacity_scenario,
            base_capacities=base_capacities,
            use_bayesian=use_bayesian,
        )
        return _summarize(table, ledger, candidate_log)

    with_bayesian = _run(True)
    without_bayesian = _run(False)
    headline = "avoided_penalty_per_normalized_resource_unit"
    delta = round(with_bayesian[headline] - without_bayesian[headline], 4)
    return {
        "capacity_scenario": capacity_scenario,
        "with_bayesian_term": with_bayesian,
        "without_bayesian_term": without_bayesian,
        "headline_delta_with_minus_without": delta,
        "bayesian_term_adds_value": bool(delta > 0),
        "qualification": (
            "Ablation of POLICY_CURRENT's structural_reduction term only -- everything "
            "else (eligible pool, exploration, execution feasibility, capacity, "
            "resource ranking) is identical between the two runs. Not required to "
            "favor the Bayesian term; reported honestly either way."
        ),
    }


def expected_vs_realized_rank_correlation(
    context: SeedContext, *, use_bayesian: bool = True
) -> dict[str, Any]:
    """Evaluation-only diagnostic (never used to make or influence any decision):
    Spearman rank correlation between ``POLICY_CURRENT``'s expected value-density
    priority score and each candidate's *realized* avoided-penalty-per-normalized-
    resource-unit from the twin's own common-random-number potential outcomes.

    Computed over the *full* eligible candidate population (not just capacity-
    accepted orders), so capacity acceptance/exploration selection never distorts
    this ranking-quality check; capacity-scenario-independent, since
    ``POLICY_CURRENT``'s priority score already is (see ``_expected_value_density``).
    """
    base_capacities = default_daily_capacities(context.dataset)
    candidates = _candidate_frame(
        POLICY_CURRENT,
        context.decisions,
        context.responses,
        context.config.seed,
        base_capacities=base_capacities,
        use_bayesian=use_bayesian,
    )
    if candidates.empty:
        return {
            "n_candidates": 0,
            "spearman_rank_correlation": None,
            "qualification": "no eligible POLICY_CURRENT candidates in this seed.",
        }

    responses_indexed = context.responses.set_index(["order_id", "action_code"])
    realized_densities: list[float] = []
    for _, row in candidates.iterrows():
        key = (row["order_id"], row["action_code"])
        avoided = (
            float(responses_indexed.loc[key, "avoided_penalty"])
            if key in responses_indexed.index
            else 0.0
        )
        resource_type = row["resource_type"]
        base_capacity = base_capacities.pool(resource_type).get(row["resource_id"], 0.0)
        demand = _demand_units(resource_type, float(row.get("quantity_at_risk", 0.0)))
        normalized_fraction = demand / base_capacity if base_capacity > 0 else demand
        realized_densities.append(
            avoided / max(normalized_fraction, VALUE_DENSITY_MIN_RESOURCE_FRACTION)
        )

    frame = pd.DataFrame(
        {
            "expected_density": candidates["priority_score"].to_numpy(),
            "realized_density": realized_densities,
        }
    )
    correlation = frame["expected_density"].corr(frame["realized_density"], method="spearman")
    return {
        "n_candidates": int(len(frame)),
        "spearman_rank_correlation": None if pd.isna(correlation) else round(float(correlation), 4),
        "qualification": (
            "Evaluation-only diagnostic: compares the decision-time expected "
            "value-density ranking against the twin's own realized potential outcome "
            "for the same chosen action. Never used to make, filter, or revise any "
            "decision -- the twin's realized outcomes are only ever read here, after "
            "the fact, for measurement."
        ),
    }


def counterfactual_action_ranking(context: SeedContext) -> dict[str, Any]:
    """Compare the Bayesian structural top intervention against the twin's
    counterfactually best feasible action, the strongest-signal heuristic,
    and a random feasible action -- action-ranking-only, capacity-agnostic.

    Restricted to orders with at least one active leading signal (a
    plausible actionable cause) so "no cause detected" orders do not dilute
    agreement rates. ``bayesian_action`` (derived from a **model-scenario**
    ranking of the fixed Bayesian network's do-operator scenarios) is
    explicitly kept distinct from the twin's **simulator-evaluation**
    ``best_action`` -- one is not used to validate the other's existence,
    only their agreement rate and value regret are reported.
    """
    from .pipeline import _top_intervention_cause

    decisions = context.decisions
    action_rows = context.responses.loc[context.responses["action_code"] != NO_ACTION]
    best_idx = action_rows.groupby("order_id")["avoided_penalty"].idxmax()
    best = action_rows.loc[best_idx].set_index("order_id")

    candidates = decisions.loc[decisions.get("active_leading_signal_count", 0) > 0].copy()
    if "intervention_scenarios_json" not in candidates.columns:
        return {
            "note": "intervention_scenarios_json not present on scored frame; diagnostics skipped.",
            "n_orders": 0,
        }
    candidates["bayesian_top_cause"] = candidates["intervention_scenarios_json"].map(
        _top_intervention_cause
    )
    candidates["bayesian_top_action"] = candidates["bayesian_top_cause"].map(CAUSE_TO_ACTION)
    candidates["strongest_signal_action"] = candidates["primary_cause"].map(CAUSE_TO_ACTION)

    action_values = action_rows.set_index(["order_id", "action_code"])["avoided_penalty"]

    def _value_for(order_id: str, action: str | None) -> float:
        if action is None:
            return 0.0
        try:
            return float(action_values.loc[(order_id, action)])
        except KeyError:
            return 0.0

    rows = []
    for _, row in candidates.iterrows():
        order_id = row["order_id"]
        if order_id not in best.index:
            continue
        best_row = best.loc[order_id]
        best_action = best_row["action_code"]
        best_value = float(best_row["avoided_penalty"])

        random_draws = deterministic_uniforms(
            context.config.seed, order_id, "ranking::random_action"
        )
        random_action = ACTIONS[int(random_draws[0] * len(ACTIONS)) % len(ACTIONS)]

        bayesian_action = row["bayesian_top_action"]
        signal_action = row["strongest_signal_action"]
        rows.append(
            {
                "order_id": order_id,
                "best_action": best_action,
                "best_value": best_value,
                "bayesian_action": bayesian_action,
                "bayesian_value": _value_for(order_id, bayesian_action),
                "signal_action": signal_action,
                "signal_value": _value_for(order_id, signal_action),
                "random_action": random_action,
                "random_value": _value_for(order_id, random_action),
                "bayesian_agrees": bayesian_action == best_action,
                "signal_agrees": signal_action == best_action,
                "random_agrees": random_action == best_action,
            }
        )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return {"note": "no orders with an active leading signal in this seed.", "n_orders": 0}

    def _agreement(column: str) -> float:
        return round(float(diagnostics[column].mean()), 4)

    def _regret(value_column: str) -> float:
        return round(float((diagnostics["best_value"] - diagnostics[value_column]).mean()), 4)

    return {
        "n_orders": int(len(diagnostics)),
        "top_action_agreement": {
            "bayesian_vs_best": _agreement("bayesian_agrees"),
            "strongest_signal_vs_best": _agreement("signal_agrees"),
            "random_vs_best": _agreement("random_agrees"),
        },
        "mean_value_regret": {
            "bayesian_vs_best": _regret("bayesian_value"),
            "strongest_signal_vs_best": _regret("signal_value"),
            "random_vs_best": _regret("random_value"),
        },
        "bayesian_adds_value_over_strongest_signal": bool(
            _regret("bayesian_value") < _regret("signal_value")
        ),
        "bayesian_adds_value_over_random": bool(
            _regret("bayesian_value") < _regret("random_value")
        ),
        "qualification": (
            "bayesian_action is a fixed-structure Bayesian do-operator scenario ranking "
            "(model-scenario), never itself a simulator-evaluation outcome; best_action is "
            "the twin's own counterfactually-best potential outcome (simulator-evaluation). "
            "Agreement/regret compare the two -- the Bayesian ranking is not assumed to win."
        ),
    }


def run_seed_evaluation(config: PrototypeConfig) -> dict[str, Any]:
    """Build one seed's context and run the capacity-scenario policy
    evaluation (25%/50%/100% capacity -- see
    ``evaluate_policies_across_capacity_scenarios``), the counterfactual
    action-ranking diagnostics, the value-aware-policy Bayesian ablation and
    expected-vs-realized rank-correlation diagnostics, plus a content fingerprint.
    """
    context = build_seed_context(config)
    capacity_sensitivity = evaluate_policies_across_capacity_scenarios(context)
    # Preserved for continuity with pre-sensitivity-analysis reports: the
    # unscaled 100%-capacity scenario, under the ``policy_evaluation`` key
    # every earlier consumer of this report already expects -- a diagnostic
    # now, never the acceptance-gate headline (see
    # ``PRIMARY_CAPACITY_SCENARIO``/``summarize_multi_seed``).
    policy_results = capacity_sensitivity["scenarios"][DIAGNOSTIC_CAPACITY_SCENARIO]
    ranking = counterfactual_action_ranking(context)
    fingerprint = content_fingerprint(
        {
            "config": {
                "seed": config.seed,
                "n_orders": config.n_orders,
                "start_date": config.start_date,
                "scenario_order_count": config.scenario_order_count,
            },
            "policy_evaluation_version": POLICY_EVALUATION_VERSION,
        }
    )
    return {
        "seed": config.seed,
        "n_orders": config.n_orders,
        "evaluation_fingerprint": fingerprint,
        "policy_evaluation": policy_results,
        "capacity_sensitivity": capacity_sensitivity,
        "counterfactual_action_ranking": ranking,
        "bayesian_ablation": bayesian_ablation_diagnostic(context),
        "expected_vs_realized_rank_correlation": expected_vs_realized_rank_correlation(context),
        "no_action_identity_gate": _no_action_identity_gate(context),
    }


def _no_action_identity_gate(context: SeedContext) -> dict[str, Any]:
    """Confirm ``NO_ACTION`` potential outcomes exactly reproduce the twin's
    original outcomes for every order in this seed (a hard Stage 1 gate).
    """
    no_action = context.responses.loc[context.responses["action_code"] == NO_ACTION].set_index(
        "order_id"
    )
    original = context.outcomes.set_index("order_id")
    joined = no_action.join(original, rsuffix="_original")
    timestamp_matches = bool(
        (joined["delivered_timestamp"] == joined["delivered_timestamp_original"]).all()
    )
    qty_matches = bool(
        (joined["delivered_qty"].round(6) == joined["delivered_qty_original"].round(6)).all()
    )
    otif_matches = bool((joined["otif_miss"] == joined["otif_miss_original"]).all())
    return {
        "delivered_timestamp_identical": timestamp_matches,
        "delivered_qty_identical": qty_matches,
        "otif_miss_identical": otif_matches,
        "passed": bool(timestamp_matches and qty_matches and otif_matches),
    }


def _scenario_metric(
    seed_reports: list[dict[str, Any]], scenario: str, policy: str, metric: str
) -> list[float]:
    return [
        report["capacity_sensitivity"]["scenarios"][scenario]["policies"][policy][metric]
        for report in seed_reports
    ]


def _paired_seed_deltas(
    seed_reports: list[dict[str, Any]],
    scenario: str,
    policy_a: str,
    policy_b: str,
    metric: str = "avoided_penalty_per_normalized_resource_unit",
    *,
    tie_tolerance: float = CAPACITY_SCENARIO_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Per-seed ``policy_a - policy_b`` deltas at ``scenario`` plus
    win/tie/loss counts (``policy_a`` wins/ties/loses per seed).

    A tie is any ``|delta| <= tie_tolerance`` -- a small, fixed, documented
    rounding-noise guard (metrics are already rounded to 4 decimal places),
    not a manufactured significance threshold: no seed is ever reclassified
    after the fact, and the tolerance is identical for every seed/scenario.
    """
    values_a = _scenario_metric(seed_reports, scenario, policy_a, metric)
    values_b = _scenario_metric(seed_reports, scenario, policy_b, metric)
    deltas = [round(a - b, 6) for a, b in zip(values_a, values_b, strict=True)]
    wins = sum(1 for delta in deltas if delta > tie_tolerance)
    losses = sum(1 for delta in deltas if delta < -tie_tolerance)
    ties = len(deltas) - wins - losses
    return {
        "capacity_scenario": scenario,
        "metric": metric,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "seeds": [report["seed"] for report in seed_reports],
        "per_seed_delta": deltas,
        "tie_tolerance": tie_tolerance,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def summarize_multi_seed(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate several ``run_seed_evaluation`` reports into per-capacity-
    scenario medians, paired per-seed deltas, and acceptance gates.

    Every metric is reported at all three pre-specified capacity scenarios
    (``resources.CAPACITY_SCENARIOS``); acceptance gates are measured at
    ``PRIMARY_CAPACITY_SCENARIO`` (50% capacity) -- the base-capacity (100%)
    comparisons are retained as ``diagnostic_*`` gates for continuity, never
    as the headline (see the module docstring).
    """
    primary = PRIMARY_CAPACITY_SCENARIO
    diagnostic = DIAGNOSTIC_CAPACITY_SCENARIO

    def _median_by_policy(scenario: str, metric: str) -> dict[str, float]:
        return {
            policy: round(
                statistics.median(_scenario_metric(seed_reports, scenario, policy, metric)), 4
            )
            for policy in POLICIES
        }

    median_headline_by_scenario: dict[str, dict[str, float]] = {}
    median_action_precision_by_scenario: dict[str, dict[str, float]] = {}
    median_regret_by_scenario: dict[str, dict[str, float]] = {}
    median_avoidable_miss_coverage_by_scenario: dict[str, dict[str, float]] = {}
    median_contested_rate_by_scenario: dict[str, dict[str, float]] = {}
    median_capacity_binding_rate_by_scenario: dict[str, dict[str, float]] = {}

    for scenario in CAPACITY_SCENARIOS:
        median_headline_by_scenario[scenario] = _median_by_policy(
            scenario, "avoided_penalty_per_normalized_resource_unit"
        )
        median_action_precision_by_scenario[scenario] = _median_by_policy(
            scenario, "action_precision"
        )
        median_regret_by_scenario[scenario] = _median_by_policy(scenario, "regret_vs_oracle")
        median_avoidable_miss_coverage_by_scenario[scenario] = _median_by_policy(
            scenario, "avoidable_miss_coverage"
        )
        median_contested_rate_by_scenario[scenario] = _median_by_policy(scenario, "contested_rate")
        median_capacity_binding_rate_by_scenario[scenario] = _median_by_policy(
            scenario, "capacity_binding_rate"
        )

    def _median_metric_by_regime(policy: str, regime: str, scenario: str) -> float:
        values = [
            report["capacity_sensitivity"]["scenarios"][scenario]["policies"][policy][
                "value_by_regime"
            ].get(regime, 0.0)
            for report in seed_reports
        ]
        return round(statistics.median(values), 4) if values else 0.0

    paired_vs_random = _paired_seed_deltas(seed_reports, primary, POLICY_CURRENT, POLICY_RANDOM)
    paired_vs_highest_risk = _paired_seed_deltas(
        seed_reports, primary, POLICY_CURRENT, POLICY_HIGHEST_RISK
    )
    paired_vs_legacy = _paired_seed_deltas(seed_reports, primary, POLICY_CURRENT, POLICY_LEGACY)

    primary_median = median_headline_by_scenario[primary]
    diagnostic_median = median_headline_by_scenario[diagnostic]
    primary_action_precision = median_action_precision_by_scenario[primary]

    n_seeds = len(seed_reports)
    # "At least 3/5 non-tie wins" generalizes to any seed count as
    # ceil(3/5 * n_seeds) -- 3 when n_seeds == 5, the benchmark's default.
    win_threshold = math.ceil(n_seeds * 3 / 5) if n_seeds else 0

    gates: dict[str, Any] = {
        "no_action_identity": all(
            report["no_action_identity_gate"]["passed"] for report in seed_reports
        ),
        "primary_capacity_scenario": primary,
        "win_threshold": win_threshold,
        # Primary Stage 1 gate: measured at 50% capacity, where capacity is
        # actually scarce enough to be discriminative between policies (see
        # module docstring). This is the gate that determines pass/fail.
        "current_beats_random_at_primary_capacity": bool(
            primary_median[POLICY_CURRENT] > primary_median[POLICY_RANDOM]
        ),
        "current_beats_highest_risk_at_primary_capacity": bool(
            primary_median[POLICY_CURRENT] > primary_median[POLICY_HIGHEST_RISK]
        ),
        "current_beats_legacy_at_primary_capacity": bool(
            primary_median[POLICY_CURRENT] > primary_median[POLICY_LEGACY]
        ),
        "current_wins_at_least_win_threshold_vs_random": bool(
            paired_vs_random["wins"] >= win_threshold
        ),
        "current_wins_at_least_win_threshold_vs_legacy": bool(
            paired_vs_legacy["wins"] >= win_threshold
        ),
        "current_policy_value_by_regime": {
            "normal": _median_metric_by_regime(POLICY_CURRENT, "normal", primary),
            "drift": _median_metric_by_regime(POLICY_CURRENT, "drift", primary),
        },
        "n_seeds": n_seeds,
        # Base-capacity (100%) comparisons: retained for continuity with
        # pre-sensitivity-analysis reports, but diagnostic only -- this
        # twin's default capacities are rarely binding at 100%, so these
        # gates are not a reliable test of whether the ranking matters (see
        # module docstring). Never the pass/fail headline.
        "diagnostic_current_beats_random_at_base_capacity": bool(
            diagnostic_median[POLICY_CURRENT] > diagnostic_median[POLICY_RANDOM]
        ),
        "diagnostic_current_beats_highest_risk_at_base_capacity": bool(
            diagnostic_median[POLICY_CURRENT] > diagnostic_median[POLICY_HIGHEST_RISK]
        ),
        "diagnostic_current_beats_legacy_at_base_capacity": bool(
            diagnostic_median[POLICY_CURRENT] > diagnostic_median[POLICY_LEGACY]
        ),
    }
    gates["current_policy_value_positive_in_both_regimes"] = bool(
        gates["current_policy_value_by_regime"]["normal"] >= 0
        and gates["current_policy_value_by_regime"]["drift"] >= 0
    )
    # No action-precision collapse: CURRENT_POLICY's action precision must not fall
    # more than 5 percentage points below HIGHEST_RISK_AT_CAPACITY's at the primary
    # capacity scenario (a value-aware ranking that spends capacity on much lower-
    # precision actions than a naive risk-only ranking would be a regression, even if
    # its headline density is higher).
    action_precision_gap = round(
        primary_action_precision[POLICY_HIGHEST_RISK] - primary_action_precision[POLICY_CURRENT],
        4,
    )
    gates["action_precision_gap_vs_highest_risk"] = action_precision_gap
    gates["no_action_precision_collapse"] = bool(action_precision_gap <= 0.05)
    gates["primary_gate_passed"] = bool(
        gates["current_beats_random_at_primary_capacity"]
        and gates["current_beats_highest_risk_at_primary_capacity"]
        and gates["current_beats_legacy_at_primary_capacity"]
        and gates["current_wins_at_least_win_threshold_vs_random"]
        and gates["current_wins_at_least_win_threshold_vs_legacy"]
        and gates["current_policy_value_positive_in_both_regimes"]
        and gates["no_action_precision_collapse"]
    )
    ranking_gate_values = [
        report["counterfactual_action_ranking"].get("bayesian_adds_value_over_random")
        for report in seed_reports
        if "bayesian_adds_value_over_random" in report["counterfactual_action_ranking"]
    ]
    gates["bayesian_adds_value_over_random_in_majority_of_seeds"] = (
        bool(
            sum(bool(value) for value in ranking_gate_values)
            >= (len(ranking_gate_values) + 1) // 2
        )
        if ranking_gate_values
        else False
    )

    def _median_current_metric(scenario: str, metric: str) -> float | None:
        values = [
            report["capacity_sensitivity"]["scenarios"][scenario]["policies"][POLICY_CURRENT].get(
                metric
            )
            for report in seed_reports
        ]
        values = [value for value in values if value is not None]
        return round(statistics.median(values), 4) if values else None

    def _median_action_mix(scenario: str, metric: str) -> dict[str, float]:
        mixes = [
            report["capacity_sensitivity"]["scenarios"][scenario]["policies"][POLICY_CURRENT].get(
                metric, {}
            )
            for report in seed_reports
        ]
        actions = sorted({action for mix in mixes for action in mix})
        return {
            action: round(statistics.median([mix.get(action, 0.0) for mix in mixes]), 4)
            for action in actions
        }

    correlations = [
        report["expected_vs_realized_rank_correlation"]["spearman_rank_correlation"]
        for report in seed_reports
        if report.get("expected_vs_realized_rank_correlation", {}).get(
            "spearman_rank_correlation"
        )
        is not None
    ]

    def _ablation_metric(key: str) -> list[float]:
        return [
            report["bayesian_ablation"][key]["avoided_penalty_per_normalized_resource_unit"]
            for report in seed_reports
            if "bayesian_ablation" in report
        ]

    ablation_with = _ablation_metric("with_bayesian_term")
    ablation_without = _ablation_metric("without_bayesian_term")
    ablation_deltas = [
        report["bayesian_ablation"]["headline_delta_with_minus_without"]
        for report in seed_reports
        if "bayesian_ablation" in report
    ]
    bayesian_ablation_summary = {
        "capacity_scenario": primary,
        "median_with_bayesian_term": (
            round(statistics.median(ablation_with), 4) if ablation_with else None
        ),
        "median_without_bayesian_term": (
            round(statistics.median(ablation_without), 4) if ablation_without else None
        ),
        "median_delta_with_minus_without": (
            round(statistics.median(ablation_deltas), 4) if ablation_deltas else None
        ),
        "seeds_where_bayesian_term_adds_value": sum(
            1
            for report in seed_reports
            if report.get("bayesian_ablation", {}).get("bayesian_term_adds_value")
        ),
        "n_seeds": len(ablation_with),
    }

    value_aware_policy_diagnostics = {
        "candidate_action_coverage_median": _median_current_metric(
            primary, "candidate_action_coverage"
        ),
        "bayesian_evidence_rate_median": _median_current_metric(primary, "bayesian_evidence_rate"),
        "mean_candidate_actions_per_order_median": _median_current_metric(
            primary, "mean_candidate_actions_per_order"
        ),
        "candidate_action_mix_median": _median_action_mix(primary, "candidate_action_mix"),
        "chosen_action_mix_median": _median_action_mix(primary, "chosen_action_mix"),
        "expected_vs_realized_rank_correlation_median": (
            round(statistics.median(correlations), 4) if correlations else None
        ),
        "bayesian_ablation": bayesian_ablation_summary,
    }

    return {
        "seeds": [report["seed"] for report in seed_reports],
        "primary_capacity_scenario": primary,
        "diagnostic_capacity_scenario": diagnostic,
        "capacity_scenarios": dict(CAPACITY_SCENARIOS),
        "median_headline_by_capacity_scenario": median_headline_by_scenario,
        "median_action_precision_by_capacity_scenario": median_action_precision_by_scenario,
        "median_regret_vs_oracle_by_capacity_scenario": median_regret_by_scenario,
        "median_avoidable_miss_coverage_by_capacity_scenario": (
            median_avoidable_miss_coverage_by_scenario
        ),
        "median_contested_rate_by_capacity_scenario": median_contested_rate_by_scenario,
        "median_capacity_binding_rate_by_capacity_scenario": (
            median_capacity_binding_rate_by_scenario
        ),
        "paired_seed_deltas": {
            "current_vs_random": paired_vs_random,
            "current_vs_highest_risk": paired_vs_highest_risk,
            "current_vs_legacy": paired_vs_legacy,
        },
        "value_aware_policy_diagnostics": value_aware_policy_diagnostics,
        "acceptance_gates": gates,
    }
