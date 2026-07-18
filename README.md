# OTIF Risk Intelligence

An explainable supply-chain control-tower prototype that predicts open orders at risk
of missing On-Time-In-Full delivery, identifies likely contributing factors down to the
affected SKU/line, recommends resource-aware mitigation actions, replays a local
daily operating loop (scoring, closures, drift detection, versioned retraining), and offers
a read-only, cited **AI Copilot** (deterministic fallback plus optional live OpenAI) that
explains and drafts but never decides.

- **Architecture**: [`docs/architecture/current.mmd`](docs/architecture/current.mmd) /
  [`docs/architecture/current.svg`](docs/architecture/current.svg)
- **Model card** (measured benchmark numbers, honest limitations): [`docs/model-card.md`](docs/model-card.md)
- **Judge-facing demo script**: [`docs/demo-script.md`](docs/demo-script.md)

The SVG is rendered by a small, checked-in, deterministic Python renderer
(`docs/architecture/diagrams.py` + `docs/architecture/generate.py`) — standard-library
only, no Mermaid CLI or other internet tooling installed or invoked. Regenerate with
`uv run python docs/architecture/generate.py`.

## Design principles

- One risk score and one intervention decision per order, with affected-SKU/line
  evidence and a compact causal pathway underneath that single decision.
- A **noisy, partially-observable digital twin** replaces deterministic synthetic
  separability: outcomes fall out of accumulated per-stage delay/shortfall, never
  pre-selected; safe orders can show warning signals, and missed orders can have weak or
  missing early evidence.
- Features must be available at an explicit `as_of_timestamp` — the same contract used
  for historical snapshots and for daily open-order scoring.
- Fusion weight and decision threshold are selected on **validation only**, with a
  fixed 10%-increment weight grid (no stacking model) and a Brier-plus-recall-guardrail
  selection rule.
- The benchmark target is a **range, not a tuning objective**: 5 fixed seeds, reported
  as median/range; seeds outside range are reported, not adjusted away.

## Architecture

1. **Digital twin** (`data.py`): stable vendor/SKU/DC/lane/customer traits, seasonality,
   correlated disruption shocks, partial observability (missing events, attenuated
   exception logging), measurement noise, and five deterministic named scenarios
   (multi-cause propagation, two orders contesting shared recovery capacity, a
   line-level stockout, and a genuine unexplained miss) reserved regardless of seed.
   Simulator truth (`simulator_truth.csv`, `line_truth.csv`, `shocks.csv`) is persisted
   separately from the model-facing tables and never used as a feature.
2. Fail-fast schema/referential/logical validation (`validation.py`), extended to the
   new `skus` table.
3. Seven-category retrospective multi-cause derivation (`root_causes.py`) from the
   *final* resolved outcome — used for training labels and evaluation, never as a
   feature.
4. Point-in-time order features (`features.py`) with an explicit `as_of_timestamp`
   contract, matured 30/90-day/all-time rolling vendor/DC/lane/customer/SKU history,
   freshness/missingness/remaining-slack/DC-utilization-trend features, and a
   timestamp-*group* chronological split (identical timestamps never straddle a
   train/validation/test boundary).
5. Per-line/SKU evidence (`line_evidence.py`) from capture-time-safe fields only
   (initial ATP allocation, inventory snapshot, SKU criticality, the order's own
   point-in-time vendor-exception signal), aggregated into safe order-level features
   (`worst_line_shortage_ratio`, `affected_line_count`, `critical_sku_share`, line
   quantity concentration) and evaluated against simulator truth.
6. Calibrated XGBoost OTIF-risk model (`model.py`; OpenMP-backed on macOS via Homebrew
   `libomp`) plus SHAP explanations with a deterministic perturbation fallback
   (`explain.py`).
7. A 10-node **mechanism Bayesian network** (`bayesian.py`) matching the actual OTIF
   definition -- "on time" AND "in full" -- instead of one flat seven-cause star:
   `ORDER_CAPTURE→LATE_DELIVERY`, `VENDOR_FAILURE→INVENTORY_SHORTAGE`,
   `{INVENTORY_SHORTAGE, DC_CAPACITY}→WAREHOUSE_OPS`, `WAREHOUSE_OPS→TRANSPORT`,
   `INVENTORY_SHORTAGE→IN_FULL_FAILURE`,
   `{ORDER_CAPTURE, WAREHOUSE_OPS, TRANSPORT, CUSTOMER_DELIVERY}→LATE_DELIVERY`,
   `{IN_FULL_FAILURE, LATE_DELIVERY}→OTIF_MISS`. `IN_FULL_FAILURE`/`LATE_DELIVERY` are
   fit directly from each training order's own `1 - in_full` / `1 - on_time` outcome, not
   inferred from the seven-category failure-only cause labels. CPTs are smoothed counts
   fit on training-only operational stage history, including disruptions absorbed by
   orders that still achieved OTIF; a node is only given as hard evidence once its stage
   has actually been observed as of the as-of timestamp, so unobserved intermediate
   stages (including the two mechanism nodes themselves, which are never directly
   observable before an order closes) are marginalized out via exact inference rather
   than assumed absent. Exact inference uses `pgmpy` variable elimination when available,
   or a numerically-identical brute-force joint enumeration over this small 10-node
   network otherwise (see "Bayesian inference mode" below).
8. **Structural intervention scenarios and causal attribution** (`bayesian.py`): for
   every scored order, `BayesianBundle.intervene` computes exact
   `P(OTIF_MISS | evidence, do(node=value))` under the fixed network -- a genuine
   do-operator computation (an intervened node's fitted CPT is replaced by a fixed
   value, severing its parents' influence on it) always via brute-force enumeration,
   never through the pgmpy observational-query path. Only operational cause nodes may be
   intervened on; invalid nodes/values are rejected. Every scored order gets a do(node=0)
   scenario for each of its active evidence nodes plus one combined-mitigation scenario,
   each reporting baseline/post-intervention Bayesian posteriors, absolute/relative risk
   reduction, the mechanism route(s), the assumed operational action, and an explicit
   "Fixed-structure scenario analysis -- not a proven treatment effect" qualification.
   Leave-one-evidence-out **evidence attribution** reports how much each active cause
   node's posterior contribution would change if it were withheld (marginalized) instead
   of conditioned on -- explicitly labeled `evidence_attribution_leave_one_out`, not SHAP
   and not a causal-effect estimate. Neither interventions nor attribution ever feed the
   XGBoost score, the fused score, or the operational decision; they are persisted
   diagnostics (`causal_attribution_json`, `intervention_scenarios_json`,
   `causal_confidence`, `evidence_coverage`, `late_delivery_probability`,
   `in_full_failure_probability`) surfaced only in the Causal Intelligence Studio view.
9. Evidence-based fusion (`fusion.py`): compares XGBoost-only, Bayesian-only, the fixed
   70/30 blend, and every other convex weight in 10% increments on validation, selecting
   a blend within 0.002 Brier score of the best eligible candidate under a
   fixed-capacity recall guardrail, then preferring more Bayesian contribution among
   practically equivalent candidates (no stacking model). The operating threshold is
   tuned separately for the chosen weight. See
   "Fusion weight selection" below.
10. Generic resource-aware interventions (`decisions.py` / `resources.py`): a lookup-table
    mitigation policy plus a capacity-aware conflict check (DC recovery units, lane
    alternate capacity, vendor escalation slots, customer appointment slots), greedily
    allocated by priority; overflow is marked `CONTESTED` with the competing orders
    listed (`contested_with`).
11. Vendor, DC, lane, customer, order-type, and SKU rollups plus service-impact
    assumptions (`decisions.py`).
12. Templated, structured planner narratives (risk → evidence → pathway → affected SKUs
    → action → resource status, optionally noting the dominant mechanism and
    highest-potential structural intervention; `narratives.py`), append-only CSV feedback
    (`feedback.py`), and a nine-view Streamlit control tower (`app.py`) -- including the
    **Causal Intelligence Studio** and **AI Copilot** views -- that reuses only persisted
    decisions.
13. A local daily **operations replay** (`operations.py`): trains an initial model on a
    historical window, then for each simulated day scores every still-open order as of
    that day, allocates daily resource capacities, persists the queue, closes resolved
    orders, derives their actual cause, appends feedback, computes drift (PSI,
    score-distribution shift, missingness change, recent OTIF-rate change), and retrains
    on a documented cadence or drift trigger — persisting a versioned model registry.
14. A multi-seed benchmark (`benchmark.py`) with explicit acceptance gates, including
    mechanism PR-AUC/Brier, evidence-coverage distribution, and low-confidence rate.
15. **Decision Value Lab** (`action_response.py`, `policy_evaluation.py`,
    `policy_benchmark.py`): a heterogeneous, probabilistic action-response digital twin
    plus capacity-constrained multi-policy evaluation that **measures** simulated
    intervention value instead of assuming a fixed effectiveness fraction. See
    "Decision Value Lab" below.
16. **Deterministic run manifests** (`manifest.py`): git SHA/dirty state, package and key
    dependency versions, normalized config/policy/schema versions, input-table
    row-count/schema/date-range/content-hash fingerprints, a feature-schema hash,
    training/validation/test windows, and SHA-256 checksums of every other artifact in
    the run — split into a **deterministic content ID** (ignores timestamps/run
    instance/output path; identical seed/config/code/model-facing artifacts always
    produce the same ID) and run-instance metadata. `verify_manifest` recomputes and
    reports checksum status (tamper-evident). See "Governance" below.
17. **Production-shaped source adapters and service contracts** (`adapters.py`,
    `service_contracts.py`): typed local-CSV ERP/WMS/TMS/SRM adapters that redact
    not-yet-occurred event timestamps and reconstruct a `PrototypeDataset`-compatible
    table set; a `ScoreRequest`/`ScoreResponse` contract; an idempotent
    upsert-by-decision-key JSONL/CSV `DecisionSink`; and a proven offline/batch parity
    guarantee (same order/as-of snapshot -> identical feature vector and score whether
    scored directly or through the adapter/service boundary). No web framework, database,
    or message broker is introduced.
18. **Decision/action/outcome ledger** (`decision_ledger.py`): an append/upsert,
    idempotent ledger of every operations-replay decision (feasible/chosen/rejected
    actions, risk/threshold, resource before/after, planner disposition, execution
    status) reconciled against matured OTIF outcomes, plus an **observational** cohort
    report (accepted/rejected/monitored miss rates and penalties, minimum-sample
    guarded, explicitly labeled `observational_not_causal`) — kept fully separate from
    the Decision Value Lab's exact, causally-interpretable potential-outcome policy
    value.
19. **Champion/challenger registry and promotion gate** (`registry.py`): immutable
    versions, append-only lifecycle events, an atomically-written `active_model.json`
    pointer, and `PromotionDecision` gates on PR-AUC/Brier/calibration/recall/alert-rate/
    drift-regime quality/policy value at 50% capacity/schema-leakage/manifest
    verification, with explicit tolerances. `PROMOTED`/`HELD`/`ROLLED_BACK` states;
    rollback only to a verified version. See "Governance" below.
20. **Rolling monitoring and SLO reporting** (`monitoring.py`): rolling-origin realized
    PR-AUC/precision/recall/calibration/alert-rate and normal-vs-drift-regime quality on
    matured ledger decisions (minimum-sample guarded), time-to-detection, feature
    freshness, measured **local-only** scoring/retrain runtime, and soft data-quality
    metrics (completeness, uniqueness, referential health, contract failures) with
    transparent SLO targets.
21. **Policy Value and Governance Streamlit views** (`app.py`): capacity-scenario
    selector with paired seed win/tie/loss, the value-density explanation, action mix,
    normal/drift value, and the Bayesian ablation shown honestly; a manifest trust card,
    lifecycle timeline, champion/challenger metric-delta table, decision ledger and
    observational cohorts, monitoring/SLO cards, and offline/batch parity status — all
    loaded from persisted artifacts, never recomputed in the UI.


## Point-in-time signals, leakage, and the digital twin

`leading_signal_*` is derived entirely in `features.py` from operational fields/events
already filtered to `event_timestamp <= as_of_timestamp` (vendor ready delay/exception,
warehouse/transport exceptions), from fields known at order capture
(`capture_delay_hours`, initial ATP allocation), from the DC capacity snapshot as of the
as-of date, or from customer master data known well in advance. None of these read the
generator's latent disruption cause directly.

The digital twin decouples "knowable at capture" from "true final outcome": a
capture-time inventory shortfall has a documented chance of being backfilled by an
expedited replenishment before shipment, so `stockout_flag`/`allocation_ratio` are
genuine *risk factors*, not certainties about the final shipped quantity.
`tests/test_features.py`
proves: (a) `leading_signal_*` is not present on the raw generator output, (b) it is not
a lossless proxy for the ground-truth cause, and (c) mutating *every* future
event/outcome cannot change an earlier as-of snapshot's row.

Cause consistency is evaluated only on held-out OTIF misses (successful orders have no
failure cause to recover). Bayesian CPTs use separate `stage_X` incident flags that are
recorded for every closed order, including disruptions that were absorbed without an
OTIF miss, plus each order's own `on_time`/`in_full` outcome for the two mechanism
nodes.

## Threshold and fusion-weight selection

1. Scores validation/test with XGBoost, Bayesian, and every fused weight on a
   0.0–1.0 grid in 0.1 increments.
2. Selects the fusion weight on **validation only**: candidates must satisfy the
   top-planner-capacity recall guardrail and fall within 0.002 Brier score of the best
   eligible candidate. Among those practically equivalent candidates, the policy
   prefers more Bayesian contribution. The comparison uses a fixed operating point,
   not each candidate's independently re-tuned threshold.
3. Tunes the final decision threshold separately, once, for the chosen weight, using the
   configured strategy (`recall_floor` by default) on the chosen weight's fused
   validation scores.
4. Persists the full comparison table (`fusion_comparison.csv`,
   `metrics.json.fusion_comparison`) and the chosen weight/label/rationale
   (`architecture.fusion_chosen_weight/label/fusion`), regardless of which candidate wins.

## Bayesian inference mode

The network is fit **only on the training split's resolved history**
(`bayesian_training_history`), matching the same chronological boundary enforced for the
risk model, and includes `on_time`/`in_full` so the two mechanism nodes
(`IN_FULL_FAILURE`, `LATE_DELIVERY`) are fit directly from that split's own resolved
outcomes. `pgmpy` exact inference is used when importable/constructible; when it is not,
a brute-force joint enumeration over the small 10-node binary network is used instead —
verified numerically identical to `pgmpy`'s result in
`tests/test_bayesian.py::test_brute_force_fallback_matches_pgmpy_exact_inference_for_every_query_node`.
Both are exact for ordinary observational queries; there is no approximate/empirical
fallback. `architecture.bayesian_inference_mode` records which one ran (`pgmpy_exact` or
`brute_force_exact`), with `architecture.bayesian_engine_build_error` set when `pgmpy` was
unavailable. **Structural interventions always use brute-force enumeration**, regardless
of which engine scored the observational query, because this prototype does not
implement/verify pgmpy's do-operator support (see
`tests/test_bayesian.py::test_intervention_severs_parent_influence_and_can_differ_from_conditioning`,
which shows a genuine do-vs-conditioning divergence at a collider node).

## Resource conflicts

`decisions.py`'s DC conflicts remain quantity/capacity aware (using
`dc_daily_capacity_units * dc_capacity_recovery_fraction`); vendor/lane/customer
conflicts remain count-based (documented assumption: no equivalent numeric
recovery-capacity field exists for those dimensions in this prototype). Every
`CONTESTED` order's `contested_with` column lists the competing order IDs. The
operations replay uses `resources.py`'s generalized daily-capacity engine (DC recovery
units, lane alternate capacity, vendor escalation slots, customer appointment slots,
each with a transparent demand unit), greedily allocated by priority and reset each
simulated day. Both are a deterministic priority-and-capacity policy, never a MILP
optimizer.

## Decision Value Lab (Stage 1: measuring intervention value)

`decisions.py`'s `estimated_avoidable_penalty` (a fixed 60% effectiveness assumption
applied to every recommended action) is a deployed-UI heuristic, not a measurement. The
Decision Value Lab is a separate, evaluation-only laboratory that **replays every
feasible intervention through the digital twin's own lifecycle mechanics** and measures
what actually would have happened, so "this action creates value" becomes falsifiable
rather than assumed.

**Heterogeneous action-response twin (`action_response.py`).** For every
`(seed, order_id, action_code)` the module draws a deterministic, common random number
(SHA-256-derived, independent of row/iteration order) so every policy under evaluation
sees the exact same potential outcome for the same order/action. Six actions are
modeled, one per feasible mitigation: `VENDOR_ESCALATION`, `INVENTORY_REALLOCATION`,
`WAREHOUSE_EXPEDITE` (covers both `DC_CAPACITY` and `WAREHOUSE_OPS`, since the twin
already adds both effects into the same `warehouse_delay_hours` mechanism),
`ALTERNATE_TRANSPORT`, `APPOINTMENT_COORDINATION`, and `ORDER_CAPTURE_CORRECTION`.
Response probability is a transparent weighted sum of five documented `[0, 1]`
components — action/cause match (0.40), intervention timing/slack (0.15), existing
stage severity (0.15), a stable resource-flexibility trait (0.15), and a stable
resource-availability signal (0.10) — clipped to `[0.02, 0.97]`; a successful action
reduces only its targeted stage delay or quantity shortfall, and the delivered
timestamp/quantity, on-time, in-full, OTIF-miss, and a realized penalty are recomputed
through the **same shared lifecycle helpers** the twin itself uses
(`data.recompute_lifecycle_timestamps`, `root_causes.compute_service_outcome`), so both
the twin and the evaluation lab share one set of equations, never a second approximate
model. A failed, mismatched attempt can also be
adverse (a small, documented chance the targeted mechanism gets slightly worse). Every
potential-outcome field is evaluation-only and is asserted, by test, to never enter
`features.build_feature_table`.

`ORDER_CAPTURE_CORRECTION` is a **documented structural no-op**: this twin's ship/transit
timing depends only on `order_date`, never `capture_delay_hours`, so correcting order
capture can never mechanically change the delivered timestamp/quantity — verified across
every seed tested (`tests/test_action_response.py`). This is reported as a genuine twin
limitation, not hidden.

**Capacity-constrained policy evaluation (`policy_evaluation.py`).** Every order is
scored via **rolling-origin, chronological cross-fitting**: the calendar is split into 5
equal chronological folds; the first is warm-up history (excluded from evaluation
entirely — there is no prior data yet to train a model on); every later fold is scored
only by a model trained, and its fusion weight/threshold selected, on history strictly
before that fold (its own internal 75/25 train/validation split), so every evaluated
order is genuinely out-of-sample and no fold's own rows ever influence its model.
Coverage (~80% of the calendar; the excluded warm-up fraction is reported per run) and
every fold's own threshold are in `scoring_coverage` of each evaluation report. Eight
policies are then compared against the identical out-of-sample order population, the
identical common-random-number potential outcomes, and, at every one of three
pre-specified **capacity-stress scenarios** (see below), the identical daily resource
capacities (`resources.build_capacity_schedule` / the same `allocate_under_capacity`
engine `operations.py`'s daily replay already uses — no new optimizer): `NO_ACTION`,
`RANDOM_AT_CAPACITY`, `HIGHEST_RISK_AT_CAPACITY`, `HIGHEST_FINANCIAL_AT_CAPACITY`,
`STRONGEST_SIGNAL_HEURISTIC`, `SINGLE_CAUSE_PRIORITY_BASELINE` (the transparent
`recommend_orders` risk×tier×value ranking, with the single `CAUSE_TO_ACTION`-implied
action — kept unchanged as a deployable baseline), `CURRENT_POLICY` (the **value-aware**
deployable policy described below, with the same ~10%-of-capacity seeded-exploration
carve-out among near-equal eligible orders and a full per-decision log — assignment
probability, selection mode, policy version, capacity before/after, chosen action,
rejected feasible actions, and an idempotency decision key), and `ORACLE_EVALUATION_ONLY`
(an evaluation-only, unattainable ceiling that may choose any action, used solely to
compute regret — never a deployable recommendation). The six non-`NO_ACTION`/non-oracle
policies share the exact same eligible pool (`combined_risk_score >= fused_threshold`
and `primary_cause` mapped by `CAUSE_TO_ACTION`), differing only in *which* eligible
orders get scarce capacity first and, for `CURRENT_POLICY` alone, *which* feasible action
each order is assigned.

### Value-aware `CURRENT_POLICY`: what changed and why

`SINGLE_CAUSE_PRIORITY_BASELINE` (`decisions.recommend_orders`'s risk×tier×value
`priority_score`) and three of the other baselines (`HIGHEST_RISK_AT_CAPACITY`,
`HIGHEST_FINANCIAL_AT_CAPACITY`, `RANDOM_AT_CAPACITY`, when the fused-threshold-gated
pool is small relative to capacity) frequently rank orders identically or accept every
eligible order regardless of rank — every eligible order gets the *one* action its
`primary_cause` implies, so there is nothing for a smarter ranking to improve except
*order*, and under this twin's capacities that ranking rarely binds. The value-aware
`CURRENT_POLICY` instead changes *what gets decided*, not just *whose turn is first*: for
every eligible order it considers **every point-in-time-feasible action**, not only the
one `primary_cause` implies, and ranks order-action pairs by an explainable proxy for
**expected avoided penalty per normalized resource capacity consumed**.

**Candidate actions** (`_value_aware_candidate_actions`): one candidate per *active*
`leading_signal_{cause}` that maps to a feasible action (`action_response.CAUSE_TO_ACTION`)
— reading only point-in-time-observable evidence, never `root_causes`' retrospective rule
evaluation, the simulator's response draw, or any potential-outcome/oracle field. If a
persisted Bayesian structural `do(node=0)` intervention scenario
(`intervention_scenarios_json`, from evidence observed strictly at/before decision time)
exists for that cause, its `relative_risk_reduction` becomes the candidate's *structural
reduction* term; otherwise it falls back to the existing, documented 60% deployed-UI
`avoided_risk_fraction` (`decisions.ImpactAssumptions`) scaled by a match confidence (1.0
if the cause is the order's primary cause, 0.4 if a corroborating active secondary
signal). In the rare case no active signal maps to a feasible action, the order's own
`primary_cause` mapping is used as a last-resort single candidate (`primary_cause_fallback`)
— identical to the single-cause baseline's mapping, so the value-aware policy is never
worse off than "no candidate."

**Expected value density** (`_expected_value_density`) — a short, fixed, exact formula,
no learned weighting and no access to potential-outcome fields:

```
expected_benefit = estimated_penalty_exposure × structural_reduction × execution_feasibility
value_density     = expected_benefit / normalized_resource_fraction
```

- `estimated_penalty_exposure` (dollars) is the existing fused-risk-weighted exposure
  already used everywhere else in this prototype (order value × penalty rate × combined
  risk) — *how much risk-weighted value is at stake*.
- `structural_reduction` (the term above, bounded `[0, 1]`) is *what share of that risk
  this specific action removes* — never re-deriving or double-counting the risk term
  itself.
- `execution_feasibility` (bounded `[0, 1]`) is a fixed weighted sum of three observable,
  point-in-time proxies: remaining promise slack (0.4, normalized over the 7-day
  prediction horizon), an action-specific resource trait (0.4 — vendor reliability for
  `VENDOR_ESCALATION`, DC utilization headroom for `WAREHOUSE_EXPEDITE`, SKU scarcity for
  `INVENTORY_REALLOCATION`, lane transit variability for `ALTERNATE_TRANSPORT`, customer
  appointment context for `APPOINTMENT_COORDINATION`), and Bayesian evidence coverage (0.2
  — how much of the 10-node network's evidence is actually observed for this order, a
  causal-confidence proxy that avoids double-counting the risk term again).
- `normalized_resource_fraction` divides the action's resource demand
  (`resources.demand_units_for`) by that resource pool's *scenario-independent default*
  daily capacity (`resources.default_daily_capacities`), so the ranking is identical
  across the 25%/50%/100% capacity-stress scenarios below — only the *acceptance* cutoff
  varies by scenario, never the priority order.

The deployed action for each eligible order is the candidate with the highest positive
`value_density`; the same score ranks resource allocation across orders, with the same
~10% seeded-exploration carve-out as before. Every weight/term/fallback above was fixed
before any benchmark run below and was never retuned after seeing results.


`CURRENT_POLICY`'s exploration carve-out uses two capacity-preserving designs. Discrete
(`vendor`/`customer`) slot pools split *whole-slot counts* via seeded stochastic
rounding, so a pool's full `min(capacity, candidates)` slots are always filled and every
candidate's exact marginal selection propensity is logged. Continuous (`dc`/`lane`)
pools fill the explore stage up to the pool's *actual* remaining capacity, not a fixed
fractional slice; because their order sizes differ, an exact per-order propensity has no
closed form there, so those rows honestly log `assignment_probability=None` with a
separately named `pool_reservation_ratio` instead of a mislabeled point estimate.

### Capacity-stress sensitivity analysis: why 100% capacity isn't the right question

At this twin's *default* (100%) daily capacities, resource pools are rarely binding —
they were sized generously relative to typical eligible-order volume — so a
capacity-priority *ranking* has very little to actually be discriminative about: on most
seeds, every deployable policy (including plain random selection) simply accepts every
eligible order it sees, regardless of priority. That is a genuine, measured property of
this twin, not a bug, but it also means "does `CURRENT_POLICY` beat simpler baselines at
100% capacity" is close to the wrong question for a resource-constrained business.

`resources.CAPACITY_SCENARIOS` therefore re-runs **every** policy — including the
oracle — at three pre-specified capacity multipliers applied *uniformly* to every
resource pool: `SCARCE_25_PERCENT` (0.25×), `SCARCE_50_PERCENT` (0.5×), and
`BASE_100_PERCENT` (1.0×, retained as a full-capacity diagnostic).
**`SCARCE_50_PERCENT` is Stage 1's headline scenario** — the acceptance gates below are
measured there, not at the unscaled baseline. Continuous pools (`dc`/`lane`) scale
directly (`capacity * multiplier`, no rounding). Discrete pools (`vendor`: 1 slot/day,
`customer`: 2 slots/day by default) would often round to a fractional slot below one
whole unit at 25%/50% (e.g. a 1-slot vendor pool at 50% targets 0.5 slots/day); instead
of silently flooring that to a permanently-zero pool, `resources.build_capacity_schedule`
walks a deterministic whole-slot accumulator (an error-diffusion/"Bresenham line"
schedule) across the full evaluated calendar, so a 1-slot pool at 50% realizes exactly
`0, 1, 0, 1, ...` and at 25% realizes a slot on exactly one day in four — the same
schedule, built once per scenario and shared unmodified across every policy, so no
policy ever sees a different realized capacity than any other for the same scenario/day.

Every policy/scenario now reports two discriminativeness metrics alongside the existing
ones: **contested rate** (fraction of eligible candidates that lost out to capacity —
`0` means the ranking never mattered that day) and **capacity-binding rate** (fraction
of pool-days where the pool's capacity was fully consumed). Both rise sharply as
capacity shrinks, confirming the stress scenarios are genuinely discriminative, not a
re-run of the same slack baseline three times. `CURRENT_POLICY`'s contested/binding
rates are *lower* than the single-action baselines' at every scenario (e.g. 3.95% vs.
7.86% contested at 50% capacity) — an expected, honest side effect of spreading eligible
orders' chosen actions across more resource pools instead of concentrating every order on
the one pool its `primary_cause` alone would imply:

| Capacity scenario | Median contested rate (`CURRENT_POLICY` / `SINGLE_CAUSE_PRIORITY_BASELINE`) | Median capacity-binding rate (`CURRENT_POLICY` / `SINGLE_CAUSE_PRIORITY_BASELINE`) |
| --- | ---: | ---: |
| `BASE_100_PERCENT` (diagnostic) | 0.4% / 1.4% | 6.0% / 12.0% |
| `SCARCE_50_PERCENT` (**primary**) | 3.9% / 7.9% | 43.0% / 45.9% |
| `SCARCE_25_PERCENT` | 27.1% / 30.4% | 43.0% / 45.9% |

**Measured 5-seed results across all three capacity scenarios** (seeds 1–5, 2,500
orders/seed; `artifacts/policy_benchmark.json`, reproduced with
`uv run otif-policy-benchmark --seeds 1 2 3 4 5 --orders 2500`):

| Policy | 25% capacity (median) | **50% capacity — primary headline** | 100% capacity (diagnostic) |
| --- | ---: | ---: | ---: |
| `NO_ACTION` | 0.00 | 0.00 | 0.00 |
| `RANDOM_AT_CAPACITY` | 8.20 | 9.17 | 15.09 |
| `HIGHEST_RISK_AT_CAPACITY` | 8.06 | 9.17 | 14.96 |
| `HIGHEST_FINANCIAL_AT_CAPACITY` | 8.06 | 9.17 | 15.09 |
| `STRONGEST_SIGNAL_HEURISTIC` | 8.10 | 9.17 | 15.09 |
| `SINGLE_CAUSE_PRIORITY_BASELINE` | 8.06 | 9.17 | 15.12 |
| **`CURRENT_POLICY`** (value-aware) | **8.35** | **9.96** | **17.02** |
| `ORACLE_EVALUATION_ONLY` (ceiling) | 56.45 | 74.45 | 119.76 |

(Median avoided penalty per normalized scarce-resource unit; full per-seed table,
action precision, avoidable-miss coverage, and regret vs. oracle at every scenario are
in `artifacts/policy_benchmark.json`'s `summary.median_*_by_capacity_scenario`.)

**At the primary 50%-capacity headline, `CURRENT_POLICY` beats every deployable
baseline — `RANDOM_AT_CAPACITY`, `HIGHEST_RISK_AT_CAPACITY`, `HIGHEST_FINANCIAL_AT_CAPACITY`,
`STRONGEST_SIGNAL_HEURISTIC`, and `SINGLE_CAUSE_PRIORITY_BASELINE` — and the primary
acceptance gate is measured and reported as PASSED** (`acceptance_gates.primary_gate_passed
= true` in `artifacts/policy_benchmark.json`). Per-seed deltas
(`CURRENT_POLICY − baseline`, tie tolerance `1e-6`, at `SCARCE_50_PERCENT`):

| Seed | 1 | 2 | 3 | 4 | 5 | Wins | Ties | Losses |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Δ vs. `RANDOM_AT_CAPACITY` | +0.2171 | +1.1062 | +0.1896 | **−0.8500** | +0.1183 | 4 | 0 | 1 |
| Δ vs. `HIGHEST_RISK_AT_CAPACITY` | +0.2171 | +1.1062 | +0.1896 | **−0.7119** | +0.0406 | 4 | 0 | 1 |
| Δ vs. `SINGLE_CAUSE_PRIORITY_BASELINE` | +0.2171 | +1.1062 | +0.1896 | **−0.8075** | +0.0406 | 4 | 0 | 1 |

`CURRENT_POLICY` wins 4 of 5 seeds against every baseline (only seed 4 loses — reported
as measured, not adjusted away), clearing the `win_threshold = 3` (`ceil(5 × 3/5)`) gate
against both `RANDOM_AT_CAPACITY` and `SINGLE_CAUSE_PRIORITY_BASELINE`. The 5-seed median
(9.964) beats `RANDOM_AT_CAPACITY`/`HIGHEST_RISK_AT_CAPACITY`/`SINGLE_CAUSE_PRIORITY_BASELINE`'s
shared median (9.1744) by +0.79 (+8.6%). `CURRENT_POLICY` also beats every baseline at
25% capacity (8.35 vs. 8.06–8.10) and at the unscaled 100%-capacity diagnostic (17.02
vs. 14.96–15.12), so both diagnostic base-capacity gates
(`diagnostic_current_beats_random_at_base_capacity`,
`diagnostic_current_beats_highest_risk_at_base_capacity`,
`diagnostic_current_beats_legacy_at_base_capacity`) pass as well. `CURRENT_POLICY` is
positive in both the normal window (median 1191.18 total avoided penalty) and the
scripted drift window (median 588.62) at the primary 50%-capacity scenario, and its
action precision (32.34%) is within the required 5-percentage-point tolerance of
`HIGHEST_RISK_AT_CAPACITY`'s (32.56%, a 0.22-point gap — `no_action_precision_collapse =
true`). `CURRENT_POLICY` also beats every baseline on two secondary metrics never part of
the acceptance gate: median regret vs. oracle (3382.91 vs. 3504.16 for the baselines —
*lower* is better) and median avoidable-miss coverage (43.8% vs. 41.3%) — the value-aware
ranking is not winning the headline by trading away these other measures. See
`docs/model-card.md` for the full per-seed/per-scenario table.

**Value-aware-policy diagnostics** (`summary.value_aware_policy_diagnostics`, 5-seed
medians at 50% capacity): candidate-action coverage (fraction of eligible orders whose
chosen action came from real active-signal/Bayesian evidence, not the last-resort
`primary_cause_fallback`) is 100%; the Bayesian-evidence rate (fraction of chosen actions
backed by a persisted Bayesian scenario rather than the leading-signal-only fallback) is
also 100% in this twin, because a `leading_signal` can only be active once its underlying
evidence has posted, which is exactly when the matching Bayesian scenario also exists.
Each eligible order has ~1.56 feasible candidate actions on average. The **expected-vs-
realized value-density Spearman rank correlation** (an evaluation-only diagnostic that
never influences any decision) is a median of **0.41** across seeds — the decision-time
ranking is meaningfully, though imperfectly, predictive of the twin's own realized
outcome. Chosen action mix (median): `INVENTORY_REALLOCATION` 51.3%,
`APPOINTMENT_COORDINATION` 33.8%, `WAREHOUSE_EXPEDITE` 13.4%, `VENDOR_ESCALATION` 2.1%,
`ORDER_CAPTURE_CORRECTION` 1.9%, `ALTERNATE_TRANSPORT` 1.5%.

**Bayesian-ablation diagnostic** (`bayesian_ablation_diagnostic`; value-aware policy
*with* vs. *without* its Bayesian structural-reduction term, everything else held fixed):
reported honestly, not required to favor the Bayesian term. In this twin, the ablated
variant *without* the Bayesian term actually scores **higher** at the primary 50%-capacity
scenario (median 11.34 vs. 9.96 *with* the term — the simpler leading-signal-only
fallback formula outperforms the Bayesian-scenario-driven one here), and the Bayesian
term added value in 0 of 5 seeds. This is reported as measured, not adjusted away: it
does not change the gate result (`CURRENT_POLICY` uses the Bayesian term by design and
still passes the primary gate), but it is an honest finding that the Bayesian structural
scenarios are not currently pulling their weight in this formula/twin combination, worth
revisiting once Stage 2 governance can A/B two `CURRENT_POLICY` variants directly.

Counterfactual ranking (same 5 seeds, `n≈1,150–1,350` signaled orders/seed — unaffected
by the `CURRENT_POLICY` redefinition above, since this diagnostic is capacity/threshold-
independent): the Bayesian top intervention beats a random feasible action's value regret
in every seed and beats the strongest-signal heuristic's value regret in 3 of 5 seeds —
but its raw top-action **agreement** rate (~0.126–0.139) is consistently *lower* than the
strongest-signal heuristic's (~0.167–0.200) across all 5 seeds. The Bayesian ranking adds
measurable value without being forced to dominate every diagnostic. This diagnostic does
not depend on resource capacity or the fused threshold, so it is unaffected by the
capacity-stress scenarios above.

All measured numbers above are regenerated, not hand-tuned: `resources.py`'s existing
capacity engine, `decisions.py`'s transparent single-cause priority formula,
the response-probability weights, the value-aware formula's weights/fallbacks/
normalizations (fixed in `policy_evaluation.py` before this benchmark ran), and the three
capacity-stress multipliers are all fixed before any benchmark run, and no seed, baseline,
or formula term was adjusted after seeing results. Full per-seed breakdowns, the
resource-consumption ledger, and the current-policy decision log are in
`artifacts/policy_benchmark.json`, alongside its `policy_benchmark_manifest.json` run
manifest.

### Governance (Stage 2): production readiness without external infrastructure

Stage 1's GO recommendation above has been acted on. Stage 2 adds deterministic run
manifests, production-shaped source adapters/service contracts with a proven
offline/batch parity guarantee, a decision/outcome ledger with an observational cohort
report, a champion/challenger promotion gate with an auditable active-model pointer and
append-only lifecycle events, rolling monitoring/SLO reporting, and two new Streamlit
views — all local, stdlib/existing-dependency only (no web framework, database, cloud
service, message broker, feature store, external causal/uplift model, MILP, RL, or live
LLM). Every number below is from a real, freshly-regenerated local run
(`uv run python -m otif_risk.pipeline --seed 42`,
`uv run python -m otif_risk.policy_benchmark`,
`uv run python -m otif_risk.operations --seed 42 --replay-days 90`); rerun those commands
to reproduce your own copies (`git_sha`/`content_id` will differ if the checkout is dirty
or on a different commit).

**Run manifests (`manifest.py`).** Every pipeline run, policy-benchmark run, and
operations replay writes a `run_manifest.json` capturing git SHA/dirty state, package and
key dependency versions, normalized config/policy/schema versions, input-table
fingerprints (row counts, schema, date ranges, content hashes — the model-facing source
tables only, never the evaluation-only truth tables), a feature-schema hash,
training/validation/test windows, and SHA-256 checksums of every other artifact already
written to that run's directory (never checksumming the manifest itself). The canonical
seed-42 run's manifest recorded 29 checksummed artifacts and verified clean
(`manifest.verify_manifest` → `verified: true`); the 90-day operations replay's manifest
checksummed 121 artifacts, also fully verified. `content_id` is a separate, deterministic
hash that excludes `generated_at_utc`/`run_instance_id`/`run_directory` — two runs of
identical seed/config/code/model-facing artifacts produce the *same* `content_id` even
run at different times or in different output directories (`test_manifest.py` proves
this, plus tamper detection: mutating a checksummed file after the manifest is written
flips `verified` to `false` and names the mismatched file).

**Production-shaped adapters and service contracts (`adapters.py`,
`service_contracts.py`).** Local-CSV `LocalCsvERPAdapter`/`WMSAdapter`/`TMSAdapter`/
`SRMAdapter` implementations of a typed `SourceAdapter.load(as_of_timestamp)` protocol
reconstruct the canonical `PrototypeDataset` source tables from persisted CSVs, redacting
(never dropping) any event whose `event_timestamp` has not genuinely occurred yet as of
the requested snapshot — the same point-in-time contract `features.build_feature_table`
already enforces, applied again at the source boundary. `ScoreRequest`
(`as_of_timestamp`, order IDs, `source_snapshot_id`, `idempotency_key`) and
`ScoreResponse` (model/policy version, manifest content ID, risk score, threshold,
confidence, explanation, decision/resource status) are validated dataclasses;
`JsonlDecisionSink`/`CsvDecisionSink` upsert by decision key, so retrying an identical
write never creates a duplicate row (`test_service_contracts`-style idempotency tests in
`test_adapters.py`). The canonical seed-42 run's `parity_check.json` proves the explicit
offline/batch parity claim: the same 15 orders at the same as-of snapshot, scored once
directly against the in-memory dataset and once through the adapter/service boundary
(CSV round-tripped tables), produced byte-identical feature vectors and scores
(`passed: true`, `mismatched_order_ids: []`).

**Decision/outcome ledger and observational cohorts (`decision_ledger.py`).** The 90-day
operations replay logged 6,226 decisions (one per open order scored per day) to an
idempotent, upsert-by-decision-key CSV ledger, then reconciled 5,923 of them against
matured OTIF outcomes (303 orders were still open at the replay's end). The resulting
**observational, non-causal** cohort report (`observational_cohort_report.json`,
`observational_not_causal: true`) compared realized miss rates: orders the policy
**accepted** (716 matured decisions) had a 57.96% miss rate versus 10.08% for orders left
**monitored** (5,206 matured decisions) — expected and unremarkable on its own, since
this prototype's daily replay closes every order against its own pre-generated outcome
regardless of the decision made (action does not yet feed back into the replayed
lifecycle here), so the gap reflects *which orders the policy prioritized* (correctly,
higher-risk ones), never a measured causal effect of acting. Each ledger row also carries
the order's real `order_value`/`penalty_rate` (from `decisions.attach_business_context`,
captured at decision time), so `realized_penalty` reflects genuine dollar exposure rather
than a placeholder zero — the accepted cohort's mean realized penalty ($113.54) is
markedly higher than the monitored cohort's ($13.33), consistent with the policy
prioritizing higher-value/higher-risk orders. `intervention_outcomes.json` breaks this
down further by intervention type (`chosen_action`) versus a no-intervention baseline —
e.g. `INVENTORY_REALLOCATION` (431 matured decisions, 70.3% miss rate) was chosen almost
exclusively for the hardest, most at-risk orders. Both reports explicitly carry the same
qualification and a minimum-sample guard (cohorts/action types below the guard, e.g. the
lone `REJECTED`-cohort order here, have their rate withheld, not reported on too few
observations). The Decision Value Lab above remains the only causally-interpretable
(exact, common-random-number potential-outcome) policy value in this codebase, and is
kept fully separate from this ledger.

**Champion/challenger registry and promotion (`registry.py`).** `evaluate_promotion`
compares a challenger against the current champion on PR-AUC, Brier, calibration error,
recall, alert rate, drift-regime PR-AUC, and policy value at the 50%-capacity scenario,
each against a fixed, explicit tolerance (e.g. 0.02 absolute PR-AUC, 5% relative policy
value); any failed check holds the challenger and leaves `active_model.json` (written
atomically) untouched, and every promotion/hold/rollback is appended to an
append-only event log that is never rewritten. The 90-day canonical replay demonstrates
both halves of this gate:

- **Real retrain lifecycle**: 9 model versions were trained (1 initial + 8
  cadence/drift-triggered retrains). Every one of the 8 real challengers was **held** —
  each retrain's own held-out test PR-AUC (measured on that retrain's necessarily small,
  recent slice of matured history in this short 90-day window) regressed well beyond the
  0.02 tolerance versus the still-strong initial champion (e.g. v9's 0.343 PR-AUC vs.
  v1's 0.717); the gate correctly kept the safer, better-measured champion active for the
  entire replay rather than chasing noisy short-window retrains. This is an honest,
  reported finding, not a bug: see "Limitations" below for what it costs.
- **Demo governance-lifecycle scenario** (`demo_lifecycle_scenario.json`), built only from
  Stage 1's own already-measured `policy_benchmark.json` numbers (never fabricated),
  distinctly labeled `demo_lifecycle_scenario_from_measured_stage1_policy_benchmark` so it
  is never confused with the real retrain lifecycle above: (1) **promotes** the
  value-aware action policy without Bayesian action ranking (5-seed median policy value
  11.345 at 50% capacity) over `SINGLE_CAUSE_PRIORITY_BASELINE` (9.174); (2) **holds**
  a Bayesian action-ranking challenger (9.964) — a −12.2% relative
  regression against the 5%-tolerance floor, so it is correctly held with the reason
  recorded verbatim (`policy value at 50% capacity regressed 11.3445 -> 9.9640 (floor
  10.7773, tolerance fraction 0.05)`); (3) **rolls back** the active pointer to `v1`, a
  real, manifest-verified version from the retrain lifecycle above
  (`rolled_back: true`). The replay's final `active_model.json` therefore points to `v1`.

**Rolling monitoring and SLOs (`monitoring.py`).** The replay's `monitoring_report.json`
reports rolling-origin realized PR-AUC/precision/recall/calibration/alert-rate on
matured, minimum-sample-guarded ledger windows, normal-vs-drift regime quality, a 90-day
freshness cadence (1 calendar day, structural), and measured **local-only** runtime
(mean scoring 1.48s/day, mean retrain 3.35s over 8 retrains — explicitly labeled
`measured_local_runtime_only_not_a_production_latency_claim`, no production-latency claim
is made). Soft data-quality metrics (completeness, uniqueness, referential health) all
passed with zero contract failures. One SLO honestly **failed**: rolling realized PR-AUC
(0.386) fell short of the 0.55 target — the direct, reported cost of the promotion gate
correctly keeping the safer `v1` champion active for the full 90 days rather than
promoting any of the noisier later retrains (see "Limitations" below).

**Policy Value and Governance Streamlit views (`app.py`).** The Policy Value view adds a
25%/50%/100%-capacity selector over `policy_benchmark.json`, the per-policy avoided
penalty/resource/action-precision/coverage/regret table (oracle labeled evaluation-only),
paired per-seed win/tie/loss counts, the value-density formula explanation, action mix,
normal/drift regime value, and the Bayesian ablation shown honestly (negative, not
hidden). The Governance view adds a manifest trust card (git SHA, content ID, checksum
verification, schema versions), the champion/challenger lifecycle timeline with reasons,
a metric-delta table, the decision ledger and observational cohorts with a visible
non-causal badge, a realized-outcomes-by-intervention-type breakdown, monitoring/SLO
cards, and offline/batch-parity status — every number loaded from persisted artifacts,
nothing recomputed at render time. Both use the same
warm-paper/charcoal/signal-blue/safety-orange palette as the rest of the control tower,
with green reserved for verified/passed/promoted states and red for held/regression/
failed states.

## AI Copilot

A read-only explanation/drafting layer sits on top of the already-governed decision above.
**It never decides anything** — it cannot change a score, a threshold, `RECOMMENDED`/
`CONTESTED`/`MONITOR`, a resource allocation, or a model/policy promotion; it can only explain,
answer a fixed catalog of questions, and draft text for a planner to copy manually.

**Evidence packet (`copilot_context.py`).** Deterministic code — never the LLM — builds a
compact, allowlisted, cited "evidence packet" for one order or one fixed portfolio question:
identity/status, model/policy/manifest versions, XGBoost/BBN/fused risk with the operating
threshold, the top SHAP/perturbation factors (explicitly labeled `association_not_causation`),
the Bayesian mechanism route/priors/posteriors and best fixed-structure scenario (explicitly
labeled a scenario, not a proven treatment effect), affected SKUs, observed/missing lifecycle
events, resource contention, and business impact (order value, penalty exposure, quantity at
risk, and any simulated policy-value figure labeled as simulator evaluation). Every fact has a
stable `id` (e.g. `risk.combined`, `shap.1`, `sku.SKU0042`) that both live and fallback answers
must cite. Secrets, file paths, git remotes, raw planner-feedback text, and simulator/line
ground truth are never read into it; lists/strings are truncated to fixed, deterministic limits.
Portfolio questions come from a fixed catalog of hand-written aggregations
(`PORTFOLIO_QUESTIONS`) — there is no unrestricted DataFrame/SQL/code-execution path.

**Deterministic fallback (`copilot_fallback.py`).** Produces the exact same structured response
schema (below) from the evidence packet with plain Python — no network call, no API key, no
randomness — for the order question catalog ("Explain this order simply", "Why was it flagged?",
"Which SKU is affected?", "Why is the action contested?", "Draft a supplier escalation") and every
portfolio question. This is the demo's always-on mode and also what live mode falls back to.

**Live OpenAI integration (`llm_copilot.py`).** Uses the official `openai` Python SDK's Responses
API with Structured Outputs (a strict JSON schema matching the response shape below) when
`OPENAI_API_KEY` is configured. Config: `OPENAI_API_KEY` (required only for live mode),
`OPENAI_MODEL` (default `gpt-5-mini`, a small cost-efficient current-generation model — override
freely), `OTIF_LLM_MODE=auto|live|fallback` (`auto` tries live and falls back automatically on any
failure; `live` still falls back gracefully rather than showing a blank page; `fallback` never
calls the network). Timeout and max output tokens are fixed in code; model sampling uses the
provider default for compatibility. The system prompt requires: facts only from the supplied evidence, a citation
for every material claim, `association_not_causation` for SHAP separated from the Bayesian
fixed-structure-scenario framing, "unknown" when evidence is absent, the persisted decision
preserved (never overridden), honest labeling of simulated values, refusal of prompt-injection
attempts to reveal secrets/instructions, and concise planner language. See `.env.example`.

**Structured response schema** (identical for live and fallback):

```json
{
  "headline": "One-sentence summary",
  "what_happened": ["..."],
  "why_flagged": [{"text": "...", "citations": ["risk.combined", "shap.1"]}],
  "affected_items": [{"text": "...", "citations": ["sku.SKU0042"]}],
  "recommended_next_step": {
    "text": "...", "citations": ["decision.action"], "preserves_persisted_decision": true
  },
  "uncertainties": [{"text": "...", "citations": ["risk.evidence_coverage"]}],
  "draft_message": "Optional supplier/customer/operations draft, display-only",
  "disclaimer": "Explanation support only; production decision unchanged."
}
```

**Citation and hallucination guard (`copilot_validation.py`).** Rejects any citation ID not
present in the evidence packet, requires at least one citation per factual explanation item,
requires `preserves_persisted_decision: true`, rejects a response whose next-step text asserts a
decision status inconsistent with the persisted one, rejects non-finite values and oversized
output, and strips unsupported HTML/URLs. On any validation failure the caller falls back to the
deterministic response — a judge can force this by pointing `OPENAI_API_KEY` at an invalid key.
This does not guarantee an LLM can never hallucinate; it guarantees unsupported claims are never
presented as grounded.

**Audit (`copilot_audit.py`).** Every request appends one line to `<run_directory>/
copilot_audit.jsonl`: request ID/timestamp, order ID or portfolio query type, provider/model/mode
(configured vs. actually used), an evidence-packet hash (not its contents), a prompt-template
version, latency, token usage when reported, validation status/fallback reason, and cited fact
IDs — never the API key, the full response text, or any chain-of-thought.

**Streamlit view.** The "AI Copilot" sidebar view adds an Order Copilot tab (order selector,
live/fallback mode badge, evidence-packet preview, a question selector, per-order chat history for
the session, citation badges that resolve to the underlying fact, an uncertainty panel, and a
copy-only draft-message code block — no send/execute action) and a Portfolio Copilot tab (the
fixed question catalog only), plus a Copilot health card (live/fallback counts, validation
pass rate, median latency, estimated token usage, recent audit entries).

**Evaluation (`copilot_evaluation.py`, `uv run otif-copilot-eval`).** Because no human-labeled
explanation dataset exists, this evaluates a deterministic set of representative orders (high-risk
inventory miss, timing-driven miss, multi-cause, contested action, low-confidence, safe/monitor,
unknown-cause — each reported as unavailable rather than guessed if this dataset has none) against
every supported question: citation validity, decision-status/action preservation, required-section
completeness, fallback success rate, and latency/token usage, plus one live-vs-fallback smoke
comparison when an API key is actually configured. No BLEU/factuality/human-preference claim is
made. Writes `artifacts/copilot_evaluation.json`.

```bash
# AI Copilot representative-order evaluation (fallback-only unless OPENAI_API_KEY is set)
uv run otif-copilot-eval --artifacts-root artifacts --output artifacts/copilot_evaluation.json
```



This project uses Python 3.12 because SHAP's native dependencies are not compatible with
the available Python 3.14 runtime.

```bash
uv sync --extra dev
brew install libomp  # macOS only, required for XGBoost

# Canonical single scoring run
uv run otif-risk --orders 2500 --seed 42

# Multi-seed benchmark (median/range + acceptance gates)
uv run otif-benchmark --seeds 1 2 3 4 5 --orders 2500 --output-dir artifacts \
  --benchmark-path artifacts/benchmark.json

# Decision Value Lab: multi-seed policy-value benchmark at 3 capacity-stress
# scenarios (25%/50%/100% of default capacity; 50% is the primary headline --
# avoided penalty per normalized resource unit, action precision, regret vs.
# oracle, contested/binding rates, paired deltas, ranking diagnostics)
uv run otif-policy-benchmark --seeds 1 2 3 4 5 --orders 2500 \
  --benchmark-path artifacts/policy_benchmark.json

# Local daily operations replay (scoring, closures, drift, versioned retraining,
# Stage 2 governance: manifests, decision ledger, champion/challenger promotion,
# monitoring/SLOs, and the demo promoted/held/rolled-back lifecycle scenario)
uv run otif-ops --orders 2500 --seed 42 --replay-days 90 --output-dir artifacts \
  --policy-value-reference-path artifacts/policy_benchmark.json

# Streamlit control tower (reads whichever run-*/ops-*/benchmark.json are present,
# including the Policy Value, Governance, and AI Copilot views)
uv run streamlit run src/otif_risk/app.py
```

Copy [`.env.example`](.env.example) to `.env` (never commit it) to configure the AI Copilot's
live OpenAI mode; the Copilot runs fully on its deterministic fallback with no `.env` at all.

Threshold tuning defaults to `recall_floor` with `target_recall=0.65` and
`min_precision=0.30`, applied to the fused score (see above).

Artifacts are written under `artifacts/run-<config-hash>/` (single pipeline runs) and
`artifacts/ops-<config-hash>/` (operations replays), including source tables, simulator
truth, outcomes, root causes, feature tables, scored orders and lines, fusion
comparison, rollups, models, metrics, model registry, daily queues, an append-only
planner/system feedback log, a deterministic `run_manifest.json`, a `parity_check.json`
(pipeline runs), and — for operations replays — a `decision_ledger.csv`,
`observational_cohort_report.json`, `intervention_outcomes.json`, `monitoring_report.json`,
`demo_lifecycle_scenario.json`, and a `registry/` directory (`registry_versions.json`,
`registry_events.jsonl`, `active_model.json`). Rerunning an identical configuration never
overwrites a prior run — a monotonically increasing numeric suffix is appended.

## Validation

```bash
uv run pytest
uv run ruff check .
uv run python -m build
```

## Scope and honesty

- The risk model predicts OTIF failure. It does not directly predict one forced cause;
  the causal chain and root-cause derivation are produced and evaluated separately.
- SHAP factors and Bayesian pathways are associations, not proof of causality; the
  pathway's `interpretation` field says so explicitly, and `cause_fidelity` in
  `metrics.json` compares the evidence-derived primary cause against the retrospective
  rule-derived cause. Because both use operational evidence, this is a consistency
  diagnostic rather than latent-cause recovery.
- **Structural intervention scenarios are not proven treatment effects.**
  `intervention_scenarios_json`/`BayesianBundle.intervene` compute an exact
  `do(node=value)` posterior under this fixed, fitted network's assumptions -- a
  "fixed-structure causal scenario analysis," never an identified or randomized causal
  effect. They never feed the XGBoost score, the fused score, or the operational
  decision; `causal_consistency` in `metrics.json` reports agreement rates against
  independent reference labels as a *consistency* diagnostic, explicitly distinguished
  from causal validation.
- **Evidence attribution is not SHAP.** `causal_attribution_json`'s leave-one-evidence-out
  contribution measures this fixed network's sensitivity to withholding one observed
  cause node, labeled `evidence_attribution_leave_one_out`.
- The fusion weight is chosen on validation only, from a fixed, explainable grid; no
  stacking model is fit.
- Financial impact in the deployed decision table (`decisions.py`'s
  `estimated_avoidable_penalty`) still uses a fixed, documented 60% effectiveness
  assumption — it is a fast, transparent UI heuristic, not the measured figure.
  **Measured** simulated policy value now comes from the separate Decision Value Lab
  (`action_response.py`/`policy_evaluation.py`, see above), which replays every feasible
  action through the twin's own lifecycle mechanics with common random numbers; it is
  still an evaluation against a synthetic twin, never an observed real-world causal
  effect, and the oracle policy is an explicitly unattainable, evaluation-only ceiling.
- **At the Decision Value Lab's primary capacity-stress headline (50% of default
  capacity), the value-aware `CURRENT_POLICY` beats `RANDOM_AT_CAPACITY`,
  `HIGHEST_RISK_AT_CAPACITY`, and `SINGLE_CAUSE_PRIORITY_BASELINE` on the
  5-seed median (wins 4 of 5 seeds against each), and the acceptance gate is measured
  and reported as passed** (`acceptance_gates.primary_gate_passed = true` in
  `artifacts/policy_benchmark.json`); it also wins the 25%/100%-capacity sensitivity
  scenarios and improves regret-vs-oracle/avoidable-miss-coverage without trading them
  away. The honest caveat: an ablation shows the policy's Bayesian structural-reduction
  term does not currently add value over its own simpler fallback in this twin (see
  "Decision Value Lab" above for the full per-scenario table and per-seed win/tie/loss
  counts; see "Governance (Stage 2)" for how the promotion gate uses this exact finding
  to hold a Bayesian-enhanced policy challenger).
- Intervention "avoided misses"/savings in the operations loop's daily replay remain
  simulated, non-causal estimates (the operations replay closes each order against its
  pre-generated outcome and does not yet call the Decision Value Lab's action-response
  twin); the Decision Value Lab's own avoided-penalty figures are the measured ones.
- The AI Copilot (see "AI Copilot" below) is a read-only explanation/drafting layer over the
  already-governed decision: it can call the live OpenAI Responses API when configured, but it
  never changes a score, threshold, decision, resource allocation, or governance action, and it
  always has a fully offline deterministic fallback with the same structured, cited schema.
- Held-out metrics on this synthetic dataset should not be read as production
  readiness evidence; see `docs/model-card.md` for the measured multi-seed benchmark and
  its honest limitations.
- **Governance is implemented, and its limits are reported, not hidden.** The 90-day
  canonical operations replay's promotion gate correctly held all 8 real retrain
  challengers (each regressed held-out PR-AUC beyond tolerance versus the initial
  champion, on this short window's necessarily small/noisy test splits), so the active
  model never advanced past `v1` for the whole replay — the direct, honestly-reported
  cost is a rolling-monitoring SLO failure (realized PR-AUC 0.386 versus a 0.55 target).
  The demo governance-lifecycle scenario (promoted/held/rolled-back) is built only from
  Stage 1's own already-measured `policy_benchmark.json` numbers and is explicitly
  labeled/kept distinct from the real per-day retrain lifecycle — see "Governance
  (Stage 2)" above.
- Real per-retrain promotion checks reuse a single reference **policy value** (Stage 1's
  measured 5-seed median at 50% capacity for the deployed `CURRENT_POLICY`) rather than
  re-running the full rolling-origin policy-value lab for every retrain (prohibitively
  expensive across a 90-day replay); only the demo lifecycle scenario compares genuinely
  different policy-value numbers (single-cause baseline, with/without the Bayesian term).
- Adapters/service contracts are typed Python interfaces plus local-CSV fixture
  implementations — production-shaped, but deliberately not a web framework, database,
  message broker, or cloud deployment (none of that is needed to prove the contract or
  the offline/batch parity guarantee locally).
