# OTIF Risk Intelligence

An explainable supply-chain control-tower prototype that predicts open orders at risk
of missing On-Time-In-Full delivery, identifies likely contributing factors down to the
affected SKU/line, recommends resource-aware mitigation actions, and replays a local
daily operating loop (scoring, closures, drift detection, versioned retraining).

- **Current architecture**: [`docs/architecture/current.mmd`](docs/architecture/current.mmd) /
  [`docs/architecture/current.svg`](docs/architecture/current.svg)
- **Target architecture** (this iteration): [`docs/architecture/target.mmd`](docs/architecture/target.mmd) /
  [`docs/architecture/target.svg`](docs/architecture/target.svg)
- **Model card** (measured benchmark numbers, honest limitations): [`docs/model-card.md`](docs/model-card.md)
- **Judge-facing demo script**: [`docs/demo-script.md`](docs/demo-script.md)

Both SVGs are rendered by a small, checked-in, deterministic Python renderer
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
7. A compact causal Bayesian chain (`bayesian.py`) replacing the direct seven-cause
   star network:
   `ORDER_CAPTURE→OTIF_MISS`, `VENDOR_FAILURE→INVENTORY_SHORTAGE`,
   `{INVENTORY_SHORTAGE, DC_CAPACITY}→WAREHOUSE_OPS`, `WAREHOUSE_OPS→TRANSPORT`,
   `TRANSPORT→OTIF_MISS`, `CUSTOMER_DELIVERY→OTIF_MISS`. CPTs are smoothed counts fit on
   the *training split's* resolved root causes only; a node is only given as hard
   evidence once its stage has actually been observed as of the as-of timestamp, so
   unobserved intermediate stages are marginalized out via exact inference rather than
   assumed absent. Exact inference uses `pgmpy` variable elimination when available, or
   a numerically-identical brute-force joint enumeration over this small 8-node network
   otherwise (see "Bayesian inference mode" below).
8. Evidence-based fusion (`fusion.py`): compares XGBoost-only, Bayesian-only, the fixed
   70/30 blend, and every other convex weight in 10% increments on validation, selecting
   the lowest-Brier candidate under a fixed-capacity recall guardrail (no stacking
   model); the operating threshold is then tuned separately for the chosen weight. See
   "Fusion weight selection" below.
9. Generic resource-aware interventions (`decisions.py` / `resources.py`): a lookup-table
   mitigation policy plus a capacity-aware conflict check (DC recovery units, lane
   alternate capacity, vendor escalation slots, customer appointment slots), greedily
   allocated by priority; overflow is marked `CONTESTED` with the competing orders
   listed (`contested_with`).
10. Vendor, DC, lane, customer, order-type, and SKU rollups plus service-impact
    assumptions (`decisions.py`).
11. Templated, structured planner narratives (risk → evidence → pathway → affected SKUs
    → action → resource status; `narratives.py`), append-only CSV feedback
    (`feedback.py`), and a five-view Streamlit control tower (`app.py`) that reuses only
    persisted decisions.
12. A local daily **operations replay** (`operations.py`): trains an initial model on a
    historical window, then for each simulated day scores every still-open order as of
    that day, allocates daily resource capacities, persists the queue, closes resolved
    orders, derives their actual cause, appends feedback, computes drift (PSI,
    score-distribution shift, missingness change, recent OTIF-rate change), and retrains
    on a documented cadence or drift trigger — persisting a versioned model registry.
13. A multi-seed benchmark (`benchmark.py`) with explicit acceptance gates.

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

Cause fidelity is evaluated only on held-out OTIF misses (successful orders have no
failure cause to recover). **Remaining honest caveat**: retrospective `cause_X` flags
are only ever 1 for orders that missed OTIF and matched that rule, which is a real
simplification affecting Bayesian calibration more than ranking — see
`docs/model-card.md`.

## Threshold and fusion-weight selection

1. Scores validation/test with XGBoost, Bayesian, and every fused weight on a
   0.0–1.0 grid in 0.1 increments.
2. Selects the fusion weight on **validation only**: lowest Brier score among
   candidates whose top-planner-capacity recall is within a small tolerance of the best
   candidate's capacity recall — a fixed, comparable operating point, not each
   candidate's own independently re-tuned recall-floor threshold (using each candidate's
   own threshold search here would let a miscalibrated candidate win purely by having an
   inflated F1-fallback recall; `tests/test_fusion.py` proves this can't happen).
3. Tunes the final decision threshold separately, once, for the chosen weight, using the
   configured strategy (`recall_floor` by default) on the chosen weight's fused
   validation scores.
4. Persists the full comparison table (`fusion_comparison.csv`,
   `metrics.json.fusion_comparison`) and the chosen weight/label/rationale
   (`architecture.fusion_chosen_weight/label/fusion`), regardless of which candidate wins.

## Bayesian inference mode

The chain is fit **only on the training split's resolved history**
(`bayesian_training_history`), matching the same chronological boundary enforced for the
risk model. `pgmpy` exact inference is used when importable/constructible; when it is
not, a brute-force joint enumeration over the small 8-node binary network is used
instead — verified numerically identical to `pgmpy`'s result in
`tests/test_bayesian.py::test_brute_force_fallback_matches_pgmpy_exact_inference`. Both
are exact; there is no approximate/empirical fallback. `architecture.bayesian_inference_mode`
records which one ran (`pgmpy_exact` or `brute_force_exact`), with
`architecture.bayesian_engine_build_error` set when `pgmpy` was unavailable.

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

## Set up and run

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

# Local daily operations replay (scoring, closures, drift, versioned retraining)
uv run otif-ops --orders 1200 --seed 42 --replay-days 90 --output-dir artifacts

# Streamlit control tower (reads whichever run-*/ops-*/benchmark.json are present)
uv run streamlit run src/otif_risk/app.py
```

Threshold tuning defaults to `recall_floor` with `target_recall=0.65` and
`min_precision=0.30`, applied to the fused score (see above).

Artifacts are written under `artifacts/run-<config-hash>/` (single pipeline runs) and
`artifacts/ops-<config-hash>/` (operations replays), including source tables, simulator
truth, outcomes, root causes, feature tables, scored orders and lines, fusion
comparison, rollups, models, metrics, model registry, daily queues, and an append-only
planner/system feedback log. Rerunning an identical configuration never overwrites a
prior run — a monotonically increasing numeric suffix is appended.

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
- The fusion weight is chosen on validation only, from a fixed, explainable grid; no
  stacking model is fit.
- Financial impact uses documented assumptions and is illustrative.
- Intervention "avoided misses"/savings in the operations loop are simulated estimates,
  never observed causal impact.
- The LLM layer is represented by a deterministic narrative template — no live LLM,
  cloud service, or external infrastructure is used anywhere in this prototype.
- Held-out metrics on this synthetic dataset should not be read as production
  readiness evidence; see `docs/model-card.md` for the measured multi-seed benchmark and
  its honest limitations.
