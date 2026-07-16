# PDF-Aligned OTIF Prototype

Standalone implementation of the complete architecture described in the OTIF project
PDF. It is intentionally a clear demonstration prototype rather than a production
reliability research platform.

It follows the PDF’s complete demonstration flow while retaining necessary,
disclosed corrections:

- the predictive target is whether an open order will miss OTIF (binary), not the
  PDF's inconsistently worded seven-class root-cause classifier — root-cause/pathway
  output is still produced and evaluated separately (see "Cause fidelity" below);
- features must be available at the declared scoring timestamp;
- the decision threshold is selected on, and applied to, the **fused** risk score —
  the score space decisions/UI actually use — not the standalone XGBoost score;
- a vendor's own rolling reliability metric only counts misses the vendor was at
  fault for, so it is not penalized for DC/transport/customer-caused misses.

## Architecture

1. Normalized synthetic order, line, vendor, DC, lane, customer and event tables.
2. Fail-fast schema, referential and logical validation.
3. Seven-category retrospective multi-cause derivation for closed orders.
4. Point-in-time order features and chronological train/validation/test splits.
   `leading_signal_*` columns are derived features (not raw generator output):
   each is a function of operational fields/events already filtered to
   `event_timestamp <= prediction_timestamp` (or of fields known at capture/from
   customer master data), so a cause only contributes a signal once real evidence
   has posted. See "Point-in-time signals and leakage" below.
5. Calibrated XGBoost OTIF-risk model (OpenMP-backed on macOS via Homebrew `libomp`).
6. Validation threshold tuning with recall/precision trade-offs (`recall_floor` default).
7. SHAP local explanations, with a deterministic perturbation fallback.
8. Seven-cause Bayesian network fit on the **training split's resolved history only**
   (not validation/test outcomes), with exact `pgmpy` inference and an explicit,
   reported empirical-table fallback. See "Bayesian inference mode" below.
9. Transparent fusion: 70% risk-model probability plus 30% Bayesian probability.
   XGBoost, Bayesian, and fused scores are each evaluated independently on the same
   held-out labels with the same metric set; **the operating threshold is selected
   on, and applied to, the fused validation scores**, and only that fused threshold
   drives decisions/UI (see "Threshold and score-space consistency" below).
10. Lookup-table mitigations and a capacity/quantity-aware DC conflict check
    (vendor/lane/customer conflicts remain count-based; see "Resource conflicts").
11. Vendor, DC, lane, customer, order-type, and SKU rollups (percentage at risk,
    value at risk, penalty exposure, quantity at risk, dominant cause) plus
    service-impact assumptions.
12. Templated planner narratives, append-only CSV feedback and three Streamlit views
    that reuse the pipeline's persisted decisions rather than recomputing them.

## Point-in-time signals and leakage

Earlier prototype code generated `leading_signal_*` columns directly from the
generator's *latent* disruption cause (a noisy function of `has(cause)`), so a
signal could appear on an order regardless of whether that cause had actually
become observable by `prediction_timestamp`. That leaked the label-adjacent
generator state into model features and produced near-perfect held-out AUCs that
were a leakage artifact, not evidence of model quality.

`leading_signal_*` is now derived entirely in `features.py` from operational
fields/events already filtered to `event_timestamp <= prediction_timestamp` (vendor
ready delay/exception, warehouse/transport exceptions), from fields known at order
capture (`capture_delay_hours`), from allocation/stockout state known at order time
(`INVENTORY_SHORTAGE`), from the DC capacity snapshot as of the prediction date
(`DC_CAPACITY`), or from customer master data known well in advance
(`customer_appointment_required`, the best available point-in-time proxy for
`CUSTOMER_DELIVERY` since the DELIVERED event itself always posts after
`prediction_timestamp`). `tests/test_features.py` proves: (a) `leading_signal_*` is
not present on the raw generator output, (b) it is not a lossless proxy for the
ground-truth cause (some matched causes show no signal because their evidence had
not posted yet), and (c) mutating an order's own not-yet-observed event cannot
change that order's feature row.

Cause fidelity is evaluated only on held-out OTIF misses. Successful orders are
excluded because they have no failure cause to recover; mixing `ON_TIME` rows into
this diagnostic would conflate outcome classification with pathway/cause fidelity.

**Remaining caveat**: this synthetic generator still assigns each cause's own
operational fields with deterministic, low-noise thresholds (for example, capture
delay > 24h vs. 0–8h with no overlap), so once a cause's evidence *has* posted, it
remains highly separable. Held-out metrics on this dataset should be read as a
leakage/separability diagnostic for this specific synthetic generator — reported
directly in `metrics.json` under `data.synthetic_data_note` and alongside a
prevalence baseline (`model_scores.prevalence_baseline`) — not as evidence of
production-grade predictive skill.

## Threshold and score-space consistency

A prior version selected the decision threshold on the calibrated XGBoost
probability (`risk_model_score`) but applied it to the fused probability
(`combined_risk_score`), which on one run pushed the threshold to ~0.999 and
classified all 500 test orders as `MONITOR` with zero business impact. The
pipeline now:

1. Scores validation and test with XGBoost, Bayesian, and fused probabilities.
2. Selects a threshold independently in *each* score space via
   `model.select_threshold`, using the same configured strategy
   (`recall_floor` by default).
3. Evaluates PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, Brier score,
   and flagged-order count for all three, at their own thresholds, on the same
   held-out labels — persisted under `metrics.json`'s `model_scores.xgb/bbn/fused`.
4. Uses **only the fused threshold** (`model_scores.fused.threshold`, also mirrored
   at the top level as `threshold` for backward compatibility) to build decisions
   and drive the Streamlit UI. The standalone XGB/BBN thresholds are reported for
   comparison only and never used for decisions.

## Bayesian inference mode

The Bayesian network is fit **only on the training split's resolved history**
(`order_id` restricted to `split.train`), not on the full dataset, matching the
same chronological boundary already enforced for the risk model. The empirical
fallback table is only used when the `pgmpy` exact-inference engine could not be
constructed at all (recorded explicitly as `architecture.bayesian_inference_mode`
= `"pgmpy_exact"` or `"empirical_table"`, with `architecture.bayesian_engine_build_error`
set when it falls back). Once an engine is successfully constructed, any error
raised while querying it surfaces immediately — it is never silently swallowed
into the fallback path.

## Resource conflicts

DC conflicts are quantity/capacity aware: when the scored frame carries the DC's
real `dc_daily_capacity_units`, candidates are accepted in priority order while
their cumulative `quantity_at_risk` stays within `dc_daily_capacity_units *
dc_capacity_recovery_fraction` (20% by default); once that allowance is exceeded,
remaining candidates — including a single order whose own `quantity_at_risk` alone
exceeds it — are marked `CONTESTED`. Vendor, lane, and customer conflicts remain a
documented count-based limit (`DEFAULT_RESOURCE_LIMITS`) because this prototype has
no equivalent numeric recovery-capacity field for those dimensions; DC also falls
back to the count-based limit when capacity data is unavailable.
`tests/test_decisions.py` proves both the capacity-aware DC behavior (a
deterministic, guaranteed conflict from real quantity/capacity math) and the
count-based fallback.

## Rollups and SKU representation

Rollups now cover vendor, DC, lane, customer, order type (the closest correctly
modeled proxy for the PDF's "order type" — this prototype only models order
*priority*, STANDARD/EXPEDITE), and SKU. Every rollup reports order count,
actionable orders, `pct_at_risk`, average risk, penalty exposure, `value_at_risk`
(order value weighted by risk — a broader revenue-exposure figure than the
penalty-rate-scaled `penalty_exposure`), quantity at risk, and the dominant primary
cause. The order-level scored table still carries a single `representative_sku`
(the first line's SKU) because an order can span multiple SKUs and its single
order-level risk score is not itself SKU-specific — attributing one score across
several SKUs would overstate precision. The SKU rollup instead uses *exploded*
order-line logic: every line is joined to its order's decision, so a multi-SKU
order contributes to every SKU it actually touches.

## Set up and run

This project uses Python 3.12 because SHAP’s native dependencies are not compatible with
the available Python 3.14 runtime.

```bash
uv sync --extra dev
brew install libomp  # macOS only, required for XGBoost
uv run otif-pdf --orders 2500 --seed 42
uv run streamlit run src/otif_pdf/app.py
```

Threshold tuning defaults to `recall_floor` with `target_recall=0.55` and
`min_precision=0.35`, selected on and applied to the fused score (see above). The
metrics artifact also records the older capacity-based baseline for comparison.

Artifacts are written under `artifacts/run-<config-hash>/`, including source tables,
outcomes, root causes, feature tables, fused scores, recommendations, rollups, models,
metrics and an empty planner feedback log. Rerunning an identical configuration never
overwrites or deletes a prior run: `_run_directory` appends a monotonically increasing
numeric suffix (`run-<hash>-2`, `run-<hash>-3`, ...) so every run remains distinguishable
on disk. `metrics.json` also carries `provenance` (generation timestamp, package
version, artifact schema version, run directory name) and `schema` (persisted column
lists for the scored-orders, feature, and root-cause tables) so consumers can detect
drift across reruns and code versions.

## Validation

```bash
uv run pytest
uv run ruff check .
uv run python -m build
```

## Scope and honesty

- The risk model predicts OTIF failure. It does not directly predict one forced cause.
- SHAP factors and Bayesian pathways are associations, not proof of causality; the
  `cause_fidelity` block in `metrics.json` compares the evidence-derived primary
  cause against the generator's ground truth as a fidelity diagnostic only.
- The fusion weights are fixed and transparent for the demo, not learned from test data.
- Financial impact uses documented assumptions and is illustrative.
- The LLM layer is represented by a deterministic narrative template.
- Held-out metrics on this synthetic dataset should not be read as production
  readiness evidence (see "Point-in-time signals and leakage" above).
- A production version would require real data validation, monitoring, governance and
  measured intervention outcomes.
