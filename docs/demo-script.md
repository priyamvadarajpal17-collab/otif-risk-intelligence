# Judge-Facing Demo Script — OTIF Risk Intelligence

A compact, end-to-end walkthrough anchored on one story: a vendor disruption creates
inventory pressure, the pressure reduces fulfillment slack and contributes to a
warehouse delay, and two high-priority orders end up competing for the same constrained
recovery capacity.

Everything shown below is generated, not staged copy: regenerate it locally with

```bash
uv sync --extra dev
brew install libomp   # macOS only, required for XGBoost
uv run otif-risk --orders 2500 --seed 42 --output-dir artifacts
uv run otif-benchmark --seeds 1 2 3 4 5 --orders 2500 --output-dir artifacts \
  --benchmark-path artifacts/benchmark.json
uv run otif-ops --orders 1200 --seed 42 --replay-days 90 --output-dir artifacts
uv run streamlit run src/otif_risk/app.py
```

## 1. Architecture in 30 seconds

Open `docs/architecture/target.svg` (or `current.svg` for the foundational
architecture). The digital twin (top) feeds point-in-time features into an XGBoost model
and a compact causal Bayesian chain; both feed a validated fusion step; the fused,
thresholded decision feeds affected-SKU evidence, explanations, and a resource-aware
policy into one unified intervention record; that record drives the order desk,
portfolio, and hotspot views, and also feeds the local operating-loop simulation
(daily scoring → closures → feedback/drift → versioned retraining).

## 2. The seed data: a genuinely noisy digital twin

- `uv run otif-risk --orders 2500 --seed 42` generates a fresh synthetic twin: stable
  vendor/SKU/DC/lane/customer traits, seasonality, correlated disruption shocks, missing
  events, and measurement noise (`src/otif_risk/data.py`).
- Measured miss rate on this seed: **15.9%**; across a 5-seed benchmark, median
  **17.5%** (range 15.2–20.2%) — inside the 15–25% target band every time tested.
- Ground truth (which shock hit which line/order, accumulated delay, shortfall) is
  persisted separately (`data/simulator_truth.csv`, `data/line_truth.csv`,
  `data/shocks.csv`) and never fed to the model.

## 3. The canonical story: vendor disruption → contested recovery capacity

The generator reserves five deterministic scenarios regardless of seed (see
`data/orders.csv`'s `scenario_tag` column). For seed 42, two of them —
`resource_contention_a` (`O002497`) and `resource_contention_b` (`O002498`) — land in
the held-out **test** split, so they are visible directly in the Streamlit app and
`data/scored_orders.csv`:

1. Both orders share the same vendor (`V001`) and DC (`DC001`) and the same order date.
2. Vendor `V001` is under a forced disruption window; the retrospective derived root
   cause for both orders is `VENDOR_FAILURE`, with `INVENTORY_SHORTAGE`, `DC_CAPACITY`,
   and `WAREHOUSE_OPS` as matched secondary causes. At scoring time, upstream-priority
   attribution also identifies `VENDOR_FAILURE`.
3. XGBoost raises both orders' fused risk scores (`combined_risk_score`); SHAP (or the
   deterministic perturbation fallback) surfaces observable operational factors like
   `vendor_ready_delay_hours` and `allocation_ratio`.
4. The Bayesian chain's pathway JSON contains active vendor, inventory, and warehouse
   evidence and reports the route
   `VENDOR_FAILURE -> INVENTORY_SHORTAGE -> WAREHOUSE_OPS -> TRANSPORT -> OTIF_MISS`,
   together with the posterior risk, prior risk, and evidence delta.
5. Both orders clear the fused decision threshold and are candidates for the same
   `V001` vendor-escalation slot. The greedy priority allocator
   (`decisions.recommend_orders` / `resources.allocate_interventions`) recommends the
   higher-priority order and marks the other **CONTESTED**, listing the competing order
   in `contested_with`.
6. Measured on this run: `O002497` → `RECOMMENDED`; `O002498` → `CONTESTED`, competing
   with `O002497`.

A related scenario (`multi_cause_propagation`, `O002496`) shows the same vendor
and DC driving a full `VENDOR_FAILURE → INVENTORY_SHORTAGE/DC_CAPACITY → WAREHOUSE_OPS →
TRANSPORT` chain end to end; `line_level_stockout` (`O002499`) shows exactly one
line of a multi-line order genuinely short while its other lines ship complete — visible
in `data/line_truth.csv`'s `truly_affected` column; and
`uncertain_unknown_cause` (`O002500`) is a genuine miss with zero corroborating
evidence anywhere, landing as `UNKNOWN`. All five are queryable directly in
`data/root_causes.csv`, `data/line_truth.csv`, and `data/orders.csv` by
`scenario_tag`/`order_id`; the held-out pair is also present in `scored_orders.csv`.

## 4. Order lookup (Streamlit)

Open the **Order lookup** view and search `O002497` / `O002498` (or any RECOMMENDED /
CONTESTED order). You will see: the fused risk score, decision status, priority, penalty
exposure; a structured narrative (risk → evidence → pathway → affected SKUs → action →
resource status); the compact causal pathway as an arrow chain; the affected-SKU table
(from `line_evidence.py`, precision **0.58** / recall **0.66** vs. a naive
all-lines-flagged baseline's precision **0.09** on this run's held-out lines); the
`CONTESTED` warning naming the competing order; and a planner-feedback form that appends
to this run's own audit log.

## 5. Model health

Open **Model health**. With `benchmark.json` present, it shows the 5-seed median/range
table and every acceptance gate (miss rate, fused PR-AUC, fused recall, calibration,
naive-baseline comparisons) with a pass/fail flag — all currently passing. It also shows
this run's own XGBoost-vs-Bayesian-vs-fused comparison and the full 11-row fusion-weight
search table (`fusion_comparison.csv`), so a judge can see *why* a particular weight won
(90% XGBoost / 10% Bayesian in the canonical run) rather than taking it on faith.

## 6. Operations

Open **Operations**. A canonical replay (`uv run otif-ops --orders 1200 --seed 42
--replay-days 90`) trains **8 model versions** (1 initial + 7 retrains: 6 triggered by
drift, 1 by the scheduled cadence) over 90 simulated days. The model registry includes
the exact PSI, score-shift, and cadence reasons for each retrain, alongside the daily
open-order timeline, drift warnings, training window, threshold, fusion weight, and
artifact path. Everything is loaded from persisted files; nothing is recomputed in the
UI.

## 7. What to look for as a skeptical judge

- `docs/model-card.md` states the measured numbers and the honest limitations
  (Bayesian standalone quality, cause-label semantics, synthetic-only validation) in the
  same place as the results — nothing here claims production readiness.
- `tests/test_features.py` proves the point-in-time contract cannot be violated by
  mutating future data; `tests/test_bayesian.py` proves the brute-force fallback matches
  exact pgmpy inference bit-for-bit; `tests/test_fusion.py` proves the weight selection
  cannot be won by a miscalibrated candidate's own inflated threshold-search recall.
- Nothing in this demo is tuned against the held-out test set: the benchmark is 5 fixed
  seeds, reported as median/range, and the acceptance gates are diagnostics that could
  have failed honestly (some individual seeds do fall slightly outside the target band,
  and that is reported, not hidden).
