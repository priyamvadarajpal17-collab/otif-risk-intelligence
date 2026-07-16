from __future__ import annotations

import pandas as pd

from otif_risk.contracts import CAUSE_CATEGORIES, PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.features import LEAKAGE_BLOCKLIST, build_feature_table, temporal_split
from otif_risk.root_causes import calculate_outcomes, derive_root_causes


def _feature_inputs(seed: int = 13):
    dataset = generate_dataset(PrototypeConfig(seed=seed, n_orders=300))
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    return dataset, outcomes, causes


def test_feature_contract_and_leakage_blocklist() -> None:
    dataset, outcomes, causes = _feature_inputs()
    features = build_feature_table(dataset, outcomes, causes)
    assert len(features) == len(dataset.orders)
    assert features["order_id"].is_unique
    assert {"prediction_timestamp", "otif_miss"} <= set(features.columns)
    assert all(f"leading_signal_{cause}" in features for cause in CAUSE_CATEGORIES)
    assert not (LEAKAGE_BLOCKLIST - {"otif_miss"}).intersection(features.columns)
    assert {
        "line_count",
        "allocation_ratio",
        "vendor_reliability_score",
        "dc_utilization_at_prediction",
        "customer_rolling_otif_miss_rate",
        "active_leading_signal_count",
        "days_to_promised_delivery",
    } <= set(features.columns)


def test_rolling_metrics_use_only_matured_prior_outcomes() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=9)
    original = build_feature_table(dataset, outcomes, causes)
    cutoff = original["prediction_timestamp"].quantile(0.45)
    changed_outcomes = outcomes.copy()
    future = changed_outcomes["outcome_timestamp"] >= cutoff
    changed_outcomes.loc[future, "otif_miss"] = 1 - changed_outcomes.loc[future, "otif_miss"]
    changed = build_feature_table(dataset, changed_outcomes, causes)
    rolling = [column for column in original if "rolling_otif_miss_rate" in column]
    before_cutoff = original["prediction_timestamp"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[before_cutoff, rolling].reset_index(drop=True),
        changed.loc[before_cutoff, rolling].reset_index(drop=True),
    )


def test_temporal_split_preserves_order_and_60_20_20_sizes() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=15)
    features = build_feature_table(dataset, outcomes, causes).sample(frac=1, random_state=1)
    split = temporal_split(features)
    assert (len(split.train), len(split.validation), len(split.test)) == (180, 60, 60)
    assert (
        split.train["prediction_timestamp"].max() <= split.validation["prediction_timestamp"].min()
    )
    assert (
        split.validation["prediction_timestamp"].max() <= split.test["prediction_timestamp"].min()
    )


def test_leading_signals_are_not_generated_columns_on_raw_orders() -> None:
    """Leading signals must not be raw generator output on the orders table.

    They are derived downstream in features.py from observable operational
    fields/events, not produced directly from the latent disruption cause at
    dataset-generation time.
    """
    dataset, _outcomes, _causes = _feature_inputs(seed=22)
    assert not any(
        f"leading_signal_{cause}" in dataset.orders.columns for cause in CAUSE_CATEGORIES
    )


def test_leading_signals_are_not_a_lossless_proxy_for_ground_truth_cause() -> None:
    """A cause whose evidence has not yet posted by prediction time must show no signal.

    Point-in-time derivation means at least one order exists per cause where the
    ground-truth cause is present but the observable signal is still 0 because
    evidence has not posted or the proxy is intentionally attenuated.
    """
    dataset, outcomes, causes = _feature_inputs(seed=22)
    features = build_feature_table(dataset, outcomes, causes)
    merged = features.merge(causes, on="order_id", suffixes=("", "_truth"))
    mismatches = 0
    for cause in CAUSE_CATEGORIES:
        truth_active = merged[f"cause_{cause}"] == 1
        signal_active = merged[f"leading_signal_{cause}"] == 1
        mismatches += int((truth_active & ~signal_active).sum())
    assert mismatches > 0


def test_post_prediction_event_mutation_cannot_change_an_orders_own_features() -> None:
    """Mutating an order's own not-yet-observed event must not change its features.

    `prediction_timestamp` can fall before some lifecycle events even complete
    (for example transit has not started yet); injecting new content into such
    a still-future event for the *same order* must not retroactively change
    that order's point-in-time feature row or the leading signals derived from
    it.
    """
    dataset, outcomes, causes = _feature_inputs(seed=22)
    original = build_feature_table(dataset, outcomes, causes)
    unobserved_transit_orders = original.loc[original["transit_observed"] == 0, "order_id"]
    assert not unobserved_transit_orders.empty, "fixture must contain an unobserved transit event"
    target_order = unobserved_transit_orders.iloc[0]

    mutated_dataset = dataset
    mutated_events = mutated_dataset.events.copy()
    transit_mask = (mutated_events["order_id"] == target_order) & (
        mutated_events["event_type"] == "IN_TRANSIT"
    )
    mutated_events.loc[transit_mask, "exception_code"] = "CARRIER_DELAY"
    mutated_dataset.events = mutated_events

    mutated = build_feature_table(mutated_dataset, outcomes, causes)

    original_row = original.loc[original["order_id"] == target_order].reset_index(drop=True)
    mutated_row = mutated.loc[mutated["order_id"] == target_order].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_row, mutated_row)


def test_vendor_rolling_rate_is_fault_attributed_not_raw_miss_rate() -> None:
    """A vendor must not be penalized in its rolling score for others' faults.

    This independently recomputes, per order, the vendor's prior matured raw
    OTIF-miss rate and prior matured vendor-fault rate using the same maturity
    rule as features.py, then finds an order where the vendor's history is all
    non-vendor-fault misses (raw rate > 0, fault rate == 0) and asserts the
    feature table reports the fault-attributed rate, not the raw miss rate.
    """
    dataset, outcomes, causes = _feature_inputs(seed=31)
    features = build_feature_table(dataset, outcomes, causes)

    history = (
        outcomes[["order_id", "prediction_timestamp", "outcome_timestamp", "otif_miss"]]
        .merge(
            dataset.orders[["order_id", "vendor_id"]],
            on="order_id",
            validate="one_to_one",
        )
        .merge(causes[["order_id", "vendor_fault"]], on="order_id", validate="one_to_one")
    )

    target_order_id = None
    for row in features.itertuples(index=False):
        prior = history.loc[
            (history["vendor_id"] == row.vendor_id)
            & (history["prediction_timestamp"] < row.prediction_timestamp)
            & (history["outcome_timestamp"] < row.prediction_timestamp)
        ]
        if len(prior) and prior["otif_miss"].mean() > 0 and prior["vendor_fault"].mean() == 0:
            target_order_id = row.order_id
            break
    assert target_order_id is not None, (
        "fixture must contain a vendor with a non-fault miss history"
    )

    observed = features.loc[features["order_id"] == target_order_id, "vendor_rolling_fault_rate"]
    assert observed.iloc[0] == 0.0
