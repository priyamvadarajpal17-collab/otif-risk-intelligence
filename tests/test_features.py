from __future__ import annotations

import pandas as pd

from otif_risk.contracts import CAUSE_CATEGORIES, PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.features import (
    LEAKAGE_BLOCKLIST,
    attach_line_evidence_features,
    build_feature_table,
    temporal_split,
)
from otif_risk.root_causes import calculate_outcomes, derive_root_causes


def _feature_inputs(seed: int = 13, n_orders: int = 400):
    dataset = generate_dataset(PrototypeConfig(seed=seed, n_orders=n_orders))
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    return dataset, outcomes, causes


def test_feature_contract_and_leakage_blocklist() -> None:
    dataset, outcomes, causes = _feature_inputs()
    features = build_feature_table(dataset, outcomes, causes)
    assert len(features) == len(dataset.orders)
    assert features["order_id"].is_unique
    assert {"as_of_timestamp", "otif_miss"} <= set(features.columns)
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
        "remaining_slack_hours",
        "hours_since_last_observed_event",
        "missing_event_stage_count",
        "dc_utilization_trend_7d",
    } <= set(features.columns)


def test_rolling_windows_are_present_for_every_entity() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=5)
    features = build_feature_table(dataset, outcomes, causes)
    for entity in ("vendor", "dc", "lane", "customer"):
        for suffix in ("30d", "90d", "all_time"):
            rate_name = "rolling_fault_rate" if entity == "vendor" else "rolling_otif_miss_rate"
            assert f"{entity}_{rate_name}_{suffix}" in features.columns
            assert f"{entity}_prior_matured_orders_{suffix}" in features.columns
    for suffix in ("30d", "90d", "all_time"):
        assert f"sku_rolling_shortfall_rate_max_{suffix}" in features.columns
        assert f"sku_rolling_shortfall_rate_mean_{suffix}" in features.columns


def test_simulator_truth_never_changes_model_features() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=6)
    original = build_feature_table(dataset, outcomes, causes)

    mutated_truth = dataset.line_truth.copy()
    mutated_truth["truly_affected"] = 1 - mutated_truth["truly_affected"].astype(int)
    dataset.line_truth = mutated_truth
    mutated = build_feature_table(dataset, outcomes, causes)

    pd.testing.assert_frame_equal(original, mutated)


def test_as_of_timestamp_defaults_to_each_orders_own_prediction_timestamp() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=9)
    features = build_feature_table(dataset, outcomes, causes)
    own_prediction = dataset.orders.set_index("order_id")["prediction_timestamp"]
    aligned = own_prediction.reindex(features["order_id"]).to_numpy()
    assert (features["as_of_timestamp"].to_numpy() == aligned).all()


def test_explicit_as_of_timestamp_scores_open_orders_as_of_one_shared_day() -> None:
    """Daily replay scores every open order as of one explicit "today"."""
    dataset, outcomes, causes = _feature_inputs(seed=17)
    # Pick a as-of date comfortably after most orders' own capture.
    shared_as_of = dataset.orders["order_date"].max()
    open_orders = dataset.orders.loc[dataset.orders["order_date"] <= shared_as_of, "order_id"]
    features = build_feature_table(
        dataset, outcomes, causes, as_of_timestamp=shared_as_of, order_ids=open_orders
    )
    assert (features["as_of_timestamp"] == shared_as_of).all()
    assert set(features["order_id"]) == set(open_orders)


def test_as_of_timestamp_before_order_date_is_rejected() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=21)
    too_early = dataset.orders["order_date"].min() - pd.Timedelta(days=1)
    try:
        build_feature_table(dataset, outcomes, causes, as_of_timestamp=too_early)
    except ValueError as error:
        assert "precedes" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for as_of before order_date")


def test_rolling_metrics_use_only_matured_prior_outcomes() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=9)
    original = build_feature_table(dataset, outcomes, causes)
    cutoff = original["as_of_timestamp"].quantile(0.45)
    changed_outcomes = outcomes.copy()
    future = changed_outcomes["outcome_timestamp"] >= cutoff
    changed_outcomes.loc[future, "otif_miss"] = 1 - changed_outcomes.loc[future, "otif_miss"]
    changed = build_feature_table(dataset, changed_outcomes, causes)
    rolling = [column for column in original if "rolling_otif_miss_rate" in column]
    before_cutoff = original["as_of_timestamp"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[before_cutoff, rolling].reset_index(drop=True),
        changed.loc[before_cutoff, rolling].reset_index(drop=True),
    )


def test_future_event_mutation_cannot_change_an_earlier_as_of_feature_row() -> None:
    """Mutating any future event/outcome must not change an earlier snapshot's row.

    This is the strengthened leakage test: it mutates *all* future events
    (not just one order's) and asserts every row at/before the cutoff is
    byte-for-byte identical.
    """
    dataset, outcomes, causes = _feature_inputs(seed=22)
    original = build_feature_table(dataset, outcomes, causes)
    cutoff = original["as_of_timestamp"].quantile(0.5)

    mutated_events = dataset.events.copy()
    future_event_mask = mutated_events["event_timestamp"] > cutoff
    assert future_event_mask.any(), "fixture must contain future events to mutate"
    mutated_events.loc[future_event_mask, "exception_code"] = "INJECTED_FUTURE_EXCEPTION"
    mutated_events.loc[future_event_mask, "event_timestamp"] = mutated_events.loc[
        future_event_mask, "event_timestamp"
    ] - pd.Timedelta(days=1000)

    mutated_outcomes = outcomes.copy()
    future_outcome_mask = mutated_outcomes["outcome_timestamp"] > cutoff
    mutated_outcomes.loc[future_outcome_mask, "otif_miss"] = 1 - mutated_outcomes.loc[
        future_outcome_mask, "otif_miss"
    ]

    dataset.events = mutated_events
    mutated_features = build_feature_table(dataset, mutated_outcomes, causes)

    # Only compare orders whose *own* label was not directly flipped by the
    # mutation above (that would trivially change that order's own row,
    # which is expected -- not a leak from some other order). An order can
    # have as_of_timestamp <= cutoff yet still resolve after cutoff, so
    # "before cutoff" here means *matured* strictly before the cutoff.
    safe_order_ids = outcomes.loc[outcomes["outcome_timestamp"] <= cutoff, "order_id"]
    before_cutoff = (original["as_of_timestamp"] <= cutoff) & (
        original["order_id"].isin(set(safe_order_ids))
    )
    original_rows = original.loc[before_cutoff].reset_index(drop=True)
    assert len(original_rows) > 0, "fixture must contain at least one matured before-cutoff order"
    mutated_rows = mutated_features.loc[
        mutated_features["order_id"].isin(original_rows["order_id"])
    ].sort_values(["as_of_timestamp", "order_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(original_rows, mutated_rows)


def test_temporal_split_groups_identical_timestamps_together() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=15, n_orders=300)
    features = build_feature_table(dataset, outcomes, causes).sample(frac=1, random_state=1)
    split = temporal_split(features)
    assert len(split.train) + len(split.validation) + len(split.test) == len(features)
    assert split.train["as_of_timestamp"].max() <= split.validation["as_of_timestamp"].min()
    assert split.validation["as_of_timestamp"].max() <= split.test["as_of_timestamp"].min()
    # No timestamp value may appear on both sides of a boundary.
    train_times = set(split.train["as_of_timestamp"])
    validation_times = set(split.validation["as_of_timestamp"])
    test_times = set(split.test["as_of_timestamp"])
    assert not (train_times & validation_times)
    assert not (validation_times & test_times)
    # Roughly 60/20/20 (timestamp grouping means it won't be exact).
    total = len(features)
    assert 0.45 * total <= len(split.train) <= 0.75 * total
    assert len(split.validation) > 0
    assert len(split.test) > 0


def test_leading_signals_are_not_generated_columns_on_raw_orders() -> None:
    dataset, _outcomes, _causes = _feature_inputs(seed=22)
    assert not any(
        f"leading_signal_{cause}" in dataset.orders.columns for cause in CAUSE_CATEGORIES
    )


def test_leading_signals_are_not_a_lossless_proxy_for_ground_truth_cause() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=22)
    features = build_feature_table(dataset, outcomes, causes)
    merged = features.merge(causes, on="order_id", suffixes=("", "_truth"))
    mismatches = 0
    for cause in CAUSE_CATEGORIES:
        truth_active = merged[f"cause_{cause}"] == 1
        signal_active = merged[f"leading_signal_{cause}"] == 1
        mismatches += int((truth_active & ~signal_active).sum())
    assert mismatches > 0


def test_vendor_rolling_rate_is_fault_attributed_not_raw_miss_rate() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=31, n_orders=600)
    features = build_feature_table(dataset, outcomes, causes)

    history = (
        outcomes[["order_id", "prediction_timestamp", "outcome_timestamp", "otif_miss"]]
        .merge(dataset.orders[["order_id", "vendor_id"]], on="order_id", validate="one_to_one")
        .merge(causes[["order_id", "vendor_fault"]], on="order_id", validate="one_to_one")
    )

    target_order_id = None
    for row in features.itertuples(index=False):
        prior = history.loc[
            (history["vendor_id"] == row.vendor_id)
            & (history["prediction_timestamp"] < row.as_of_timestamp)
            & (history["outcome_timestamp"] < row.as_of_timestamp)
        ]
        if len(prior) and prior["otif_miss"].mean() > 0 and prior["vendor_fault"].mean() == 0:
            target_order_id = row.order_id
            break
    assert target_order_id is not None, (
        "fixture must contain a vendor with a non-fault miss history"
    )

    observed = features.loc[
        features["order_id"] == target_order_id, "vendor_rolling_fault_rate_all_time"
    ]
    assert observed.iloc[0] == 0.0


def test_attach_line_evidence_features_adds_safe_order_aggregates() -> None:
    dataset, outcomes, causes = _feature_inputs(seed=3)
    features = build_feature_table(dataset, outcomes, causes)
    enriched = attach_line_evidence_features(dataset, features)
    for column in (
        "worst_line_shortage_ratio",
        "affected_line_count",
        "max_line_risk_evidence",
        "critical_sku_share",
        "line_qty_concentration",
    ):
        assert column in enriched.columns
        assert enriched[column].notna().all()
    assert len(enriched) == len(features)
