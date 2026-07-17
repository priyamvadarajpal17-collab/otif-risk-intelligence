from __future__ import annotations

import pandas as pd
import pytest

from otif_risk.contracts import PrototypeConfig
from otif_risk.data import SCENARIO_TAGS, generate_dataset
from otif_risk.root_causes import calculate_outcomes, derive_root_causes
from otif_risk.validation import validate_dataset


def test_generator_is_reproducible_and_has_expected_miss_rate() -> None:
    config = PrototypeConfig(seed=17, n_orders=500)
    first = generate_dataset(config)
    second = generate_dataset(config)
    for table_name in first.tables():
        pd.testing.assert_frame_equal(first.tables()[table_name], second.tables()[table_name])

    miss_rate = calculate_outcomes(first)["otif_miss"].mean()
    assert 0.15 <= miss_rate <= 0.25


def test_miss_rate_stays_in_range_across_several_seeds() -> None:
    for seed in (1, 2, 3, 4, 5):
        dataset = generate_dataset(PrototypeConfig(seed=seed, n_orders=1500))
        miss_rate = calculate_outcomes(dataset)["otif_miss"].mean()
        assert 0.12 <= miss_rate <= 0.28, f"seed {seed} miss rate {miss_rate} out of range"


def test_normalized_references_are_valid() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=4, n_orders=250))
    validate_dataset(dataset)
    assert set(dataset.order_lines["order_id"]) == set(dataset.orders["order_id"])
    assert set(dataset.events["order_id"]) == set(dataset.orders["order_id"])
    assert set(dataset.orders["vendor_id"]) <= set(dataset.vendors["vendor_id"])
    assert set(dataset.orders["dc_id"]) <= set(dataset.dcs["dc_id"])
    assert set(dataset.order_lines["sku_id"]) <= set(dataset.skus["sku_id"])


def test_validation_fails_fast_on_broken_reference() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=8, n_orders=200))
    dataset.orders.loc[0, "vendor_id"] = "MISSING"
    with pytest.raises(ValueError, match="unknown vendor_id"):
        validate_dataset(dataset)


def test_simulator_truth_is_separate_from_model_facing_tables() -> None:
    """Evaluation-only ground truth must never be part of the model-facing tables."""
    dataset = generate_dataset(PrototypeConfig(seed=6, n_orders=250))
    assert "simulator_truth" not in dataset.tables()
    assert "line_truth" not in dataset.tables()
    assert "shocks" not in dataset.tables()
    truth = dataset.truth_tables()
    assert {"simulator_truth", "line_truth", "shocks"} == set(truth)
    assert set(truth["simulator_truth"]["order_id"]) == set(dataset.orders["order_id"])
    assert not set(dataset.orders.columns) & {
        "vendor_shock_active",
        "accumulated_delay_hours",
        "unknown_extra_hours",
    }


def test_named_demo_scenarios_are_deterministically_present() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=99, n_orders=800))
    tagged = dataset.orders.loc[dataset.orders["scenario_tag"] != ""]
    assert set(tagged["scenario_tag"]) == set(SCENARIO_TAGS)
    assert len(tagged) == len(SCENARIO_TAGS)

    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    merged = tagged.merge(outcomes, on="order_id").merge(causes, on="order_id")
    by_tag = merged.set_index("scenario_tag")

    # Multi-cause propagation: the full chain must actually be present.
    multi = by_tag.loc["multi_cause_propagation"]
    assert multi["otif_miss"] == 1

    # Resource contention pair: same vendor/dc, same order_date, both miss.
    contention_a = by_tag.loc["resource_contention_a"]
    contention_b = by_tag.loc["resource_contention_b"]
    assert contention_a["vendor_id"] == contention_b["vendor_id"]
    assert contention_a["dc_id"] == contention_b["dc_id"]
    assert contention_a["order_date"] == contention_b["order_date"]
    assert contention_a["otif_miss"] == 1
    assert contention_b["otif_miss"] == 1

    # Line-level stockout: order misses due to inventory, not every line affected.
    line_level_order_id = merged.loc[
        merged["scenario_tag"] == "line_level_stockout", "order_id"
    ].iloc[0]
    line_level = by_tag.loc["line_level_stockout"]
    assert line_level["otif_miss"] == 1
    line_truth = dataset.line_truth.loc[dataset.line_truth["order_id"] == line_level_order_id]
    assert 1 <= line_truth["truly_affected"].sum() < len(line_truth)

    # Uncertain/unknown cause: a genuine miss with no observable rule evidence.
    unknown = by_tag.loc["uncertain_unknown_cause"]
    assert unknown["otif_miss"] == 1
    assert unknown["primary_cause"] == "UNKNOWN"


def test_scenario_slice_does_not_make_the_whole_dataset_deterministic() -> None:
    """Only the reserved scenario slice is scripted; the rest stays seed-random."""
    first = generate_dataset(PrototypeConfig(seed=101, n_orders=500))
    second = generate_dataset(PrototypeConfig(seed=202, n_orders=500))
    non_scenario_first = first.orders.loc[first.orders["scenario_tag"] == "", "vendor_id"]
    non_scenario_second = second.orders.loc[second.orders["scenario_tag"] == "", "vendor_id"]
    assert not non_scenario_first.reset_index(drop=True).equals(
        non_scenario_second.reset_index(drop=True)
    )
