from __future__ import annotations

import json

from otif_risk.contracts import CAUSE_CATEGORIES, PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.root_causes import calculate_outcomes, derive_root_causes


def test_outcomes_are_one_row_per_order_and_match_components() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=21, n_orders=300))
    outcomes = calculate_outcomes(dataset)
    assert len(outcomes) == len(dataset.orders)
    assert outcomes["order_id"].is_unique
    expected = ((outcomes["on_time"] == 0) | (outcomes["in_full"] == 0)).astype(int)
    assert outcomes["otif_miss"].equals(expected)


def test_all_rules_are_evaluated_and_multicauses_are_retained() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=3, n_orders=600))
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    binary_columns = [f"cause_{cause}" for cause in CAUSE_CATEGORIES]
    assert set(binary_columns) <= set(causes.columns)
    multi = causes.loc[causes[binary_columns].sum(axis=1) > 1]
    assert not multi.empty
    row = multi.iloc[0]
    assert json.loads(row["secondary_causes"])
    matched = [cause for cause in CAUSE_CATEGORIES if row[f"cause_{cause}"]]
    assert row["primary_cause"] == matched[0]


def test_unexplained_miss_is_unknown() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=5, n_orders=300))
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    unknown = causes.loc[causes["primary_cause"] == "UNKNOWN"]
    assert not unknown.empty
    assert unknown.filter(like="cause_").sum(axis=1).eq(0).all()
    assert unknown["confidence"].lt(0.5).all()
