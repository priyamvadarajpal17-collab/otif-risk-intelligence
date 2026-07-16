from __future__ import annotations

import pandas as pd
import pytest

from otif_pdf.contracts import PrototypeConfig
from otif_pdf.data import generate_dataset
from otif_pdf.root_causes import calculate_outcomes
from otif_pdf.validation import validate_dataset


def test_generator_is_reproducible_and_has_expected_miss_rate() -> None:
    config = PrototypeConfig(seed=17, n_orders=500)
    first = generate_dataset(config)
    second = generate_dataset(config)
    for table_name in first.tables():
        pd.testing.assert_frame_equal(first.tables()[table_name], second.tables()[table_name])

    miss_rate = calculate_outcomes(first)["otif_miss"].mean()
    assert 0.15 <= miss_rate <= 0.25


def test_normalized_references_are_valid() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=4, n_orders=250))
    validate_dataset(dataset)
    assert set(dataset.order_lines["order_id"]) == set(dataset.orders["order_id"])
    assert set(dataset.events["order_id"]) == set(dataset.orders["order_id"])
    assert set(dataset.orders["vendor_id"]) <= set(dataset.vendors["vendor_id"])
    assert set(dataset.orders["dc_id"]) <= set(dataset.dcs["dc_id"])


def test_validation_fails_fast_on_broken_reference() -> None:
    dataset = generate_dataset(PrototypeConfig(seed=8, n_orders=200))
    dataset.orders.loc[0, "vendor_id"] = "MISSING"
    with pytest.raises(ValueError, match="unknown vendor_id"):
        validate_dataset(dataset)
