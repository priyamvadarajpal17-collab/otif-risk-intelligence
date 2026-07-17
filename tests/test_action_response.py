from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from otif_risk.action_response import (
    ACTION_RESOURCE_TYPE,
    ACTION_TARGET_CAUSES,
    ACTIONS,
    CAUSE_TO_ACTION,
    NO_ACTION,
    deterministic_uniforms,
    simulate_action_response,
    with_avoided_penalty,
)
from otif_risk.contracts import PrototypeConfig
from otif_risk.data import generate_dataset
from otif_risk.features import build_feature_table
from otif_risk.root_causes import calculate_outcomes, derive_root_causes


@pytest.fixture(scope="module")
def small_dataset():
    config = PrototypeConfig(seed=11, n_orders=250)
    dataset = generate_dataset(config)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    return config, dataset, outcomes, causes


@pytest.fixture(scope="module")
def responses(small_dataset):
    config, dataset, outcomes, causes = small_dataset
    return simulate_action_response(dataset, outcomes, causes, seed=config.seed)


def test_deterministic_uniforms_is_a_pure_function_of_its_key():
    a = deterministic_uniforms(1, "O000001", "VENDOR_ESCALATION")
    b = deterministic_uniforms(1, "O000001", "VENDOR_ESCALATION")
    assert np.array_equal(a, b)
    # Different action/order/seed must (with overwhelming probability) differ.
    c = deterministic_uniforms(1, "O000001", "ALTERNATE_TRANSPORT")
    d = deterministic_uniforms(2, "O000001", "VENDOR_ESCALATION")
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_deterministic_uniforms_independent_of_calling_order():
    # Simulate two "iteration orders" by just calling in a different sequence;
    # the result for a given key must not depend on what was drawn before it.
    keys = [("O1", "VENDOR_ESCALATION"), ("O2", "ALTERNATE_TRANSPORT"), ("O1", "VENDOR_ESCALATION")]
    first_pass = [deterministic_uniforms(5, order_id, action) for order_id, action in keys]
    reordered = [deterministic_uniforms(5, order_id, action) for order_id, action in reversed(keys)]
    assert np.array_equal(first_pass[0], first_pass[2])
    assert np.array_equal(first_pass[0], reordered[-1])


def test_no_action_potential_outcome_reproduces_original_outcomes_exactly(small_dataset, responses):
    _, _, outcomes, _ = small_dataset
    no_action = responses.loc[responses["action_code"] == NO_ACTION].set_index("order_id")
    original = outcomes.set_index("order_id")
    joined = no_action.join(original, rsuffix="_original")

    assert (joined["delivered_timestamp"] == joined["delivered_timestamp_original"]).all()
    assert (joined["delivered_qty"] == joined["delivered_qty_original"]).all()
    assert (joined["on_time"] == joined["on_time_original"]).all()
    assert (joined["in_full"] == joined["in_full_original"]).all()
    assert (joined["otif_miss"] == joined["otif_miss_original"]).all()


def test_responses_cover_every_order_and_every_action_plus_no_action(small_dataset, responses):
    _, dataset, _, _ = small_dataset
    expected_rows = len(dataset.orders) * (len(ACTIONS) + 1)
    assert len(responses) == expected_rows
    assert set(responses["action_code"]) == {NO_ACTION, *ACTIONS}
    counts = responses["action_code"].value_counts()
    assert (counts == len(dataset.orders)).all()


def test_order_capture_correction_never_changes_the_simulated_outcome(responses):
    """Documented structural limitation: capture delay is not wired into the
    twin's downstream ship/transit timestamps, so this action can never
    mechanically change delivered timestamp/quantity/otif outcome."""
    occ = responses.loc[responses["action_code"] == "ORDER_CAPTURE_CORRECTION"].set_index(
        "order_id"
    )
    no_action = responses.loc[responses["action_code"] == NO_ACTION].set_index("order_id")
    joined = occ.join(no_action, rsuffix="_no_action")
    assert (joined["otif_miss"] == joined["otif_miss_no_action"]).all()
    assert (joined["on_time"] == joined["on_time_no_action"]).all()
    assert (joined["in_full"] == joined["in_full_no_action"]).all()


def test_target_stage_isolation_vendor_escalation_only_touches_vendor_delay(small_dataset):
    """A vendor escalation must never change customer-delay-driven cause
    signals: reduce vendor delay for a vendor-caused order and its delivered
    quantity must be identical to no_action (this action never touches
    quantity)."""
    config, dataset, outcomes, causes = small_dataset
    responses = simulate_action_response(dataset, outcomes, causes, seed=config.seed)
    vendor_rows = responses.loc[responses["action_code"] == "VENDOR_ESCALATION"]
    no_action = responses.loc[responses["action_code"] == NO_ACTION].set_index("order_id")
    joined = vendor_rows.set_index("order_id").join(no_action, rsuffix="_no_action")
    # Vendor escalation never changes delivered quantity (it targets delay
    # only); compare with a floating-point tolerance since delivered_qty is
    # recomputed via requested_qty * (1 - shortfall_fraction) rather than
    # copied verbatim.
    assert joined["delivered_qty"].to_numpy() == pytest.approx(
        joined["delivered_qty_no_action"].to_numpy(), abs=1e-6
    )


def test_action_response_heterogeneity_varies_by_match_and_severity(small_dataset):
    config, dataset, outcomes, causes = small_dataset
    responses = simulate_action_response(dataset, outcomes, causes, seed=config.seed)
    vendor_rows = responses.loc[responses["action_code"] == "VENDOR_ESCALATION"]
    matched = vendor_rows.loc[vendor_rows["match_score"] == 1.0, "response_probability"]
    unmatched = vendor_rows.loc[vendor_rows["match_score"] == 0.1, "response_probability"]
    assert len(matched) > 0 and len(unmatched) > 0
    # A matched (primary-cause) vendor escalation must be more likely to
    # succeed, on average, than one attempted on an unrelated order.
    assert matched.mean() > unmatched.mean()


def test_includes_adverse_and_no_benefit_outcomes(responses):
    """Not every action succeeds, and not every attempt is harmless."""
    with_penalty = with_avoided_penalty(responses)
    action_rows = with_penalty.loc[with_penalty["action_code"] != NO_ACTION]
    assert (action_rows["avoided_penalty"] < 0).any(), "expected some adverse outcomes"
    assert (action_rows["avoided_penalty"] == 0).any(), "expected some no-benefit outcomes"
    assert (action_rows["avoided_penalty"] > 0).any(), "expected some beneficial outcomes"
    assert (~action_rows["success"]).any(), "expected some failed attempts"


def test_response_probability_is_bounded(responses):
    assert (responses["response_probability"] >= 0.0).all()
    assert (responses["response_probability"] <= 1.0).all()


def test_action_target_causes_and_resource_types_are_fully_mapped():
    assert set(ACTION_TARGET_CAUSES) == set(ACTIONS)
    assert set(ACTION_RESOURCE_TYPE) == set(ACTIONS)
    for action, resource_type in ACTION_RESOURCE_TYPE.items():
        assert resource_type in {"dc", "lane", "vendor", "customer"}, action
    # Every cause with a mapped action must round-trip to one of our actions.
    for cause, action in CAUSE_TO_ACTION.items():
        assert action in ACTIONS
        assert cause in ACTION_TARGET_CAUSES[action]


def test_potential_outcomes_never_enter_the_model_feature_table(small_dataset):
    """Evaluation-only potential-outcome fields must never leak into features."""
    _, dataset, outcomes, causes = small_dataset
    feature_table = build_feature_table(dataset, outcomes, causes)
    forbidden = {
        "action_code",
        "response_probability",
        "success",
        "adverse",
        "realized_penalty",
        "avoided_penalty",
        "no_action_penalty",
        "match_score",
        "mechanism_note",
    }
    assert forbidden.isdisjoint(feature_table.columns)


def test_simulate_action_response_is_row_order_invariant(small_dataset):
    config, dataset, outcomes, causes = small_dataset
    shuffled_outcomes = outcomes.sample(frac=1.0, random_state=99).reset_index(drop=True)
    shuffled_causes = causes.sample(frac=1.0, random_state=123).reset_index(drop=True)

    original = simulate_action_response(dataset, outcomes, causes, seed=config.seed)
    shuffled = simulate_action_response(
        dataset, shuffled_outcomes, shuffled_causes, seed=config.seed
    )

    original_sorted = original.sort_values(["order_id", "action_code"]).reset_index(drop=True)
    shuffled_sorted = shuffled.sort_values(["order_id", "action_code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(original_sorted, shuffled_sorted)


def test_simulate_action_response_is_deterministic_across_repeated_runs(small_dataset):
    config, dataset, outcomes, causes = small_dataset
    first = simulate_action_response(dataset, outcomes, causes, seed=config.seed)
    second = simulate_action_response(dataset, outcomes, causes, seed=config.seed)
    pd.testing.assert_frame_equal(first, second)


def test_with_avoided_penalty_no_action_rows_have_zero_avoided_penalty(responses):
    annotated = with_avoided_penalty(responses)
    no_action = annotated.loc[annotated["action_code"] == NO_ACTION]
    assert (no_action["avoided_penalty"] == 0).all()


def test_failed_no_effect_actions_reproduce_no_action_outcome(responses):
    """A failed, non-adverse action changes no lifecycle input, so it must
    reproduce the ``NO_ACTION`` potential outcome almost exactly -- any
    larger drift would mean an order's recorded outcome was constructed
    inconsistently with the shared lifecycle cascade (`recompute_lifecycle_
    timestamps`) that both ``data.py`` and this module replay."""
    no_action = responses.loc[responses["action_code"] == NO_ACTION].set_index("order_id")
    failed_clean = responses.loc[
        (responses["action_code"] != NO_ACTION)
        & (~responses["success"])
        & (~responses["adverse"])
    ]
    joined = failed_clean.set_index("order_id").join(no_action, rsuffix="_no_action")
    drift_seconds = (
        (joined["delivered_timestamp"] - joined["delivered_timestamp_no_action"])
        .dt.total_seconds()
        .abs()
    )
    assert len(drift_seconds) > 0
    assert (drift_seconds < 60).all()


def test_uncertain_unknown_cause_scenario_is_cascade_consistent():
    """Regression test: ``uncertain_unknown_cause``'s recorded miss must be
    produced by feeding the shared lifecycle cascade (via
    ``unknown_extra_hours``, the one mechanism no action targets), never by
    overwriting ``delivered_timestamp`` after the fact. A post-hoc overwrite
    would silently desynchronize this order's ``NO_ACTION`` outcome from what
    a failed action recomputation produces from this same row's own stored
    delay columns -- previously a ~25 hour drift, i.e. a potential-outcome
    consistency violation for a no-effect action."""
    config = PrototypeConfig(seed=42, n_orders=2500)
    dataset = generate_dataset(config)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    responses = simulate_action_response(dataset, outcomes, causes, seed=config.seed)

    order_id = dataset.orders.loc[
        dataset.orders["scenario_tag"] == "uncertain_unknown_cause", "order_id"
    ].iloc[0]
    order_rows = responses.loc[responses["order_id"] == order_id].set_index("action_code")
    assert outcomes.set_index("order_id").loc[order_id, "otif_miss"] == 1

    no_action_delivered = order_rows.loc[NO_ACTION, "delivered_timestamp"]
    checked_any = False
    for action in ACTIONS:
        row = order_rows.loc[action]
        if row["success"] or row["adverse"]:
            continue
        checked_any = True
        drift = abs((row["delivered_timestamp"] - no_action_delivered).total_seconds())
        assert drift < 60, (action, drift)
    assert checked_any
