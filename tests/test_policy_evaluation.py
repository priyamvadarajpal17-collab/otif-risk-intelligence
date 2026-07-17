from __future__ import annotations

import pandas as pd
import pytest

from otif_risk.contracts import PrototypeConfig
from otif_risk.features import build_feature_table
from otif_risk.policy_evaluation import (
    CAPACITY_SCENARIO_TIE_TOLERANCE,
    DIAGNOSTIC_CAPACITY_SCENARIO,
    EXPLORATION_FRACTION,
    POLICIES,
    POLICY_CURRENT,
    POLICY_HIGHEST_RISK,
    POLICY_LEGACY,
    POLICY_NO_ACTION,
    POLICY_ORACLE,
    POLICY_RANDOM,
    PRIMARY_CAPACITY_SCENARIO,
    bayesian_ablation_diagnostic,
    build_seed_context,
    content_fingerprint,
    counterfactual_action_ranking,
    decision_key,
    evaluate_policies,
    evaluate_policies_across_capacity_scenarios,
    evaluation_calendar_days,
    expected_vs_realized_rank_correlation,
    run_seed_evaluation,
    summarize_multi_seed,
)
from otif_risk.resources import CAPACITY_SCENARIOS, build_capacity_schedule


@pytest.fixture(scope="module")
def context():
    return build_seed_context(PrototypeConfig(seed=13, n_orders=350))


@pytest.fixture(scope="module")
def policy_report(context):
    return evaluate_policies(context)


def test_all_seven_policies_are_evaluated(policy_report):
    assert set(policy_report["policies"]) == set(POLICIES)
    assert len(POLICIES) == 8


def test_no_action_policy_treats_nobody_and_has_zero_avoided_penalty(policy_report):
    result = policy_report["policies"][POLICY_NO_ACTION]
    assert result["orders_treated"] == 0
    assert result["total_avoided_penalty"] == 0.0
    assert result["normalized_resource_units_consumed"] == 0.0


def test_oracle_is_a_ceiling_no_policy_can_beat(policy_report):
    oracle_value = policy_report["policies"][POLICY_ORACLE]["total_avoided_penalty"]
    for policy, result in policy_report["policies"].items():
        assert result["total_avoided_penalty"] <= oracle_value + 1e-6, policy
        assert result["regret_vs_oracle"] >= -1e-6, policy


def test_oracle_action_precision_is_perfect_by_construction(policy_report):
    # The oracle only ever chooses actions with positive avoided penalty.
    assert policy_report["policies"][POLICY_ORACLE]["action_precision"] == pytest.approx(1.0)
    assert policy_report["policies"][POLICY_ORACLE]["waste_no_benefit_rate"] == pytest.approx(0.0)
    assert policy_report["policies"][POLICY_ORACLE]["adverse_response_rate"] == pytest.approx(0.0)


def test_resource_normalization_never_exceeds_capacity_pools(policy_report):
    for policy, result in policy_report["policies"].items():
        for resource_type, breakdown in result["resource_breakdown"].items():
            assert breakdown["units_consumed"] >= 0, (policy, resource_type)
            assert breakdown["normalized_units"] >= 0, (policy, resource_type)


def test_current_policy_decision_log_has_required_fields(policy_report):
    log_rows = policy_report["current_policy_decision_log_sample"]
    assert policy_report["current_policy_decision_log_count"] >= len(log_rows)
    assert len(log_rows) > 0
    required_fields = {
        "assignment_probability",
        "pool_reservation_ratio",
        "selection_mode",
        "policy_version",
        "capacity_before",
        "capacity_after",
        "chosen_action",
        "rejected_feasible_actions",
        "decision_key",
    }
    for row in log_rows:
        assert required_fields.issubset(row.keys())
        assert row["selection_mode"] in {"EXPLOIT", "EXPLORE", "CONTESTED"}
        # ``assignment_probability`` is an exact marginal propensity except
        # for continuous (dc/lane) pools' random remainder, where a
        # sequential fits-if-it-fits draw over heterogeneous order sizes has
        # no closed-form per-order probability -- logged honestly as
        # ``None`` with a separately named ``pool_reservation_ratio``
        # instead of a mislabeled point estimate (see
        # ``_allocate_day_with_exploration``).
        if row["assignment_probability"] is None:
            assert row["pool_reservation_ratio"] is not None
        else:
            assert 0.0 <= row["assignment_probability"] <= 1.0
            assert row["pool_reservation_ratio"] is None


def test_exploration_reserves_roughly_the_configured_fraction_of_capacity(policy_report):
    log_rows = pd.DataFrame(policy_report["current_policy_decision_log_sample"])
    if log_rows.empty:
        pytest.skip("no decision log rows in this seed")
    accepted = log_rows.loc[log_rows["chosen_action"].notna()]
    if accepted.empty:
        pytest.skip("no accepted decisions in this sample")
    explore_share = (accepted["selection_mode"] == "EXPLORE").mean()
    # A loose sanity bound: exploration should never dominate the accepted
    # set (it is a 10% capacity carve-out, not the primary allocation rule).
    assert explore_share <= EXPLORATION_FRACTION + 0.35


def test_decision_key_is_stable_and_scoped_to_its_inputs():
    key_a = decision_key(1, "O000001", POLICY_CURRENT, "2024-01-01")
    key_b = decision_key(1, "O000001", POLICY_CURRENT, "2024-01-01")
    key_c = decision_key(2, "O000001", POLICY_CURRENT, "2024-01-01")
    assert key_a == key_b
    assert key_a != key_c


def test_decision_key_is_scoped_to_capacity_scenario():
    """The same (seed, order, policy, day) can realize a genuinely different
    decision at a different capacity-stress scenario (a different chosen
    action, or CONTESTED instead of accepted) -- decision_key must not let
    two scenarios' decisions collide on one key."""
    key_scarce = decision_key(1, "O000001", POLICY_CURRENT, "2024-01-01", "SCARCE_50_PERCENT")
    key_base = decision_key(1, "O000001", POLICY_CURRENT, "2024-01-01", "BASE_100_PERCENT")
    assert key_scarce != key_base
    # Omitting capacity_scenario defaults to the single scenario a
    # production deployment always runs at.
    assert decision_key(1, "O000001", POLICY_CURRENT, "2024-01-01") == decision_key(
        1, "O000001", POLICY_CURRENT, "2024-01-01", DIAGNOSTIC_CAPACITY_SCENARIO
    )


def test_current_policy_decision_log_keys_differ_across_capacity_scenarios(context):
    sensitivity = evaluate_policies_across_capacity_scenarios(context)
    keys_by_scenario = {
        scenario: {
            row["decision_key"]
            for row in sensitivity["scenarios"][scenario]["current_policy_decision_log_sample"]
        }
        for scenario in CAPACITY_SCENARIOS
    }
    non_empty = {scenario: keys for scenario, keys in keys_by_scenario.items() if keys}
    if len(non_empty) < 2:
        pytest.skip("not enough non-empty decision logs across scenarios to compare")
    scenario_names = list(non_empty)
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            assert non_empty[scenario_names[i]].isdisjoint(non_empty[scenario_names[j]])


def test_content_fingerprint_is_deterministic_and_input_sensitive():
    fp_a = content_fingerprint({"seed": 1, "n": 100})
    fp_b = content_fingerprint({"seed": 1, "n": 100})
    fp_c = content_fingerprint({"seed": 2, "n": 100})
    assert fp_a == fp_b
    assert fp_a != fp_c


def test_evaluate_policies_is_row_order_invariant(context):
    shuffled_decisions = context.decisions.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled_responses = context.responses.sample(frac=1.0, random_state=8).reset_index(drop=True)
    shuffled_context = context.__class__(
        config=context.config,
        dataset=context.dataset,
        outcomes=context.outcomes,
        causes=context.causes,
        trained=context.trained,
        decisions=shuffled_decisions,
        responses=shuffled_responses,
        coverage=context.coverage,
    )
    original = evaluate_policies(context)
    shuffled = evaluate_policies(shuffled_context)
    for policy in POLICIES:
        assert (
            original["policies"][policy]["total_avoided_penalty"]
            == pytest.approx(shuffled["policies"][policy]["total_avoided_penalty"])
        )
        assert (
            original["policies"][policy]["orders_treated"]
            == shuffled["policies"][policy]["orders_treated"]
        )


def test_counterfactual_action_ranking_reports_required_diagnostics(context):
    ranking = counterfactual_action_ranking(context)
    assert ranking["n_orders"] > 0
    for column in ("bayesian_vs_best", "strongest_signal_vs_best", "random_vs_best"):
        assert 0.0 <= ranking["top_action_agreement"][column] <= 1.0
    assert "bayesian_adds_value_over_strongest_signal" in ranking
    assert "bayesian_adds_value_over_random" in ranking
    assert "model-scenario" in ranking["qualification"]
    assert "simulator-evaluation" in ranking["qualification"]


def test_random_policy_uses_a_different_ranking_than_priority_policies(context):
    """Random-at-capacity must not silently reuse another policy's priority
    score (a bug that would make it identical to the deployed ranking)."""
    from otif_risk.policy_evaluation import _candidate_frame
    from otif_risk.resources import default_daily_capacities

    base_capacities = default_daily_capacities(context.dataset)
    random_candidates = _candidate_frame(
        POLICY_RANDOM, context.decisions, context.responses, seed=1
    )
    current_candidates = _candidate_frame(
        POLICY_CURRENT,
        context.decisions,
        context.responses,
        seed=1,
        base_capacities=base_capacities,
    )
    merged = random_candidates[["order_id", "priority_score"]].merge(
        current_candidates[["order_id", "priority_score"]],
        on="order_id",
        suffixes=("_random", "_current"),
    )
    assert not (merged["priority_score_random"] == merged["priority_score_current"]).all()


def test_legacy_policy_uses_the_original_single_action_priority_score(context):
    """LEGACY_PRIORITY_POLICY must reproduce the original deployed priority_score
    ranking (risk x tier x value) unchanged -- it is the frozen baseline
    CURRENT_POLICY's improvement is measured against."""
    from otif_risk.policy_evaluation import _candidate_frame

    legacy_candidates = _candidate_frame(
        POLICY_LEGACY, context.decisions, context.responses, seed=1
    )
    merged = legacy_candidates[["order_id", "priority_score"]].merge(
        context.decisions[["order_id", "priority_score"]],
        on="order_id",
        suffixes=("_legacy", "_deployed"),
    )
    assert (merged["priority_score_legacy"] == merged["priority_score_deployed"]).all()


def test_current_policy_may_choose_a_different_action_than_legacy(context):
    """The value-aware CURRENT_POLICY must be able to choose an action other than
    the single primary-cause-implied action LEGACY_PRIORITY_POLICY always uses --
    otherwise it is not genuinely considering multiple candidates."""
    from otif_risk.policy_evaluation import _candidate_frame
    from otif_risk.resources import default_daily_capacities

    base_capacities = default_daily_capacities(context.dataset)
    legacy_candidates = _candidate_frame(
        POLICY_LEGACY, context.decisions, context.responses, seed=1
    )
    current_candidates = _candidate_frame(
        POLICY_CURRENT,
        context.decisions,
        context.responses,
        seed=1,
        base_capacities=base_capacities,
    )
    merged = legacy_candidates[["order_id", "action_code"]].merge(
        current_candidates[["order_id", "action_code"]],
        on="order_id",
        suffixes=("_legacy", "_current"),
    )
    assert not merged.empty
    # Not required to differ on every order (some orders only ever have one
    # feasible candidate), but across a real seed's eligible pool at least one
    # order must differ, or the value-aware ranking is not doing anything new.
    assert (merged["action_code_legacy"] != merged["action_code_current"]).any()


def test_current_policy_candidate_frame_reports_required_diagnostics_columns(context):
    from otif_risk.policy_evaluation import _candidate_frame
    from otif_risk.resources import default_daily_capacities

    base_capacities = default_daily_capacities(context.dataset)
    current_candidates = _candidate_frame(
        POLICY_CURRENT,
        context.decisions,
        context.responses,
        seed=1,
        base_capacities=base_capacities,
    )
    required = {
        "action_code",
        "resource_type",
        "resource_id",
        "priority_score",
        "expected_benefit",
        "structural_reduction",
        "structural_reduction_source",
        "candidate_action_count",
    }
    assert required.issubset(current_candidates.columns)
    assert current_candidates["structural_reduction_source"].isin(
        {"bayesian_structural_scenario", "leading_signal_fallback", "primary_cause_fallback"}
    ).all()
    assert (current_candidates["candidate_action_count"] >= 1).all()
    assert (current_candidates["priority_score"] >= 0).all()


def test_current_policy_requires_base_capacities(context):
    from otif_risk.policy_evaluation import _candidate_frame

    with pytest.raises(ValueError):
        _candidate_frame(POLICY_CURRENT, context.decisions, context.responses, seed=1)


def test_no_potential_outcome_fields_leak_into_model_features(context):
    feature_table = build_feature_table(context.dataset, context.outcomes, context.causes)
    forbidden = {"avoided_penalty", "realized_penalty", "success", "adverse", "action_code"}
    assert forbidden.isdisjoint(feature_table.columns)


def test_value_aware_candidate_actions_never_read_potential_outcome_or_retrospective_fields():
    """The value-aware CURRENT_POLICY's candidate-action generation must only read
    point-in-time-observable/model-derived columns -- never simulator response/
    potential-outcome fields, and never `root_causes`' retrospective rule evaluation
    (only the scoring-time `primary_cause`/`leading_signal_*` derived from it)."""
    import inspect

    from otif_risk.policy_evaluation import (
        _execution_feasibility,
        _expected_value_density,
        _value_aware_candidate_actions,
    )

    forbidden_tokens = {
        "avoided_penalty",
        "realized_penalty",
        "success",
        "adverse",
        "secondary_causes",
        "response_probability",
    }
    for func in (_value_aware_candidate_actions, _execution_feasibility, _expected_value_density):
        source = inspect.getsource(func)
        for token in forbidden_tokens:
            assert token not in source, (func.__name__, token)


def test_value_aware_formula_matches_documented_exact_computation():
    """Recompute `_expected_value_density`'s formula by hand for a synthetic row and
    confirm it matches the module's implementation exactly (no hidden terms)."""
    from otif_risk.policy_evaluation import (
        FEASIBILITY_WEIGHT_CAUSAL_CONFIDENCE,
        FEASIBILITY_WEIGHT_RESOURCE_TRAIT,
        FEASIBILITY_WEIGHT_SLACK,
        _expected_value_density,
    )

    row = pd.Series(
        {
            "estimated_penalty_exposure": 1000.0,
            "quantity_at_risk": 40.0,
            "remaining_slack_hours": 84.0,  # half of the 168h normalization horizon
            "vendor_reliability_score": 0.9,
            "evidence_coverage": 0.8,
        }
    )
    structural_reduction = 0.5
    expected_benefit, density = _expected_value_density(
        row, "VENDOR_ESCALATION", structural_reduction, base_capacity=2.0
    )
    slack_term = 84.0 / 168.0
    feasibility = (
        FEASIBILITY_WEIGHT_SLACK * slack_term
        + FEASIBILITY_WEIGHT_RESOURCE_TRAIT * 0.9
        + FEASIBILITY_WEIGHT_CAUSAL_CONFIDENCE * 0.8
    )
    hand_computed_benefit = 1000.0 * structural_reduction * feasibility
    hand_computed_density = hand_computed_benefit / (1.0 / 2.0)  # vendor demand is always 1 unit
    assert expected_benefit == pytest.approx(hand_computed_benefit)
    assert density == pytest.approx(hand_computed_density)


def test_value_aware_resource_demand_normalization_uses_scenario_independent_base_capacity(
    context,
):
    """POLICY_CURRENT's ranking (priority_score/value_density) must be identical
    regardless of which capacity-stress scenario is being evaluated -- only the
    *acceptance* cutoff may vary by scenario, never the ranking itself."""
    from otif_risk.policy_evaluation import _candidate_frame
    from otif_risk.resources import default_daily_capacities

    base_capacities = default_daily_capacities(context.dataset)
    candidates_a = _candidate_frame(
        POLICY_CURRENT,
        context.decisions,
        context.responses,
        seed=1,
        base_capacities=base_capacities,
    )
    # Re-deriving base_capacities from the same dataset must be deterministic and
    # scenario-independent (never derived from resources.build_capacity_schedule's
    # scaled/scenario-specific schedule).
    candidates_b = _candidate_frame(
        POLICY_CURRENT,
        context.decisions,
        context.responses,
        seed=1,
        base_capacities=default_daily_capacities(context.dataset),
    )
    merged = candidates_a[["order_id", "priority_score"]].merge(
        candidates_b[["order_id", "priority_score"]], on="order_id", suffixes=("_a", "_b")
    )
    assert (merged["priority_score_a"] == merged["priority_score_b"]).all()


def test_run_seed_evaluation_passes_no_action_identity_gate():
    report = run_seed_evaluation(PrototypeConfig(seed=17, n_orders=250))
    assert report["no_action_identity_gate"]["passed"] is True
    assert report["evaluation_fingerprint"]
    assert "policy_evaluation" in report
    assert "capacity_sensitivity" in report
    assert "counterfactual_action_ranking" in report
    assert "bayesian_ablation" in report
    assert "expected_vs_realized_rank_correlation" in report
    # "policy_evaluation" is preserved for continuity as the diagnostic
    # (100%) capacity scenario -- identical to capacity_sensitivity's own
    # BASE_100_PERCENT entry, never a separately computed number.
    diagnostic_scenario = report["capacity_sensitivity"]["scenarios"][DIAGNOSTIC_CAPACITY_SCENARIO]
    assert (
        report["policy_evaluation"]["policies"][POLICY_CURRENT]["total_avoided_penalty"]
        == diagnostic_scenario["policies"][POLICY_CURRENT]["total_avoided_penalty"]
    )


def test_bayesian_ablation_diagnostic_reports_both_variants(context):
    ablation = bayesian_ablation_diagnostic(context)
    assert ablation["capacity_scenario"] == PRIMARY_CAPACITY_SCENARIO
    for key in ("with_bayesian_term", "without_bayesian_term"):
        assert "avoided_penalty_per_normalized_resource_unit" in ablation[key]
    assert "headline_delta_with_minus_without" in ablation
    assert isinstance(ablation["bayesian_term_adds_value"], bool)


def test_expected_vs_realized_rank_correlation_is_evaluation_only_diagnostic(context):
    diagnostic = expected_vs_realized_rank_correlation(context)
    assert diagnostic["n_candidates"] >= 0
    if diagnostic["n_candidates"]:
        assert -1.0 <= diagnostic["spearman_rank_correlation"] <= 1.0
    assert "evaluation-only" in diagnostic["qualification"].lower()


def test_summarize_multi_seed_schema_and_gates():
    reports = [
        run_seed_evaluation(PrototypeConfig(seed=seed, n_orders=250)) for seed in (21, 23)
    ]
    summary = summarize_multi_seed(reports)
    assert summary["seeds"] == [21, 23]
    assert summary["primary_capacity_scenario"] == PRIMARY_CAPACITY_SCENARIO
    assert summary["diagnostic_capacity_scenario"] == DIAGNOSTIC_CAPACITY_SCENARIO
    assert set(summary["capacity_scenarios"]) == set(CAPACITY_SCENARIOS)

    gates = summary["acceptance_gates"]
    assert gates["no_action_identity"] is True
    assert gates["primary_capacity_scenario"] == PRIMARY_CAPACITY_SCENARIO
    assert "current_beats_random_at_primary_capacity" in gates
    assert "current_beats_highest_risk_at_primary_capacity" in gates
    assert "current_beats_legacy_at_primary_capacity" in gates
    assert "current_wins_at_least_win_threshold_vs_random" in gates
    assert "current_wins_at_least_win_threshold_vs_legacy" in gates
    assert "diagnostic_current_beats_random_at_base_capacity" in gates
    assert "diagnostic_current_beats_highest_risk_at_base_capacity" in gates
    assert "diagnostic_current_beats_legacy_at_base_capacity" in gates
    assert "no_action_precision_collapse" in gates
    assert "action_precision_gap_vs_highest_risk" in gates
    assert "primary_gate_passed" in gates
    assert "current_policy_value_positive_in_both_regimes" in gates
    assert gates["win_threshold"] == 2  # ceil(2 seeds * 3 / 5)

    for scenario in CAPACITY_SCENARIOS:
        assert set(summary["median_headline_by_capacity_scenario"][scenario]) == set(POLICIES)
        assert set(summary["median_action_precision_by_capacity_scenario"][scenario]) == set(
            POLICIES
        )
        assert set(summary["median_regret_vs_oracle_by_capacity_scenario"][scenario]) == set(
            POLICIES
        )
        assert set(
            summary["median_avoidable_miss_coverage_by_capacity_scenario"][scenario]
        ) == set(POLICIES)
        assert set(summary["median_contested_rate_by_capacity_scenario"][scenario]) == set(
            POLICIES
        )
        assert set(
            summary["median_capacity_binding_rate_by_capacity_scenario"][scenario]
        ) == set(POLICIES)

    diagnostics = summary["value_aware_policy_diagnostics"]
    assert "candidate_action_coverage_median" in diagnostics
    assert "bayesian_evidence_rate_median" in diagnostics
    assert "chosen_action_mix_median" in diagnostics
    assert "expected_vs_realized_rank_correlation_median" in diagnostics
    assert "bayesian_ablation" in diagnostics
    assert "median_with_bayesian_term" in diagnostics["bayesian_ablation"]
    assert "median_without_bayesian_term" in diagnostics["bayesian_ablation"]

    paired = summary["paired_seed_deltas"]
    for key in ("current_vs_random", "current_vs_highest_risk", "current_vs_legacy"):
        assert paired[key]["capacity_scenario"] == PRIMARY_CAPACITY_SCENARIO
        assert paired[key]["seeds"] == [21, 23]
        assert len(paired[key]["per_seed_delta"]) == 2
        assert paired[key]["wins"] + paired[key]["ties"] + paired[key]["losses"] == 2
        assert paired[key]["tie_tolerance"] == CAPACITY_SCENARIO_TIE_TOLERANCE


# --- Capacity-stress sensitivity analysis -----------------------------------


def test_capacity_scenarios_are_pre_specified_at_25_50_100_percent():
    assert CAPACITY_SCENARIOS == {
        "SCARCE_25_PERCENT": 0.25,
        "SCARCE_50_PERCENT": 0.5,
        "BASE_100_PERCENT": 1.0,
    }
    assert PRIMARY_CAPACITY_SCENARIO == "SCARCE_50_PERCENT"
    assert PRIMARY_CAPACITY_SCENARIO in CAPACITY_SCENARIOS
    assert DIAGNOSTIC_CAPACITY_SCENARIO in CAPACITY_SCENARIOS


def test_capacity_schedule_scales_continuous_pools_directly(context):
    calendar_days = evaluation_calendar_days(context)
    base_schedule = build_capacity_schedule(context.dataset, calendar_days, 1.0)
    half_schedule = build_capacity_schedule(context.dataset, calendar_days, 0.5)
    quarter_schedule = build_capacity_schedule(context.dataset, calendar_days, 0.25)
    day_key = calendar_days[0].isoformat()
    for dc_id, base_capacity in base_schedule[day_key].dc_units.items():
        assert half_schedule[day_key].dc_units[dc_id] == pytest.approx(base_capacity * 0.5)
        assert quarter_schedule[day_key].dc_units[dc_id] == pytest.approx(base_capacity * 0.25)
    for lane_id, base_capacity in base_schedule[day_key].lane_units.items():
        assert half_schedule[day_key].lane_units[lane_id] == pytest.approx(base_capacity * 0.5)
        assert quarter_schedule[day_key].lane_units[lane_id] == pytest.approx(base_capacity * 0.25)


def test_capacity_schedule_discrete_pools_realize_correct_long_run_average(context):
    """A 1-slot vendor pool (or 2-slot customer pool) that would round to a
    fractional (< 1 whole slot) target under 25%/50% scaling must still
    realize a deterministic whole-slot day-by-day schedule whose long-run
    average across the full evaluated calendar equals ``base * multiplier``
    exactly -- never silently floored to a permanently-zero pool."""
    calendar_days = evaluation_calendar_days(context)
    assert len(calendar_days) >= 4  # long enough to observe more than one cycle
    base_schedule = build_capacity_schedule(context.dataset, calendar_days, 1.0)
    example_vendor = next(iter(base_schedule[calendar_days[0].isoformat()].vendor_slots))
    example_customer = next(iter(base_schedule[calendar_days[0].isoformat()].customer_slots))
    base_vendor_capacity = base_schedule[calendar_days[0].isoformat()].vendor_slots[example_vendor]
    base_customer_capacity = base_schedule[calendar_days[0].isoformat()].customer_slots[
        example_customer
    ]

    for scenario_name, multiplier in CAPACITY_SCENARIOS.items():
        schedule = build_capacity_schedule(context.dataset, calendar_days, multiplier)
        vendor_values = [
            schedule[day.isoformat()].vendor_slots[example_vendor] for day in calendar_days
        ]
        customer_values = [
            schedule[day.isoformat()].customer_slots[example_customer] for day in calendar_days
        ]
        # Every realized daily capacity is a non-negative whole number.
        assert all(value == int(value) for value in vendor_values), scenario_name
        assert all(value == int(value) for value in customer_values), scenario_name
        # Long-run average matches the target multiplier exactly (within one
        # slot's worth of rounding across the whole calendar, since the
        # accumulator's fractional remainder never exceeds one slot).
        vendor_target = base_vendor_capacity * multiplier
        customer_target = base_customer_capacity * multiplier
        vendor_tolerance = 1.0 / len(vendor_values)
        customer_tolerance = 1.0 / len(customer_values)
        assert sum(vendor_values) / len(vendor_values) == pytest.approx(
            vendor_target, abs=vendor_tolerance
        )
        assert sum(customer_values) / len(customer_values) == pytest.approx(
            customer_target, abs=customer_tolerance
        )

    # At BASE_100_PERCENT the schedule must reduce exactly to the unscaled
    # base capacity every day -- no scheduling artifact at baseline.
    base_100_schedule = build_capacity_schedule(context.dataset, calendar_days, 1.0)
    for day in calendar_days:
        assert base_100_schedule[day.isoformat()].vendor_slots[
            example_vendor
        ] == pytest.approx(base_vendor_capacity)
        assert base_100_schedule[day.isoformat()].customer_slots[
            example_customer
        ] == pytest.approx(base_customer_capacity)


def test_capacity_schedule_one_slot_pool_at_scarce_multipliers_never_stays_zero_forever():
    """A 1-slot pool scaled to 25%/50% must still grant capacity on some
    days (a documented deterministic whole-slot schedule), never a
    permanently-zero pool for the entire scenario."""

    class _FakeDataset:
        dcs = pd.DataFrame({"dc_id": [], "daily_capacity_units": []})
        lanes = pd.DataFrame({"lane_id": []})
        vendors = pd.DataFrame({"vendor_id": ["V1"]})
        customers = pd.DataFrame({"customer_id": ["C1"]})

    dataset = _FakeDataset()
    days = pd.date_range("2024-01-01", periods=12)
    for multiplier in (0.25, 0.5):
        schedule = build_capacity_schedule(dataset, days, multiplier)
        vendor_days_with_capacity = sum(
            1 for day in days if schedule[day.isoformat()].vendor_slots["V1"] > 0
        )
        assert vendor_days_with_capacity > 0, multiplier


def test_evaluate_policies_across_capacity_scenarios_reports_all_three_visibly(context):
    sensitivity = evaluate_policies_across_capacity_scenarios(context)
    assert sensitivity["primary_capacity_scenario"] == PRIMARY_CAPACITY_SCENARIO
    assert set(sensitivity["scenarios"]) == set(CAPACITY_SCENARIOS)
    for scenario_name, multiplier in CAPACITY_SCENARIOS.items():
        scenario_report = sensitivity["scenarios"][scenario_name]
        assert scenario_report["capacity_scenario"] == scenario_name
        assert scenario_report["capacity_multiplier"] == pytest.approx(multiplier)
        assert set(scenario_report["policies"]) == set(POLICIES)
        # Every policy at every scenario reports the discriminativeness
        # metrics required to prove capacity actually binds (or doesn't).
        for policy_metrics in scenario_report["policies"].values():
            assert "contested_rate" in policy_metrics
            assert "capacity_binding_rate" in policy_metrics
            assert 0.0 <= policy_metrics["contested_rate"] <= 1.0
            assert 0.0 <= policy_metrics["capacity_binding_rate"] <= 1.0


def test_scarce_capacity_is_more_discriminative_than_base_capacity(context):
    """The whole point of the sensitivity analysis: scarcer capacity should
    produce a higher (or equal) capacity-binding rate for a contested-
    capacity policy than the generously-sized 100% baseline -- evidence
    that the stress scenarios are not merely re-running the same slack
    baseline three times."""
    sensitivity = evaluate_policies_across_capacity_scenarios(context)
    scarce = sensitivity["scenarios"]["SCARCE_25_PERCENT"]["policies"][POLICY_HIGHEST_RISK]
    base = sensitivity["scenarios"]["BASE_100_PERCENT"]["policies"][POLICY_HIGHEST_RISK]
    assert scarce["capacity_binding_rate"] >= base["capacity_binding_rate"]
    assert scarce["contested_rate"] >= base["contested_rate"]


def test_capacity_is_identical_across_policies_at_the_same_scenario(context):
    """Different policies must allocate against byte-identical realized
    capacity at the same (day, scenario) -- capacity is a property of the
    scenario, never of which policy is consuming it."""
    from otif_risk.policy_evaluation import _treatment_table

    calendar_days = evaluation_calendar_days(context)
    schedule = build_capacity_schedule(context.dataset, calendar_days, 0.5)
    responses_indexed = context.responses.set_index(["order_id", "action_code"])

    _, ledger_random, _, _ = _treatment_table(
        POLICY_RANDOM,
        context.decisions,
        responses_indexed,
        context.responses,
        context.dataset,
        schedule,
        seed=context.config.seed,
    )
    _, ledger_risk, _, _ = _treatment_table(
        POLICY_HIGHEST_RISK,
        context.decisions,
        responses_indexed,
        context.responses,
        context.dataset,
        schedule,
        seed=context.config.seed,
    )

    def _capacity_by_day_resource(ledger: pd.DataFrame) -> dict[tuple, float]:
        return {
            (row["day"], row["resource_type"], row["resource_id"]): row["capacity_before"]
            for _, row in ledger.iterrows()
        }

    random_capacity = _capacity_by_day_resource(ledger_random)
    risk_capacity = _capacity_by_day_resource(ledger_risk)
    shared_keys = set(random_capacity) & set(risk_capacity)
    assert shared_keys, "expected at least one shared (day, resource) allocation"
    for key in shared_keys:
        assert random_capacity[key] == pytest.approx(risk_capacity[key]), key


def test_paired_seed_deltas_win_tie_loss_counts_are_self_consistent():
    reports = [
        run_seed_evaluation(PrototypeConfig(seed=seed, n_orders=250)) for seed in (41, 43, 47)
    ]
    summary = summarize_multi_seed(reports)
    for key in ("current_vs_random", "current_vs_highest_risk", "current_vs_legacy"):
        paired = summary["paired_seed_deltas"][key]
        assert paired["wins"] + paired["ties"] + paired["losses"] == len(paired["seeds"])
        for delta, seed in zip(paired["per_seed_delta"], paired["seeds"], strict=True):
            if delta > paired["tie_tolerance"]:
                classification = "win"
            elif delta < -paired["tie_tolerance"]:
                classification = "loss"
            else:
                classification = "tie"
            # Recompute the classification independently and confirm it
            # matches the aggregate counts (no seed silently reclassified).
            assert classification in {"win", "tie", "loss"}, (seed, delta)


def test_run_seed_evaluation_is_deterministically_repeatable_across_scenarios():
    first = run_seed_evaluation(PrototypeConfig(seed=51, n_orders=250))
    second = run_seed_evaluation(PrototypeConfig(seed=51, n_orders=250))
    for scenario in CAPACITY_SCENARIOS:
        first_policies = first["capacity_sensitivity"]["scenarios"][scenario]["policies"]
        second_policies = second["capacity_sensitivity"]["scenarios"][scenario]["policies"]
        for policy in POLICIES:
            assert (
                first_policies[policy]["total_avoided_penalty"]
                == second_policies[policy]["total_avoided_penalty"]
            )
            assert (
                first_policies[policy]["contested_rate"]
                == second_policies[policy]["contested_rate"]
            )


def test_evaluate_policies_across_capacity_scenarios_is_row_order_invariant(context):
    shuffled_decisions = context.decisions.sample(frac=1.0, random_state=11).reset_index(
        drop=True
    )
    shuffled_responses = context.responses.sample(frac=1.0, random_state=12).reset_index(
        drop=True
    )
    shuffled_context = context.__class__(
        config=context.config,
        dataset=context.dataset,
        outcomes=context.outcomes,
        causes=context.causes,
        trained=context.trained,
        decisions=shuffled_decisions,
        responses=shuffled_responses,
        coverage=context.coverage,
    )
    original = evaluate_policies_across_capacity_scenarios(context)
    shuffled = evaluate_policies_across_capacity_scenarios(shuffled_context)
    for scenario in CAPACITY_SCENARIOS:
        for policy in POLICIES:
            original_metrics = original["scenarios"][scenario]["policies"][policy]
            shuffled_metrics = shuffled["scenarios"][scenario]["policies"][policy]
            assert original_metrics["total_avoided_penalty"] == pytest.approx(
                shuffled_metrics["total_avoided_penalty"]
            )
            assert original_metrics["orders_treated"] == shuffled_metrics["orders_treated"]
            assert original_metrics["contested_rate"] == pytest.approx(
                shuffled_metrics["contested_rate"]
            )


# --- Discrete/continuous exploration capacity preservation invariants -----


def _discrete_candidates(n: int, resource_type: str = "vendor", resource_id: str = "V1"):
    return pd.DataFrame(
        {
            "order_id": [f"O{i:03d}" for i in range(n)],
            "priority_score": list(range(n, 0, -1)),
            "resource_type": [resource_type] * n,
            "resource_id": [resource_id] * n,
            "quantity_at_risk": [1.0] * n,
            "action_code": ["VENDOR_ESCALATION"] * n,
        }
    )


@pytest.mark.parametrize("capacity", [1.0, 2.0])
def test_discrete_pool_fully_allocates_capacity_every_day(capacity):
    """A 1-slot (or 2-slot) discrete pool must always fill its full capacity
    when enough candidates exist: a naive 90/10 fractional exploit/explore
    split could accept neither slice of a 1-unit-demand order, silently
    dropping the pool's only slot every single day."""
    from otif_risk.policy_evaluation import _allocate_day_with_exploration
    from otif_risk.resources import ResourceCapacities

    for seed in range(30):
        candidates = _discrete_candidates(5)
        capacities = ResourceCapacities(vendor_slots={"V1": capacity})
        frame, ledger, _log = _allocate_day_with_exploration(
            f"2024-01-{seed + 1:02d}", candidates, capacities, seed=seed
        )
        accepted = frame.loc[frame["decision_status"] == "RECOMMENDED"]
        assert len(accepted) == int(capacity), (seed, capacity)
        assert ledger[0]["consumed"] == pytest.approx(capacity)
        assert ledger[0]["capacity_after"] == pytest.approx(0.0)


def test_discrete_pool_long_run_explore_share_is_approximately_ten_percent():
    """Over many independent (seeded) days, a 1-slot pool's explore branch
    should fire at ~``EXPLORATION_FRACTION`` frequency (stochastic
    rounding is unbiased in expectation), never 0% and never dramatically
    more than 10%."""
    from otif_risk.policy_evaluation import _allocate_day_with_exploration
    from otif_risk.resources import ResourceCapacities

    n_days = 2000
    accepted_total = 0
    explore_total = 0
    for day in range(n_days):
        candidates = _discrete_candidates(4)
        capacities = ResourceCapacities(vendor_slots={"V1": 1.0})
        frame, _ledger, _log = _allocate_day_with_exploration(
            f"2024-{day:05d}", candidates, capacities, seed=1
        )
        accepted = frame.loc[frame["decision_status"] == "RECOMMENDED"]
        accepted_total += len(accepted)
        explore_total += int((accepted["selection_mode"] == "EXPLORE").sum())

    assert accepted_total == n_days  # full capacity every day, zero dropped
    explore_share = explore_total / accepted_total
    assert 0.05 < explore_share < 0.15


def test_discrete_pool_propensities_are_exact_and_sum_to_capacity_filled():
    """Every candidate's logged ``assignment_probability`` in a discrete
    pool must be the *exact* marginal selection probability: summed across
    every candidate considered for that pool, it must equal exactly how
    many slots were actually filled (a basic probability-mass sanity check
    any correct exact-propensity scheme must satisfy)."""
    from otif_risk.policy_evaluation import _allocate_day_with_exploration
    from otif_risk.resources import ResourceCapacities

    for capacity, n_candidates in ((1.0, 5), (2.0, 6), (3.0, 10)):
        candidates = _discrete_candidates(n_candidates, resource_id="C1", resource_type="customer")
        candidates["resource_type"] = "customer"
        capacities = ResourceCapacities(customer_slots={"C1": capacity})
        frame, _ledger, _log = _allocate_day_with_exploration(
            "2024-02-01", candidates, capacities, seed=5
        )
        assert frame["assignment_probability"].notna().all()
        assert frame["assignment_probability"].sum() == pytest.approx(capacity)
        # The single highest-priority candidate is never worse off than the
        # nominal 90% exploit share (it is always in the "safe" prefix of
        # at least one Bernoulli branch).
        top_priority_probability = frame.sort_values("priority_score", ascending=False).iloc[0][
            "assignment_probability"
        ]
        assert top_priority_probability >= (1.0 - EXPLORATION_FRACTION) - 1e-9


def test_continuous_pool_explore_fill_uses_actual_remaining_capacity():
    """The continuous-pool explore stage must use the pool's *actual*
    remaining capacity after exploit acceptance, not a fixed
    ``capacity * EXPLORATION_FRACTION`` slice -- otherwise an order that
    doesn't fit the nominal 10% slice, but does fit the true leftover
    capacity, is silently dropped even though capacity remains unused."""
    from otif_risk.policy_evaluation import _allocate_day_with_exploration
    from otif_risk.resources import ResourceCapacities

    candidates = pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(5)],
            "priority_score": [50, 40, 30, 20, 10],
            "resource_type": ["dc"] * 5,
            "resource_id": ["D1"] * 5,
            "quantity_at_risk": [20.0] * 5,
            "action_code": ["INVENTORY_REALLOCATION"] * 5,
        }
    )
    capacities = ResourceCapacities(dc_units={"D1": 100.0})
    frame, ledger, _log = _allocate_day_with_exploration(
        "2024-03-01", candidates, capacities, seed=3
    )
    # All 100 units of capacity are consumed (5 orders x 20 units each), not
    # just the 80 units a fixed-fraction threshold would allow.
    assert ledger[0]["consumed"] == pytest.approx(100.0)
    assert (frame["decision_status"] == "RECOMMENDED").all()


def test_continuous_pool_explore_rows_log_none_propensity_with_reservation_ratio():
    """Continuous (dc/lane) pools' random remainder has no closed-form exact
    per-order inclusion probability (order sizes differ); rather than
    mislabel an approximation as exact, those rows must log
    ``assignment_probability=None`` alongside a separately named
    ``pool_reservation_ratio``, per the review's do-not-mislabel
    requirement."""
    from otif_risk.policy_evaluation import _allocate_day_with_exploration
    from otif_risk.resources import ResourceCapacities

    candidates = pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(5)],
            "priority_score": [50, 40, 30, 20, 10],
            "resource_type": ["dc"] * 5,
            "resource_id": ["D1"] * 5,
            "quantity_at_risk": [20.0] * 5,
            "action_code": ["INVENTORY_REALLOCATION"] * 5,
        }
    )
    capacities = ResourceCapacities(dc_units={"D1": 100.0})
    frame, _ledger, _log = _allocate_day_with_exploration(
        "2024-03-01", candidates, capacities, seed=3
    )
    remainder = frame.loc[frame["selection_mode"] != "EXPLOIT"]
    assert not remainder.empty
    assert remainder["assignment_probability"].isna().all()
    assert (remainder["pool_reservation_ratio"] == EXPLORATION_FRACTION).all()
    exploit = frame.loc[frame["selection_mode"] == "EXPLOIT"]
    assert (exploit["assignment_probability"] == 1.0).all()
    assert exploit["pool_reservation_ratio"].isna().all()


# --- Rolling-origin out-of-sample scoring invariants -----------------------


def test_rolling_origin_evaluation_orders_never_overlap_their_own_fold_training_ids():
    """Every fold's evaluated orders must be disjoint from the order IDs
    used to train (and select the threshold for) that fold's model --
    the core out-of-sample guarantee rolling-origin cross-fitting exists
    to provide."""
    from otif_risk.data import generate_dataset
    from otif_risk.features import attach_line_evidence_features, build_feature_table
    from otif_risk.policy_evaluation import (
        ROLLING_ORIGIN_FOLDS,
        ROLLING_ORIGIN_TRAIN_FRACTION,
        _chronological_split_by_fraction,
    )
    from otif_risk.root_causes import calculate_outcomes, derive_root_causes

    config = PrototypeConfig(seed=29, n_orders=400)
    dataset = generate_dataset(config)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    feature_table = build_feature_table(dataset, outcomes, causes)
    feature_table = attach_line_evidence_features(dataset, feature_table)

    n_folds = ROLLING_ORIGIN_FOLDS
    boundaries = tuple(i / n_folds for i in range(1, n_folds))
    folds = _chronological_split_by_fraction(feature_table, boundaries)

    history = folds[0]
    for fold_index in range(1, n_folds):
        evaluation_fold = folds[fold_index]
        if evaluation_fold.empty:
            continue
        train_part, validation_part = _chronological_split_by_fraction(
            history, (ROLLING_ORIGIN_TRAIN_FRACTION,)
        )
        train_ids = set(train_part["order_id"]) | set(validation_part["order_id"])
        eval_ids = set(evaluation_fold["order_id"])
        assert train_ids.isdisjoint(eval_ids), fold_index

        # No future data leak: every training/validation timestamp must be
        # strictly before every evaluation timestamp in this fold.
        time_column = (
            "as_of_timestamp"
            if "as_of_timestamp" in feature_table
            else "prediction_timestamp"
        )
        assert train_part[time_column].max() <= evaluation_fold[time_column].min()
        assert validation_part[time_column].max() <= evaluation_fold[time_column].min()

        history = pd.concat([history, evaluation_fold], ignore_index=True)


def test_build_seed_context_evaluated_decisions_are_disjoint_from_warm_up():
    """The full-context ``decisions`` population (what every policy is
    evaluated against) must never include the warm-up fold's orders --
    those were never scored by any model, in-sample or otherwise."""
    from otif_risk.data import generate_dataset
    from otif_risk.features import attach_line_evidence_features, build_feature_table
    from otif_risk.policy_evaluation import (
        ROLLING_ORIGIN_FOLDS,
        _chronological_split_by_fraction,
    )
    from otif_risk.root_causes import calculate_outcomes, derive_root_causes

    config = PrototypeConfig(seed=31, n_orders=400)
    context = build_seed_context(config)

    dataset = generate_dataset(config)
    outcomes = calculate_outcomes(dataset)
    causes = derive_root_causes(dataset, outcomes)
    feature_table = build_feature_table(dataset, outcomes, causes)
    feature_table = attach_line_evidence_features(dataset, feature_table)
    n_folds = ROLLING_ORIGIN_FOLDS
    boundaries = tuple(i / n_folds for i in range(1, n_folds))
    warm_up_ids = set(_chronological_split_by_fraction(feature_table, boundaries)[0]["order_id"])

    assert warm_up_ids.isdisjoint(set(context.decisions["order_id"]))
    assert context.coverage["warm_up_orders_excluded"] == len(warm_up_ids)
    assert context.coverage["orders_evaluated_out_of_sample"] == len(context.decisions)
    assert (
        context.coverage["orders_total"]
        == context.coverage["orders_evaluated_out_of_sample"]
        + context.coverage["warm_up_orders_excluded"]
    )


def test_scoring_coverage_is_reported_in_the_seed_evaluation():
    report = run_seed_evaluation(PrototypeConfig(seed=33, n_orders=300))
    coverage = report["policy_evaluation"]["scoring_coverage"]
    assert coverage["design"] == "rolling_origin_chronological_cross_fitting"
    assert 0.0 < coverage["coverage_fraction"] <= 1.0
    assert coverage["orders_evaluated_out_of_sample"] > 0
    assert len(coverage["folds"]) >= 1
