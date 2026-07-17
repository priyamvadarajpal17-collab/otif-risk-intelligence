from __future__ import annotations

import copy
import itertools
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from otif_risk.bayesian import (
    ALL_ROUTES,
    CAUSE_NODES,
    CHAIN_NODES,
    CHAIN_PARENTS,
    IN_FULL_FAILURE,
    LATE_DELIVERY,
    MECHANISM_NODES,
    SIGNAL_COLUMNS,
    _confidence_band,
    _posterior,
    fit_bayesian_network,
)
from otif_risk.contracts import CAUSE_CATEGORIES
from otif_risk.model import ENDPOINT


def _correlated_history(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthetic training history with genuine upstream -> downstream dependence.

    Unlike independent random cause flags, every stage's incident flag here
    causally depends on its declared parents (mirroring the real pipeline's
    ``stage_*`` incident evidence), so the fitted CPTs actually encode
    propagation/collider structure -- required for the do-vs-conditioning
    divergence test below.
    """
    rng = np.random.default_rng(seed)
    order_capture = rng.random(n) < 0.10
    vendor_failure = rng.random(n) < 0.15
    dc_capacity = rng.random(n) < 0.20
    customer_delivery = rng.random(n) < 0.10
    inventory_shortage = (vendor_failure & (rng.random(n) < 0.70)) | (
        ~vendor_failure & (rng.random(n) < 0.05)
    )
    warehouse_ops = ((inventory_shortage | dc_capacity) & (rng.random(n) < 0.75)) | (
        ~(inventory_shortage | dc_capacity) & (rng.random(n) < 0.05)
    )
    transport = (warehouse_ops & (rng.random(n) < 0.50)) | (
        ~warehouse_ops & (rng.random(n) < 0.05)
    )
    in_full_failure = (inventory_shortage & (rng.random(n) < 0.60)) | (
        ~inventory_shortage & (rng.random(n) < 0.02)
    )
    late_delivery = (
        (order_capture | warehouse_ops | transport | customer_delivery) & (rng.random(n) < 0.85)
    ) | (~(order_capture | warehouse_ops | transport | customer_delivery) & (rng.random(n) < 0.02))
    otif_miss = in_full_failure | late_delivery
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(n)],
            "cause_ORDER_CAPTURE": order_capture.astype(int),
            "cause_VENDOR_FAILURE": vendor_failure.astype(int),
            "cause_INVENTORY_SHORTAGE": inventory_shortage.astype(int),
            "cause_DC_CAPACITY": dc_capacity.astype(int),
            "cause_WAREHOUSE_OPS": warehouse_ops.astype(int),
            "cause_TRANSPORT": transport.astype(int),
            "cause_CUSTOMER_DELIVERY": customer_delivery.astype(int),
            "on_time": (~late_delivery).astype(int),
            "in_full": (~in_full_failure).astype(int),
            "otif_miss": otif_miss.astype(int),
        }
    )


def _history(n_repeats: int = 6) -> pd.DataFrame:
    """Synthetic training history spanning every combination of 7 binary causes."""
    rows = []
    rng = np.random.default_rng(0)
    for repeat in range(n_repeats):
        for index, combination in enumerate(itertools.product((0, 1), repeat=7)):
            base_p_late = 0.05 + 0.4 * combination[1] + 0.3 * combination[5] + 0.05 * sum(
                combination
            )
            base_p_full = 0.05 + 0.5 * combination[2] + 0.05 * sum(combination)
            late = int(rng.random() < min(base_p_late, 0.95))
            in_full_failure = int(rng.random() < min(base_p_full, 0.95))
            miss = int(late or in_full_failure)
            rows.append(
                {
                    "order_id": f"{repeat}-{index}",
                    **dict(zip(CAUSE_NODES, combination, strict=True)),
                    "on_time": 1 - late,
                    "in_full": 1 - in_full_failure,
                    "otif_miss": miss,
                }
            )
    frame = pd.DataFrame(rows)
    return frame.rename(columns={cause: f"cause_{cause}" for cause in CAUSE_NODES})


def _evidence(*combinations: tuple[int, ...], observed: bool = True) -> pd.DataFrame:
    rows = [
        {
            "order_id": f"order-{index}",
            **dict(zip(SIGNAL_COLUMNS, combination, strict=True)),
            "vendor_ready_observed": int(observed),
            "shipped_observed": int(observed),
            "transit_observed": int(observed),
        }
        for index, combination in enumerate(combinations)
    ]
    return pd.DataFrame(rows)


# --- Topology ----------------------------------------------------------------


def test_chain_topology_matches_the_10_node_mechanism_design():
    assert CHAIN_PARENTS["INVENTORY_SHORTAGE"] == ("VENDOR_FAILURE",)
    assert set(CHAIN_PARENTS["WAREHOUSE_OPS"]) == {"INVENTORY_SHORTAGE", "DC_CAPACITY"}
    assert CHAIN_PARENTS["TRANSPORT"] == ("WAREHOUSE_OPS",)
    assert CHAIN_PARENTS[IN_FULL_FAILURE] == ("INVENTORY_SHORTAGE",)
    assert set(CHAIN_PARENTS[LATE_DELIVERY]) == {
        "ORDER_CAPTURE",
        "WAREHOUSE_OPS",
        "TRANSPORT",
        "CUSTOMER_DELIVERY",
    }
    assert set(CHAIN_PARENTS["OTIF_MISS"]) == {IN_FULL_FAILURE, LATE_DELIVERY}
    assert CHAIN_PARENTS["ORDER_CAPTURE"] == ()
    assert MECHANISM_NODES == (IN_FULL_FAILURE, LATE_DELIVERY)
    assert len(CHAIN_NODES) == 10
    assert set(CHAIN_NODES) == {*CAUSE_NODES, IN_FULL_FAILURE, LATE_DELIVERY, "OTIF_MISS"}

    # Parents must precede children in the topological order.
    position = {node: index for index, node in enumerate(CHAIN_NODES)}
    for node, parents in CHAIN_PARENTS.items():
        for parent in parents:
            assert position[parent] < position[node]


def test_all_edges_are_accounted_for_in_the_topology():
    edges = [(parent, node) for node, parents in CHAIN_PARENTS.items() for parent in parents]
    assert len(edges) == 11
    assert ("VENDOR_FAILURE", "INVENTORY_SHORTAGE") in edges
    assert ("INVENTORY_SHORTAGE", "WAREHOUSE_OPS") in edges
    assert ("DC_CAPACITY", "WAREHOUSE_OPS") in edges
    assert ("WAREHOUSE_OPS", "TRANSPORT") in edges
    assert ("INVENTORY_SHORTAGE", IN_FULL_FAILURE) in edges
    assert (IN_FULL_FAILURE, "OTIF_MISS") in edges
    assert (LATE_DELIVERY, "OTIF_MISS") in edges


def test_mechanism_routes_include_multiple_paths_for_shared_upstream_causes():
    """VENDOR_FAILURE/INVENTORY_SHORTAGE feed both mechanisms -- multiple routes."""
    vendor_routes = ALL_ROUTES["VENDOR_FAILURE"]
    assert len(vendor_routes) >= 2
    assert any(route[-2] == IN_FULL_FAILURE for route in vendor_routes)
    assert any(route[-2] == LATE_DELIVERY for route in vendor_routes)
    for route in vendor_routes:
        assert route[0] == "VENDOR_FAILURE"
        assert route[-1] == "OTIF_MISS"

    # DC_CAPACITY only ever feeds the timing mechanism.
    dc_routes = ALL_ROUTES["DC_CAPACITY"]
    assert all(route[-2] == LATE_DELIVERY for route in dc_routes)


# --- Fitting / mechanism truth construction -----------------------------------


def test_fit_requires_on_time_and_in_full_columns():
    history = _history().drop(columns=["on_time"])
    with pytest.raises(ValueError, match="missing columns"):
        fit_bayesian_network(history)


def test_fit_requires_all_cause_columns_and_target():
    incomplete_history = _history().drop(columns=["cause_VENDOR_FAILURE"])
    with pytest.raises(ValueError, match="missing columns"):
        fit_bayesian_network(incomplete_history)


def test_mechanism_truth_is_built_from_on_time_and_in_full_not_failure_only_causes():
    """IN_FULL_FAILURE / LATE_DELIVERY are 1 - in_full / 1 - on_time from outcomes."""
    history = _correlated_history()
    bundle = fit_bayesian_network(history)

    # With INVENTORY_SHORTAGE clear, IN_FULL_FAILURE should be rare; with it
    # active, it should be substantially more likely -- this can only be true
    # if IN_FULL_FAILURE was actually derived from `in_full`, not copied from
    # a cause_* label (which never encoded this at all in earlier iterations).
    assert bundle.cpts[IN_FULL_FAILURE][(0,)] < 0.10
    assert bundle.cpts[IN_FULL_FAILURE][(1,)] > 0.40

    # LATE_DELIVERY with every timing parent clear should be rare; with any
    # active it should be much more likely.
    assert bundle.cpts[LATE_DELIVERY][(0, 0, 0, 0)] < 0.10
    assert bundle.cpts[LATE_DELIVERY][(0, 1, 0, 0)] > 0.40


def test_stage_history_is_preferred_over_failure_only_cause_labels() -> None:
    history = _history()
    for cause in CAUSE_NODES:
        history[f"stage_{cause}"] = history[f"cause_{cause}"]
        history[f"cause_{cause}"] = 0

    bundle = fit_bayesian_network(history)

    assert any(abs(lift) > 0 for lift in bundle.cause_lifts.values())


def test_cause_lifts_cover_every_cause_category():
    bundle = fit_bayesian_network(_history())
    assert set(bundle.cause_lifts) == set(CAUSE_CATEGORIES)


# --- Scoring / pathway ---------------------------------------------------------


def test_bayesian_bundle_scores_and_returns_pathway_with_mechanism_routes():
    bundle = fit_bayesian_network(_correlated_history())
    evidence = _evidence((0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0))

    scored = bundle.score(evidence)
    pathway_no_evidence = json.loads(scored.loc[0, "causal_pathway"])
    pathway_vendor = json.loads(scored.loc[1, "causal_pathway"])

    expected_columns = {
        "order_id",
        "bbn_risk_score",
        "in_full_failure_probability",
        "late_delivery_probability",
        "evidence_coverage",
        "causal_confidence",
        "causal_pathway",
        "causal_attribution_json",
        "intervention_scenarios_json",
    }
    assert expected_columns <= set(scored.columns)
    assert 0 <= scored["bbn_risk_score"].min() <= scored["bbn_risk_score"].max() <= 1
    assert pathway_no_evidence["active_evidence"] == []
    assert pathway_no_evidence["route"] == []
    assert pathway_no_evidence["routes"] == []

    assert pathway_vendor["active_evidence"] == ["VENDOR_FAILURE"]
    assert pathway_vendor["endpoint"] == "OTIF_MISS"
    assert pathway_vendor["active_evidence_count"] == 1
    routes = pathway_vendor["routes"]
    assert len(routes) >= 2
    assert any(route[-2] == IN_FULL_FAILURE for route in routes)
    assert any(route[-2] == LATE_DELIVERY for route in routes)
    assert pathway_vendor["route"][0] == "VENDOR_FAILURE"
    assert pathway_vendor["route"][-1] == "OTIF_MISS"
    assert IN_FULL_FAILURE in pathway_vendor["mechanism_posteriors"]
    assert LATE_DELIVERY in pathway_vendor["mechanism_posteriors"]
    assert pathway_vendor["evidence_coverage"] == pytest.approx(1.0)
    assert pathway_vendor["confidence"] == "HIGH"
    assert "interpretation" in pathway_vendor
    assert "evidence_delta" in pathway_vendor


def test_unobserved_intermediate_stages_are_marginalized_not_assumed_zero():
    """A node whose stage hasn't posted yet must be excluded from hard evidence."""
    bundle = fit_bayesian_network(_correlated_history())
    observed = _evidence((0, 1, 0, 0, 0, 0, 0), observed=True)
    unobserved = _evidence((0, 1, 0, 0, 0, 0, 0), observed=False)

    observed_score = bundle.score(observed)
    unobserved_score = bundle.score(unobserved)
    observed_pathway = json.loads(observed_score.loc[0, "causal_pathway"])
    unobserved_pathway = json.loads(unobserved_score.loc[0, "causal_pathway"])

    # VENDOR_FAILURE itself has no observability gate, so it's evidenced either way,
    # but WAREHOUSE_OPS/TRANSPORT should drop out of the evidence dict when unobserved.
    assert "WAREHOUSE_OPS" in observed_pathway["evidence"]
    assert "TRANSPORT" in observed_pathway["evidence"]
    assert "WAREHOUSE_OPS" not in unobserved_pathway["evidence"]
    assert "TRANSPORT" not in unobserved_pathway["evidence"]
    assert unobserved_pathway["confidence"] == "LOW"
    assert observed_pathway["confidence"] == "HIGH"
    assert np.isfinite(observed_score.loc[0, "bbn_risk_score"])
    assert np.isfinite(unobserved_score.loc[0, "bbn_risk_score"])


def test_evidence_coverage_and_confidence_band_thresholds():
    assert _confidence_band(4 / 7) == "LOW"
    assert _confidence_band(5 / 7) == "MEDIUM"
    assert _confidence_band(6 / 7) == "MEDIUM"
    assert _confidence_band(1.0) == "HIGH"


def test_bayesian_bundle_rejects_missing_evidence():
    bundle = fit_bayesian_network(_correlated_history())
    incomplete = _evidence((0, 0, 0, 0, 0, 0, 0)).drop(columns=SIGNAL_COLUMNS[-1])

    with pytest.raises(ValueError, match="missing Bayesian evidence"):
        bundle.score(incomplete)


def test_continuous_and_boolean_signals_are_binarized():
    bundle = fit_bayesian_network(_correlated_history())
    values = np.array([0.2, 0.8, True, False, 1.0, 0.49, 0.51], dtype=object)

    scored = bundle.score(_evidence(tuple(values)))

    assert np.isfinite(scored.loc[0, "bbn_risk_score"])


# --- Inference engine ----------------------------------------------------------


def test_fitted_bundle_reports_pgmpy_exact_inference_mode():
    bundle = fit_bayesian_network(_correlated_history())

    assert bundle.inference_mode == "pgmpy_exact"
    assert bundle.engine_build_error is None
    assert bundle.inference_engine is not None


def test_brute_force_fallback_matches_pgmpy_exact_inference_for_every_query_node():
    """The brute-force enumeration fallback must be numerically exact, not approximate."""
    bundle = fit_bayesian_network(_correlated_history())
    evidence = _evidence((0, 1, 1, 0, 0, 0, 0))
    pgmpy_scored = bundle.score(evidence)

    bundle.inference_engine = None
    fallback_scored = bundle.score(evidence)

    for column in (
        "bbn_risk_score",
        "in_full_failure_probability",
        "late_delivery_probability",
    ):
        assert fallback_scored.loc[0, column] == pytest.approx(
            pgmpy_scored.loc[0, column], abs=1e-6
        )


def test_engine_construction_failure_is_recorded_explicitly(monkeypatch):
    """The only legitimate fallback trigger is explicit engine-build failure."""
    import otif_risk.bayesian as bayesian_module

    def _fail_to_build(cpts):
        return None, "pgmpy is unavailable in this environment: simulated ImportError"

    monkeypatch.setattr(bayesian_module, "_build_pgmpy_engine", _fail_to_build)
    bundle = fit_bayesian_network(_correlated_history())

    assert bundle.inference_mode == "brute_force_exact"
    assert bundle.engine_build_error is not None
    assert bundle.inference_engine is None
    scored = bundle.score(_evidence((0, 1, 0, 0, 0, 0, 0)))
    assert np.isfinite(scored.loc[0, "bbn_risk_score"])


def test_query_time_error_from_an_available_engine_surfaces():
    """An available engine's runtime error must not be silently swallowed."""
    bundle = fit_bayesian_network(_correlated_history())
    broken_engine = SimpleNamespace(query=MagicMock(side_effect=RuntimeError("boom")))
    bundle.inference_engine = broken_engine

    with pytest.raises(RuntimeError, match="boom"):
        bundle.score(_evidence((0, 0, 0, 0, 0, 0, 0)))


# --- Structural interventions ---------------------------------------------------


def test_intervention_severs_parent_influence_and_can_differ_from_conditioning():
    """do(WAREHOUSE_OPS=0) must not equal conditioning WAREHOUSE_OPS=0 as evidence.

    WAREHOUSE_OPS is a collider of INVENTORY_SHORTAGE and DC_CAPACITY.
    Conditioning on it as hard evidence "explains away" INVENTORY_SHORTAGE
    given DC_CAPACITY=1 is already observed active, which changes the belief
    feeding IN_FULL_FAILURE (and therefore OTIF_MISS); a structural
    intervention severs that backward influence and must not.
    """
    bundle = fit_bayesian_network(_correlated_history(n=20000, seed=3))

    evidence = {"DC_CAPACITY": 1}
    conditioning_evidence = {**evidence, "WAREHOUSE_OPS": 0}
    p_condition = _posterior(bundle.cpts, conditioning_evidence, ENDPOINT)
    p_do = _posterior(bundle.cpts, evidence, ENDPOINT, do={"WAREHOUSE_OPS": 0})

    assert p_condition != pytest.approx(p_do, abs=1e-9)

    result = bundle.intervene(evidence, {"WAREHOUSE_OPS": 0})
    assert result["post_intervention_bayesian_posterior"] == pytest.approx(p_do, abs=1e-6)


def test_intervention_matches_conditioning_for_a_root_node():
    """A parentless root node: do(X=x) and conditioning on X=x always agree.

    A root node's own CPT factor is a constant P(X=x) that does not depend on
    any other free variable, so whether it is included (conditioning) or
    replaced by a point mass (do) only rescales the joint by a constant that
    cancels out in the normalization -- there is no parent influence to sever
    in the first place. CUSTOMER_DELIVERY has no parents in this network.
    """
    bundle = fit_bayesian_network(_correlated_history(n=20000, seed=4))
    evidence: dict[str, int] = {}
    p_condition = _posterior(bundle.cpts, {"CUSTOMER_DELIVERY": 0}, ENDPOINT)
    p_do = _posterior(bundle.cpts, evidence, ENDPOINT, do={"CUSTOMER_DELIVERY": 0})
    assert p_condition == pytest.approx(p_do, abs=1e-9)


def test_intervene_baseline_matches_plain_query_with_no_do():
    bundle = fit_bayesian_network(_correlated_history())
    evidence = {"VENDOR_FAILURE": 1}
    result = bundle.intervene(evidence, {"VENDOR_FAILURE": 0})
    assert result["baseline_bayesian_posterior"] == pytest.approx(
        bundle._query(evidence, ENDPOINT), abs=1e-6
    )
    assert result["absolute_risk_reduction"] == pytest.approx(
        result["baseline_bayesian_posterior"] - result["post_intervention_bayesian_posterior"],
        abs=1e-5,
    )
    assert result["type"] == "single_node_mitigation"
    assert result["qualification"].startswith("Fixed-structure scenario analysis")
    assert result["inference_mode_used"] == "brute_force_exact"


def test_intervention_rejects_non_cause_nodes():
    bundle = fit_bayesian_network(_correlated_history())
    with pytest.raises(ValueError, match="operational cause"):
        bundle.intervene({"VENDOR_FAILURE": 1}, {"OTIF_MISS": 0})
    with pytest.raises(ValueError, match="operational cause"):
        bundle.intervene({"VENDOR_FAILURE": 1}, {IN_FULL_FAILURE: 0})


def test_intervention_rejects_invalid_values():
    bundle = fit_bayesian_network(_correlated_history())
    with pytest.raises(ValueError, match="0 or 1"):
        bundle.intervene({"VENDOR_FAILURE": 1}, {"VENDOR_FAILURE": 2})


def test_intervention_rejects_empty_do():
    bundle = fit_bayesian_network(_correlated_history())
    with pytest.raises(ValueError, match="at least one"):
        bundle.intervene({"VENDOR_FAILURE": 1}, {})


def test_evidence_dict_with_invalid_node_or_value_is_rejected():
    bundle = fit_bayesian_network(_correlated_history())
    with pytest.raises(ValueError, match="operational cause"):
        bundle.intervene({"OTIF_MISS": 1}, {"VENDOR_FAILURE": 0})
    with pytest.raises(ValueError, match="0 or 1"):
        bundle.intervene({"VENDOR_FAILURE": 5}, {"VENDOR_FAILURE": 0})


def test_intervention_does_not_mutate_evidence_or_do_or_cpts():
    bundle = fit_bayesian_network(_correlated_history())
    evidence = {"VENDOR_FAILURE": 1, "DC_CAPACITY": 0}
    do = {"VENDOR_FAILURE": 0}
    evidence_before = copy.deepcopy(evidence)
    do_before = copy.deepcopy(do)
    cpts_before = copy.deepcopy(bundle.cpts)

    bundle.intervene(evidence, do)

    assert evidence == evidence_before
    assert do == do_before
    assert bundle.cpts == cpts_before


def test_mitigating_an_active_cause_reduces_the_posterior():
    bundle = fit_bayesian_network(_correlated_history())
    evidence = {"VENDOR_FAILURE": 1}
    result = bundle.intervene(evidence, {"VENDOR_FAILURE": 0})
    assert result["post_intervention_bayesian_posterior"] <= result["baseline_bayesian_posterior"]
    assert result["absolute_risk_reduction"] >= 0
    assert 0 <= result["relative_risk_reduction"] <= 1


def test_intervention_scenarios_json_includes_single_and_combined_scenarios():
    bundle = fit_bayesian_network(_correlated_history())
    evidence = _evidence((0, 1, 1, 0, 0, 0, 0))
    scored = bundle.score(evidence)
    scenarios = json.loads(scored.loc[0, "intervention_scenarios_json"])

    types = [scenario["type"] for scenario in scenarios]
    assert types.count("single_node_mitigation") == 2
    assert types.count("combined_mitigation") == 1
    combined = next(s for s in scenarios if s["type"] == "combined_mitigation")
    assert set(combined["intervened_nodes"]) == {"VENDOR_FAILURE", "INVENTORY_SHORTAGE"}
    for scenario in scenarios:
        assert scenario["inference_mode_used"] == "brute_force_exact"
        assert "not a proven treatment effect" in scenario["qualification"]


def test_intervention_scenarios_are_empty_with_no_active_evidence():
    bundle = fit_bayesian_network(_correlated_history())
    evidence = _evidence((0, 0, 0, 0, 0, 0, 0))
    scored = bundle.score(evidence)
    assert json.loads(scored.loc[0, "intervention_scenarios_json"]) == []
    assert json.loads(scored.loc[0, "causal_attribution_json"]) == []


# --- Evidence attribution -------------------------------------------------------


def test_evidence_attribution_reports_leave_one_out_contribution_for_active_nodes():
    bundle = fit_bayesian_network(_correlated_history())
    evidence = _evidence((0, 1, 1, 0, 0, 0, 0))
    scored = bundle.score(evidence)
    attribution = json.loads(scored.loc[0, "causal_attribution_json"])

    assert {row["node"] for row in attribution} == {"VENDOR_FAILURE", "INVENTORY_SHORTAGE"}
    for row in attribution:
        assert row["method"] == "evidence_attribution_leave_one_out"
        assert row["direction"] in {"increases_risk", "decreases_risk", "no_effect"}
        assert row["observed_value"] == 1
    # Sorted by descending absolute contribution.
    contributions = [abs(row["contribution"]) for row in attribution]
    assert contributions == sorted(contributions, reverse=True)


def test_evidence_attribution_screened_off_by_observed_mediator_is_zero():
    """Once a mediator is hard-evidenced, upstream evidence adds no information."""
    bundle = fit_bayesian_network(_correlated_history())
    # INVENTORY_SHORTAGE observed clear screens off VENDOR_FAILURE's effect on
    # every one of INVENTORY_SHORTAGE's descendants (a textbook d-separation).
    evidence = _evidence((0, 1, 0, 0, 0, 0, 0))
    scored = bundle.score(evidence)
    attribution = {
        row["node"]: row for row in json.loads(scored.loc[0, "causal_attribution_json"])
    }
    assert attribution["VENDOR_FAILURE"]["contribution"] == pytest.approx(0.0, abs=1e-9)
    assert attribution["VENDOR_FAILURE"]["direction"] == "no_effect"


# --- Mechanism probabilities -----------------------------------------------------


def test_mechanism_probabilities_respond_to_their_own_upstream_evidence():
    bundle = fit_bayesian_network(_correlated_history())
    baseline = bundle.score(_evidence((0, 0, 0, 0, 0, 0, 0))).iloc[0]
    inventory_active = bundle.score(_evidence((0, 0, 1, 0, 0, 0, 0))).iloc[0]
    order_capture_active = bundle.score(_evidence((1, 0, 0, 0, 0, 0, 0))).iloc[0]

    # INVENTORY_SHORTAGE feeds IN_FULL_FAILURE directly.
    assert inventory_active["in_full_failure_probability"] > baseline["in_full_failure_probability"]
    # ORDER_CAPTURE feeds LATE_DELIVERY directly, not IN_FULL_FAILURE.
    assert order_capture_active["late_delivery_probability"] > baseline["late_delivery_probability"]


def test_fit_bayesian_model_alias():
    from otif_risk.bayesian import fit_bayesian_model

    bundle = fit_bayesian_model(_correlated_history())
    assert bundle.inference_mode in {"pgmpy_exact", "brute_force_exact"}
