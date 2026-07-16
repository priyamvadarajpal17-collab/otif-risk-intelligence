from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from otif_risk.bayesian import (
    CAUSE_NODES,
    CHAIN_NODES,
    CHAIN_PARENTS,
    SIGNAL_COLUMNS,
    fit_bayesian_network,
)
from otif_risk.contracts import CAUSE_CATEGORIES


def _history(n_repeats: int = 6) -> pd.DataFrame:
    """Synthetic training history spanning every combination of 7 binary causes."""
    rows = []
    rng = np.random.default_rng(0)
    for repeat in range(n_repeats):
        for index, combination in enumerate(itertools.product((0, 1), repeat=7)):
            # A miss is likelier when vendor failure/transport-adjacent causes fire.
            base_p = 0.05 + 0.5 * combination[1] + 0.3 * combination[5] + 0.1 * sum(combination)
            miss = int(rng.random() < min(base_p, 0.97))
            rows.append(
                {
                    "order_id": f"{repeat}-{index}",
                    **dict(zip(CAUSE_NODES, combination, strict=True)),
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


def test_chain_topology_matches_the_compact_causal_design():
    assert CHAIN_PARENTS["INVENTORY_SHORTAGE"] == ("VENDOR_FAILURE",)
    assert set(CHAIN_PARENTS["WAREHOUSE_OPS"]) == {"INVENTORY_SHORTAGE", "DC_CAPACITY"}
    assert CHAIN_PARENTS["TRANSPORT"] == ("WAREHOUSE_OPS",)
    assert set(CHAIN_PARENTS["OTIF_MISS"]) == {
        "ORDER_CAPTURE",
        "INVENTORY_SHORTAGE",
        "WAREHOUSE_OPS",
        "TRANSPORT",
        "CUSTOMER_DELIVERY",
    }
    assert CHAIN_PARENTS["ORDER_CAPTURE"] == ()
    assert "OTIF_MISS" in CHAIN_NODES
    # Parents must precede children in the topological order.
    position = {node: index for index, node in enumerate(CHAIN_NODES)}
    for node, parents in CHAIN_PARENTS.items():
        for parent in parents:
            assert position[parent] < position[node]


def test_bayesian_bundle_scores_and_returns_pathway_with_route():
    bundle = fit_bayesian_network(_history())
    evidence = _evidence((0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0))

    scored = bundle.score(evidence)
    pathway_no_evidence = json.loads(scored.loc[0, "causal_pathway"])
    pathway_vendor = json.loads(scored.loc[1, "causal_pathway"])

    assert list(scored.columns) == ["order_id", "bbn_risk_score", "causal_pathway"]
    assert 0 <= scored["bbn_risk_score"].min() <= scored["bbn_risk_score"].max() <= 1
    assert pathway_no_evidence["active_evidence"] == []
    assert pathway_no_evidence["route"] == []
    assert pathway_vendor["active_evidence"] == ["VENDOR_FAILURE"]
    assert pathway_vendor["route"] == [
        "VENDOR_FAILURE",
        "INVENTORY_SHORTAGE",
        "WAREHOUSE_OPS",
        "TRANSPORT",
        "OTIF_MISS",
    ]
    assert pathway_vendor["endpoint"] == "OTIF_MISS"
    assert "interpretation" in pathway_vendor
    assert "evidence_delta" in pathway_vendor


def test_unobserved_intermediate_stages_are_marginalized_not_assumed_zero():
    """A node whose stage hasn't posted yet must be excluded from hard evidence."""
    bundle = fit_bayesian_network(_history())
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
    assert np.isfinite(observed_score.loc[0, "bbn_risk_score"])
    assert np.isfinite(unobserved_score.loc[0, "bbn_risk_score"])


def test_fitted_bundle_reports_pgmpy_exact_inference_mode():
    bundle = fit_bayesian_network(_history())

    assert bundle.inference_mode == "pgmpy_exact"
    assert bundle.engine_build_error is None
    assert bundle.inference_engine is not None


def test_brute_force_fallback_matches_pgmpy_exact_inference():
    """The brute-force enumeration fallback must be numerically exact, not approximate."""
    bundle = fit_bayesian_network(_history())
    evidence = _evidence((0, 1, 1, 0, 0, 0, 0))
    pgmpy_scored = bundle.score(evidence)

    bundle.inference_engine = None
    fallback_scored = bundle.score(evidence)

    assert fallback_scored.loc[0, "bbn_risk_score"] == pytest.approx(
        pgmpy_scored.loc[0, "bbn_risk_score"], abs=1e-6
    )


def test_engine_construction_failure_is_recorded_explicitly(monkeypatch):
    """The only legitimate fallback trigger is explicit engine-build failure."""
    import otif_risk.bayesian as bayesian_module

    def _fail_to_build(cpts):
        return None, "pgmpy is unavailable in this environment: simulated ImportError"

    monkeypatch.setattr(bayesian_module, "_build_pgmpy_engine", _fail_to_build)
    bundle = fit_bayesian_network(_history())

    assert bundle.inference_mode == "brute_force_exact"
    assert bundle.engine_build_error is not None
    assert bundle.inference_engine is None
    scored = bundle.score(_evidence((0, 1, 0, 0, 0, 0, 0)))
    assert np.isfinite(scored.loc[0, "bbn_risk_score"])


def test_query_time_error_from_an_available_engine_surfaces():
    """An available engine's runtime error must not be silently swallowed."""
    bundle = fit_bayesian_network(_history())
    broken_engine = SimpleNamespace(query=MagicMock(side_effect=RuntimeError("boom")))
    bundle.inference_engine = broken_engine

    with pytest.raises(RuntimeError, match="boom"):
        bundle.score(_evidence((0, 0, 0, 0, 0, 0, 0)))


def test_bayesian_bundle_rejects_missing_evidence():
    bundle = fit_bayesian_network(_history())
    incomplete = _evidence((0, 0, 0, 0, 0, 0, 0)).drop(columns=SIGNAL_COLUMNS[-1])

    with pytest.raises(ValueError, match="missing Bayesian evidence"):
        bundle.score(incomplete)


def test_continuous_and_boolean_signals_are_binarized():
    bundle = fit_bayesian_network(_history())
    values = np.array([0.2, 0.8, True, False, 1.0, 0.49, 0.51], dtype=object)

    scored = bundle.score(_evidence(tuple(values)))

    assert np.isfinite(scored.loc[0, "bbn_risk_score"])


def test_fit_requires_all_cause_columns_and_target():
    incomplete_history = _history().drop(columns=["cause_VENDOR_FAILURE"])
    with pytest.raises(ValueError, match="missing columns"):
        fit_bayesian_network(incomplete_history)


def test_cause_lifts_cover_every_cause_category():
    bundle = fit_bayesian_network(_history())
    assert set(bundle.cause_lifts) == set(CAUSE_CATEGORIES)


def test_stage_history_is_preferred_over_failure_only_cause_labels() -> None:
    history = _history()
    for cause in CAUSE_NODES:
        history[f"stage_{cause}"] = history[f"cause_{cause}"]
        history[f"cause_{cause}"] = 0

    bundle = fit_bayesian_network(history)

    assert any(abs(lift) > 0 for lift in bundle.cause_lifts.values())
