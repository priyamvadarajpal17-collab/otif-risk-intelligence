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
    SIGNAL_COLUMNS,
    fit_bayesian_network,
)
from otif_risk.contracts import CAUSE_CATEGORIES


def _history() -> pd.DataFrame:
    rows = []
    for repeat in range(4):
        for index, combination in enumerate(itertools.product((0, 1), repeat=7)):
            miss = int(sum(combination) >= 3 or combination[1] or combination[5])
            rows.append(
                {
                    "order_id": f"{repeat}-{index}",
                    **dict(zip(CAUSE_NODES, combination, strict=True)),
                    "otif_miss": miss,
                }
            )
    return pd.DataFrame(rows)


def _evidence(*combinations: tuple[int, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_id": f"order-{index}",
                **dict(zip(SIGNAL_COLUMNS, combination, strict=True)),
            }
            for index, combination in enumerate(combinations)
        ]
    )


def test_bayesian_bundle_scores_seven_leading_signals_and_pathway():
    bundle = fit_bayesian_network(_history())
    evidence = _evidence((0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1))

    scored = bundle.score(evidence)
    pathway = json.loads(scored.loc[1, "causal_pathway"])

    assert list(scored.columns) == ["order_id", "bbn_risk_score", "causal_pathway"]
    assert scored.loc[1, "bbn_risk_score"] > scored.loc[0, "bbn_risk_score"]
    assert 0 <= scored["bbn_risk_score"].min() <= scored["bbn_risk_score"].max() <= 1
    assert pathway["endpoint"] == "OTIF_MISS"
    assert set(pathway["observed_leading_signals"]) == set(CAUSE_CATEGORIES)
    assert pathway["interpretation"] == "probabilistic_association"


def test_fitted_bundle_reports_pgmpy_exact_inference_mode():
    """Item 3: the report/metrics must be able to say which inference mode was used."""
    bundle = fit_bayesian_network(_history())

    assert bundle.inference_mode == "pgmpy_exact"
    assert bundle.engine_build_error is None
    assert bundle.inference_engine is not None


def test_empirical_fallback_matches_fitted_cpt():
    bundle = fit_bayesian_network(_history(), smoothing=0.5)
    bundle.inference_engine = None
    combination = (0, 1, 0, 0, 0, 1, 0)

    scored = bundle.score(_evidence(combination))

    assert scored.loc[0, "bbn_risk_score"] == pytest.approx(
        bundle.outcome_probabilities[combination]
    )


def test_engine_construction_failure_is_recorded_explicitly(monkeypatch):
    """Item 3: the only legitimate fallback trigger is explicit engine-build failure."""
    import otif_risk.bayesian as bayesian_module

    def _fail_to_build(priors, probabilities, combinations):
        return None, "pgmpy is unavailable in this environment: simulated ImportError"

    monkeypatch.setattr(bayesian_module, "_build_pgmpy_engine", _fail_to_build)
    bundle = fit_bayesian_network(_history())

    assert bundle.inference_mode == "empirical_table"
    assert bundle.engine_build_error is not None
    assert bundle.inference_engine is None
    # Scoring must still work via the recorded fallback.
    scored = bundle.score(_evidence((0, 1, 0, 0, 0, 1, 0)))
    assert np.isfinite(scored.loc[0, "bbn_risk_score"])


def test_query_time_error_from_an_available_engine_surfaces():
    """Item 3: an available engine's runtime error must not be silently swallowed."""
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

