"""Seven-cause Bayesian network for the OTIF-miss endpoint."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from otif_pdf.contracts import CAUSE_CATEGORIES
from otif_pdf.model import ENDPOINT, TARGET_COLUMN

CAUSE_NODES = tuple(f"cause_{category}" for category in CAUSE_CATEGORIES)
SIGNAL_COLUMNS = tuple(f"leading_signal_{category}" for category in CAUSE_CATEGORIES)


@dataclass
class BayesianBundle:
    """Serializable exact/fallback Bayesian scorer for OTIF-miss risk."""

    outcome_probabilities: dict[tuple[int, ...], float]
    cause_priors: dict[str, float]
    cause_lifts: dict[str, float]
    inference_engine: Any | None = field(default=None, repr=False)
    endpoint: str = ENDPOINT
    inference_mode: str = "empirical_table"
    engine_build_error: str | None = None

    def score(self, evidence_frame: pd.DataFrame) -> pd.DataFrame:
        if "order_id" not in evidence_frame:
            raise ValueError("evidence_frame must contain order_id")
        missing = sorted(set(SIGNAL_COLUMNS) - set(evidence_frame.columns))
        if missing:
            raise ValueError(f"missing Bayesian evidence columns: {missing}")

        risks: list[float] = []
        pathways: list[str] = []
        for _, row in evidence_frame.iterrows():
            evidence_values = tuple(_as_binary(row[column]) for column in SIGNAL_COLUMNS)
            evidence = dict(zip(CAUSE_NODES, evidence_values, strict=True))
            risks.append(self._query(evidence, evidence_values))
            active = [
                category
                for category, value in zip(CAUSE_CATEGORIES, evidence_values, strict=True)
                if value
            ]
            active.sort(
                key=lambda category: abs(self.cause_lifts[f"cause_{category}"]),
                reverse=True,
            )
            pathways.append(
                json.dumps(
                    {
                        "endpoint": self.endpoint,
                        "observed_leading_signals": active,
                        "interpretation": "probabilistic_association",
                        "inference_mode": self.inference_mode,
                    },
                    separators=(",", ":"),
                )
            )
        return pd.DataFrame(
            {
                "order_id": evidence_frame["order_id"].to_numpy(),
                "bbn_risk_score": np.clip(risks, 0.0, 1.0),
                "causal_pathway": pathways,
            }
        )

    def _query(self, evidence: dict[str, int], values: tuple[int, ...]) -> float:
        if self.inference_engine is not None:
            # An engine was successfully constructed, so this is the sole,
            # explicitly recorded (`inference_mode == "pgmpy_exact"`) path that may
            # use it. Any query-time error here is a real defect (bad evidence,
            # library incompatibility, etc.) and must surface rather than being
            # silently swallowed into a fallback the report cannot see.
            result = self.inference_engine.query(
                variables=[ENDPOINT],
                evidence=evidence,
                show_progress=False,
            )
            state_names = result.state_names.get(ENDPOINT, [0, 1])
            positive_index = state_names.index(1)
            return float(np.asarray(result.values)[positive_index])
        # No engine is available. This only happens when pgmpy itself could not be
        # imported/constructed at fit time (`engine_build_error` records why), which
        # is an explicit, reported condition — not a silent runtime catch-all.
        return self.outcome_probabilities[values]


def fit_bayesian_network(
    historical: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> BayesianBundle:
    """Fit binary CPTs for seven cause nodes pointing to ``OTIF_MISS``."""
    required = {*CAUSE_NODES, TARGET_COLUMN}
    missing = sorted(required - set(historical.columns))
    if missing:
        raise ValueError(f"historical frame is missing columns: {missing}")
    if historical.empty:
        raise ValueError("historical frame must not be empty")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")

    binary = historical.loc[:, [*CAUSE_NODES, TARGET_COLUMN]].apply(
        lambda column: column.map(_as_binary)
    )
    cause_priors = {
        cause: float((binary[cause].sum() + smoothing) / (len(binary) + 2 * smoothing))
        for cause in CAUSE_NODES
    }
    global_risk = float(binary[TARGET_COLUMN].mean())
    cause_lifts = {}
    for cause in CAUSE_NODES:
        active = binary.loc[binary[cause] == 1, TARGET_COLUMN]
        active_risk = (
            float((active.sum() + smoothing) / (len(active) + 2 * smoothing))
            if len(active)
            else global_risk
        )
        cause_lifts[cause] = active_risk - global_risk

    probabilities: dict[tuple[int, ...], float] = {}
    combinations = list(itertools.product((0, 1), repeat=len(CAUSE_NODES)))
    for combination in combinations:
        mask = np.ones(len(binary), dtype=bool)
        for cause, value in zip(CAUSE_NODES, combination, strict=True):
            mask &= binary[cause].to_numpy() == value
        outcomes = binary.loc[mask, TARGET_COLUMN]
        probabilities[combination] = float(
            (outcomes.sum() + smoothing) / (len(outcomes) + 2 * smoothing)
        )

    engine, engine_build_error = _build_pgmpy_engine(cause_priors, probabilities, combinations)
    inference_mode = "pgmpy_exact" if engine is not None else "empirical_table"
    return BayesianBundle(
        probabilities,
        cause_priors,
        cause_lifts,
        engine,
        inference_mode=inference_mode,
        engine_build_error=engine_build_error,
    )


def fit_bayesian_model(
    historical: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> BayesianBundle:
    """Compatibility alias for fitting the Bayesian network."""
    return fit_bayesian_network(historical, smoothing=smoothing)


def _build_pgmpy_engine(
    priors: dict[str, float],
    probabilities: dict[tuple[int, ...], float],
    combinations: list[tuple[int, ...]],
) -> tuple[Any | None, str | None]:
    """Build the exact pgmpy inference engine, or explicitly report why not.

    The only legitimate, silent-fallback trigger is that pgmpy itself is not
    importable/compatible in this environment (an explicit availability check).
    Any error while constructing the network from our own CPDs (shape/model
    errors) indicates a bug in this code and must propagate rather than being
    hidden behind the empirical fallback.
    """
    try:
        try:
            from pgmpy.models import DiscreteBayesianNetwork as Network
        except ImportError:
            from pgmpy.models import BayesianNetwork as Network
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
    except ImportError as exc:
        return None, f"pgmpy is unavailable in this environment: {exc}"

    network = Network([(cause, ENDPOINT) for cause in CAUSE_NODES])
    cpds = [
        TabularCPD(
            variable=cause,
            variable_card=2,
            values=[[1 - priors[cause]], [priors[cause]]],
            state_names={cause: [0, 1]},
        )
        for cause in CAUSE_NODES
    ]
    positive = [probabilities[combination] for combination in combinations]
    outcome_cpd = TabularCPD(
        variable=ENDPOINT,
        variable_card=2,
        values=[[1 - value for value in positive], positive],
        evidence=list(CAUSE_NODES),
        evidence_card=[2] * len(CAUSE_NODES),
        state_names={ENDPOINT: [0, 1], **{cause: [0, 1] for cause in CAUSE_NODES}},
    )
    network.add_cpds(*cpds, outcome_cpd)
    network.check_model()
    return VariableElimination(network), None


def _as_binary(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return 1
        if normalized in {"false", "no", "n", "0", ""}:
            return 0
    try:
        return int(float(value) >= 0.5)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot convert {value!r} to binary evidence") from error
