"""A compact, interpretable causal-chain Bayesian network for OTIF-miss risk.

Replaces a direct "all seven causes point straight at OTIF_MISS" star network
with a small propagation graph that tells a clearer compounding-risk story:

    ORDER_CAPTURE ------------------------------------------------------+
    VENDOR_FAILURE -> INVENTORY_SHORTAGE -+                            |
    DC_CAPACITY --------------------------+-> WAREHOUSE_OPS -> TRANSPORT -> OTIF_MISS
                                                                         |
    CUSTOMER_DELIVERY -------------------------------------------------+

CPTs are learned (with additive smoothing) from the *training split's*
resolved operational stage history. Stage failures are recorded for every
closed order, including orders that still achieved OTIF, so the network can
learn whether disruption propagated or was absorbed. Scoring uses point-in-time evidence: a
node's binary leading signal is only used as *hard* evidence once its stage
has actually been observed (``vendor_ready_observed``, ``shipped_observed``,
``transit_observed``); unobserved intermediate nodes are marginalized out via
exact inference rather than assumed to be 0, which is what makes the chain's
propagation (e.g. an early vendor-failure signal raising OTIF risk before the
warehouse/transport stages have even happened) show up in the score.

Two exact-inference engines are supported and explicitly reported:
``pgmpy_exact`` (variable elimination) when pgmpy is importable, and
``brute_force_exact`` (full joint enumeration over this small 8-node binary
network) when it is not. Both are mathematically exact for a network this
size; there is no approximate/empirical fallback.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from otif_risk.contracts import CAUSE_CATEGORIES
from otif_risk.model import ENDPOINT, TARGET_COLUMN

#: Chain topology: node -> tuple of parent node names (empty tuple = root).
CHAIN_PARENTS: dict[str, tuple[str, ...]] = {
    "ORDER_CAPTURE": (),
    "VENDOR_FAILURE": (),
    "DC_CAPACITY": (),
    "CUSTOMER_DELIVERY": (),
    "INVENTORY_SHORTAGE": ("VENDOR_FAILURE",),
    "WAREHOUSE_OPS": ("INVENTORY_SHORTAGE", "DC_CAPACITY"),
    "TRANSPORT": ("WAREHOUSE_OPS",),
    ENDPOINT: (
        "ORDER_CAPTURE",
        "INVENTORY_SHORTAGE",
        "WAREHOUSE_OPS",
        "TRANSPORT",
        "CUSTOMER_DELIVERY",
    ),
}
#: Topological order (parents always precede children).
CHAIN_NODES: tuple[str, ...] = (
    "ORDER_CAPTURE",
    "VENDOR_FAILURE",
    "DC_CAPACITY",
    "CUSTOMER_DELIVERY",
    "INVENTORY_SHORTAGE",
    "WAREHOUSE_OPS",
    "TRANSPORT",
    ENDPOINT,
)
CAUSE_NODES = tuple(cause for cause in CAUSE_CATEGORIES)
SIGNAL_COLUMNS = tuple(f"leading_signal_{category}" for category in CAUSE_CATEGORIES)

#: The observed-by-prediction-time flag column for each cause node, or ``None``
#: when that node's evidence is always knowable (order capture, initial
#: allocation/ATP, DC capacity snapshot, customer master data).
OBSERVABILITY_COLUMN: dict[str, str | None] = {
    "ORDER_CAPTURE": None,
    "VENDOR_FAILURE": "vendor_ready_observed",
    "INVENTORY_SHORTAGE": None,
    "DC_CAPACITY": None,
    "WAREHOUSE_OPS": "shipped_observed",
    "TRANSPORT": "transit_observed",
    "CUSTOMER_DELIVERY": None,
}

#: For narrating a route, the downstream path from each root/near-root cause
#: to the endpoint, following the fixed chain edges above.
CHAIN_ROUTES: dict[str, tuple[str, ...]] = {
    "ORDER_CAPTURE": ("ORDER_CAPTURE", ENDPOINT),
    "VENDOR_FAILURE": (
        "VENDOR_FAILURE",
        "INVENTORY_SHORTAGE",
        "WAREHOUSE_OPS",
        "TRANSPORT",
        ENDPOINT,
    ),
    "INVENTORY_SHORTAGE": ("INVENTORY_SHORTAGE", "WAREHOUSE_OPS", "TRANSPORT", ENDPOINT),
    "DC_CAPACITY": ("DC_CAPACITY", "WAREHOUSE_OPS", "TRANSPORT", ENDPOINT),
    "WAREHOUSE_OPS": ("WAREHOUSE_OPS", "TRANSPORT", ENDPOINT),
    "TRANSPORT": ("TRANSPORT", ENDPOINT),
    "CUSTOMER_DELIVERY": ("CUSTOMER_DELIVERY", ENDPOINT),
}

QUALIFICATION = (
    "probabilistic_association_within_a_fixed_chain_structure_not_a_proven_causal_mechanism"
)


@dataclass
class BayesianBundle:
    """Serializable exact-inference scorer for the compact causal chain."""

    cpts: dict[str, dict[tuple[int, ...], float]]
    cause_lifts: dict[str, float]
    prior_risk: float
    inference_engine: Any | None = field(default=None, repr=False)
    endpoint: str = ENDPOINT
    inference_mode: str = "brute_force_exact"
    engine_build_error: str | None = None

    def score(self, evidence_frame: pd.DataFrame) -> pd.DataFrame:
        """Score orders and return risk + a pathway JSON per order."""
        if "order_id" not in evidence_frame:
            raise ValueError("evidence_frame must contain order_id")
        missing = sorted(set(SIGNAL_COLUMNS) - set(evidence_frame.columns))
        if missing:
            raise ValueError(f"missing Bayesian evidence columns: {missing}")

        risks: list[float] = []
        pathways: list[str] = []
        for _, row in evidence_frame.iterrows():
            evidence = self._row_evidence(row)
            posterior = self._query(evidence)
            risks.append(posterior)
            pathways.append(json.dumps(self._pathway(evidence, posterior), separators=(",", ":")))
        return pd.DataFrame(
            {
                "order_id": evidence_frame["order_id"].to_numpy(),
                "bbn_risk_score": np.clip(risks, 0.0, 1.0),
                "causal_pathway": pathways,
            }
        )

    def _row_evidence(self, row: pd.Series) -> dict[str, int]:
        """Only include a node as hard evidence once its stage is observed."""
        evidence: dict[str, int] = {}
        for cause in CAUSE_NODES:
            observability_column = OBSERVABILITY_COLUMN[cause]
            observed = (
                True
                if observability_column is None
                else bool(row.get(observability_column, True))
            )
            if observed:
                evidence[cause] = _as_binary(row[f"leading_signal_{cause}"])
        return evidence

    def _query(self, evidence: dict[str, int]) -> float:
        if self.inference_engine is not None:
            # An engine was successfully constructed, so this is the sole,
            # explicitly recorded (`inference_mode == "pgmpy_exact"`) path
            # that may use it. Any query-time error here is a real defect
            # and must surface rather than being silently swallowed.
            result = self.inference_engine.query(
                variables=[ENDPOINT],
                evidence=evidence,
                show_progress=False,
            )
            state_names = result.state_names.get(ENDPOINT, [0, 1])
            positive_index = state_names.index(1)
            return float(np.asarray(result.values)[positive_index])
        # No pgmpy engine is available (`engine_build_error` records why, an
        # explicit reported condition). The brute-force enumeration below is
        # exact for this 8-node binary network, not an approximation.
        return _enumerate_posterior(self.cpts, evidence)

    def _pathway(self, evidence: dict[str, int], posterior: float) -> dict[str, Any]:
        active = [cause for cause, value in evidence.items() if value == 1 and cause in CAUSE_NODES]
        active.sort(key=lambda cause: abs(self.cause_lifts.get(cause, 0.0)), reverse=True)
        primary_route: tuple[str, ...] = ()
        if active:
            upstream_cause = next(
                cause for cause in CAUSE_CATEGORIES if evidence.get(cause) == 1
            )
            primary_route = CHAIN_ROUTES.get(
                upstream_cause, (upstream_cause, ENDPOINT)
            )
        return {
            "endpoint": self.endpoint,
            "evidence": evidence,
            "active_evidence": active,
            "route": list(primary_route) if primary_route else [],
            "posterior_risk": round(posterior, 6),
            "prior_risk": round(self.prior_risk, 6),
            "evidence_delta": round(posterior - self.prior_risk, 6),
            "inference_mode": self.inference_mode,
            "interpretation": QUALIFICATION,
        }


def fit_bayesian_network(
    historical: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> BayesianBundle:
    """Fit smoothed binary CPTs for the compact causal chain."""
    stage_columns = {f"stage_{cause}" for cause in CAUSE_NODES}
    cause_columns = {f"cause_{cause}" for cause in CAUSE_NODES}
    if stage_columns <= set(historical.columns):
        source_prefix = "stage_"
    elif cause_columns <= set(historical.columns):
        source_prefix = "cause_"
    else:
        missing = sorted(stage_columns - set(historical.columns))
        raise ValueError(f"historical frame is missing columns: {missing}")
    if TARGET_COLUMN not in historical:
        raise ValueError(f"historical frame is missing columns: ['{TARGET_COLUMN}']")
    if historical.empty:
        raise ValueError("historical frame must not be empty")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")

    binary = pd.DataFrame(
        {
            cause: historical[f"{source_prefix}{cause}"].map(_as_binary)
            for cause in CAUSE_NODES
        }
    )
    binary[ENDPOINT] = historical[TARGET_COLUMN].map(_as_binary)

    cpts: dict[str, dict[tuple[int, ...], float]] = {}
    for node in CHAIN_NODES:
        parents = CHAIN_PARENTS[node]
        cpts[node] = _fit_node_cpt(binary, node, parents, smoothing)

    global_risk = float(binary[ENDPOINT].mean())
    cause_lifts = {}
    for cause in CAUSE_NODES:
        active = binary.loc[binary[cause] == 1, ENDPOINT]
        active_risk = (
            float((active.sum() + smoothing) / (len(active) + 2 * smoothing))
            if len(active)
            else global_risk
        )
        cause_lifts[cause] = active_risk - global_risk

    prior_risk = _enumerate_posterior(cpts, {})
    engine, engine_build_error = _build_pgmpy_engine(cpts)
    inference_mode = "pgmpy_exact" if engine is not None else "brute_force_exact"
    return BayesianBundle(
        cpts=cpts,
        cause_lifts=cause_lifts,
        prior_risk=prior_risk,
        inference_engine=engine,
        inference_mode=inference_mode,
        engine_build_error=engine_build_error,
    )


def fit_bayesian_model(historical: pd.DataFrame, *, smoothing: float = 1.0) -> BayesianBundle:
    """Convenience alias for fitting the Bayesian network."""
    return fit_bayesian_network(historical, smoothing=smoothing)


def _fit_node_cpt(
    binary: pd.DataFrame,
    node: str,
    parents: tuple[str, ...],
    smoothing: float,
) -> dict[tuple[int, ...], float]:
    cpt: dict[tuple[int, ...], float] = {}
    for combination in itertools.product((0, 1), repeat=len(parents)):
        if parents:
            mask = np.ones(len(binary), dtype=bool)
            for parent, value in zip(parents, combination, strict=True):
                mask &= binary[parent].to_numpy() == value
            subset = binary.loc[mask, node]
        else:
            subset = binary[node]
        cpt[combination] = float(
            (subset.sum() + smoothing) / (len(subset) + 2 * smoothing)
        )
    return cpt


def _enumerate_posterior(
    cpts: dict[str, dict[tuple[int, ...], float]], evidence: dict[str, int]
) -> float:
    """Exact P(OTIF_MISS=1 | evidence) via brute-force joint enumeration.

    The network has only 8 binary nodes, so enumerating every assignment of
    the (at most 7) non-evidenced cause nodes -- 2^7 = 128 combinations -- is
    both exact and computationally trivial, unlike a real production causal
    graph where this would not scale.
    """
    free_nodes = [node for node in CAUSE_NODES if node not in evidence]
    total_positive = 0.0
    total_negative = 0.0
    for combination in itertools.product((0, 1), repeat=len(free_nodes)):
        assignment = dict(evidence)
        assignment.update(zip(free_nodes, combination, strict=True))
        for endpoint_value in (0, 1):
            assignment[ENDPOINT] = endpoint_value
            joint = 1.0
            for node in CHAIN_NODES:
                parents = CHAIN_PARENTS[node]
                parent_values = tuple(assignment[parent] for parent in parents)
                p1 = cpts[node][parent_values]
                joint *= p1 if assignment[node] == 1 else (1 - p1)
            if endpoint_value == 1:
                total_positive += joint
            else:
                total_negative += joint
    denominator = total_positive + total_negative
    if denominator <= 0:
        return 0.0
    return total_positive / denominator


def _build_pgmpy_engine(
    cpts: dict[str, dict[tuple[int, ...], float]],
) -> tuple[Any | None, str | None]:
    """Build the exact pgmpy inference engine, or explicitly report why not.

    The only legitimate, silent-fallback trigger is that pgmpy itself is not
    importable/compatible in this environment (an explicit availability
    check). Any error while constructing the network from our own CPDs
    indicates a bug in this code and must propagate.
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

    edges = [
        (parent, node) for node, parents in CHAIN_PARENTS.items() for parent in parents
    ]
    network = Network(edges)
    cpds = []
    for node in CHAIN_NODES:
        parents = CHAIN_PARENTS[node]
        if not parents:
            p1 = cpts[node][()]
            cpds.append(
                TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[1 - p1], [p1]],
                    state_names={node: [0, 1]},
                )
            )
            continue
        combinations = list(itertools.product((0, 1), repeat=len(parents)))
        positive = [cpts[node][combination] for combination in combinations]
        cpds.append(
            TabularCPD(
                variable=node,
                variable_card=2,
                values=[[1 - value for value in positive], positive],
                evidence=list(parents),
                evidence_card=[2] * len(parents),
                state_names={node: [0, 1], **{parent: [0, 1] for parent in parents}},
            )
        )
    network.add_cpds(*cpds)
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
