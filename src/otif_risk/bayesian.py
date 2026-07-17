"""An interpretable 10-node mechanism Bayesian network for OTIF-miss risk.

The network mirrors the actual OTIF definition -- "on time" AND "in full" --
instead of pointing all seven operational cause nodes directly at one opaque
OTIF_MISS node:

    ORDER_CAPTURE --------------------------------------------------+
    VENDOR_FAILURE -> INVENTORY_SHORTAGE -+                         |
    DC_CAPACITY ---------------------------+-> WAREHOUSE_OPS -> TRANSPORT -+
                                            |                              |
                                            +-> IN_FULL_FAILURE            +-> LATE_DELIVERY
    CUSTOMER_DELIVERY -----------------------------------------------------+
                                            |                              |
                                            +---------> OTIF_MISS <--------+

``INVENTORY_SHORTAGE`` feeds ``IN_FULL_FAILURE`` directly (a quantity failure);
``ORDER_CAPTURE``, ``WAREHOUSE_OPS``, ``TRANSPORT``, and ``CUSTOMER_DELIVERY``
feed ``LATE_DELIVERY`` (a timing failure); both mechanism nodes feed
``OTIF_MISS``. This creates an immediately explainable split between "we ran
short" and "we ran late" -- something the earlier flat structure could not
express -- while retaining the same operational propagation
(``VENDOR_FAILURE -> INVENTORY_SHORTAGE``, ``{INVENTORY_SHORTAGE,
DC_CAPACITY} -> WAREHOUSE_OPS``, ``WAREHOUSE_OPS -> TRANSPORT``).

CPTs are learned (with additive smoothing) from the *training split's*
resolved operational stage history, including ``IN_FULL_FAILURE = 1 -
in_full`` and ``LATE_DELIVERY = 1 - on_time`` computed directly from that
split's own resolved outcomes -- never from the seven-category failure-only
cause labels alone. Stage failures are recorded for every closed order,
including orders that still achieved OTIF, so the network can learn whether
disruption propagated or was absorbed. Scoring uses point-in-time evidence: a
cause node's binary leading signal is only used as *hard* evidence once its
stage has actually been observed (``vendor_ready_observed``,
``shipped_observed``, ``transit_observed``); unobserved intermediate nodes --
including the two mechanism nodes, which are never directly observable before
an order closes -- are marginalized out via exact inference rather than
assumed to be 0 or 1.

Two exact-inference engines are supported and explicitly reported:
``pgmpy_exact`` (variable elimination) when pgmpy is importable, and
``brute_force_exact`` (full joint enumeration over this small 10-node binary
network) when it is not. Both are mathematically exact for a network this
size for ordinary observational queries. **Structural interventions**
(``do(node=value)``) always use the brute-force enumeration path, deliberately,
because this prototype does not implement/verify pgmpy's do-operator support;
see :meth:`BayesianBundle.intervene`.

Note on ``OTIF_MISS``: in the training data it is *deterministically*
``IN_FULL_FAILURE OR LATE_DELIVERY`` (see ``root_causes.calculate_outcomes``).
Its CPT is nonetheless fit the same way as every other node's -- smoothed
counts from data, never hand-coded as a boolean gate -- so it converges very
close to, but not exactly, a logical OR (e.g. ``P(OTIF_MISS=1 |
IN_FULL_FAILURE=0, LATE_DELIVERY=0)`` is a small positive number, not exactly
0). This is a deliberate consistency choice (every node in this network is
learned, none is asserted), not a claim that the two mechanisms combine via
anything other than OR.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from otif_risk.contracts import CAUSE_CATEGORIES
from otif_risk.decisions import FALLBACK_RECOMMENDATION, RECOMMENDATION_TABLE
from otif_risk.model import ENDPOINT, TARGET_COLUMN

#: The two intermediate "mechanism" nodes that split the OTIF definition into
#: its quantity (in-full) and timing (on-time) halves.
IN_FULL_FAILURE = "IN_FULL_FAILURE"
LATE_DELIVERY = "LATE_DELIVERY"
MECHANISM_NODES: tuple[str, ...] = (IN_FULL_FAILURE, LATE_DELIVERY)

#: Chain topology: node -> tuple of parent node names (empty tuple = root).
CHAIN_PARENTS: dict[str, tuple[str, ...]] = {
    "ORDER_CAPTURE": (),
    "VENDOR_FAILURE": (),
    "DC_CAPACITY": (),
    "CUSTOMER_DELIVERY": (),
    "INVENTORY_SHORTAGE": ("VENDOR_FAILURE",),
    "WAREHOUSE_OPS": ("INVENTORY_SHORTAGE", "DC_CAPACITY"),
    "TRANSPORT": ("WAREHOUSE_OPS",),
    IN_FULL_FAILURE: ("INVENTORY_SHORTAGE",),
    LATE_DELIVERY: ("ORDER_CAPTURE", "WAREHOUSE_OPS", "TRANSPORT", "CUSTOMER_DELIVERY"),
    ENDPOINT: (IN_FULL_FAILURE, LATE_DELIVERY),
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
    IN_FULL_FAILURE,
    LATE_DELIVERY,
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

QUALIFICATION = (
    "probabilistic_association_within_a_fixed_chain_structure_not_a_proven_causal_mechanism"
)
#: Explicit qualification attached to every structural-intervention scenario.
#: Deliberately verbose: this is the one piece of qualifying language the UI
#: and any downstream consumer must never drop or paraphrase away.
INTERVENTION_QUALIFICATION = (
    "Fixed-structure scenario analysis -- not a proven treatment effect. "
    "do(node=value) replaces the intervened node's fitted structural equation "
    "(CPT) with a fixed value in this Bayesian network, severing its parents' "
    "influence on it while its children still respond to the fixed value; it "
    "is an exact computation under the model's fixed assumptions, not an "
    "identified or randomized causal effect estimate."
)
#: Explicit qualification attached to every evidence-attribution row.
ATTRIBUTION_QUALIFICATION = (
    "evidence_attribution_leave_one_out -- not SHAP and not a causal effect "
    "estimate. Measures how much this fixed network's posterior changes when "
    "one observed cause node is withheld from evidence (marginalized out) "
    "instead of conditioned on."
)

#: Deterministic evidence-coverage thresholds for the causal_confidence band.
#: Four of the seven cause nodes are always observable (no gating column), so
#: coverage never falls below 4/7 (~0.571); these breakpoints line up with
#: "no gated stage observed yet" (LOW), "some gated stages observed" (MEDIUM),
#: and "every gated stage observed" (HIGH).
EVIDENCE_COVERAGE_HIGH_THRESHOLD = 0.95
EVIDENCE_COVERAGE_MEDIUM_THRESHOLD = 0.65


def _build_children(parents: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {node: [] for node in parents}
    for node, node_parents in parents.items():
        for parent in node_parents:
            children[parent].append(node)
    return children


CHAIN_CHILDREN: dict[str, list[str]] = _build_children(CHAIN_PARENTS)


def _all_routes(node: str, target: str, children: dict[str, list[str]]) -> list[tuple[str, ...]]:
    """Every simple directed path from ``node`` to ``target`` following ``children``."""
    if node == target:
        return [(node,)]
    routes: list[tuple[str, ...]] = []
    for child in children.get(node, []):
        for suffix in _all_routes(child, target, children):
            routes.append((node, *suffix))
    return routes


#: Every mechanism route from each non-endpoint node to OTIF_MISS. Cause nodes
#: that feed both mechanisms (e.g. VENDOR_FAILURE, INVENTORY_SHORTAGE) have
#: more than one route; this is the "mechanism route(s), not one route only"
#: contract used by pathway/attribution/intervention reporting.
ALL_ROUTES: dict[str, tuple[tuple[str, ...], ...]] = {
    node: tuple(_all_routes(node, ENDPOINT, CHAIN_CHILDREN))
    for node in CHAIN_NODES
    if node != ENDPOINT
}


@dataclass
class BayesianBundle:
    """Serializable exact-inference scorer for the 10-node mechanism network."""

    cpts: dict[str, dict[tuple[int, ...], float]]
    cause_lifts: dict[str, float]
    prior_risk: float
    inference_engine: Any | None = field(default=None, repr=False)
    endpoint: str = ENDPOINT
    inference_mode: str = "brute_force_exact"
    engine_build_error: str | None = None

    def score(self, evidence_frame: pd.DataFrame) -> pd.DataFrame:
        """Score orders and return risk, mechanism posteriors, and diagnostics per order."""
        if "order_id" not in evidence_frame:
            raise ValueError("evidence_frame must contain order_id")
        missing = sorted(set(SIGNAL_COLUMNS) - set(evidence_frame.columns))
        if missing:
            raise ValueError(f"missing Bayesian evidence columns: {missing}")

        rows: list[dict[str, Any]] = []
        for _, row in evidence_frame.iterrows():
            evidence = self._row_evidence(row)
            posterior = self._query(evidence, ENDPOINT)
            p_in_full = self._query(evidence, IN_FULL_FAILURE)
            p_late = self._query(evidence, LATE_DELIVERY)
            coverage = len(evidence) / len(CAUSE_NODES)
            confidence = _confidence_band(coverage)
            attribution = self._evidence_attribution(evidence, posterior)
            scenarios = self._intervention_scenarios(evidence, posterior)
            pathway = self._pathway(
                evidence,
                posterior,
                {IN_FULL_FAILURE: p_in_full, LATE_DELIVERY: p_late},
                coverage,
                confidence,
            )
            rows.append(
                {
                    "order_id": row["order_id"],
                    "bbn_risk_score": float(np.clip(posterior, 0.0, 1.0)),
                    "in_full_failure_probability": float(np.clip(p_in_full, 0.0, 1.0)),
                    "late_delivery_probability": float(np.clip(p_late, 0.0, 1.0)),
                    "evidence_coverage": round(coverage, 6),
                    "causal_confidence": confidence,
                    "causal_pathway": json.dumps(pathway, separators=(",", ":")),
                    "causal_attribution_json": json.dumps(attribution, separators=(",", ":")),
                    "intervention_scenarios_json": json.dumps(scenarios, separators=(",", ":")),
                }
            )
        return pd.DataFrame(rows)

    def intervene(self, evidence: dict[str, int], do: dict[str, int]) -> dict[str, Any]:
        """Exact ``P(OTIF_MISS=1 | evidence, do(node=value))`` under this fixed BN.

        This is a **structural intervention scenario**, not a proven causal
        effect: it replaces each intervened node's fitted structural equation
        with the given constant (severing its parents' influence on it, while
        its children still respond normally to the fixed value), then
        re-enumerates the exact joint posterior. It is deliberately computed
        via brute-force enumeration only -- never through the pgmpy
        observational-query engine, whose do-operator support is not
        implemented/verified here -- so the result is always exact and
        auditable from the fitted CPTs alone.

        Only operational cause nodes may be intervened on (never the
        mechanism nodes or the endpoint itself); invalid node names or
        non-binary values are rejected. Overlapping ``evidence``/``do`` values
        for the same node are allowed by design (that is precisely a
        "mitigate this observed active cause" scenario); neither ``evidence``
        nor ``do`` nor ``self.cpts`` is mutated by this call.
        """
        self._validate_evidence(evidence)
        self._validate_intervention(do)
        baseline = self._query(evidence, ENDPOINT)
        post = _posterior(self.cpts, evidence, ENDPOINT, do=do)
        return self._scenario_payload(baseline, post, tuple(do), dict(do))

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

    def _query(self, evidence: dict[str, int], query_node: str) -> float:
        if self.inference_engine is not None:
            # An engine was successfully constructed, so this is the sole,
            # explicitly recorded (`inference_mode == "pgmpy_exact"`) path
            # that may use it. Any query-time error here is a real defect
            # and must surface rather than being silently swallowed.
            result = self.inference_engine.query(
                variables=[query_node],
                evidence=evidence,
                show_progress=False,
            )
            state_names = result.state_names.get(query_node, [0, 1])
            positive_index = state_names.index(1)
            return float(np.asarray(result.values)[positive_index])
        # No pgmpy engine is available (`engine_build_error` records why, an
        # explicit reported condition). The brute-force enumeration below is
        # exact for this 10-node binary network, not an approximation.
        return _posterior(self.cpts, evidence, query_node)

    def _evidence_attribution(
        self, evidence: dict[str, int], baseline_posterior: float
    ) -> list[dict[str, Any]]:
        """Leave-one-evidence-out contribution for every active observed cause node."""
        active = [cause for cause, value in evidence.items() if value == 1]
        rows: list[dict[str, Any]] = []
        for cause in active:
            withheld_evidence = {k: v for k, v in evidence.items() if k != cause}
            withheld_posterior = self._query(withheld_evidence, ENDPOINT)
            contribution = baseline_posterior - withheld_posterior
            if contribution > 1e-9:
                direction = "increases_risk"
            elif contribution < -1e-9:
                direction = "decreases_risk"
            else:
                direction = "no_effect"
            rows.append(
                {
                    "node": cause,
                    "observed": True,
                    "observed_value": 1,
                    "baseline_posterior": round(baseline_posterior, 6),
                    "withheld_posterior": round(withheld_posterior, 6),
                    "contribution": round(contribution, 6),
                    "direction": direction,
                    "method": "evidence_attribution_leave_one_out",
                    "qualification": ATTRIBUTION_QUALIFICATION,
                }
            )
        rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return rows

    def _intervention_scenarios(
        self, evidence: dict[str, int], baseline_posterior: float
    ) -> list[dict[str, Any]]:
        """One do(node=0) scenario per active cause node, plus a combined scenario."""
        active = tuple(cause for cause, value in evidence.items() if value == 1)
        scenarios: list[dict[str, Any]] = []
        for cause in active:
            do = {cause: 0}
            post = _posterior(self.cpts, evidence, ENDPOINT, do=do)
            scenarios.append(self._scenario_payload(baseline_posterior, post, (cause,), do))
        if len(active) > 1:
            do = {cause: 0 for cause in active}
            post = _posterior(self.cpts, evidence, ENDPOINT, do=do)
            scenarios.append(self._scenario_payload(baseline_posterior, post, active, do))
        return scenarios

    def _scenario_payload(
        self,
        baseline_posterior: float,
        post_intervention_posterior: float,
        nodes: tuple[str, ...],
        do: dict[str, int],
    ) -> dict[str, Any]:
        absolute = baseline_posterior - post_intervention_posterior
        relative = (absolute / baseline_posterior) if baseline_posterior > 1e-9 else 0.0
        routes = sorted({route for node in nodes for route in ALL_ROUTES.get(node, ())})
        assumed_actions = [
            {
                "node": node,
                "action": RECOMMENDATION_TABLE.get(node, FALLBACK_RECOMMENDATION)["action"],
                "owner": RECOMMENDATION_TABLE.get(node, FALLBACK_RECOMMENDATION)["owner"],
            }
            for node in nodes
        ]
        return {
            "type": "combined_mitigation" if len(nodes) > 1 else "single_node_mitigation",
            "intervened_nodes": list(nodes),
            "do": dict(do),
            "baseline_bayesian_posterior": round(baseline_posterior, 6),
            "post_intervention_bayesian_posterior": round(post_intervention_posterior, 6),
            "absolute_risk_reduction": round(absolute, 6),
            "relative_risk_reduction": round(relative, 6),
            "routes": [list(route) for route in routes],
            "assumed_actions": assumed_actions,
            "inference_mode_used": "brute_force_exact",
            "qualification": INTERVENTION_QUALIFICATION,
        }

    def _pathway(
        self,
        evidence: dict[str, int],
        posterior: float,
        mechanism_posteriors: dict[str, float],
        coverage: float,
        confidence: str,
    ) -> dict[str, Any]:
        active = [cause for cause, value in evidence.items() if value == 1 and cause in CAUSE_NODES]
        active.sort(key=lambda cause: abs(self.cause_lifts.get(cause, 0.0)), reverse=True)
        routes = sorted({route for cause in active for route in ALL_ROUTES.get(cause, ())})
        primary_route: tuple[str, ...] = ()
        if active:
            candidate_routes = ALL_ROUTES.get(active[0], ((active[0], self.endpoint),))
            primary_route = min(candidate_routes, key=len)
        return {
            "endpoint": self.endpoint,
            "evidence": evidence,
            "active_evidence": active,
            "active_evidence_count": len(active),
            "route": list(primary_route) if primary_route else [],
            "routes": [list(route) for route in routes],
            "mechanism_posteriors": {
                IN_FULL_FAILURE: round(mechanism_posteriors[IN_FULL_FAILURE], 6),
                LATE_DELIVERY: round(mechanism_posteriors[LATE_DELIVERY], 6),
            },
            "posterior_risk": round(posterior, 6),
            "prior_risk": round(self.prior_risk, 6),
            "evidence_delta": round(posterior - self.prior_risk, 6),
            "evidence_coverage": round(coverage, 6),
            "confidence": confidence,
            "inference_mode": self.inference_mode,
            "interpretation": QUALIFICATION,
        }

    def _validate_evidence(self, evidence: dict[str, int]) -> None:
        invalid_nodes = sorted(set(evidence) - set(CAUSE_NODES))
        if invalid_nodes:
            raise ValueError(
                f"evidence may only include operational cause nodes, got: {invalid_nodes}"
            )
        invalid_values = {node: value for node, value in evidence.items() if value not in (0, 1)}
        if invalid_values:
            raise ValueError(f"evidence values must be 0 or 1, got: {invalid_values}")

    def _validate_intervention(self, do: dict[str, int]) -> None:
        if not do:
            raise ValueError("do must specify at least one operational cause node")
        invalid_nodes = sorted(set(do) - set(CAUSE_NODES))
        if invalid_nodes:
            raise ValueError(
                "structural interventions are only permitted on operational cause "
                f"nodes, got: {invalid_nodes}"
            )
        invalid_values = {node: value for node, value in do.items() if value not in (0, 1)}
        if invalid_values:
            raise ValueError(f"intervention values must be 0 or 1, got: {invalid_values}")


def fit_bayesian_network(
    historical: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> BayesianBundle:
    """Fit smoothed binary CPTs for the 10-node mechanism network."""
    stage_columns = {f"stage_{cause}" for cause in CAUSE_NODES}
    cause_columns = {f"cause_{cause}" for cause in CAUSE_NODES}
    if stage_columns <= set(historical.columns):
        source_prefix = "stage_"
    elif cause_columns <= set(historical.columns):
        source_prefix = "cause_"
    else:
        missing = sorted(stage_columns - set(historical.columns))
        raise ValueError(f"historical frame is missing columns: {missing}")
    required_outcome_columns = {TARGET_COLUMN, "on_time", "in_full"}
    if missing := sorted(required_outcome_columns - set(historical.columns)):
        raise ValueError(f"historical frame is missing columns: {missing}")
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
    binary[IN_FULL_FAILURE] = 1 - historical["in_full"].map(_as_binary)
    binary[LATE_DELIVERY] = 1 - historical["on_time"].map(_as_binary)
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

    prior_risk = _posterior(cpts, {}, ENDPOINT)
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


def _confidence_band(coverage: float) -> str:
    """Deterministic LOW/MEDIUM/HIGH band from evidence coverage.

    Four of the seven cause nodes have no observability gate at all, so
    coverage never drops below 4/7 (~0.571). The thresholds below therefore
    map onto "no gated stage observed yet" (LOW), "some gated stages observed"
    (MEDIUM), and "every gated stage observed" (HIGH) rather than an arbitrary
    percentage split.
    """
    if coverage >= EVIDENCE_COVERAGE_HIGH_THRESHOLD:
        return "HIGH"
    if coverage >= EVIDENCE_COVERAGE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _node_factor(
    cpts: dict[str, dict[tuple[int, ...], float]],
    node: str,
    assignment: dict[str, int],
    do: dict[str, int],
) -> float:
    """This node's contribution to the joint weight for one full assignment.

    A node in ``do`` is structurally intervened: its own fitted CPT (its
    parents' influence on it) is replaced with a fixed value, so it
    contributes a factor of 1 rather than its natural conditional
    probability -- the "mutilated graph" semantics of ``do(node=value)``. Its
    children still read ``assignment[node]`` normally, so the fixed value
    still propagates forward exactly as it would if observed.
    """
    if node in do:
        return 1.0
    parents = CHAIN_PARENTS[node]
    parent_values = tuple(assignment[parent] for parent in parents)
    p1 = cpts[node][parent_values]
    return p1 if assignment[node] == 1 else (1 - p1)


def _joint_weight(
    cpts: dict[str, dict[tuple[int, ...], float]],
    assignment: dict[str, int],
    do: dict[str, int],
) -> float:
    weight = 1.0
    for node in CHAIN_NODES:
        weight *= _node_factor(cpts, node, assignment, do)
        if weight == 0.0:
            return 0.0
    return weight


def _marginal_probabilities(
    cpts: dict[str, dict[tuple[int, ...], float]],
    evidence: dict[str, int],
    do: dict[str, int] | None,
    target_nodes: tuple[str, ...],
) -> dict[tuple[int, ...], float]:
    """Exact ``P(target_nodes | evidence, do)`` via brute-force joint enumeration.

    The network has only 10 binary nodes, so enumerating every assignment of
    the non-evidenced, non-intervened, non-target nodes (at most 9) --
    2^9 = 512 combinations -- is both exact and computationally trivial,
    unlike a real production causal graph where this would not scale. Neither
    ``evidence`` nor ``do`` is mutated: both are only read from, and every
    assignment enumerated below is a fresh local dict.
    """
    do = do or {}
    fixed = dict(evidence)
    fixed.update(do)
    free_nodes = [node for node in CHAIN_NODES if node not in fixed and node not in target_nodes]
    target_combinations = list(itertools.product((0, 1), repeat=len(target_nodes)))
    totals: dict[tuple[int, ...], float] = dict.fromkeys(target_combinations, 0.0)
    for free_combination in itertools.product((0, 1), repeat=len(free_nodes)):
        base_assignment = dict(fixed)
        base_assignment.update(zip(free_nodes, free_combination, strict=True))
        for target_combination in target_combinations:
            assignment = dict(base_assignment)
            assignment.update(zip(target_nodes, target_combination, strict=True))
            totals[target_combination] += _joint_weight(cpts, assignment, do)
    denominator = sum(totals.values())
    if denominator <= 0:
        uniform = 1.0 / len(totals)
        return dict.fromkeys(totals, uniform)
    return {combination: value / denominator for combination, value in totals.items()}


def _posterior(
    cpts: dict[str, dict[tuple[int, ...], float]],
    evidence: dict[str, int],
    query_node: str,
    do: dict[str, int] | None = None,
) -> float:
    """Exact ``P(query_node=1 | evidence, do)``."""
    table = _marginal_probabilities(cpts, evidence, do, (query_node,))
    return table[(1,)]


def _build_pgmpy_engine(
    cpts: dict[str, dict[tuple[int, ...], float]],
) -> tuple[Any | None, str | None]:
    """Build the exact pgmpy inference engine, or explicitly report why not.

    The only legitimate, silent-fallback trigger is that pgmpy itself is not
    importable/compatible in this environment (an explicit availability
    check). Any error while constructing the network from our own CPDs
    indicates a bug in this code and must propagate. This engine is used only
    for ordinary observational queries (`.score()`'s baseline/mechanism/
    attribution posteriors); structural interventions always use the
    brute-force path in `_posterior(..., do=...)` regardless of engine
    availability (see `BayesianBundle.intervene`).
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
