"""Typed, allowlisted evidence packets for the read-only AI Copilot.

This module is the single source of truth for what the Copilot (live OpenAI
or deterministic fallback) is allowed to see and cite. Every fact carries a
stable ``id`` that both the live model and the fallback narrator must cite;
the Streamlit UI resolves those IDs back to human-readable evidence so a
judge can trace any claim to its source.

Design rules (see the Grounded LLM Copilot Plan):

* Deterministic code builds the packet -- never the LLM. The LLM/fallback
  only reads and summarizes it.
* Only an explicit allowlist of ``scored_orders.csv`` columns may become
  facts. Anything not in the allowlist is silently ignored, so a future
  column added to the pipeline is never leaked by accident.
* Secrets, file paths, git remotes/SHAs, raw planner-feedback text, and
  simulator/line ground truth (``simulator_truth``/``line_truth``/``shocks``)
  are never included -- those tables are not even read here.
* Lists and strings are truncated deterministically (fixed limits below), so
  packet size is bounded regardless of how large a run is.
* Portfolio questions never get unrestricted DataFrame/SQL access -- see
  ``PORTFOLIO_QUESTIONS`` and ``build_portfolio_evidence_packet``, which is a
  fixed, hand-coded aggregation catalog.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from otif_risk.narratives import parse_top_factors

# --------------------------------------------------------------------------
# Deterministic size limits (kept small and fixed, never model-configurable).
# --------------------------------------------------------------------------
MAX_FACTS_PER_PACKET = 60
MAX_LIST_ITEMS = 5
MAX_STRING_LEN = 220
MAX_ROUTE_NODES = 8

#: Explicit allowlist of scored_orders.csv columns this module may read.
#: Anything else on the row (including any future column) is ignored.
ALLOWED_ORDER_COLUMNS: frozenset[str] = frozenset(
    {
        "order_id",
        "as_of_timestamp",
        "customer_tier",
        "order_priority",
        "decision_status",
        "recommended_action",
        "action_owner",
        "resource_type",
        "resource_id",
        "contested_with",
        "xgb_risk_score",
        "bbn_risk_score",
        "combined_risk_score",
        "causal_confidence",
        "evidence_coverage",
        "top_factors_json",
        "primary_cause",
        "causal_pathway",
        "intervention_scenarios_json",
        "late_delivery_probability",
        "in_full_failure_probability",
        "affected_skus_json",
        "affected_sku_count",
        "affected_line_count",
        "observed_event_count",
        "missing_event_stage_count",
        "remaining_slack_hours",
        "vendor_id",
        "dc_id",
        "lane_id",
        "customer_id",
        "order_value",
        "penalty_rate",
        "estimated_penalty_exposure",
        "estimated_avoidable_penalty",
        "quantity_at_risk",
        "priority_score",
        "total_order_qty",
    }
)

#: Defense-in-depth denylist: even if a caller passes a wider mapping (e.g. a
#: full pandas row with extra columns), any key matching these substrings is
#: refused, regardless of the allowlist above.
_DENYLIST_SUBSTRINGS = (
    "simulator",
    "line_truth",
    "shock",
    "git_",
    "remote",
    "secret",
    "api_key",
    "token",
    "password",
    "file_path",
    "filepath",
    "__file__",
    "feedback_reason",
    "feedback_text",
    "planner_feedback",
)


def _is_denied(key: str) -> bool:
    lowered = key.lower()
    return any(bad in lowered for bad in _DENYLIST_SUBSTRINGS)


def _truncate_text(value: Any, limit: int = MAX_STRING_LEN) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class Fact:
    """One citable, allowlisted fact. ``id`` is the stable citation key."""

    id: str
    label: str
    value: Any
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "value": self.value, "category": self.category}


@dataclass(frozen=True)
class EvidencePacket:
    """Base packet: an ordered, size-bounded list of citable facts."""

    scope: str  # "order" or "portfolio"
    subject: str  # order_id, or portfolio question_id
    generated_at_utc: str
    facts: tuple[Fact, ...] = field(default_factory=tuple)
    persisted_decision_status: str | None = None
    persisted_recommended_action: str | None = None

    def fact_ids(self) -> frozenset[str]:
        return frozenset(fact.id for fact in self.facts)

    def get(self, fact_id: str) -> Fact | None:
        for fact in self.facts:
            if fact.id == fact_id:
                return fact
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "subject": self.subject,
            "generated_at_utc": self.generated_at_utc,
            "persisted_decision_status": self.persisted_decision_status,
            "persisted_recommended_action": self.persisted_recommended_action,
            "facts": [fact.to_dict() for fact in self.facts],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def evidence_hash(self) -> str:
        """Stable SHA-256 over the packet contents, for audit correlation."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _add(facts: list[Fact], seen: set[str], fact: Fact) -> None:
    if fact.id in seen:
        return
    if len(facts) >= MAX_FACTS_PER_PACKET:
        return
    seen.add(fact.id)
    facts.append(fact)


def build_order_evidence_packet(
    order: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> EvidencePacket:
    """Deterministically build the allowlisted, cited evidence packet for one order.

    ``order`` is typically one row of ``scored_orders.csv`` (as a mapping/dict
    or a ``pandas.Series``). Only ``ALLOWED_ORDER_COLUMNS`` are ever read from
    it; anything else -- including any accidental extra column -- is ignored.
    """

    def col(name: str) -> Any:
        if name not in ALLOWED_ORDER_COLUMNS or _is_denied(name):
            return None
        return order.get(name)

    facts: list[Fact] = []
    seen: set[str] = set()

    order_id = _truncate_text(col("order_id"), 64) or "unknown"

    # --- Identity and status ------------------------------------------------
    _add(facts, seen, Fact("identity.order_id", "Order ID", order_id, "identity"))
    _add(
        facts,
        seen,
        Fact("identity.as_of", "As-of timestamp", _truncate_text(col("as_of_timestamp")), "identity"),
    )
    _add(
        facts,
        seen,
        Fact("identity.customer_tier", "Customer tier", _truncate_text(col("customer_tier")), "identity"),
    )
    _add(
        facts,
        seen,
        Fact("identity.order_priority", "Order priority", _truncate_text(col("order_priority")), "identity"),
    )

    decision_status = _truncate_text(col("decision_status"), 32) or "MONITOR"
    recommended_action = _truncate_text(col("recommended_action"))
    _add(facts, seen, Fact("decision.status", "Decision status", decision_status, "decision"))
    _add(facts, seen, Fact("decision.action", "Recommended action", recommended_action, "decision"))
    _add(
        facts,
        seen,
        Fact("decision.owner", "Accountable owner", _truncate_text(col("action_owner")), "decision"),
    )
    _add(
        facts,
        seen,
        Fact(
            "decision.primary_cause",
            "Primary cause category",
            _truncate_text(col("primary_cause")),
            "decision",
        ),
    )

    # --- Model/policy/manifest versions -------------------------------------
    architecture = (metrics or {}).get("architecture", {}) if isinstance(metrics, Mapping) else {}
    if isinstance(architecture, Mapping):
        _add(
            facts,
            seen,
            Fact(
                "version.risk_model",
                "Risk model",
                _truncate_text(architecture.get("risk_model"), 64),
                "version",
            ),
        )
        _add(
            facts,
            seen,
            Fact(
                "version.fusion_policy",
                "Fusion policy",
                _truncate_text(architecture.get("fusion_chosen_label"), 64),
                "version",
            ),
        )
    manifest_map = manifest if isinstance(manifest, Mapping) else {}
    schema_versions = manifest_map.get("schema_versions") if isinstance(manifest_map, Mapping) else None
    if isinstance(schema_versions, Mapping):
        _add(
            facts,
            seen,
            Fact(
                "version.artifact_schema",
                "Artifact schema version",
                _truncate_text(schema_versions.get("artifact_schema_version"), 32),
                "version",
            ),
        )
    run_directory_name = manifest_map.get("run_directory") if isinstance(manifest_map, Mapping) else None
    if run_directory_name:
        _add(
            facts,
            seen,
            Fact("version.run_id", "Pipeline run identifier", _truncate_text(run_directory_name, 40), "version"),
        )

    # --- Risk ----------------------------------------------------------------
    xgb = _finite_float(col("xgb_risk_score"))
    bbn = _finite_float(col("bbn_risk_score"))
    combined = _finite_float(col("combined_risk_score"))
    threshold = _finite_float((metrics or {}).get("threshold")) if isinstance(metrics, Mapping) else None
    _add(facts, seen, Fact("risk.xgboost", "XGBoost risk score", xgb, "risk"))
    _add(facts, seen, Fact("risk.bayesian", "Bayesian network risk score", bbn, "risk"))
    _add(facts, seen, Fact("risk.combined", "Combined OTIF risk score", combined, "risk"))
    if threshold is not None:
        _add(facts, seen, Fact("risk.threshold", "Decision risk threshold", threshold, "risk"))
    _add(
        facts,
        seen,
        Fact(
            "risk.confidence",
            "Causal-confidence band",
            _truncate_text(col("causal_confidence"), 16),
            "risk",
        ),
    )
    coverage = _finite_float(col("evidence_coverage"))
    _add(facts, seen, Fact("risk.evidence_coverage", "Evidence coverage (share of observed nodes)", coverage, "risk"))
    _add(
        facts,
        seen,
        Fact(
            "risk.calibration_note",
            "Calibration note",
            "Scores are probability estimates, not guarantees; fusion weight is tuned on a "
            "held-out validation window, not on this order.",
            "risk",
        ),
    )

    # --- XGBoost explanation (SHAP/perturbation) ------------------------------
    top_factors = _safe_json_loads(col("top_factors_json"))
    if isinstance(top_factors, list):
        for index, item in enumerate(top_factors[:MAX_LIST_ITEMS], start=1):
            if not isinstance(item, Mapping):
                continue
            factor_name = _truncate_text(item.get("factor"), 80)
            contribution = _finite_float(item.get("contribution"))
            direction = _truncate_text(item.get("direction"), 24)
            _add(
                facts,
                seen,
                Fact(
                    f"shap.{index}",
                    f"SHAP/perturbation factor: {factor_name}" if factor_name else f"SHAP factor {index}",
                    {
                        "factor": factor_name,
                        "contribution": contribution,
                        "direction": direction,
                        "interpretation": "association_not_causation",
                    },
                    "shap",
                ),
            )
    else:
        parsed_names = parse_top_factors(col("top_factors_json"))
        for index, name in enumerate(parsed_names[:MAX_LIST_ITEMS], start=1):
            _add(
                facts,
                seen,
                Fact(f"shap.{index}", f"SHAP/perturbation factor: {name}", name, "shap"),
            )
    _add(
        facts,
        seen,
        Fact(
            "shap.limitation",
            "SHAP/perturbation limitation",
            "association_not_causation: these factors describe model attribution, not a proven "
            "causal effect.",
            "shap",
        ),
    )

    # --- Mechanism explanation (Bayesian pathway) -----------------------------
    late_probability = _finite_float(col("late_delivery_probability"))
    in_full_probability = _finite_float(col("in_full_failure_probability"))
    _add(facts, seen, Fact("mechanism.late_delivery_probability", "Late-delivery probability", late_probability, "mechanism"))
    _add(
        facts,
        seen,
        Fact("mechanism.in_full_failure_probability", "In-full-failure probability", in_full_probability, "mechanism"),
    )
    pathway = _safe_json_loads(col("causal_pathway"))
    if isinstance(pathway, Mapping):
        active_evidence = [str(node) for node in (pathway.get("active_evidence") or [])][:MAX_LIST_ITEMS]
        if active_evidence:
            _add(
                facts,
                seen,
                Fact("mechanism.active_evidence", "Active evidence nodes", active_evidence, "mechanism"),
            )
        route = [str(node) for node in (pathway.get("route") or [])][:MAX_ROUTE_NODES]
        if route:
            _add(facts, seen, Fact("mechanism.route", "Bayesian propagation route", route, "mechanism"))
        prior = _finite_float(pathway.get("prior_risk"))
        posterior = _finite_float(pathway.get("posterior_risk"))
        delta = _finite_float(pathway.get("evidence_delta"))
        if prior is not None:
            _add(facts, seen, Fact("mechanism.prior_risk", "Prior risk (before evidence)", prior, "mechanism"))
        if posterior is not None:
            _add(facts, seen, Fact("mechanism.posterior_risk", "Posterior risk (after evidence)", posterior, "mechanism"))
        if delta is not None:
            _add(facts, seen, Fact("mechanism.evidence_delta", "Evidence-driven risk delta", delta, "mechanism"))
    intervention_scenarios = _safe_json_loads(col("intervention_scenarios_json"))
    if isinstance(intervention_scenarios, list):
        single_node = [
            item
            for item in intervention_scenarios
            if isinstance(item, Mapping) and item.get("type") == "single_node_mitigation"
        ]
        if single_node:
            best = max(single_node, key=lambda item: item.get("absolute_risk_reduction", 0.0) or 0.0)
            reduction = _finite_float(best.get("absolute_risk_reduction"))
            nodes = [str(node) for node in (best.get("intervened_nodes") or [])][:2]
            if reduction is not None and reduction > 0 and nodes:
                _add(
                    facts,
                    seen,
                    Fact(
                        "mechanism.best_scenario",
                        "Best fixed-structure mitigation scenario",
                        {"nodes": nodes, "absolute_risk_reduction": reduction},
                        "mechanism",
                    ),
                )
    _add(
        facts,
        seen,
        Fact(
            "mechanism.limitation",
            "Bayesian/scenario limitation",
            "probabilistic_association_within_a_fixed_chain_structure_not_a_proven_causal_mechanism: "
            "scenarios are exact do()-calculus under a fixed network, not measured treatment effects.",
            "mechanism",
        ),
    )

    # --- Operational detail ---------------------------------------------------
    affected_skus = _safe_json_loads(col("affected_skus_json"))
    if isinstance(affected_skus, list):
        for item in affected_skus[:MAX_LIST_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            sku_id = _truncate_text(item.get("sku_id"), 32)
            if not sku_id:
                continue
            _add(
                facts,
                seen,
                Fact(
                    f"sku.{sku_id}",
                    f"Affected SKU {sku_id}",
                    {
                        "sku_id": sku_id,
                        "criticality_tier": _truncate_text(item.get("criticality_tier"), 24),
                        "allocation_gap_ratio": _finite_float(item.get("allocation_gap_ratio")),
                        "evidence_strength": _finite_float(item.get("evidence_strength")),
                    },
                    "operational",
                ),
            )
    observed_events = col("observed_event_count")
    missing_events = col("missing_event_stage_count")
    if observed_events is not None:
        _add(
            facts,
            seen,
            Fact("events.observed_count", "Observed lifecycle events", _finite_float(observed_events), "operational"),
        )
    if missing_events is not None:
        _add(
            facts,
            seen,
            Fact(
                "events.missing_stage_count",
                "Missing lifecycle event stages",
                _finite_float(missing_events),
                "operational",
            ),
        )
    slack = _finite_float(col("remaining_slack_hours"))
    if slack is not None:
        _add(facts, seen, Fact("operational.remaining_slack_hours", "Remaining delivery slack (hours)", slack, "operational"))
    for entity, fact_id, label in (
        ("vendor_id", "operational.vendor", "Vendor"),
        ("dc_id", "operational.dc", "Distribution center"),
        ("lane_id", "operational.lane", "Lane"),
        ("customer_id", "operational.customer", "Customer"),
    ):
        value = col(entity)
        if value is not None:
            _add(facts, seen, Fact(fact_id, label, _truncate_text(value, 32), "operational"))
    resource_type = col("resource_type")
    resource_id = col("resource_id")
    if resource_type is not None or resource_id is not None:
        _add(
            facts,
            seen,
            Fact(
                "operational.resource_pool",
                "Contended resource pool",
                {"resource_type": _truncate_text(resource_type, 32), "resource_id": _truncate_text(resource_id, 32)},
                "operational",
            ),
        )
    contested_with = col("contested_with")
    if contested_with is not None and str(contested_with).strip() and str(contested_with).lower() != "nan":
        contested_orders = [item.strip() for item in str(contested_with).split(",") if item.strip()][:MAX_LIST_ITEMS]
        _add(
            facts,
            seen,
            Fact("operational.contested_with", "Contesting orders for the same resource", contested_orders, "operational"),
        )

    # --- Business impact --------------------------------------------------------
    order_value = _finite_float(col("order_value"))
    penalty_exposure = _finite_float(col("estimated_penalty_exposure"))
    avoidable_penalty = _finite_float(col("estimated_avoidable_penalty"))
    quantity_at_risk = _finite_float(col("quantity_at_risk"))
    priority_score = _finite_float(col("priority_score"))
    if order_value is not None:
        _add(facts, seen, Fact("business.order_value", "Order value", order_value, "business"))
    if penalty_exposure is not None:
        _add(facts, seen, Fact("business.penalty_exposure", "Estimated penalty exposure", penalty_exposure, "business"))
    if quantity_at_risk is not None:
        _add(facts, seen, Fact("business.quantity_at_risk", "Quantity at risk", quantity_at_risk, "business"))
    if priority_score is not None:
        _add(facts, seen, Fact("business.priority_score", "Planner priority score", priority_score, "business"))
    if avoidable_penalty is not None:
        _add(
            facts,
            seen,
            Fact(
                "business.simulated_avoidable_penalty",
                "Simulated avoidable penalty (simulator evaluation only, not observed)",
                avoidable_penalty,
                "business",
            ),
        )

    return EvidencePacket(
        scope="order",
        subject=order_id,
        generated_at_utc=datetime.now(UTC).isoformat(),
        facts=tuple(facts),
        persisted_decision_status=decision_status,
        persisted_recommended_action=recommended_action,
    )


# ==========================================================================
# Portfolio evidence packet: a fixed, deterministic query catalog.
# ==========================================================================

PORTFOLIO_QUESTIONS: dict[str, str] = {
    "highest_risk_orders": "Which orders carry the highest OTIF risk right now?",
    "decision_mix": "How many orders are recommended, contested, or monitor-only?",
    "largest_penalty_exposure": "Which orders have the most penalty exposure?",
    "hotspots": "Which vendors, DCs, lanes, or customers are hotspots?",
    "dominant_causes": "What are the dominant root causes across the portfolio?",
    "capacity_conflicts": "Which resource pools have the most contested orders?",
    "low_confidence_orders": "Which orders have the lowest-confidence evidence?",
    "model_health": "What is current model and threshold health?",
    "planner_focus_today": "Where should planners focus today?",
}


def _top_n(df: pd.DataFrame, column: str, n: int = MAX_LIST_ITEMS, ascending: bool = False) -> pd.DataFrame:
    if column not in df.columns or df.empty:
        return df.iloc[0:0]
    return df.sort_values(column, ascending=ascending).head(n)


def _order_summary_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "order_id": _truncate_text(row.get("order_id"), 32),
                "combined_risk_score": _finite_float(row.get("combined_risk_score")),
                "decision_status": _truncate_text(row.get("decision_status"), 16),
                "primary_cause": _truncate_text(row.get("primary_cause"), 32),
                "estimated_penalty_exposure": _finite_float(row.get("estimated_penalty_exposure")),
            }
        )
    return rows


def _compute_highest_risk_orders(decisions: pd.DataFrame, _metrics: Mapping[str, Any] | None) -> list[Fact]:
    top = _top_n(decisions, "combined_risk_score")
    return [Fact("portfolio.highest_risk_orders", "Highest-risk orders", _order_summary_rows(top), "portfolio")]


def _compute_decision_mix(decisions: pd.DataFrame, _metrics: Mapping[str, Any] | None) -> list[Fact]:
    if "decision_status" not in decisions.columns:
        return []
    counts = decisions["decision_status"].value_counts().to_dict()
    return [
        Fact(
            "portfolio.decision_mix",
            "Decision status counts",
            {str(k): int(v) for k, v in counts.items()},
            "portfolio",
        )
    ]


def _compute_largest_penalty_exposure(decisions: pd.DataFrame, _metrics: Mapping[str, Any] | None) -> list[Fact]:
    top = _top_n(decisions, "estimated_penalty_exposure")
    total = _finite_float(decisions.get("estimated_penalty_exposure", pd.Series(dtype=float)).sum())
    facts = [
        Fact("portfolio.largest_penalty_exposure", "Orders with largest penalty exposure", _order_summary_rows(top), "portfolio")
    ]
    if total is not None:
        facts.append(Fact("portfolio.total_penalty_exposure", "Total penalty exposure across scored orders", total, "portfolio"))
    return facts


def _compute_hotspots(decisions: pd.DataFrame, _metrics: Mapping[str, Any] | None) -> list[Fact]:
    facts = []
    for entity, fact_id, label in (
        ("vendor_id", "portfolio.vendor_hotspots", "Vendor hotspots (order count)"),
        ("dc_id", "portfolio.dc_hotspots", "DC hotspots (order count)"),
        ("lane_id", "portfolio.lane_hotspots", "Lane hotspots (order count)"),
        ("customer_id", "portfolio.customer_hotspots", "Customer hotspots (order count)"),
    ):
        if entity not in decisions.columns:
            continue
        counts = decisions[entity].value_counts().head(MAX_LIST_ITEMS)
        facts.append(Fact(fact_id, label, {str(k): int(v) for k, v in counts.items()}, "portfolio"))
    return facts


def _compute_dominant_causes(decisions: pd.DataFrame, _metrics: Mapping[str, Any] | None) -> list[Fact]:
    if "primary_cause" not in decisions.columns:
        return []
    counts = decisions["primary_cause"].value_counts().head(MAX_LIST_ITEMS)
    return [Fact("portfolio.dominant_causes", "Dominant root-cause categories", {str(k): int(v) for k, v in counts.items()}, "portfolio")]


def _compute_capacity_conflicts(decisions: pd.DataFrame, _metrics: Mapping[str, Any] | None) -> list[Fact]:
    if "decision_status" not in decisions.columns or "resource_type" not in decisions.columns:
        return []
    contested = decisions[decisions["decision_status"] == "CONTESTED"]
    if contested.empty:
        return [Fact("portfolio.capacity_conflicts", "Contested orders by resource pool", {}, "portfolio")]
    counts = contested["resource_type"].value_counts().head(MAX_LIST_ITEMS)
    exposure = (
        contested.groupby("resource_type")["estimated_penalty_exposure"].sum().round(2)
        if "estimated_penalty_exposure" in contested.columns
        else pd.Series(dtype=float)
    )
    return [
        Fact(
            "portfolio.capacity_conflicts",
            "Contested orders by resource pool",
            {str(k): int(v) for k, v in counts.items()},
            "portfolio",
        ),
        Fact(
            "portfolio.capacity_conflict_exposure",
            "Penalty exposure by contested resource pool",
            {str(k): _finite_float(v) for k, v in exposure.items()},
            "portfolio",
        ),
    ]


def _compute_low_confidence_orders(decisions: pd.DataFrame, _metrics: Mapping[str, Any] | None) -> list[Fact]:
    if "evidence_coverage" not in decisions.columns:
        return []
    top = _top_n(decisions, "evidence_coverage", ascending=True)
    return [Fact("portfolio.low_confidence_orders", "Lowest evidence-coverage orders", _order_summary_rows(top), "portfolio")]


def _compute_model_health(decisions: pd.DataFrame, metrics: Mapping[str, Any] | None) -> list[Fact]:
    facts = []
    metrics = metrics or {}
    threshold = _finite_float(metrics.get("threshold")) if isinstance(metrics, Mapping) else None
    if threshold is not None:
        facts.append(Fact("portfolio.threshold", "Current decision threshold", threshold, "portfolio"))
    architecture = metrics.get("architecture", {}) if isinstance(metrics, Mapping) else {}
    if isinstance(architecture, Mapping) and architecture.get("risk_model"):
        facts.append(Fact("portfolio.risk_model", "Risk model in production", _truncate_text(architecture.get("risk_model"), 64), "portfolio"))
    facts.append(Fact("portfolio.scored_order_count", "Total scored orders in this run", int(len(decisions)), "portfolio"))
    return facts


def _compute_planner_focus_today(decisions: pd.DataFrame, metrics: Mapping[str, Any] | None) -> list[Fact]:
    facts = list(_compute_decision_mix(decisions, metrics))
    facts.extend(_compute_largest_penalty_exposure(decisions, metrics))
    facts.extend(_compute_capacity_conflicts(decisions, metrics))
    # De-duplicate ids while preserving order (a question can reuse other computations).
    seen: set[str] = set()
    unique: list[Fact] = []
    for fact in facts:
        if fact.id in seen:
            continue
        seen.add(fact.id)
        unique.append(fact)
    return unique[:MAX_FACTS_PER_PACKET]


_PORTFOLIO_COMPUTATIONS: dict[str, Callable[[pd.DataFrame, Mapping[str, Any] | None], list[Fact]]] = {
    "highest_risk_orders": _compute_highest_risk_orders,
    "decision_mix": _compute_decision_mix,
    "largest_penalty_exposure": _compute_largest_penalty_exposure,
    "hotspots": _compute_hotspots,
    "dominant_causes": _compute_dominant_causes,
    "capacity_conflicts": _compute_capacity_conflicts,
    "low_confidence_orders": _compute_low_confidence_orders,
    "model_health": _compute_model_health,
    "planner_focus_today": _compute_planner_focus_today,
}


def build_portfolio_evidence_packet(
    question_id: str,
    decisions: pd.DataFrame,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> EvidencePacket:
    """Compute one fixed portfolio question's facts deterministically.

    ``question_id`` must be a key of ``PORTFOLIO_QUESTIONS``. There is no
    arbitrary code/SQL execution path here: each question maps to one
    hand-written, reviewable aggregation function.
    """

    if question_id not in PORTFOLIO_QUESTIONS:
        raise ValueError(f"Unknown portfolio question_id: {question_id!r}")
    compute = _PORTFOLIO_COMPUTATIONS[question_id]
    facts = compute(decisions, metrics)[:MAX_FACTS_PER_PACKET]
    return EvidencePacket(
        scope="portfolio",
        subject=question_id,
        generated_at_utc=datetime.now(UTC).isoformat(),
        facts=tuple(facts),
        persisted_decision_status=None,
        persisted_recommended_action=None,
    )
