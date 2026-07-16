"""Deterministic, audit-friendly narratives for scored orders."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


def _display(value: Any, fallback: str = "not available") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def parse_top_factors(value: Any, *, limit: int = 3) -> list[str]:
    """Normalize top-factor JSON from common pipeline output shapes."""

    if limit < 1:
        raise ValueError("limit must be positive")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in text.split(",") if part.strip()]

    factors: list[str] = []
    if isinstance(parsed, Mapping):
        parsed = sorted(parsed.items(), key=lambda item: abs(float(item[1])), reverse=True)
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    for item in parsed:
        if isinstance(item, Mapping):
            name = item.get("factor") or item.get("feature") or item.get("name")
        elif isinstance(item, (list, tuple)) and item:
            name = item[0]
        else:
            name = item
        if name is not None and str(name).strip():
            factors.append(str(name).strip().replace("_", " "))
        if len(factors) == limit:
            break
    return factors


def _pathway_text(value: Any) -> str:
    """Render the causal pathway compactly: a route arrow-chain when available."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "no causal pathway"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, Mapping) and "route" in parsed:
            route = parsed.get("route") or []
            if route:
                return " -> ".join(str(node) for node in route)
            return "no active evidence route"
    return _display(value, "no causal pathway")


def _affected_sku_text(order: Mapping[str, Any]) -> str:
    raw = order.get("affected_skus_json")
    if not raw or (isinstance(raw, float) and math.isnan(raw)):
        return "no affected-SKU evidence"
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return "no affected-SKU evidence"
    if not parsed:
        return "no affected-SKU evidence"
    skus = ", ".join(
        str(item.get("sku_id", "?")) for item in parsed[:3] if isinstance(item, Mapping)
    )
    return f"affected SKUs: {skus}" if skus else "no affected-SKU evidence"


def _resource_status_text(order: Mapping[str, Any]) -> str:
    status = _display(order.get("decision_status"), "MONITOR")
    resource_type = order.get("resource_type")
    resource_id = order.get("resource_id")
    contested_with = order.get("contested_with")
    if status.upper() == "CONTESTED" and contested_with:
        resource_label = resource_type or "shared"
        return f"resource status: contested for {resource_label} capacity with {contested_with}"
    if resource_type:
        return f"resource status: {resource_type} {resource_id or 'n/a'} ({status.lower()})"
    return f"resource status: {status.lower()}"


def _mechanism_text(order: Mapping[str, Any]) -> str:
    """Optional 'dominant mechanism' clause when mechanism probabilities are present."""
    late = order.get("late_delivery_probability")
    in_full = order.get("in_full_failure_probability")
    try:
        late_value = float(late)
        in_full_value = float(in_full)
    except (TypeError, ValueError):
        return ""
    if math.isnan(late_value) or math.isnan(in_full_value):
        return ""
    if late_value >= in_full_value:
        label, value = "late delivery", late_value
    else:
        label, value = "in-full failure", in_full_value
    return f"dominant mechanism: {label} ({value:.0%})"


def _top_intervention_text(order: Mapping[str, Any]) -> str:
    """Optional 'highest-potential structural intervention' clause, when available.

    Only ever describes a *fixed-structure scenario*, explicitly labeled as
    such -- never a proven treatment effect -- and never affects the score or
    decision it is attached to.
    """
    raw = order.get("intervention_scenarios_json")
    if not raw or (isinstance(raw, float) and math.isnan(raw)):
        return ""
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ""
    if not isinstance(parsed, list) or not parsed:
        return ""
    single_node = [
        item
        for item in parsed
        if isinstance(item, Mapping) and item.get("type") == "single_node_mitigation"
    ]
    if not single_node:
        return ""
    best = max(single_node, key=lambda item: item.get("absolute_risk_reduction", 0.0))
    nodes = best.get("intervened_nodes") or []
    reduction = best.get("absolute_risk_reduction")
    if not nodes or reduction is None or float(reduction) <= 0:
        # Only narrate a scenario that actually reduces the modeled posterior;
        # a non-positive "best" (every active node screened off or, at a
        # collider node, structurally increasing risk) has nothing genuinely
        # mitigating to report, so stay silent rather than mislabel it.
        return ""
    node_text = str(nodes[0]).replace("_", " ").title()
    return (
        f"top structural scenario: mitigating {node_text} could cut the modeled "
        f"posterior by {float(reduction):.0%} (fixed-structure scenario analysis, "
        "not a proven treatment effect)"
    )


def order_narrative(order: Mapping[str, Any]) -> str:
    """One-line summary: risk -> evidence -> pathway -> SKUs -> action -> resource status."""

    order_id = _display(order.get("order_id"), "unknown")
    risk_value = order.get("combined_risk_score", 0.0)
    try:
        risk = min(max(float(risk_value), 0.0), 1.0)
    except (TypeError, ValueError):
        risk = 0.0
    cause = _display(order.get("primary_cause"), "unclassified").replace("_", " ").lower()
    factors = parse_top_factors(order.get("top_factors_json"))
    factor_text = ", ".join(factors) if factors else "no ranked factors"
    pathway = _pathway_text(order.get("causal_pathway"))
    affected_text = _affected_sku_text(order)
    action = _display(
        order.get("recommended_action"),
        "review the exception and confirm a recovery plan",
    )
    status = _display(order.get("decision_status"), "MONITOR")
    resource_text = _resource_status_text(order)
    optional_clauses = "; ".join(
        clause for clause in (_mechanism_text(order), _top_intervention_text(order)) if clause
    )
    tail = f"; {optional_clauses}" if optional_clauses else ""
    return (
        f"Order {order_id} has {risk:.0%} OTIF risk, led by {cause}; "
        f"top factors: {factor_text}; pathway: {pathway}; "
        f"{affected_text}; "
        f"{status.lower()} action: {action}; {resource_text}{tail}."
    )
