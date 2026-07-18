"""Deterministic, structured Copilot answers -- no network, no API key.

This module produces the same structured response schema as the live OpenAI
path (see ``llm_copilot.py``), built entirely from an ``EvidencePacket`` with
plain, reviewable Python -- no model call, no randomness. It is the demo's
"always works" mode, and it is also what ``llm_copilot`` falls back to
whenever the live path is unavailable, times out, or fails validation.

Every list item that makes a factual claim carries at least one citation
into the packet's fact IDs, so the same citation-badge UI that resolves live
answers also resolves fallback answers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from otif_risk.copilot_context import EvidencePacket

DISCLAIMER = "Explanation support only; production decision unchanged."

#: Fixed catalog of supported order-scoped questions (the Streamlit "chips").
ORDER_QUESTIONS: dict[str, str] = {
    "simple_explanation": "Explain this order simply",
    "why_flagged": "Why was it flagged?",
    "sku_impact": "Which SKU is affected?",
    "contention": "Why is the action contested?",
    "draft_supplier_escalation": "Draft a supplier escalation",
}


def _cite(text: str, citations: list[str]) -> dict[str, Any]:
    return {"text": text, "citations": [c for c in citations if c]}


def _fact_value(packet: EvidencePacket, fact_id: str, default: Any = None) -> Any:
    fact = packet.get(fact_id)
    return fact.value if fact is not None else default


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "unknown"


def _skeleton(packet: EvidencePacket) -> dict[str, Any]:
    return {
        "headline": "",
        "what_happened": [],
        "why_flagged": [],
        "affected_items": [],
        "recommended_next_step": {
            "text": "",
            "citations": [],
            "preserves_persisted_decision": True,
        },
        "uncertainties": [],
        "draft_message": None,
        "disclaimer": DISCLAIMER,
    }


def _base_facts(packet: EvidencePacket) -> dict[str, Any]:
    order_id = packet.subject
    status = packet.persisted_decision_status or "MONITOR"
    action = packet.persisted_recommended_action or "review the exception"
    cause = _fact_value(packet, "decision.primary_cause", "an unclassified factor")
    combined_risk = _fact_value(packet, "risk.combined")
    return {
        "order_id": order_id,
        "status": status,
        "action": action,
        "cause": str(cause).replace("_", " ").title() if cause else "Unclassified",
        "combined_risk": combined_risk,
    }


def _why_flagged_items(packet: EvidencePacket) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    combined = packet.get("risk.combined")
    if combined is not None:
        items.append(
            _cite(
                f"Combined OTIF risk is {_pct(combined.value)}, driven by XGBoost association "
                "and Bayesian mechanism evidence.",
                ["risk.combined", "risk.xgboost", "risk.bayesian"],
            )
        )
    shap_ids = [f"shap.{i}" for i in range(1, 6) if packet.get(f"shap.{i}") is not None]
    if shap_ids:
        factor_labels = []
        for fid in shap_ids:
            value = packet.get(fid).value
            name = value.get("factor") if isinstance(value, Mapping) else value
            if name:
                factor_labels.append(str(name))
        items.append(
            _cite(
                "Top contributing factors (association, not causation): "
                + ", ".join(factor_labels)
                + ".",
                [*shap_ids, "shap.limitation"],
            )
        )
    route_fact = packet.get("mechanism.route")
    if route_fact is not None and route_fact.value:
        items.append(
            _cite(
                "The Bayesian mechanism route is " + " -> ".join(route_fact.value) + ".",
                ["mechanism.route", "mechanism.limitation"],
            )
        )
    return items


def _affected_items(packet: EvidencePacket) -> list[dict[str, Any]]:
    sku_facts = [fact for fact in packet.facts if fact.id.startswith("sku.")]
    items = []
    for fact in sku_facts:
        value = fact.value if isinstance(fact.value, Mapping) else {}
        strength = value.get("evidence_strength")
        strength_text = f" (evidence strength {strength:.0%})" if isinstance(strength, (int, float)) else ""
        items.append(_cite(f"{fact.id.split('.', 1)[1]}{strength_text}", [fact.id]))
    if not items:
        items.append(
            _cite("No affected-SKU evidence is present on this order.", ["identity.order_id"])
        )
    return items


def _uncertainties(packet: EvidencePacket) -> list[dict[str, Any]]:
    items = []
    coverage = packet.get("risk.evidence_coverage")
    if coverage is not None and coverage.value is not None:
        items.append(
            _cite(
                f"Evidence coverage is {_pct(coverage.value)}; some lifecycle stages are unobserved.",
                ["risk.evidence_coverage"],
            )
        )
    missing = packet.get("events.missing_stage_count")
    if missing is not None and missing.value:
        items.append(
            _cite(
                f"{int(missing.value)} lifecycle event stage(s) are missing for this order.",
                ["events.missing_stage_count"],
            )
        )
    confidence = packet.get("risk.confidence")
    if confidence is not None and confidence.value:
        items.append(_cite(f"Causal-confidence band is {confidence.value}.", ["risk.confidence"]))
    if not items:
        items.append(
            _cite("No material evidence gaps were flagged for this order.", ["identity.order_id"])
        )
    return items


def _simple_explanation(packet: EvidencePacket) -> dict[str, Any]:
    base = _base_facts(packet)
    response = _skeleton(packet)
    response["headline"] = (
        f"Order {base['order_id']} is {base['status'].lower()} at "
        f"{_pct(base['combined_risk'])} OTIF risk, led by {base['cause'].lower()}."
    )
    response["what_happened"] = [
        f"The order's combined risk score is {_pct(base['combined_risk'])} "
        f"(fact risk.combined), primarily attributed to {base['cause'].lower()} "
        "(fact decision.primary_cause)."
    ]
    response["why_flagged"] = _why_flagged_items(packet)
    response["affected_items"] = _affected_items(packet)
    response["recommended_next_step"] = {
        "text": f"{base['action']} (status: {base['status']}).",
        "citations": ["decision.action", "decision.status"],
        "preserves_persisted_decision": True,
    }
    response["uncertainties"] = _uncertainties(packet)
    return response


def _why_was_it_flagged(packet: EvidencePacket) -> dict[str, Any]:
    response = _simple_explanation(packet)
    base = _base_facts(packet)
    response["headline"] = f"Order {base['order_id']} was flagged mainly due to {base['cause'].lower()}."
    return response


def _sku_impact(packet: EvidencePacket) -> dict[str, Any]:
    base = _base_facts(packet)
    response = _skeleton(packet)
    sku_items = _affected_items(packet)
    sku_ids = [fact.id for fact in packet.facts if fact.id.startswith("sku.")]
    response["headline"] = (
        f"{len(sku_ids)} SKU line(s) are flagged as affected on order {base['order_id']}."
        if sku_ids
        else f"No SKU lines are flagged as affected on order {base['order_id']}."
    )
    response["what_happened"] = [item["text"] for item in sku_items]
    response["affected_items"] = sku_items
    response["why_flagged"] = _why_flagged_items(packet)[:1]
    response["recommended_next_step"] = {
        "text": f"{base['action']} (status: {base['status']}).",
        "citations": ["decision.action", "decision.status"],
        "preserves_persisted_decision": True,
    }
    response["uncertainties"] = _uncertainties(packet)
    return response


def _contention(packet: EvidencePacket) -> dict[str, Any]:
    base = _base_facts(packet)
    response = _skeleton(packet)
    resource = packet.get("operational.resource_pool")
    contested_with = packet.get("operational.contested_with")
    if base["status"] != "CONTESTED":
        response["headline"] = f"Order {base['order_id']} is not currently contested (status: {base['status']})."
        response["what_happened"] = [f"Persisted decision status is {base['status']}, not CONTESTED."]
        response["why_flagged"] = [_cite(f"Decision status is {base['status']}.", ["decision.status"])]
    else:
        pool = resource.value if resource is not None else {}
        pool_text = (
            f"{pool.get('resource_type', 'a shared')} pool (id {pool.get('resource_id', 'n/a')})"
            if isinstance(pool, Mapping)
            else "a shared resource pool"
        )
        competitors = contested_with.value if contested_with is not None else []
        response["headline"] = f"Order {base['order_id']} is contested for {pool_text}."
        response["what_happened"] = [
            f"This order competes for {pool_text} with: " + ", ".join(competitors) + "."
            if competitors
            else f"This order competes for {pool_text}."
        ]
        cites = ["decision.status"]
        if resource is not None:
            cites.append("operational.resource_pool")
        if contested_with is not None:
            cites.append("operational.contested_with")
        response["why_flagged"] = [
            _cite(
                f"Status is CONTESTED because available capacity in {pool_text} is fully claimed "
                "by higher-priority orders.",
                cites,
            )
        ]
    response["affected_items"] = _affected_items(packet)
    response["recommended_next_step"] = {
        "text": f"{base['action']} (status: {base['status']}).",
        "citations": ["decision.action", "decision.status"],
        "preserves_persisted_decision": True,
    }
    response["uncertainties"] = _uncertainties(packet)
    return response


def _draft_supplier_escalation(packet: EvidencePacket) -> dict[str, Any]:
    base = _base_facts(packet)
    response = _skeleton(packet)
    vendor = packet.get("operational.vendor")
    vendor_text = vendor.value if vendor is not None else "the responsible supplier"
    response["headline"] = f"Draft supplier escalation prepared for order {base['order_id']}."
    response["what_happened"] = [
        f"Order {base['order_id']} carries {_pct(base['combined_risk'])} OTIF risk, "
        f"attributed to {base['cause'].lower()}."
    ]
    response["why_flagged"] = _why_flagged_items(packet)
    response["affected_items"] = _affected_items(packet)
    response["recommended_next_step"] = {
        "text": f"{base['action']} (status: {base['status']}).",
        "citations": ["decision.action", "decision.status"],
        "preserves_persisted_decision": True,
    }
    response["uncertainties"] = _uncertainties(packet)
    sku_ids = ", ".join(fact.id.split(".", 1)[1] for fact in packet.facts if fact.id.startswith("sku."))
    response["draft_message"] = (
        f"Subject: Escalation -- Order {base['order_id']} at risk\n\n"
        f"Hello {vendor_text} team,\n\n"
        f"Order {base['order_id']} is currently {base['status']} with an estimated OTIF risk of "
        f"{_pct(base['combined_risk'])}, primarily attributed to {base['cause'].lower()}."
        + (f" Affected SKU(s): {sku_ids}." if sku_ids else "")
        + f"\n\nRequested action: {base['action']}.\n\n"
        "Please confirm revised supply/ship commitments at your earliest opportunity.\n\n"
        "This message is a draft for planner review only; it has not been sent automatically, "
        "and it does not change the persisted decision or status above."
    )
    return response


_ORDER_HANDLERS = {
    "simple_explanation": _simple_explanation,
    "why_flagged": _why_was_it_flagged,
    "sku_impact": _sku_impact,
    "contention": _contention,
    "draft_supplier_escalation": _draft_supplier_escalation,
}


def order_fallback_response(question_id: str, packet: EvidencePacket) -> dict[str, Any]:
    """Build the deterministic structured response for one supported order question."""

    if packet.scope != "order":
        raise ValueError("order_fallback_response requires an order-scoped EvidencePacket")
    handler = _ORDER_HANDLERS.get(question_id)
    if handler is None:
        raise ValueError(f"Unsupported order question_id: {question_id!r}")
    return handler(packet)


def portfolio_fallback_response(question_id: str, packet: EvidencePacket) -> dict[str, Any]:
    """Build the deterministic structured response for one fixed portfolio question."""

    if packet.scope != "portfolio":
        raise ValueError("portfolio_fallback_response requires a portfolio-scoped EvidencePacket")
    response = _skeleton(packet)
    fact_ids = [fact.id for fact in packet.facts]
    if not fact_ids:
        response["headline"] = "No portfolio facts are available for this question."
        response["recommended_next_step"]["text"] = "No data-backed recommendation is available."
        response["recommended_next_step"]["citations"] = []
        return response
    response["headline"] = _portfolio_headline(question_id, packet)
    response["what_happened"] = [
        _describe_fact(fact) for fact in packet.facts
    ]
    response["why_flagged"] = [_cite(_describe_fact(fact), [fact.id]) for fact in packet.facts]
    response["recommended_next_step"] = {
        "text": _portfolio_recommendation(question_id, packet),
        "citations": fact_ids[: min(3, len(fact_ids))],
        "preserves_persisted_decision": True,
    }
    response["uncertainties"] = [
        _cite(
            "Portfolio facts summarize the currently loaded scored-orders snapshot only; they are "
            "not a live production feed.",
            [fact_ids[0]],
        )
    ]
    return response


def _describe_fact(fact: Any) -> str:
    return f"{fact.label}: {fact.value}"


def _portfolio_headline(question_id: str, packet: EvidencePacket) -> str:
    headlines = {
        "highest_risk_orders": "Highest-risk orders in the current scored run.",
        "decision_mix": "Current decision-status mix across the portfolio.",
        "largest_penalty_exposure": "Orders and totals with the largest penalty exposure.",
        "hotspots": "Vendor/DC/lane/customer concentration hotspots.",
        "dominant_causes": "Most common root-cause categories in this run.",
        "capacity_conflicts": "Resource pools with the most contested orders.",
        "low_confidence_orders": "Orders with the lowest evidence coverage.",
        "model_health": "Current model, threshold, and run health.",
        "planner_focus_today": "Where planners should focus today.",
    }
    return headlines.get(question_id, f"Portfolio summary for {question_id}.")


def _portfolio_recommendation(question_id: str, packet: EvidencePacket) -> str:
    if question_id == "planner_focus_today":
        return (
            "Prioritize CONTESTED orders and the resource pools with the most contested exposure "
            "first, then the largest penalty-exposure RECOMMENDED orders."
        )
    if question_id == "capacity_conflicts":
        return "Review the top contested resource pool for capacity reallocation options."
    if question_id == "low_confidence_orders":
        return "Prioritize gathering missing lifecycle events for the lowest-coverage orders before acting."
    return "Use these cited facts to prioritize planner attention; no automatic action is taken."
