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


def order_narrative(order: Mapping[str, Any]) -> str:
    """Create a stable one-line summary with risk, evidence, pathway, and action."""

    order_id = _display(order.get("order_id"), "unknown")
    risk_value = order.get("combined_risk_score", 0.0)
    try:
        risk = min(max(float(risk_value), 0.0), 1.0)
    except (TypeError, ValueError):
        risk = 0.0
    cause = _display(order.get("primary_cause"), "unclassified").replace("_", " ").lower()
    factors = parse_top_factors(order.get("top_factors_json"))
    factor_text = ", ".join(factors) if factors else "no ranked factors"
    pathway = _display(order.get("causal_pathway"), "no causal pathway")
    action = _display(
        order.get("recommended_action"),
        "review the exception and confirm a recovery plan",
    )
    status = _display(order.get("decision_status"), "MONITOR")
    return (
        f"Order {order_id} has {risk:.0%} OTIF risk, led by {cause}; "
        f"top factors: {factor_text}; pathway: {pathway}; "
        f"{status.lower()} action: {action}."
    )
