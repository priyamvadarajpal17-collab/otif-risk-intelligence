"""Citation and hallucination guard for Copilot responses (live or fallback).

This is the last line of defense before a response reaches the Streamlit UI
or the audit log: it enforces the structured response shape, requires every
factual claim to cite a fact ID that actually exists in the evidence packet,
requires the persisted decision to be preserved, rejects non-finite/oversized
output, and strips unsupported HTML/URLs.

It is not a guarantee that an upstream LLM can never hallucinate -- it makes
unsupported claims visible (as a validation failure) and ensures they are
never presented to a planner as grounded output; the caller (``llm_copilot``)
falls back to the deterministic response on any failure here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from otif_risk.copilot_context import EvidencePacket

REQUIRED_KEYS = (
    "headline",
    "what_happened",
    "why_flagged",
    "affected_items",
    "recommended_next_step",
    "uncertainties",
    "draft_message",
    "disclaimer",
)
DECISION_STATUS_WORDS = ("RECOMMENDED", "CONTESTED", "MONITOR")

#: Deterministic, fixed size limits -- never model-configurable.
MAX_RESPONSE_JSON_BYTES = 24_000
MAX_STRING_FIELD_LEN = 2_000
MAX_LIST_LEN = 20

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    sanitized_response: dict[str, Any] | None = None


def _strip_unsafe_text(value: str) -> str:
    cleaned = _HTML_TAG_RE.sub("", value)
    cleaned = _URL_RE.sub("[link removed]", cleaned)
    if len(cleaned) > MAX_STRING_FIELD_LEN:
        cleaned = cleaned[: MAX_STRING_FIELD_LEN - 1].rstrip() + "\u2026"
    return cleaned


def _sanitize(value: Any) -> Any:
    """Recursively strip HTML/URLs from every string; leave structure intact."""
    if isinstance(value, str):
        return _strip_unsafe_text(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value[:MAX_LIST_LEN]]
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _validate_cited_item(item: Any, allowed_ids: frozenset[str], errors: list[str], field_name: str) -> None:
    if not isinstance(item, dict):
        errors.append(f"{field_name} entries must be objects with text/citations")
        return
    text = item.get("text")
    citations = item.get("citations")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"{field_name} entry missing non-empty text")
        return
    if not isinstance(citations, list) or not citations:
        errors.append(f"{field_name} entry missing required citations: {text[:60]!r}")
        return
    unknown = [c for c in citations if c not in allowed_ids]
    if unknown:
        errors.append(f"{field_name} entry cites unknown fact id(s): {unknown}")


def validate_response(response: Any, packet: EvidencePacket) -> ValidationResult:
    """Validate + sanitize a structured Copilot response against its evidence packet.

    Returns a ``ValidationResult``. ``sanitized_response`` is populated (HTML/
    URL-stripped, size-capped) whenever the shape could be parsed, even if
    validation ultimately fails, so callers/audits can inspect what was
    rejected -- but only a ``passed=True`` result should ever be shown to a
    planner as a grounded answer.
    """

    errors: list[str] = []
    if not isinstance(response, dict):
        return ValidationResult(passed=False, errors=["response is not a JSON object"], sanitized_response=None)

    missing = [key for key in REQUIRED_KEYS if key not in response]
    if missing:
        errors.append(f"response is missing required keys: {missing}")
        return ValidationResult(passed=False, errors=errors, sanitized_response=None)

    try:
        import json

        size = len(json.dumps(response, default=str))
    except (TypeError, ValueError):
        size = MAX_RESPONSE_JSON_BYTES + 1
    if size > MAX_RESPONSE_JSON_BYTES:
        errors.append(f"response exceeds size limit ({size} > {MAX_RESPONSE_JSON_BYTES} bytes)")

    if not _all_finite(response):
        errors.append("response contains non-finite numeric value(s)")

    sanitized = _sanitize(response)

    allowed_ids = packet.fact_ids()

    if not isinstance(sanitized.get("headline"), str) or not sanitized["headline"].strip():
        errors.append("headline must be a non-empty string")

    if not isinstance(sanitized.get("what_happened"), list):
        errors.append("what_happened must be a list")

    for field_name in ("why_flagged", "affected_items", "uncertainties"):
        items = sanitized.get(field_name)
        if not isinstance(items, list):
            errors.append(f"{field_name} must be a list")
            continue
        for item in items:
            _validate_cited_item(item, allowed_ids, errors, field_name)

    next_step = sanitized.get("recommended_next_step")
    if not isinstance(next_step, dict):
        errors.append("recommended_next_step must be an object")
    else:
        if next_step.get("preserves_persisted_decision") is not True:
            errors.append("recommended_next_step.preserves_persisted_decision must be true")
        _validate_cited_item(next_step, allowed_ids, errors, "recommended_next_step")

        if packet.scope == "order" and packet.persisted_decision_status:
            # Only the next-step directive is checked (not the free-form headline, which may
            # legitimately discuss a *different* status in passing, e.g. "is not contested").
            haystack = str(next_step.get("text", "")).upper()
            persisted = packet.persisted_decision_status.upper()
            mentioned = {word for word in DECISION_STATUS_WORDS if word in haystack}
            off_status = mentioned - {persisted} if persisted in haystack else mentioned
            if off_status:
                errors.append(
                    "response mentions a decision status inconsistent with the persisted status "
                    f"({packet.persisted_decision_status}): {sorted(off_status)}"
                )

    draft_message = sanitized.get("draft_message")
    if draft_message is not None and not isinstance(draft_message, str):
        errors.append("draft_message must be a string or null")

    disclaimer = sanitized.get("disclaimer")
    if not isinstance(disclaimer, str) or not disclaimer.strip():
        errors.append("disclaimer must be a non-empty string")

    return ValidationResult(passed=not errors, errors=errors, sanitized_response=sanitized)
