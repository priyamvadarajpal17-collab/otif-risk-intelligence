"""Validated append-only planner feedback."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FEEDBACK_ACTIONS = ("ACCEPT", "REJECT", "OVERRIDE")
FEEDBACK_COLUMNS = (
    "timestamp_utc",
    "order_id",
    "feedback_action",
    "original_status",
    "original_recommendation",
    "override_recommendation",
    "reason",
    "actor",
)


def validate_feedback(
    *,
    order_id: Any,
    feedback_action: str,
    reason: str = "",
    override_recommendation: str = "",
) -> None:
    """Validate user-entered feedback before it reaches the audit log."""

    if not str(order_id).strip():
        raise ValueError("order_id is required")
    normalized_action = str(feedback_action).strip().upper()
    if normalized_action not in FEEDBACK_ACTIONS:
        raise ValueError(f"feedback_action must be one of {', '.join(FEEDBACK_ACTIONS)}")
    if normalized_action in {"REJECT", "OVERRIDE"} and not reason.strip():
        raise ValueError("a reason is required for reject or override feedback")
    if normalized_action == "OVERRIDE" and not override_recommendation.strip():
        raise ValueError("override_recommendation is required for override feedback")
    if normalized_action != "OVERRIDE" and override_recommendation.strip():
        raise ValueError("override_recommendation is only valid for override feedback")


def append_feedback(
    path: str | Path,
    *,
    order_id: Any,
    feedback_action: str,
    original_status: str = "",
    original_recommendation: str = "",
    override_recommendation: str = "",
    reason: str = "",
    actor: str = "prototype-user",
    timestamp: datetime | None = None,
) -> dict[str, str]:
    """Append one feedback event without rewriting existing entries."""

    validate_feedback(
        order_id=order_id,
        feedback_action=feedback_action,
        reason=reason,
        override_recommendation=override_recommendation,
    )
    target = Path(path)
    if target.exists() and target.is_dir():
        raise ValueError("feedback path must point to a CSV file")
    target.parent.mkdir(parents=True, exist_ok=True)
    now = timestamp or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    row = {
        "timestamp_utc": now.astimezone(UTC).isoformat(),
        "order_id": str(order_id).strip(),
        "feedback_action": feedback_action.strip().upper(),
        "original_status": original_status.strip(),
        "original_recommendation": original_recommendation.strip(),
        "override_recommendation": override_recommendation.strip(),
        "reason": reason.strip(),
        "actor": actor.strip() or "prototype-user",
    }
    needs_header = not target.exists() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_COLUMNS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    return row
