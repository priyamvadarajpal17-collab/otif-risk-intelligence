"""Append-only Copilot request/response audit log.

Each line of ``copilot_audit.jsonl`` is one JSON object describing a single
Copilot request: who/what was asked, which provider answered, whether
validation passed, and observability metadata (latency, token counts). It
never contains the API key, the full response text, or any chain-of-thought
-- only cited fact IDs and validation status, consistent with the plan's
"no hidden reasoning" requirement.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_FILENAME = "copilot_audit.jsonl"
PROMPT_VERSION = "copilot-v1"


@dataclass
class AuditRecord:
    request_id: str
    timestamp_utc: str
    scope: str  # "order" | "portfolio"
    order_id: str | None
    query_type: str
    provider: str  # "openai" | "fallback"
    model: str | None
    mode_configured: str
    mode_used: str  # "live" | "fallback"
    evidence_hash: str
    prompt_version: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    validation_status: str  # "passed" | "failed"
    fallback_reason: str | None
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_request_id() -> str:
    return uuid.uuid4().hex


def build_audit_record(
    *,
    scope: str,
    order_id: str | None,
    query_type: str,
    provider: str,
    model: str | None,
    mode_configured: str,
    mode_used: str,
    evidence_hash: str,
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
    validation_status: str,
    fallback_reason: str | None,
    citations: list[str],
    request_id: str | None = None,
) -> AuditRecord:
    return AuditRecord(
        request_id=request_id or new_request_id(),
        timestamp_utc=datetime.now(UTC).isoformat(),
        scope=scope,
        order_id=order_id,
        query_type=query_type,
        provider=provider,
        model=model,
        mode_configured=mode_configured,
        mode_used=mode_used,
        evidence_hash=evidence_hash,
        prompt_version=PROMPT_VERSION,
        latency_ms=round(latency_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        validation_status=validation_status,
        fallback_reason=fallback_reason,
        citations=list(citations),
    )


def append_audit_record(path: str | Path, record: AuditRecord) -> None:
    """Append one JSON line. Creates parent directories and the file if needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), sort_keys=True, default=str)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_audit_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Best-effort read of existing audit records for the health card / evaluation."""

    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        records = records[-limit:]
    return records


def default_audit_path(run_directory: str | Path) -> Path:
    return Path(run_directory) / AUDIT_FILENAME
