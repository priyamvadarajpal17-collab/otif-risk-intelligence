"""Representative-order Copilot evaluation harness.

Because no human-labeled explanation dataset exists for this prototype, this
module evaluates the Copilot against a deterministic set of *representative*
orders (high-risk inventory miss, timing-driven miss, multi-cause, contested
action, low-confidence, safe/monitor, unknown-cause) rather than claiming any
BLEU/factuality/human-preference score. It measures only what can be measured
mechanically: citation validity, decision-status/action preservation,
required-section completeness, fallback success, and latency/token usage --
and, only when an API key is actually configured, live-vs-fallback agreement
on key facts from one live smoke sample.

Run as a script: ``python -m otif_risk.copilot_evaluation``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from otif_risk.app import load_run_artifacts
from otif_risk.contracts import CAUSE_CATEGORIES
from otif_risk.copilot_context import EvidencePacket, build_order_evidence_packet
from otif_risk.copilot_fallback import ORDER_QUESTIONS
from otif_risk.copilot_validation import validate_response
from otif_risk.llm_copilot import get_order_copilot_response, is_live_configured

DEFAULT_OUTPUT_PATH = "artifacts/copilot_evaluation.json"
REQUIRED_SECTIONS = (
    "headline",
    "what_happened",
    "why_flagged",
    "affected_items",
    "recommended_next_step",
    "uncertainties",
    "disclaimer",
)


def _safe_json(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _active_evidence_count(row: pd.Series) -> int:
    pathway = _safe_json(row.get("causal_pathway"))
    if isinstance(pathway, dict):
        count = pathway.get("active_evidence_count")
        if isinstance(count, (int, float)):
            return int(count)
        active = pathway.get("active_evidence")
        if isinstance(active, list):
            return len(active)
    return 0


def select_representative_orders(decisions: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    """Deterministically pick one order per representative category.

    Returns a mapping category -> order row (as a dict) or ``None`` if this
    dataset has no matching order for that category (reported honestly rather
    than guessed).
    """

    selected: dict[str, dict[str, Any] | None] = {}

    def pick(frame: pd.DataFrame, sort_col: str | None, ascending: bool = False) -> dict[str, Any] | None:
        if frame.empty:
            return None
        if sort_col and sort_col in frame.columns:
            frame = frame.sort_values(sort_col, ascending=ascending)
        return frame.iloc[0].to_dict()

    inventory = decisions[decisions.get("primary_cause") == "INVENTORY_SHORTAGE"]
    selected["high_risk_inventory_miss"] = pick(inventory, "combined_risk_score")

    if {"late_delivery_probability", "in_full_failure_probability"}.issubset(decisions.columns):
        timing = decisions[
            decisions["late_delivery_probability"] > decisions["in_full_failure_probability"]
        ]
    else:
        timing = decisions.iloc[0:0]
    selected["timing_driven_miss"] = pick(timing, "combined_risk_score")

    multi_cause_mask = decisions.apply(_active_evidence_count, axis=1) >= 3
    selected["multi_cause_order"] = pick(decisions[multi_cause_mask], "combined_risk_score")

    contested = decisions[decisions.get("decision_status") == "CONTESTED"]
    selected["contested_action"] = pick(contested, "priority_score")

    if "causal_confidence" in decisions.columns:
        low_confidence = decisions[decisions["causal_confidence"] == "LOW"]
        if low_confidence.empty and "evidence_coverage" in decisions.columns:
            low_confidence = decisions
            selected["low_confidence_order"] = pick(low_confidence, "evidence_coverage", ascending=True)
        else:
            selected["low_confidence_order"] = pick(low_confidence, "evidence_coverage", ascending=True)
    else:
        selected["low_confidence_order"] = None

    monitor = decisions[decisions.get("decision_status") == "MONITOR"]
    selected["safe_monitor_order"] = pick(monitor, "combined_risk_score", ascending=True)

    if "primary_cause" in decisions.columns:
        unknown = decisions[~decisions["primary_cause"].isin(CAUSE_CATEGORIES)]
    else:
        unknown = decisions.iloc[0:0]
    selected["unknown_cause_order"] = pick(unknown, "combined_risk_score")

    return selected


def _evaluate_response(response: dict[str, Any], packet: EvidencePacket) -> dict[str, Any]:
    validation = validate_response(response, packet)
    completeness = all(
        section in response and response[section] not in (None, "", [])
        for section in REQUIRED_SECTIONS
    )
    preserves_decision = bool(
        isinstance(response.get("recommended_next_step"), dict)
        and response["recommended_next_step"].get("preserves_persisted_decision") is True
    )
    all_citations = []
    for field_name in ("why_flagged", "affected_items", "uncertainties"):
        for item in response.get(field_name, []) or []:
            if isinstance(item, dict):
                all_citations.extend(item.get("citations", []) or [])
    next_step = response.get("recommended_next_step") or {}
    all_citations.extend(next_step.get("citations", []) or [])
    allowed = packet.fact_ids()
    valid_citations = [c for c in all_citations if c in allowed]
    citation_validity_rate = (
        len(valid_citations) / len(all_citations) if all_citations else 1.0
    )
    return {
        "validation_passed": validation.passed,
        "validation_errors": validation.errors,
        "completeness": completeness,
        "preserves_persisted_decision": preserves_decision,
        "citation_validity_rate": citation_validity_rate,
        "citation_count": len(all_citations),
    }


def run_evaluation(
    decisions: pd.DataFrame,
    *,
    metrics: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    run_live_smoke: bool | None = None,
) -> dict[str, Any]:
    """Evaluate fallback (and optionally one live smoke) across representative orders."""

    representative = select_representative_orders(decisions)
    run_live_smoke = is_live_configured() if run_live_smoke is None else run_live_smoke

    per_order_results: dict[str, Any] = {}
    latencies_ms: list[float] = []
    fallback_successes = 0
    fallback_attempts = 0
    live_smoke_result: dict[str, Any] | None = None
    live_smoke_done = False

    for category, order in representative.items():
        if order is None:
            per_order_results[category] = {
                "available": False,
                "note": "No order in this dataset matches this representative category.",
            }
            continue
        order_id = str(order.get("order_id"))
        question_results: dict[str, Any] = {}
        for question_id in ORDER_QUESTIONS:
            mode = "fallback"
            if run_live_smoke and not live_smoke_done:
                mode = "auto"
            answer = get_order_copilot_response(
                order, question_id, metrics=metrics, manifest=manifest, mode=mode
            )
            evaluation = _evaluate_response(answer.response, answer.packet)
            latencies_ms.append(answer.latency_ms)
            fallback_attempts += 1
            if evaluation["validation_passed"]:
                fallback_successes += 1
            question_results[question_id] = {
                "mode_used": answer.mode_used,
                "provider": answer.provider,
                "model": answer.model,
                "latency_ms": answer.latency_ms,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "fallback_reason": answer.fallback_reason,
                **evaluation,
            }
            if run_live_smoke and not live_smoke_done and mode == "auto":
                live_smoke_done = True
                live_fallback_answer = get_order_copilot_response(
                    order, question_id, metrics=metrics, manifest=manifest, mode="fallback"
                )
                agree_status = (
                    answer.response.get("recommended_next_step", {}).get("preserves_persisted_decision")
                    == live_fallback_answer.response.get("recommended_next_step", {}).get(
                        "preserves_persisted_decision"
                    )
                )
                live_smoke_result = {
                    "order_id": order_id,
                    "question_id": question_id,
                    "live_mode_used": answer.mode_used,
                    "live_validation_passed": evaluation["validation_passed"],
                    "agrees_with_fallback_on_decision_preservation": agree_status,
                }
        per_order_results[category] = {"available": True, "order_id": order_id, "questions": question_results}

    summary = {
        "orders_evaluated": sum(1 for v in per_order_results.values() if v.get("available")),
        "categories_available": [k for k, v in per_order_results.items() if v.get("available")],
        "categories_unavailable": [k for k, v in per_order_results.items() if not v.get("available")],
        "fallback_success_rate": (
            fallback_successes / fallback_attempts if fallback_attempts else None
        ),
        "citation_validity_rate": (
            statistics.mean(
                r["citation_validity_rate"]
                for cat in per_order_results.values()
                if cat.get("available")
                for r in cat["questions"].values()
            )
            if any(v.get("available") for v in per_order_results.values())
            else None
        ),
        "decision_preservation_rate": (
            statistics.mean(
                1.0 if r["preserves_persisted_decision"] else 0.0
                for cat in per_order_results.values()
                if cat.get("available")
                for r in cat["questions"].values()
            )
            if any(v.get("available") for v in per_order_results.values())
            else None
        ),
        "completeness_rate": (
            statistics.mean(
                1.0 if r["completeness"] else 0.0
                for cat in per_order_results.values()
                if cat.get("available")
                for r in cat["questions"].values()
            )
            if any(v.get("available") for v in per_order_results.values())
            else None
        ),
        "latency_ms": {
            "count": len(latencies_ms),
            "median": statistics.median(latencies_ms) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
        },
        "live_smoke_exercised": run_live_smoke and live_smoke_done,
        "live_smoke_result": live_smoke_result,
    }
    return {
        "note": (
            "Deterministic mechanical checks only (citation validity, decision-status/action "
            "preservation, required-section completeness, fallback success, latency). No "
            "BLEU/factuality/human-preference claim is made -- no labeled explanation dataset "
            "exists for this prototype."
        ),
        "representative_categories": list(representative.keys()),
        "per_category": per_order_results,
        "summary": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        default=os.environ.get("OTIF_ARTIFACTS_DIR", "artifacts"),
        help="Directory containing run-* pipeline output (default: artifacts or $OTIF_ARTIFACTS_DIR).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the evaluation JSON (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--live-smoke",
        action="store_true",
        default=None,
        help="Force one live-mode smoke request even if not auto-detected (requires OPENAI_API_KEY).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.artifacts_root)
    run_directory, metrics, decisions = load_run_artifacts(str(root.resolve()))
    manifest_path = Path(run_directory) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None

    report = run_evaluation(decisions, metrics=metrics, manifest=manifest, run_live_smoke=args.live_smoke)
    report["run_directory"] = str(run_directory)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote Copilot evaluation report to {output_path}")
    print(json.dumps(report["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
