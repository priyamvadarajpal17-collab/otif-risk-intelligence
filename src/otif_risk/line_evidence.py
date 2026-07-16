"""Point-in-time line/SKU evidence beneath the single order-level decision.

The order-level risk score and decision stay singular (see ``decisions.py``),
but planners need to know *which lines/SKUs* are likely driving that risk.
This module derives that evidence from fields that are genuinely knowable at
(or before) an order's ``prediction_timestamp`` -- the initial ATP allocation
recorded at order capture, the observed inventory snapshot, SKU criticality,
and the order's own point-in-time vendor-exception signal -- never from the
simulator's retrospective ``shipped_qty``/``line_truth`` truth columns, which
are reserved for evaluation only.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PrototypeDataset

#: Minimum evidence_strength for a line to be flagged "likely affected".
LIKELY_AFFECTED_THRESHOLD = 0.20


def build_line_evidence(dataset: PrototypeDataset, features: pd.DataFrame) -> pd.DataFrame:
    """Build one row per order line using only as-of-capture-time-safe fields.

    ``allocated_qty``/``inventory_available_at_order`` are known at order
    capture (well before ``prediction_timestamp``); ``shipped_qty`` and the
    simulator's ``line_truth`` are intentionally excluded here -- they are
    the post-hoc truth this evidence is *evaluated against*, not an input.
    """
    required = {"order_id", "leading_signal_VENDOR_FAILURE", "leading_signal_INVENTORY_SHORTAGE"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features missing required columns: {sorted(missing)}")

    lines = dataset.order_lines.merge(
        dataset.skus[["sku_id", "criticality_tier", "base_unit_value"]],
        on="sku_id",
        how="left",
        validate="many_to_one",
    )
    order_signal = features.set_index("order_id")[
        ["leading_signal_VENDOR_FAILURE", "leading_signal_INVENTORY_SHORTAGE"]
    ]
    lines = lines.merge(order_signal, left_on="order_id", right_index=True, how="left")

    requested = lines["requested_qty"].astype(float).clip(lower=1)
    allocated = lines["allocated_qty"].astype(float)
    inventory = lines["inventory_available_at_order"].astype(float)

    allocation_gap_qty = (requested - allocated).clip(lower=0)
    allocation_gap_ratio = (allocation_gap_qty / requested).clip(0, 1)
    inventory_coverage_ratio = (inventory / requested).clip(lower=0)
    line_value = requested * lines["base_unit_value"].fillna(0.0)
    is_critical = (lines["criticality_tier"] == "CRITICAL").astype(float)
    vendor_signal = lines["leading_signal_VENDOR_FAILURE"].fillna(0).astype(float)
    inventory_signal = lines["leading_signal_INVENTORY_SHORTAGE"].fillna(0).astype(float)

    evidence_strength = (
        0.55 * allocation_gap_ratio
        + 0.20 * vendor_signal
        + 0.15 * inventory_signal
        + 0.10 * is_critical
    ).clip(0, 1)
    likely_affected = (evidence_strength >= LIKELY_AFFECTED_THRESHOLD) | (
        allocation_gap_ratio > 0.02
    )

    evidence = pd.DataFrame(
        {
            "order_line_id": lines["order_line_id"],
            "order_id": lines["order_id"],
            "sku_id": lines["sku_id"],
            "criticality_tier": lines["criticality_tier"],
            "requested_qty": lines["requested_qty"],
            "allocated_qty": lines["allocated_qty"],
            "inventory_available_at_order": lines["inventory_available_at_order"],
            "allocation_gap_qty": allocation_gap_qty.round(2),
            "allocation_gap_ratio": allocation_gap_ratio.round(4),
            "inventory_coverage_ratio": inventory_coverage_ratio.round(4),
            "line_value": line_value.round(2),
            "evidence_strength": evidence_strength.round(4),
            "likely_affected": likely_affected,
        }
    )
    return evidence


def order_line_aggregates(line_evidence: pd.DataFrame) -> pd.DataFrame:
    """Safe, order-level rollups of line evidence for the order feature table.

    Only aggregates of point-in-time-safe line evidence -- never raw
    per-line/SKU identity -- so an order's single risk score still reflects
    order-grain modeling while gaining shortfall-shape context.
    """
    if line_evidence.empty:
        return pd.DataFrame(
            columns=[
                "order_id",
                "worst_line_shortage_ratio",
                "affected_line_count",
                "max_line_risk_evidence",
                "critical_sku_share",
                "line_qty_concentration",
            ]
        )

    def _concentration(group: pd.DataFrame) -> float:
        qty = group["requested_qty"].astype(float)
        total = qty.sum()
        if total <= 0:
            return 1.0
        shares = qty / total
        return float((shares**2).sum())

    def _critical_share(group: pd.DataFrame) -> float:
        total_value = group["line_value"].sum()
        if total_value <= 0:
            return 0.0
        critical_value = group.loc[group["criticality_tier"] == "CRITICAL", "line_value"].sum()
        return float(critical_value / total_value)

    grouped = line_evidence.groupby("order_id")
    aggregates = grouped.apply(
        lambda group: pd.Series(
            {
                "worst_line_shortage_ratio": float(group["allocation_gap_ratio"].max()),
                "affected_line_count": int(group["likely_affected"].sum()),
                "max_line_risk_evidence": float(group["evidence_strength"].max()),
                "critical_sku_share": _critical_share(group),
                "line_qty_concentration": _concentration(group),
            }
        ),
        include_groups=False,
    ).reset_index()
    return aggregates


def affected_sku_summary(line_evidence: pd.DataFrame, *, top_n: int = 3) -> pd.DataFrame:
    """Return a compact, order-level JSON summary of the most likely-affected SKUs."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rows: list[dict[str, Any]] = []
    for order_id, group in line_evidence.groupby("order_id"):
        affected = group.loc[group["likely_affected"]].sort_values(
            "evidence_strength", ascending=False
        )
        top = affected.head(top_n)
        summary = [
            {
                "sku_id": row.sku_id,
                "criticality_tier": row.criticality_tier,
                "allocation_gap_ratio": round(float(row.allocation_gap_ratio), 4),
                "evidence_strength": round(float(row.evidence_strength), 4),
            }
            for row in top.itertuples(index=False)
        ]
        rows.append(
            {
                "order_id": order_id,
                "affected_sku_count": int(len(affected)),
                "affected_skus_json": json.dumps(summary, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def evaluate_line_evidence(
    line_evidence: pd.DataFrame,
    line_truth: pd.DataFrame,
) -> dict[str, Any]:
    """Precision/recall of ``likely_affected`` against simulator line truth.

    Also reports a naive "attribute every line on a miss order" baseline so
    the report can show the targeted evidence is demonstrably better than
    blaming every SKU on an at-risk order.
    """
    merged = line_evidence.merge(
        line_truth[["order_line_id", "truly_affected"]], on="order_line_id", validate="one_to_one"
    )
    truth = merged["truly_affected"].astype(bool).to_numpy()
    predicted = merged["likely_affected"].astype(bool).to_numpy()
    naive_all = np.ones_like(predicted, dtype=bool)

    def _precision_recall(pred: np.ndarray) -> dict[str, float]:
        true_positive = int((pred & truth).sum())
        predicted_positive = int(pred.sum())
        actual_positive = int(truth.sum())
        precision = true_positive / predicted_positive if predicted_positive else float("nan")
        recall = true_positive / actual_positive if actual_positive else float("nan")
        return {"precision": precision, "recall": recall, "flagged_lines": predicted_positive}

    return {
        "evaluated_lines": int(len(merged)),
        "truly_affected_lines": int(truth.sum()),
        "targeted_evidence": _precision_recall(predicted),
        "naive_all_lines_baseline": _precision_recall(naive_all),
        "note": (
            "targeted_evidence uses allocation-gap/criticality/vendor-signal "
            "evidence per line; naive_all_lines_baseline always flags every "
            "line on the scored orders. Precision should be materially higher "
            "for targeted_evidence at comparable or better recall."
        ),
    }
