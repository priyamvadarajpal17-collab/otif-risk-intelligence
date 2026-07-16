"""Streamlit presentation layer for OTIF risk intelligence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from otif_risk.decisions import (
    DEFAULT_RISK_THRESHOLD,
    build_rollups,
    recommend_orders,
    service_impact_summary,
)
from otif_risk.feedback import append_feedback
from otif_risk.narratives import order_narrative

SCORED_ORDERS_FILENAME = "scored_orders.csv"
ORDER_LINES_FILENAME = "order_lines.csv"
#: Decision fields the pipeline already persists to scored_orders.csv. When all
#: of these are present, the UI reuses them as-is instead of recomputing a
#: (potentially different) policy, keeping Streamlit aligned with the audited
#: pipeline run rather than silently overriding it.
PERSISTED_DECISION_COLUMNS = {
    "decision_status",
    "recommended_action",
    "action_owner",
    "resource_type",
    "resource_id",
    "priority_score",
    "estimated_penalty_exposure",
    "estimated_avoidable_penalty",
    "quantity_at_risk",
}


def latest_run_directory(artifacts_root: str | Path) -> Path:
    """Return the newest complete run directory."""

    root = Path(artifacts_root)
    candidates = [
        path
        for path in root.glob("run-*")
        if path.is_dir() and (path / "metrics.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run-* directory found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _find_scored_orders(run_directory: Path) -> Path:
    direct = run_directory / SCORED_ORDERS_FILENAME
    if direct.is_file():
        return direct
    matches = sorted(run_directory.glob(f"**/{SCORED_ORDERS_FILENAME}"))
    if not matches:
        csv_files = sorted(run_directory.glob("**/*.csv"))
        for path in csv_files:
            if {"order_id", "combined_risk_score"}.issubset(pd.read_csv(path, nrows=1).columns):
                return path
        raise FileNotFoundError(f"No scored-orders CSV found under {run_directory}")
    return matches[0]


@st.cache_data(show_spinner=False)
def load_run_artifacts(artifacts_root: str) -> tuple[str, dict[str, Any], pd.DataFrame]:
    """Load metrics and scored orders from the latest pipeline run.

    Reuses the pipeline's persisted decision fields (decision_status, priority,
    recommended action, exposure, etc.) whenever `scored_orders.csv` already
    carries them, so the UI reflects exactly the decisions the pipeline
    computed and wrote to disk. `recommend_orders` is only invoked as a
    fallback for artifacts that lack a persisted decision, using the fused
    decision threshold recorded in metrics.json —
    never a hardcoded default silently overriding the run's own policy.
    """

    run_directory = latest_run_directory(artifacts_root)
    with (run_directory / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain a JSON object")
    scored_orders = pd.read_csv(_find_scored_orders(run_directory))
    if PERSISTED_DECISION_COLUMNS <= set(scored_orders.columns):
        decisions = scored_orders
    else:
        threshold = float(metrics.get("threshold", DEFAULT_RISK_THRESHOLD))
        decisions = recommend_orders(scored_orders, risk_threshold=threshold)
    return str(run_directory), metrics, decisions


@st.cache_data(show_spinner=False)
def _load_order_lines(run_directory: str) -> pd.DataFrame | None:
    """Best-effort load of order_lines.csv for the SKU rollup, if it was persisted."""

    matches = sorted(Path(run_directory).glob(f"**/{ORDER_LINES_FILENAME}"))
    if not matches:
        return None
    return pd.read_csv(matches[0])


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        :root { --ink: #16201c; --signal: #e3522c; --steel: #53635c; --paper: #f2f0e8; }
        .stApp {
            background:
              linear-gradient(rgba(22,32,28,.035) 1px, transparent 1px),
              linear-gradient(90deg, rgba(22,32,28,.035) 1px, transparent 1px),
              var(--paper);
            background-size: 28px 28px;
            font-family: "IBM Plex Sans", sans-serif;
            color: var(--ink);
        }
        h1, h2, h3 {
            font-family: "Barlow Condensed", sans-serif !important; letter-spacing: .02em;
        }
        h1 { text-transform: uppercase; border-left: 8px solid var(--signal); padding-left: .6rem; }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.72); border-top: 3px solid var(--ink);
            padding: .8rem 1rem; box-shadow: 3px 3px 0 rgba(22,32,28,.12);
        }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(22,32,28,.22); }
        .status-strip {
            display:inline-block; padding:.25rem .55rem; background:var(--ink); color:white;
            font-family:"Barlow Condensed",sans-serif; letter-spacing:.08em;
            text-transform:uppercase;
        }
        .run-stamp { color:var(--steel); font-size:.78rem; letter-spacing:.05em; }
        div[data-testid="stDataFrame"] { border: 1px solid rgba(22,32,28,.24); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _header(run_directory: str, metrics: dict[str, Any]) -> None:
    st.title("OTIF Intervention Desk")
    run_name = Path(run_directory).name
    model_note = metrics.get("model", metrics.get("model_name", "combined XGB + BBN"))
    st.markdown(
        f'<span class="status-strip">Latest scored run</span> '
        f'<span class="run-stamp">{run_name} · {model_note}</span>',
        unsafe_allow_html=True,
    )


def _order_lookup(decisions: pd.DataFrame, run_directory: str) -> None:
    st.header("Order lookup")
    order_ids = decisions["order_id"].astype(str).tolist()
    selected_id = st.selectbox("Order ID", order_ids)
    order = decisions.loc[decisions["order_id"].astype(str) == selected_id].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Combined risk", f"{order['combined_risk_score']:.1%}")
    columns[1].metric("Decision", str(order["decision_status"]))
    columns[2].metric("Priority", f"{order['priority_score']:.1f}")
    columns[3].metric(
        "Penalty exposure", _format_currency(float(order["estimated_penalty_exposure"]))
    )
    st.info(order_narrative(order.to_dict()))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Evidence")
        st.write(f"**Primary cause:** {str(order['primary_cause']).replace('_', ' ').title()}")
        st.write(f"**Causal pathway:** {order.get('causal_pathway', 'Not available')}")
        st.write(f"**Recommended action:** {order['recommended_action']}")
        st.caption(f"Accountable owner: {order['action_owner']}")
    with right:
        st.subheader("Planner feedback")
        action = st.radio("Disposition", ["ACCEPT", "REJECT", "OVERRIDE"], horizontal=True)
        override = (
            st.text_input("Replacement action") if action == "OVERRIDE" else ""
        )
        reason = st.text_input("Reason", placeholder="Required for reject or override")
        if st.button("Record feedback", type="primary"):
            try:
                append_feedback(
                    Path(run_directory) / "planner_feedback.csv",
                    order_id=selected_id,
                    feedback_action=action,
                    original_status=str(order["decision_status"]),
                    original_recommendation=str(order["recommended_action"]),
                    override_recommendation=override,
                    reason=reason,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Feedback appended to this run's audit log.")


def _portfolio(decisions: pd.DataFrame) -> None:
    st.header("Ranked portfolio")
    status_filter = st.multiselect(
        "Decision status",
        ["RECOMMENDED", "CONTESTED", "MONITOR"],
        default=["RECOMMENDED", "CONTESTED"],
    )
    filtered = decisions[decisions["decision_status"].isin(status_filter)].sort_values(
        ["priority_score", "combined_risk_score"], ascending=False
    )
    st.caption(f"{len(filtered):,} orders in the current work queue")
    columns = [
        "order_id",
        "decision_status",
        "priority_score",
        "combined_risk_score",
        "primary_cause",
        "recommended_action",
        "estimated_penalty_exposure",
    ]
    st.dataframe(
        filtered[[column for column in columns if column in filtered]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "combined_risk_score": st.column_config.ProgressColumn(
                "Risk", min_value=0.0, max_value=1.0, format="%.0%%"
            ),
            "estimated_penalty_exposure": st.column_config.NumberColumn(
                "Penalty exposure", format="$%.2f"
            ),
        },
    )


def _hotspots_and_impact(decisions: pd.DataFrame, run_directory: str) -> None:
    st.header("Hotspots + impact")
    impact = service_impact_summary(decisions)
    columns = st.columns(4)
    columns[0].metric("Recommended", impact["recommended_orders"])
    columns[1].metric("Resource conflicts", impact["contested_orders"])
    columns[2].metric("Penalty exposure", _format_currency(impact["penalty_exposure"]))
    columns[3].metric(
        "Potentially avoidable", _format_currency(impact["estimated_avoidable_penalty"])
    )
    rollups = build_rollups(decisions, order_lines=_load_order_lines(run_directory))
    entities = ("vendor", "dc", "lane", "customer", "order_type", "sku")
    tabs = st.tabs(["Vendors", "DCs", "Lanes", "Customers", "Order type", "SKUs"])
    for tab, entity in zip(tabs, entities, strict=True):
        with tab:
            if rollups[entity].empty:
                st.caption(f"No {entity} identifiers in this scored run.")
            else:
                st.dataframe(rollups[entity], hide_index=True, use_container_width=True)
    with st.expander("Impact assumptions"):
        st.write(impact["assumptions"]["penalty"])
        st.write(impact["assumptions"]["service_impact"])


def main(artifacts_root: str | Path | None = None) -> None:
    """Render the three-view prototype application."""

    st.set_page_config(page_title="OTIF Intervention Desk", page_icon="▦", layout="wide")
    _inject_style()
    root = Path(
        artifacts_root
        or os.environ.get("OTIF_ARTIFACTS_DIR", Path.cwd() / "artifacts")
    ).expanduser()
    try:
        run_directory, metrics, decisions = load_run_artifacts(str(root.resolve()))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        st.title("OTIF Intervention Desk")
        st.error(f"Artifacts are not ready: {exc}")
        st.caption("Run the scoring pipeline, then refresh this page.")
        return

    _header(run_directory, metrics)
    with st.sidebar:
        st.subheader("Control tower")
        view = st.radio(
            "View",
            ["Order lookup", "Ranked portfolio", "Hotspots + impact"],
            label_visibility="collapsed",
        )
        st.caption(f"Loaded {len(decisions):,} scored orders")
    if view == "Order lookup":
        _order_lookup(decisions, run_directory)
    elif view == "Ranked portfolio":
        _portfolio(decisions)
    else:
        _hotspots_and_impact(decisions, run_directory)


if __name__ == "__main__":
    main()
