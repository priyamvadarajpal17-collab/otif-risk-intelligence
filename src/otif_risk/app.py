"""Streamlit presentation layer for OTIF risk intelligence."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from otif_risk.bayesian import CHAIN_PARENTS
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


def _find_latest_ops_directory(artifacts_root: str | Path) -> Path | None:
    root = Path(artifacts_root)
    candidates = [
        path
        for path in root.glob("ops-*")
        if path.is_dir() and (path / "operations_summary.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _find_benchmark_path(artifacts_root: str | Path) -> Path | None:
    candidate = Path(artifacts_root) / "benchmark.json"
    return candidate if candidate.is_file() else None


def _parse_pathway_route(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return [str(node) for node in parsed.get("route", [])]
    return []


def _parse_pathway(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _bayesian_graph_svg(pathway: dict[str, Any]) -> str:
    """Render the fixed Bayesian graph, highlighting one order's evidence and route."""
    positions = {
        "ORDER_CAPTURE": (95, 55),
        "VENDOR_FAILURE": (95, 155),
        "DC_CAPACITY": (95, 255),
        "CUSTOMER_DELIVERY": (95, 355),
        "INVENTORY_SHORTAGE": (350, 155),
        "WAREHOUSE_OPS": (570, 215),
        "TRANSPORT": (790, 215),
        "OTIF_MISS": (1010, 215),
    }
    node_width, node_height = 170, 50
    route = [str(node) for node in pathway.get("route", [])]
    route_edges = set(zip(route, route[1:], strict=False))
    evidence = pathway.get("evidence", {})
    active = set(str(node) for node in pathway.get("active_evidence", []))

    edge_parts: list[str] = []
    for target, parents in CHAIN_PARENTS.items():
        for source in parents:
            source_x, source_y = positions[source]
            target_x, target_y = positions[target]
            highlighted = (source, target) in route_edges
            color = "#e3522c" if highlighted else "#7b8078"
            width = 4 if highlighted else 2
            edge_parts.append(
                f'<path d="M {source_x + node_width / 2:.0f} {source_y:.0f} '
                f'L {target_x - node_width / 2:.0f} {target_y:.0f}" '
                f'stroke="{color}" stroke-width="{width}" fill="none" '
                f'marker-end="url(#arrow-{"hot" if highlighted else "base"})"/>'
            )

    node_parts: list[str] = []
    for node, (x, y) in positions.items():
        is_active = node in active
        is_route = node in route
        fill = "#f7e6da" if is_route else ("#e7eef5" if is_active else "#fffdf7")
        stroke = "#e3522c" if is_route else ("#2b5b8c" if is_active else "#34362f")
        value = evidence.get(node)
        status = "ACTIVE EVIDENCE" if value == 1 else ("OBSERVED CLEAR" if value == 0 else "")
        label = escape(node.replace("_", " "))
        node_parts.append(
            f'<g><rect x="{x - node_width / 2}" y="{y - node_height / 2}" '
            f'width="{node_width}" height="{node_height}" rx="5" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{3 if is_route or is_active else 1.5}"/>'
            f'<text x="{x}" y="{y - 2}" text-anchor="middle" fill="#16201c" '
            f'font-size="13" font-weight="700">{label}</text>'
            f'<text x="{x}" y="{y + 15}" text-anchor="middle" fill="#53635c" '
            f'font-size="9">{status}</text></g>'
        )

    return (
        '<svg viewBox="0 0 1120 420" width="100%" role="img" '
        'aria-label="Bayesian causal network">'
        '<defs><marker id="arrow-base" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#7b8078"/></marker>'
        '<marker id="arrow-hot" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#e3522c"/></marker></defs>'
        '<rect width="1120" height="420" fill="#f2f0e8" rx="8"/>'
        + "".join(edge_parts)
        + "".join(node_parts)
        + '<text x="24" y="402" fill="#53635c" font-size="11">'
        "Blue = active evidence · Orange = selected route · Gray = fixed model structure"
        "</text></svg>"
    )


def _parse_affected_skus(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


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
        :root {
            --ink: #16201c; --signal: #e3522c; --steel: #53635c;
            --paper: #f2f0e8; --panel: #fffdf7; --line: #c8c6bc;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background:
              linear-gradient(rgba(22,32,28,.035) 1px, transparent 1px),
              linear-gradient(90deg, rgba(22,32,28,.035) 1px, transparent 1px),
              var(--paper);
            background-size: 28px 28px;
            font-family: "IBM Plex Sans", sans-serif;
            color: var(--ink) !important;
        }
        .stApp p, .stApp span:not(.status-strip), .stApp label,
        .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"], .stApp [role="tab"] {
            color: var(--ink) !important;
        }
        h1, h2, h3 {
            color: var(--ink) !important;
            font-family: "Barlow Condensed", sans-serif !important; letter-spacing: .02em;
        }
        h1 { text-transform: uppercase; border-left: 8px solid var(--signal); padding-left: .6rem; }
        [data-testid="stMetric"] {
            background: rgba(255,253,247,.94); border-top: 3px solid var(--ink);
            padding: .8rem 1rem; box-shadow: 3px 3px 0 rgba(22,32,28,.12);
        }
        [data-testid="stSidebar"] {
            background: #e6e3d8 !important; border-right: 1px solid rgba(22,32,28,.22);
        }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="popover"] {
            background-color: var(--panel) !important;
            color: var(--ink) !important;
        }
        .stApp input::placeholder, .stApp textarea::placeholder {
            color: #71756d !important; opacity: 1;
        }
        .stApp [data-testid="stAlert"] {
            color: var(--ink) !important;
        }
        .stApp code {
            background: #dedbd0 !important; color: #20211d !important;
        }
        .stApp button {
            color: var(--ink);
        }
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
        st.subheader("Evidence + causal pathway")
        st.write(f"**Primary cause:** {str(order['primary_cause']).replace('_', ' ').title()}")
        route = _parse_pathway_route(order.get("causal_pathway"))
        if route:
            st.markdown(
                " → ".join(f"`{node}`" for node in route),
            )
        else:
            st.caption("No active evidence route (order shows no elevated leading signals).")
        st.write(f"**Recommended action:** {order['recommended_action']}")
        st.caption(f"Accountable owner: {order['action_owner']}")
        if str(order["decision_status"]) == "CONTESTED" and order.get("contested_with"):
            st.warning(f"Contested resource — competing with: {order['contested_with']}")

        st.subheader("Affected SKUs")
        affected = _parse_affected_skus(order.get("affected_skus_json"))
        if affected:
            st.dataframe(pd.DataFrame(affected), hide_index=True, use_container_width=True)
        else:
            st.caption("No line-level evidence flags this order's SKUs as likely affected.")
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


def _operations_view(artifacts_root: str) -> None:
    st.header("Operations")
    ops_dir = _find_latest_ops_directory(artifacts_root)
    if ops_dir is None:
        st.info(
            "No completed operations replay found under this artifacts directory. "
            "Run `uv run python -m otif_risk.operations` to generate one."
        )
        return
    summary = json.loads((ops_dir / "operations_summary.json").read_text(encoding="utf-8"))
    columns = st.columns(4)
    columns[0].metric("Replay days completed", summary["replay_days_completed"])
    columns[1].metric("Model versions trained", summary["model_versions_trained"])
    columns[2].metric("Retrain events", len(summary["retrain_events"]))
    columns[3].metric("Drift warning days", len(summary["drift_warning_days"]))
    st.caption(
        f"Replay directory: {ops_dir.name} · simulated cutoff {summary['initial_cutoff']} · "
        f"current model v{summary['final_model_version']} (threshold "
        f"{summary['final_threshold']:.3f}, fusion weight {summary['final_fusion_weight']:.1f})"
    )

    daily_log_path = ops_dir / "daily_log.json"
    if daily_log_path.is_file():
        daily = pd.DataFrame(json.loads(daily_log_path.read_text(encoding="utf-8")))
        st.subheader("Daily open-order timeline")
        display_columns = [
            "simulated_day",
            "open_orders",
            "newly_closed_orders",
            "recommended",
            "contested",
            "monitor",
            "model_version",
            "retrained",
            "retrain_trigger",
        ]
        st.dataframe(
            daily[[column for column in display_columns if column in daily]],
            hide_index=True,
            use_container_width=True,
        )
        if "open_orders" in daily:
            chart_columns = ["open_orders", "recommended", "contested"]
            st.line_chart(daily.set_index("simulated_day")[chart_columns])

    registry_path = ops_dir / "model_registry.json"
    if registry_path.is_file():
        st.subheader("Model registry")
        registry = pd.DataFrame(json.loads(registry_path.read_text(encoding="utf-8")))
        st.dataframe(
            registry[
                [
                    column
                    for column in (
                        "version",
                        "trained_at_simulated_day",
                        "n_training_orders",
                        "threshold",
                        "fusion_weight",
                        "fusion_label",
                        "trigger",
                    )
                    if column in registry
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )


def _model_health_view(artifacts_root: str, metrics: dict[str, Any]) -> None:
    st.header("Model health")
    benchmark_path = _find_benchmark_path(artifacts_root)
    if benchmark_path is not None:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        st.caption(
            f"Multi-seed benchmark across seeds {benchmark['seeds']} "
            f"({benchmark['n_orders']} orders each)."
        )
        summary_rows = [
            {"metric": name, **stats} for name, stats in benchmark["summary"].items()
        ]
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
        st.subheader("Acceptance gates")
        gate_rows = [
            {"gate": name, **(value if isinstance(value, dict) else {"pass": value})}
            for name, value in benchmark["acceptance_gates"].items()
        ]
        st.dataframe(pd.DataFrame(gate_rows), hide_index=True, use_container_width=True)
        st.caption(benchmark["note"])
    else:
        st.info("No benchmark.json found; showing this run's own model comparison instead.")

    model_scores = metrics.get("model_scores")
    if model_scores:
        st.subheader("This run: XGBoost vs. Bayesian vs. fused")
        rows = []
        for label in ("xgb", "bbn", "fused"):
            space = model_scores.get(label, {})
            test_metrics = space.get("test_metrics", {})
            rows.append(
                {
                    "model": label,
                    "pr_auc": test_metrics.get("pr_auc"),
                    "roc_auc": test_metrics.get("roc_auc"),
                    "precision": test_metrics.get("precision"),
                    "recall": test_metrics.get("recall"),
                    "brier": test_metrics.get("brier"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    fusion_comparison = metrics.get("fusion_comparison")
    if fusion_comparison:
        st.subheader("Fusion weight comparison (validation)")
        st.dataframe(pd.DataFrame(fusion_comparison), hide_index=True, use_container_width=True)
    architecture = metrics.get("architecture", {})
    chosen_weight = architecture.get("fusion_chosen_weight")
    if chosen_weight is not None:
        if float(chosen_weight) >= 0.999:
            st.info(
                "Validation selected XGBoost-only risk scoring because the Bayesian "
                "probability did not improve calibration without reducing recall. "
                "The Bayesian network remains active for causal-pathway inference; "
                "its probability is shown as a diagnostic and is not forced into the final score."
            )
        else:
            st.info(
                f"Validation selected {float(chosen_weight):.0%} XGBoost and "
                f"{1 - float(chosen_weight):.0%} Bayesian contribution to the final score."
            )
    if architecture.get("fusion"):
        st.caption(architecture["fusion"])


def _bayesian_network_view(decisions: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.header("Bayesian network")
    st.caption(
        "A compact operational pathway model. It explains how observed disruption can "
        "propagate; XGBoost remains the primary predictive model."
    )
    order_ids = decisions["order_id"].astype(str).tolist()
    default_index = next(
        (
            index
            for index, status in enumerate(decisions["decision_status"].astype(str))
            if status in {"RECOMMENDED", "CONTESTED"}
        ),
        0,
    )
    selected_id = st.selectbox("Inspect order", order_ids, index=default_index, key="bbn-order")
    order = decisions.loc[decisions["order_id"].astype(str) == selected_id].iloc[0]
    pathway = _parse_pathway(order.get("causal_pathway"))

    columns = st.columns(4)
    columns[0].metric("Bayesian posterior", f"{float(order.get('bbn_risk_score', 0)):.1%}")
    columns[1].metric("XGBoost risk", f"{float(order.get('xgb_risk_score', 0)):.1%}")
    columns[2].metric("Combined risk", f"{float(order.get('combined_risk_score', 0)):.1%}")
    columns[3].metric(
        "Evidence delta",
        f"{float(pathway.get('evidence_delta', 0)):+.1%}",
    )
    st.markdown(_bayesian_graph_svg(pathway), unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Order pathway")
        route = pathway.get("route", [])
        if route:
            st.markdown(" → ".join(f"`{node}`" for node in route))
        else:
            st.caption("No elevated route for this order.")
        st.write(
            f"**Prior risk:** {float(pathway.get('prior_risk', 0)):.1%}  \n"
            f"**Posterior risk:** {float(pathway.get('posterior_risk', 0)):.1%}  \n"
            f"**Inference:** {pathway.get('inference_mode', 'not available')}"
        )
    with right:
        st.subheader("Model role")
        architecture = metrics.get("architecture", {})
        xgb_weight = float(architecture.get("fusion_chosen_weight", 1.0))
        st.write(
            f"Final score contribution: **{xgb_weight:.0%} XGBoost / "
            f"{1 - xgb_weight:.0%} Bayesian**"
        )
        bbn_metrics = metrics.get("model_scores", {}).get("bbn", {}).get("test_metrics", {})
        if bbn_metrics:
            st.write(
                f"Bayesian standalone PR-AUC: **{float(bbn_metrics.get('pr_auc', 0)):.3f}**  \n"
                f"Bayesian precision: **{float(bbn_metrics.get('precision', 0)):.1%}**  \n"
                f"Bayesian recall: **{float(bbn_metrics.get('recall', 0)):.1%}**"
            )
        st.caption(
            "The pathway is a probabilistic association within a fixed expert-defined "
            "structure, not proof of causality."
        )


def main(artifacts_root: str | Path | None = None) -> None:
    """Render the control-tower prototype application."""

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
            [
                "Order lookup",
                "Ranked portfolio",
                "Hotspots + impact",
                "Operations",
                "Model health",
                "Bayesian network",
            ],
            label_visibility="collapsed",
        )
        st.caption(f"Loaded {len(decisions):,} scored orders")
    if view == "Order lookup":
        _order_lookup(decisions, run_directory)
    elif view == "Ranked portfolio":
        _portfolio(decisions)
    elif view == "Hotspots + impact":
        _hotspots_and_impact(decisions, run_directory)
    elif view == "Operations":
        _operations_view(str(root.resolve()))
    elif view == "Bayesian network":
        _bayesian_network_view(decisions, metrics)
    else:
        _model_health_view(str(root.resolve()), metrics)


if __name__ == "__main__":
    main()
