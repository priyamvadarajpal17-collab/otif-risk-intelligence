"""Streamlit presentation layer for OTIF risk intelligence."""

from __future__ import annotations

import json
import os
import statistics
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from otif_risk.bayesian import CHAIN_PARENTS, IN_FULL_FAILURE, LATE_DELIVERY
from otif_risk.copilot_audit import default_audit_path, read_audit_records
from otif_risk.copilot_context import (
    PORTFOLIO_QUESTIONS,
    EvidencePacket,
    build_order_evidence_packet,
)
from otif_risk.copilot_fallback import ORDER_QUESTIONS
from otif_risk.decisions import (
    DEFAULT_RISK_THRESHOLD,
    build_rollups,
    recommend_orders,
    service_impact_summary,
)
from otif_risk.feedback import append_feedback
from otif_risk.llm_copilot import (
    CopilotAnswer,
    get_order_copilot_response,
    get_portfolio_copilot_response,
    is_live_configured,
)
from otif_risk.manifest import verify_manifest
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


def _find_policy_benchmark_path(artifacts_root: str | Path) -> Path | None:
    candidate = Path(artifacts_root) / "policy_benchmark.json"
    return candidate if candidate.is_file() else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_manifest(run_directory: str | Path) -> dict[str, Any] | None:
    """Best-effort load of run_manifest.json for Copilot version facts."""
    manifest_path = Path(run_directory) / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return None


def _badge(text: str, kind: str) -> str:
    """A small, high-contrast lifecycle/status badge (see ``_inject_style``'s
    ``.gov-badge-*`` classes): green only for verified/passed/promoted,
    red for held/regression/failed, blue for neutral/informational,
    orange for warnings."""
    return f'<span class="gov-badge gov-badge-{kind}">{escape(text)}</span>'


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


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


#: Fixed layout for the 10-node mechanism graph. Roots on the left, the two
#: operational propagation stages in the middle, the IN_FULL_FAILURE (quantity)
#: and LATE_DELIVERY (timing) mechanism nodes just before the OTIF_MISS
#: endpoint on the right -- mirroring the actual OTIF ("on time" AND "in
#: full") definition instead of one flat star of causes.
_GRAPH_POSITIONS: dict[str, tuple[int, int]] = {
    "ORDER_CAPTURE": (95, 55),
    "VENDOR_FAILURE": (95, 175),
    "DC_CAPACITY": (95, 295),
    "CUSTOMER_DELIVERY": (95, 415),
    "INVENTORY_SHORTAGE": (365, 175),
    "WAREHOUSE_OPS": (620, 245),
    "TRANSPORT": (875, 245),
    IN_FULL_FAILURE: (1105, 105),
    LATE_DELIVERY: (1105, 375),
    "OTIF_MISS": (1340, 245),
}
_GRAPH_NODE_WIDTH, _GRAPH_NODE_HEIGHT = 190, 52
_GRAPH_STATUS_STYLE: dict[str, dict[str, str]] = {
    "evidence": {"fill": "#e7eef5", "stroke": "#2b5b8c", "label": "ACTIVE EVIDENCE", "dash": ""},
    "propagation": {
        "fill": "#f7e6da",
        "stroke": "#e3522c",
        "label": "ACTIVE PROPAGATION",
        "dash": "",
    },
    "intervened": {
        "fill": "#e1f0e5",
        "stroke": "#2f7d4f",
        "label": "INTERVENED (SCENARIO)",
        "dash": "",
    },
    "observed_clear": {
        "fill": "#fffdf7",
        "stroke": "#34362f",
        "label": "OBSERVED CLEAR",
        "dash": "",
    },
    "unknown": {"fill": "#efece2", "stroke": "#53635c", "label": "UNOBSERVED", "dash": "5,4"},
}


def _node_status(node: str, pathway: dict[str, Any], intervened_nodes: set[str]) -> str:
    if node in intervened_nodes:
        return "intervened"
    evidence = pathway.get("evidence", {}) or {}
    active = {str(item) for item in pathway.get("active_evidence", [])}
    routes = pathway.get("routes") or ([pathway["route"]] if pathway.get("route") else [])
    on_route = any(node in route for route in routes)
    if node in active:
        return "evidence"
    if on_route:
        return "propagation"
    if node in evidence:
        return "observed_clear"
    return "unknown"


def _causal_graph_svg(
    pathway: dict[str, Any], intervened_nodes: set[str] | None = None
) -> str:
    """Render the 10-node mechanism graph, highlighting one order's evidence,

    active propagation route(s), and -- when a structural-intervention
    scenario is being inspected -- the intervened node(s) in green (green is
    reserved exclusively for simulated risk reduction in this app, never for
    an actual observed/decision state).
    """
    intervened_nodes = intervened_nodes or set()
    evidence = pathway.get("evidence", {}) or {}
    routes = pathway.get("routes") or ([pathway["route"]] if pathway.get("route") else [])
    route_edges = {edge for route in routes for edge in zip(route, route[1:], strict=False)}

    edge_parts: list[str] = []
    for target, parents in CHAIN_PARENTS.items():
        for source in parents:
            source_x, source_y = _GRAPH_POSITIONS[source]
            target_x, target_y = _GRAPH_POSITIONS[target]
            severed = target in intervened_nodes
            highlighted = (source, target) in route_edges
            if severed:
                color, width, dash, marker = "#2f7d4f", 2.5, "6,4", "severed"
            elif highlighted:
                color, width, dash, marker = "#e3522c", 4, "", "hot"
            else:
                color, width, dash, marker = "#8a8d84", 2, "", "base"
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            edge_parts.append(
                f'<path d="M {source_x + _GRAPH_NODE_WIDTH / 2:.0f} {source_y:.0f} '
                f'L {target_x - _GRAPH_NODE_WIDTH / 2:.0f} {target_y:.0f}" '
                f'stroke="{color}" stroke-width="{width}" fill="none"{dash_attr} '
                f'marker-end="url(#arrow-{marker})"/>'
            )

    node_parts: list[str] = []
    for node, (x, y) in _GRAPH_POSITIONS.items():
        status = _node_status(node, pathway, intervened_nodes)
        style = _GRAPH_STATUS_STYLE[status]
        value = evidence.get(node)
        detail = style["label"]
        if status in {"evidence", "observed_clear"} and value is not None:
            detail = f"{style['label']} ({value})"
        label = escape(node.replace("_", " "))
        dash_attr = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        node_parts.append(
            f'<g><rect x="{x - _GRAPH_NODE_WIDTH / 2}" y="{y - _GRAPH_NODE_HEIGHT / 2}" '
            f'width="{_GRAPH_NODE_WIDTH}" height="{_GRAPH_NODE_HEIGHT}" rx="5" '
            f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="3"{dash_attr}/>'
            f'<text x="{x}" y="{y - 3}" text-anchor="middle" fill="#16201c" '
            f'font-size="13" font-weight="700">{label}</text>'
            f'<text x="{x}" y="{y + 15}" text-anchor="middle" fill="#53635c" '
            f'font-size="8.5">{escape(detail)}</text></g>'
        )

    width, height = 1440, 480
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        'aria-label="Causal intelligence mechanism network">'
        '<defs>'
        '<marker id="arrow-base" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#8a8d84"/></marker>'
        '<marker id="arrow-hot" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#e3522c"/></marker>'
        '<marker id="arrow-severed" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#2f7d4f"/></marker>'
        "</defs>"
        f'<rect width="{width}" height="{height}" fill="#f2f0e8" rx="8"/>'
        + "".join(edge_parts)
        + "".join(node_parts)
        + f'<text x="24" y="{height - 14}" fill="#53635c" font-size="11">'
        "Blue = active evidence &#183; Orange = active propagation &#183; "
        "Green = intervened (scenario) &#183; Gray dashed = unobserved"
        "</text></svg>"
    )


#: Kept as a private alias: earlier iterations called this the "Bayesian graph".
_bayesian_graph_svg = _causal_graph_svg


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
            --signal-blue: #2b5b8c; --safe-green: #2f7d4f; --danger-red: #a63b2a;
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
        .gov-badge {
            display:inline-block; padding:.2rem .6rem; border-radius:2px;
            font-family:"Barlow Condensed",sans-serif; letter-spacing:.06em;
            text-transform:uppercase; font-weight:700; font-size:.82rem; color:#fff !important;
        }
        .gov-badge-pass { background: var(--safe-green); }
        .gov-badge-fail { background: var(--danger-red); }
        .gov-badge-info { background: var(--signal-blue); }
        .gov-badge-warn { background: var(--signal); }
        .gov-card {
            background: rgba(255,253,247,.96); border: 1px solid rgba(22,32,28,.18);
            border-left: 5px solid var(--signal-blue); padding: .8rem 1rem; margin-bottom:.6rem;
        }
        .gov-card.evaluation-only { border-left-color: var(--signal); }
        .oracle-note {
            font-family:"Barlow Condensed",sans-serif; letter-spacing:.05em;
            text-transform:uppercase;
            color: var(--signal); font-weight:700; font-size:.8rem;
        }
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
                "Risk", min_value=0.0, max_value=1.0, format="percent"
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


def _confidence_badge_color(band: str) -> str:
    return {"HIGH": "#2f7d4f", "MEDIUM": "#c8862b", "LOW": "#a63b2a"}.get(band, "#53635c")


def _causal_intelligence_view(decisions: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.header("Causal intelligence")
    st.caption(
        "A 10-node mechanism graph splits OTIF risk into its two real components -- "
        "IN_FULL_FAILURE (quantity) and LATE_DELIVERY (timing) -- feeding one OTIF_MISS "
        "endpoint. XGBoost remains the primary predictive model; this view is "
        "decision-support analysis, not a second decision engine."
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
    selected_id = st.selectbox("Inspect order", order_ids, index=default_index, key="causal-order")
    order = decisions.loc[decisions["order_id"].astype(str) == selected_id].iloc[0]
    pathway = _parse_pathway(order.get("causal_pathway"))
    attribution = _parse_json_list(order.get("causal_attribution_json"))
    scenarios = _parse_json_list(order.get("intervention_scenarios_json"))
    confidence = str(order.get("causal_confidence", pathway.get("confidence", "MEDIUM")))
    coverage = float(order.get("evidence_coverage", pathway.get("evidence_coverage", 0.0)))

    headline = st.columns(5)
    headline[0].metric("Bayesian posterior", f"{float(order.get('bbn_risk_score', 0)):.1%}")
    headline[1].metric("XGBoost risk", f"{float(order.get('xgb_risk_score', 0)):.1%}")
    headline[2].metric("Combined risk", f"{float(order.get('combined_risk_score', 0)):.1%}")
    headline[3].metric("Evidence coverage", f"{coverage:.0%}")
    headline[4].markdown(
        f'<div style="padding-top:1.6rem"><span class="status-strip" '
        f'style="background:{_confidence_badge_color(confidence)}">{confidence} '
        "CONFIDENCE</span></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Mechanism split: why an order misses OTIF")
    mechanism_columns = st.columns(2)
    mechanism_posteriors = pathway.get("mechanism_posteriors", {})
    late_probability = float(
        order.get("late_delivery_probability", mechanism_posteriors.get(LATE_DELIVERY, 0))
    )
    in_full_probability = float(
        order.get("in_full_failure_probability", mechanism_posteriors.get(IN_FULL_FAILURE, 0))
    )
    with mechanism_columns[0]:
        st.metric("P(Late delivery) -- timing failure", f"{late_probability:.1%}")
        st.progress(min(max(late_probability, 0.0), 1.0))
    with mechanism_columns[1]:
        st.metric("P(In-full failure) -- quantity failure", f"{in_full_probability:.1%}")
        st.progress(min(max(in_full_probability, 0.0), 1.0))

    st.subheader("Mechanism graph")
    scenario_labels = ["Baseline (no intervention)"] + [
        _scenario_label(scenario) for scenario in scenarios
    ]
    scenario_choice = st.selectbox(
        "Highlight a structural-intervention scenario",
        scenario_labels,
        key="causal-scenario",
    )
    chosen_scenario = (
        scenarios[scenario_labels.index(scenario_choice) - 1]
        if scenario_choice != "Baseline (no intervention)"
        else None
    )
    intervened_nodes = set(chosen_scenario["intervened_nodes"]) if chosen_scenario else set()
    st.markdown(_causal_graph_svg(pathway, intervened_nodes), unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Evidence attribution")
        st.caption(
            "Leave-one-evidence-out contribution -- not SHAP, not a causal effect estimate."
        )
        if attribution:
            attribution_table = pd.DataFrame(
                [
                    {
                        "node": row["node"],
                        "contribution": row["contribution"],
                        "direction": row["direction"],
                        "observed": row.get("observed", True),
                    }
                    for row in attribution
                ]
            )
            st.dataframe(
                attribution_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "contribution": st.column_config.NumberColumn(
                        "Contribution to posterior", format="%+.3f"
                    ),
                },
            )
        else:
            st.caption("No active evidence to attribute for this order.")
        st.write(
            f"**Route(s):** {_routes_text(pathway)}  \n"
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

    st.subheader("Structural intervention scenarios")
    st.markdown(
        '<span class="status-strip" style="background:#2f7d4f">Fixed-structure scenario '
        "analysis — not proven treatment effect</span>",
        unsafe_allow_html=True,
    )
    if scenarios:
        scenario_table = pd.DataFrame(
            [
                {
                    "scenario": _scenario_label(scenario),
                    "baseline_posterior": scenario["baseline_bayesian_posterior"],
                    "post_intervention_posterior": scenario["post_intervention_bayesian_posterior"],
                    "absolute_risk_reduction": scenario["absolute_risk_reduction"],
                    "relative_risk_reduction": scenario["relative_risk_reduction"],
                }
                for scenario in scenarios
            ]
        )
        st.dataframe(
            scenario_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "baseline_posterior": st.column_config.NumberColumn(
                    "Baseline Bayesian posterior", format="percent"
                ),
                "post_intervention_posterior": st.column_config.NumberColumn(
                    "Post-intervention posterior", format="percent"
                ),
                "absolute_risk_reduction": st.column_config.NumberColumn(
                    "Simulated absolute reduction", format="percent"
                ),
                "relative_risk_reduction": st.column_config.NumberColumn(
                    "Simulated relative reduction", format="percent"
                ),
            },
        )
        if chosen_scenario:
            st.info(_why_this_changed_text(chosen_scenario))
        st.caption(
            "Selecting a scenario only changes what is highlighted above -- it never "
            "recomputes the persisted XGBoost score, fused score, or decision status "
            "shown elsewhere in this app."
        )
    else:
        st.caption(
            "No active evidence on this order, so there is nothing to mitigate: no "
            "structural-intervention scenarios are computed."
        )

    st.subheader("Model diagnostics")
    diagnostics_columns = st.columns(4)
    if bbn_metrics:
        diagnostics_columns[0].metric(
            "Bayesian standalone Brier", f"{float(bbn_metrics.get('brier', 0)):.3f}"
        )
    mechanism_metrics = metrics.get("mechanism_metrics", {})
    late_mechanism = mechanism_metrics.get("late_delivery", {})
    in_full_mechanism = mechanism_metrics.get("in_full_failure", {})
    diagnostics_columns[1].metric(
        "Late-delivery mechanism PR-AUC", f"{float(late_mechanism.get('pr_auc', 0) or 0):.3f}"
    )
    diagnostics_columns[2].metric(
        "In-full mechanism PR-AUC", f"{float(in_full_mechanism.get('pr_auc', 0) or 0):.3f}"
    )
    confidence_diag = metrics.get("causal_confidence_diagnostics", {})
    diagnostics_columns[3].metric(
        "Low-confidence rate", f"{float(confidence_diag.get('low_confidence_rate', 0) or 0):.0%}"
    )
    consistency = metrics.get("causal_consistency", {})
    if consistency:
        attribution_agreement = float(consistency.get("top_attribution_vs_rule_cause", 0) or 0)
        intervention_agreement = float(
            consistency.get("top_intervention_vs_simulator_responsive_cause", 0) or 0
        )
        st.caption(
            f"Top-attribution vs. rule-derived cause agreement: {attribution_agreement:.0%} · "
            "Top-intervention vs. simulator-responsive cause agreement: "
            f"{intervention_agreement:.0%} (consistency diagnostics, not causal validation)."
        )


def _scenario_label(scenario: dict[str, Any]) -> str:
    nodes = ", ".join(str(node).replace("_", " ").title() for node in scenario["intervened_nodes"])
    reduction = float(scenario.get("absolute_risk_reduction", 0.0))
    kind = "Combined mitigation" if scenario.get("type") == "combined_mitigation" else "Mitigate"
    # Signed format (not a manual "-" prefix): a scenario can structurally
    # *increase* the modeled posterior (e.g. a screened-off or collider node),
    # and this must render as a genuine increase, not a double negative.
    return f"{kind}: {nodes} ({-reduction:+.0%} modeled risk)"


def _routes_text(pathway: dict[str, Any]) -> str:
    routes = pathway.get("routes") or ([pathway["route"]] if pathway.get("route") else [])
    if not routes:
        return "no active evidence route"
    return "; ".join(" → ".join(route) for route in routes)


def _why_this_changed_text(scenario: dict[str, Any]) -> str:
    nodes = ", ".join(str(node).replace("_", " ").title() for node in scenario["intervened_nodes"])
    baseline = float(scenario["baseline_bayesian_posterior"])
    post = float(scenario["post_intervention_bayesian_posterior"])
    absolute = float(scenario["absolute_risk_reduction"])
    relative = float(scenario["relative_risk_reduction"])
    actions = ", ".join(item["action"] for item in scenario.get("assumed_actions", []))
    return (
        f"Why this changed: mitigating {nodes} (assumed action: {actions or 'operational fix'}) "
        f"moves the Bayesian posterior from {baseline:.1%} to {post:.1%} -- a "
        f"{absolute:+.1%} absolute / {relative:.0%} relative reduction under this fixed "
        "network's assumptions. Fixed-structure scenario analysis — not a proven "
        "treatment effect; it does not change the XGBoost score, the fused score, or "
        "the operational decision."
    )


def _policy_value_view(artifacts_root: str) -> None:
    st.header("Policy value")
    path = _find_policy_benchmark_path(artifacts_root)
    if path is None:
        st.info(
            "No policy_benchmark.json found under this artifacts directory. Run "
            "`uv run python -m otif_risk.policy_benchmark` to generate one."
        )
        return
    payload = _load_json(path)
    summary = payload["summary"]
    gates = summary["acceptance_gates"]

    st.markdown(
        _badge(
            "Primary gate passed" if gates["primary_gate_passed"] else "Primary gate failed",
            "pass" if gates["primary_gate_passed"] else "fail",
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Seeds {payload['seeds']} ({payload['n_orders']} orders each). Headline metric: "
        "avoided penalty per normalized resource unit -- one resource unit is the fraction "
        "of a resource pool's daily capacity consumed, so a weak action type can never hide "
        "behind a strong one in the combined number (see per-resource breakdowns in the raw "
        "report). Primary capacity scenario: "
        f"{summary['primary_capacity_scenario']} (win threshold {gates['win_threshold']} of "
        f"{gates['n_seeds']} seeds)."
    )

    scenarios = summary["capacity_scenarios"]
    scenario_names = list(scenarios)
    default_index = (
        scenario_names.index(summary["primary_capacity_scenario"])
        if summary["primary_capacity_scenario"] in scenario_names
        else 0
    )
    def _capacity_scenario_label(name: str) -> str:
        return f"{name} ({int(round(scenarios[name] * 100))}% of default capacity)"

    selected_scenario = st.selectbox(
        "Capacity-stress scenario",
        scenario_names,
        index=default_index,
        format_func=_capacity_scenario_label,
    )
    if selected_scenario != summary["primary_capacity_scenario"]:
        st.caption(
            f"Diagnostic/sensitivity view -- the acceptance gates above are measured only at "
            f"{summary['primary_capacity_scenario']}."
        )

    headline = summary["median_headline_by_capacity_scenario"][selected_scenario]
    precision = summary["median_action_precision_by_capacity_scenario"][selected_scenario]
    coverage = summary["median_avoidable_miss_coverage_by_capacity_scenario"][selected_scenario]
    regret = summary["median_regret_vs_oracle_by_capacity_scenario"][selected_scenario]
    rows = [
        {
            "policy": policy,
            "avoided_penalty_per_resource_unit": headline.get(policy),
            "action_precision": precision.get(policy),
            "avoidable_miss_coverage": coverage.get(policy),
            "regret_vs_oracle": regret.get(policy),
            "evaluation_only": policy == "ORACLE_EVALUATION_ONLY",
        }
        for policy in headline
    ]
    table = pd.DataFrame(rows).sort_values(
        "avoided_penalty_per_resource_unit", ascending=False
    )
    st.subheader("Policies compared at this capacity scenario (median across seeds)")
    st.markdown(
        '<p class="oracle-note">ORACLE_EVALUATION_ONLY is an evaluation-only, unattainable '
        "ceiling used solely to compute regret -- never a deployable recommendation.</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Paired per-seed win/tie/loss (CURRENT_POLICY vs. baselines, primary capacity)")
    deltas = summary["paired_seed_deltas"]
    delta_rows = [
        {
            "comparison": label.replace("_", " "),
            "wins": entry["wins"],
            "ties": entry["ties"],
            "losses": entry["losses"],
            "per_seed_delta": entry["per_seed_delta"],
        }
        for label, entry in deltas.items()
    ]
    st.dataframe(pd.DataFrame(delta_rows), hide_index=True, use_container_width=True)

    st.subheader("Normal vs. drift regime value (CURRENT_POLICY, primary capacity)")
    regime = gates["current_policy_value_by_regime"]
    columns = st.columns(2)
    columns[0].metric("Normal regime", regime["normal"])
    columns[1].metric("Drift regime", regime["drift"])
    st.markdown(
        _badge(
            "Positive in both regimes"
            if gates["current_policy_value_positive_in_both_regimes"]
            else "Regressed in a regime",
            "pass" if gates["current_policy_value_positive_in_both_regimes"] else "fail",
        ),
        unsafe_allow_html=True,
    )

    diagnostics = summary["value_aware_policy_diagnostics"]
    st.subheader("Current vs. baseline: value-density explanation")
    st.markdown(
        "`value_density = (estimated_penalty_exposure × structural_reduction × "
        "execution_feasibility) / normalized_resource_fraction` -- CURRENT_POLICY ranks "
        "every eligible order/action candidate by this explainable density and resources the "
        "highest-density candidates first, with a 10% seeded-exploration carve-out. "
        "`structural_reduction` uses a persisted Bayesian do-operator scenario when available, "
        "else a fixed, documented fallback fraction."
    )
    mix_columns = st.columns(2)
    mix_columns[0].caption("Candidate action mix (median)")
    mix_columns[0].dataframe(
        pd.DataFrame(
            list(diagnostics["candidate_action_mix_median"].items()),
            columns=["action", "share"],
        ),
        hide_index=True,
        use_container_width=True,
    )
    mix_columns[1].caption("Chosen (capacity-accepted) action mix (median)")
    mix_columns[1].dataframe(
        pd.DataFrame(
            list(diagnostics["chosen_action_mix_median"].items()), columns=["action", "share"]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Bayesian ablation -- shown honestly")
    ablation = diagnostics["bayesian_ablation"]
    delta = ablation["median_delta_with_minus_without"] or 0.0
    st.markdown(
        _badge(
            "Bayesian term regresses measured policy value"
            if delta < 0
            else "Bayesian term adds value",
            "fail" if delta < 0 else "pass",
        ),
        unsafe_allow_html=True,
    )
    st.write(
        f"With Bayesian structural-reduction term: **{ablation['median_with_bayesian_term']}** · "
        f"without (leading-signal fallback only): **{ablation['median_without_bayesian_term']}** · "
        f"median delta: **{ablation['median_delta_with_minus_without']}** across "
        f"{ablation['n_seeds']} seeds ({ablation['seeds_where_bayesian_term_adds_value']} of "
        "which favored the Bayesian term)."
    )
    st.caption(
        "The Bayesian mechanism network still drives structured explanation and candidate "
        "action generation; this ablation shows adding its structural-reduction estimate to "
        "the deployed policy's ranking currently regresses measured policy value at 50% "
        "capacity on every benchmarked seed. Stage 2 governance holds a Bayesian-enhanced "
        "policy challenger built from this exact number rather than promoting it -- see the "
        "Governance tab's demo lifecycle scenario."
    )


def _governance_view(artifacts_root: str) -> None:
    st.header("Governance")
    ops_dir = _find_latest_ops_directory(artifacts_root)
    if ops_dir is None:
        st.info(
            "No completed operations replay found under this artifacts directory. Run "
            "`uv run python -m otif_risk.operations` to generate one."
        )
        return
    st.caption(f"Replay directory: {ops_dir.name}")

    st.subheader("Manifest trust card")
    manifest_path = ops_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        verification = verify_manifest(ops_dir)
        columns = st.columns(4)
        columns[0].metric("Git SHA", (manifest["git"]["sha"] or "unknown")[:10])
        columns[1].metric("Content ID", manifest["content_id"][:12])
        columns[2].metric("Files verified", verification.get("files_verified", 0))
        columns[3].metric(
            "Dirty working tree", "yes" if manifest["git"]["dirty"] else "no"
        )
        st.markdown(
            _badge(
                "Checksums verified" if verification["verified"] else "Verification FAILED",
                "pass" if verification["verified"] else "fail",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Feature schema hash: {manifest.get('feature_schema_hash') or 'n/a'} · "
            f"data/feature schema versions: {manifest.get('schema_versions')}"
        )
    else:
        st.warning("No run_manifest.json found for this replay.")

    st.subheader("Champion/challenger lifecycle timeline")
    registry_dir = ops_dir / "registry"
    events_path = registry_dir / "registry_events.jsonl"
    active_path = registry_dir / "active_model.json"
    if events_path.is_file():
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = []
        for event in events:
            if event["event"] == "ROLLED_BACK":
                reason = event.get("reason", "")
            else:
                reason = "; ".join(event.get("reasons", [])) or "all gates passed"
            rows.append(
                {
                    "event": event["event"],
                    "version": event["version_id"],
                    "timestamp_utc": event.get("timestamp_utc"),
                    "reason": reason,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if active_path.is_file():
            active = _load_json(active_path)
            st.markdown(
                f"**Active pointer:** `{active['active_version_id']}` "
                f"(previous: `{active.get('previous_version_id')}`, set by "
                f"{active['set_by_event']} at {active['updated_at_utc']})"
            )
    else:
        st.info("No governance lifecycle events recorded yet.")

    st.subheader("Champion/challenger metric deltas and gate statuses")
    versions_path = registry_dir / "registry_versions.json"
    if versions_path.is_file():
        versions = _load_json(versions_path)
        rows = []
        for version_id, version_payload in versions.items():
            metrics = version_payload["metrics"]
            rows.append(
                {
                    "version": version_id,
                    "note": version_payload.get("note"),
                    "pr_auc": metrics["pr_auc"],
                    "brier": metrics["brier"],
                    "calibration_error": metrics["calibration_error"],
                    "recall": metrics["recall"],
                    "alert_rate": metrics["alert_rate"],
                    "policy_value_50pct_capacity": metrics["policy_value_50pct_capacity"],
                    "manifest_verified": metrics["manifest_verified"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    demo_path = ops_dir / "demo_lifecycle_scenario.json"
    if demo_path.is_file():
        demo = _load_json(demo_path)
        st.subheader("Demo governance-lifecycle scenario (measured Stage 1 numbers)")
        if demo.get("enabled"):
            st.markdown(
                _badge("PROMOTED", "pass")
                + " value-aware CURRENT_POLICY over the legacy single-cause baseline"
                + "<br/>"
                + _badge("HELD", "fail")
                + " Bayesian-enhanced challenger -- measured policy-value regression"
                + "<br/>"
                + _badge("ROLLED BACK", "info")
                + f" active pointer restored to `{demo['rollback']['version_id']}`",
                unsafe_allow_html=True,
            )
            st.caption(
                "; ".join(demo["promotion_2_bayesian_enhanced_held"]["reasons"])
                or "all gates passed"
            )
        else:
            st.info(demo.get("reason", "demo lifecycle scenario not available"))

    st.subheader("Decision ledger and observational outcome cohorts")
    ledger_path = ops_dir / "decision_ledger.csv"
    if ledger_path.is_file():
        ledger = pd.read_csv(ledger_path)
        st.caption(f"{len(ledger):,} decisions logged across the replay.")
        st.dataframe(
            ledger.head(100)[
                [
                    column
                    for column in (
                        "decision_id",
                        "order_id",
                        "decision_timestamp",
                        "model_version",
                        "chosen_action",
                        "planner_decision",
                        "execution_status",
                        "matured",
                        "matured_otif_miss",
                        "realized_penalty",
                    )
                    if column in ledger.columns
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )
    cohort_path = ops_dir / "observational_cohort_report.json"
    if cohort_path.is_file():
        cohort = _load_json(cohort_path)
        st.markdown(_badge("Observational -- not causal", "warn"), unsafe_allow_html=True)
        st.caption(cohort["qualification"])
        cohort_rows = [
            {"cohort": name, **stats} for name, stats in cohort["cohorts"].items()
        ]
        st.dataframe(pd.DataFrame(cohort_rows), hide_index=True, use_container_width=True)

    intervention_outcomes_path = ops_dir / "intervention_outcomes.json"
    if intervention_outcomes_path.is_file():
        intervention_outcomes = _load_json(intervention_outcomes_path)
        st.subheader("Realized outcomes by intervention type")
        st.markdown(_badge("Observational -- not causal", "warn"), unsafe_allow_html=True)
        st.caption(intervention_outcomes["qualification"])
        action_rows = [
            {"intervention_type": name, **stats}
            for name, stats in intervention_outcomes["outcomes_by_intervention_type"].items()
        ]
        action_rows.append(
            {
                "intervention_type": "NO_INTERVENTION_BASELINE",
                **intervention_outcomes["no_intervention_baseline"],
            }
        )
        st.dataframe(pd.DataFrame(action_rows), hide_index=True, use_container_width=True)

    st.subheader("Monitoring and SLO status")
    monitoring_path = ops_dir / "monitoring_report.json"
    if monitoring_path.is_file():
        monitoring = _load_json(monitoring_path)
        slo_rows = [
            {"slo": name, **stats} for name, stats in monitoring["slo_status"].items()
        ]
        st.dataframe(pd.DataFrame(slo_rows), hide_index=True, use_container_width=True)
        st.caption(monitoring["runtime"]["scope"])
        regime = monitoring.get("regime_quality", {})
        if regime:

            def _regime_summary(name: str, stats: dict[str, Any]) -> str:
                if stats.get("sufficient_sample"):
                    return f"{name}: n={stats['n_matured_decisions']}, pr_auc={stats.get('pr_auc')}"
                return f"{name}: n={stats['n_matured_decisions']} (insufficient sample)"

            st.caption(
                "Regime quality (observational, matured decisions): "
                + "; ".join(_regime_summary(name, stats) for name, stats in regime.items())
            )
    else:
        st.info("No monitoring_report.json found for this replay.")

    st.subheader("Offline/batch parity status")
    try:
        canonical_run_dir = latest_run_directory(artifacts_root)
        parity_path = canonical_run_dir / "parity_check.json"
    except FileNotFoundError:
        parity_path = None
    if parity_path is not None and parity_path.is_file():
        parity = _load_json(parity_path)
        parity_label = "Parity verified" if parity["passed"] else "Parity FAILED"
        st.markdown(
            _badge(parity_label, "pass" if parity["passed"] else "fail"),
            unsafe_allow_html=True,
        )
        st.caption(
            f"{parity['n_orders_checked']} orders checked at as-of "
            f"{parity['as_of_timestamp']} -- {parity['qualification']}"
        )
    else:
        st.info("No parity_check.json found for the latest canonical pipeline run.")


# ==========================================================================
# AI Copilot view (read-only, grounded, cited explanation layer).
# ==========================================================================


def _truncate_badge_value(value: Any, limit: int = 90) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, default=str)
    else:
        text = str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _render_citation_badges(citation_ids: list[str], packet: EvidencePacket) -> None:
    if not citation_ids:
        return
    badges = []
    for citation_id in citation_ids:
        fact = packet.get(citation_id)
        if fact is not None:
            value_text = _truncate_badge_value(fact.value)
            badges.append(_badge(f"{citation_id} · {fact.label} = {value_text}", "info"))
        else:
            badges.append(_badge(f"{citation_id} · unresolved", "warn"))
    st.markdown(" ".join(badges), unsafe_allow_html=True)


def _render_cited_list(title: str, items: list[dict[str, Any]], packet: EvidencePacket) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        if not isinstance(item, dict):
            continue
        st.write(f"- {item.get('text', '')}")
        _render_citation_badges(item.get("citations", []) or [], packet)


def _render_copilot_answer(question_label: str, answer: CopilotAnswer) -> None:
    packet = answer.packet
    response = answer.response
    mode_kind = "pass" if answer.mode_used == "live" else "info"
    validation_kind = "pass" if answer.validation_status == "passed" else "fail"
    st.markdown(f"##### Q: {question_label}")
    st.markdown(
        _badge(f"{answer.mode_used.upper()} · {answer.provider}", mode_kind)
        + " "
        + _badge(f"validation: {answer.validation_status}", validation_kind)
        + (f" {_badge(answer.model, 'info')}" if answer.model else ""),
        unsafe_allow_html=True,
    )
    st.markdown(f"**{response.get('headline', '')}**")
    for item in response.get("what_happened", []) or []:
        st.write(f"- {item}")
    _render_cited_list("Why flagged", response.get("why_flagged", []), packet)
    _render_cited_list("Affected items", response.get("affected_items", []), packet)
    next_step = response.get("recommended_next_step") or {}
    st.markdown("**Recommended next step**")
    st.write(next_step.get("text", ""))
    _render_citation_badges(next_step.get("citations", []) or [], packet)
    if next_step.get("preserves_persisted_decision") is not True:
        st.error(
            "This response did not preserve the persisted decision and was rejected by the "
            "validator; the deterministic fallback is shown instead."
        )
    _render_cited_list("Uncertainties", response.get("uncertainties", []), packet)
    if response.get("draft_message"):
        st.markdown("**Draft message** (copy manually -- never sent or executed automatically)")
        st.code(response["draft_message"], language=None)
    st.caption(response.get("disclaimer", ""))
    if answer.mode_used == "fallback" and answer.fallback_reason:
        st.caption(f"Fallback reason: {answer.fallback_reason}")
    st.caption(
        f"Evidence hash {answer.packet.evidence_hash()[:16]}\u2026 · "
        f"{len(packet.facts)} cited facts · latency {answer.latency_ms:.0f} ms"
    )


def _copilot_mode_badge() -> None:
    mode_configured = os.environ.get("OTIF_LLM_MODE", "auto").strip().lower() or "auto"
    live_ready = is_live_configured()
    if live_ready:
        st.markdown(
            _badge(f"Live OpenAI ready \u00b7 mode={mode_configured}", "pass"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _badge(f"Deterministic fallback \u00b7 mode={mode_configured} \u00b7 no OPENAI_API_KEY", "info"),
            unsafe_allow_html=True,
        )
    st.caption(
        "The Copilot only explains, answers questions, and drafts text from the evidence below -- "
        "it never changes a score, threshold, decision, or resource allocation."
    )


def _order_copilot_tab(decisions: pd.DataFrame, metrics: dict[str, Any], run_directory: str) -> None:
    manifest = _load_run_manifest(run_directory)
    order_ids = decisions["order_id"].astype(str).tolist()
    selected_id = st.selectbox("Order ID", order_ids, key="copilot_order_select")
    order_row = decisions.loc[decisions["order_id"].astype(str) == selected_id].iloc[0]
    packet = build_order_evidence_packet(order_row.to_dict(), metrics=metrics, manifest=manifest)

    columns = st.columns(3)
    columns[0].metric("Decision status", str(order_row.get("decision_status", "n/a")))
    columns[1].metric("Combined risk", f"{float(order_row.get('combined_risk_score', 0.0)):.1%}")
    columns[2].metric("Cited facts available", len(packet.facts))

    with st.expander("Evidence packet preview (facts the Copilot may cite)"):
        st.caption(
            "Allowlisted, deterministically built, and size-limited -- the same facts back both "
            "live and fallback answers below."
        )
        st.json(packet.to_dict())

    question_label_to_id = {label: qid for qid, label in ORDER_QUESTIONS.items()}
    question_label = st.selectbox(
        "Ask the copilot", list(question_label_to_id.keys()), key="copilot_order_question"
    )
    question_id = question_label_to_id[question_label]

    history_key = f"copilot_history_{selected_id}"
    if st.button("Ask", key="copilot_order_ask", type="primary"):
        answer = get_order_copilot_response(
            order_row.to_dict(),
            question_id,
            metrics=metrics,
            manifest=manifest,
            run_directory=run_directory,
        )
        st.session_state.setdefault(history_key, [])
        st.session_state[history_key].append({"question": question_label, "answer": answer})

    history = st.session_state.get(history_key, [])
    if not history:
        st.info("Ask a question above to see a grounded, cited answer scoped to this order.")
        return
    st.caption(f"{len(history)} question(s) asked for order {selected_id} in this session.")
    for turn in reversed(history):
        _render_copilot_answer(turn["question"], turn["answer"])
        st.divider()


def _portfolio_copilot_tab(decisions: pd.DataFrame, metrics: dict[str, Any], run_directory: str) -> None:
    st.caption(
        "Fixed question catalog only -- no unrestricted DataFrame/SQL access. Each question maps "
        "to one reviewable, deterministic aggregation."
    )
    question_label_to_id = {label: qid for qid, label in PORTFOLIO_QUESTIONS.items()}
    question_label = st.selectbox(
        "Portfolio question", list(question_label_to_id.keys()), key="copilot_portfolio_question"
    )
    question_id = question_label_to_id[question_label]

    if st.button("Ask portfolio copilot", key="copilot_portfolio_ask", type="primary"):
        answer = get_portfolio_copilot_response(
            question_id, decisions, metrics=metrics, run_directory=run_directory
        )
        st.session_state["copilot_portfolio_last"] = {"question": question_label, "answer": answer}

    turn = st.session_state.get("copilot_portfolio_last")
    if turn is None:
        st.info("Ask a portfolio question above.")
        return
    _render_copilot_answer(turn["question"], turn["answer"])


def _copilot_health_card(run_directory: str) -> None:
    st.subheader("Copilot health")
    records = read_audit_records(default_audit_path(run_directory))
    if not records:
        st.caption("No Copilot requests recorded yet for this run.")
        return
    total = len(records)
    live_count = sum(1 for record in records if record.get("mode_used") == "live")
    fallback_count = total - live_count
    passed = sum(1 for record in records if record.get("validation_status") == "passed")
    latencies = [
        record.get("latency_ms") for record in records if isinstance(record.get("latency_ms"), (int, float))
    ]
    median_latency = statistics.median(latencies) if latencies else None
    total_tokens = sum(
        (record.get("input_tokens") or 0) + (record.get("output_tokens") or 0) for record in records
    )
    columns = st.columns(5)
    columns[0].metric("Total requests", total)
    columns[1].metric("Live", live_count)
    columns[2].metric("Fallback", fallback_count)
    columns[3].metric("Validation pass", f"{passed}/{total}")
    columns[4].metric(
        "Median latency (ms)", f"{median_latency:.0f}" if median_latency is not None else "n/a"
    )
    st.caption(
        f"Estimated token usage (when reported by the provider): {total_tokens:,}. "
        "The validator rejects any response that fails to preserve the persisted decision, so no "
        "unsupported decision override can reach this log."
    )
    with st.expander("Recent Copilot audit entries"):
        st.dataframe(pd.DataFrame(records[-25:]), hide_index=True, use_container_width=True)


def _ai_copilot_view(decisions: pd.DataFrame, metrics: dict[str, Any], run_directory: str) -> None:
    st.header("AI Copilot")
    st.caption(
        "Read-only planning copilot: explains, drafts, and cites evidence -- it never decides. "
        "The order decision, governance, and operations views above remain the authoritative "
        "operational surfaces."
    )
    _copilot_mode_badge()
    tabs = st.tabs(["Order Copilot", "Portfolio Copilot"])
    with tabs[0]:
        _order_copilot_tab(decisions, metrics, run_directory)
    with tabs[1]:
        _portfolio_copilot_tab(decisions, metrics, run_directory)
    st.divider()
    _copilot_health_card(run_directory)


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
                "Causal intelligence",
                "Policy value",
                "Governance",
                "AI Copilot",
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
    elif view == "Causal intelligence":
        _causal_intelligence_view(decisions, metrics)
    elif view == "Policy value":
        _policy_value_view(str(root.resolve()))
    elif view == "Governance":
        _governance_view(str(root.resolve()))
    elif view == "AI Copilot":
        _ai_copilot_view(decisions, metrics, run_directory)
    else:
        _model_health_view(str(root.resolve()), metrics)


if __name__ == "__main__":
    main()
