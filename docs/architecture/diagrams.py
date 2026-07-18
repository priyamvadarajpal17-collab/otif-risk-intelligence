"""Deterministic, checked-in SVG renderer for the OTIF architecture diagrams.

This module is the single source of truth for the *visual* (SVG) rendering of
the two canonical Mermaid diagrams (``current.mmd`` / ``target.mmd``). It does
not parse Mermaid; instead each diagram's nodes/edges/bands are declared once,
in a small typed data model, and rendered directly to SVG using nothing but
the Python standard library (no network access, no external CLI, no Mermaid
tooling). Keeping the node/edge lists here in the same order and grouping as
the ``.mmd`` sources keeps the two representations easy to eyeball for drift.

Visual language ("industrial operations room"):
  - warm paper background (``PAPER``) with a faint charcoal grid, evoking a
    control-room whiteboard rather than a generic gradient/cloud diagram;
  - charcoal (``INK``) for structural boxes and default connectors;
  - signal blue (``SIGNAL_BLUE``) for the statistical/model path (XGBoost,
    Bayesian chain, calibration, SHAP);
  - safety orange (``SIGNAL_ORANGE``) for the intervention/decision path
    (fusion, threshold, resource policy, unified intervention record);
  - subgraphs ("bands") are drawn as labelled dashed containers, matching the
    Mermaid ``subgraph`` blocks.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# --- Palette -----------------------------------------------------------------
PAPER = "#f2ede1"
INK = "#20211d"
STEEL = "#5b5f57"
SIGNAL_ORANGE = "#d8531f"
SIGNAL_BLUE = "#2b5b8c"
PAPER_PANEL = "#faf6ec"
GRID_LINE = "#20211d14"

# --- Grid geometry -------------------------------------------------------------
COL_WIDTH = 236
ROW_HEIGHT = 92
NODE_WIDTH = 208
NODE_HEIGHT = 62
MARGIN = 48


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    col: float
    row: float
    kind: str = "data"  # data | model | decision | product | loop
    col_span: float = 1.0


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str = "flow"  # flow | model | intervention | loop
    label: str = ""


@dataclass(frozen=True)
class Band:
    label: str
    node_ids: tuple[str, ...]
    kind: str = "twin"  # twin | loop


@dataclass(frozen=True)
class Diagram:
    title: str
    subtitle: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    bands: tuple[Band, ...] = field(default_factory=tuple)


KIND_STYLE = {
    "data": {"fill": PAPER_PANEL, "stroke": INK, "text": INK},
    "model": {"fill": "#e7eef5", "stroke": SIGNAL_BLUE, "text": INK},
    "decision": {"fill": "#f7e6da", "stroke": SIGNAL_ORANGE, "text": INK},
    "product": {"fill": PAPER_PANEL, "stroke": INK, "text": INK},
    "loop": {"fill": "#efe9d8", "stroke": STEEL, "text": INK},
}

EDGE_STYLE = {
    "flow": {"stroke": INK, "width": 2},
    "model": {"stroke": SIGNAL_BLUE, "width": 2.4},
    "intervention": {"stroke": SIGNAL_ORANGE, "width": 2.6},
    "loop": {"stroke": STEEL, "width": 2},
}


def _node_center(node: Node) -> tuple[float, float]:
    x = MARGIN + node.col * COL_WIDTH + (node.col_span * COL_WIDTH) / 2
    y = MARGIN + node.row * ROW_HEIGHT + NODE_HEIGHT / 2
    return x, y


def _node_box(node: Node) -> tuple[float, float, float, float]:
    x = MARGIN + node.col * COL_WIDTH
    y = MARGIN + node.row * ROW_HEIGHT
    width = node.col_span * COL_WIDTH - 24
    return x, y, width, NODE_HEIGHT


def _wrap_label(label: str, width_chars: int = 24) -> list[str]:
    return textwrap.wrap(label, width=width_chars, break_long_words=False) or [label]


def _render_node(node: Node) -> str:
    style = KIND_STYLE[node.kind]
    x, y, width, height = _node_box(node)
    lines = _wrap_label(node.label)
    line_height = 15
    text_start_y = y + height / 2 - (len(lines) - 1) * line_height / 2 + 5
    text_svg = "\n".join(
        f'    <tspan x="{x + width / 2:.1f}" y="{text_start_y + index * line_height:.1f}">'
        f"{_escape(line)}</tspan>"
        for index, line in enumerate(lines)
    )
    return (
        f'  <g class="node node-{node.kind}">\n'
        f'    <rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="6" '
        f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="1.6"/>\n'
        f'    <text text-anchor="middle" font-family="IBM Plex Sans, Helvetica, Arial, sans-serif" '
        f'font-size="12.5" fill="{style["text"]}">\n{text_svg}\n    </text>\n'
        "  </g>"
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _render_edge(edge: Edge, nodes: dict[str, Node]) -> str:
    style = EDGE_STYLE[edge.kind]
    source, target = nodes[edge.source], nodes[edge.target]
    sx, sy = _node_center(source)
    tx, ty = _node_center(target)
    sbx, sby, sbw, sbh = _node_box(source)
    tbx, tby, tbw, tbh = _node_box(target)
    # Route from the closer edge of each box (below if lower row, else side).
    if abs(target.row - source.row) >= abs(target.col - source.col):
        start = (sx, sby + sbh if target.row > source.row else sby)
        end = (tx, tby if target.row > source.row else tby + tbh)
    else:
        start = (sbx + sbw if target.col > source.col else sbx, sy)
        end = (tbx if target.col > source.col else tbx + tbw, ty)
    marker = f"url(#arrow-{edge.kind})"
    dash = ' stroke-dasharray="6,4"' if edge.kind == "loop" else ""
    path = (
        f'  <path d="M {start[0]:.1f} {start[1]:.1f} L {end[0]:.1f} {end[1]:.1f}" '
        f'stroke="{style["stroke"]}" stroke-width="{style["width"]}" fill="none" '
        f'marker-end="{marker}"{dash}/>'
    )
    if not edge.label:
        return path
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 6
    label = (
        f'  <text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" '
        f'font-family="IBM Plex Sans, Helvetica, Arial, sans-serif" font-size="10.5" '
        f'font-style="italic" fill="{STEEL}">{_escape(edge.label)}</text>'
    )
    return f"{path}\n{label}"


def _render_band(band: Band, nodes: dict[str, Node]) -> str:
    boxes = [_node_box(nodes[node_id]) for node_id in band.node_ids]
    min_x = min(box[0] for box in boxes) - 16
    min_y = min(box[1] for box in boxes) - 26
    max_x = max(box[0] + box[2] for box in boxes) + 16
    max_y = max(box[1] + box[3] for box in boxes) + 16
    stroke = STEEL if band.kind == "loop" else SIGNAL_BLUE
    return (
        f'  <g class="band">\n'
        f'    <rect x="{min_x:.1f}" y="{min_y:.1f}" width="{max_x - min_x:.1f}" '
        f'height="{max_y - min_y:.1f}" rx="10" fill="none" stroke="{stroke}" '
        f'stroke-width="1.4" stroke-dasharray="3,5"/>\n'
        f'    <text x="{min_x + 12:.1f}" y="{min_y + 16:.1f}" '
        f'font-family="Barlow Condensed, Helvetica, Arial, sans-serif" font-size="13" '
        f'letter-spacing="0.06em" fill="{stroke}">{_escape(band.label.upper())}</text>\n'
        "  </g>"
    )


def render_svg(diagram: Diagram) -> str:
    """Render ``diagram`` to a self-contained, deterministic SVG document."""
    nodes = {node.id: node for node in diagram.nodes}
    max_col = max(node.col + node.col_span for node in diagram.nodes)
    max_row = max(node.row for node in diagram.nodes) + 1
    width = MARGIN * 2 + max_col * COL_WIDTH
    height = MARGIN * 2 + max_row * ROW_HEIGHT + 64

    grid_lines = []
    for gx in range(0, int(width), 28):
        grid_lines.append(
            f'<line x1="{gx}" y1="0" x2="{gx}" y2="{height:.0f}" stroke="{GRID_LINE}"/>'
        )
    for gy in range(0, int(height), 28):
        grid_lines.append(
            f'<line x1="0" y1="{gy}" x2="{width:.0f}" y2="{gy}" stroke="{GRID_LINE}"/>'
        )

    bands_svg = "\n".join(_render_band(band, nodes) for band in diagram.bands)
    edges_svg = "\n".join(_render_edge(edge, nodes) for edge in diagram.edges)
    nodes_svg = "\n".join(_render_node(node) for node in diagram.nodes)

    markers = "\n".join(
        f'    <marker id="arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">\n'
        f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{style["stroke"]}"/>\n'
        "    </marker>"
        for kind, style in EDGE_STYLE.items()
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" \
role="img" aria-labelledby="title desc" font-family="IBM Plex Sans, Helvetica, Arial, sans-serif">
  <title id="title">{_escape(diagram.title)}</title>
  <desc id="desc">{_escape(diagram.subtitle)}</desc>
  <defs>
{markers}
  </defs>
  <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="{PAPER}"/>
  <g opacity="0.55">
{chr(10).join(grid_lines)}
  </g>
  <text x="{MARGIN}" y="28" font-family="Barlow Condensed, Helvetica, Arial, sans-serif" \
font-size="21" letter-spacing="0.04em" fill="{INK}">{_escape(diagram.title.upper())}</text>
  <text x="{MARGIN}" y="46" font-size="11.5" fill="{STEEL}">{_escape(diagram.subtitle)}</text>
  <g transform="translate(0, 20)">
{bands_svg}
{edges_svg}
{nodes_svg}
  </g>
</svg>
"""


# ---------------------------------------------------------------------------
# Diagram definitions (mirrors current.mmd / target.mmd)
# ---------------------------------------------------------------------------

CURRENT_DIAGRAM = Diagram(
    title="Current architecture",
    subtitle=(
        "Shipped realistic-OTIF-demo pipeline: noisy digital twin, compact 8-node "
        "causal Bayesian chain (one OTIF_MISS endpoint), evidence-based fusion, "
        "resource-aware ops loop."
    ),
    nodes=(
        Node("A1", "Stable vendor/SKU/lane/DC/customer traits", 0, 0, "loop"),
        Node("A2", "Seasonality + correlated shocks", 1, 0, "loop"),
        Node("A3", "Partial observability + noise", 2, 0, "loop"),
        Node("A4", "Line + order lifecycle simulation", 1, 1, "loop"),
        Node("A5", "Ground-truth causes, outcomes, response", 1, 2, "loop"),
        Node("B", "Data contracts + quality report", 1, 3, "data"),
        Node("C1", "Retrospective multi-cause derivation", 0.4, 4, "data"),
        Node("C2", "Point-in-time feature snapshots", 1.9, 4, "data"),
        Node("C3", "Cause + line-evidence truth", 0.4, 5, "data"),
        Node("D", "Rolling-origin train/validation/test", 1.9, 5, "data"),
        Node("E1", "XGBoost order-risk model", 1.4, 6, "model"),
        Node("F1", "Compact causal Bayesian chain (8 nodes, 1 endpoint)", 2.4, 6, "model"),
        Node("E2", "Calibration + SHAP", 1.4, 7, "model"),
        Node("F2", "Exact posterior + pathway", 2.4, 7, "model"),
        Node("G", "Validated score fusion", 1.9, 8, "decision"),
        Node("H", "Capacity-aware operating threshold", 1.9, 9, "decision"),
        Node("I1", "Affected-SKU evidence", 0.2, 10, "decision"),
        Node("I2", "Order explanation", 1.4, 10, "decision"),
        Node("I3", "Resource-aware intervention policy", 2.6, 10, "decision"),
        Node("J", "Unified intervention record", 1.4, 11, "decision"),
        Node("K1", "Order desk", 0.2, 12, "product"),
        Node("K2", "Portfolio + hotspot views", 1.4, 12, "product"),
        Node("K3", "Model health + Bayesian network view", 2.6, 12, "product"),
        Node("L1", "Daily open-order scoring", 0.2, 13.3, "loop"),
        Node("L2", "Orders close", 1.1, 13.3, "loop"),
        Node("L3", "Outcomes + derived causes", 2.0, 13.3, "loop"),
        Node("L4", "Feedback + drift + performance", 0.6, 14.3, "loop"),
        Node("L5", "Versioned retraining", 1.6, 14.3, "loop"),
    ),
    edges=(
        Edge("A1", "A4"),
        Edge("A2", "A4"),
        Edge("A3", "A4"),
        Edge("A4", "A5"),
        Edge("A5", "B"),
        Edge("B", "C1"),
        Edge("B", "C2"),
        Edge("C1", "C3"),
        Edge("C2", "D"),
        Edge("D", "E1", "model"),
        Edge("E1", "E2", "model"),
        Edge("D", "F1", "model"),
        Edge("F1", "F2", "model"),
        Edge("E1", "G", "model"),
        Edge("F2", "G", "model"),
        Edge("G", "H", "intervention"),
        Edge("C3", "I1", "intervention"),
        Edge("E2", "I2", "intervention"),
        Edge("F2", "I2", "intervention"),
        Edge("H", "I3", "intervention"),
        Edge("I1", "J", "intervention"),
        Edge("I2", "J", "intervention"),
        Edge("I3", "J", "intervention"),
        Edge("J", "K1"),
        Edge("J", "K2"),
        Edge("J", "K3"),
        Edge("J", "L1"),
        Edge("L1", "L2", "loop"),
        Edge("L2", "L3", "loop"),
        Edge("L3", "L4", "loop"),
        Edge("L4", "L5", "loop", label="scheduled or triggered"),
        Edge("L5", "L1", "loop"),
    ),
    bands=(
        Band("Noisy supply-chain digital twin", ("A1", "A2", "A3", "A4", "A5"), "twin"),
        Band("Local operating-loop simulation", ("L1", "L2", "L3", "L4", "L5"), "loop"),
    ),
)

TARGET_DIAGRAM = Diagram(
    title="Target architecture",
    subtitle=(
        "Causal Intelligence Studio -- a 10-node mechanism Bayesian network "
        "(IN_FULL_FAILURE / LATE_DELIVERY -> OTIF_MISS) plus exact structural "
        "do-operator scenarios, decision-support only -- and a Decision Value Lab that "
        "measures simulated policy value against an evaluation-only oracle ceiling."
    ),
    nodes=(
        Node("A1", "Stable vendor/SKU/lane/DC/customer traits", 0, 0, "loop"),
        Node("A2", "Seasonality + correlated shocks", 1, 0, "loop"),
        Node("A3", "Partial observability + noise", 2, 0, "loop"),
        Node("A4", "Line + order lifecycle simulation", 1, 1, "loop"),
        Node("A5", "Ground-truth causes, outcomes, response", 1, 2, "loop"),
        Node("B", "Data contracts + quality report", 1, 3, "data"),
        Node("C1", "Retrospective multi-cause derivation", 0.4, 4, "data"),
        Node("C2", "Point-in-time feature snapshots", 1.9, 4, "data"),
        Node("C3", "Cause + line-evidence truth", 0.4, 5, "data"),
        Node("D", "Rolling-origin train/validation/test", 1.9, 5, "data"),
        Node("E1", "XGBoost order-risk model", 1.4, 6, "model"),
        Node(
            "F1",
            "10-node mechanism Bayesian network (IN_FULL_FAILURE / LATE_DELIVERY -> OTIF_MISS)",
            2.5,
            6,
            "model",
            col_span=1.3,
        ),
        Node("E2", "Calibration + SHAP", 1.4, 7, "model"),
        Node("F2", "Exact posterior + mechanism split + attribution", 2.2, 7, "model"),
        Node("F3", "Exact do-operator intervention scenarios", 3.4, 7, "model"),
        Node("G", "Validated score fusion", 1.9, 8, "decision"),
        Node("H", "Capacity-aware operating threshold", 1.9, 9, "decision"),
        Node("I1", "Affected-SKU evidence", 0.2, 10, "decision"),
        Node("I2", "Order explanation", 1.4, 10, "decision"),
        Node("I3", "Resource-aware intervention policy", 2.6, 10, "decision"),
        Node("J", "Unified intervention record", 1.4, 11, "decision"),
        Node("K1", "Order desk", 0.2, 12, "product"),
        Node("K2", "Portfolio + hotspot views", 1.4, 12, "product"),
        Node(
            "K3",
            "Causal Intelligence Studio (mechanism graph, attribution, scenarios)",
            2.6,
            12,
            "product",
        ),
        Node("L1", "Daily open-order scoring", 0.2, 13.3, "loop"),
        Node("L2", "Orders close", 1.1, 13.3, "loop"),
        Node("L3", "Outcomes + derived causes", 2.0, 13.3, "loop"),
        Node("L4", "Feedback + drift + performance", 0.6, 14.3, "loop"),
        Node("L5", "Versioned retraining", 1.6, 14.3, "loop"),
        Node("M1", "Heterogeneous action-response twin (evaluation-only)", 0.2, 15.6, "decision"),
        Node("M2", "7-policy capacity-constrained evaluation", 1.5, 15.6, "decision", col_span=1.2),
        Node("M3", "Oracle regret + counterfactual action ranking", 2.9, 15.6, "decision"),
    ),
    edges=(
        Edge("A1", "A4"),
        Edge("A2", "A4"),
        Edge("A3", "A4"),
        Edge("A4", "A5"),
        Edge("A5", "B"),
        Edge("B", "C1"),
        Edge("B", "C2"),
        Edge("C1", "C3"),
        Edge("C2", "D"),
        Edge("D", "E1", "model"),
        Edge("E1", "E2", "model"),
        Edge("D", "F1", "model"),
        Edge("F1", "F2", "model"),
        Edge("F1", "F3", "model"),
        Edge("E1", "G", "model"),
        Edge("F2", "G", "model"),
        Edge("G", "H", "intervention"),
        Edge("C3", "I1", "intervention"),
        Edge("E2", "I2", "intervention"),
        Edge("F2", "I2", "intervention"),
        Edge("H", "I3", "intervention"),
        Edge("I1", "J", "intervention"),
        Edge("I2", "J", "intervention"),
        Edge("I3", "J", "intervention"),
        Edge("J", "K1"),
        Edge("J", "K2"),
        Edge("J", "K3"),
        Edge("F3", "K3", "loop", label="decision-support only, never feeds G/H"),
        Edge("J", "L1"),
        Edge("L1", "L2", "loop"),
        Edge("L2", "L3", "loop"),
        Edge("L3", "L4", "loop"),
        Edge("L4", "L5", "loop", label="scheduled or triggered"),
        Edge("L5", "L1", "loop"),
        Edge("A5", "M1", "loop", label="evaluation-only common random numbers"),
        Edge("I3", "M2", "loop", label="deployed policy under evaluation"),
        Edge("M1", "M2", "loop"),
        Edge("F3", "M3", "loop", label="model-scenario vs. simulator-evaluation"),
        Edge("M2", "M3", "loop"),
    ),
    bands=(
        Band("Noisy supply-chain digital twin", ("A1", "A2", "A3", "A4", "A5"), "twin"),
        Band("Local operating-loop simulation", ("L1", "L2", "L3", "L4", "L5"), "loop"),
        Band(
            "Decision Value Lab (Stage 1, evaluation-only, never feeds G/H/J)",
            ("M1", "M2", "M3"),
            "loop",
        ),
    ),
)


def write_diagram(diagram: Diagram, path: Path) -> None:
    path.write_text(render_svg(diagram), encoding="utf-8")
