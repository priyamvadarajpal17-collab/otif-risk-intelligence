"""Shared contracts for the OTIF risk intelligence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CAUSE_CATEGORIES = (
    "ORDER_CAPTURE",
    "VENDOR_FAILURE",
    "INVENTORY_SHORTAGE",
    "DC_CAPACITY",
    "WAREHOUSE_OPS",
    "TRANSPORT",
    "CUSTOMER_DELIVERY",
)


#: Reserved slice of orders (per config) deterministically scripted into named
#: demonstration scenarios (see ``data.py``'s ``SCENARIO_*`` constants). These
#: orders always exist regardless of ``seed``; the remaining orders stay fully
#: seed-random so the overall benchmark is not manufactured.
DEFAULT_SCENARIO_ORDER_COUNT = 5


@dataclass(frozen=True)
class PrototypeConfig:
    seed: int = 42
    n_orders: int = 2_500
    start_date: str = "2024-01-01"
    prediction_horizon_days: int = 7
    planner_capacity_fraction: float = 0.15
    threshold_strategy: str = "recall_floor"
    target_recall: float = 0.65
    min_precision: float = 0.30
    scenario_order_count: int = DEFAULT_SCENARIO_ORDER_COUNT
    output_dir: Path = Path("artifacts")

    def __post_init__(self) -> None:
        if self.n_orders < 200:
            raise ValueError("n_orders must be at least 200")
        if self.prediction_horizon_days < 1:
            raise ValueError("prediction_horizon_days must be positive")
        if not 0 < self.planner_capacity_fraction < 1:
            raise ValueError("planner_capacity_fraction must be in (0, 1)")
        if self.threshold_strategy not in {"capacity", "recall_floor", "f1_max"}:
            raise ValueError("threshold_strategy must be capacity, recall_floor, or f1_max")
        if not 0 < self.target_recall <= 1:
            raise ValueError("target_recall must be in (0, 1]")
        if not 0 <= self.min_precision <= 1:
            raise ValueError("min_precision must be in [0, 1]")
        if self.scenario_order_count < 0 or self.scenario_order_count >= self.n_orders:
            raise ValueError("scenario_order_count must be in [0, n_orders)")


@dataclass
class PrototypeDataset:
    orders: pd.DataFrame
    order_lines: pd.DataFrame
    events: pd.DataFrame
    vendors: pd.DataFrame
    dcs: pd.DataFrame
    lanes: pd.DataFrame
    customers: pd.DataFrame
    skus: pd.DataFrame
    capacity_snapshots: pd.DataFrame
    #: Evaluation-only ground truth (latent shocks, accumulated delay/shortfall,
    #: intervention responsiveness). Never merged into model feature tables.
    simulator_truth: pd.DataFrame
    #: Evaluation-only per-line ground truth (which lines a shock actually hit).
    line_truth: pd.DataFrame
    #: Correlated disruption shocks applied to vendors/DCs/lanes during generation.
    shocks: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "orders": self.orders,
            "order_lines": self.order_lines,
            "events": self.events,
            "vendors": self.vendors,
            "dcs": self.dcs,
            "lanes": self.lanes,
            "customers": self.customers,
            "skus": self.skus,
            "capacity_snapshots": self.capacity_snapshots,
        }

    def truth_tables(self) -> dict[str, pd.DataFrame]:
        """Evaluation-only ground truth, kept separate from model-facing tables."""
        return {
            "simulator_truth": self.simulator_truth,
            "line_truth": self.line_truth,
            "shocks": self.shocks,
        }
