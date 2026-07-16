"""Small shared contracts for the standalone PDF prototype."""

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


@dataclass(frozen=True)
class PrototypeConfig:
    seed: int = 42
    n_orders: int = 2_500
    start_date: str = "2024-01-01"
    prediction_horizon_days: int = 7
    planner_capacity_fraction: float = 0.15
    threshold_strategy: str = "recall_floor"
    target_recall: float = 0.55
    min_precision: float = 0.35
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


@dataclass
class PrototypeDataset:
    orders: pd.DataFrame
    order_lines: pd.DataFrame
    events: pd.DataFrame
    vendors: pd.DataFrame
    dcs: pd.DataFrame
    lanes: pd.DataFrame
    customers: pd.DataFrame
    capacity_snapshots: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "orders": self.orders,
            "order_lines": self.order_lines,
            "events": self.events,
            "vendors": self.vendors,
            "dcs": self.dcs,
            "lanes": self.lanes,
            "customers": self.customers,
            "capacity_snapshots": self.capacity_snapshots,
        }
