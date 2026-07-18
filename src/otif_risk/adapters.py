"""Production-shaped source adapters (Stage 2 governance).

Models the "how does data enter" half of the target architecture's
production-shaped source boundary without any real ERP/WMS/TMS/SRM
integration: each adapter is a typed :class:`SourceAdapter` that knows how
to ``load(as_of_timestamp)`` its own canonical table(s) from a local CSV
snapshot (the same tables ``pipeline.run_pipeline`` already persists under
a run's ``data/`` directory), filtered to what would genuinely be knowable
as of that timestamp -- the identical point-in-time contract
``features.build_feature_table`` already enforces, applied again here at
the source boundary rather than only inside the feature builder.

Only the model-facing ``PrototypeDataset.tables()`` columns are ever
adapter outputs. Adapters never carry ``PrototypeDataset.truth_tables()``
(``simulator_truth``/``line_truth``/``shocks``) -- those are
evaluation-only ground truth and must never look like a legitimate
production source (see the module docstring of ``contracts.py`` and the
plan's "no simulator truth/potential outcomes in model/policy service
inputs" constraint). ``assemble_prototype_dataset`` fills those three
fields with empty placeholder frames when building a dataset for the
serving boundary, since ``pipeline.score_orders`` (the serving-path
scoring function) never reads them -- only ``run_pipeline``'s own
offline evaluation/reporting does.

Canonical source-system ownership used here (a defensible, documented
mapping; this prototype's twin does not itself distinguish "systems"):

- **ERP** -- order capture and customer master data: ``orders``,
  ``order_lines``, ``customers``.
- **WMS** -- warehouse/DC operations: ``dcs``, ``capacity_snapshots``, and
  ``SHIPPED`` events.
- **TMS** -- transportation: ``lanes``, and ``IN_TRANSIT``/``DELIVERED``
  events.
- **SRM** -- supplier/vendor and SKU master data: ``vendors``, ``skus``,
  and ``VENDOR_READY`` events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from otif_risk.contracts import PrototypeDataset

#: The complete set of model-facing source tables every assembled dataset
#: must carry (mirrors ``PrototypeDataset.tables()``'s keys).
REQUIRED_SOURCE_TABLES: frozenset[str] = frozenset(
    {
        "orders",
        "order_lines",
        "events",
        "vendors",
        "dcs",
        "lanes",
        "customers",
        "skus",
        "capacity_snapshots",
    }
)

#: Event types owned by each transportation/warehouse/supplier adapter.
WMS_EVENT_TYPES = ("SHIPPED",)
TMS_EVENT_TYPES = ("IN_TRANSIT", "DELIVERED")
SRM_EVENT_TYPES = ("VENDOR_READY",)


@runtime_checkable
class SourceAdapter(Protocol):
    """A typed, production-shaped boundary for one upstream source system."""

    #: Short, stable identifier (e.g. ``"erp"``) used in error messages and
    #: the decision ledger's ``source_snapshot_id``.
    source_name: str

    def load(self, as_of_timestamp: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Return every canonical table this source owns, as of ``as_of_timestamp``.

        Implementations must never return rows/values that would not
        genuinely be knowable at ``as_of_timestamp`` in a real deployment.
        """
        ...


def _read_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"source CSV not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates or [])


def _redact_future_events(
    events: pd.DataFrame, as_of_timestamp: pd.Timestamp, event_types: tuple[str, ...]
) -> pd.DataFrame:
    """Keep only ``event_types`` rows, blanking ``event_timestamp`` for any
    event that has not genuinely occurred by ``as_of_timestamp`` yet.

    Mirrors ``features._event_features``'s own ``event_timestamp <=
    as_of_timestamp`` filter, applied here at the source boundary: a real
    WMS/TMS/SRM feed would simply not yet have recorded a future event, but
    ``planned_timestamp`` (the schedule/promise) is legitimately already
    known, so only the *realized* timestamp is redacted, never the row.
    """
    subset = events.loc[events["event_type"].isin(event_types)].copy()
    occurred = pd.to_datetime(subset["event_timestamp"]) <= as_of_timestamp
    subset.loc[~occurred, "event_timestamp"] = pd.NaT
    return subset


class LocalCsvERPAdapter:
    """Order-capture + customer-master source: ``orders``, ``order_lines``, ``customers``."""

    source_name = "erp"

    def __init__(self, orders_path: Path, order_lines_path: Path, customers_path: Path) -> None:
        self.orders_path = Path(orders_path)
        self.order_lines_path = Path(order_lines_path)
        self.customers_path = Path(customers_path)

    def load(self, as_of_timestamp: pd.Timestamp) -> dict[str, pd.DataFrame]:
        as_of_timestamp = pd.Timestamp(as_of_timestamp)
        orders = _read_csv(
            self.orders_path,
            parse_dates=[
                "order_date",
                "order_capture_timestamp",
                "prediction_timestamp",
                "requested_delivery_date",
                "promised_delivery_date",
            ],
        )
        # An order not yet captured by as_of_timestamp cannot exist in a real
        # ERP feed queried at that moment.
        orders = orders.loc[orders["order_date"] <= as_of_timestamp].reset_index(drop=True)
        order_lines = _read_csv(self.order_lines_path)
        order_lines = order_lines.loc[
            order_lines["order_id"].isin(orders["order_id"])
        ].reset_index(drop=True)
        customers = _read_csv(self.customers_path)
        return {"orders": orders, "order_lines": order_lines, "customers": customers}


class LocalCsvWMSAdapter:
    """Warehouse/DC operations source: ``dcs``, ``capacity_snapshots``, ``SHIPPED`` events."""

    source_name = "wms"

    def __init__(self, dcs_path: Path, capacity_snapshots_path: Path, events_path: Path) -> None:
        self.dcs_path = Path(dcs_path)
        self.capacity_snapshots_path = Path(capacity_snapshots_path)
        self.events_path = Path(events_path)

    def load(self, as_of_timestamp: pd.Timestamp) -> dict[str, pd.DataFrame]:
        as_of_timestamp = pd.Timestamp(as_of_timestamp)
        dcs = _read_csv(self.dcs_path)
        snapshots = _read_csv(self.capacity_snapshots_path, parse_dates=["snapshot_date"])
        snapshots = snapshots.loc[
            snapshots["snapshot_date"] <= as_of_timestamp
        ].reset_index(drop=True)
        events = _read_csv(
            self.events_path, parse_dates=["planned_timestamp", "event_timestamp"]
        )
        wms_events = _redact_future_events(events, as_of_timestamp, WMS_EVENT_TYPES)
        return {"dcs": dcs, "capacity_snapshots": snapshots, "events": wms_events}


class LocalCsvTMSAdapter:
    """Transportation source: ``lanes``, ``IN_TRANSIT``/``DELIVERED`` events."""

    source_name = "tms"

    def __init__(self, lanes_path: Path, events_path: Path) -> None:
        self.lanes_path = Path(lanes_path)
        self.events_path = Path(events_path)

    def load(self, as_of_timestamp: pd.Timestamp) -> dict[str, pd.DataFrame]:
        as_of_timestamp = pd.Timestamp(as_of_timestamp)
        lanes = _read_csv(self.lanes_path)
        events = _read_csv(
            self.events_path, parse_dates=["planned_timestamp", "event_timestamp"]
        )
        tms_events = _redact_future_events(events, as_of_timestamp, TMS_EVENT_TYPES)
        return {"lanes": lanes, "events": tms_events}


class LocalCsvSRMAdapter:
    """Supplier/vendor + SKU master data source: ``vendors``, ``skus``, ``VENDOR_READY`` events."""

    source_name = "srm"

    def __init__(self, vendors_path: Path, skus_path: Path, events_path: Path) -> None:
        self.vendors_path = Path(vendors_path)
        self.skus_path = Path(skus_path)
        self.events_path = Path(events_path)

    def load(self, as_of_timestamp: pd.Timestamp) -> dict[str, pd.DataFrame]:
        as_of_timestamp = pd.Timestamp(as_of_timestamp)
        vendors = _read_csv(self.vendors_path)
        skus = _read_csv(self.skus_path)
        events = _read_csv(
            self.events_path, parse_dates=["planned_timestamp", "event_timestamp"]
        )
        srm_events = _redact_future_events(events, as_of_timestamp, SRM_EVENT_TYPES)
        return {"vendors": vendors, "skus": skus, "events": srm_events}


def default_adapters(data_dir: Path) -> list[SourceAdapter]:
    """Construct the four canonical local-CSV adapters over one run's ``data/`` directory."""
    data_dir = Path(data_dir)
    return [
        LocalCsvERPAdapter(
            data_dir / "orders.csv", data_dir / "order_lines.csv", data_dir / "customers.csv"
        ),
        LocalCsvWMSAdapter(
            data_dir / "dcs.csv", data_dir / "capacity_snapshots.csv", data_dir / "events.csv"
        ),
        LocalCsvTMSAdapter(data_dir / "lanes.csv", data_dir / "events.csv"),
        LocalCsvSRMAdapter(
            data_dir / "vendors.csv", data_dir / "skus.csv", data_dir / "events.csv"
        ),
    ]


def assemble_source_tables(
    adapters: list[SourceAdapter], as_of_timestamp: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Load every adapter and merge their tables into one canonical set.

    ``events`` is contributed by three adapters (WMS/TMS/SRM, each owning a
    disjoint subset of ``event_type`` values) and is concatenated rather
    than treated as a collision; every other table must be contributed by
    exactly one adapter.
    """
    tables: dict[str, pd.DataFrame] = {}
    event_parts: list[pd.DataFrame] = []
    for adapter in adapters:
        for name, frame in adapter.load(as_of_timestamp).items():
            if name == "events":
                event_parts.append(frame)
                continue
            if name in tables:
                raise ValueError(
                    f"duplicate source table '{name}' contributed by adapter "
                    f"'{adapter.source_name}'"
                )
            tables[name] = frame
    if event_parts:
        tables["events"] = pd.concat(event_parts, ignore_index=True).sort_values(
            ["order_id", "event_type"]
        ).reset_index(drop=True)

    missing = REQUIRED_SOURCE_TABLES - set(tables)
    if missing:
        raise ValueError(f"assembled source tables missing: {sorted(missing)}")
    return tables


def assemble_prototype_dataset(
    adapters: list[SourceAdapter],
    as_of_timestamp: pd.Timestamp,
    *,
    truth_tables: dict[str, pd.DataFrame] | None = None,
) -> PrototypeDataset:
    """Build a ``PrototypeDataset`` from adapter-sourced canonical tables.

    ``truth_tables`` (``simulator_truth``/``line_truth``/``shocks``) exist
    purely to satisfy ``PrototypeDataset``'s dataclass shape -- the serving
    boundary (``pipeline.score_orders``, ``features.build_feature_table``)
    never reads them. When omitted, empty placeholder frames are used;
    passing the real truth tables (e.g. from an in-memory evaluation
    harness) is only ever done for offline/batch parity *tests*, never for
    the model/policy serving inputs themselves -- see the plan's "no
    simulator truth/potential outcomes in model/policy service inputs"
    constraint.
    """
    tables = assemble_source_tables(adapters, as_of_timestamp)
    truth = truth_tables or {}
    return PrototypeDataset(
        orders=tables["orders"],
        order_lines=tables["order_lines"],
        events=tables["events"],
        vendors=tables["vendors"],
        dcs=tables["dcs"],
        lanes=tables["lanes"],
        customers=tables["customers"],
        skus=tables["skus"],
        capacity_snapshots=tables["capacity_snapshots"],
        simulator_truth=truth.get("simulator_truth", pd.DataFrame()),
        line_truth=truth.get("line_truth", pd.DataFrame()),
        shocks=truth.get("shocks", pd.DataFrame()),
    )


def data_quality_report(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Soft data-quality/contract diagnostics over one assembled table set.

    Deliberately simple, transparent checks (no external DQ framework):
    completeness (fraction of non-null cells per table), uniqueness (primary
    key duplication), and referential health (order_lines/events/orders
    foreign keys resolve). Used by ``monitoring.py`` to report contract
    failure counts alongside prediction/policy quality.
    """
    primary_keys = {
        "orders": "order_id",
        "order_lines": "order_line_id",
        "vendors": "vendor_id",
        "dcs": "dc_id",
        "lanes": "lane_id",
        "customers": "customer_id",
        "skus": "sku_id",
    }
    completeness: dict[str, float] = {}
    uniqueness: dict[str, Any] = {}
    for name, frame in tables.items():
        cells = frame.shape[0] * max(frame.shape[1], 1)
        completeness[name] = round(float(1 - frame.isna().sum().sum() / cells), 4) if cells else 1.0
        key = primary_keys.get(name)
        if key and key in frame.columns:
            duplicates = int(frame.duplicated(subset=[key]).sum())
            uniqueness[name] = {"duplicate_keys": duplicates, "passed": duplicates == 0}

    referential_checks = {
        "order_lines_order_id_resolves": bool(
            tables["order_lines"]["order_id"].isin(tables["orders"]["order_id"]).all()
        )
        if "order_lines" in tables and "orders" in tables
        else None,
        "events_order_id_resolves": bool(
            tables["events"]["order_id"].isin(tables["orders"]["order_id"]).all()
        )
        if "events" in tables and "orders" in tables
        else None,
        "orders_vendor_id_resolves": bool(
            tables["orders"]["vendor_id"].isin(tables["vendors"]["vendor_id"]).all()
        )
        if "orders" in tables and "vendors" in tables
        else None,
    }
    contract_failures = sum(1 for value in referential_checks.values() if value is False)
    contract_failures += sum(1 for entry in uniqueness.values() if not entry["passed"])

    return {
        "completeness_by_table": completeness,
        "uniqueness_by_table": uniqueness,
        "referential_checks": referential_checks,
        "contract_failure_count": contract_failures,
        "passed": contract_failures == 0,
    }
