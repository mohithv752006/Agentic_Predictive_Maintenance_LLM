"""DuckDB access layer over the harmonized Parquet files.

Views:
  ``smart``  - daily SMART snapshots of the selected models (hive partitions
               model/year/quarter, files sorted by serial_number, date)
  ``drives`` - one row per drive: model, capacity, first/last date, failure date

All public methods open a fresh cursor, so a ``Store`` can be shared across threads.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pdm_agents.config import AppConfig
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)


def _posix(p: Path) -> str:
    return p.resolve().as_posix()


def connect(cfg: AppConfig) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with the configured resource limits."""
    con = duckdb.connect()
    tmp = cfg.paths.parquet.parent / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{cfg.data.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(cfg.data.duckdb_threads)}")
    con.execute(f"SET temp_directory='{_posix(tmp)}'")
    con.execute("SET preserve_insertion_order=false")
    return con


class Store:
    """Query helper over ``smart`` and ``drives``."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.smart_dir = cfg.paths.parquet / "smart"
        self.meta_dir = cfg.paths.parquet / "meta"
        self._con = connect(cfg)
        self._lock = threading.Lock()
        self.ready = self.smart_dir.exists() and (self.meta_dir / "drives.parquet").exists()
        if self.ready:
            self._create_views()

    def _create_views(self) -> None:
        glob = _posix(self.smart_dir) + "/**/*.parquet"
        self._con.execute(
            f"CREATE OR REPLACE VIEW smart AS SELECT * FROM read_parquet('{glob}', "
            "hive_partitioning=true, union_by_name=true)"
        )
        drives = _posix(self.meta_dir / "drives.parquet")
        self._con.execute(
            f"CREATE OR REPLACE VIEW drives AS SELECT * FROM read_parquet('{drives}')"
        )

    def cursor(self) -> duckdb.DuckDBPyConnection:
        with self._lock:
            return self._con.cursor()

    def df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return self.cursor().execute(sql, params or []).df()

    # ------------------------------------------------------------------ meta

    def models_meta(self) -> dict[str, Any]:
        path = self.meta_dir / "models.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def drive_info(self, serial: str) -> dict[str, Any] | None:
        rows = self.df("SELECT * FROM drives WHERE serial_number = ?", [serial])
        return None if rows.empty else rows.iloc[0].to_dict()

    # --------------------------------------------------------------- queries

    def drive_history(
        self, serial: str, start: date | None = None, end: date | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """All snapshots of one drive (oldest first), optionally within [start, end]."""
        info = self.drive_info(serial)
        if info is None:
            return pd.DataFrame()
        cols = ", ".join(["date", "serial_number", "capacity_bytes", "failure", *(columns or [])])
        if columns is None:
            cols = "*"
        sql = f"SELECT {cols} FROM smart WHERE model = ? AND serial_number = ?"
        params: list[Any] = [info["model"], serial]
        if start is not None:
            sql += " AND date >= ?"
            params.append(start)
        if end is not None:
            sql += " AND date <= ?"
            params.append(end)
        return self.df(sql + " ORDER BY date", params)

    def drive_window(self, serial: str, as_of: date, days: int) -> pd.DataFrame:
        """Snapshots in (as_of - days, as_of]; never anything after ``as_of``."""
        start = pd.Timestamp(as_of) - pd.Timedelta(days=days - 1)
        return self.drive_history(serial, start.date(), as_of)

    def summary(self) -> pd.DataFrame:
        """Drives, drive-days, failures and date range per model."""
        return self.df(
            """
            SELECT d.model, count(*) AS drives, sum(d.n_days) AS drive_days,
                   sum(CASE WHEN d.failure_date IS NOT NULL THEN 1 ELSE 0 END) AS failures,
                   min(d.first_date) AS first_date, max(d.last_date) AS last_date
            FROM drives d GROUP BY d.model ORDER BY failures DESC
            """
        )


def summarize(store: Store) -> str:
    """Human-readable data summary."""
    meta = store.models_meta()
    df = store.summary()
    lines = ["Data summary (selected models)", df.to_string(index=False)]
    if meta:
        lines.append(f"training models: {meta.get('training_models')}")
        lines.append(f"unseen (held-out) models: {meta.get('holdout_models')}")
    return "\n".join(lines)
