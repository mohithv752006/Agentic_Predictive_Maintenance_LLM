"""Harmonize Backblaze daily CSVs (inside quarterly zips) into partitioned Parquet.

SMART columns change between quarters, so each daily CSV is read with only the
configured columns that it actually has; absent ones are filled with nulls and
every SMART column is stored as float64. Files are processed one at a time
(streamed out of the zip, never extracted to disk), so memory stays bounded.

Steps:
1. ``stage_quarters``: one staging Parquet file per day (all drive models),
   resumable: days already staged are skipped.
2. ``select_models``: rank models by failures and drive-days (computed, not
   hard-coded) -> training models + held-out unseen models.
3. ``write_final``: copy the selected models into
   ``parquet/smart/model=.../year=.../quarter=.../*.parquet`` (deduplicated,
   sorted by serial and date) and write ``parquet/meta/models.json``.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from pdm_agents.config import AppConfig
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)

CORE_COLUMNS = ["date", "serial_number", "model", "capacity_bytes", "failure"]
CORE_TYPES: dict[str, pa.DataType] = {
    "date": pa.date32(),
    "serial_number": pa.string(),
    "model": pa.string(),
    "capacity_bytes": pa.int64(),
    "failure": pa.int8(),
}
DAY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.csv$")
ZIP_RE = re.compile(r"data_Q([1-4])_(\d{4})\.zip$")


@dataclass(frozen=True)
class DayFile:
    zip_path: Path
    member: str
    day: date

    @property
    def year(self) -> int:
        return self.day.year

    @property
    def quarter(self) -> int:
        return (self.day.month - 1) // 3 + 1


def target_schema(smart_columns: list[str]) -> pa.Schema:
    fields = [pa.field(c, CORE_TYPES[c]) for c in CORE_COLUMNS]
    fields += [pa.field(c, pa.float64()) for c in smart_columns]
    return pa.schema(fields)


def list_day_files(zip_paths: list[Path]) -> list[DayFile]:
    """Every daily CSV inside the given zips (macOS resource forks skipped)."""
    out = []
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if "__MACOSX" in name:
                    continue
                m = DAY_RE.search(name)
                if m:
                    out.append(DayFile(zp, name, date(*map(int, m.groups()))))
    return sorted(out, key=lambda d: d.day)


def read_day_csv(data: bytes, smart_columns: list[str]) -> pa.Table:
    """Parse one daily CSV into the harmonized schema."""
    header = data[: data.find(b"\n")].decode("utf-8", errors="replace").strip().split(",")
    header = [h.strip() for h in header]
    missing_core = [c for c in CORE_COLUMNS if c not in header]
    if missing_core:
        raise ValueError(f"CSV lacks core columns {missing_core}")
    present_smart = [c for c in smart_columns if c in header]
    include = CORE_COLUMNS + present_smart
    types = {**CORE_TYPES, **{c: pa.float64() for c in present_smart}}
    table = pacsv.read_csv(
        io.BytesIO(data),
        convert_options=pacsv.ConvertOptions(
            include_columns=include, column_types=types, strings_can_be_null=True
        ),
    )
    schema = target_schema(smart_columns)
    n = table.num_rows
    arrays = []
    for f in schema:
        if f.name in table.column_names:
            col = table[f.name]
            if f.name in ("model", "serial_number"):
                col = pa.compute.utf8_trim_whitespace(col)
            arrays.append(col.cast(f.type))
        else:
            arrays.append(pa.nulls(n, type=f.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _stage_path(staging: Path, d: DayFile) -> Path:
    return staging / f"year={d.year}" / f"quarter={d.quarter}" / f"{d.day.isoformat()}.parquet"


def _stage_one(d: DayFile, staging: Path, smart_columns: list[str]) -> tuple[date, int]:
    out = _stage_path(staging, d)
    if out.exists():
        return d.day, -1
    with zipfile.ZipFile(d.zip_path) as zf:
        data = zf.read(d.member)
    table = read_day_csv(data, smart_columns)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as fh:  # file handle: avoids native path encoding issues
        pq.write_table(table, fh, compression="zstd")
    tmp.replace(out)
    return d.day, table.num_rows


def stage_quarters(cfg: AppConfig, zip_paths: list[Path], workers: int = 4) -> Path:
    """Stage every daily CSV of ``zip_paths`` as Parquet; returns the staging dir."""
    staging = cfg.paths.parquet / "staging"
    days = list_day_files(zip_paths)
    if not days:
        raise RuntimeError("no daily CSV files found in the zips")
    log.info("staging %d daily files from %d zips", len(days), len(zip_paths))
    smart = cfg.data.smart_columns
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for day, rows in pool.map(lambda d: _stage_one(d, staging, smart), days):
            done += 1
            if rows >= 0 and (done % 20 == 0 or done == len(days)):
                log.info("staged %d/%d (%s: %d rows)", done, len(days), day, rows)
    return staging


def configured_zips(cfg: AppConfig) -> list[Path]:
    """Downloaded zips for the configured years (incomplete ``.part`` files ignored)."""
    out = []
    for p in sorted(cfg.paths.raw.glob("data_Q*_*.zip")):
        m = ZIP_RE.search(p.name)
        if m and int(m.group(2)) in cfg.data.years:
            out.append(p)
    return out


def iter_staged_files(staging: Path) -> Iterator[Path]:
    yield from sorted(staging.glob("year=*/quarter=*/*.parquet"))


def select_models(cfg: AppConfig, staging: Path) -> dict:
    """Rank drive models in the staged data and pick training + unseen models.

    Ranking: models with at least ``min_failures_per_model`` failures, ordered by
    failures then drive-days. The first ``top_k_models`` are used for training;
    the next ``holdout_models`` are held out entirely (unseen-model test set).
    An explicit ``data.drive_models`` list overrides the training choice.
    """
    from pdm_agents.data.store import connect

    con = connect(cfg)
    glob = staging.resolve().as_posix() + "/year=*/quarter=*/*.parquet"
    stats = con.execute(
        f"""
        SELECT model, count(DISTINCT serial_number) AS drives, count(*) AS drive_days,
               sum(failure)::BIGINT AS failures, min(date) AS first_date, max(date) AS last_date
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE model IS NOT NULL AND serial_number IS NOT NULL
        GROUP BY model
        """
    ).df()
    eligible = stats[stats["failures"] >= cfg.data.min_failures_per_model].sort_values(
        ["failures", "drive_days"], ascending=False
    )
    ranked = eligible["model"].tolist()
    if cfg.data.drive_models == "auto":
        training = ranked[: cfg.data.top_k_models]
    else:
        training = [m for m in cfg.data.drive_models if m in set(stats["model"])]
    holdout = [m for m in ranked if m not in training][: cfg.data.holdout_models]
    if not training:
        raise RuntimeError("no drive model satisfies min_failures_per_model")
    top = stats.sort_values("failures", ascending=False).head(15).copy()
    for c in ("first_date", "last_date"):
        top[c] = top[c].astype(str)
    return {
        "training_models": training,
        "holdout_models": holdout,
        "ranking_rule": "failures desc, drive_days desc; min_failures_per_model="
        f"{cfg.data.min_failures_per_model}",
        "candidates": top.to_dict(orient="records"),
    }


def write_final(cfg: AppConfig, staging: Path, selection: dict) -> Path:
    """Write selected models to partitioned Parquet + ``meta/drives.parquet``."""
    from pdm_agents.data.store import connect

    models = selection["training_models"] + selection["holdout_models"]
    smart_dir = cfg.paths.parquet / "smart"
    meta_dir = cfg.paths.parquet / "meta"
    if smart_dir.exists():
        import shutil

        shutil.rmtree(smart_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    con = connect(cfg)
    smart_cols = ", ".join(cfg.data.smart_columns)
    con.execute("CREATE TEMP TABLE sel(model VARCHAR)")
    con.executemany("INSERT INTO sel VALUES (?)", [[m] for m in models])
    quarters = sorted(p for p in staging.glob("year=*/quarter=*") if p.is_dir())
    log.info("writing final Parquet for %d models, %d quarters", len(models), len(quarters))
    # One quarter at a time: dedup + sort fit in memory (a single global sort spills
    # tens of GB to disk and is I/O-bound on a laptop). Duplicates are per (serial, date),
    # so they never cross a quarter boundary.
    for qdir in quarters:
        glob = qdir.resolve().as_posix() + "/*.parquet"
        con.execute(
            f"""
            COPY (
                SELECT date, serial_number, capacity_bytes, failure, {smart_cols},
                       model, year(date) AS year, quarter(date) AS quarter
                FROM read_parquet('{glob}', hive_partitioning=false)
                WHERE model IN (SELECT model FROM sel)
                QUALIFY row_number() OVER (PARTITION BY serial_number, date
                                           ORDER BY failure DESC) = 1
                ORDER BY model, serial_number, date
            ) TO '{smart_dir.resolve().as_posix()}'
            (FORMAT parquet, COMPRESSION zstd, PARTITION_BY (model, year, quarter),
             OVERWRITE_OR_IGNORE)
            """
        )
        log.info("wrote %s/%s", qdir.parent.name, qdir.name)
    holdout = set(selection["holdout_models"])
    drives = con.execute(
        f"""
        SELECT serial_number, any_value(model) AS model, max(capacity_bytes) AS capacity_bytes,
               min(date) AS first_date, max(date) AS last_date, count(*) AS n_days,
               min(CASE WHEN failure = 1 THEN date END) AS failure_date
        FROM read_parquet('{smart_dir.resolve().as_posix()}/**/*.parquet', hive_partitioning=true)
        GROUP BY serial_number
        """
    ).df()
    drives["holdout_model"] = drives["model"].isin(holdout)
    with (meta_dir / "drives.parquet").open("wb") as fh:
        drives.to_parquet(fh, index=False)
    (meta_dir / "models.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    n_failed = int(drives["failure_date"].notna().sum())
    log.info("final data: %d drives, %d failures", len(drives), n_failed)
    return smart_dir


def prepare(cfg: AppConfig, workers: int = 4, keep_staging: bool = True) -> dict:
    """Full harmonization: stage -> select models -> final Parquet. Returns the selection."""
    zips = configured_zips(cfg)
    if not zips:
        raise RuntimeError(f"no downloaded zips for years {cfg.data.years} in {cfg.paths.raw}")
    staging = stage_quarters(cfg, zips, workers=workers)
    selection = select_models(cfg, staging)
    log.info("selected training models %s, unseen %s", selection["training_models"],
             selection["holdout_models"])
    write_final(cfg, staging, selection)
    if not keep_staging:
        import shutil

        shutil.rmtree(staging)
    return selection
