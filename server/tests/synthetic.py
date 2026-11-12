"""Tiny synthetic Backblaze-like dataset for tests (deterministic).

Mimics the real layout: quarterly zips ``data_Q{q}_{year}.zip`` holding one CSV per
day. Quarters have different SMART column sets (Q1 lacks 184/242, Q2 adds an
unknown column) to exercise harmonization. Failing drives degrade over their last
~25 days (media + some mechanical/thermal signal); a few drives are decommissioned
without failing; one duplicate row is injected.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

START = date(2023, 1, 1)
END = date(2023, 6, 30)
MODELS = {  # model -> (n_drives, failure fraction, capacity)
    "SYN A1": (60, 0.25, 12_000_138_625_024),
    "SYN B2": (50, 0.25, 14_000_519_643_136),
    "SYN C3": (40, 0.25, 16_000_900_661_248),
    "SYN D4": (36, 0.25, 8_001_563_222_016),
    "SYN E5": (5, 0.0, 4_000_787_030_016),
}
Q1_SMART = [1, 3, 4, 5, 7, 9, 10, 12, 187, 188, 190, 192, 193, 194, 197, 198, 199, 240, 241]
Q2_SMART = [*Q1_SMART, 184, 242]


@dataclass
class Drive:
    serial: str
    model: str
    capacity: int
    start: date
    end: date
    fail_date: date | None
    seed: int
    rows: dict[date, dict[str, float]] = field(default_factory=dict)


def _make_drives(rng: np.random.Generator) -> list[Drive]:
    drives = []
    n_days = (END - START).days
    for mi, (model, (n, frac, cap)) in enumerate(MODELS.items()):
        n_fail = round(n * frac)
        for i in range(n):
            serial = f"Z{mi}{i:04d}{rng.integers(10, 99)}"
            start = START if rng.random() > 0.1 else START + timedelta(int(rng.integers(5, 40)))
            fail = None
            end = END
            if i < n_fail:
                fail = START + timedelta(int(rng.integers(45, n_days - 3)))
                end = fail
            elif rng.random() < 0.05:
                end = START + timedelta(int(rng.integers(60, n_days - 10)))  # decommissioned
            drives.append(Drive(serial, model, cap, start, end, fail, int(rng.integers(1e9))))
    return drives


def _simulate(d: Drive) -> None:
    rng = np.random.default_rng(d.seed)
    age0 = float(rng.integers(2_000, 30_000))
    realloc = float(rng.choice([0, 0, 0, 0, 8]))
    pending = 0.0
    load_cycles = float(rng.integers(1_000, 50_000))
    lba_w = float(rng.integers(1e10, 1e12))
    base_temp = float(rng.integers(26, 38))
    day = d.start
    while day <= d.end:
        k = (day - d.start).days
        to_fail = (d.fail_date - day).days if d.fail_date else None
        degrading = to_fail is not None and to_fail <= 25
        if degrading:
            realloc += float(rng.poisson(3 + (25 - to_fail)))
            pending = max(0.0, pending + float(rng.integers(-1, 6)))
        elif rng.random() < 0.01:
            realloc += 1
        load_cycles += float(rng.integers(5, 40))
        lba_w += float(rng.integers(1e8, 5e9))
        temp = base_temp + rng.normal(0, 1.2) + (3.0 if degrading and to_fail < 10 else 0.0)
        row = {
            "smart_1_raw": float(rng.integers(0, 2e8)),
            "smart_1_normalized": 80.0,
            "smart_3_raw": 0.0,
            "smart_3_normalized": 98.0,
            "smart_4_raw": 12.0,
            "smart_4_normalized": 100.0,
            "smart_5_raw": realloc,
            "smart_5_normalized": max(1.0, 100 - realloc / 20),
            "smart_7_raw": float(rng.integers(0, 1e9)),
            "smart_7_normalized": 85.0 - (8.0 if degrading else 0.0) + rng.normal(0, 1),
            "smart_9_raw": age0 + 24 * k,
            "smart_9_normalized": 70.0,
            "smart_10_raw": 0.0,
            "smart_10_normalized": 100.0,
            "smart_12_raw": 12.0 + (k // 60),
            "smart_12_normalized": 100.0,
            "smart_184_raw": 0.0,
            "smart_184_normalized": 100.0,
            "smart_187_raw": float(int(realloc / 4)) if degrading else 0.0,
            "smart_187_normalized": 100.0,
            "smart_188_raw": float(rng.poisson(0.5)) if degrading else 0.0,
            "smart_188_normalized": 100.0,
            "smart_190_raw": round(temp),
            "smart_190_normalized": 100 - round(temp),
            "smart_192_raw": 3.0,
            "smart_192_normalized": 100.0,
            "smart_193_raw": load_cycles,
            "smart_193_normalized": 90.0,
            "smart_194_raw": round(temp),
            "smart_194_normalized": round(temp),
            "smart_197_raw": pending,
            "smart_197_normalized": 100.0,
            "smart_198_raw": float(int(pending / 2)),
            "smart_198_normalized": 100.0,
            "smart_199_raw": 0.0,
            "smart_199_normalized": 200.0,
            "smart_240_raw": age0 + 23 * k,
            "smart_240_normalized": 100.0,
            "smart_241_raw": lba_w,
            "smart_241_normalized": 100.0,
            "smart_242_raw": lba_w * 1.7,
            "smart_242_normalized": 100.0,
        }
        d.rows[day] = row
        day += timedelta(days=1)


def build_synthetic_zips(raw_dir: Path, seed: int = 7) -> dict:
    """Write the zips into ``raw_dir``; returns ground truth (failure dates per serial)."""
    rng = np.random.default_rng(seed)
    drives = _make_drives(rng)
    for d in drives:
        _simulate(d)
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth = {d.serial: {"model": d.model, "fail_date": d.fail_date, "end": d.end} for d in drives}
    for q, (q_start, q_end, smart_ids, extra) in {
        1: (date(2023, 1, 1), date(2023, 3, 31), Q1_SMART, False),
        2: (date(2023, 4, 1), date(2023, 6, 30), Q2_SMART, True),
    }.items():
        smart_cols = [f"smart_{a}_{t}" for a in smart_ids for t in ("normalized", "raw")]
        header = ["date", "serial_number", "model", "capacity_bytes", "failure", "datacenter",
                  *smart_cols] + (["smart_999_raw"] if extra else [])
        if q == 1:  # different column order in Q1
            header = header[:5] + list(reversed(header[5:]))
        zpath = raw_dir / f"data_Q{q}_2023.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("__MACOSX/data_Q1/._junk.csv", "junk")
            day = q_start
            while day <= q_end:
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\n")
                w.writerow(header)
                for d in drives:
                    if day not in d.rows:
                        continue
                    row = d.rows[day]
                    rec = {"date": day.isoformat(), "serial_number": d.serial, "model": d.model,
                           "capacity_bytes": d.capacity,
                           "failure": int(d.fail_date == day), "datacenter": "syn",
                           "smart_999_raw": 1, **row}
                    vals = [rec.get(c, "") for c in header]
                    w.writerow(vals)
                    if d.serial == drives[0].serial and day == date(2023, 2, 1):
                        w.writerow(vals)  # duplicate row
                zf.writestr(f"data_Q{q}_2023/{day.isoformat()}.csv", buf.getvalue())
                day += timedelta(days=1)
    return truth


def synthetic_overrides(root: Path) -> dict:
    """Config overrides pointing every path into ``root`` and shrinking the problem."""
    return {
        "paths": {
            "raw": str(root / "raw"),
            "parquet": str(root / "parquet"),
            "artifacts": str(root / "artifacts"),
            "reports": str(root / "reports"),
            "runtime": str(root / "runtime"),
        },
        "data": {
            "years": [2023],
            "top_k_models": 3,
            "holdout_models": 1,
            "min_failures_per_model": 3,
            "neg_pos_ratio": 5,
            "negative_stride_days": 5,
            "eval_max_windows": 100_000,
            "duckdb_memory_limit": "1GB",
            "duckdb_threads": 2,
        },
        "models": {
            "lgbm": {"n_estimators": 60, "early_stopping_rounds": 10, "min_child_samples": 5},
            "sequence": {"epochs": 2, "hidden": 16, "batch_size": 128},
        },
        "rag": {"case_index_max_negatives": 2000},
        "evaluation": {"bootstrap_samples": 50, "llm_sample_size": 12,
                       "llm_sample_max_positives": 4},
        "seeds": [0],
    }
