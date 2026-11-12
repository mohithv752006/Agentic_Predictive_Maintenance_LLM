"""Horizon labels, leakage-safe splits and window sampling.

Splits (all enforced by tests):

* **Serial-disjoint**: every serial of a training model is hashed (md5) into
  exactly one of train / val / test. Serials of held-out models form ``unseen``.
* **Temporal**: the data's date range is cut into consecutive train / val / test
  periods. A split only samples as-of dates inside its own period, and only when
  every horizon label is observable before the period ends
  (``as_of + max_horizon <= period_end``), so training labels never look into
  the validation or test period. ``unseen`` uses the test period.
* Calibrators, scalers, thresholds and case memory are fit on train/val only
  (enforced where they are fit).

Labels for a window ending at ``as_of`` (inputs use only snapshots <= as_of):
``y_h = 1`` if the drive's failure row is 1..h days after ``as_of``; ``0`` if it is
known to survive the horizon; ``null`` (censored) if it left the fleet before
``as_of + h`` without failing. Windows on or after the failure day are excluded.

Sampling: training keeps all windows positive at the longest horizon plus
``neg_pos_ratio`` negatives per positive (drawn from a stride grid). Validation,
test and unseen use a uniform sample capped at ``eval_max_windows``, which
preserves the natural class distribution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pdm_agents.config import AppConfig
from pdm_agents.data.store import Store
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test", "unseen")
# Every positive window of failing test/unseen drives (not capped): for lead-time metrics.
LEADTIME_SETS = {"leadtime_test": "test", "leadtime_unseen": "unseen"}


def label_column(h: int) -> str:
    return f"y_{h}"


@dataclass(frozen=True)
class Periods:
    """Inclusive date ranges of each split's as-of dates."""

    data_start: date
    data_end: date
    train_end: date
    val_end: date

    def bounds(self, split: str) -> tuple[date, date]:
        """Period [start, end] of a split (unseen shares the test period)."""
        if split == "train":
            return self.data_start, self.train_end
        if split == "val":
            return self.train_end + timedelta(days=1), self.val_end
        return self.val_end + timedelta(days=1), self.data_end

    def to_json(self) -> dict:
        return {k: v.isoformat() for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, d: dict) -> Periods:
        return cls(**{k: date.fromisoformat(v) for k, v in d.items()})


def compute_periods(cfg: AppConfig, data_start: date, data_end: date) -> Periods:
    span = (data_end - data_start).days
    f = cfg.data.split.time_fractions
    train_end = data_start + timedelta(days=int(span * f["train"]))
    val_end = data_start + timedelta(days=int(span * (f["train"] + f["val"])))
    return Periods(data_start, data_end, train_end, val_end)


def serial_unit(serial: str) -> float:
    """Stable pseudo-uniform number in [0, 1) from the serial number."""
    return int(hashlib.md5(serial.encode("utf-8")).hexdigest()[:12], 16) / float(1 << 48)


def assign_splits(drives: pd.DataFrame, cfg: AppConfig, holdout_models: list[str]) -> pd.Series:
    """Split name per drive (index aligned with ``drives``)."""
    f = cfg.data.split.serial_fractions
    u = drives["serial_number"].map(serial_unit)
    split = np.where(u < f["train"], "train", np.where(u < f["train"] + f["val"], "val", "test"))
    split = np.where(drives["model"].isin(holdout_models), "unseen", split)
    return pd.Series(split, index=drives.index, name="split")


def build_split_table(cfg: AppConfig, store: Store) -> tuple[pd.DataFrame, Periods]:
    """Drives with their split; also writes ``meta/splits.json`` and ``meta/drive_splits``."""
    drives = store.df("SELECT * FROM drives")
    meta = store.models_meta()
    drives["split"] = assign_splits(drives, cfg, meta.get("holdout_models", []))
    periods = compute_periods(
        cfg,
        pd.Timestamp(drives["first_date"].min()).date(),
        pd.Timestamp(drives["last_date"].max()).date(),
    )
    out = store.meta_dir
    with (out / "drive_splits.parquet").open("wb") as fh:
        drives[["serial_number", "model", "split"]].to_parquet(fh, index=False)
    (out / "splits.json").write_text(
        json.dumps({"periods": periods.to_json(),
                    "serial_fractions": cfg.data.split.serial_fractions,
                    "time_fractions": cfg.data.split.time_fractions,
                    "counts": drives["split"].value_counts().to_dict()}, indent=2),
        encoding="utf-8",
    )
    return drives, periods


def load_periods(store: Store) -> Periods:
    data = json.loads((store.meta_dir / "splits.json").read_text(encoding="utf-8"))
    return Periods.from_json(data["periods"])


def label_sql(horizons: list[int]) -> str:
    """SQL expressions for the horizon labels (needs as_of, failure_date, last_date)."""
    parts = []
    for h in horizons:
        parts.append(
            f"""CASE
                WHEN failure_date IS NOT NULL
                     AND datediff('day', as_of, failure_date) BETWEEN 1 AND {h} THEN 1
                WHEN failure_date IS NULL AND datediff('day', as_of, last_date) < {h} THEN NULL
                ELSE 0 END::FLOAT AS {label_column(h)}"""
        )
    return ",\n".join(parts)


def candidate_windows(
    cfg: AppConfig, store: Store, model: str, split: str, periods: Periods, serials: pd.DataFrame
) -> pd.DataFrame:
    """All eligible (serial, as_of) windows of one model inside a split's period."""
    start, end = periods.bounds(split)
    end = min(end, periods.data_end) - timedelta(days=cfg.max_horizon)
    if end < start:
        return pd.DataFrame()
    w = cfg.data.window_days
    h = cfg.max_horizon
    # Training negatives are only drawn from a stride grid: filter in SQL so the
    # (much larger) off-grid negatives never leave DuckDB. Positives are all kept.
    grid = ""
    if split == "train":
        grid = (f"AND ((d.failure_date IS NOT NULL AND datediff('day', o.as_of, d.failure_date)"
                f" BETWEEN 1 AND {h}) OR datediff('day', d.first_date, o.as_of)"
                f" % {cfg.data.negative_stride_days} = 0)")
    cur = store.cursor()
    cur.register("split_serials", serials[["serial_number"]])
    df = cur.execute(
        f"""
        WITH obs AS (
            SELECT s.serial_number, s.date AS as_of,
                   count(*) OVER (PARTITION BY s.serial_number ORDER BY s.date
                                  RANGE BETWEEN INTERVAL {w - 1} DAYS PRECEDING
                                  AND CURRENT ROW) AS n_obs
            FROM smart s
            WHERE s.model = ? AND s.serial_number IN (SELECT serial_number FROM split_serials)
              AND s.date <= ?
        )
        SELECT o.serial_number, o.as_of, o.n_obs, d.model, d.capacity_bytes, d.first_date,
               d.failure_date, d.last_date,
               datediff('day', o.as_of, d.failure_date) AS days_to_failure,
               {label_sql(cfg.horizons)}
        FROM obs o JOIN drives d USING (serial_number)
        WHERE o.as_of BETWEEN ? AND ?
          AND o.n_obs >= ?
          AND (d.failure_date IS NULL OR o.as_of < d.failure_date)
          {grid}
        """,
        [model, end, start, end, cfg.data.min_window_obs],
    ).df()
    cur.unregister("split_serials")
    return df


def sample_windows(
    cfg: AppConfig, candidates: pd.DataFrame, split: str, seed: int
) -> pd.DataFrame:
    """Training: all positives + ratio of strided negatives. Eval: uniform capped sample."""
    if candidates.empty:
        return candidates
    rng = np.random.default_rng(seed)
    ymax = label_column(cfg.max_horizon)
    if split == "train":
        pos = candidates[candidates[ymax] == 1]
        offset = (pd.to_datetime(candidates["as_of"]) - pd.to_datetime(candidates["first_date"]))
        on_grid = (offset.dt.days % cfg.data.negative_stride_days) == 0
        neg = candidates[(candidates[ymax] == 0) & on_grid]
        k = min(len(neg), max(1, cfg.data.neg_pos_ratio * max(len(pos), 1)))
        neg = neg.iloc[rng.choice(len(neg), size=k, replace=False)] if k < len(neg) else neg
        out = pd.concat([pos, neg])
    else:
        out = candidates
    return out


def build_samples(cfg: AppConfig, store: Store, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Sampled windows per split with labels (no features yet)."""
    drives, periods = build_split_table(cfg, store)
    result: dict[str, list[pd.DataFrame]] = {s: [] for s in [*SPLITS, *LEADTIME_SETS]}
    ymax = label_column(cfg.max_horizon)
    for model in sorted(drives["model"].unique()):
        for split in SPLITS:
            serials = drives[(drives["model"] == model) & (drives["split"] == split)]
            if serials.empty:
                continue
            cand = candidate_windows(cfg, store, model, split, periods, serials)
            for name, src in LEADTIME_SETS.items():
                if src == split and not cand.empty:
                    result[name].append(cand[cand[ymax] == 1].assign(split=split))
            sampled = sample_windows(cfg, cand, split, seed)
            if not sampled.empty:
                sampled = sampled.assign(split=split)
                result[split].append(sampled)
            log.info("%s / %s: %d candidates -> %d windows", model, split, len(cand),
                     len(sampled))
    out: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)
    for split, parts in result.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if split in SPLITS and split != "train" and len(df) > cfg.data.eval_max_windows:
            # uniform cap across models keeps the natural class distribution
            df = df.iloc[np.sort(rng.choice(len(df), cfg.data.eval_max_windows, replace=False))]
        if not df.empty:
            df = df.sort_values(["model", "serial_number", "as_of"]).reset_index(drop=True)
            df["sample_id"] = [f"{split}-{i}" for i in range(len(df))]
        out[split] = df
    return out
