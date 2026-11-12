"""Per-family window features, per-model normalization and sequence tensors.

For every SMART column owned by an agent (``agents.yaml``) and a window ending
at ``as_of`` (only snapshots in ``(as_of - window_days, as_of]``):

  last, delta (last - first), slope (per day), acceleration (slope of the
  second half minus slope of the first half), max, n_increases (day-over-day
  increases) and, for raw counters, days since the attribute first became
  non-zero (using all history <= as_of).

Feature columns are named ``f__<column>__<stat>``. The same SQL is used for
training tables and for single-drive inference, so train/serve skew is impossible.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_agents.config import AppConfig
from pdm_agents.data.labels import build_samples, label_column
from pdm_agents.data.store import Store
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)

STATS = ("last", "delta", "slope", "accel", "max", "n_inc", "days_nz")
CONTEXT_FEATURES = ["f__ctx__capacity_tb", "f__ctx__n_obs"]
META_COLUMNS = ["sample_id", "serial_number", "model", "as_of", "split", "capacity_bytes",
                "n_obs", "first_date", "last_date", "failure_date", "days_to_failure"]


def feature_name(column: str, stat: str) -> str:
    return f"f__{column}__{stat}"


def agent_columns(cfg: AppConfig) -> dict[str, list[str]]:
    """SMART columns per numeric agent (all agents in agents.yaml, enabled or not)."""
    return {n: list(a.attributes) for n, a in cfg.agents.items() if a.kind == "numeric"}


def all_feature_columns(cfg: AppConfig) -> list[str]:
    cols: list[str] = []
    for attrs in agent_columns(cfg).values():
        cols.extend(c for c in attrs if c not in cols)
    return cols


def family_features(cfg: AppConfig, agent: str, available: list[str] | None = None) -> list[str]:
    """Feature names of one agent's family (+ context), optionally limited to ``available``."""
    names = [feature_name(c, s) for c in agent_columns(cfg)[agent] for s in _stats_for(c)]
    names += CONTEXT_FEATURES
    if available is not None:
        names = [n for n in names if n in set(available)]
    return names


def global_features(cfg: AppConfig, available: list[str] | None = None) -> list[str]:
    names = [feature_name(c, s) for c in all_feature_columns(cfg) for s in _stats_for(c)]
    names += CONTEXT_FEATURES
    if available is not None:
        names = [n for n in names if n in set(available)]
    return names


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f__")]


def _stats_for(column: str) -> tuple[str, ...]:
    return STATS if column.endswith("_raw") else STATS[:-1]


def _feature_sql(columns: list[str], window_days: int) -> tuple[str, str, str]:
    """(lag select list, aggregate select list, first-nonzero select list)."""
    half = window_days / 2.0
    lags, aggs, fnz = [], [], []
    for c in columns:
        lags.append(f"{c} - lag({c}) OVER (PARTITION BY serial_number ORDER BY date) AS d__{c}")
        last = f"arg_max({c}, date) FILTER (WHERE {c} IS NOT NULL)"
        first = f"arg_min({c}, date) FILTER (WHERE {c} IS NOT NULL)"
        s2 = f"regr_slope({c}, t) FILTER (WHERE t > -{half})"
        s1 = f"regr_slope({c}, t) FILTER (WHERE t <= -{half})"
        aggs += [
            f"{last} AS {feature_name(c, 'last')}",
            f"{last} - {first} AS {feature_name(c, 'delta')}",
            f"regr_slope({c}, t) AS {feature_name(c, 'slope')}",
            f"{s2} - {s1} AS {feature_name(c, 'accel')}",
            f"max({c}) AS {feature_name(c, 'max')}",
            f"count(*) FILTER (WHERE d__{c} > 0) AS {feature_name(c, 'n_inc')}",
        ]
        if c.endswith("_raw"):
            fnz.append(f"min(date) FILTER (WHERE {c} > 0) AS fnz__{c}")
    return ", ".join(lags), ",\n".join(aggs), ", ".join(fnz)


def compute_features(
    cfg: AppConfig, store: Store, samples: pd.DataFrame, smart_view: str = "smart"
) -> pd.DataFrame:
    """Window features for ``samples`` (needs sample_id, serial_number, model, as_of).

    ``smart_view`` lets tests substitute a truncated view to prove no future data is used.
    """
    if samples.empty:
        return pd.DataFrame()
    available = set(store.df(f"SELECT * FROM {smart_view} LIMIT 0").columns)
    columns = [c for c in all_feature_columns(cfg) if c in available]
    w = cfg.data.window_days
    lag_sql, agg_sql, fnz_sql = _feature_sql(columns, w)
    parts = []
    for model, chunk in samples.groupby("model", sort=True):
        cur = store.cursor()
        keys = chunk[["sample_id", "serial_number", "as_of"]].copy()
        keys["as_of"] = pd.to_datetime(keys["as_of"]).dt.date
        cur.register("samp", keys)
        max_as_of = keys["as_of"].max()
        feats = cur.execute(
            f"""
            WITH base AS (
                SELECT serial_number, date, {", ".join(columns)}, {lag_sql}
                FROM {smart_view}
                WHERE model = ? AND serial_number IN (SELECT DISTINCT serial_number FROM samp)
                  AND date <= ?
            ),
            j AS (
                SELECT p.sample_id, b.*, datediff('day', p.as_of, b.date) AS t
                FROM samp p JOIN base b ON b.serial_number = p.serial_number
                 AND b.date > p.as_of - INTERVAL {w} DAY AND b.date <= p.as_of
            )
            SELECT sample_id, count(*) AS f__ctx__n_obs,
                   {agg_sql}
            FROM j GROUP BY sample_id
            """,
            [model, max_as_of],
        ).df()
        if fnz_sql:
            fnz = cur.execute(
                f"""
                SELECT serial_number, {fnz_sql} FROM {smart_view}
                WHERE model = ? AND serial_number IN (SELECT DISTINCT serial_number FROM samp)
                  AND date <= ?
                GROUP BY serial_number
                """,
                [model, max_as_of],
            ).df()
            merged = keys.merge(fnz, on="serial_number", how="left")
            as_of = pd.to_datetime(merged["as_of"])
            for c in columns:
                if not c.endswith("_raw"):
                    continue
                first_nz = pd.to_datetime(merged[f"fnz__{c}"])
                days = (as_of - first_nz).dt.days.astype("float64")
                merged[feature_name(c, "days_nz")] = days.where(first_nz <= as_of)
            keep = ["sample_id"] + [feature_name(c, "days_nz") for c in columns
                                    if c.endswith("_raw")]
            feats = feats.merge(merged[keep], on="sample_id", how="left")
        cur.unregister("samp")
        parts.append(feats)
    feats = pd.concat(parts, ignore_index=True)
    out = samples.merge(feats, on="sample_id", how="left")
    out["f__ctx__capacity_tb"] = out["capacity_bytes"].astype("float64") / 1e12
    fcols = feature_columns(out)
    out[fcols] = out[fcols].astype("float64")
    return out


# ---------------------------------------------------------------- normalization


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class Normalizer:
    """Per-drive-model robust scaling on signed-log values, fit on training data only.

    Models never seen in training use statistics pooled over all training models.
    """

    columns: list[str]
    per_model: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    pooled: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> Normalizer:
        norm = cls(columns=list(columns))
        norm.pooled = cls._stats(df[columns])
        for model, chunk in df.groupby("model"):
            norm.per_model[str(model)] = cls._stats(chunk[columns])
        return norm

    @staticmethod
    def _stats(frame: pd.DataFrame) -> dict[str, list[float]]:
        x = _slog(frame.to_numpy(dtype="float64"))
        with warnings.catch_warnings():  # attributes absent for a model are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(x, axis=0)
            q75, q25 = np.nanpercentile(x, 75, axis=0), np.nanpercentile(x, 25, axis=0)
            std = np.nanstd(x, axis=0)
        scale = (q75 - q25) / 1.349
        scale = np.where(np.isfinite(scale), scale, 0.0)
        std = np.where(np.isfinite(std), std, 0.0)
        scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
        med = np.where(np.isfinite(med), med, 0.0)
        return {"median": med.tolist(), "scale": scale.tolist()}

    def stats_for(self, model: str) -> dict[str, list[float]]:
        return self.per_model.get(model, self.pooled)

    def transform_array(self, x: np.ndarray, model: str) -> np.ndarray:
        """Normalize an array whose last axis follows ``self.columns``; NaN -> 0, clipped."""
        st = self.stats_for(model)
        z = (_slog(x) - np.asarray(st["median"])) / np.asarray(st["scale"])
        return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(self.columns)), dtype="float32")
        for model, idx in df.groupby("model").indices.items():
            out[idx] = self.transform_array(df.iloc[idx][self.columns].to_numpy("float64"),
                                            str(model))
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"columns": self.columns, "per_model": self.per_model,
                                    "pooled": self.pooled}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Normalizer:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(columns=d["columns"], per_model=d["per_model"], pooled=d["pooled"])


# -------------------------------------------------------------------- sequences


def build_sequences(
    cfg: AppConfig, store: Store, samples: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    """Raw window tensors ``(n_samples, window_days, n_columns)``; missing days are NaN.

    Day index 0 is ``as_of - window_days + 1``, the last index is ``as_of``.
    """
    w = cfg.data.window_days
    out = np.full((len(samples), w, len(columns)), np.nan, dtype="float32")
    if samples.empty:
        return out
    pos = {sid: i for i, sid in enumerate(samples["sample_id"])}
    for model, chunk in samples.groupby("model", sort=True):
        cur = store.cursor()
        keys = chunk[["sample_id", "serial_number", "as_of"]].copy()
        keys["as_of"] = pd.to_datetime(keys["as_of"]).dt.date
        cur.register("samp", keys)
        rows = cur.execute(
            f"""
            SELECT p.sample_id, {w - 1} + datediff('day', p.as_of, s.date) AS k,
                   {", ".join("s." + c for c in columns)}
            FROM samp p JOIN smart s ON s.serial_number = p.serial_number
             AND s.date > p.as_of - INTERVAL {w} DAY AND s.date <= p.as_of
            WHERE s.model = ?
            """,
            [model],
        ).df()
        cur.unregister("samp")
        idx = rows["sample_id"].map(pos).to_numpy()
        k = rows["k"].to_numpy().astype(int)
        out[idx, k, :] = rows[columns].to_numpy("float32")
    return out


# ------------------------------------------------------------------ orchestration


def features_dir(cfg: AppConfig) -> Path:
    return cfg.paths.parquet / "features"


def build_feature_tables(cfg: AppConfig, store: Store, seed: int = 0) -> str:
    """Sample windows for every split, compute features, write ``features/<split>.parquet``."""
    out_dir = features_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(cfg, store, seed=seed)
    lines = ["Feature tables"]
    for split in samples:
        df = samples[split]
        path = out_dir / f"{split}.parquet"
        if df.empty:
            path.unlink(missing_ok=True)
            lines.append(f"  {split}: no windows")
            continue
        feats = compute_features(cfg, store, df)
        with path.open("wb") as fh:
            feats.to_parquet(fh, index=False)
        pos = {h: int((feats[label_column(h)] == 1).sum()) for h in cfg.horizons}
        lines.append(
            f"  {split}: {len(feats)} windows, {feats['serial_number'].nunique()} drives, "
            f"positives {pos}, as_of {feats['as_of'].min()}..{feats['as_of'].max()}"
        )
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    return report


def load_split(cfg: AppConfig, split: str) -> pd.DataFrame:
    path = features_dir(cfg) / f"{split}.parquet"
    if not path.exists():
        return pd.DataFrame()
    with path.open("rb") as fh:
        return pd.read_parquet(fh)
