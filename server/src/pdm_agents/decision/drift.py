"""Feature drift per SMART family (PSI, with KS as a secondary statistic).

The reference distribution is the validation period (natural class distribution,
fit-time data only). Drift for a drive model at a date compares a sample of that
model's current windows against the reference of the same model (pooled reference
for models unseen in training). Drift is label-free, so it may use any data up to
the as-of date without leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pdm_agents.config import AppConfig
from pdm_agents.data.features import family_features
from pdm_agents.schemas import DriftReport

POOLED = "__pooled__"
MIN_SAMPLES = 30
# Raw attributes whose level is not a lifetime counter (temperature, pending sectors).
NON_CUMULATIVE_RAW = {"smart_190_raw", "smart_194_raw", "smart_197_raw"}
LEVEL_STATS = ("__last", "__max", "__days_nz")


def drift_feature(name: str) -> bool:
    """Features monitored for drift.

    Levels of cumulative raw counters (power-on hours, LBAs written, error totals) grow
    as every drive ages, so their PSI would flag normal aging as drift. For those only
    rate statistics (delta, slope, acceleration, increases) are monitored; levels are
    kept for normalized values and non-cumulative raw attributes.
    """
    if name.startswith("f__ctx"):
        return False
    column = name[3:].rsplit("__", 1)[0]
    if column.endswith("_raw") and column not in NON_CUMULATIVE_RAW:
        return not name.endswith(LEVEL_STATS)
    return True


def psi(ref_props: np.ndarray, cur_props: np.ndarray, eps: float = 1e-4) -> float:
    """Population stability index between two binned distributions."""
    r = np.clip(ref_props, eps, None)
    c = np.clip(cur_props, eps, None)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class _FeatureRef:
    edges: list[float]
    props: list[float]
    sample: list[float]  # small reference sample for KS


@dataclass
class DriftMonitor:
    families: dict[str, list[str]]
    bins: int = 10
    refs: dict[str, dict[str, _FeatureRef]] = field(default_factory=dict)  # model -> feature

    @classmethod
    def fit(cls, cfg: AppConfig, reference: pd.DataFrame, agents: list[str]) -> DriftMonitor:
        avail = list(reference.columns)
        families = {a: [f for f in family_features(cfg, a, avail) if drift_feature(f)]
                    for a in agents}
        mon = cls(families=families, bins=cfg.drift.bins)
        groups = {POOLED: reference, **dict(tuple(reference.groupby("model")))}
        rng = np.random.default_rng(0)
        for model, df in groups.items():
            refs = {}
            for feats in families.values():
                for f in feats:
                    x = df[f].dropna().to_numpy("float64")
                    if len(x) < MIN_SAMPLES:
                        continue
                    edges = np.unique(np.quantile(x, np.linspace(0, 1, mon.bins + 1)[1:-1]))
                    props = np.bincount(np.searchsorted(edges, x, side="right"),
                                        minlength=len(edges) + 1) / len(x)
                    sample = rng.choice(x, size=min(len(x), 500), replace=False)
                    refs[f] = _FeatureRef(edges.tolist(), props.tolist(), sample.tolist())
            mon.refs[str(model)] = refs
        return mon

    def report(self, current: pd.DataFrame, model: str, cfg: AppConfig,
               reference_name: str = "validation period") -> DriftReport:
        """PSI per family = mean PSI of the family's features (current vs reference)."""
        refs = self.refs.get(model) or self.refs.get(POOLED, {})
        by_family: dict[str, float] = {}
        ks_by_family: dict[str, float] = {}
        for fam, feats in self.families.items():
            vals, ks_vals = [], []
            for f in feats:
                ref = refs.get(f)
                if ref is None or f not in current:
                    continue
                x = current[f].dropna().to_numpy("float64")
                if len(x) < MIN_SAMPLES:
                    continue
                edges = np.asarray(ref.edges)
                cur = np.bincount(np.searchsorted(edges, x, side="right"),
                                  minlength=len(edges) + 1) / len(x)
                vals.append(psi(np.asarray(ref.props), cur))
                ks_vals.append(float(ks_2samp(ref.sample, x).statistic))
            if vals:
                by_family[fam] = round(float(np.mean(vals)), 4)
                ks_by_family[fam] = round(float(np.mean(ks_vals)), 4)
        worst = max(by_family.values(), default=0.0)
        level = ("alert" if worst >= cfg.drift.psi_alert
                 else "warn" if worst >= cfg.drift.psi_warn else "ok")
        name = reference_name if model in self.refs else f"{reference_name} (pooled models)"
        return DriftReport(psi_by_family=by_family, level=level,
                           reference=f"{name}; KS {ks_by_family}")

    def save(self, path: Path) -> None:
        data = {"families": self.families, "bins": self.bins,
                "refs": {m: {f: r.__dict__ for f, r in refs.items()}
                         for m, refs in self.refs.items()}}
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DriftMonitor:
        d = json.loads(path.read_text(encoding="utf-8"))
        mon = cls(families=d["families"], bins=d["bins"])
        mon.refs = {m: {f: _FeatureRef(**r) for f, r in refs.items()}
                    for m, refs in d["refs"].items()}
        return mon


class DriftService:
    """On-demand drift report per (drive model, date), cached.

    Samples up to ``drift.sample_drives`` drives of the model that reported on the
    date (deterministic order by serial hash), computes their window features with
    the same code as training, and compares them with the reference.
    """

    def __init__(self, cfg: AppConfig, store, monitor: DriftMonitor) -> None:
        self.cfg = cfg
        self.store = store
        self.monitor = monitor
        self._cache: dict[tuple[str, str], DriftReport] = {}

    def report(self, model: str, as_of) -> DriftReport:
        """Drift for ``model`` in the week of ``as_of`` (computed on the week's first day,
        which is never after ``as_of``; drift is a slow fleet-level signal)."""
        from datetime import timedelta

        from pdm_agents.data.features import compute_features

        as_of = pd.Timestamp(as_of).date()
        as_of = as_of - timedelta(days=as_of.weekday())
        key = (model, str(as_of))
        if key in self._cache:
            return self._cache[key]
        drives = self.store.df(
            """
            SELECT serial_number, capacity_bytes FROM smart
            WHERE model = ? AND date = ?
            ORDER BY hash(serial_number) LIMIT ?
            """,
            [model, as_of, self.cfg.drift.sample_drives],
        )
        if drives.empty:
            rep = DriftReport(psi_by_family={}, level="ok",
                              reference=f"no drives reported on {as_of}")
        else:
            samples = drives.assign(model=model, as_of=as_of,
                                    sample_id=[f"drift-{i}" for i in range(len(drives))])
            feats = compute_features(self.cfg, self.store, samples)
            rep = self.monitor.report(feats, model, self.cfg)
        self._cache[key] = rep
        return rep


def relevant_level(report: DriftReport, weights: dict[str, float] | None,
                   cfg: AppConfig) -> str:
    """Drift level counting only families the fusion relies on.

    Drift in a family with (near) zero fusion weight cannot change a prediction, so it
    should not escalate cases or trigger recalibration. Without weights, all families count.
    """
    fams = report.psi_by_family
    if weights is not None:
        fams = {f: v for f, v in fams.items()
                if weights.get(f, 0.0) >= cfg.drift.min_family_weight}
    worst = max(fams.values(), default=0.0)
    if worst >= cfg.drift.psi_alert:
        return "alert"
    return "warn" if worst >= cfg.drift.psi_warn else "ok"
