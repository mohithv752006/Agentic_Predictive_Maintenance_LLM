"""Trained artifact bundle: train every model for one seed, save/load, and score windows.

Contents (per horizon unless noted):
  family LightGBM models (one per numeric agent) + isotonic/Platt calibrators,
  global all-features LightGBM + calibrator, sequence model (all horizons) +
  calibrators, fusion (stacking / log-odds / majority vote), per-model base rates,
  the feature normalizer (case memory) and the drift monitor.

Fit-data discipline: models fit on the training split (early stopping on a
serial-grouped 15% slice of it); calibrators, fusion, base rates and the drift
reference fit on the validation split; calibrated inputs to fusion are
out-of-fold (5 serial-grouped folds) so fusion never sees in-sample calibrations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pdm_agents.config import AppConfig
from pdm_agents.data.features import (
    Normalizer,
    all_feature_columns,
    build_sequences,
    family_features,
    feature_columns,
    global_features,
    load_split,
)
from pdm_agents.data.labels import label_column, serial_unit
from pdm_agents.data.store import Store
from pdm_agents.decision.drift import DriftMonitor
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.base import grouped_holdout, logit, sigmoid
from pdm_agents.models.calibration import Calibrator
from pdm_agents.models.fusion import Fusion
from pdm_agents.models.lgbm import LGBMModel
from pdm_agents.models.sequence import SequenceBatch, SequenceModel

log = get_logger(__name__)

POOLED = "__pooled__"
CASE_AGENT = "case_retrieval"
N_FOLDS = 5
BASE_RATE_SMOOTHING = 1000.0


def monotone_in_horizon(risk: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Enforce P(fail within h) non-decreasing in h (running maximum over sorted horizons).

    Per-horizon models and calibrators are fit independently and can violate this
    logical constraint slightly; the running maximum is the smallest correction.
    """
    out, running = {}, None
    for h in sorted(risk):
        cur = np.asarray(risk[h], dtype="float64")
        running = cur if running is None else np.maximum(running, cur)
        out[h] = running
    return out


def bundle_dir(cfg: AppConfig, seed: int) -> Path:
    return cfg.paths.artifacts / "models" / f"seed_{seed}"


@dataclass
class ModelBundle:
    horizons: list[int]
    family_models: dict[str, dict[int, LGBMModel]]
    family_cals: dict[str, dict[int, Calibrator]]
    global_models: dict[int, LGBMModel]
    global_cals: dict[int, Calibrator]
    fusion: dict[int, Fusion]
    base_rates: dict[str, dict[int, float]]
    normalizer: Normalizer
    drift: DriftMonitor
    sequence: SequenceModel | None = None
    sequence_cals: dict[int, Calibrator] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    val_oof: pd.DataFrame | None = None  # out-of-fold calibrated val risks (for refits)
    case_cals: dict[int, Calibrator] = field(default_factory=dict)  # kNN case-risk calibrators
    rag_fusion: dict[int, Fusion] = field(default_factory=dict)  # fusion incl. case_retrieval

    @property
    def families(self) -> list[str]:
        return list(self.family_models)

    @property
    def has_case_fusion(self) -> bool:
        return bool(self.rag_fusion) and bool(self.case_cals)

    def agent_weights(self, h: int, with_case: bool = False) -> dict[str, float]:
        """Normalized non-negative fusion weights per agent (for disagreement)."""
        f = self.rag_fusion[h] if with_case and self.has_case_fusion else self.fusion[h]
        return dict(zip(f.agents, f.weights_, strict=True))

    # ----------------------------------------------------------------- scoring

    def prior(self, models: list[str] | np.ndarray) -> dict[int, np.ndarray]:
        pooled = self.base_rates[POOLED]
        return {h: np.array([self.base_rates.get(str(m), pooled)[h] for m in models])
                for h in self.horizons}

    def family_risk(self, agent: str, df: pd.DataFrame,
                    calibrated: bool = True) -> dict[int, np.ndarray]:
        out = {}
        for h in self.horizons:
            raw = self.family_models[agent][h].predict_proba(df)
            out[h] = self.family_cals[agent][h].transform(raw) if calibrated else raw
        return monotone_in_horizon(out)

    def all_family_risks(self, df: pd.DataFrame, agents: list[str] | None = None,
                         calibrated: bool = True) -> dict[str, dict[int, np.ndarray]]:
        agents = [a for a in (agents or self.families) if a in self.family_models]
        return {a: self.family_risk(a, df, calibrated) for a in agents}

    def global_risk(self, df: pd.DataFrame, calibrated: bool = True) -> dict[int, np.ndarray]:
        out = {}
        for h in self.horizons:
            raw = self.global_models[h].predict_proba(df)
            out[h] = self.global_cals[h].transform(raw) if calibrated else raw
        return monotone_in_horizon(out)

    def fuse(self, family_risks: dict[str, dict[int, np.ndarray]], models: list[str],
             method: str = "stacking", calibrated: bool = True) -> dict[int, np.ndarray]:
        """Fused probability per horizon. ``calibrated=False`` = uncalibrated ablation:
        raw risks pooled in log-odds space with the fitted weights, no recalibration.
        If ``family_risks`` contains ``case_retrieval`` (calibrated kNN risk) and a
        retrieval-aware fusion was fitted, that fusion is used."""
        prior = self.prior(models)
        use_case = calibrated and CASE_AGENT in family_risks and self.has_case_fusion
        out = {}
        for h in self.horizons:
            risks = {a: r[h] for a, r in family_risks.items()
                     if use_case or a != CASE_AGENT}
            f = self.rag_fusion[h] if use_case else self.fusion[h]
            if calibrated:
                out[h] = f.predict(risks, prior[h], method)
            else:
                X = f._matrix(risks, prior[h])
                out[h] = sigmoid(f._pool_logit(X))
        return monotone_in_horizon(out)

    def sequence_risk(self, cfg: AppConfig, store: Store, df: pd.DataFrame,
                      calibrated: bool = True, chunk: int = 20_000) -> dict[int, np.ndarray]:
        if self.sequence is None:
            raise RuntimeError("bundle has no sequence model")
        parts = []
        for i in range(0, len(df), chunk):
            part = df.iloc[i : i + chunk]
            seq = build_sequences(cfg, store, part, self.sequence.columns)
            parts.append(self.sequence.predict_proba(
                SequenceBatch(seq, part["model"].astype(str).to_numpy())))
        probs = np.concatenate(parts) if parts else np.zeros((0, len(self.horizons)))
        out = {}
        for j, h in enumerate(self.sequence.horizons):
            out[h] = self.sequence_cals[h].transform(probs[:, j]) if calibrated else probs[:, j]
        return monotone_in_horizon(out)

    # ------------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for agent, by_h in self.family_models.items():
            for h, m in by_h.items():
                m.save(path / "family" / f"{agent}_h{h}")
        for h, m in self.global_models.items():
            m.save(path / "global" / f"global_h{h}")
        cals = {
            "family": {a: {str(h): c.to_dict() for h, c in by_h.items()}
                       for a, by_h in self.family_cals.items()},
            "global": {str(h): c.to_dict() for h, c in self.global_cals.items()},
            "sequence": {str(h): c.to_dict() for h, c in self.sequence_cals.items()},
        }
        (path / "calibrators.json").write_text(json.dumps(cals), encoding="utf-8")
        (path / "fusion.json").write_text(
            json.dumps({str(h): f.to_dict() for h, f in self.fusion.items()}), encoding="utf-8")
        (path / "base_rates.json").write_text(json.dumps(
            {m: {str(h): v for h, v in r.items()} for m, r in self.base_rates.items()}),
            encoding="utf-8")
        self.normalizer.save(path / "normalizer.json")
        self.drift.save(path / "drift.json")
        if self.sequence is not None:
            self.sequence.save(path / "sequence")
        if self.val_oof is not None:
            with (path / "val_oof.parquet").open("wb") as fh:
                self.val_oof.to_parquet(fh, index=False)
        if self.has_case_fusion:
            (path / "case_fusion.json").write_text(json.dumps({
                "case_cals": {str(h): c.to_dict() for h, c in self.case_cals.items()},
                "rag_fusion": {str(h): f.to_dict() for h, f in self.rag_fusion.items()},
            }), encoding="utf-8")
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str),
                                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        horizons = [int(h) for h in meta["horizons"]]
        cals = json.loads((path / "calibrators.json").read_text(encoding="utf-8"))
        family_models = {
            a: {h: LGBMModel.load(path / "family" / f"{a}_h{h}") for h in horizons}
            for a in meta["families"]
        }
        seq_path = path / "sequence"
        val_oof = None
        if (path / "val_oof.parquet").exists():
            with (path / "val_oof.parquet").open("rb") as fh:
                val_oof = pd.read_parquet(fh)
        case_cals, rag_fusion = {}, {}
        if (path / "case_fusion.json").exists():
            cf = json.loads((path / "case_fusion.json").read_text(encoding="utf-8"))
            case_cals = {int(h): Calibrator.from_dict(c) for h, c in cf["case_cals"].items()}
            rag_fusion = {int(h): Fusion.from_dict(d) for h, d in cf["rag_fusion"].items()}
        return cls(
            horizons=horizons,
            family_models=family_models,
            family_cals={a: {int(h): Calibrator.from_dict(c) for h, c in by_h.items()}
                         for a, by_h in cals["family"].items()},
            global_models={h: LGBMModel.load(path / "global" / f"global_h{h}")
                           for h in horizons},
            global_cals={int(h): Calibrator.from_dict(c) for h, c in cals["global"].items()},
            fusion={int(h): Fusion.from_dict(d) for h, d in json.loads(
                (path / "fusion.json").read_text(encoding="utf-8")).items()},
            base_rates={m: {int(h): v for h, v in r.items()} for m, r in json.loads(
                (path / "base_rates.json").read_text(encoding="utf-8")).items()},
            normalizer=Normalizer.load(path / "normalizer.json"),
            drift=DriftMonitor.load(path / "drift.json"),
            sequence=SequenceModel.load(seq_path) if seq_path.exists() else None,
            sequence_cals={int(h): Calibrator.from_dict(c)
                           for h, c in cals.get("sequence", {}).items()},
            meta=meta,
            val_oof=val_oof,
            case_cals=case_cals,
            rag_fusion=rag_fusion,
        )


# ------------------------------------------------------------------- training


def _folds(serials: pd.Series) -> np.ndarray:
    return (serials.map(serial_unit).to_numpy() * 997 % N_FOLDS).astype(int)


def _crossfit(method: str, raw: np.ndarray, y: np.ndarray,
              folds: np.ndarray) -> tuple[np.ndarray, Calibrator]:
    """Out-of-fold calibrated predictions + a calibrator fit on all rows."""
    oof = np.empty_like(raw)
    for k in range(N_FOLDS):
        m = folds == k
        if not m.any():
            continue
        oof[m] = Calibrator(method).fit(raw[~m], y[~m]).transform(raw[m])
    return oof, Calibrator(method).fit(raw, y)


def _base_rates(val: pd.DataFrame, horizons: list[int]) -> dict[str, dict[int, float]]:
    rates: dict[str, dict[int, float]] = {POOLED: {}}
    for h in horizons:
        y = val[label_column(h)]
        rates[POOLED][h] = float(y.mean())
    for model, df in val.groupby("model"):
        rates[str(model)] = {}
        for h in horizons:
            y = df[label_column(h)].dropna()
            a = BASE_RATE_SMOOTHING
            rates[str(model)][h] = float((y.sum() + a * rates[POOLED][h]) / (len(y) + a))
    return rates


def _sequence_rows(df: pd.DataFrame, cap: int, ymax: str, seed: int) -> pd.DataFrame:
    """All positives + random others up to ``cap`` (train), or uniform cap (val)."""
    if len(df) <= cap:
        return df
    rng = np.random.default_rng(seed)
    pos = df[df[ymax] == 1]
    rest = df.drop(pos.index)
    k = max(0, cap - len(pos))
    return pd.concat([pos, rest.iloc[rng.choice(len(rest), size=k, replace=False)]])


def train_bundle(cfg: AppConfig, store: Store, seed: int = 0,
                 train_sequence: bool = True) -> ModelBundle:
    """Train every model for ``seed`` from the feature tables."""
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    if train.empty or val.empty:
        raise RuntimeError("feature tables missing: run `python -m scripts.prepare` first")
    avail = [c for c in feature_columns(train) if train[c].notna().any()]
    agents = [n for n, a in cfg.agents.items() if a.kind == "numeric"]
    hold = grouped_holdout(train["serial_number"].to_numpy(), 0.15, seed)
    method = cfg.models.calibration.method
    folds = _folds(val["serial_number"])

    family_models: dict[str, dict[int, LGBMModel]] = {}
    family_cals: dict[str, dict[int, Calibrator]] = {}
    global_models, global_cals, fusion = {}, {}, {}
    skipped: list[str] = []
    val_metrics: dict[str, dict] = {}
    base_rates = _base_rates(val, cfg.horizons)
    val_oof = val[["sample_id", "serial_number", "model"]].copy()

    for h in cfg.horizons:
        yc = label_column(h)
        tr_mask = train[yc].notna().to_numpy()
        fit_df, es_df = train[tr_mask & ~hold], train[tr_mask & hold]
        vmask = val[yc].notna().to_numpy()
        v, y_val, f_val = val[vmask], val.loc[vmask, yc].to_numpy(), folds[vmask]
        oof: dict[str, np.ndarray] = {}
        for agent in agents:
            feats = family_features(cfg, agent, avail)
            if not [f for f in feats if not f.startswith("f__ctx")]:
                if agent not in skipped:
                    skipped.append(agent)
                    log.warning("agent %s: none of its attributes exist in the data", agent)
                continue
            m = LGBMModel(f"{agent}_h{h}", feats, cfg.models.lgbm, seed)
            m.fit(fit_df, fit_df[yc].to_numpy(), es_df, es_df[yc].to_numpy())
            family_models.setdefault(agent, {})[h] = m
            oof[agent], cal = _crossfit(method, m.predict_proba(v), y_val, f_val)
            family_cals.setdefault(agent, {})[h] = cal
            log.info("h=%d %s: %d trees, val AUPRC (oof calibrated) %.4f", h, agent,
                     m.best_iteration, _auprc(y_val, oof[agent]))
        g = LGBMModel(f"global_h{h}", global_features(cfg, avail), cfg.models.lgbm, seed)
        g.fit(fit_df, fit_df[yc].to_numpy(), es_df, es_df[yc].to_numpy())
        global_models[h] = g
        g_oof, global_cals[h] = _crossfit(method, g.predict_proba(v), y_val, f_val)
        prior = np.array([base_rates.get(str(mm), base_rates[POOLED])[h] for mm in v["model"]])
        fusion[h] = Fusion(list(oof), h).fit(oof, prior, y_val)
        fused_oof = fusion[h].predict(oof, prior, "stacking")
        for agent, p in oof.items():
            val_oof.loc[vmask, f"oof__{agent}__{h}"] = p
        val_oof.loc[vmask, f"oof__global__{h}"] = g_oof
        val_oof[f"y_{h}"] = val[yc].to_numpy()
        val_oof.loc[vmask, f"prior_{h}"] = prior
        val_metrics[str(h)] = {
            "val_positives": int(np.nansum(y_val)),
            "global_auprc": _auprc(y_val, g_oof),
            "fused_auprc_in_sample": _auprc(y_val, fused_oof),
            "families": {a: _auprc(y_val, p) for a, p in oof.items()},
        }

    normalizer = Normalizer.fit(train, global_features(cfg, avail))
    drift = DriftMonitor.fit(cfg, val, list(family_models))

    sequence, seq_cals = None, {}
    if train_sequence:
        sequence, seq_cals = _train_sequence(cfg, store, train, val, seed)

    meta = {
        "seed": seed,
        "horizons": cfg.horizons,
        "families": list(family_models),
        "skipped_agents": skipped,
        "calibration": method,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train": len(train),
        "n_val": len(val),
        "training_models": sorted(train["model"].unique().tolist()),
        "available_features": len(avail),
        "val_metrics": val_metrics,
        "top_features": {f"{a}_h{cfg.max_horizon}": family_models[a][cfg.max_horizon]
                         .feature_importance(5) for a in family_models},
    }
    return ModelBundle(cfg.horizons, family_models, family_cals, global_models, global_cals,
                       fusion, base_rates, normalizer, drift, sequence, seq_cals, meta,
                       val_oof=val_oof)


def fit_case_fusion(cfg: AppConfig, bundle: ModelBundle, cases) -> None:
    """Calibrate the case-memory kNN risk and fit a fusion that includes it (in place).

    Uses validation data only: kNN risk of each validation window against the
    training-only case memory, isotonic/Platt calibrators cross-fitted by serial,
    and a stacking fusion over the out-of-fold family risks plus the case risk.
    """
    if bundle.val_oof is None:
        raise RuntimeError("bundle lacks out-of-fold validation risks; retrain it")
    val = load_split(cfg, "val")
    oof = bundle.val_oof.set_index("sample_id").loc[val["sample_id"]].reset_index()
    prior = bundle.prior(val["model"].astype(str).tolist())
    raw = cases.knn_risk(val, cfg.rag.case_top_k, bundle.horizons, prior,
                         same_model_only=cfg.rag.same_model_only)
    folds = _folds(val["serial_number"])
    method = cfg.models.calibration.method
    for h in bundle.horizons:
        y = oof[f"y_{h}"].to_numpy()
        m = ~np.isnan(y)
        case_oof, bundle.case_cals[h] = _crossfit(method, raw[h][m], y[m], folds[m])
        col = np.full(len(oof), np.nan)
        col[m] = case_oof
        bundle.val_oof = bundle.val_oof.merge(
            pd.DataFrame({"sample_id": oof["sample_id"], f"oof__{CASE_AGENT}__{h}": col}),
            on="sample_id", how="left", suffixes=("_old", ""))
        bundle.val_oof = bundle.val_oof.drop(
            columns=[c for c in bundle.val_oof.columns if c.endswith("_old")])
        risks = {a: oof.loc[m, f"oof__{a}__{h}"].to_numpy() for a in bundle.families}
        risks[CASE_AGENT] = case_oof
        bundle.rag_fusion[h] = Fusion([*bundle.families, CASE_AGENT], h).fit(
            risks, prior[h][m], y[m])
        bundle.meta.setdefault("case_fusion", {})[str(h)] = {
            "case_auprc_oof": _auprc(y[m], case_oof),
            "case_coef": bundle.rag_fusion[h].coef_[-1],
        }


def case_risk(
    cfg: AppConfig, bundle: ModelBundle, cases, df: pd.DataFrame
) -> dict[int, np.ndarray]:
    """Calibrated kNN case risk for feature rows (requires ``fit_case_fusion``)."""
    prior = bundle.prior(df["model"].astype(str).tolist())
    raw = cases.knn_risk(df, cfg.rag.case_top_k, bundle.horizons, prior,
                         same_model_only=cfg.rag.same_model_only)
    return monotone_in_horizon({h: bundle.case_cals[h].transform(raw[h]) for h in bundle.horizons})


def _train_sequence(cfg: AppConfig, store: Store, train: pd.DataFrame, val: pd.DataFrame,
                    seed: int) -> tuple[SequenceModel, dict[int, Calibrator]]:
    available = set(store.df("SELECT * FROM smart LIMIT 0").columns)
    columns = [c for c in all_feature_columns(cfg) if c in available]
    cap = cfg.models.sequence.max_windows
    ymax = label_column(cfg.max_horizon)
    tr = _sequence_rows(train, cap, ymax, seed).reset_index(drop=True)
    labels = tr[[label_column(h) for h in cfg.horizons]].to_numpy("float32")
    log.info("sequence model: building %d training windows", len(tr))
    x_tr = build_sequences(cfg, store, tr, columns)
    model = SequenceModel(columns, cfg.horizons, cfg.models.sequence, seed)
    model.fit(SequenceBatch(x_tr, tr["model"].astype(str).to_numpy(),
                            tr["serial_number"].to_numpy()), labels)
    del x_tr
    rng = np.random.default_rng(seed)
    vs = val if len(val) <= cap else val.iloc[np.sort(rng.choice(len(val), cap, replace=False))]
    vs = vs.reset_index(drop=True)
    probs = model.predict_proba(SequenceBatch(build_sequences(cfg, store, vs, columns),
                                              vs["model"].astype(str).to_numpy()))
    cals = {}
    for j, h in enumerate(cfg.horizons):
        y = vs[label_column(h)].to_numpy()
        m = ~np.isnan(y)
        cals[h] = Calibrator(cfg.models.calibration.method).fit(probs[m, j], y[m])
    return model, cals


def _auprc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y)
    m = ~np.isnan(y)
    if len(np.unique(y[m])) < 2:
        return None
    return round(float(average_precision_score(y[m], p[m])), 4)


def load_bundle(cfg: AppConfig, seed: int | None = None) -> ModelBundle | None:
    """Load the bundle for ``seed`` (default: first configured seed), or ``None``."""
    path = bundle_dir(cfg, cfg.seeds[0] if seed is None else seed)
    if not (path / "meta.json").exists():
        return None
    return ModelBundle.load(path)


__all__ = ["ModelBundle", "bundle_dir", "load_bundle", "logit", "train_bundle"]
