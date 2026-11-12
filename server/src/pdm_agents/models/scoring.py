"""Batch numeric scoring (no LLM): family risks, fusion, uncertainty, disagreement, policy.

Shared by evaluation (full test sets) and the fleet snapshot behind ``GET /drives``.
The single-drive graph computes the same quantities node by node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdm_agents.config import AppConfig
from pdm_agents.data.features import compute_features
from pdm_agents.data.store import Store
from pdm_agents.decision.policy import decide_batch
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.bundle import CASE_AGENT, ModelBundle, case_risk
from pdm_agents.models.fusion import disagreement_batch, uncertainty

log = get_logger(__name__)


@dataclass
class NumericScores:
    family: dict[str, dict[int, np.ndarray]]
    fused: dict[int, np.ndarray]
    uncertainty: np.ndarray
    disagreement: np.ndarray


def numeric_scores(cfg: AppConfig, bundle: ModelBundle, df: pd.DataFrame,
                   agents: list[str] | None = None, method: str | None = None,
                   calibrated: bool | None = None,
                   case: dict[int, np.ndarray] | None = None) -> NumericScores:
    """Score every row of a feature table the way the graph would without an LLM.

    ``case`` = calibrated case-memory risk (``bundle.case_risk``); when given (and the
    bundle has a retrieval-aware fusion) it is fused like the graph's case agent.
    """
    agents = [a for a in (agents or cfg.numeric_agents()) if a in bundle.family_models]
    calibrated = cfg.features.use_calibration if calibrated is None else calibrated
    method = method or cfg.models.fusion.method
    family = bundle.all_family_risks(df, agents, calibrated=calibrated)
    inputs = {**family, CASE_AGENT: case} if case is not None else family
    fused = bundle.fuse(inputs, df["model"].astype(str).tolist(), method=method,
                        calibrated=calibrated)
    h = cfg.max_horizon
    with_case = case is not None and calibrated and bundle.has_case_fusion
    agent_risk = {a: r[h] for a, r in inputs.items() if with_case or a != CASE_AGENT}
    disagree = disagreement_batch(agent_risk, bundle.agent_weights(h, with_case=with_case)) \
        if family else np.zeros(len(df))
    n_numeric = max(len(cfg.numeric_agents()), 1)
    window_cov = np.clip(df["f__ctx__n_obs"].to_numpy("float64") / cfg.data.window_days, 0, 1)
    coverage = window_cov * (len(family) / n_numeric)
    unc = np.array([uncertainty(p, c) for p, c in zip(fused[h], coverage, strict=True)])
    return NumericScores(family, fused, unc, disagree)


def fleet_snapshot(cfg: AppConfig, store: Store, bundle: ModelBundle,
                   cases=None) -> pd.DataFrame:
    """Numeric risk + policy action for every drive at its latest snapshot."""
    drives = store.df("SELECT * FROM drives")
    splits_path = store.meta_dir / "drive_splits.parquet"
    if splits_path.exists():
        with splits_path.open("rb") as fh:
            splits = pd.read_parquet(fh)[["serial_number", "split"]]
        drives = drives.merge(splits, on="serial_number", how="left")
    samples = drives.rename(columns={"last_date": "as_of"})[
        ["serial_number", "model", "as_of", "capacity_bytes"]].copy()
    samples["sample_id"] = samples["serial_number"]
    log.info("fleet snapshot: scoring %d drives", len(samples))
    feats = compute_features(cfg, store, samples)
    use_case = (cases is not None and bundle.has_case_fusion and cfg.features.use_rag
                and cfg.features.use_case_retrieval)
    scores = numeric_scores(cfg, bundle, feats,
                            case=case_risk(cfg, bundle, cases, feats) if use_case else None)
    final, _, _ = decide_batch(cfg, scores.fused, cfg.decision.default_cost_ratio,
                               scores.uncertainty, scores.disagreement)
    out = feats[["serial_number", "model", "capacity_bytes", "as_of"]].copy()
    out = out.rename(columns={"as_of": "last_date"})
    for h, p in scores.fused.items():
        out[f"p_{h}"] = p
    out["uncertainty"] = scores.uncertainty
    out["action"] = final
    meta = drives.set_index("serial_number")
    out["failed"] = out["serial_number"].map(meta["failure_date"].notna())
    if "split" in meta:
        out["split"] = out["serial_number"].map(meta["split"])
    return out


def snapshot_path(cfg: AppConfig):
    return cfg.paths.artifacts / "fleet_snapshot.parquet"
