"""Evaluation context and the systems compared (baselines + full system).

A *system* maps a feature table to calibrated probabilities per horizon plus the
uncertainty/disagreement signals the decision policy uses. Numeric systems are
scored on the full test sets. Systems with ``graph`` overrides are also run
through the real LangGraph pipeline on the stratified LLM sample (see runner).

Baselines (spec section 11): LightGBM on all features, the sequence model, a
single LLM agent over all attributes, multi-agent without RAG, and the full system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pdm_agents.config import AppConfig
from pdm_agents.data.features import load_split
from pdm_agents.data.store import Store
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.bundle import CASE_AGENT, ModelBundle, case_risk, fit_case_fusion
from pdm_agents.models.fusion import disagreement_batch, uncertainty
from pdm_agents.rag.case_memory import CaseMemory

log = get_logger(__name__)


@dataclass
class SystemOutput:
    p: dict[int, np.ndarray]
    uncertainty: np.ndarray
    disagreement: np.ndarray


class EvalContext:
    """Per-seed scoring context with cached model outputs per table."""

    def __init__(self, cfg: AppConfig, store: Store, bundle: ModelBundle,
                 cases: CaseMemory | None, tables: list[str]) -> None:
        self.cfg, self.store, self.bundle, self.cases = cfg, store, bundle, cases
        self.tables = {t: load_split(cfg, t) for t in tables}
        self._cache: dict[tuple, object] = {}
        if cases is not None and not bundle.has_case_fusion:
            fit_case_fusion(cfg, bundle, cases)

    def df(self, table: str) -> pd.DataFrame:
        return self.tables[table]

    def _cached(self, key: tuple, fn: Callable[[], object]):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def family(self, table: str, calibrated: bool = True) -> dict[str, dict[int, np.ndarray]]:
        return self._cached(("family", table, calibrated),
                            lambda: self.bundle.all_family_risks(self.df(table),
                                                                 calibrated=calibrated))

    def case(self, table: str) -> dict[int, np.ndarray] | None:
        if self.cases is None or not self.bundle.has_case_fusion:
            return None
        return self._cached(("case", table),
                            lambda: case_risk(self.cfg, self.bundle, self.cases, self.df(table)))

    def global_risk(self, table: str) -> dict[int, np.ndarray]:
        return self._cached(("global", table), lambda: self.bundle.global_risk(self.df(table)))

    def sequence(self, table: str) -> dict[int, np.ndarray]:
        return self._cached(("sequence", table), lambda: self.bundle.sequence_risk(
            self.cfg, self.store, self.df(table)))

    def models(self, table: str) -> list[str]:
        return self.df(table)["model"].astype(str).tolist()

    def coverage(self, table: str) -> np.ndarray:
        n = self.df(table)["f__ctx__n_obs"].to_numpy("float64")
        return np.clip(n / self.cfg.data.window_days, 0, 1)

    def output(self, table: str, p: dict[int, np.ndarray],
               agents: dict[str, dict[int, np.ndarray]] | None = None,
               agent_cov: float = 1.0, weights: dict[str, float] | None = None) -> SystemOutput:
        """Package probabilities with uncertainty and weighted agent disagreement."""
        h = self.cfg.max_horizon
        cov = self.coverage(table) * agent_cov
        unc = np.array([uncertainty(x, c) for x, c in zip(p[h], cov, strict=True)])
        if agents and weights:
            dis = disagreement_batch({a: r[h] for a, r in agents.items()}, weights)
        else:
            dis = np.zeros(len(unc))
        return SystemOutput(p, unc, dis)


@dataclass
class SystemDef:
    name: str
    group: str  # baseline | full | ablation
    score: Callable[[EvalContext, str], SystemOutput]
    description: str
    cost_policy: bool = True
    drift_recalibration: bool = False
    graph: dict | None = None  # config overrides to run this system through the graph
    note: str = ""
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------- scoring fns


def score_global(ctx: EvalContext, table: str) -> SystemOutput:
    return ctx.output(table, ctx.global_risk(table))


def score_sequence(ctx: EvalContext, table: str) -> SystemOutput:
    return ctx.output(table, ctx.sequence(table))


def score_fusion(ctx: EvalContext, table: str, with_case: bool = True,
                 method: str | None = None, agents: list[str] | None = None,
                 calibrated: bool = True) -> SystemOutput:
    fam = ctx.family(table, calibrated=calibrated)
    if agents is not None:
        fam = {a: r for a, r in fam.items() if a in agents}
    inputs = dict(fam)
    case = ctx.case(table) if with_case and calibrated else None
    if case is not None:
        inputs[CASE_AGENT] = case
    p = ctx.bundle.fuse(inputs, ctx.models(table),
                        method=method or ctx.cfg.models.fusion.method, calibrated=calibrated)
    weights = ctx.bundle.agent_weights(ctx.cfg.max_horizon, with_case=case is not None)
    return ctx.output(table, p, inputs, weights=weights)


def _all_attributes_agent(cfg: AppConfig) -> dict:
    attrs: list[str] = []
    for a in cfg.numeric_agents():
        attrs.extend(c for c in cfg.agents[a].attributes if c not in attrs)
    disabled = {n: {"enabled": False} for n in cfg.agents}
    return {"agents": {**disabled, "all_attributes": {
        "class": "pdm_agents.agents.all_attributes:AllAttributesAgent", "enabled": True,
        "kind": "global", "description": "Every SMART attribute in one agent.",
        "attributes": attrs}}}


BASELINES: dict[str, Callable[[AppConfig], SystemDef]] = {
    "lgbm_global": lambda cfg: SystemDef(
        "lgbm_global", "baseline", score_global,
        "LightGBM on all features (calibrated on validation)."),
    "sequence": lambda cfg: SystemDef(
        "sequence", "baseline", score_sequence,
        f"PyTorch {cfg.models.sequence.kind.upper()} over the raw 30-day window (calibrated)."),
    "single_llm_agent": lambda cfg: SystemDef(
        "single_llm_agent", "baseline", score_global,
        "One LLM agent seeing all attributes with the same tools; risk from the "
        "all-features model.", graph=_all_attributes_agent(cfg),
        note="numeric risk identical to lgbm_global by construction; differs only in the "
             "graph run (planning, explanation)"),
    "multi_agent_no_rag": lambda cfg: SystemDef(
        "multi_agent_no_rag", "baseline",
        lambda ctx, t: score_fusion(ctx, t, with_case=False),
        "Specialist agents + calibrated stacking fusion, no retrieval.",
        graph={"features": {"use_rag": False}}),
    "full_system": lambda cfg: SystemDef(
        "full_system", "full", lambda ctx, t: score_fusion(ctx, t, with_case=True),
        "Specialists + case-memory risk + calibrated fusion + cost policy + abstention + "
        "drift-aware recalibration.", drift_recalibration=True, graph={}),
}
