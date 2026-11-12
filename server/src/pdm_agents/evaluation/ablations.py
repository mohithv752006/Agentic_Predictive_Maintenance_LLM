"""Config-driven ablations of the full system (one component removed at a time).

Removing an agent refits the fusion without it on the stored out-of-fold
validation risks (no retraining of the family models, no test data touched).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pdm_agents.config import AppConfig
from pdm_agents.evaluation.baselines import EvalContext, SystemDef, SystemOutput, score_fusion
from pdm_agents.models.bundle import CASE_AGENT
from pdm_agents.models.fusion import Fusion


def _refit_without(ctx: EvalContext, drop: str) -> dict[int, Fusion]:
    """Fusion over (families + case) minus ``drop``, fit on validation OOF risks."""
    key = ("fusion_without", drop)
    if key in ctx._cache:
        return ctx._cache[key]
    oof = ctx.bundle.val_oof
    if oof is None:
        raise RuntimeError("bundle has no out-of-fold validation risks; retrain")
    keep = [a for a in [*ctx.bundle.families, CASE_AGENT]
            if a != drop and f"oof__{a}__{ctx.cfg.max_horizon}" in oof]
    out = {}
    for h in ctx.bundle.horizons:
        y = oof[f"y_{h}"].to_numpy()
        m = ~np.isnan(y)
        risks = {a: oof.loc[m, f"oof__{a}__{h}"].to_numpy() for a in keep}
        out[h] = Fusion(keep, h).fit(risks, oof.loc[m, f"prior_{h}"].to_numpy(), y[m])
    ctx._cache[key] = out
    return out


def score_without_agent(ctx: EvalContext, table: str, drop: str) -> SystemOutput:
    fusion = _refit_without(ctx, drop)
    fam = {a: r for a, r in ctx.family(table).items() if a != drop}
    inputs = dict(fam)
    case = ctx.case(table)
    if case is not None and drop != CASE_AGENT:
        inputs[CASE_AGENT] = case
    prior = ctx.bundle.prior(ctx.models(table))
    p = {h: fusion[h].predict({a: r[h] for a, r in inputs.items()}, prior[h], "stacking")
         for h in ctx.bundle.horizons}
    from pdm_agents.models.bundle import monotone_in_horizon

    n = len(ctx.cfg.numeric_agents())
    h = ctx.cfg.max_horizon
    weights = dict(zip(fusion[h].agents, fusion[h].weights_, strict=True))
    return ctx.output(table, monotone_in_horizon(p), inputs, agent_cov=len(fam) / max(n, 1),
                      weights=weights)


def ablations(cfg: AppConfig) -> dict[str, Callable[[AppConfig], SystemDef]]:
    out: dict[str, Callable[[AppConfig], SystemDef]] = {}
    for agent in cfg.numeric_agents():
        out[f"no_{agent}"] = lambda c, a=agent: SystemDef(
            f"no_{a}", "ablation", lambda ctx, t, a=a: score_without_agent(ctx, t, a),
            f"Full system without the {a} agent (fusion refit without it).",
            drift_recalibration=True,
            graph={"agents": {a: {"enabled": False}}})
    out.update({
        "no_case_retrieval": lambda c: SystemDef(
            "no_case_retrieval", "ablation", lambda ctx, t: score_fusion(ctx, t, with_case=False),
            "Full system without case-memory retrieval.", drift_recalibration=True,
            graph={"features": {"use_case_retrieval": False}}),
        "no_knowledge_retrieval": lambda c: SystemDef(
            "no_knowledge_retrieval", "ablation",
            lambda ctx, t: score_fusion(ctx, t, with_case=True),
            "Full system without knowledge retrieval.", drift_recalibration=True,
            graph={"features": {"use_knowledge_retrieval": False}},
            note="knowledge snippets only ground explanations; numeric results equal the "
                 "full system, differences appear in the graph-run explanation metrics"),
        "no_planner": lambda c: SystemDef(
            "no_planner", "ablation", lambda ctx, t: score_fusion(ctx, t, with_case=True),
            "Full system without the LLM planner (every enabled agent runs in a fixed order).",
            drift_recalibration=True, graph={"features": {"use_planner": False}},
            note="the planner only orders and selects agents; numeric results equal the full "
                 "system, differences appear in the graph-run metrics"),
        "no_calibration": lambda c: SystemDef(
            "no_calibration", "ablation",
            lambda ctx, t: score_fusion(ctx, t, with_case=False, calibrated=False),
            "Raw (uncalibrated) agent scores pooled in log-odds space, no recalibration.",
            graph={"features": {"use_calibration": False}}),
        "majority_vote": lambda c: SystemDef(
            "majority_vote", "ablation",
            lambda ctx, t: score_fusion(ctx, t, with_case=True, method="majority_vote"),
            "Naive majority voting of agents instead of calibrated fusion.",
            graph={"models": {"fusion": {"method": "majority_vote"}}}),
        "no_cost_policy": lambda c: SystemDef(
            "no_cost_policy", "ablation", lambda ctx, t: score_fusion(ctx, t, with_case=True),
            f"Fixed probability threshold ({c.decision.fixed_threshold}) instead of the "
            "expected-cost policy.", cost_policy=False, drift_recalibration=True,
            graph={"features": {"use_cost_policy": False}}),
        "no_drift_recalibration": lambda c: SystemDef(
            "no_drift_recalibration", "ablation",
            lambda ctx, t: score_fusion(ctx, t, with_case=True),
            "Full system with static calibration (no drift-triggered recalibration).",
            drift_recalibration=False),
    })
    return out
