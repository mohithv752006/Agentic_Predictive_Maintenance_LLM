"""Action selection: lowest expected cost, plus abstain/escalate rules.

Escalate to a human when the fused prediction is too uncertain, when specialist
agents disagree too much, or when drift for the drive's model is at alert level
(each rule can be switched off through config). With ``use_cost_policy: false``
a fixed probability threshold replaces the cost rule (ablation baseline).
"""

from __future__ import annotations

import numpy as np

from pdm_agents.config import AppConfig
from pdm_agents.decision.cost import cheapest_action, expected_costs_batch
from pdm_agents.decision.drift import relevant_level
from pdm_agents.schemas import Decision, DriftReport, FusedRisk

ESCALATE = "escalate_to_human"


def threshold_actions(probs: dict[int, np.ndarray], threshold: float) -> np.ndarray:
    """Fixed-threshold baseline: replace now if P(7d) >= t, schedule if P(30d) >= t."""
    h_short, h_long = min(probs), max(probs)
    p7, p30 = np.asarray(probs[h_short]), np.asarray(probs[h_long])
    return np.where(p7 >= threshold, "replace_now",
                    np.where(p30 >= threshold, "schedule_replacement", "monitor")).astype(object)


def decide_batch(
    cfg: AppConfig,
    probs: dict[int, np.ndarray],
    cost_ratio: float,
    uncertainty: np.ndarray | None = None,
    disagreement: np.ndarray | None = None,
    drift_alert: np.ndarray | None = None,
    allow_escalation: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Vectorized policy. Returns (final action, pre-escalation action, expected costs)."""
    costs = expected_costs_batch(probs, cost_ratio, cfg.actions)
    if cfg.features.use_cost_policy:
        base = cheapest_action(costs, cfg.actions)
    else:
        base = threshold_actions(probs, cfg.decision.fixed_threshold)
    n = len(base)
    esc = np.zeros(n, dtype=bool)
    if allow_escalation and ESCALATE in cfg.actions:
        if uncertainty is not None:
            esc |= np.asarray(uncertainty) > cfg.decision.escalate_if_uncertainty_above
        if disagreement is not None:
            esc |= np.asarray(disagreement) > cfg.decision.escalate_if_agent_disagreement_above
        if drift_alert is not None and cfg.features.use_drift:
            esc |= np.asarray(drift_alert, dtype=bool)
    final = np.where(esc, ESCALATE, base).astype(object)
    return final, base, costs


def decide(cfg: AppConfig, fused: FusedRisk, cost_ratio: float,
           drift: DriftReport | None = None, failed_agents: list[str] | None = None,
           drift_weights: dict[str, float] | None = None) -> Decision:
    """Decision for one drive, with human-readable reasons.

    ``drift_weights`` (fusion weight per agent family) restricts the drift-alert rule to
    families the fusion relies on (see ``drift.relevant_level``).
    """
    probs = {h: np.array([p]) for h, p in fused.probabilities.items()}
    alert = drift is not None and relevant_level(drift, drift_weights, cfg) == "alert"
    if alert and drift_weights is not None:
        drift = drift.model_copy(update={"psi_by_family": {
            f: v for f, v in drift.psi_by_family.items()
            if drift_weights.get(f, 0.0) >= cfg.drift.min_family_weight}})
    final, base, costs = decide_batch(
        cfg, probs, cost_ratio, np.array([fused.uncertainty]), np.array([fused.disagreement]),
        np.array([alert]),
    )
    action, base_action = str(final[0]), str(base[0])
    exp = {a: round(float(c[0]), 6) for a, c in costs.items()}
    reasons: list[str] = []
    if cfg.features.use_cost_policy:
        reasons.append(f"lowest expected cost: {base_action} ({exp[base_action]:.3f} units at "
                       f"cost ratio {cost_ratio:g})")
    else:
        reasons.append(f"fixed threshold {cfg.decision.fixed_threshold:g}: {base_action}")
    d = cfg.decision
    if action == ESCALATE:
        if fused.uncertainty > d.escalate_if_uncertainty_above:
            reasons.append(f"uncertainty {fused.uncertainty:.2f} > "
                           f"{d.escalate_if_uncertainty_above:g}")
        if fused.disagreement > d.escalate_if_agent_disagreement_above:
            reasons.append(f"agent disagreement {fused.disagreement:.2f} > "
                           f"{d.escalate_if_agent_disagreement_above:g}")
        if alert and cfg.features.use_drift:
            worst = max(drift.psi_by_family.items(), key=lambda kv: kv[1])
            reasons.append(f"drift alert: PSI {worst[1]:.2f} for {worst[0]}")
        reasons.append(f"suggested action for the reviewer: {base_action}")
    if failed_agents:
        reasons.append(f"agents failed: {', '.join(failed_agents)}")
    return Decision(action=action, expected_costs=exp, cost_ratio=cost_ratio,
                    escalated=action == ESCALATE, reasons=reasons)
