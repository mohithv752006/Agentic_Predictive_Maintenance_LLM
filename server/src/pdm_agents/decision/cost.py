"""Expected-cost computation from calibrated probabilities and the action table.

Units: one planned replacement = 1. A missed failure costs ``cost_ratio``.
See ``configs/actions.yaml`` for the definition of each action.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pdm_agents.config import ActionSpec

EscalationOutcome = Literal["model", "oracle"]


def _non_delegating(actions: dict[str, ActionSpec]) -> list[str]:
    return [a for a, s in actions.items() if not s.delegates]


def expected_costs_batch(
    probs: dict[int, np.ndarray], cost_ratio: float, actions: dict[str, ActionSpec]
) -> dict[str, np.ndarray]:
    """Expected cost of every action for every row."""
    n = len(next(iter(probs.values())))
    out: dict[str, np.ndarray] = {}
    for name, spec in actions.items():
        if spec.delegates:
            continue
        cost = np.full(n, spec.fixed_cost, dtype="float64")
        if spec.exposure_horizon is not None and spec.exposure_fraction > 0:
            cost = cost + spec.exposure_fraction * np.asarray(probs[spec.exposure_horizon]) \
                * cost_ratio
        out[name] = cost
    best = np.min(np.column_stack(list(out.values())), axis=1)
    for name, spec in actions.items():
        if spec.delegates:
            out[name] = spec.fixed_cost + best
    return out


def expected_costs(
    probs: dict[int, float], cost_ratio: float, actions: dict[str, ActionSpec]
) -> dict[str, float]:
    """Expected cost of every action for one drive."""
    batch = expected_costs_batch({h: np.array([p]) for h, p in probs.items()}, cost_ratio,
                                 actions)
    return {a: round(float(c[0]), 6) for a, c in batch.items()}


def cheapest_action(costs: dict[str, np.ndarray], actions: dict[str, ActionSpec]) -> np.ndarray:
    """Name of the lowest-expected-cost non-delegating action per row."""
    names = _non_delegating(actions)
    matrix = np.column_stack([costs[a] for a in names])
    return np.asarray(names, dtype=object)[np.argmin(matrix, axis=1)]


def realized_costs(
    chosen: np.ndarray,
    labels: dict[int, np.ndarray],
    cost_ratio: float,
    actions: dict[str, ActionSpec],
    fallback_action: np.ndarray | None = None,
    escalation_outcome: EscalationOutcome = "model",
) -> np.ndarray:
    """Cost actually incurred given true outcomes (``labels[h]`` = failed within h days).

    For an escalated row the reviewer's fixed cost is added to the cost of the action
    the reviewer takes: ``model`` = the policy's own best action (conservative),
    ``oracle`` = the best action in hindsight (optimistic). Rows whose needed label
    is censored (NaN) get NaN.
    """
    chosen = np.asarray(chosen, dtype=object)
    n = len(chosen)
    per_action: dict[str, np.ndarray] = {}
    for name, spec in actions.items():
        if spec.delegates:
            continue
        cost = np.full(n, spec.fixed_cost, dtype="float64")
        if spec.exposure_horizon is not None and spec.exposure_fraction > 0:
            y = np.asarray(labels[spec.exposure_horizon], dtype="float64")
            cost = cost + spec.exposure_fraction * y * cost_ratio  # NaN propagates
        per_action[name] = cost
    out = np.full(n, np.nan)
    for name, cost in per_action.items():
        m = chosen == name
        out[m] = cost[m]
    hindsight = np.min(np.column_stack(list(per_action.values())), axis=1)
    for name, spec in actions.items():
        if not spec.delegates:
            continue
        m = chosen == name
        if not m.any():
            continue
        if escalation_outcome == "oracle" or fallback_action is None:
            out[m] = spec.fixed_cost + hindsight[m]
        else:
            fb = np.asarray(fallback_action, dtype=object)
            follow = np.full(n, np.nan)
            for a, cost in per_action.items():
                sel = fb == a
                follow[sel] = cost[sel]
            out[m] = spec.fixed_cost + follow[m]
    return out
