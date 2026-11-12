"""Decision policy: expected-cost math and escalation rules."""

from __future__ import annotations

import numpy as np
import pytest

from pdm_agents.config import load_config
from pdm_agents.decision.cost import expected_costs, realized_costs
from pdm_agents.decision.policy import decide, decide_batch
from pdm_agents.schemas import DriftReport, FusedRisk


@pytest.fixture
def cfg():
    return load_config()


def test_expected_cost_math(cfg):
    costs = expected_costs({7: 0.02, 14: 0.05, 30: 0.10}, 50, cfg.actions)
    assert costs["monitor"] == pytest.approx(0.10 * 50)
    assert costs["schedule_replacement"] == pytest.approx(1.0 + 0.02 * 50)
    assert costs["replace_now"] == pytest.approx(1.5)
    assert costs["escalate_to_human"] == pytest.approx(0.25 + 1.5)


def test_lowest_cost_action_changes_with_cost_ratio(cfg):
    fused = FusedRisk(probabilities={7: 0.004, 14: 0.008, 30: 0.03}, uncertainty=0.1,
                      disagreement=0.0)
    assert decide(cfg, fused, 10).action == "monitor"  # 0.3 < 1.04 < 1.5
    assert decide(cfg, fused, 50).action == "schedule_replacement"  # 1.5 / 1.2 / 1.5
    assert decide(cfg, fused, 200).action == "replace_now"  # 6.0 / 1.8 / 1.5


def test_escalation_rules(cfg):
    base = {"probabilities": {7: 0.001, 14: 0.002, 30: 0.003}}
    d = decide(cfg, FusedRisk(**base, uncertainty=0.9, disagreement=0.0), 50)
    assert d.escalated and d.action == "escalate_to_human"
    assert any("uncertainty" in r for r in d.reasons)
    assert any("suggested action" in r for r in d.reasons)
    d = decide(cfg, FusedRisk(**base, uncertainty=0.1, disagreement=0.6), 50)
    assert d.escalated and any("disagreement" in r for r in d.reasons)
    alert = DriftReport(psi_by_family={"thermal": 0.4}, level="alert")
    d = decide(cfg, FusedRisk(**base, uncertainty=0.1, disagreement=0.0), 50, alert)
    assert d.escalated and any("drift" in r for r in d.reasons)
    no_drift = load_config({"features": {"use_drift": False}})
    assert not decide(no_drift, FusedRisk(**base, uncertainty=0.1, disagreement=0.0), 50,
                      alert).escalated
    calm = decide(cfg, FusedRisk(**base, uncertainty=0.1, disagreement=0.1), 50)
    assert not calm.escalated and calm.action == "monitor"


def test_drift_only_counts_for_families_the_fusion_uses(cfg):
    fused = FusedRisk(probabilities={7: 0.001, 14: 0.002, 30: 0.003}, uncertainty=0.1,
                      disagreement=0.0)
    thermal_drift = DriftReport(psi_by_family={"thermal": 1.3, "media_health": 0.02},
                                level="alert")
    weights = {"media_health": 0.9, "thermal": 0.0}
    assert not decide(cfg, fused, 50, thermal_drift, drift_weights=weights).escalated
    media_drift = DriftReport(psi_by_family={"thermal": 0.0, "media_health": 0.5},
                              level="alert")
    d = decide(cfg, fused, 50, media_drift, drift_weights=weights)
    assert d.escalated and any("media_health" in r for r in d.reasons)


def test_fixed_threshold_policy():
    cfg = load_config({"features": {"use_cost_policy": False}})
    probs = {7: np.array([0.6, 0.1, 0.1]), 14: np.array([0.7, 0.3, 0.1]),
             30: np.array([0.8, 0.6, 0.2])}
    _, base, _ = decide_batch(cfg, probs, 50)
    assert list(base) == ["replace_now", "schedule_replacement", "monitor"]


def test_realized_costs_and_escalation_outcomes(cfg):
    chosen = np.array(["monitor", "monitor", "replace_now", "escalate_to_human",
                       "escalate_to_human"], dtype=object)
    fallback = np.array(["monitor", "monitor", "replace_now", "monitor", "monitor"],
                        dtype=object)
    y = np.array([0, 1, 0, 1, 0.0])
    labels = {7: y, 14: y, 30: y}
    model = realized_costs(chosen, labels, 50, cfg.actions, fallback, "model")
    oracle = realized_costs(chosen, labels, 50, cfg.actions, fallback, "oracle")
    assert list(model) == [0, 50, 1.5, 0.25 + 50, 0.25]
    assert list(oracle) == [0, 50, 1.5, 0.25 + 1.5, 0.25]
    nan = {h: np.array([np.nan]) for h in (7, 14, 30)}
    censored = realized_costs(np.array(["monitor"], dtype=object), nan, 50, cfg.actions)
    assert np.isnan(censored[0])
