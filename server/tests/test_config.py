"""Configuration loading, interpolation, overrides and environment detection."""

from __future__ import annotations

import pytest

from pdm_agents.config import SERVER_ROOT, deep_merge, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.horizons == [7, 14, 30]
    assert cfg.decision.cost_ratios == [10, 50, 200]
    assert cfg.resolved_env in {"local", "colab"}
    expected_actions = {"monitor", "schedule_replacement", "replace_now", "escalate_to_human"}
    assert set(cfg.actions) == expected_actions
    assert "media_health" in cfg.numeric_agents()
    assert cfg.paths.knowledge_base == SERVER_ROOT / "knowledge_base"


def test_env_interpolation(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "some/model")
    cfg = load_config()
    assert cfg.llm.base_url == "https://example.test/v1"
    assert cfg.llm.model == "some/model"


def test_overrides_deep_merge_and_flags():
    cfg = load_config({"features": {"use_rag": False}, "decision": {"default_cost_ratio": 10}})
    assert cfg.decision.default_cost_ratio == 10
    assert cfg.decision.cost_ratios == [10, 50, 200]
    assert "case_retrieval" not in cfg.enabled_agents()
    assert "knowledge_retrieval" not in cfg.enabled_agents()


def test_colab_paths(monkeypatch):
    monkeypatch.setenv("PDM_ENV", "colab")
    cfg = load_config()
    assert cfg.resolved_env == "colab"
    assert cfg.paths.raw.as_posix().startswith("/content/drive/MyDrive/drive-failure-agents")
    # knowledge base ships with the code, not with the data
    assert cfg.paths.knowledge_base == SERVER_ROOT / "knowledge_base"


def test_invalid_action_horizon_rejected():
    with pytest.raises(ValueError, match="exposure_horizon"):
        load_config({"actions": {"monitor": {"exposure_horizon": 90}}})


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        load_config({"decision": {"not_a_key": 1}})


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 3}})
    assert out == {"a": {"b": 3, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
