"""Each agent returns a valid AgentFinding with FakeLLM, offline, and when a tool raises."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest
from langchain_core.tools import tool

from pdm_agents.agents.registry import build_agents, load_agent_class
from pdm_agents.graph.nodes import build_context
from pdm_agents.llm.client import LLMClient
from pdm_agents.llm.fake import FakeLLM
from pdm_agents.schemas import AgentFinding


@pytest.fixture(scope="module")
def ctx(resources, sample_serial):
    serial, as_of = sample_serial
    return build_context(resources, serial, date.fromisoformat(as_of))


def test_context_uses_only_past_data(ctx):
    assert max(ctx.window.dates) <= ctx.as_of_date
    assert ctx.features["f__ctx__n_obs"] == len(ctx.window.dates)
    assert ctx.base_rates and ctx.model_seen_in_training


@pytest.mark.parametrize("llm_mode", ["ok", "offline"])
def test_every_agent_returns_valid_finding(resources, ctx, llm_mode):
    llm = None if llm_mode == "offline" else LLMClient(chat_model=FakeLLM(), model_name="fake")
    agents = build_agents(resources.cfg, resources)
    assert set(agents) == set(resources.cfg.enabled_agents())
    for name, agent in agents.items():
        f = agent.run(ctx, llm)
        AgentFinding.model_validate(f.model_dump())
        assert f.agent == name and f.serial_number == ctx.serial_number
        assert f.status == ("ok" if llm else "degraded"), (name, f.error)
        assert 0 <= f.confidence <= 1 and f.summary
        if resources.cfg.agents[name].kind == "numeric":
            assert set(f.risk) == set(resources.cfg.horizons)
            assert all(0 <= p <= 1 for p in f.risk.values())
        if name == "case_retrieval":
            cases = [e for e in f.evidence if e.kind == "case"]
            assert cases and all(e.id.startswith("C-") for e in cases)
        if f.risk:  # P(fail within h) cannot decrease as h grows
            ps = [f.risk[h] for h in sorted(f.risk)]
            assert ps == sorted(ps), (name, f.risk)
        if name == "knowledge_retrieval":
            assert f.evidence and all(e.id.startswith("K-") for e in f.evidence)


@tool("trend_analysis")
def broken_trend(column: str, values: list) -> dict:
    """Always fails."""
    raise RuntimeError("tool exploded")


@tool("case_retrieval")
def broken_cases(serial_number: str, model: str, features: dict, k: int = 5) -> list:
    """Always fails."""
    raise RuntimeError("index unavailable")


def test_agent_failure_is_contained(resources, ctx):
    tools = {**resources.tools, "trend_analysis": broken_trend, "case_retrieval": broken_cases}
    broken = dataclasses.replace(resources, tools=tools)
    agents = build_agents(resources.cfg, broken, ["media_health", "case_retrieval", "thermal"])
    assert len(agents) == 3
    for agent in agents.values():
        f = agent.run(ctx, None)  # must not raise
        assert f.status == "failed"
        assert "exploded" in f.error or "unavailable" in f.error


def test_registry_rejects_non_agents():
    with pytest.raises(TypeError):
        load_agent_class("pdm_agents.schemas:AgentFinding")
    with pytest.raises(ValueError):
        load_agent_class("no_colon_here")
