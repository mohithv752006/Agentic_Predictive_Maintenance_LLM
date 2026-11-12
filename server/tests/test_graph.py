"""LangGraph end to end: FakeLLM, one failing agent, replanning, LLM offline, extension."""

from __future__ import annotations

from datetime import date

import pytest

from pdm_agents.agents.media_health import MediaHealthAgent
from pdm_agents.agents.thermal import ThermalAgent
from pdm_agents.config import load_config
from pdm_agents.graph.builder import PredictionEngine, load_resources
from pdm_agents.llm.client import LLMClient
from pdm_agents.llm.fake import FakeLLM
from pdm_agents.schemas import PredictRequest, PredictResponse
from tests.synthetic import synthetic_overrides


def _engine(resources, mode="ok", **kw):
    llm = LLMClient(chat_model=FakeLLM(mode=mode), model_name="fake")
    return PredictionEngine(resources.cfg, resources=resources, llm=llm, **kw)


def _req(sample_serial, **kw):
    serial, as_of = sample_serial
    return PredictRequest(serial_number=serial, as_of_date=date.fromisoformat(as_of), **kw)


def test_end_to_end_with_fake_llm(resources, sample_serial):
    engine = _engine(resources)
    resp = engine.predict(_req(sample_serial))
    PredictResponse.model_validate(resp.model_dump())
    cfg = resources.cfg
    assert resp.llm_available and resp.llm_calls > 0
    assert {f.agent for f in resp.findings} == set(cfg.enabled_agents())
    assert all(f.status == "ok" for f in resp.findings)
    assert set(resp.fused.probabilities) == set(cfg.horizons)
    assert resp.decision.action in cfg.actions
    assert resp.explanation.source == "llm"
    assert set(resp.explanation.cited_finding_ids) <= {f.id for f in resp.findings}
    assert set(resp.explanation.cited_case_ids) <= {c.id for c in resp.retrieved_cases}
    nodes = [t.node for t in resp.trace]
    for n in ("load_context", "planner", "retrieve", "fuse", "decide", "explain"):
        assert n in nodes
    assert sum(n.startswith("agent_") for n in nodes) == len(cfg.enabled_agents())
    assert nodes.count("retrieve") == 1  # fan-in: retrieve runs once after all agents
    state = engine.graph.get_state({"configurable": {"thread_id": resp.run_id}})
    assert state.values["decision"].action == resp.decision.action
    assert engine.runs.get(resp.run_id).run_id == resp.run_id


def test_stream_emits_agent_events_before_result(resources, sample_serial):
    events = list(_engine(resources).stream(_req(sample_serial)))
    assert events[-1]["type"] == "result"
    agent_events = [e for e in events[:-1] if e["node"].startswith("agent_")]
    assert agent_events and all(e["findings"] for e in agent_events)


def test_one_agent_failing(resources, sample_serial, monkeypatch):
    def boom(self, ctx):
        raise RuntimeError("media tool down")

    monkeypatch.setattr(MediaHealthAgent, "run_tools", boom)
    resp = _engine(resources).predict(_req(sample_serial))
    by_agent = {f.agent: f for f in resp.findings}
    assert by_agent["media_health"].status == "failed"
    assert "media_health" not in resp.fused.contributing_agents
    assert resp.decision is not None and resp.explanation is not None
    assert resp.context_summary["attempts"] == 1  # a single failure does not replan
    assert any("media_health" in e for e in resp.errors)


def test_two_failures_trigger_one_replan(resources, sample_serial, monkeypatch):
    def boom(self, ctx):
        raise RuntimeError("down")

    monkeypatch.setattr(MediaHealthAgent, "run_tools", boom)
    monkeypatch.setattr(ThermalAgent, "run_tools", boom)
    resp = _engine(resources).predict(_req(sample_serial))
    assert resp.context_summary["attempts"] == 2
    assert [t.node for t in resp.trace].count("planner") == 2
    assert resp.decision is not None


def test_llm_offline_degrades_gracefully(resources, sample_serial):
    resp = _engine(resources, mode="raise").predict(_req(sample_serial))
    assert resp.llm_available is False
    assert resp.decision is not None and resp.fused is not None
    assert resp.explanation.source == "template"
    assert any(f.status == "degraded" for f in resp.findings)
    assert all(f.status in ("degraded", "ok") for f in resp.findings)
    assert resp.explanation.cited_finding_ids  # template still cites real ids
    offline = _engine(resources).predict(_req(sample_serial, offline=True))
    assert offline.llm_calls == 0 and offline.llm_available is False


def test_unknown_drive_and_bad_horizon(resources):
    resp = _engine(resources).predict(PredictRequest(serial_number="NOPE"))
    assert resp.fused is None and resp.decision is None
    assert any("unknown drive" in e for e in resp.errors)
    with pytest.raises(ValueError):
        _engine(resources).predict(PredictRequest(serial_number="X", horizons=[90]))


def test_dummy_agent_added_with_one_file_and_one_yaml_entry(indexed_cfg, sample_serial):
    over = synthetic_overrides(indexed_cfg.paths.raw.parent)
    over["agents"] = {"dummy": {"class": "tests.dummy_agent:DummyAgent", "enabled": True,
                                "kind": "context", "description": "Counts observed days."}}
    cfg = load_config(over)
    engine = PredictionEngine(cfg, resources=load_resources(cfg),
                              llm=LLMClient(chat_model=FakeLLM(), model_name="fake"))
    resp = engine.predict(_req(sample_serial))
    dummy = next(f for f in resp.findings if f.agent == "dummy")
    assert dummy.status == "ok"
    assert "agent_dummy" in [t.node for t in resp.trace]
