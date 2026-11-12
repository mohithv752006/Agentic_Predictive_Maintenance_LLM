"""LLM client: think-stripping, structured output + repair, circuit breaker, health."""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from pdm_agents.config import load_config
from pdm_agents.llm.client import LLMClient, LLMUnavailableError, create_llm, strip_think
from pdm_agents.llm.fake import FakeLLM
from pdm_agents.llm.health import check_health
from pdm_agents.llm.structured import (
    extract_json_object,
    load_prompt,
    parse_structured,
    structured_call,
)
from pdm_agents.schemas import AgentInterpretation, ExplanationDraft, PlannerOutput

PLANNER_VARS = {
    "serial_number": "S1",
    "drive_model": "M",
    "as_of_date": "2025-01-01",
    "available_agents": "media_health, thermal",
    "feature_summary": "smart_5_raw rising",
    "previous_attempt": "none",
}


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<think>hmm</think>\n\nanswer", "answer"),
        ("reasoning only closes</think>answer", "answer"),
        ("plain", "plain"),
        ("<think>a</think>x<think>b</think>y", "xy"),
    ],
)
def test_strip_think(raw, clean):
    assert strip_think(raw) == clean


def test_extract_json_object_handles_prose_and_braces_in_strings():
    text = 'Sure! {"a": "x}{y", "b": {"c": 1}} trailing {"other": 2}'
    assert extract_json_object(text) == '{"a": "x}{y", "b": {"c": 1}}'
    assert extract_json_object("no json here") is None
    out = parse_structured('```json\n{"agents": ["a"], "rationale": "r"}\n```', PlannerOutput)
    assert out.agents == ["a"]


def test_all_prompts_render():
    load_prompt("planner").format_messages(**PLANNER_VARS, format_instructions="F")
    load_prompt("specialist").format_messages(
        agent_name="thermal", agent_description="d", serial_number="S", drive_model="M",
        window_start="a", window_end="b", risk_hint="low", evidence_block="E",
        format_instructions="F")
    load_prompt("explainer").format_messages(
        serial_number="S", drive_model="M", as_of_date="d", decision_block="x", fused_block="x",
        findings_block="x", cases_block="x", knowledge_block="x", format_instructions="F")
    load_prompt("repair").format_messages(
        error="e", previous_answer="p", original_prompt="o", format_instructions="F")


def _client(mode: str) -> LLMClient:
    return LLMClient(chat_model=FakeLLM(mode=mode), model_name="fake")


def test_structured_ok_and_think_stripped():
    for mode in ("ok", "think"):
        llm = _client(mode)
        res = structured_call(llm, "planner", PLANNER_VARS, PlannerOutput)
        assert res.status == "ok"
        assert res.value.agents == ["media_health", "thermal"]
        assert llm.calls == 1


def test_structured_repair_path():
    llm = _client("malformed_once")
    res = structured_call(llm, "planner", PLANNER_VARS, PlannerOutput)
    assert res.status == "repaired"
    assert res.value.agents == ["media_health", "thermal"]
    assert llm.calls == 2


def test_structured_fails_after_one_repair():
    llm = _client("malformed")
    res = structured_call(llm, "planner", PLANNER_VARS, PlannerOutput)
    assert res.status == "failed" and res.value is None
    assert llm.calls == 2


def test_unreachable_trips_circuit_breaker():
    llm = _client("raise")
    res = structured_call(llm, "planner", PLANNER_VARS, PlannerOutput)
    assert res.status == "unavailable"
    assert llm.available is False
    with pytest.raises(LLMUnavailableError):
        llm.invoke([HumanMessage("x")])
    assert llm.calls == 1  # fails fast, no second network attempt
    assert structured_call(None, "planner", PLANNER_VARS, PlannerOutput).status == "unavailable"


def test_session_has_fresh_counters():
    base = _client("raise")
    structured_call(base, "planner", PLANNER_VARS, PlannerOutput)
    sess = base.session()
    assert sess.available and sess.calls == 0 and sess.chat_model is base.chat_model


def test_specialist_and_explainer_schemas_with_fake():
    llm = _client("ok")
    res = structured_call(
        llm, "specialist",
        {"agent_name": "media_health", "agent_description": "d", "serial_number": "S",
         "drive_model": "M", "window_start": "a", "window_end": "b", "risk_hint": "high",
         "evidence_block": "[E-media_health-0] slope 3.2\n[E-media_health-1] max 48"},
        AgentInterpretation)
    assert res.ok and res.value.concern == "high"
    assert res.value.key_evidence_ids == ["E-media_health-0", "E-media_health-1"]
    res = structured_call(
        llm, "explainer",
        {"serial_number": "S", "drive_model": "M", "as_of_date": "d", "decision_block": "x",
         "fused_block": "x", "findings_block": "[F-0123abcd] media", "cases_block": "[C-00aa11bb]",
         "knowledge_block": "[K-smart_5-0] text"},
        ExplanationDraft)
    assert res.ok and res.value.cited_finding_ids == ["F-0123abcd"]


def test_create_llm_modes():
    cfg = load_config({"llm": {"base_url": ""}}).llm
    assert create_llm(cfg, "offline") is None
    assert create_llm(cfg, "live") is None  # nothing configured
    assert isinstance(create_llm(cfg, "fake").chat_model, FakeLLM)


def test_health_unconfigured_and_unreachable():
    cfg = load_config({"llm": {"base_url": ""}}).llm
    assert check_health(cfg).configured is False
    cfg = load_config({"llm": {"base_url": "http://127.0.0.1:9/v1", "health_timeout_s": 2}}).llm
    h = check_health(cfg)
    assert h.configured and not h.reachable and h.error


@pytest.mark.live_llm
@pytest.mark.skipif(not os.environ.get("LLM_BASE_URL"), reason="LLM_BASE_URL not set")
def test_live_endpoint():
    cfg = load_config().llm
    health = check_health(cfg)
    assert health.reachable, health.error
    llm = create_llm(cfg, "live")
    res = structured_call(llm, "planner", PLANNER_VARS, PlannerOutput)
    assert res.ok, res.error
    assert set(res.value.agents) <= {"media_health", "thermal"}
