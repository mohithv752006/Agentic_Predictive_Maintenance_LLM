"""Schema round-trips and FakeLLM determinism."""

from __future__ import annotations

import json
from datetime import date

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from pdm_agents.llm.fake import FakeLLM, FakeLLMError
from pdm_agents.schemas import (
    AgentFinding,
    ExplanationDraft,
    FusedRisk,
    PlannerOutput,
    PredictRequest,
    PredictResponse,
)


def test_horizon_keys_round_trip_through_json():
    fused = FusedRisk(probabilities={7: 0.1, 30: 0.4}, uncertainty=0.2, disagreement=0.1)
    again = FusedRisk.model_validate_json(fused.model_dump_json())
    assert again.probabilities == {7: 0.1, 30: 0.4}


def test_predict_request_normalizes_horizons():
    req = PredictRequest(serial_number="X", horizons=[30, 7, 7])
    assert req.horizons == [7, 30]
    with pytest.raises(ValueError):
        PredictRequest(serial_number="X", horizons=[0])


def test_finding_ids_unique_and_status_validated():
    a = AgentFinding(agent="thermal", serial_number="X", summary="s", status="ok")
    b = AgentFinding(agent="thermal", serial_number="X", summary="s", status="ok")
    assert a.id != b.id and a.id.startswith("F-")
    with pytest.raises(ValueError):
        AgentFinding(agent="thermal", serial_number="X", summary="s", status="weird")


def test_predict_response_minimal():
    resp = PredictResponse(
        run_id="r",
        request=PredictRequest(serial_number="X", as_of_date=date(2025, 1, 1)),
        fused=None,
        decision=None,
        explanation=None,
        findings=[],
        retrieved_cases=[],
        drift=None,
        llm_available=False,
        latency_ms=1.0,
        llm_calls=0,
    )
    assert json.loads(resp.model_dump_json())["llm_available"] is False


def _ask(llm: FakeLLM, system: str, user: str) -> str:
    return str(llm.invoke([SystemMessage(system), HumanMessage(user)]).content)


def test_fake_llm_planner_and_explainer_are_deterministic():
    llm = FakeLLM()
    out = _ask(llm, "ROLE: planner", "AVAILABLE_AGENTS: thermal, mechanical")
    assert PlannerOutput.model_validate_json(out).agents == ["thermal", "mechanical"]
    user = "findings F-0123abcd and F-0123abce, case C-00aa11bb, knowledge K-media_health-0"
    draft = ExplanationDraft.model_validate_json(_ask(llm, "ROLE: explainer", user))
    assert draft.cited_finding_ids == ["F-0123abcd", "F-0123abce"]
    assert draft.cited_case_ids == ["C-00aa11bb"]
    assert draft.cited_knowledge_ids == ["K-media_health-0"]
    assert _ask(llm, "ROLE: explainer", user) == _ask(llm, "ROLE: explainer", user)


def test_fake_llm_failure_modes():
    assert _ask(FakeLLM(mode="think"), "ROLE: planner", "AVAILABLE_AGENTS: a").startswith("<think>")
    once = FakeLLM(mode="malformed_once")
    assert "not valid json" in _ask(once, "ROLE: planner", "x")
    assert json.loads(_ask(once, "ROLE: planner", "AVAILABLE_AGENTS: a"))["agents"] == ["a"]
    with pytest.raises(FakeLLMError):
        _ask(FakeLLM(mode="raise"), "ROLE: planner", "x")
