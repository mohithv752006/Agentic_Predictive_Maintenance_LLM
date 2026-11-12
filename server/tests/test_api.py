"""API routes respond with valid schemas (FakeLLM engine, synthetic data)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from pdm_agents.api.deps import AppState
from pdm_agents.api.main import create_app
from pdm_agents.config import load_config
from pdm_agents.graph.builder import PredictionEngine, load_resources
from pdm_agents.llm.client import LLMClient
from pdm_agents.llm.fake import FakeLLM
from pdm_agents.schemas import (
    DriveHistory,
    DriveListResponse,
    EscalationItem,
    FeedbackRecord,
    FleetStats,
    HealthResponse,
    PredictResponse,
    RuntimeSettingsView,
)
from tests.synthetic import synthetic_overrides


def _client(indexed_cfg, tmp_path, extra: dict | None = None) -> TestClient:
    over = synthetic_overrides(indexed_cfg.paths.raw.parent)
    over["paths"]["runtime"] = str(tmp_path / "runtime")
    # independent of a developer's server/.env pointing at a live LLM
    over["llm"] = {"base_url": "", "api_key": ""}
    if extra:
        from pdm_agents.config import deep_merge

        over = deep_merge(over, extra)
    cfg = load_config(over)
    engine = PredictionEngine(cfg, resources=load_resources(cfg),
                              llm=LLMClient(chat_model=FakeLLM(), model_name="fake"))
    return TestClient(create_app(cfg, AppState(cfg, engine=engine)))


@pytest.fixture
def client(indexed_cfg, tmp_path):
    with _client(indexed_cfg, tmp_path) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = HealthResponse.model_validate(r.json())
    assert h.data_ready and h.models_ready and h.indexes_ready
    assert h.llm.configured is False and h.status == "degraded"


def test_settings_roundtrip(client):
    view = RuntimeSettingsView.model_validate(client.get("/settings").json())
    assert set(view.enabled_agents) == set(view.available_agents)
    r = client.put("/settings", json={"enabled_agents": ["media_health", "thermal"],
                                      "default_cost_ratio": 10})
    assert r.status_code == 200
    assert r.json()["enabled_agents"] == ["media_health", "thermal"]
    assert client.put("/settings", json={"enabled_agents": ["nope"]}).status_code == 422
    r = client.put("/settings", json={"llm_base_url": "http://127.0.0.1:9/v1",
                                      "llm_api_key": "sk-test-123456"})
    assert r.json()["llm_api_key_hint"] == "...3456" and "sk-test" not in r.text
    t = client.post("/settings/test").json()
    assert t["configured"] and not t["reachable"]


def test_drives_list_and_history(client):
    r = client.get("/drives", params={"page_size": 5})
    assert r.status_code == 200
    body = DriveListResponse.model_validate(r.json())
    assert body.total > 5 and len(body.items) == 5 and body.models
    risks = [i.latest_risk for i in body.items]
    assert risks == sorted(risks, reverse=True)
    model = body.models[0]
    only = DriveListResponse.model_validate(client.get("/drives", params={"model": model}).json())
    assert all(i.model == model for i in only.items)
    serial = body.items[0].serial_number
    hist = DriveHistory.model_validate(client.get(f"/drives/{serial}").json())
    assert hist.dates and hist.series and all(len(s.values) == len(hist.dates)
                                              for s in hist.series)
    assert client.get("/drives/NOPE").status_code == 404
    stats = FleetStats.model_validate(client.get("/drives/stats").json())
    assert stats.total == body.total and stats.snapshot_available
    assert sum(b.count for b in stats.risk_histogram) <= stats.total
    assert sum(stats.by_action.values()) == stats.total


def test_predict_and_feedback(client, sample_serial):
    serial, as_of = sample_serial
    r = client.post("/predict", json={"serial_number": serial, "as_of_date": as_of})
    assert r.status_code == 200
    resp = PredictResponse.model_validate(r.json())
    assert resp.decision is not None and resp.llm_available
    assert client.get(f"/runs/{resp.run_id}").json()["run_id"] == resp.run_id
    fb = client.post("/feedback", json={"run_id": resp.run_id, "serial_number": serial,
                                        "verdict": "override", "final_action": "replace_now",
                                        "note": "spare available"})
    assert fb.status_code == 200
    FeedbackRecord.model_validate(fb.json())
    bad = client.post("/feedback", json={"run_id": "missing", "serial_number": serial,
                                         "verdict": "approve", "final_action": "monitor"})
    assert bad.status_code == 404
    assert client.post("/predict", json={"serial_number": "NOPE"}).status_code == 404


def test_predict_stream_sse(client, sample_serial):
    serial, as_of = sample_serial
    events = []
    with client.stream("GET", "/predict/stream",
                       params={"serial_number": serial, "as_of_date": as_of}) as r:
        assert r.status_code == 200
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event:
                events.append((event, line.split(":", 1)[1].strip()))
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "result"
    assert set(json.loads(events[0][1])["agents"]) >= {"media_health"}
    updates = [json.loads(d) for e, d in events if e == "update"]
    assert any(u["node"].startswith("agent_") for u in updates)
    PredictResponse.model_validate_json(events[-1][1])


def test_escalation_queue(indexed_cfg, tmp_path, sample_serial):
    always = {"decision": {"escalate_if_uncertainty_above": -1.0}}
    with _client(indexed_cfg, tmp_path, always) as c:
        serial, as_of = sample_serial
        resp = c.post("/predict", json={"serial_number": serial, "as_of_date": as_of}).json()
        assert resp["decision"]["escalated"]
        c.post("/feedback", json={"run_id": resp["run_id"], "serial_number": serial,
                                  "verdict": "approve", "final_action": "schedule_replacement"})
        items = [EscalationItem.model_validate(i) for i in c.get("/escalations").json()]
        assert items and items[0].run_id == resp["run_id"]
        assert items[0].feedback.final_action == "schedule_replacement"


def test_experiments_empty_ok(client):
    r = client.get("/experiments")
    assert r.status_code == 200 and isinstance(r.json(), list)
    assert client.get("/experiments/does-not-exist").status_code == 404
