"""Build the LangGraph StateGraph from config and run predictions.

Graph: load_context -> planner -> dispatch -(Send, parallel)-> agent_<name>...
       -> retrieve -> fuse -> [replan: planner | decide] -> explain -> END

One node is added per enabled agent in ``agents.yaml``; nothing here names an
agent, so adding an agent needs no change in this module.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from pdm_agents.agents.base import AgentResources
from pdm_agents.agents.registry import build_agents
from pdm_agents.config import AppConfig
from pdm_agents.data.store import Store
from pdm_agents.decision.drift import DriftService
from pdm_agents.graph import nodes
from pdm_agents.graph.checkpoint import RunStore, make_checkpointer
from pdm_agents.graph.state import GraphDeps, GraphState, latest_findings
from pdm_agents.llm.client import LLMClient, LLMMode, create_llm
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.bundle import load_bundle
from pdm_agents.rag.case_memory import CaseMemory
from pdm_agents.rag.knowledge_index import KnowledgeIndex, index_dir
from pdm_agents.schemas import PredictRequest, PredictResponse
from pdm_agents.tools.registry import build_tools

log = get_logger(__name__)


def load_resources(cfg: AppConfig, seed: int | None = None) -> AgentResources:
    """Load store, model bundle, indexes and tools; missing artifacts are tolerated."""
    store = Store(cfg)
    bundle = load_bundle(cfg, seed) if store.ready else None
    cases = knowledge = None
    idx = index_dir(cfg)
    if (idx / "cases" / "indexes.json").exists():
        cases = CaseMemory.load(idx / "cases")
    if (idx / "knowledge" / "snippets.json").exists():
        knowledge = KnowledgeIndex.load(cfg, idx / "knowledge")
        knowledge.embedder.encode(["warm-up"], query=True)  # load the embedder once, up front
    drift = DriftService(cfg, store, bundle.drift) if bundle is not None else None
    tools = build_tools(cfg, bundle, cases, knowledge)
    return AgentResources(cfg=cfg, store=store, bundle=bundle, cases=cases,
                          knowledge=knowledge, tools=tools, drift=drift)


def build_graph(cfg: AppConfig, checkpointer: Any | None = None):
    """Compile the graph; one ``agent_<name>`` node per enabled agent."""
    g = StateGraph(GraphState, context_schema=GraphDeps)
    g.add_node("load_context", nodes.load_context)
    g.add_node("planner", nodes.planner)
    g.add_node("dispatch", nodes.dispatch)
    agent_nodes = []
    for name in cfg.enabled_agents():
        node = f"agent_{name}"
        g.add_node(node, nodes.make_agent_node(name))
        g.add_edge(node, "retrieve")
        agent_nodes.append(node)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("fuse", nodes.fuse)
    g.add_node("decide", nodes.decide)
    g.add_node("explain", nodes.explain)

    g.add_edge(START, "load_context")
    g.add_conditional_edges("load_context", nodes.route_after_context,
                            {"planner": "planner", "end": END})
    g.add_edge("planner", "dispatch")
    g.add_conditional_edges("dispatch", nodes.route_dispatch, [*agent_nodes, "retrieve"])
    g.add_edge("retrieve", "fuse")
    g.add_conditional_edges("fuse", nodes.route_after_fuse,
                            {"planner": "planner", "decide": "decide"})
    g.add_edge("decide", "explain")
    g.add_edge("explain", END)
    return g.compile(checkpointer=checkpointer)


class PredictionEngine:
    """Runs predictions end to end; owns resources, the compiled graph and the LLM client."""

    def __init__(self, cfg: AppConfig, resources: AgentResources | None = None,
                 llm_mode: LLMMode = "live", llm: LLMClient | None = None,
                 checkpoint: bool = True, record_runs: bool = True) -> None:
        self.cfg = cfg
        self.res = resources or load_resources(cfg)
        self.llm = llm if llm is not None else create_llm(cfg.llm, llm_mode)
        self.graph = build_graph(cfg, make_checkpointer(cfg) if checkpoint else None)
        self.runs = RunStore.for_config(cfg) if record_runs else None
        self.enabled_override: list[str] | None = None
        self.default_cost_ratio = cfg.decision.default_cost_ratio

    def set_llm(self, llm: LLMClient | None) -> None:
        self.llm = llm

    def _agents_for(self, request: PredictRequest) -> dict:
        names = request.enabled_agents or self.enabled_override
        return build_agents(self.cfg, self.res, names)

    def stream(self, request: PredictRequest) -> Iterator[dict]:
        """Yield ``{"type": "event", ...}`` per graph update, then ``{"type": "result"}``."""
        bad = [h for h in request.horizons if h not in self.cfg.horizons]
        if bad:
            raise ValueError(f"unsupported horizons {bad}; configured: {self.cfg.horizons}")
        run_id = uuid.uuid4().hex[:12]
        session = None if request.offline or self.llm is None else self.llm.session()
        deps = GraphDeps(res=self.res, agents=self._agents_for(request), llm=session,
                         cost_ratio=request.cost_ratio or self.default_cost_ratio)
        state: GraphState = {"run_id": run_id, "request": request, "context": None,
                             "plan": [], "findings": [], "retrieved_cases": [],
                             "knowledge": [], "fused": None, "drift": None, "decision": None,
                             "explanation": None, "attempts": 0, "errors": [], "trace": []}
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 50}
        t0 = time.perf_counter()
        final: dict[str, Any] = dict(state)
        for update in self.graph.stream(state, config, context=deps, stream_mode="updates"):
            for node, delta in update.items():
                if not isinstance(delta, dict):
                    continue
                for key, value in delta.items():
                    if key in ("findings", "errors", "trace"):
                        final[key] = final.get(key, []) + list(value)
                    else:
                        final[key] = value
                yield {"type": "event", "node": node, "run_id": run_id,
                       "trace": [t.model_dump(mode="json") for t in delta.get("trace", [])],
                       "findings": [f.model_dump(mode="json")
                                    for f in delta.get("findings", [])]}
        latency = (time.perf_counter() - t0) * 1000
        resp = self._response(run_id, request, final, session, latency)
        if self.runs is not None:
            self.runs.save(resp)
        yield {"type": "result", "run_id": run_id, "data": resp}

    def predict(self, request: PredictRequest) -> PredictResponse:
        result = None
        for event in self.stream(request):
            if event["type"] == "result":
                result = event["data"]
        assert result is not None
        return result

    def _response(self, run_id: str, request: PredictRequest, final: dict,
                  session: LLMClient | None, latency_ms: float) -> PredictResponse:
        ctx = final.get("context")
        summary: dict[str, Any] = {}
        if ctx is not None:
            summary = {"model": ctx.model, "capacity_bytes": ctx.capacity_bytes,
                       "as_of_date": ctx.as_of_date.isoformat(),
                       "window_days_observed": len(ctx.window.dates),
                       "window_days": self.cfg.data.window_days,
                       "model_seen_in_training": ctx.model_seen_in_training,
                       "base_rates": {str(h): p for h, p in ctx.base_rates.items()},
                       "missing_attributes": ctx.missing_attributes,
                       "attempts": final.get("attempts", 0)}
        return PredictResponse(
            run_id=run_id,
            request=request,
            context_summary=summary,
            plan=final.get("plan", []),
            fused=final.get("fused"),
            decision=final.get("decision"),
            explanation=final.get("explanation"),
            findings=list(latest_findings(final.get("findings", [])).values()),
            retrieved_cases=final.get("retrieved_cases", []),
            knowledge=final.get("knowledge", []),
            drift=final.get("drift"),
            llm_available=bool(session is not None and session.available and session.calls > 0
                               and session.failures < session.calls),
            latency_ms=round(latency_ms, 1),
            llm_calls=session.calls if session else 0,
            errors=final.get("errors", []),
            trace=final.get("trace", []),
        )
