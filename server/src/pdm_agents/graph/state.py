"""LangGraph state, per-agent task payload and run-time dependencies."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from pdm_agents.agents.base import AgentResources, SpecialistAgent
from pdm_agents.llm.client import LLMClient
from pdm_agents.schemas import (
    AgentFinding,
    Decision,
    DriftReport,
    DriveContext,
    Explanation,
    FusedRisk,
    KnowledgeSnippet,
    PredictRequest,
    RetrievedCase,
    TraceEvent,
)


class GraphState(TypedDict, total=False):
    run_id: str
    request: PredictRequest
    context: DriveContext | None
    plan: list[str]  # agent names to run
    findings: Annotated[list[AgentFinding], operator.add]  # reducer for parallel fan-in
    retrieved_cases: list[RetrievedCase]
    knowledge: list[KnowledgeSnippet]
    fused: FusedRisk | None
    drift: DriftReport | None
    decision: Decision | None
    explanation: Explanation | None
    attempts: int
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]  # UI timeline + audit trail


class AgentTask(TypedDict):
    """Payload sent to one agent node by the dispatcher (``Send``)."""

    agent: str
    context: DriveContext
    attempt: int


@dataclass
class GraphDeps:
    """Run-time context (not checkpointed): resources, agents for this run, LLM session."""

    res: AgentResources
    agents: dict[str, SpecialistAgent]
    llm: LLMClient | None
    cost_ratio: float


def latest_findings(findings: list[AgentFinding]) -> dict[str, AgentFinding]:
    """Most recent finding per agent (a replan appends newer findings)."""
    out: dict[str, AgentFinding] = {}
    for f in findings:
        out[f.agent] = f
    return out
