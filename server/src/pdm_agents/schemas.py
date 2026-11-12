"""All Pydantic data contracts shared by the pipeline, graph, API and client.

Horizon-keyed dicts use ``int`` keys (days); JSON serializes them as strings and
Pydantic parses them back. Action names are plain strings validated against
``configs/actions.yaml`` at decision time, so the action set stays config-driven.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AgentStatus = Literal["ok", "degraded", "failed"]
DriftLevel = Literal["ok", "warn", "alert"]
TraceStatus = Literal["started", "ok", "degraded", "failed", "skipped", "info"]
EvidenceKind = Literal["metric", "trend", "model", "case", "knowledge", "context", "note"]
FeedbackVerdict = Literal["approve", "override"]


def new_id(prefix: str) -> str:
    """Short unique id with a readable prefix, e.g. ``F-1a2b3c4d``."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- core


class PredictRequest(Contract):
    serial_number: str
    as_of_date: date | None = None  # None = latest snapshot available for the drive
    horizons: list[int] = Field(default_factory=lambda: [7, 14, 30])
    cost_ratio: float | None = None  # None = config default
    enabled_agents: list[str] | None = None  # None = all enabled agents
    offline: bool = False  # force the no-LLM path

    @field_validator("horizons")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("horizons must be positive integers")
        return sorted(set(v))


class WindowData(Contract):
    """The drive's SMART snapshots inside the look-back window (oldest first)."""

    dates: list[date]
    columns: dict[str, list[float | None]]


class DriveContext(Contract):
    serial_number: str
    model: str
    capacity_bytes: int | None
    as_of_date: date
    window_df_ref: str  # stable key "<serial>@<as_of_date>" for the window cache
    window: WindowData
    features: dict[str, float | None]
    model_seen_in_training: bool = True
    base_rates: dict[int, float] = Field(default_factory=dict)  # per-model prior per horizon
    missing_attributes: list[str] = Field(default_factory=list)


class EvidenceItem(Contract):
    id: str
    kind: EvidenceKind
    label: str
    value: float | str | None = None
    detail: str = ""


class ToolEvidence(Contract):
    agent: str
    metrics: dict[str, float]
    notes: list[str] = Field(default_factory=list)
    risk: dict[int, float] = Field(default_factory=dict)  # calibrated family-model risk
    evidence: list[EvidenceItem] = Field(default_factory=list)


class AgentFinding(Contract):
    id: str = Field(default_factory=lambda: new_id("F"))
    agent: str
    serial_number: str
    window_start: date | None = None
    window_end: date | None = None
    summary: str
    risk: dict[int, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    status: AgentStatus
    error: str | None = None


class RetrievedCase(Contract):
    id: str
    serial_number: str
    model: str
    similarity: float
    outcome_failed: bool
    days_to_failure: int | None = None
    summary: str


class KnowledgeSnippet(Contract):
    id: str
    source_file: str
    text: str
    score: float | None = None


class FusedRisk(Contract):
    probabilities: dict[int, float]
    uncertainty: float = Field(ge=0.0, le=1.0)
    disagreement: float = Field(ge=0.0, le=1.0)
    method: str = "stacking"
    contributing_agents: list[str] = Field(default_factory=list)


class DriftReport(Contract):
    psi_by_family: dict[str, float]
    level: DriftLevel
    reference: str = ""


class Decision(Contract):
    action: str
    expected_costs: dict[str, float]
    cost_ratio: float
    escalated: bool
    reasons: list[str]


class Explanation(Contract):
    text: str
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_case_ids: list[str] = Field(default_factory=list)
    cited_knowledge_ids: list[str] = Field(default_factory=list)
    source: Literal["llm", "template"] = "template"


class TraceEvent(Contract):
    ts: datetime = Field(default_factory=utcnow)
    node: str
    agent: str | None = None
    status: TraceStatus
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(Contract):
    run_id: str
    request: PredictRequest
    context_summary: dict[str, Any] = Field(default_factory=dict)
    plan: list[str] = Field(default_factory=list)
    fused: FusedRisk | None
    decision: Decision | None
    explanation: Explanation | None
    findings: list[AgentFinding]
    retrieved_cases: list[RetrievedCase]
    knowledge: list[KnowledgeSnippet] = Field(default_factory=list)
    drift: DriftReport | None
    llm_available: bool
    latency_ms: float
    llm_calls: int
    errors: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)


# ------------------------------------------------------------- LLM output schemas


class PlannerOutput(Contract):
    """What the planner LLM must return."""

    agents: list[str] = Field(description="Agent names to run, highest priority first.")
    rationale: str = Field(description="One or two sentences on why.")


class AgentInterpretation(Contract):
    """What a specialist agent's LLM interpretation must return.

    The LLM never produces probabilities; it summarizes tool evidence and grades concern.
    """

    summary: str = Field(description="2-4 sentences interpreting the evidence, citing its ids.")
    concern: Literal["low", "medium", "high"] = Field(description="Qualitative concern level.")
    key_evidence_ids: list[str] = Field(default_factory=list, description="Evidence ids relied on.")


class ExplanationDraft(Contract):
    """What the explainer LLM must return."""

    text: str = Field(description="Explanation for an operator; cite ids in [brackets].")
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_case_ids: list[str] = Field(default_factory=list)
    cited_knowledge_ids: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------------- API


class LLMHealth(Contract):
    configured: bool
    reachable: bool
    base_url: str = ""
    model_id: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    guided_json: bool = False


class HealthResponse(Contract):
    status: Literal["ok", "degraded"]
    version: str
    env: str
    data_ready: bool
    models_ready: bool
    indexes_ready: bool
    llm: LLMHealth


class RuntimeSettings(Contract):
    """Settings editable from the UI; persisted to ``outputs/runtime/settings.json``."""

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "Qwen/Qwen3.5-4B"
    default_cost_ratio: float = 50
    enabled_agents: list[str] = Field(default_factory=list)


class RuntimeSettingsView(Contract):
    """Settings as returned to the client (API key masked)."""

    llm_base_url: str
    llm_api_key_set: bool
    llm_api_key_hint: str
    llm_model: str
    default_cost_ratio: float
    enabled_agents: list[str]
    available_agents: list[str]
    cost_ratios: list[float]


class RuntimeSettingsUpdate(Contract):
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    default_cost_ratio: float | None = Field(default=None, gt=0)
    enabled_agents: list[str] | None = None


class DriveSummary(Contract):
    serial_number: str
    model: str
    capacity_bytes: int | None
    last_date: date
    failed: bool
    latest_risk: float | None = None  # calibrated global-model P(fail within max horizon)
    action: str | None = None
    split: str | None = None


class DriveListResponse(Contract):
    items: list[DriveSummary]
    total: int
    page: int
    page_size: int
    models: list[str]
    risk_horizon: int


class HistogramBin(Contract):
    start: float
    end: float
    count: int


class FleetStats(Contract):
    total: int
    failed: int
    by_action: dict[str, int]
    by_model: dict[str, int]
    risk_horizon: int
    risk_histogram: list[HistogramBin]
    high_risk_count: int  # snapshot action is a replacement
    escalated_count: int  # snapshot escalations + escalated runs
    snapshot_available: bool


class SmartSeries(Contract):
    column: str
    family: str
    values: list[float | None]


class DriveHistory(Contract):
    serial_number: str
    model: str
    capacity_bytes: int | None
    dates: list[date]
    series: list[SmartSeries]
    failure_date: date | None = None


class FeedbackRequest(Contract):
    run_id: str
    serial_number: str
    verdict: FeedbackVerdict
    final_action: str
    note: str = ""


class FeedbackRecord(FeedbackRequest):
    id: str = Field(default_factory=lambda: new_id("FB"))
    created_at: datetime = Field(default_factory=utcnow)


class EscalationItem(Contract):
    run_id: str
    serial_number: str
    model: str
    as_of_date: date
    action: str
    reasons: list[str]
    probabilities: dict[int, float]
    created_at: datetime
    feedback: FeedbackRecord | None = None


class ExperimentSummary(Contract):
    id: str
    name: str
    created_at: str
    systems: list[str]
    headline: dict[str, Any] = Field(default_factory=dict)


class ExperimentDetail(Contract):
    id: str
    name: str
    created_at: str
    metrics: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    plots: list[str]
    config: dict[str, Any]
