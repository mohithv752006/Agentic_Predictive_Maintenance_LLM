"""Specialist agent contract and the shared numeric-family implementation.

Numbers come from tools and trained models (``run_tools``, no LLM). The LLM only
interprets the evidence (``interpret``). ``run`` never raises: any exception
becomes a finding with ``status="failed"``; an unavailable or invalid LLM answer
becomes a deterministic template summary with ``status="degraded"``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.tools import BaseTool

from pdm_agents.config import AgentSpec, AppConfig
from pdm_agents.data.features import family_features
from pdm_agents.data.store import Store
from pdm_agents.decision.drift import DriftService
from pdm_agents.llm.client import LLMClient
from pdm_agents.llm.structured import prompt_exists, structured_call
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.bundle import ModelBundle
from pdm_agents.rag.case_memory import CaseMemory
from pdm_agents.rag.knowledge_index import KnowledgeIndex
from pdm_agents.rag.summarizer import column_label
from pdm_agents.schemas import (
    AgentFinding,
    AgentInterpretation,
    DriveContext,
    EvidenceItem,
    ToolEvidence,
)

log = get_logger(__name__)
Concern = Literal["low", "medium", "high"]


@dataclass
class AgentResources:
    """Everything agents and graph nodes need at run time (loaded once per process)."""

    cfg: AppConfig
    store: Store
    bundle: ModelBundle | None
    cases: CaseMemory | None
    knowledge: KnowledgeIndex | None
    tools: dict[str, BaseTool]
    drift: DriftService | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def tool(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise RuntimeError(f"tool {name!r} unavailable (artifacts missing?)")
        return self.tools[name]


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if abs(v) >= 1e6:
        return f"{v:.3g}"
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.3f}"


class SpecialistAgent(ABC):
    """Base class. Subclass + one ``agents.yaml`` entry adds an agent."""

    def __init__(self, name: str, spec: AgentSpec, resources: AgentResources) -> None:
        self.name = name
        self.spec = spec
        self.attributes = list(spec.attributes)
        self.description = spec.description
        self.res = resources

    # --------------------------------------------------------------- contract

    @abstractmethod
    def run_tools(self, ctx: DriveContext) -> ToolEvidence:
        """Deterministic evidence from tools and models (no LLM)."""

    def interpret(self, evidence: ToolEvidence, llm: LLMClient | None,
                  ctx: DriveContext | None = None) -> AgentFinding:
        """LLM interpretation of the evidence; template fallback marks the finding degraded."""
        hint = self.risk_hint(evidence, ctx)
        prompt = f"agent_{self.name}" if prompt_exists(f"agent_{self.name}") else "specialist"
        result = structured_call(llm, prompt, {
            "agent_name": self.name,
            "agent_description": self.description,
            "serial_number": ctx.serial_number if ctx else "",
            "drive_model": ctx.model if ctx else "",
            "window_start": str(ctx.window.dates[0]) if ctx and ctx.window.dates else "",
            "window_end": str(ctx.as_of_date) if ctx else "",
            "risk_hint": hint,
            "evidence_block": self.evidence_block(evidence),
        }, AgentInterpretation)
        valid_ids = {e.id for e in evidence.evidence}
        if result.ok:
            summary = result.value.summary.strip()
            status = "ok"
            cited = set(result.value.key_evidence_ids)
            if cited and not cited <= valid_ids:
                summary += " (note: some cited evidence ids were not recognised)"
        else:
            summary = self.template_summary(evidence, hint)
            status = "degraded"
        return AgentFinding(
            agent=self.name,
            serial_number=ctx.serial_number if ctx else "",
            window_start=ctx.window.dates[0] if ctx and ctx.window.dates else None,
            window_end=ctx.as_of_date if ctx else None,
            summary=summary,
            risk=evidence.risk,
            confidence=self.confidence(evidence, ctx),
            evidence=evidence.evidence,
            status=status,
            error=None if result.ok else f"llm {result.status}: {result.error}",
        )

    def run(self, ctx: DriveContext, llm: LLMClient | None) -> AgentFinding:
        """Tools then interpretation. Never raises."""
        try:
            evidence = self.run_tools(ctx)
            return self.interpret(evidence, llm, ctx)
        except Exception as exc:  # one agent must never crash the graph
            log.warning("agent %s failed: %s", self.name, exc, exc_info=True)
            return AgentFinding(
                agent=self.name, serial_number=ctx.serial_number,
                window_end=ctx.as_of_date, summary=f"{self.name} could not run: {exc}",
                status="failed", error=f"{type(exc).__name__}: {exc}", confidence=0.0,
            )

    # ---------------------------------------------------------------- helpers

    def risk_hint(self, evidence: ToolEvidence, ctx: DriveContext | None) -> Concern:
        """Concern implied by the calibrated risk relative to the fleet base rate."""
        if not evidence.risk:
            return "low"
        h = max(evidence.risk)
        p = evidence.risk[h]
        base = (ctx.base_rates.get(h) if ctx else None) or 0.0
        ratio = p / base if base > 0 else float("inf") if p > 0 else 0.0
        if p >= 0.2 or ratio >= 10:
            return "high"
        if p >= 0.05 or ratio >= 3:
            return "medium"
        return "low"

    def evidence_block(self, evidence: ToolEvidence) -> str:
        lines = [f"[{e.id}] {e.label}: {e.value if e.value is not None else ''} {e.detail}".strip()
                 for e in evidence.evidence]
        lines += [f"note: {n}" for n in evidence.notes]
        return "\n".join(lines) if lines else "(no evidence)"

    def template_summary(self, evidence: ToolEvidence, hint: Concern) -> str:
        """Deterministic summary used when the LLM is unavailable or invalid."""
        parts = []
        if evidence.risk:
            risk = ", ".join(f"{h}d {p:.1%}" for h, p in sorted(evidence.risk.items()))
            parts.append(f"Calibrated {self.name} risk: {risk} ({hint} concern)")
        notable = [e for e in evidence.evidence if e.kind in ("trend", "case", "knowledge")][:3]
        if notable:
            parts.append("; ".join(f"{e.label} [{e.id}]" for e in notable))
        parts += evidence.notes[:2]
        return ". ".join(parts) + "." if parts else f"No usable {self.name} evidence."

    def confidence(self, evidence: ToolEvidence, ctx: DriveContext | None) -> float:
        """Data coverage: observed fraction of the window x fraction of attributes present."""
        if ctx is None:
            return 0.5
        w = self.res.cfg.data.window_days
        coverage = min(1.0, len(ctx.window.dates) / w)
        if self.attributes:
            present = [a for a in self.attributes if a not in ctx.missing_attributes
                       and a in ctx.window.columns]
            coverage *= len(present) / len(self.attributes)
        return round(coverage, 3)


class FamilyAgent(SpecialistAgent):
    """Numeric agent owning a SMART attribute family and its calibrated family model."""

    model_label = "calibrated family-model risk"

    def domain_notes(self, trends: dict[str, dict], ctx: DriveContext) -> list[str]:
        """Agent-specific observations (override in subclasses)."""
        return []

    def model_risk(self, ctx: DriveContext) -> dict | None:
        """``{"risk": {h: p}, "base_rate": {h: p}}`` from this agent's model, or None."""
        bundle = self.res.bundle
        if "family_risk" not in self.res.tools or bundle is None \
                or self.name not in bundle.family_models:
            return None
        feats = {f: ctx.features.get(f) for f in
                 family_features(self.res.cfg, self.name, list(ctx.features))}
        return self.res.tool("family_risk").invoke(
            {"agent": self.name, "model": ctx.model, "features": feats})

    def run_tools(self, ctx: DriveContext) -> ToolEvidence:
        present = [a for a in self.attributes
                   if a in ctx.window.columns and any(v is not None for v in ctx.window.columns[a])]
        if not present:
            raise RuntimeError(f"none of {self.attributes} reported for model {ctx.model}")
        trend_tool = self.res.tool("trend_analysis")
        trends: dict[str, dict] = {}
        metrics: dict[str, float] = {}
        items: list[EvidenceItem] = []
        for col in present:
            t = trend_tool.invoke({"column": col, "values": ctx.window.columns[col]})
            trends[col] = t
            for key in ("last", "delta", "slope", "acceleration", "max", "spikes", "increases"):
                if t.get(key) is not None:
                    metrics[f"{col}.{key}"] = float(t[key])
            detail = (f"{_fmt(t.get('first'))} -> {_fmt(t.get('last'))} over {t['n']} days, "
                      f"slope {_fmt(t.get('slope'))}/day, {t.get('increases', 0)} increases")
            cp = t.get("change_point")
            if cp and abs(cp["shift_in_std"]) >= 1.5:
                detail += f", level shift {cp['shift']:+.3g} at day {cp['day_index']}"
            items.append(EvidenceItem(id=f"E-{self.name}-{len(items)}", kind="trend",
                                      label=column_label(col), value=t.get("last"),
                                      detail=detail))
        risk: dict[int, float] = {}
        notes: list[str] = []
        out = self.model_risk(ctx)
        if out is not None:
            risk = {int(h): float(p) for h, p in out["risk"].items()}
            base = {int(h): float(p) for h, p in out["base_rate"].items()}
            text = ", ".join(f"{h}d {p:.2%} (base {base[h]:.2%})" for h, p in sorted(risk.items()))
            items.append(EvidenceItem(id=f"E-{self.name}-{len(items)}", kind="model",
                                      label=self.model_label, value=None, detail=text))
            for h, p in risk.items():
                metrics[f"risk_{h}d"] = p
        else:
            notes.append("no trained model available; trend evidence only")
        missing = [a for a in self.attributes if a not in present]
        if missing:
            notes.append(f"not reported by this drive: {', '.join(missing)}")
        notes += self.domain_notes(trends, ctx)
        return ToolEvidence(agent=self.name, metrics=metrics, notes=notes, risk=risk,
                            evidence=items)
