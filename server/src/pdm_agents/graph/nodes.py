"""Graph nodes: load_context, planner, dispatch, agent nodes, retrieve, fuse, decide, explain.

Every node returns partial state updates and appends ``TraceEvent``s, which the
API streams to the UI timeline and which are stored with the run for audit.
"""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from langgraph.runtime import Runtime
from langgraph.types import Send

from pdm_agents.agents.base import AgentResources
from pdm_agents.agents.knowledge_retrieval import knowledge_query
from pdm_agents.data.features import agent_columns, compute_features
from pdm_agents.decision.policy import decide as policy_decide
from pdm_agents.graph.state import AgentTask, GraphDeps, GraphState, latest_findings
from pdm_agents.llm.structured import structured_call
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.bundle import CASE_AGENT
from pdm_agents.models.fusion import disagreement, uncertainty
from pdm_agents.rag.summarizer import family_summary
from pdm_agents.schemas import (
    AgentFinding,
    DriveContext,
    Explanation,
    ExplanationDraft,
    FusedRisk,
    KnowledgeSnippet,
    PlannerOutput,
    RetrievedCase,
    TraceEvent,
    WindowData,
)

log = get_logger(__name__)
MAX_ATTEMPTS = 2
CITE_RE = re.compile(r"\[((?:F|C|K)-[0-9A-Za-z_\-]+)\]")


class DriveNotFoundError(LookupError):
    pass


def _ev(node: str, status: str, message: str = "", agent: str | None = None,
        **data: Any) -> TraceEvent:
    return TraceEvent(node=node, status=status, message=message, agent=agent, data=data)


# --------------------------------------------------------------------- context


def build_context(res: AgentResources, serial: str, as_of: date | None) -> DriveContext:
    """Window + features for one drive, using the same feature code as training."""
    cfg, store = res.cfg, res.store
    info = store.drive_info(serial)
    if info is None:
        raise DriveNotFoundError(f"unknown drive {serial!r}")
    first = pd.Timestamp(info["first_date"]).date()
    last = pd.Timestamp(info["last_date"]).date()
    as_of = as_of or last
    if as_of < first or as_of > last:
        raise ValueError(f"as_of_date {as_of} outside the drive's data range {first}..{last}")
    columns: list[str] = []
    for cols in agent_columns(cfg).values():
        columns.extend(c for c in cols if c not in columns)
    window = store.drive_window(serial, as_of, cfg.data.window_days)
    available = [c for c in columns if c in window.columns]
    sample = pd.DataFrame([{"sample_id": f"{serial}@{as_of}", "serial_number": serial,
                            "model": info["model"], "as_of": as_of,
                            "capacity_bytes": info["capacity_bytes"]}])
    feats = compute_features(cfg, store, sample)
    features = {c: (None if pd.isna(v) else float(v))
                for c, v in feats.iloc[0].items() if c.startswith("f__")}
    win = WindowData(
        dates=[pd.Timestamp(d).date() for d in window["date"]],
        columns={c: [None if pd.isna(v) else float(v) for v in window[c]] for c in available},
    )
    bundle = res.bundle
    seen = bool(bundle and info["model"] in bundle.meta.get("training_models", []))
    base = {h: float(v[0]) for h, v in bundle.prior([info["model"]]).items()} if bundle else {}
    missing = [c for c in columns if c not in available or all(v is None for v in win.columns[c])]
    return DriveContext(
        serial_number=serial, model=str(info["model"]),
        capacity_bytes=None if pd.isna(info["capacity_bytes"]) else int(info["capacity_bytes"]),
        as_of_date=as_of, window_df_ref=f"{serial}@{as_of}", window=win, features=features,
        model_seen_in_training=seen, base_rates=base, missing_attributes=missing,
    )


def load_context(state: GraphState, runtime: Runtime[GraphDeps]) -> dict:
    res = runtime.context.res
    req = state["request"]
    t0 = time.perf_counter()
    try:
        ctx = build_context(res, req.serial_number, req.as_of_date)
    except Exception as exc:
        return {"context": None, "errors": [f"load_context: {exc}"],
                "trace": [_ev("load_context", "failed", str(exc))]}
    drift = None
    if res.cfg.features.use_drift and res.drift is not None:
        try:
            drift = res.drift.report(ctx.model, ctx.as_of_date)
        except Exception as exc:  # drift is advisory; never block a prediction
            log.warning("drift report failed: %s", exc)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"context": ctx, "drift": drift, "trace": [_ev(
        "load_context", "ok",
        f"{ctx.model}, {len(ctx.window.dates)} days up to {ctx.as_of_date}",
        duration_ms=ms, drift=drift.level if drift else None)]}


def route_after_context(state: GraphState) -> str:
    return "planner" if state.get("context") is not None else "end"


# --------------------------------------------------------------------- planner


def planner(state: GraphState, runtime: Runtime[GraphDeps]) -> dict:
    deps = runtime.context
    cfg = deps.res.cfg
    ctx = state["context"]
    attempts = state.get("attempts", 0) + 1
    allowed = list(deps.agents)
    numeric = [a for a in allowed if cfg.agents[a].kind == "numeric"]
    latest = latest_findings(state.get("findings", []))
    failed = [a for a, f in latest.items() if f.status == "failed" and a in allowed]
    default = allowed if attempts == 1 else failed
    previous = "none" if attempts == 1 else "; ".join(
        f"{a} failed: {latest[a].error}" for a in failed)
    plan, source, note = default, "default", ""
    if cfg.features.use_planner and deps.llm is not None:
        result = structured_call(deps.llm, "planner", {
            "serial_number": ctx.serial_number, "drive_model": ctx.model,
            "as_of_date": str(ctx.as_of_date), "available_agents": ", ".join(allowed),
            "feature_summary": family_summary(ctx.features, cfg),
            "previous_attempt": previous,
        }, PlannerOutput)
        if result.ok:
            proposed = list(dict.fromkeys(a for a in result.value.agents if a in allowed))
            if attempts == 1 and not set(proposed) & set(numeric):
                note = "LLM plan had no numeric agent; using default plan"
            elif proposed:
                plan, source, note = proposed, "llm", result.value.rationale
            else:
                note = "LLM plan named no available agent; using default plan"
        else:
            note = f"planner LLM {result.status}; using default plan"
    return {"plan": plan, "attempts": attempts, "trace": [_ev(
        "planner", "ok" if source == "llm" else "degraded" if deps.llm else "ok",
        f"attempt {attempts}: {', '.join(plan) or 'nothing to run'}",
        plan=plan, source=source, note=note)]}


# -------------------------------------------------------------------- dispatch


def dispatch(state: GraphState) -> dict:
    return {"trace": [_ev("dispatch", "started", "running", agent=a, attempt=state["attempts"])
                      for a in state.get("plan", [])]}


def route_dispatch(state: GraphState) -> list[Send] | str:
    sends = [Send(f"agent_{a}", AgentTask(agent=a, context=state["context"],
                                          attempt=state["attempts"]))
             for a in state.get("plan", [])]
    return sends or "retrieve"


def make_agent_node(name: str):
    """Node running one specialist agent (added per enabled agent by the builder)."""

    def agent_node(task: AgentTask, runtime: Runtime[GraphDeps]) -> dict:
        deps = runtime.context
        t0 = time.perf_counter()
        agent = deps.agents.get(name)
        if agent is None:
            finding = AgentFinding(agent=name, serial_number=task["context"].serial_number,
                                   summary=f"{name} is disabled for this run",
                                   status="failed", error="disabled")
        else:
            finding = agent.run(task["context"], deps.llm)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        out: dict = {"findings": [finding], "trace": [_ev(
            f"agent_{name}", finding.status, finding.summary[:300], agent=name,
            finding_id=finding.id, risk={str(h): p for h, p in finding.risk.items()},
            confidence=finding.confidence, duration_ms=ms, attempt=task["attempt"])]}
        if finding.status == "failed":
            out["errors"] = [f"{name}: {finding.error}"]
        return out

    agent_node.__name__ = f"agent_{name}"
    return agent_node


# -------------------------------------------------------------------- retrieve


def retrieve(state: GraphState, runtime: Runtime[GraphDeps]) -> dict:
    res = runtime.context.res
    cfg = res.cfg
    ctx = state["context"]
    if not cfg.features.use_rag:
        return {"retrieved_cases": [], "knowledge": [],
                "trace": [_ev("retrieve", "skipped", "RAG disabled")]}
    cases: list[RetrievedCase] = []
    knowledge: list[KnowledgeSnippet] = []
    notes = []
    try:
        if cfg.features.use_case_retrieval and "case_retrieval" in res.tools:
            raw = res.tool("case_retrieval").invoke({
                "serial_number": ctx.serial_number, "model": ctx.model,
                "features": ctx.features, "k": cfg.rag.case_top_k})
            cases = [RetrievedCase(**c) for c in raw]
        if cfg.features.use_knowledge_retrieval and "knowledge_retrieval" in res.tools:
            query = _knowledge_query(state, res)
            raw = res.tool("knowledge_retrieval").invoke(
                {"query": query, "k": cfg.rag.knowledge_top_k})
            knowledge = [KnowledgeSnippet(**s) for s in raw]
            # keep snippets the knowledge agent already cited
            seen = {k.id for k in knowledge}
            for f in latest_findings(state.get("findings", [])).values():
                for e in f.evidence:
                    if e.kind == "knowledge" and e.id not in seen and res.knowledge:
                        snip = res.knowledge.get(e.id)
                        if snip:
                            knowledge.append(snip)
                            seen.add(e.id)
    except Exception as exc:
        notes.append(f"retrieval error: {exc}")
        log.warning("retrieval failed: %s", exc)
    status = "failed" if notes else "ok"
    return {"retrieved_cases": cases, "knowledge": knowledge, "errors": notes,
            "trace": [_ev("retrieve", status,
                          f"{len(cases)} similar drives, {len(knowledge)} knowledge snippets",
                          case_ids=[c.id for c in cases], knowledge_ids=[k.id for k in knowledge])]}


def _knowledge_query(state: GraphState, res: AgentResources) -> str:
    """Query focused on the agents reporting the highest risk."""
    cfg = res.cfg
    latest = latest_findings(state.get("findings", []))
    h = cfg.max_horizon
    numeric = sorted((f for a, f in latest.items() if cfg.agents.get(a) and
                      cfg.agents[a].kind == "numeric" and f.risk),
                     key=lambda f: f.risk.get(h, 0.0), reverse=True)[:2]
    if not numeric:
        return knowledge_query(state["context"], cfg)
    parts = []
    for f in numeric:
        trends = [e.label for e in f.evidence if e.kind == "trend"][:3]
        parts.append(f"{cfg.agents[f.agent].description} ({', '.join(trends)})")
    return "Hard drive failure risk: " + "; ".join(parts)


# ------------------------------------------------------------------------ fuse


def fuse(state: GraphState, runtime: Runtime[GraphDeps]) -> dict:
    deps = runtime.context
    cfg, bundle = deps.res.cfg, deps.res.bundle
    ctx = state["context"]
    latest = latest_findings(state.get("findings", []))

    def ok(a: str, kind: str) -> bool:
        f = latest.get(a)
        return bool(f and cfg.agents.get(a) and cfg.agents[a].kind == kind
                    and f.status != "failed" and f.risk)

    run_numeric = [a for a in deps.agents if cfg.agents[a].kind == "numeric"]
    usable = {a: latest[a] for a in run_numeric if ok(a, "numeric")}
    single = [latest[a] for a in deps.agents if ok(a, "global")]
    if bundle is None:
        return {"fused": None, "errors": ["fuse: no trained models"],
                "trace": [_ev("fuse", "failed", "no trained model bundle")]}
    h = cfg.max_horizon
    window_cov = min(1.0, len(ctx.window.dates) / cfg.data.window_days)
    case = latest.get(CASE_AGENT)
    use_case = bool(cfg.features.use_rag and cfg.features.use_case_retrieval and case
                    and case.status != "failed" and case.risk and bundle.has_case_fusion)
    if not usable and single:
        # single-agent baseline: one agent over all attributes, no fusion
        probs = {hh: np.array([p]) for hh, p in single[0].risk.items()}
        contributing, method, agent_cov = [single[0].agent], "single_agent", 1.0
    else:
        risks = {a: {hh: np.array([p]) for hh, p in f.risk.items()} for a, f in usable.items()}
        if use_case:
            risks[CASE_AGENT] = {hh: np.array([p]) for hh, p in case.risk.items()}
        probs = bundle.fuse(risks, [ctx.model], method=cfg.models.fusion.method,
                            calibrated=cfg.features.use_calibration)
        contributing = list(risks)
        method = cfg.models.fusion.method if cfg.features.use_calibration else "uncalibrated"
        agent_cov = len(usable) / len(run_numeric) if run_numeric else 0.0
    p_max = float(probs[h][0])
    agent_risk = {a: f.risk.get(h) for a, f in usable.items()}
    if use_case and usable:
        agent_risk[CASE_AGENT] = case.risk.get(h)
    weights = bundle.agent_weights(h, with_case=use_case) if usable else {}
    fused = FusedRisk(
        probabilities={hh: round(float(v[0]), 6) for hh, v in probs.items()},
        uncertainty=round(uncertainty(p_max, window_cov * agent_cov), 4),
        disagreement=round(disagreement(agent_risk, weights), 4),
        method=method,
        contributing_agents=contributing,
    )
    return {"fused": fused, "trace": [_ev(
        "fuse", "ok", ", ".join(f"{hh}d {p:.2%}" for hh, p in fused.probabilities.items()),
        probabilities={str(k): v for k, v in fused.probabilities.items()},
        uncertainty=fused.uncertainty, disagreement=fused.disagreement,
        agents=list(usable))]}


def route_after_fuse(state: GraphState) -> str:
    """Replan once when two or more agents failed; otherwise decide."""
    latest = latest_findings(state.get("findings", []))
    failed = [a for a, f in latest.items() if f.status == "failed"]
    if len(failed) >= 2 and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "planner"
    return "decide"


# ---------------------------------------------------------------------- decide


def decide(state: GraphState, runtime: Runtime[GraphDeps]) -> dict:
    deps = runtime.context
    fused = state.get("fused")
    if fused is None:
        return {"decision": None, "trace": [_ev("decide", "skipped", "no fused risk")]}
    latest = latest_findings(state.get("findings", []))
    failed = [a for a, f in latest.items() if f.status == "failed"]
    bundle = deps.res.bundle
    weights = bundle.agent_weights(deps.res.cfg.max_horizon) if bundle else None
    decision = policy_decide(deps.res.cfg, fused, deps.cost_ratio, state.get("drift"), failed,
                             drift_weights=weights)
    return {"decision": decision, "trace": [_ev(
        "decide", "ok", f"{decision.action}" + (" (escalated)" if decision.escalated else ""),
        action=decision.action, expected_costs=decision.expected_costs,
        escalated=decision.escalated)]}


# --------------------------------------------------------------------- explain


def _blocks(state: GraphState) -> dict[str, str]:
    d, fused = state.get("decision"), state.get("fused")
    latest = latest_findings(state.get("findings", []))
    decision_block = "none"
    if d:
        # Spell out the two-step logic so the explainer cannot confuse them: the cheapest
        # action is chosen on cost alone; escalation (never cheapest) comes from the rules.
        own = {a: c for a, c in d.expected_costs.items() if a != "escalate_to_human"}
        cheapest = min(own, key=own.get) if own else d.action
        costs = ", ".join(f"{a} {c:.3f}" for a, c in sorted(own.items(), key=lambda kv: kv[1]))
        lines = [f"Final recommendation: {d.action}.",
                 f"Lowest expected-cost action: {cheapest} ({own.get(cheapest, 0):.3f} units; "
                 f"cost ratio {d.cost_ratio:g}). Expected costs: {costs}."]
        if d.escalated:
            rules = [r for r in d.reasons
                     if not r.startswith(("lowest expected cost", "suggested action"))]
            lines.append("The case is escalated to a human reviewer because of these rules, not "
                         f"because escalation is cheaper: {'; '.join(rules)}. The reviewer is "
                         f"advised to {cheapest}.")
        decision_block = " ".join(lines)
    fused_block = "none"
    if fused:
        fused_block = (", ".join(f"within {h} days {p:.2%}" for h, p in
                                 sorted(fused.probabilities.items()))
                       + f"; uncertainty {fused.uncertainty:.2f}; agent disagreement "
                         f"{fused.disagreement:.2f}")
    findings_block = "\n".join(
        f"[{f.id}] {f.agent} ({f.status}"
        + (", risk " + ", ".join(f"{h}d {p:.2%}" for h, p in sorted(f.risk.items()))
           if f.risk else "") + f"): {f.summary}"
        for f in latest.values()) or "none"
    cases_block = "\n".join(
        f"[{c.id}] similarity {c.similarity:.2f}: {c.summary}"
        for c in state.get("retrieved_cases", [])) or "none"
    knowledge_block = "\n".join(
        f"[{k.id}] {k.text[:500]}" for k in state.get("knowledge", [])) or "none"
    return {"decision_block": decision_block, "fused_block": fused_block,
            "findings_block": findings_block, "cases_block": cases_block,
            "knowledge_block": knowledge_block}


def template_explanation(state: GraphState) -> Explanation:
    """Deterministic explanation citing real ids (used when the LLM is unavailable)."""
    d, fused = state.get("decision"), state.get("fused")
    latest = latest_findings(state.get("findings", []))
    ranked = sorted((f for f in latest.values() if f.status != "failed" and f.risk),
                    key=lambda f: max(f.risk.values()), reverse=True)
    cases = state.get("retrieved_cases", [])
    knowledge = state.get("knowledge", [])
    parts: list[str] = []
    if d and fused:
        h_lo, h_hi = min(fused.probabilities), max(fused.probabilities)
        parts.append(
            f"Recommended action: {d.action.replace('_', ' ')}. The calibrated probability of "
            f"failure is {fused.probabilities[h_lo]:.2%} within {h_lo} days and "
            f"{fused.probabilities[h_hi]:.2%} within {h_hi} days.")
        if d.escalated:
            parts.append("The case was escalated: " + "; ".join(d.reasons[1:]) + ".")
        else:
            parts.append(d.reasons[0][0].upper() + d.reasons[0][1:] + ".")
    for f in ranked[:2]:
        parts.append(f"{f.agent.replace('_', ' ').capitalize()}: {f.summary} [{f.id}]")
    if cases:
        n_failed = sum(c.outcome_failed for c in cases)
        cited = " ".join(f"[{c.id}]" for c in cases[:3])
        parts.append(f"{n_failed} of the {len(cases)} most similar training drives failed "
                     f"within the horizon {cited}.")
    if knowledge:
        k = knowledge[0]
        parts.append(f"Reference: {k.text.split(':', 1)[0]} [{k.id}].")
    return Explanation(
        text=" ".join(parts) or "No explanation available.",
        cited_finding_ids=[f.id for f in ranked[:2]],
        cited_case_ids=[c.id for c in cases[:3]],
        cited_knowledge_ids=[knowledge[0].id] if knowledge else [],
        source="template",
    )


def explain(state: GraphState, runtime: Runtime[GraphDeps]) -> dict:
    deps = runtime.context
    ctx = state["context"]
    if state.get("decision") is None:
        exp = template_explanation(state)
        return {"explanation": exp, "trace": [_ev("explain", "degraded", "no decision")]}
    valid = {
        "F": {f.id for f in latest_findings(state.get("findings", [])).values()},
        "C": {c.id for c in state.get("retrieved_cases", [])},
        "K": {k.id for k in state.get("knowledge", [])},
    }
    result = structured_call(deps.llm, "explainer", {
        "serial_number": ctx.serial_number, "drive_model": ctx.model,
        "as_of_date": str(ctx.as_of_date), **_blocks(state)}, ExplanationDraft)
    exp = None
    if result.ok:
        exp = _validated_explanation(result.value, valid)
    status, note = ("ok", "LLM explanation") if exp else ("degraded", "template explanation")
    if exp is None:
        exp = template_explanation(state)
        if result.ok:
            note = "LLM explanation cited no valid ids; template used"
    return {"explanation": exp, "trace": [_ev("explain", status, note, source=exp.source)]}


def _validated_explanation(draft: ExplanationDraft,
                           valid: dict[str, set[str]]) -> Explanation | None:
    """Keep only citations of real ids; strip invented bracket ids from the text."""
    text = draft.text.strip()
    cited = {k: [] for k in valid}
    for ref in [*draft.cited_finding_ids, *draft.cited_case_ids, *draft.cited_knowledge_ids,
                *CITE_RE.findall(text)]:
        kind = ref[:1]
        if kind in valid and ref in valid[kind] and ref not in cited[kind]:
            cited[kind].append(ref)
    text = CITE_RE.sub(lambda m: m.group(0) if m.group(1)[:1] in valid
                       and m.group(1) in valid[m.group(1)[:1]] else "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not any(cited.values()) and any(valid.values()):
        return None
    return Explanation(text=text, cited_finding_ids=cited["F"], cited_case_ids=cited["C"],
                       cited_knowledge_ids=cited["K"], source="llm")
