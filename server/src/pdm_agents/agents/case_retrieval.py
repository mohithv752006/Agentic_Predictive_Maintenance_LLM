"""Case retrieval: the most similar training drives (same model) and their outcomes.

Besides the neighbours themselves, the agent reports a calibrated case risk: the
similarity-weighted share of neighbours that failed within each horizon, calibrated
on validation data (``fit_case_fusion``). Fusion uses it as one more input.
"""

from __future__ import annotations

from pdm_agents.agents.base import Concern, SpecialistAgent
from pdm_agents.schemas import DriveContext, EvidenceItem, ToolEvidence


class CaseRetrievalAgent(SpecialistAgent):
    def run_tools(self, ctx: DriveContext) -> ToolEvidence:
        args = {"serial_number": ctx.serial_number, "model": ctx.model,
                "features": ctx.features, "k": self.res.cfg.rag.case_top_k}
        cases = self.res.tool("case_retrieval").invoke(args)
        items = [EvidenceItem(id=c["id"], kind="case",
                              label=f"similar drive {c['serial_number']} (similarity "
                                    f"{c['similarity']:.2f})",
                              value="failed" if c["outcome_failed"] else "survived",
                              detail=c["summary"]) for c in cases]
        sims = [max(c["similarity"], 0.0) for c in cases]
        failed = [c for c in cases if c["outcome_failed"]]
        weighted = (sum(s for s, c in zip(sims, cases, strict=True) if c["outcome_failed"])
                    / sum(sims)) if sum(sims) > 0 else 0.0
        metrics = {"n_cases": float(len(cases)), "n_failed": float(len(failed)),
                   "mean_similarity": sum(sims) / len(sims) if sims else 0.0,
                   "similarity_weighted_failed_share": weighted}
        notes = [f"{len(failed)} of {len(cases)} most similar training drives failed within "
                 f"the horizon"]
        risk: dict[int, float] = {}
        if "case_risk" in self.res.tools:
            risk = {int(h): float(p) for h, p in self.res.tool("case_risk").invoke(args).items()}
            text = ", ".join(f"{h}d {p:.2%}" for h, p in sorted(risk.items()))
            items.append(EvidenceItem(id=f"E-{self.name}-0", kind="model",
                                      label="calibrated case-memory risk", detail=text))
        else:
            notes.append("no calibrated case risk available (run scripts.index)")
        if not ctx.model_seen_in_training:
            notes.append("no same-model cases exist; neighbours come from all training models")
        return ToolEvidence(agent=self.name, metrics=metrics, notes=notes, risk=risk,
                            evidence=items)

    def risk_hint(self, evidence: ToolEvidence, ctx: DriveContext | None) -> Concern:
        if evidence.risk:
            return super().risk_hint(evidence, ctx)
        share = evidence.metrics.get("similarity_weighted_failed_share", 0.0)
        return "high" if share >= 0.6 else "medium" if share >= 0.2 else "low"

    def confidence(self, evidence: ToolEvidence, ctx: DriveContext | None) -> float:
        return round(max(0.0, min(1.0, evidence.metrics.get("mean_similarity", 0.0))), 3)
