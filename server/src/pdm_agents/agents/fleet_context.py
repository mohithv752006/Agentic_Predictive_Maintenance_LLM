"""Fleet context: drive model, capacity, per-model base rates and data coverage.

Its risk is the empirical per-model base rate (the prior that fusion starts from);
it is reported for context and not fused a second time.
"""

from __future__ import annotations

from pdm_agents.agents.base import Concern, SpecialistAgent
from pdm_agents.schemas import DriveContext, EvidenceItem, ToolEvidence


class FleetContextAgent(SpecialistAgent):
    def run_tools(self, ctx: DriveContext) -> ToolEvidence:
        items, notes = [], []
        base = dict(ctx.base_rates)
        if base:
            text = ", ".join(f"{h}d {p:.3%}" for h, p in sorted(base.items()))
            items.append(EvidenceItem(id=f"E-{self.name}-0", kind="context",
                                      label="fleet base rate for this model", detail=text))
        seen = "yes" if ctx.model_seen_in_training else "no (pooled statistics are used)"
        items.append(EvidenceItem(id=f"E-{self.name}-{len(items)}", kind="context",
                                  label="model seen in training", value=seen))
        if ctx.capacity_bytes:
            items.append(EvidenceItem(id=f"E-{self.name}-{len(items)}", kind="context",
                                      label="capacity",
                                      value=f"{ctx.capacity_bytes / 1e12:.1f} TB"))
        w = self.res.cfg.data.window_days
        items.append(EvidenceItem(id=f"E-{self.name}-{len(items)}", kind="context",
                                  label="window coverage",
                                  value=f"{len(ctx.window.dates)}/{w} days"))
        if self.res.drift is not None and self.res.cfg.features.use_drift:
            drift = self.res.drift.report(ctx.model, ctx.as_of_date)
            items.append(EvidenceItem(id=f"E-{self.name}-{len(items)}", kind="context",
                                      label="feature drift vs validation period",
                                      value=drift.level,
                                      detail=", ".join(f"{k} PSI {v:.2f}"
                                                       for k, v in drift.psi_by_family.items())))
        if not ctx.model_seen_in_training:
            notes.append("drive model was never seen in training: predictions extrapolate")
        if ctx.missing_attributes:
            notes.append(f"attributes not reported: {', '.join(ctx.missing_attributes)}")
        metrics = {f"base_rate_{h}d": p for h, p in base.items()}
        metrics["window_days_observed"] = float(len(ctx.window.dates))
        return ToolEvidence(agent=self.name, metrics=metrics, notes=notes, risk=base,
                            evidence=items)

    def risk_hint(self, evidence: ToolEvidence, ctx: DriveContext | None) -> Concern:
        return "medium" if ctx is not None and not ctx.model_seen_in_training else "low"

    def confidence(self, evidence: ToolEvidence, ctx: DriveContext | None) -> float:
        return 1.0 if ctx is None or ctx.model_seen_in_training else 0.5
