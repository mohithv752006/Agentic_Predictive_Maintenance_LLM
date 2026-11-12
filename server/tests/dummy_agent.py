"""Extension proof: a new agent in one file (+ one agents.yaml entry)."""

from __future__ import annotations

from pdm_agents.agents.base import SpecialistAgent
from pdm_agents.schemas import DriveContext, EvidenceItem, ToolEvidence


class DummyAgent(SpecialistAgent):
    def run_tools(self, ctx: DriveContext) -> ToolEvidence:
        n = len(ctx.window.dates)
        return ToolEvidence(agent=self.name, metrics={"days": float(n)},
                            evidence=[EvidenceItem(id=f"E-{self.name}-0", kind="note",
                                                   label="days observed", value=n)])
