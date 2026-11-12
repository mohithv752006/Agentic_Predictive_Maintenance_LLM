"""Single agent seeing every SMART attribute (baseline for hypothesis H1).

Same trend tools as the specialists, but its risk comes from the all-features
model and nothing is fused. Enabled only by the ``single_llm_agent`` experiment
(``kind: global`` in its agents override).
"""

from __future__ import annotations

from pdm_agents.agents.base import FamilyAgent
from pdm_agents.schemas import DriveContext


class AllAttributesAgent(FamilyAgent):
    model_label = "calibrated all-features model risk"

    def model_risk(self, ctx: DriveContext) -> dict | None:
        if "global_risk" not in self.res.tools:
            return None
        return self.res.tool("global_risk").invoke(
            {"model": ctx.model, "features": dict(ctx.features)})
