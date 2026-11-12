"""Media health: sector reallocation, pending and uncorrectable sector growth."""

from __future__ import annotations

from pdm_agents.agents.base import FamilyAgent
from pdm_agents.rag.summarizer import column_label
from pdm_agents.schemas import DriveContext


class MediaHealthAgent(FamilyAgent):
    def domain_notes(self, trends: dict[str, dict], ctx: DriveContext) -> list[str]:
        notes = []
        grew = [c for c, t in trends.items() if (t.get("delta") or 0) > 0]
        if grew:
            notes.append("error counters grew inside the window: "
                         + ", ".join(column_label(c) for c in grew))
        nonzero = [c for c, t in trends.items() if (t.get("last") or 0) > 0 and c not in grew]
        if nonzero:
            notes.append("non-zero but stable: " + ", ".join(column_label(c) for c in nonzero))
        pending = trends.get("smart_197_raw")
        if pending and (pending.get("delta") or 0) < 0:
            notes.append("pending sectors decreased (SMART 197 is not cumulative)")
        if not grew and not nonzero:
            notes.append("all media error counters are zero")
        return notes
