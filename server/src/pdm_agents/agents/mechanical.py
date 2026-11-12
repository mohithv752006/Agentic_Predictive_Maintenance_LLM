"""Mechanical: seek errors, spin retries, command timeouts, head flying hours."""

from __future__ import annotations

from pdm_agents.agents.base import FamilyAgent
from pdm_agents.schemas import DriveContext


class MechanicalAgent(FamilyAgent):
    def domain_notes(self, trends: dict[str, dict], ctx: DriveContext) -> list[str]:
        notes = []
        for col, label in (("smart_10_raw", "spin retries"), ("smart_188_raw", "command timeouts")):
            t = trends.get(col)
            if t and (t.get("delta") or 0) > 0:
                notes.append(f"{label} increased by {t['delta']:.0f} in the window")
            elif t and (t.get("last") or 0) > 0:
                notes.append(f"{label} non-zero ({t['last']:.0f}) but stable")
        seek = trends.get("smart_7_normalized")
        if seek and seek.get("slope") is not None and seek["slope"] < 0:
            notes.append(f"seek error rate (normalized, higher is better) trending down "
                         f"{seek['slope']:.3f}/day")
        return notes
