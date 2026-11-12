"""Usage and wear: age, power and load cycles, workload volume."""

from __future__ import annotations

from pdm_agents.agents.base import FamilyAgent
from pdm_agents.schemas import DriveContext

HOURS_PER_YEAR = 24 * 365


class UsageWearAgent(FamilyAgent):
    def domain_notes(self, trends: dict[str, dict], ctx: DriveContext) -> list[str]:
        notes = []
        hours = trends.get("smart_9_raw")
        if hours and hours.get("last") is not None:
            notes.append(f"power-on time {hours['last']:,.0f} (about "
                         f"{hours['last'] / HOURS_PER_YEAR:.1f} years if counted in hours)")
        cycles = trends.get("smart_12_raw")
        if cycles and (cycles.get("delta") or 0) > 0:
            notes.append(f"{cycles['delta']:.0f} power cycle(s) in the window")
        load = trends.get("smart_193_raw")
        if load and load.get("slope") is not None:
            notes.append(f"load cycles growing {load['slope']:,.1f}/day")
        return notes
