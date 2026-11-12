"""Thermal: sustained heat and temperature spikes."""

from __future__ import annotations

from pdm_agents.agents.base import FamilyAgent
from pdm_agents.schemas import DriveContext


class ThermalAgent(FamilyAgent):
    def domain_notes(self, trends: dict[str, dict], ctx: DriveContext) -> list[str]:
        notes = []
        t = trends.get("smart_194_raw") or trends.get("smart_190_raw")
        if not t or t.get("n", 0) == 0:
            return notes
        if t["max"] < 200:  # plausible Celsius reading (raw values can be byte-encoded)
            notes.append(f"temperature mean {t['mean']:.1f} C, max {t['max']:.0f} C")
        else:
            notes.append("temperature raw value looks byte-encoded; trends only")
        if t.get("spikes"):
            notes.append(f"{t['spikes']} temperature spike(s) above the window's robust range")
        cp = t.get("change_point")
        if cp and abs(cp["shift_in_std"]) >= 1.5:
            notes.append(f"temperature level shifted {cp['shift']:+.1f} at day {cp['day_index']}")
        return notes
