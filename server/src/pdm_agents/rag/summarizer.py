"""Turn a drive window's features into compact, human-readable text.

Used for case-memory summaries (similar past drives), the planner's feature
summary and template explanations. Only computed features are used, never raw
rows, and numbers are reported as computed (no rounding beyond display).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from pdm_agents.config import AppConfig
from pdm_agents.data.features import agent_columns, feature_name

ATTRIBUTE_NAMES = {
    1: "read error rate", 3: "spin-up time", 4: "start/stop count",
    5: "reallocated sectors", 7: "seek error rate", 9: "power-on hours",
    10: "spin retries", 12: "power cycles", 184: "end-to-end errors",
    187: "reported uncorrectable errors", 188: "command timeouts",
    190: "airflow temperature", 192: "power-off retracts", 193: "load cycles",
    194: "temperature", 197: "pending sectors", 198: "offline uncorrectable sectors",
    199: "UDMA CRC errors", 240: "head flying hours", 241: "LBAs written", 242: "LBAs read",
}
COL_RE = re.compile(r"smart_(\d+)_(raw|normalized)")
# counters where any growth is a warning sign
ERROR_COUNTERS = {5, 10, 184, 187, 188, 197, 198, 199}


def column_label(column: str) -> str:
    m = COL_RE.fullmatch(column)
    if not m:
        return column
    attr, kind = int(m.group(1)), m.group(2)
    name = ATTRIBUTE_NAMES.get(attr, f"attribute {attr}")
    return f"{name} (SMART {attr}{', normalized' if kind == 'normalized' else ''})"


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _fmt(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v:.3g}"
    if abs(v - round(v)) < 1e-9:
        return f"{round(v):,}"
    return f"{v:.2f}"


def column_phrase(row: Mapping[str, Any], column: str) -> str | None:
    """'reallocated sectors (SMART 5): 0 -> 48 over the window, slope +1.60/day'."""
    last = _num(row.get(feature_name(column, "last")))
    if last is None:
        return None
    delta = _num(row.get(feature_name(column, "delta"))) or 0.0
    slope = _num(row.get(feature_name(column, "slope")))
    mx = _num(row.get(feature_name(column, "max")))
    first = last - delta
    text = f"{column_label(column)}: {_fmt(first)} -> {_fmt(last)}"
    if slope is not None and abs(slope) > 1e-9:
        text += f", slope {slope:+.2f}/day"
    if mx is not None and mx > last:
        text += f", max {_fmt(mx)}"
    return text


def _interest(row: Mapping[str, Any], column: str) -> float:
    m = COL_RE.fullmatch(column)
    attr = int(m.group(1)) if m else 0
    last = _num(row.get(feature_name(column, "last"))) or 0.0
    delta = abs(_num(row.get(feature_name(column, "delta"))) or 0.0)
    score = 0.0
    if attr in ERROR_COUNTERS:
        score += 10.0 * (delta > 0) + 3.0 * (last > 0)
    return score + math.log1p(delta)


def trajectory_summary(row: Mapping[str, Any], cfg: AppConfig, max_items: int = 5) -> str:
    """One-line summary of the most informative attributes in a window."""
    cols: list[str] = []
    for attrs in agent_columns(cfg).values():
        cols.extend(c for c in attrs if c not in cols)
    ranked = sorted(cols, key=lambda c: _interest(row, c), reverse=True)
    phrases = [p for c in ranked if (p := column_phrase(row, c))][:max_items]
    head = []
    if (cap := _num(row.get("f__ctx__capacity_tb"))) is not None:
        head.append(f"{cap:.1f} TB")
    if (n := _num(row.get("f__ctx__n_obs"))) is not None:
        head.append(f"{int(n)} days observed")
    return "; ".join([", ".join(head), *phrases]) if head else "; ".join(phrases)


def family_summary(row: Mapping[str, Any], cfg: AppConfig) -> str:
    """Multi-line per-family summary (planner input)."""
    lines = []
    for agent, cols in agent_columns(cfg).items():
        phrases = [p for c in cols if (p := column_phrase(row, c))]
        lines.append(f"- {agent}: " + ("; ".join(phrases) if phrases else "no data"))
    return "\n".join(lines)
