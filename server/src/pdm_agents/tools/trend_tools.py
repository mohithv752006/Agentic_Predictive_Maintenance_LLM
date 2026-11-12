"""Deterministic trend tools over one SMART series (oldest first, None = missing day)."""

from __future__ import annotations

import math

import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field


def _clean(values: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([i for i, v in enumerate(values) if v is not None and not _nan(v)], dtype=float)
    y = np.array([float(v) for v in values if v is not None and not _nan(v)], dtype=float)
    return t, y


def _nan(v: float) -> bool:
    return isinstance(v, float) and math.isnan(v)


def slope(values: list[float | None]) -> float | None:
    """Least-squares slope per day (None with fewer than 2 points)."""
    t, y = _clean(values)
    if len(t) < 2 or np.ptp(t) == 0:
        return None
    return float(np.polyfit(t, y, 1)[0])


def acceleration(values: list[float | None]) -> float | None:
    """Slope of the second half of the window minus slope of the first half."""
    n = len(values)
    first, second = slope(values[: n // 2]), slope(values[n // 2 :])
    if first is None or second is None:
        return None
    return second - first


def change_point(values: list[float | None], min_segment: int = 3) -> dict | None:
    """Best single mean-shift split (least squares). Returns index, means and shift."""
    t, y = _clean(values)
    if len(y) < 2 * min_segment:
        return None
    best = None
    for k in range(min_segment, len(y) - min_segment + 1):
        a, b = y[:k], y[k:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, k, a.mean(), b.mean())
    _, k, before, after = best
    spread = float(np.std(y)) or 1.0
    return {"day_index": int(t[k]), "before_mean": float(before), "after_mean": float(after),
            "shift": float(after - before), "shift_in_std": float((after - before) / spread)}


def spikes(values: list[float | None], z: float = 3.5) -> int:
    """Points more than ``z`` robust deviations (MAD) above the median."""
    _, y = _clean(values)
    if len(y) < 5:
        return 0
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826
    if mad == 0:
        return int((y > med).sum()) if np.ptp(y) > 0 else 0
    return int(((y - med) / mad > z).sum())


def analyze_series(values: list[float | None]) -> dict:
    """All trend statistics for one series."""
    _, y = _clean(values)
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": len(y),
        "first": float(y[0]),
        "last": float(y[-1]),
        "delta": float(y[-1] - y[0]),
        "max": float(y.max()),
        "min": float(y.min()),
        "mean": float(y.mean()),
        "slope": slope(values),
        "acceleration": acceleration(values),
        "change_point": change_point(values),
        "spikes": spikes(values),
        "increases": int((np.diff(y) > 0).sum()),
    }


class SeriesArgs(BaseModel):
    column: str = Field(description="Harmonized SMART column, e.g. smart_5_raw")
    values: list[float | None] = Field(description="Daily values, oldest first; null = missing")


@tool("trend_analysis", args_schema=SeriesArgs)
def trend_analysis(column: str, values: list[float | None]) -> dict:
    """Slope, acceleration, change point, spikes and counts for one SMART series."""
    return {"column": column, **analyze_series(values)}
