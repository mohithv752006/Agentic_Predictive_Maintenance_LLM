"""Evaluation metrics: ranking, operating points, lead time, calibration, cost, selection.

All functions ignore censored labels (NaN). Bootstrap resampling is clustered by
drive serial because windows of the same drive are correlated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _valid(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    m = ~np.isnan(y) & ~np.isnan(p)
    return y[m], p[m]


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _valid(y, p)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _valid(y, p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def threshold_at_far(y: np.ndarray, p: np.ndarray, far: float) -> float:
    """Score threshold whose false-alarm rate on negatives is at most ``far``."""
    y, p = _valid(y, p)
    neg = np.sort(p[y == 0])
    if len(neg) == 0:
        return float("nan")
    k = int(np.floor(far * len(neg)))  # number of negatives allowed above threshold
    return float(neg[len(neg) - k - 1] if k < len(neg) else neg[0]) + 1e-12


def recall_at_far(y: np.ndarray, p: np.ndarray, far: float) -> float:
    y, p = _valid(y, p)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    thr = threshold_at_far(y, p, far)
    return float((p[y == 1] >= thr).mean())


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass (quantile) bins (suits rare events)."""
    y, p = _valid(y, p)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    total = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            total += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return float(total / len(y))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed rate per equal-mass bin (for reliability diagrams)."""
    y, p = _valid(y, p)
    order = np.argsort(p)
    rows = [{"mean_predicted": float(p[i].mean()), "observed_rate": float(y[i].mean()),
             "n": len(i)} for i in np.array_split(order, bins) if len(i)]
    return pd.DataFrame(rows)


def lead_time(df: pd.DataFrame, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Earliest alarm before failure among failing drives.

    ``df`` holds every daily window of each failing drive in the 30 days before its
    failure (``serial_number``, ``days_to_failure``). A drive is detected if any of
    its windows scores >= ``threshold``; lead time = days_to_failure of the first alarm.
    """
    d = df.assign(score=score)
    alarms = d[d["score"] >= threshold]
    first = alarms.groupby("serial_number")["days_to_failure"].max()
    n = d["serial_number"].nunique()
    return {
        "failing_drives": float(n),
        "detected_share": float(len(first) / n) if n else float("nan"),
        "mean_lead_days": float(first.mean()) if len(first) else float("nan"),
        "median_lead_days": float(first.median()) if len(first) else float("nan"),
    }


def selective_curve(correct: np.ndarray, confidence: np.ndarray,
                    coverages: list[float]) -> pd.DataFrame:
    """Accuracy of the most confident ``coverage`` fraction of decisions."""
    m = ~np.isnan(correct)
    correct, confidence = correct[m], confidence[m]
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for c in coverages:
        k = max(1, round(c * len(order)))
        rows.append({"coverage": c, "selective_accuracy": float(correct[order[:k]].mean()),
                     "n": k})
    return pd.DataFrame(rows)


def cluster_bootstrap(
    groups: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int, seed: int = 0
) -> np.ndarray:
    """Statistic on ``n_boot`` resamples of whole groups; ``stat`` receives row indices."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == g)[0] for g in range(len(uniq))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        out[b] = stat(idx)
    return out


def ci95(samples: np.ndarray) -> tuple[float, float]:
    s = samples[~np.isnan(samples)]
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired_bootstrap_pvalue(diffs: np.ndarray) -> float:
    """Two-sided p-value for H0: mean difference = 0, from bootstrap differences."""
    d = diffs[~np.isnan(diffs)]
    if len(d) == 0:
        return float("nan")
    p_one = min((d <= 0).mean(), (d >= 0).mean())
    return float(min(1.0, 2 * p_one))
