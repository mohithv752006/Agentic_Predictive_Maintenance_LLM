"""Fusion of calibrated per-agent risks into one probability per horizon.

Methods (``models.fusion.method``):

* ``stacking``      - logistic regression on agent log-odds + the model prior,
                      fit on out-of-fold calibrated validation probabilities.
* ``log_odds``      - weighted log-odds pooling around the prior; weights are the
                      non-negative stacking coefficients, then Platt-recalibrated.
* ``majority_vote`` - naive baseline (H3 ablation): each agent votes when its
                      calibrated risk exceeds its F1-optimal validation threshold;
                      the probability is the vote share.

A missing agent (disabled or failed) is imputed with the prior, i.e. "no
evidence". Also provides agent-disagreement and uncertainty scores in [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from pdm_agents.models.base import logit, sigmoid
from pdm_agents.models.calibration import Calibrator

FUSION_METHODS = ("stacking", "log_odds", "majority_vote")


class Fusion:
    """Per-horizon fusion over a fixed, ordered list of agents."""

    def __init__(self, agents: list[str], horizon: int) -> None:
        self.agents = list(agents)
        self.horizon = horizon
        self.coef_: list[float] = []
        self.intercept_: float = 0.0
        self.prior_coef_: float = 0.0
        self.weights_: list[float] = []
        self.pool_cal: Calibrator | None = None
        self.thresholds_: list[float] = []

    # ------------------------------------------------------------------ fit

    def _matrix(self, risks: dict[str, np.ndarray], prior: np.ndarray) -> np.ndarray:
        cols = []
        for a in self.agents:
            r = risks.get(a)
            col = np.where(np.isnan(r), prior, r) if r is not None else prior
            cols.append(logit(col))
        return np.column_stack([*cols, logit(prior)])

    def fit(self, risks: dict[str, np.ndarray], prior: np.ndarray, y: np.ndarray) -> Fusion:
        """Fit every method from calibrated (out-of-fold) agent risks on validation data."""
        keep = ~np.isnan(y)
        risks = {a: np.asarray(r)[keep] for a, r in risks.items()}
        prior, y = np.asarray(prior)[keep], np.asarray(y)[keep]
        X = self._matrix(risks, prior)
        if len(np.unique(y)) < 2:
            self.coef_ = [0.0] * len(self.agents)
            self.prior_coef_, self.intercept_ = 1.0, 0.0
        else:
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(X, y)
            self.coef_ = lr.coef_[0, :-1].tolist()
            self.prior_coef_ = float(lr.coef_[0, -1])
            self.intercept_ = float(lr.intercept_[0])
        w = np.clip(np.asarray(self.coef_), 0, None)
        self.weights_ = (w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / max(len(w), 1)))
        self.weights_ = self.weights_.tolist()
        self.pool_cal = Calibrator("platt").fit(sigmoid(self._pool_logit(X)), y)
        self.thresholds_ = []
        for i, a in enumerate(self.agents):
            p = sigmoid(X[:, i])
            self.thresholds_.append(_f1_threshold(p, y) if a in risks else 1.0)
        return self

    def _pool_logit(self, X: np.ndarray) -> np.ndarray:
        prior_l = X[:, -1]
        return prior_l + (X[:, :-1] - prior_l[:, None]) @ np.asarray(self.weights_)

    # -------------------------------------------------------------- predict

    def predict(self, risks: dict[str, np.ndarray], prior: np.ndarray,
                method: str = "stacking") -> np.ndarray:
        prior = np.asarray(prior, dtype="float64")
        X = self._matrix({a: np.asarray(r, dtype="float64") for a, r in risks.items()}, prior)
        if method == "stacking":
            z = X[:, :-1] @ np.asarray(self.coef_) + self.prior_coef_ * X[:, -1] + self.intercept_
            return sigmoid(z)
        if method == "log_odds":
            assert self.pool_cal is not None
            return self.pool_cal.transform(sigmoid(self._pool_logit(X)))
        if method == "majority_vote":
            present = [i for i, a in enumerate(self.agents) if a in risks]
            if not present:
                return np.zeros(len(prior))
            votes = np.column_stack([sigmoid(X[:, i]) >= self.thresholds_[i] for i in present])
            return votes.mean(axis=1)
        raise ValueError(f"unknown fusion method {method!r}")

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {"agents": self.agents, "horizon": self.horizon, "coef": self.coef_,
                "intercept": self.intercept_, "prior_coef": self.prior_coef_,
                "weights": self.weights_, "thresholds": self.thresholds_,
                "pool_cal": self.pool_cal.to_dict() if self.pool_cal else None}

    @classmethod
    def from_dict(cls, d: dict) -> Fusion:
        f = cls(d["agents"], d["horizon"])
        f.coef_, f.intercept_, f.prior_coef_ = d["coef"], d["intercept"], d["prior_coef"]
        f.weights_, f.thresholds_ = d["weights"], d["thresholds"]
        f.pool_cal = Calibrator.from_dict(d["pool_cal"]) if d["pool_cal"] else None
        return f

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Fusion:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _f1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 1.0
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 1.0


def disagreement(risks: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Conflict between the agents the fusion relies on, in [0, 1].

    ``2 x`` the weighted standard deviation of calibrated agent risks, weighted by the
    fusion's (non-negative, normalized) agent weights. Specialists are expected to
    differ (a media failure need not show in thermal data), so agents the fusion
    barely uses barely count. Without weights, all agents count equally.
    """
    items = [(a, v) for a, v in risks.items() if v is not None and not np.isnan(v)]
    if len(items) < 2:
        return 0.0
    w = np.array([(weights or {}).get(a, 1.0 if weights is None else 0.0) for a, _ in items])
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    p = np.array([v for _, v in items])
    mean = float(np.sum(w * p))
    return float(np.clip(2 * np.sqrt(np.sum(w * (p - mean) ** 2)), 0, 1))


def disagreement_batch(risks: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Vectorized ``disagreement`` over rows."""
    agents = [a for a in risks if weights.get(a, 0) > 0]
    if len(agents) < 2:
        return np.zeros(len(next(iter(risks.values())))) if risks else np.zeros(0)
    w = np.array([weights[a] for a in agents])
    w = w / w.sum()
    p = np.column_stack([risks[a] for a in agents])
    mean = p @ w
    return np.clip(2 * np.sqrt(((p - mean[:, None]) ** 2) @ w), 0, 1)


def uncertainty(p: float, coverage: float) -> float:
    """Decision uncertainty in [0, 1].

    Normalized binary entropy of the fused probability (1 at p = 0.5), inflated
    when evidence is missing: ``1 - (1 - H(p)) * coverage``, where ``coverage`` is
    the fraction of the window observed times the fraction of agents that
    contributed.
    """
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(np.clip(1 - (1 - h) * np.clip(coverage, 0, 1), 0, 1))
