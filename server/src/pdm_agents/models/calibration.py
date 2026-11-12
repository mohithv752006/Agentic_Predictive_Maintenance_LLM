"""Probability calibration (isotonic / Platt), fitted on validation data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pdm_agents.models.base import logit, sigmoid

Method = Literal["isotonic", "platt", "identity"]


class Calibrator:
    """Maps raw scores to calibrated probabilities. ``identity`` disables calibration."""

    def __init__(self, method: Method = "isotonic") -> None:
        self.method = method
        self.x_: list[float] = []
        self.y_: list[float] = []
        self.coef_: float = 1.0
        self.intercept_: float = 0.0
        self.fitted = method == "identity"

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> Calibrator:
        p_raw = np.asarray(p_raw, dtype="float64")
        y = np.asarray(y, dtype="float64")
        keep = ~np.isnan(y)
        p_raw, y = p_raw[keep], y[keep]
        if self.method == "identity":
            return self
        if len(np.unique(y)) < 2:
            # Degenerate validation data: fall back to the empirical rate.
            self.method = "platt"
            self.coef_, self.intercept_ = 0.0, float(logit(np.clip(y.mean(), 1e-4, 1 - 1e-4)))
        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p_raw, y)
            self.x_ = iso.X_thresholds_.tolist()
            self.y_ = iso.y_thresholds_.tolist()
        else:
            lr = LogisticRegression(C=1e4, max_iter=1000)
            lr.fit(logit(p_raw).reshape(-1, 1), y)
            self.coef_, self.intercept_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        self.fitted = True
        return self

    def transform(self, p_raw: np.ndarray | float) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype="float64")
        if self.method == "identity":
            return np.clip(p_raw, 0.0, 1.0)
        if self.method == "isotonic":
            return np.interp(p_raw, self.x_, self.y_)
        return sigmoid(self.coef_ * logit(p_raw) + self.intercept_)

    def to_dict(self) -> dict:
        return {"method": self.method, "x": self.x_, "y": self.y_, "coef": self.coef_,
                "intercept": self.intercept_}

    @classmethod
    def from_dict(cls, d: dict) -> Calibrator:
        cal = cls(d["method"])
        cal.x_, cal.y_ = d["x"], d["y"]
        cal.coef_, cal.intercept_ = d["coef"], d["intercept"]
        cal.fitted = True
        return cal

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
