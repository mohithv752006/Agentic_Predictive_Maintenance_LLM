"""Model protocol shared by all probability models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ProbabilityModel(Protocol):
    """Binary probability model: fit / predict_proba / save / load.

    ``X`` is a ``pandas.DataFrame`` of features for tabular models and a
    ``(n, window, channels)`` array for sequence models.
    """

    name: str

    def fit(self, X: Any, y: np.ndarray, X_val: Any | None = None,
            y_val: np.ndarray | None = None) -> ProbabilityModel: ...

    def predict_proba(self, X: Any) -> np.ndarray:
        """P(y = 1) for each row, shape ``(n,)``."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> ProbabilityModel: ...


def logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="float64")))


def grouped_holdout(groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask selecting about ``frac`` of the unique groups (e.g. serials)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    chosen = set(rng.choice(uniq, size=max(1, int(len(uniq) * frac)), replace=False))
    return np.fromiter((g in chosen for g in groups), dtype=bool, count=len(groups))
