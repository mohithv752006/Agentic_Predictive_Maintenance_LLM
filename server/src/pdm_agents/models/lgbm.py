"""LightGBM models: one per SMART family (agent) and one on all features."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from pdm_agents.config import LGBMConfig
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)


class LGBMModel:
    """Gradient-boosted trees on window features. Saved as LightGBM text + JSON metadata."""

    def __init__(self, name: str, features: list[str], params: LGBMConfig | None = None,
                 seed: int = 0) -> None:
        self.name = name
        self.features = list(features)
        self.params = params or LGBMConfig()
        self.seed = seed
        self.booster: lgb.Booster | None = None
        self.best_iteration: int | None = None
        # LightGBM boosters are not safe for concurrent predict calls (graph agents and
        # API requests run in threads), so predictions on one model are serialized.
        self._lock = threading.Lock()

    def _lgb_params(self) -> dict[str, Any]:
        p = self.params
        return {
            "objective": "binary",
            "learning_rate": p.learning_rate,
            "num_leaves": p.num_leaves,
            "min_child_samples": p.min_child_samples,
            "bagging_fraction": p.subsample,
            "bagging_freq": 1,
            "feature_fraction": p.colsample_bytree,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": p.num_threads,
            "verbosity": -1,
            "feature_pre_filter": False,
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame | None = None,
            y_val: np.ndarray | None = None) -> LGBMModel:
        """Train; early-stops on ``(X_val, y_val)`` when it contains both classes."""
        train = lgb.Dataset(X[self.features].to_numpy("float64"), label=y,
                            feature_name=self.features, free_raw_data=True)
        callbacks = []
        valid_sets = []
        if X_val is not None and y_val is not None and len(np.unique(y_val)) == 2:
            valid_sets = [lgb.Dataset(X_val[self.features].to_numpy("float64"), label=y_val,
                                      reference=train)]
            callbacks.append(lgb.early_stopping(self.params.early_stopping_rounds,
                                                verbose=False))
        self.booster = lgb.train(self._lgb_params(), train,
                                 num_boost_round=self.params.n_estimators,
                                 valid_sets=valid_sets, callbacks=callbacks)
        self.best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError(f"model {self.name} is not fitted")
        x = X[self.features].to_numpy("float64")
        with self._lock:
            return self.booster.predict(x, num_iteration=self.best_iteration)

    def feature_importance(self, top: int = 10) -> list[tuple[str, float]]:
        if self.booster is None:
            return []
        gains = self.booster.feature_importance(importance_type="gain")
        order = np.argsort(gains)[::-1][:top]
        return [(self.features[i], float(gains[i])) for i in order]

    def save(self, path: Path) -> None:
        """Write ``<path>.txt`` (model) and ``<path>.json`` (metadata) via Python file I/O."""
        if self.booster is None:
            raise RuntimeError("cannot save an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".txt").write_text(self.booster.model_to_string(), encoding="utf-8")
        meta = {"name": self.name, "features": self.features, "seed": self.seed,
                "best_iteration": self.best_iteration, "params": self.params.model_dump()}
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LGBMModel:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        model = cls(meta["name"], meta["features"], LGBMConfig(**meta["params"]), meta["seed"])
        model.booster = lgb.Booster(model_str=path.with_suffix(".txt").read_text(encoding="utf-8"))
        model.best_iteration = meta["best_iteration"]
        return model
