"""Sequence baseline: 1D-CNN or LSTM over the raw 30-day SMART window (CPU-sized).

Inputs are per-drive-model normalized SMART values (``features.Normalizer``)
plus a per-day presence mask; one network predicts every horizon at once with a
masked BCE loss (censored labels are ignored). Early stopping uses a
serial-grouped hold-out of the training split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pdm_agents.config import SequenceConfig
from pdm_agents.data.features import Normalizer
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.base import grouped_holdout

log = get_logger(__name__)


@dataclass
class SequenceBatch:
    """Raw windows ``values`` (n, window, channels) with the drive model of each row."""

    values: np.ndarray
    models: np.ndarray
    groups: np.ndarray | None = None  # serial numbers, for grouped early stopping


class _CNN(nn.Module):
    def __init__(self, c_in: int, hidden: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
        )
        self.head = nn.Linear(2 * hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (b, t, c)
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


class _LSTM(nn.Module):
    def __init__(self, c_in: int, hidden: int, n_out: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(c_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.rnn(x)
        return self.head(h[-1])


class SequenceModel:
    """Multi-horizon sequence classifier. ``predict_proba`` returns ``(n, n_horizons)``."""

    def __init__(self, columns: list[str], horizons: list[int],
                 params: SequenceConfig | None = None, seed: int = 0) -> None:
        self.name = "sequence"
        self.columns = list(columns)
        self.horizons = list(horizons)
        self.params = params or SequenceConfig()
        self.seed = seed
        self.normalizer: Normalizer | None = None
        self.net: nn.Module | None = None

    def _build(self) -> nn.Module:
        cls = _CNN if self.params.kind == "cnn" else _LSTM
        return cls(len(self.columns) + 1, self.params.hidden, len(self.horizons))

    def _prepare(self, batch: SequenceBatch) -> torch.Tensor:
        assert self.normalizer is not None
        vals = batch.values.astype("float64")
        present = (~np.isnan(vals).all(axis=2)).astype("float32")[..., None]
        out = np.zeros(vals.shape, dtype="float32")
        for model in np.unique(batch.models):
            idx = np.where(batch.models == model)[0]
            out[idx] = self.normalizer.transform_array(vals[idx], str(model))
        return torch.from_numpy(np.concatenate([out, present], axis=2))

    def fit(self, X: SequenceBatch, y: np.ndarray, X_val: SequenceBatch | None = None,
            y_val: np.ndarray | None = None) -> SequenceModel:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        flat = X.values.reshape(-1, X.values.shape[2])
        flat_models = np.repeat(X.models, X.values.shape[1])
        take = rng.choice(len(flat), size=min(len(flat), 300_000), replace=False)
        self.normalizer = Normalizer.fit(
            pd.DataFrame(flat[take], columns=self.columns).assign(model=flat_models[take]),
            self.columns,
        )
        if X_val is None:
            groups = X.groups if X.groups is not None else np.arange(len(y))
            hold = grouped_holdout(np.asarray(groups), 0.15, self.seed)
            X_val = SequenceBatch(X.values[hold], X.models[hold])
            y_val = y[hold]
            X = SequenceBatch(X.values[~hold], X.models[~hold])
            y = y[~hold]
        xt, yt = self._prepare(X), torch.from_numpy(np.asarray(y, dtype="float32"))
        xv, yv = self._prepare(X_val), torch.from_numpy(np.asarray(y_val, dtype="float32"))
        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.params.learning_rate)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(self.params.epochs):
            self.net.train()
            perm = torch.from_numpy(rng.permutation(len(xt)))
            for i in range(0, len(perm), self.params.batch_size):
                b = perm[i : i + self.params.batch_size]
                loss = _masked_bce(self.net(xt[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_loss = self._eval_loss(xv, yv)
            log.info("sequence epoch %d: val loss %.4f", epoch + 1, val_loss)
            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.params.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _eval_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        assert self.net is not None
        self.net.eval()
        with torch.no_grad():
            return float(_masked_bce(self.net(x), y))

    def predict_proba(self, X: SequenceBatch) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("sequence model is not fitted")
        self.net.eval()
        x = self._prepare(X)
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                outs.append(torch.sigmoid(self.net(x[i : i + 4096])).numpy())
        return np.concatenate(outs) if outs else np.zeros((0, len(self.horizons)))

    def save(self, path: Path) -> None:
        assert self.net is not None and self.normalizer is not None
        path.mkdir(parents=True, exist_ok=True)
        with (path / "weights.pt").open("wb") as fh:
            torch.save(self.net.state_dict(), fh)
        self.normalizer.save(path / "normalizer.json")
        (path / "meta.json").write_text(json.dumps({
            "columns": self.columns, "horizons": self.horizons, "seed": self.seed,
            "params": self.params.model_dump()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SequenceModel:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        model = cls(meta["columns"], meta["horizons"], SequenceConfig(**meta["params"]),
                    meta["seed"])
        model.normalizer = Normalizer.load(path / "normalizer.json")
        model.net = model._build()
        with (path / "weights.pt").open("rb") as fh:
            model.net.load_state_dict(torch.load(fh, weights_only=True))
        return model


def _masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(y)
    target = torch.nan_to_num(y, nan=0.0)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)
