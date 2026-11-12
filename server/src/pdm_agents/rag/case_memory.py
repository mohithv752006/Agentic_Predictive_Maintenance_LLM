"""Case memory: FAISS index over numeric trajectory vectors of TRAINING drives only.

Vectors are the window features normalized per drive model (``Normalizer`` fit on
the training split), L2-normalized so inner product = cosine similarity. One index
per drive model (``rag.same_model_only``) plus a pooled index used for models
never seen in training. Results are de-duplicated by serial number.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from pdm_agents.config import AppConfig
from pdm_agents.data.features import Normalizer
from pdm_agents.data.labels import label_column
from pdm_agents.logging_setup import get_logger
from pdm_agents.rag.summarizer import trajectory_summary
from pdm_agents.schemas import RetrievedCase

log = get_logger(__name__)

POOLED = "__pooled__"
META_COLS = ["case_id", "serial_number", "model", "as_of", "outcome_failed",
             "days_to_failure", "summary"]


def case_id(serial: str, as_of: object) -> str:
    return "C-" + hashlib.md5(f"{serial}@{as_of}".encode()).hexdigest()[:8]


def _write_index(index: faiss.Index, path: Path) -> None:
    path.write_bytes(faiss.serialize_index(index).tobytes())


def _read_index(path: Path) -> faiss.Index:
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype="uint8"))


class CaseMemory:
    def __init__(self, normalizer: Normalizer, meta: pd.DataFrame,
                 indexes: dict[str, faiss.Index], positions: dict[str, np.ndarray]) -> None:
        self.normalizer = normalizer
        self.meta = meta.reset_index(drop=True)
        self.indexes = indexes
        self.positions = positions  # model -> row numbers of ``meta`` in index order

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, cfg: AppConfig, train: pd.DataFrame, normalizer: Normalizer,
              seed: int = 0) -> CaseMemory:
        """Index training windows: every positive + up to ``case_index_max_negatives``."""
        if not train.empty and set(train["split"].unique()) != {"train"}:
            raise ValueError("case memory may only contain training-split windows")
        ymax = label_column(cfg.max_horizon)
        pos = train[train[ymax] == 1]
        neg = train[train[ymax] == 0]
        cap = cfg.rag.case_index_max_negatives
        if len(neg) > cap:
            neg = neg.iloc[np.random.default_rng(seed).choice(len(neg), cap, replace=False)]
        df = pd.concat([pos, neg]).reset_index(drop=True)
        vecs = normalizer.transform(df).astype("float32")
        faiss.normalize_L2(vecs)
        meta = pd.DataFrame({
            "case_id": [case_id(s, a) for s, a in zip(df["serial_number"], df["as_of"],
                                                       strict=True)],
            "serial_number": df["serial_number"].astype(str),
            "model": df["model"].astype(str),
            "as_of": pd.to_datetime(df["as_of"]).dt.date.astype(str),
            "outcome_failed": (df[ymax] == 1).to_numpy(),
            "days_to_failure": df["days_to_failure"].astype("float64"),
            "summary": [trajectory_summary(r, cfg) for r in df.to_dict(orient="records")],
        })
        indexes, positions = {}, {}
        groups = {POOLED: np.arange(len(df)),
                  **{str(m): idx for m, idx in df.groupby("model").indices.items()}}
        for model, idx in groups.items():
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs[idx])
            indexes[model], positions[model] = index, np.asarray(idx)
        log.info("case memory: %d cases (%d failed) across %d models", len(meta),
                 int(meta["outcome_failed"].sum()), len(groups) - 1)
        return cls(normalizer, meta, indexes, positions)

    # ------------------------------------------------------------------ query

    def query(self, row: pd.DataFrame, k: int, same_model_only: bool = True,
              exclude_serial: str | None = None) -> list[RetrievedCase]:
        """Top-``k`` similar training drives (one case per serial) for a 1-row frame."""
        model = str(row["model"].iloc[0])
        key = model if same_model_only and model in self.indexes else POOLED
        index = self.indexes[key]
        vec = self.normalizer.transform(row).astype("float32")
        faiss.normalize_L2(vec)
        sims, ids = index.search(vec, min(index.ntotal, max(k * 10, 50)))
        out, seen = [], set()
        for sim, i in zip(sims[0], ids[0], strict=True):
            if i < 0:
                continue
            m = self.meta.iloc[int(self.positions[key][i])]
            if m.serial_number in seen or m.serial_number == exclude_serial:
                continue
            seen.add(m.serial_number)
            dtf = None if pd.isna(m.days_to_failure) else int(m.days_to_failure)
            outcome = (f"failed {dtf} days after this window" if m.outcome_failed
                       else "did not fail within the horizon")
            out.append(RetrievedCase(
                id=m.case_id, serial_number=m.serial_number, model=m.model,
                similarity=round(float(sim), 4), outcome_failed=bool(m.outcome_failed),
                days_to_failure=dtf if m.outcome_failed else None,
                summary=f"{m.as_of}: {outcome}. {m.summary}"))
            if len(out) >= k:
                break
        return out

    def knn_risk(self, df: pd.DataFrame, k: int, horizons: list[int],
                 prior: dict[int, np.ndarray], same_model_only: bool = True,
                 smoothing: float = 1.0) -> dict[int, np.ndarray]:
        """Similarity-weighted share of the ``k`` nearest training drives that failed within
        each horizon, shrunk toward ``prior``. Raw (uncalibrated); one neighbour per serial.
        Rows are processed per drive model in batches."""
        out = {h: np.full(len(df), np.nan) for h in horizons}
        dtf = self.meta["days_to_failure"].to_numpy("float64")
        failed = self.meta["outcome_failed"].to_numpy(bool)
        serial_codes = pd.factorize(self.meta["serial_number"])[0]
        for model, idx in df.groupby(df["model"].astype(str)).indices.items():
            key = model if same_model_only and model in self.indexes else POOLED
            index, pos = self.indexes[key], self.positions[key]
            vec = self.normalizer.transform(df.iloc[idx]).astype("float32")
            faiss.normalize_L2(vec)
            sims, ids = index.search(vec, min(index.ntotal, k * 5))
            for row, (s_row, i_row) in enumerate(zip(sims, ids, strict=True)):
                rows = pos[i_row[i_row >= 0]]
                s_row = s_row[i_row >= 0]
                _, first = np.unique(serial_codes[rows], return_index=True)
                keep = np.sort(first)[:k]
                rows, w = rows[keep], np.clip(s_row[keep], 0, None)
                for h in horizons:
                    hit = failed[rows] & (dtf[rows] <= h)
                    p0 = prior[h][idx[row]]
                    out[h][idx[row]] = (np.sum(w * hit) + smoothing * p0) / (w.sum() + smoothing)
        return out

    @property
    def serials(self) -> set[str]:
        return set(self.meta["serial_number"])

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.normalizer.save(path / "normalizer.json")
        with (path / "cases.parquet").open("wb") as fh:
            self.meta.to_parquet(fh, index=False)
        names = {}
        for i, (model, index) in enumerate(self.indexes.items()):
            _write_index(index, path / f"index_{i}.faiss")
            np.save(path / f"positions_{i}.npy", self.positions[model])
            names[str(i)] = model
        (path / "indexes.json").write_text(json.dumps(names), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CaseMemory:
        names = json.loads((path / "indexes.json").read_text(encoding="utf-8"))
        with (path / "cases.parquet").open("rb") as fh:
            meta = pd.read_parquet(fh)
        indexes = {m: _read_index(path / f"index_{i}.faiss") for i, m in names.items()}
        positions = {m: np.load(path / f"positions_{i}.npy") for i, m in names.items()}
        return cls(Normalizer.load(path / "normalizer.json"), meta, indexes, positions)
