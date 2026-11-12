"""Knowledge index over ``knowledge_base/*.md`` with a small CPU embedder (FAISS).

Each ``## `` section of a document becomes one snippet with id
``K-<file stem>-<n>``. Embeddings: ``BAAI/bge-small-en-v1.5`` (configurable),
with the model's recommended query instruction for retrieval.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np

from pdm_agents.config import AppConfig
from pdm_agents.logging_setup import get_logger
from pdm_agents.schemas import KnowledgeSnippet

log = get_logger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SECTION_RE = re.compile(r"^## +(.+)$", re.MULTILINE)


def kb_fingerprint(kb_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(kb_dir.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def chunk_markdown(path: Path) -> list[KnowledgeSnippet]:
    """Split a document into one snippet per ``##`` section (title prepended)."""
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# +(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    stem = re.sub(r"[^0-9a-z_]+", "_", path.stem.lower())
    heads = list(SECTION_RE.finditer(text))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = " ".join(text[m.end():end].split())
        out.append(KnowledgeSnippet(id=f"K-{stem}-{i}", source_file=path.name,
                                    text=f"{title} / {m.group(1).strip()}: {body}"))
    return out


class Embedder:
    """Lazy sentence-transformers wrapper (CPU)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query and "bge" in self.model_name.lower():
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


class KnowledgeIndex:
    def __init__(self, snippets: list[KnowledgeSnippet], index: faiss.Index,
                 embedder: Embedder, fingerprint: str) -> None:
        self.snippets = snippets
        self.index = index
        self.embedder = embedder
        self.fingerprint = fingerprint

    @classmethod
    def build(cls, cfg: AppConfig, embedder: Embedder | None = None) -> KnowledgeIndex:
        kb = cfg.paths.knowledge_base
        snippets = [s for p in sorted(kb.glob("*.md")) for s in chunk_markdown(p)]
        if not snippets:
            raise RuntimeError(f"no knowledge base documents in {kb}")
        embedder = embedder or Embedder(cfg.embeddings.model, cfg.embeddings.device)
        vecs = embedder.encode([s.text for s in snippets])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        log.info("knowledge index: %d snippets from %d documents", len(snippets),
                 len({s.source_file for s in snippets}))
        return cls(snippets, index, embedder, kb_fingerprint(kb))

    def query(self, text: str, k: int) -> list[KnowledgeSnippet]:
        vec = self.embedder.encode([text], query=True)
        sims, ids = self.index.search(vec, min(k, len(self.snippets)))
        return [self.snippets[i].model_copy(update={"score": round(float(s), 4)})
                for s, i in zip(sims[0], ids[0], strict=True) if i >= 0]

    def get(self, snippet_id: str) -> KnowledgeSnippet | None:
        return next((s for s in self.snippets if s.id == snippet_id), None)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath("index.faiss").write_bytes(faiss.serialize_index(self.index).tobytes())
        path.joinpath("snippets.json").write_text(json.dumps({
            "fingerprint": self.fingerprint, "model": self.embedder.model_name,
            "snippets": [s.model_dump() for s in self.snippets]}), encoding="utf-8")

    @classmethod
    def load(cls, cfg: AppConfig, path: Path) -> KnowledgeIndex:
        data = json.loads(path.joinpath("snippets.json").read_text(encoding="utf-8"))
        index = faiss.deserialize_index(
            np.frombuffer(path.joinpath("index.faiss").read_bytes(), dtype="uint8"))
        current = kb_fingerprint(cfg.paths.knowledge_base)
        if data["fingerprint"] != current:
            log.warning("knowledge base changed since indexing; run `python -m scripts.index`")
        return cls([KnowledgeSnippet(**s) for s in data["snippets"]], index,
                   Embedder(data["model"], cfg.embeddings.device), data["fingerprint"])


def index_dir(cfg: AppConfig) -> Path:
    return cfg.paths.artifacts / "index"
