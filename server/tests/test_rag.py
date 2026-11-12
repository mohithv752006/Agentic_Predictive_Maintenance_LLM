"""RAG: case memory holds only training drives; knowledge retrieval is grounded."""

from __future__ import annotations

import pytest

from pdm_agents.data.features import load_split
from pdm_agents.data.labels import label_column
from pdm_agents.rag.case_memory import CaseMemory
from pdm_agents.rag.knowledge_index import KnowledgeIndex, chunk_markdown, index_dir
from pdm_agents.rag.summarizer import column_label, trajectory_summary


def test_case_memory_contains_only_training_serials(indexed_cfg):
    cases = CaseMemory.load(index_dir(indexed_cfg) / "cases")
    train = set(load_split(indexed_cfg, "train")["serial_number"])
    for split in ("val", "test", "unseen"):
        assert not cases.serials & set(load_split(indexed_cfg, split)["serial_number"])
    assert cases.serials <= train
    assert cases.meta["outcome_failed"].any()


def test_case_memory_refuses_non_training_rows(indexed_cfg):
    from pdm_agents.models.bundle import load_bundle

    val = load_split(indexed_cfg, "val")
    with pytest.raises(ValueError, match="training-split"):
        CaseMemory.build(indexed_cfg, val, load_bundle(indexed_cfg).normalizer)


def test_case_query_same_model_unique_serials_and_signal(indexed_cfg):
    cases = CaseMemory.load(index_dir(indexed_cfg) / "cases")
    test = load_split(indexed_cfg, "test")
    pos = test[test[label_column(7)] == 1]
    row = (pos if not pos.empty else test).head(1)
    hits = cases.query(row, k=5, same_model_only=True)
    assert 1 <= len(hits) <= 5
    assert len({h.serial_number for h in hits}) == len(hits)
    assert all(h.model == row["model"].iloc[0] for h in hits)
    assert all(h.id.startswith("C-") for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.similarity)
    if not pos.empty:  # a degrading drive should find failed neighbours
        assert any(h.outcome_failed for h in hits)
    unseen = load_split(indexed_cfg, "unseen").head(1)
    assert cases.query(unseen, k=3)  # unseen model -> pooled index


def test_knowledge_chunks_and_retrieval(indexed_cfg):
    snippets = chunk_markdown(indexed_cfg.paths.knowledge_base / "media_health.md")
    assert snippets[0].id == "K-media_health-0"
    assert all("Sources" not in s.text.split(":")[0] for s in snippets)
    kb = KnowledgeIndex.load(indexed_cfg, index_dir(indexed_cfg) / "knowledge")
    hits = kb.query("reallocated sectors increasing quickly", k=3)
    assert len(hits) == 3
    assert hits[0].source_file == "media_health.md"
    hits = kb.query("drive temperature too hot", k=2)
    assert any(h.source_file == "thermal.md" for h in hits)
    assert kb.get(hits[0].id) is not None


def test_summaries_are_readable(featured_cfg):
    row = load_split(featured_cfg, "train").iloc[0].to_dict()
    text = trajectory_summary(row, featured_cfg)
    assert "TB" in text and "->" in text
    assert column_label("smart_5_raw") == "reallocated sectors (SMART 5)"
