"""Build the case memory (training windows only), the knowledge index and the
fleet snapshot (numeric risk + action for every drive at its latest date).

Usage: python -m scripts.index [--config extra.yaml] [--seed 0]
"""

from __future__ import annotations

from pathlib import Path

import typer

from pdm_agents.config import AppConfig
from pdm_agents.data.features import load_split
from pdm_agents.data.store import Store
from pdm_agents.models.bundle import bundle_dir, fit_case_fusion, load_bundle
from pdm_agents.models.scoring import fleet_snapshot, snapshot_path
from pdm_agents.rag.case_memory import CaseMemory
from pdm_agents.rag.knowledge_index import KnowledgeIndex, index_dir
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


def build_indexes(cfg: AppConfig, seed: int | None = None) -> tuple[CaseMemory, KnowledgeIndex]:
    """Build and save both indexes; the case normalizer comes from the trained bundle."""
    bundle = load_bundle(cfg, seed)
    if bundle is None:
        raise RuntimeError("no trained model bundle; run `python -m scripts.train` first")
    cases = CaseMemory.build(cfg, load_split(cfg, "train"), bundle.normalizer)
    cases.save(index_dir(cfg) / "cases")
    fit_case_fusion(cfg, bundle, cases)  # calibrated kNN risk + retrieval-aware fusion (val)
    bundle.save(bundle_dir(cfg, cfg.seeds[0] if seed is None else seed))
    knowledge = KnowledgeIndex.build(cfg)
    knowledge.save(index_dir(cfg) / "knowledge")
    snapshot = fleet_snapshot(cfg, Store(cfg), bundle, cases)
    with snapshot_path(cfg).open("wb") as fh:
        snapshot.to_parquet(fh, index=False)
    return cases, knowledge


@app.command()
def main(
    config: Path | None = typer.Option(None, help="Extra YAML merged over default.yaml"),
    seed: int | None = typer.Option(None, help="Bundle seed whose normalizer to use"),
) -> None:
    cfg = bootstrap(config_file=config)
    cases, knowledge = build_indexes(cfg, seed)
    typer.echo(f"case memory: {len(cases.meta)} cases from {len(cases.serials)} training drives")
    typer.echo(f"knowledge index: {len(knowledge.snippets)} snippets")


if __name__ == "__main__":
    app()
