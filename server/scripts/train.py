"""Train family/global LightGBM, sequence model, calibrators and fusion for each seed.

Usage: python -m scripts.train [--config extra.yaml] [--seeds 0,1,2] [--no-sequence]
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pdm_agents.data.store import Store
from pdm_agents.models.bundle import bundle_dir, train_bundle
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path | None = typer.Option(None, help="Extra YAML merged over default.yaml"),
    seeds: str | None = typer.Option(None, help="Comma-separated seeds (default: config)"),
    no_sequence: bool = typer.Option(False, help="Skip the sequence baseline"),
) -> None:
    cfg = bootstrap(config_file=config)
    store = Store(cfg)
    if not store.ready:
        raise typer.Exit("no harmonized data; run scripts.prepare first")
    seed_list = [int(s) for s in seeds.split(",")] if seeds else cfg.seeds
    for seed in seed_list:
        bundle = train_bundle(cfg, store, seed=seed, train_sequence=not no_sequence)
        out = bundle_dir(cfg, seed)
        bundle.save(out)
        typer.echo(f"seed {seed}: saved to {out}")
        typer.echo(json.dumps(bundle.meta["val_metrics"], indent=2))


if __name__ == "__main__":
    app()
