"""Harmonize downloaded zips into Parquet, then build labels, splits and features.

Usage: python -m scripts.prepare [--config extra.yaml] [--skip-harmonize] [--workers 4]
"""

from __future__ import annotations

from pathlib import Path

import typer

from pdm_agents.data.harmonize import prepare
from pdm_agents.data.store import Store, summarize
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path | None = typer.Option(None, help="Extra YAML merged over default.yaml"),
    skip_harmonize: bool = typer.Option(False, help="Reuse existing Parquet"),
    skip_features: bool = typer.Option(False, help="Only harmonize"),
    workers: int = typer.Option(4, help="Parallel CSV readers"),
    keep_staging: bool = typer.Option(True, help="Keep per-day staging Parquet"),
) -> None:
    cfg = bootstrap(config_file=config)
    if not skip_harmonize:
        prepare(cfg, workers=workers, keep_staging=keep_staging)
    store = Store(cfg)
    if not store.ready:
        raise typer.Exit("no harmonized data found; run without --skip-harmonize")
    typer.echo(summarize(store))
    if not skip_features:
        from pdm_agents.data.features import build_feature_tables

        report = build_feature_tables(cfg, store)
        typer.echo(report)


if __name__ == "__main__":
    app()
