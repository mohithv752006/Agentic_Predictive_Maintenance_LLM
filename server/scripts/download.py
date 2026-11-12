"""Download the configured Backblaze Drive Stats quarters.

Usage: python -m scripts.download [--config extra.yaml] [--list-only]
"""

from __future__ import annotations

from pathlib import Path

import typer

from pdm_agents.data.download import discover, download_all
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path | None = typer.Option(None, help="Extra YAML merged over default.yaml"),
    list_only: bool = typer.Option(False, help="Only list the files that would be downloaded"),
) -> None:
    cfg = bootstrap(config_file=config)
    if list_only:
        for f in discover(cfg):
            typer.echo(f"{f.filename}  {f.url}")
        return
    paths = download_all(cfg)
    total = sum(p.stat().st_size for p in paths)
    typer.echo(f"{len(paths)} files, {total / 1e9:.2f} GB in {cfg.paths.raw}")


if __name__ == "__main__":
    app()
