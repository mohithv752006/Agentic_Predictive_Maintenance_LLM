"""Run one or more experiment YAMLs (baselines, ablations, cost sweep, drift) and write reports.

Usage:
  python -m scripts.evaluate --experiment configs/experiments/full.yaml
  python -m scripts.evaluate --experiment configs/experiments/ablate_no_calibration.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer

from pdm_agents.evaluation.runner import run_experiment
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


@app.command()
def main(
    experiment: list[Path] = typer.Option(..., help="Experiment YAML (repeatable)"),
) -> None:
    bootstrap()
    for path in experiment:
        out = run_experiment(path)
        typer.echo(f"{path.name}: report written to {out}")


if __name__ == "__main__":
    app()
