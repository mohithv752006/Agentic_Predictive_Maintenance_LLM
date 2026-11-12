"""Bundle evaluation results for the Colab results notebook (results_and_ablation.ipynb).

Usage:
  python -m scripts.export_results
  python -m scripts.export_results --final outputs/reports/final_results-<stamp> \
      --llm outputs/reports/ablation_llm-<stamp>

Writes outputs/results_bundle.zip with:
  final/   tables/*.csv, metrics.json, config.yaml, predictions/*.parquet
           (numeric systems + ablations; config.yaml has the API key masked)
  llm/     tables/llm_sample.csv, metrics.json                  (live graph-run ablation)
  dataset_summary.json                                          (dataset and split sizes)
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import typer

from pdm_agents.config import AppConfig
from pdm_agents.data.features import features_dir
from pdm_agents.data.labels import label_column
from pdm_agents.data.store import Store
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


def _latest(reports: Path, prefix: str) -> Path | None:
    runs = sorted(p for p in reports.glob(f"{prefix}-*") if (p / "metrics.json").exists())
    return runs[-1] if runs else None


def dataset_summary(cfg: AppConfig) -> dict:
    store = Store(cfg)
    per_model = store.summary()
    for col in ("first_date", "last_date"):
        per_model[col] = per_model[col].astype(str)
    splits_path = store.meta_dir / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else {}
    windows = {}
    for split in ("train", "val", "test", "unseen"):
        path = features_dir(cfg) / f"{split}.parquet"
        if not path.exists():
            continue
        with path.open("rb") as fh:
            y = pd.read_parquet(fh, columns=[label_column(h) for h in cfg.horizons])
        windows[split] = {"windows": len(y),
                          **{f"positives_{h}": int((y[label_column(h)] == 1).sum())
                             for h in cfg.horizons}}
    return {"per_model": per_model.to_dict(orient="records"),
            "models": store.models_meta(), "splits": splits, "windows": windows,
            "horizons": cfg.horizons}


def _add(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    method = zipfile.ZIP_STORED if src.suffix == ".parquet" else zipfile.ZIP_DEFLATED
    zf.write(src, arcname, compress_type=method)


@app.command()
def main(
    final: Path | None = typer.Option(None, help="final_results report dir (default: latest)"),
    llm: Path | None = typer.Option(None, help="ablation_llm report dir (default: latest)"),
    out: Path | None = typer.Option(None, help="zip path (default: outputs/results_bundle.zip)"),
) -> None:
    cfg = bootstrap()
    reports = cfg.paths.reports
    final = final or _latest(reports, "final_results")
    if final is None or not (final / "predictions").exists():
        raise typer.BadParameter("no final_results report with predictions; run "
                                 "configs/experiments/final_results.yaml first")
    llm = llm or _latest(reports, "ablation_llm")
    out = out or cfg.paths.reports.parent / "results_bundle.zip"
    with zipfile.ZipFile(out, "w") as zf:
        _add_dir(zf, final, "final", ["metrics.json", "config.yaml", "tables/*.csv",
                                      "predictions/*.parquet"])
        if llm is not None:
            _add_dir(zf, llm, "llm", ["metrics.json", "tables/llm_sample.csv"])
        else:
            typer.echo("warning: no ablation_llm report found; the notebook will skip that part")
        zf.writestr("dataset_summary.json", json.dumps(dataset_summary(cfg), indent=2))
    typer.echo(f"final: {final.name}")
    typer.echo(f"llm:   {llm.name if llm else '-'}")
    typer.echo(f"bundle written to {out} ({out.stat().st_size / 1e6:.1f} MB)")


def _add_dir(zf: zipfile.ZipFile, root: Path, prefix: str, patterns: list[str]) -> None:
    for pattern in patterns:
        for src in sorted(root.glob(pattern)):
            _add(zf, src, f"{prefix}/{src.relative_to(root).as_posix()}")


if __name__ == "__main__":
    app()
