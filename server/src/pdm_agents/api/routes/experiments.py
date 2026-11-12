"""GET /experiments and GET /experiments/{id}: finished evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from fastapi import APIRouter, Depends, HTTPException

from pdm_agents.api.deps import AppState, get_state
from pdm_agents.schemas import ExperimentDetail, ExperimentSummary

router = APIRouter(tags=["experiments"])


def _reports(state: AppState) -> list[Path]:
    root = state.cfg.paths.reports
    return sorted((p for p in root.glob("*/metrics.json")), key=lambda p: p.stat().st_mtime,
                  reverse=True)


@router.get("/experiments", response_model=list[ExperimentSummary])
def list_experiments(state: AppState = Depends(get_state)) -> list[ExperimentSummary]:
    out = []
    for path in _reports(state):
        m = json.loads(path.read_text(encoding="utf-8"))
        out.append(ExperimentSummary(id=path.parent.name, name=m.get("name", path.parent.name),
                                     created_at=m.get("created_at", ""),
                                     systems=m.get("systems", []),
                                     headline=m.get("headline", {})))
    return out


@router.get("/experiments/{exp_id}", response_model=ExperimentDetail)
def get_experiment(exp_id: str, state: AppState = Depends(get_state)) -> ExperimentDetail:
    root = state.cfg.paths.reports / exp_id
    if "/" in exp_id or "\\" in exp_id or not (root / "metrics.json").exists():
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id}")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    tables = {}
    for csv in sorted((root / "tables").glob("*.csv")):
        with csv.open("rb") as fh:
            df = pd.read_csv(fh)
        tables[csv.stem] = json.loads(df.to_json(orient="records"))
    plots = [f"/reports/{exp_id}/plots/{p.name}" for p in sorted((root / "plots").glob("*.png"))]
    cfg_path = root / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return ExperimentDetail(id=exp_id, name=metrics.get("name", exp_id),
                            created_at=metrics.get("created_at", ""), metrics=metrics,
                            tables=tables, plots=plots, config=config)
