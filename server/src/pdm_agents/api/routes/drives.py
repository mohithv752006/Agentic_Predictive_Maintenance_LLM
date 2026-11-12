"""GET /drives (paginated fleet list) and GET /drives/{serial} (SMART history)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from pdm_agents.api.deps import AppState, get_state
from pdm_agents.data.features import agent_columns
from pdm_agents.schemas import (
    DriveHistory,
    DriveListResponse,
    DriveSummary,
    FleetStats,
    HistogramBin,
    SmartSeries,
)

router = APIRouter(tags=["drives"])


def _fleet(state: AppState) -> pd.DataFrame:
    """Snapshot risks, overridden by the latest full run of a drive if there is one."""
    snap = state.snapshot
    if snap is None:
        drives = state.engine.res.store.df("SELECT * FROM drives") \
            if state.engine.res.store.ready else pd.DataFrame()
        if drives.empty:
            return pd.DataFrame()
        snap = drives.assign(failed=drives["failure_date"].notna(), action=None, split=None)
        snap[f"p_{state.cfg.max_horizon}"] = float("nan")
        snap["last_date"] = pd.to_datetime(snap["last_date"]).dt.date
    snap = snap.copy()
    runs = state.engine.runs.latest_by_serial() if state.engine.runs else {}
    if runs:
        mask = snap["serial_number"].isin(runs)
        snap.loc[mask, "action"] = snap.loc[mask, "serial_number"].map(
            lambda s: runs[s]["action"])
    return snap


@router.get("/drives", response_model=DriveListResponse)
def list_drives(
    page: int = Query(1, ge=1),
    page_size: int | None = Query(None, ge=1, le=500),
    model: str | None = None,
    min_risk: float | None = Query(None, ge=0, le=1),
    action: str | None = None,
    search: str | None = None,
    failed: bool | None = None,
    sort: Literal["risk", "serial", "last_date", "model"] = "risk",
    order: Literal["asc", "desc"] = "desc",
    state: AppState = Depends(get_state),
) -> DriveListResponse:
    h = state.cfg.max_horizon
    size = page_size or state.cfg.api.page_size
    df = _fleet(state)
    if df.empty:
        return DriveListResponse(items=[], total=0, page=page, page_size=size, models=[],
                                 risk_horizon=h)
    models = sorted(df["model"].astype(str).unique().tolist())
    risk_col = f"p_{h}"
    if model:
        df = df[df["model"] == model]
    if min_risk is not None:
        df = df[df[risk_col] >= min_risk]
    if action:
        df = df[df["action"] == action]
    if failed is not None:
        df = df[df["failed"] == failed]
    if search:
        df = df[df["serial_number"].str.contains(search, case=False, regex=False)]
    key = {"risk": risk_col, "serial": "serial_number", "last_date": "last_date",
           "model": "model"}[sort]
    df = df.sort_values(key, ascending=order == "asc", na_position="last")
    total = len(df)
    chunk = df.iloc[(page - 1) * size : page * size]
    items = [DriveSummary(
        serial_number=r.serial_number, model=str(r.model),
        capacity_bytes=None if pd.isna(r.capacity_bytes) else int(r.capacity_bytes),
        last_date=r.last_date, failed=bool(r.failed),
        latest_risk=None if pd.isna(getattr(r, risk_col)) else float(getattr(r, risk_col)),
        action=r.action if isinstance(r.action, str) else None,
        split=r.split if isinstance(getattr(r, "split", None), str) else None,
    ) for r in chunk.itertuples()]
    return DriveListResponse(items=items, total=total, page=page, page_size=size, models=models,
                             risk_horizon=h)


@router.get("/drives/stats", response_model=FleetStats)
def fleet_stats(state: AppState = Depends(get_state)) -> FleetStats:
    """KPI counts and a log-spaced histogram of the latest fused risk."""
    h = state.cfg.max_horizon
    df = _fleet(state)
    if df.empty:
        return FleetStats(total=0, failed=0, by_action={}, by_model={}, risk_horizon=h,
                          risk_histogram=[], high_risk_count=0, escalated_count=0,
                          snapshot_available=False)
    risk = df[f"p_{h}"].dropna().clip(1e-6, 1.0)
    edges = np.logspace(-6, 0, 13)
    counts, _ = np.histogram(risk, bins=edges)
    actions = df["action"].dropna().value_counts().to_dict()
    queue = state.engine.runs.escalations() if state.engine.runs else []
    return FleetStats(
        total=len(df), failed=int(df["failed"].sum()),
        by_action={str(k): int(v) for k, v in actions.items()},
        by_model={str(k): int(v) for k, v in df["model"].value_counts().items()},
        risk_horizon=h,
        risk_histogram=[HistogramBin(start=float(a), end=float(b), count=int(c))
                        for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)],
        high_risk_count=int(df["action"].isin(["replace_now", "schedule_replacement"]).sum()),
        escalated_count=int((df["action"] == "escalate_to_human").sum()) + len(queue),
        snapshot_available=state.snapshot is not None,
    )


@router.get("/drives/{serial}", response_model=DriveHistory)
def drive_history(serial: str, state: AppState = Depends(get_state)) -> DriveHistory:
    store = state.engine.res.store
    if not store.ready:
        raise HTTPException(status_code=503, detail="data not prepared")
    info = store.drive_info(serial)
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown drive {serial}")
    families = {c: fam for fam, cols in agent_columns(state.cfg).items() for c in cols}
    hist = store.drive_history(serial)
    series = [SmartSeries(column=c, family=fam,
                          values=[None if pd.isna(v) else float(v) for v in hist[c]])
              for c, fam in families.items() if c in hist and hist[c].notna().any()]
    fail = info.get("failure_date")
    return DriveHistory(
        serial_number=serial, model=str(info["model"]),
        capacity_bytes=None if pd.isna(info["capacity_bytes"]) else int(info["capacity_bytes"]),
        dates=[pd.Timestamp(d).date() for d in hist["date"]], series=series,
        failure_date=None if pd.isna(fail) else pd.Timestamp(fail).date())
