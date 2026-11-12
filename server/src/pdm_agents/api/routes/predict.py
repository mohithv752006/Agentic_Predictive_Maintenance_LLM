"""POST /predict (full run), GET /predict/stream (SSE), GET /runs/{run_id}."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from pdm_agents.api.deps import AppState, get_state
from pdm_agents.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


def _check_ready(state: AppState, req: PredictRequest) -> None:
    res = state.engine.res
    if not res.store.ready or res.bundle is None:
        raise HTTPException(status_code=503, detail="data or models not ready; run the CLI "
                                                    "pipeline (prepare, train, index)")
    if res.store.drive_info(req.serial_number) is None:
        raise HTTPException(status_code=404, detail=f"unknown drive {req.serial_number}")
    bad = [h for h in req.horizons if h not in state.cfg.horizons]
    if bad:
        raise HTTPException(status_code=422, detail=f"unsupported horizons {bad}")


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, state: AppState = Depends(get_state)) -> PredictResponse:
    _check_ready(state, req)
    return await run_in_threadpool(state.engine.predict, req)


@router.get("/predict/stream")
async def predict_stream(
    serial_number: str,
    as_of_date: date | None = None,
    cost_ratio: float | None = Query(None, gt=0),
    offline: bool = False,
    agents: str | None = Query(None, description="Comma-separated agent names"),
    state: AppState = Depends(get_state),
) -> EventSourceResponse:
    """Server-sent events: ``start``, one ``update`` per graph node, then ``result``."""
    req = PredictRequest(serial_number=serial_number, as_of_date=as_of_date,
                         horizons=state.cfg.horizons, cost_ratio=cost_ratio, offline=offline,
                         enabled_agents=[a for a in agents.split(",") if a] if agents else None)
    _check_ready(state, req)
    planned = list(state.engine._agents_for(req))

    async def events():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                for ev in state.engine.stream(req):
                    loop.call_soon_threadsafe(queue.put_nowait, ev)
            except Exception as exc:  # surface to the client instead of hanging
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error",
                                                             "message": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        yield ServerSentEvent(json.dumps({"agents": planned,
                                          "llm_configured": state.engine.llm is not None}),
                              event="start")
        while (ev := await queue.get()) is not None:
            if ev["type"] == "result":
                yield ServerSentEvent(ev["data"].model_dump_json(), event="result")
            elif ev["type"] == "error":
                yield ServerSentEvent(json.dumps({"message": ev["message"]}), event="error")
            else:
                yield ServerSentEvent(json.dumps(ev), event="update")

    return EventSourceResponse(events(), ping=15)


@router.get("/runs/{run_id}", response_model=PredictResponse)
def get_run(run_id: str, state: AppState = Depends(get_state)) -> PredictResponse:
    resp = state.engine.runs.get(run_id) if state.engine.runs else None
    if resp is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return resp
