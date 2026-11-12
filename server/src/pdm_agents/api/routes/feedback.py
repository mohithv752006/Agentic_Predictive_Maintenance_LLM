"""POST /feedback (human decision on an escalated run) and GET /escalations (queue)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from pdm_agents.api.deps import AppState, get_state
from pdm_agents.schemas import EscalationItem, FeedbackRecord, FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackRecord)
def post_feedback(req: FeedbackRequest, state: AppState = Depends(get_state)) -> FeedbackRecord:
    runs = state.engine.runs
    if runs is None or runs.get(req.run_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown run {req.run_id}")
    if req.final_action not in state.cfg.actions:
        raise HTTPException(status_code=422, detail=f"unknown action {req.final_action}")
    record = FeedbackRecord(**req.model_dump())
    runs.add_feedback(record)
    return record


@router.get("/escalations", response_model=list[EscalationItem])
def escalations(state: AppState = Depends(get_state)) -> list[EscalationItem]:
    return state.engine.runs.escalations() if state.engine.runs else []
