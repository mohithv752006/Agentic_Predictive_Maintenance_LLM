"""GET /health: server status and LLM reachability."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from pdm_agents import __version__
from pdm_agents.api.deps import AppState, get_state
from pdm_agents.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(refresh: bool = False, state: AppState = Depends(get_state)) -> HealthResponse:
    res = state.engine.res
    llm = state.llm_health(refresh=refresh)
    models_ready = res.bundle is not None
    ready = res.store.ready and models_ready
    return HealthResponse(
        status="ok" if ready and llm.reachable else "degraded",
        version=__version__,
        env=state.cfg.resolved_env,
        data_ready=res.store.ready,
        models_ready=models_ready,
        indexes_ready=res.cases is not None and res.knowledge is not None,
        llm=llm,
    )
