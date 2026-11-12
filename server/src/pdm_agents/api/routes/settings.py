"""GET/PUT /settings and POST /settings/test (LLM connection test)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from pdm_agents.api.deps import AppState, get_state
from pdm_agents.llm.health import check_health
from pdm_agents.schemas import LLMHealth, RuntimeSettingsUpdate, RuntimeSettingsView

router = APIRouter(tags=["settings"])


@router.get("/settings", response_model=RuntimeSettingsView)
def get_settings(state: AppState = Depends(get_state)) -> RuntimeSettingsView:
    return state.settings_view()


@router.put("/settings", response_model=RuntimeSettingsView)
def put_settings(update: RuntimeSettingsUpdate,
                 state: AppState = Depends(get_state)) -> RuntimeSettingsView:
    try:
        state.update_settings(update)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return state.settings_view()


@router.post("/settings/test", response_model=LLMHealth)
def test_connection(update: RuntimeSettingsUpdate | None = None,
                    state: AppState = Depends(get_state)) -> LLMHealth:
    """Test the saved LLM settings, or unsaved values sent in the body."""
    s = state.settings.model_copy(update=update.model_dump(exclude_none=True)) \
        if update else state.settings
    return check_health(state.llm_config(s), detect_guided=True)
