"""FastAPI app factory: CORS, lifespan (loads models/indexes once), routers, static reports.

Run with ``python -m scripts.serve`` (or ``uvicorn --factory pdm_agents.api.main:create_app``).
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pdm_agents import __version__
from pdm_agents.api.deps import AppState
from pdm_agents.api.routes import ROUTERS
from pdm_agents.config import AppConfig, ensure_dirs, get_config
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)


def create_app(cfg: AppConfig | None = None, state: AppState | None = None) -> FastAPI:
    cfg = cfg or get_config()
    ensure_dirs(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pdm = state or AppState(cfg)
        # probe the LLM (incl. guided-JSON support) without delaying startup
        threading.Thread(target=lambda: app.state.pdm.llm_health(refresh=True),
                         daemon=True).start()
        log.info("API ready (env=%s)", cfg.resolved_env)
        yield

    app = FastAPI(title="Drive failure agents", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cfg.api.cors_origins,
                       allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
    for router in ROUTERS:
        app.include_router(router)
    app.mount("/reports", StaticFiles(directory=str(cfg.paths.reports)), name="reports")
    return app
