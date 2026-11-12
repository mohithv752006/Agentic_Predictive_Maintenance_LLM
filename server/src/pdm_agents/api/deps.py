"""Application state shared by the routes: engine, runtime settings, fleet snapshot."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import Request

from pdm_agents.config import AppConfig, LLMConfig
from pdm_agents.graph.builder import PredictionEngine
from pdm_agents.llm.client import LLMClient
from pdm_agents.llm.health import check_health
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.scoring import snapshot_path
from pdm_agents.schemas import (
    LLMHealth,
    RuntimeSettings,
    RuntimeSettingsUpdate,
    RuntimeSettingsView,
)

log = get_logger(__name__)
HEALTH_TTL_S = 30.0


class AppState:
    """One per process. Settings changes re-point the engine's LLM client immediately."""

    def __init__(self, cfg: AppConfig, engine: PredictionEngine | None = None) -> None:
        self.cfg = cfg
        self.settings_path: Path = cfg.paths.runtime / "settings.json"
        self.settings = self._load_settings()
        self._lock = threading.Lock()
        self._health: tuple[float, LLMHealth] | None = None
        self.engine = engine or PredictionEngine(cfg, llm=self._client_for(self.llm_config()))
        self._apply_settings(rebuild_llm=engine is None)
        self.snapshot = self._load_snapshot()

    # --------------------------------------------------------------- settings

    def _load_settings(self) -> RuntimeSettings:
        defaults = RuntimeSettings(
            llm_base_url=self.cfg.llm.base_url, llm_api_key=self.cfg.llm.api_key,
            llm_model=self.cfg.llm.model, default_cost_ratio=self.cfg.decision.default_cost_ratio,
            enabled_agents=self.cfg.enabled_agents())
        if self.settings_path.exists():
            try:
                saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
                merged = {**defaults.model_dump(), **{k: v for k, v in saved.items() if v}}
                merged["enabled_agents"] = [a for a in merged["enabled_agents"]
                                            if a in self.cfg.enabled_agents()]
                return RuntimeSettings(**merged)
            except (ValueError, OSError) as exc:
                log.warning("ignoring unreadable settings file: %s", exc)
        return defaults

    def llm_config(self, s: RuntimeSettings | None = None) -> LLMConfig:
        s = s or self.settings
        return self.cfg.llm.model_copy(update={"base_url": s.llm_base_url,
                                               "api_key": s.llm_api_key, "model": s.llm_model})

    @staticmethod
    def _client_for(cfg: LLMConfig) -> LLMClient | None:
        return LLMClient.from_config(cfg)

    def _apply_settings(self, rebuild_llm: bool = True) -> None:
        if rebuild_llm:
            self.engine.set_llm(self._client_for(self.llm_config()))
        self.engine.enabled_override = self.settings.enabled_agents or None
        self.engine.default_cost_ratio = self.settings.default_cost_ratio
        self._health = None

    def update_settings(self, update: RuntimeSettingsUpdate) -> RuntimeSettings:
        data = update.model_dump(exclude_none=True)
        if "enabled_agents" in data:
            unknown = set(data["enabled_agents"]) - set(self.cfg.enabled_agents())
            if unknown:
                raise ValueError(f"unknown or config-disabled agents: {sorted(unknown)}")
        llm_changed = any(k.startswith("llm_") for k in data)
        with self._lock:
            self.settings = self.settings.model_copy(update=data)
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(self.settings.model_dump_json(indent=2),
                                          encoding="utf-8")
            self._apply_settings(rebuild_llm=llm_changed)
        return self.settings

    def settings_view(self) -> RuntimeSettingsView:
        key = self.settings.llm_api_key
        return RuntimeSettingsView(
            llm_base_url=self.settings.llm_base_url, llm_api_key_set=bool(key),
            llm_api_key_hint=f"...{key[-4:]}" if len(key) >= 8 else ("set" if key else ""),
            llm_model=self.settings.llm_model,
            default_cost_ratio=self.settings.default_cost_ratio,
            enabled_agents=self.settings.enabled_agents,
            available_agents=self.cfg.enabled_agents(),
            cost_ratios=self.cfg.decision.cost_ratios)

    # ----------------------------------------------------------------- health

    def llm_health(self, refresh: bool = False) -> LLMHealth:
        now = time.monotonic()
        if not refresh and self._health and now - self._health[0] < HEALTH_TTL_S:
            return self._health[1]
        health = check_health(self.llm_config(), detect_guided=refresh)
        if refresh and self.engine.llm is not None:
            self.engine.llm.guided_json = health.guided_json
        self._health = (now, health)
        return health

    # --------------------------------------------------------------- snapshot

    def _load_snapshot(self) -> pd.DataFrame | None:
        path = snapshot_path(self.cfg)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            df = pd.read_parquet(fh)
        df["last_date"] = pd.to_datetime(df["last_date"]).dt.date
        return df


def get_state(request: Request) -> AppState:
    return request.app.state.pdm
