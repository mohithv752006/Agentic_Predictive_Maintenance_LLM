"""Shared CLI bootstrap: load config (+ optional experiment overrides) and logging."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from pdm_agents.config import AppConfig, ensure_dirs, load_config
from pdm_agents.logging_setup import setup_logging


def _utf8_streams() -> None:
    """Windows consoles default to cp1252; project paths may contain non-ASCII characters."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def bootstrap(
    overrides: dict[str, Any] | None = None, config_file: Path | None = None
) -> AppConfig:
    """Load config (optionally merged with a YAML file), set up logging, create dirs."""
    _utf8_streams()
    merged: dict[str, Any] = {}
    if config_file is not None:
        with config_file.open("r", encoding="utf-8") as fh:
            merged = yaml.safe_load(fh) or {}
    if overrides:
        from pdm_agents.config import deep_merge

        merged = deep_merge(merged, overrides)
    cfg = load_config(merged or None)
    setup_logging(cfg.logging.level, cfg.logging.json_format)
    ensure_dirs(cfg)
    return cfg
