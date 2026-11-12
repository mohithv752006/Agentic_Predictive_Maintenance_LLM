"""Agent registry: instantiates the agents enabled in ``agents.yaml``.

Each entry's ``class`` ("module:ClassName") is imported with importlib, so a new
agent needs only its module and one YAML entry; the graph builder picks it up.
"""

from __future__ import annotations

import importlib

from pdm_agents.agents.base import AgentResources, SpecialistAgent
from pdm_agents.config import AppConfig


def load_agent_class(path: str) -> type[SpecialistAgent]:
    module_name, _, class_name = path.partition(":")
    if not class_name:
        raise ValueError(f"agent class path {path!r} must look like 'module:ClassName'")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, SpecialistAgent)):
        raise TypeError(f"{path} is not a SpecialistAgent subclass")
    return cls


def build_agents(cfg: AppConfig, resources: AgentResources,
                 names: list[str] | None = None) -> dict[str, SpecialistAgent]:
    """Instantiate enabled agents (optionally restricted to ``names``), in YAML order."""
    enabled = cfg.enabled_agents()
    wanted = enabled if names is None else [n for n in enabled if n in set(names)]
    return {n: load_agent_class(cfg.agents[n].class_path)(n, cfg.agents[n], resources)
            for n in wanted}
