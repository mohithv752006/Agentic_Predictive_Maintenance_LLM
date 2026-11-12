"""Risk tool: run an agent's family model and return calibrated risk per horizon."""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from pdm_agents.models.bundle import ModelBundle


class FamilyRiskArgs(BaseModel):
    agent: str = Field(description="Numeric agent whose family model to run")
    model: str = Field(description="Drive model name")
    features: dict[str, float | None] = Field(description="Window features (f__* columns)")


def make_family_risk_tool(bundle: ModelBundle, calibrated: bool = True) -> BaseTool:
    """Bind the trained bundle into a LangChain tool."""

    @tool("family_risk", args_schema=FamilyRiskArgs)
    def family_risk(agent: str, model: str, features: dict[str, float | None]) -> dict:
        """Calibrated probability of failure within each horizon from one family model."""
        if agent not in bundle.family_models:
            raise KeyError(f"no trained family model for agent {agent!r}")
        row = pd.DataFrame([{k: (float("nan") if v is None else v) for k, v in features.items()}])
        missing = [f for f in bundle.family_models[agent][bundle.horizons[0]].features
                   if f not in row]
        for f in missing:
            row[f] = float("nan")
        risk = bundle.family_risk(agent, row, calibrated=calibrated)
        prior = bundle.prior([model])
        return {
            "risk": {h: float(v[0]) for h, v in risk.items()},
            "base_rate": {h: float(v[0]) for h, v in prior.items()},
            "calibrated": calibrated,
            "missing_features": len(missing),
        }

    return family_risk


class GlobalRiskArgs(BaseModel):
    model: str = Field(description="Drive model name")
    features: dict[str, float | None] = Field(description="Window features (f__* columns)")


def make_global_risk_tool(bundle: ModelBundle, calibrated: bool = True) -> BaseTool:
    """Risk from the all-features model (used by the single-agent baseline)."""

    @tool("global_risk", args_schema=GlobalRiskArgs)
    def global_risk(model: str, features: dict[str, float | None]) -> dict:
        """Calibrated probability of failure within each horizon from the all-features model."""
        row = pd.DataFrame([{k: (float("nan") if v is None else v) for k, v in features.items()}])
        for f in bundle.global_models[bundle.horizons[0]].features:
            if f not in row:
                row[f] = float("nan")
        risk = bundle.global_risk(row, calibrated=calibrated)
        prior = bundle.prior([model])
        return {"risk": {h: float(v[0]) for h, v in risk.items()},
                "base_rate": {h: float(v[0]) for h, v in prior.items()}}

    return global_risk
