"""Retrieval tools: similar training drives (case memory) and knowledge snippets."""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from pdm_agents.config import AppConfig
from pdm_agents.models.bundle import ModelBundle, case_risk
from pdm_agents.rag.case_memory import CaseMemory
from pdm_agents.rag.knowledge_index import KnowledgeIndex


class CaseArgs(BaseModel):
    serial_number: str = Field(description="Query drive (excluded from results)")
    model: str = Field(description="Drive model name")
    features: dict[str, float | None] = Field(description="Window features (f__* columns)")
    k: int = Field(default=5, ge=1, le=50)


class KnowledgeArgs(BaseModel):
    query: str = Field(description="What to look up, in plain words")
    k: int = Field(default=3, ge=1, le=20)


def make_case_tool(memory: CaseMemory, cfg: AppConfig) -> BaseTool:
    @tool("case_retrieval", args_schema=CaseArgs)
    def case_retrieval(serial_number: str, model: str, features: dict[str, float | None],
                       k: int = 5) -> list[dict]:
        """Most similar training drives (same model when possible) and their outcomes."""
        row = pd.DataFrame([{**{f: (float("nan") if v is None else v)
                                for f, v in features.items()}, "model": model}])
        for col in memory.normalizer.columns:
            if col not in row:
                row[col] = float("nan")
        hits = memory.query(row, k=k, same_model_only=cfg.rag.same_model_only,
                            exclude_serial=serial_number)
        return [h.model_dump() for h in hits]

    return case_retrieval


def make_case_risk_tool(memory: CaseMemory, bundle: ModelBundle, cfg: AppConfig) -> BaseTool:
    @tool("case_risk", args_schema=CaseArgs)
    def case_risk_tool(serial_number: str, model: str, features: dict[str, float | None],
                       k: int = 5) -> dict:
        """Calibrated failure risk from the outcomes of the most similar training drives."""
        row = pd.DataFrame([{**{f: (float("nan") if v is None else v)
                                for f, v in features.items()}, "model": model}])
        for col in memory.normalizer.columns:
            if col not in row:
                row[col] = float("nan")
        risk = case_risk(cfg, bundle, memory, row)
        return {h: float(v[0]) for h, v in risk.items()}

    return case_risk_tool


def make_knowledge_tool(index: KnowledgeIndex) -> BaseTool:
    @tool("knowledge_retrieval", args_schema=KnowledgeArgs)
    def knowledge_retrieval(query: str, k: int = 3) -> list[dict]:
        """SMART attribute semantics, vendor caveats and policy notes relevant to a query."""
        return [s.model_dump() for s in index.query(query, k)]

    return knowledge_retrieval
