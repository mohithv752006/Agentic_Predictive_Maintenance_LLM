"""Tool registry: name -> LangChain tool, built from the loaded resources.

Add a tool = one function (or factory) in a ``tools/*.py`` module + one entry here.
Tools whose resources are missing (e.g. no case memory yet) are simply absent;
agents that need them report a failure instead of crashing.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from pdm_agents.config import AppConfig
from pdm_agents.models.bundle import ModelBundle
from pdm_agents.rag.case_memory import CaseMemory
from pdm_agents.rag.knowledge_index import KnowledgeIndex
from pdm_agents.tools.retrieval_tools import (
    make_case_risk_tool,
    make_case_tool,
    make_knowledge_tool,
)
from pdm_agents.tools.risk_tools import make_family_risk_tool, make_global_risk_tool
from pdm_agents.tools.trend_tools import trend_analysis


def build_tools(
    cfg: AppConfig,
    bundle: ModelBundle | None,
    cases: CaseMemory | None,
    knowledge: KnowledgeIndex | None,
) -> dict[str, BaseTool]:
    tools: dict[str, BaseTool] = {"trend_analysis": trend_analysis}
    if bundle is not None:
        tools["family_risk"] = make_family_risk_tool(bundle, cfg.features.use_calibration)
        tools["global_risk"] = make_global_risk_tool(bundle, cfg.features.use_calibration)
    if cases is not None:
        tools["case_retrieval"] = make_case_tool(cases, cfg)
        if bundle is not None and bundle.has_case_fusion:
            tools["case_risk"] = make_case_risk_tool(cases, bundle, cfg)
    if knowledge is not None:
        tools["knowledge_retrieval"] = make_knowledge_tool(knowledge)
    return tools
