"""Knowledge retrieval: attribute meanings and vendor caveats for grounded explanations."""

from __future__ import annotations

from pdm_agents.agents.base import SpecialistAgent
from pdm_agents.rag.summarizer import trajectory_summary
from pdm_agents.schemas import DriveContext, EvidenceItem, ToolEvidence


def knowledge_query(ctx: DriveContext, cfg) -> str:
    """Plain-language query from the window's most notable attributes."""
    summary = trajectory_summary({**ctx.features}, cfg, max_items=3)
    return f"What do these hard drive SMART observations mean for failure risk? {summary}"


class KnowledgeRetrievalAgent(SpecialistAgent):
    def run_tools(self, ctx: DriveContext) -> ToolEvidence:
        query = knowledge_query(ctx, self.res.cfg)
        snippets = self.res.tool("knowledge_retrieval").invoke(
            {"query": query, "k": self.res.cfg.rag.knowledge_top_k})
        items = [EvidenceItem(id=s["id"], kind="knowledge", label=s["text"].split(":", 1)[0],
                              value=s.get("score"), detail=s["text"][:400])
                 for s in snippets]
        return ToolEvidence(agent=self.name,
                            metrics={"n_snippets": float(len(snippets)),
                                     "top_score": float(snippets[0]["score"]) if snippets else 0},
                            notes=[f"query: {query[:200]}"], evidence=items)

    def confidence(self, evidence: ToolEvidence, ctx: DriveContext | None) -> float:
        return round(max(0.0, min(1.0, evidence.metrics.get("top_score", 0.0))), 3)
