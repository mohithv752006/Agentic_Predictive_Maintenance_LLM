"""Deterministic fake chat model for tests and offline demos.

It reads the ``ROLE:`` line from the system prompt (every prompt template in
``llm/prompts`` starts with one) and returns schema-valid JSON built only from
ids and names that appear in the prompt, so outputs are reproducible and
citations are always real. Failure modes let tests exercise repair, think-
stripping and the unreachable-LLM path.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

FakeMode = Literal["ok", "think", "malformed_once", "malformed", "raise"]

ROLE_RE = re.compile(r"^ROLE:\s*(\w+)", re.MULTILINE)
AGENTS_RE = re.compile(r"^AVAILABLE_AGENTS:\s*(.+)$", re.MULTILINE)
FINDING_RE = re.compile(r"\bF-[0-9a-f]{8}\b")
CASE_RE = re.compile(r"\bC-[0-9a-f]{8}\b")
KNOW_RE = re.compile(r"\bK-[0-9a-z_]+-\d+\b")
EVID_RE = re.compile(r"\bE-[0-9a-z_]+-\d+\b")
CONCERN_RE = re.compile(r"^RISK_HINT:\s*(low|medium|high)", re.MULTILINE)


class FakeLLMError(ConnectionError):
    """Raised in ``raise`` mode to mimic an unreachable endpoint."""


class FakeLLM(BaseChatModel):
    """Deterministic, role-aware fake chat model."""

    mode: FakeMode = "ok"
    calls: int = 0
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-pdm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        text = "\n".join(str(m.content) for m in messages)
        self.prompts.append(text)
        if self.mode == "raise":
            raise FakeLLMError("FakeLLM configured as unreachable")
        if self.mode == "malformed" or (self.mode == "malformed_once" and self.calls == 1):
            content = "Sure! Here is the answer: {not valid json"
        else:
            content = json.dumps(self._answer(text))
            if self.mode == "think":
                content = f"<think>reasoning that must be stripped</think>\n{content}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    # ------------------------------------------------------------------ answers

    def _answer(self, text: str) -> dict[str, Any]:
        role_match = ROLE_RE.search(text)
        role = role_match.group(1) if role_match else "generic"
        if role == "planner":
            agents_match = AGENTS_RE.search(text)
            agents = [a.strip() for a in agents_match.group(1).split(",")] if agents_match else []
            return {
                "agents": [a for a in agents if a],
                "rationale": "Run every available specialist; numeric families first.",
            }
        if role == "explainer":
            findings = _unique(FINDING_RE.findall(text))
            cases = _unique(CASE_RE.findall(text))
            knowledge = _unique(KNOW_RE.findall(text))
            cited = " ".join(f"[{i}]" for i in findings[:3] + cases[:2] + knowledge[:2])
            return {
                "text": (
                    "The recommendation follows from the specialist findings, similar past "
                    f"drives and attribute references {cited}."
                ).replace("  ", " "),
                "cited_finding_ids": findings[:3],
                "cited_case_ids": cases[:2],
                "cited_knowledge_ids": knowledge[:2],
            }
        if role == "repair":
            # The repair prompt embeds the original prompt; answer that one instead.
            inner = text.split("ORIGINAL_PROMPT:", 1)[-1]
            return self._answer(inner)
        # specialist interpretation (ROLE: agent_<name>) or anything else
        evidence = _unique(EVID_RE.findall(text))
        concern_match = CONCERN_RE.search(text)
        concern = concern_match.group(1) if concern_match else "low"
        cited = ", ".join(evidence[:3]) or "the tool metrics"
        return {
            "summary": f"Tool evidence ({cited}) indicates {concern} concern for this family.",
            "concern": concern,
            "key_evidence_ids": evidence[:3],
        }


def _unique(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)
