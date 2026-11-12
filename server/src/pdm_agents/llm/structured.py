"""Structured (JSON) output from a small model, with validation, one repair and fallback.

A 4B model is unreliable at native tool calling, so the default path is:
prompt with the JSON schema -> extract the JSON object -> ``PydanticOutputParser``
-> on failure one repair prompt -> otherwise report failure (caller falls back to
a deterministic template). When the backend enforces JSON schemas (detected at
startup, see ``health.detect_guided_json``), ``with_structured_output`` is tried
first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Generic, Literal, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from pdm_agents.llm.client import LLMClient, LLMUnavailableError, strip_think
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
USER_MARKER = "=== USER ==="
StructuredStatus = Literal["ok", "repaired", "failed", "unavailable"]


@dataclass
class StructuredResult(Generic[T]):
    value: T | None
    status: StructuredStatus
    raw: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


@cache
def load_prompt(name: str) -> ChatPromptTemplate:
    """Load ``llm/prompts/<name>.md``: system part, ``=== USER ===`` line, user part."""
    text = resources.files("pdm_agents.llm").joinpath("prompts", f"{name}.md").read_text("utf-8")
    if USER_MARKER not in text:
        raise ValueError(f"prompt {name!r} lacks the {USER_MARKER!r} separator")
    system, user = text.split(USER_MARKER, 1)
    return ChatPromptTemplate.from_messages([("system", system.strip()), ("human", user.strip())])


def prompt_exists(name: str) -> bool:
    return resources.files("pdm_agents.llm").joinpath("prompts", f"{name}.md").is_file()


def format_instructions(schema: type[BaseModel]) -> str:
    """Instructions embedding the compact JSON schema of ``schema``."""
    schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return (
        "Return ONLY one JSON object, with no prose and no markdown, that validates against "
        f"this JSON schema:\n{schema_json}"
    )


def extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block in ``text`` (string-aware), if any."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_structured(text: str, schema: type[T]) -> T:
    """Parse model text into ``schema``; raises ``ValueError`` on failure."""
    block = extract_json_object(strip_think(text))
    if block is None:
        raise ValueError("no JSON object found in the answer")
    try:
        return PydanticOutputParser(pydantic_object=schema).parse(block)
    except Exception as exc:  # OutputParserException wraps json/validation errors
        raise ValueError(str(exc).split("\nFor troubleshooting")[0]) from exc


def _prompt_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


def structured_call(
    llm: LLMClient | None,
    prompt_name: str,
    variables: dict[str, Any],
    schema: type[T],
) -> StructuredResult[T]:
    """Run prompt ``prompt_name`` and return a validated ``schema`` instance or a failure."""
    if llm is None or not llm.available:
        return StructuredResult(None, "unavailable", error="LLM unavailable")
    instructions = format_instructions(schema)
    messages = load_prompt(prompt_name).format_messages(
        **variables, format_instructions=instructions
    )
    raw = ""
    try:
        if llm.guided_json:
            guided = _try_guided(llm, messages, schema)
            if guided is not None:
                return guided
        raw = llm.invoke(messages)
        try:
            return StructuredResult(parse_structured(raw, schema), "ok", raw=raw)
        except ValueError as first_error:
            repair = load_prompt("repair").format_messages(
                error=str(first_error)[:500],
                original_prompt=_prompt_text(messages),
                previous_answer=raw[:2000],
                format_instructions=instructions,
            )
            raw = llm.invoke(repair)
            try:
                return StructuredResult(parse_structured(raw, schema), "repaired", raw=raw)
            except ValueError as second_error:
                log.info("structured output failed after repair (%s): %s", prompt_name,
                         second_error)
                return StructuredResult(None, "failed", raw=raw, error=str(second_error))
    except LLMUnavailableError as exc:
        return StructuredResult(None, "unavailable", raw=raw, error=str(exc))
    except Exception as exc:  # server error, bad request, ...
        log.warning("LLM call failed (%s): %s", prompt_name, exc)
        return StructuredResult(None, "failed", raw=raw, error=f"{type(exc).__name__}: {exc}")


def _try_guided(
    llm: LLMClient, messages: list[BaseMessage], schema: type[T]
) -> StructuredResult[T] | None:
    """Backend-enforced JSON schema via ``with_structured_output``; ``None`` to fall back."""
    try:
        runnable = llm.chat_model.with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        llm.calls += 1
        out = runnable.invoke(messages)
    except Exception as exc:
        log.info("guided JSON call failed, using prompt path: %s", exc)
        return None
    parsed = out.get("parsed") if isinstance(out, dict) else None
    if isinstance(parsed, schema):
        raw_msg = out.get("raw")
        return StructuredResult(parsed, "ok", raw=str(getattr(raw_msg, "content", "")))
    return None
