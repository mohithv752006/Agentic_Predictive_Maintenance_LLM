"""LLM client: LangChain ``ChatOpenAI`` pointed at the Colab endpoint.

``LLMClient`` wraps any LangChain chat model (the real endpoint or ``FakeLLM``)
and adds what the graph needs:

* ``<think>`` stripping on every response (client side, in addition to the server);
* a per-run call counter (``session()`` returns a view with fresh counters);
* a circuit breaker: after a connection-level failure the session is marked
  unavailable and later calls fail fast instead of waiting for timeouts.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from pdm_agents.config import LLMConfig
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMUnavailableError(RuntimeError):
    """The LLM could not be reached (or was marked unavailable for this run)."""


def strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` blocks and anything before an orphan ``</think>``."""
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def _is_connection_error(exc: BaseException) -> bool:
    names = {type(e).__name__ for e in _chain(exc)}
    markers = {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout",
               "ReadTimeout", "ConnectionError", "FakeLLMError", "TimeoutError",
               "AuthenticationError", "NotFoundError", "PermissionDeniedError"}
    return bool(names & markers)


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def build_chat_model(cfg: LLMConfig) -> BaseChatModel:
    """Create the ``ChatOpenAI`` model for the configured OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "not-set",
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
        # Qwen3.5 thinks by default; vLLM/SGLang/our proxy read this from the request body.
        extra_body={"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}},
    )


@dataclass
class LLMClient:
    """Thin wrapper over a chat model with counters, think-stripping and a circuit breaker."""

    chat_model: BaseChatModel
    model_name: str = "unknown"
    guided_json: bool = False
    calls: int = 0
    failures: int = 0
    available: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_config(cls, cfg: LLMConfig) -> LLMClient | None:
        """Real client, or ``None`` when no endpoint is configured."""
        if not cfg.base_url:
            return None
        return cls(chat_model=build_chat_model(cfg), model_name=cfg.model)

    def session(self) -> LLMClient:
        """A per-run view sharing the chat model but with fresh counters and breaker."""
        return LLMClient(chat_model=self.chat_model, model_name=self.model_name,
                         guided_json=self.guided_json)

    def invoke(self, messages: list[BaseMessage]) -> str:
        """Call the model and return cleaned text; raises ``LLMUnavailableError`` if down."""
        if not self.available:
            raise LLMUnavailableError("LLM marked unavailable for this run")
        with self._lock:
            self.calls += 1
        try:
            result = self.chat_model.invoke(messages)
        except Exception as exc:
            with self._lock:
                self.failures += 1
            if _is_connection_error(exc):
                self.available = False
                log.warning("LLM unreachable, switching run to degraded mode: %s", exc)
                raise LLMUnavailableError(str(exc)) from exc
            raise
        content = result.content
        if isinstance(content, list):  # content blocks
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        return strip_think(str(content))


LLMMode = Literal["live", "fake", "offline"]


def create_llm(cfg: LLMConfig, mode: LLMMode = "live") -> LLMClient | None:
    """Build the client for a mode: ``live`` endpoint, deterministic ``fake``, or ``offline``.

    ``live`` returns ``None`` when no endpoint is configured, which callers treat as offline.
    """
    if mode == "offline":
        return None
    if mode == "fake":
        from pdm_agents.llm.fake import FakeLLM

        return LLMClient(chat_model=FakeLLM(), model_name="fake")
    return LLMClient.from_config(cfg)
