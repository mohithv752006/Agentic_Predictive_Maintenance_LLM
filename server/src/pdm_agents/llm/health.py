"""LLM endpoint health: reachability, latency, served model id, guided-JSON support."""

from __future__ import annotations

import json
import time

import httpx

from pdm_agents.config import LLMConfig
from pdm_agents.logging_setup import get_logger
from pdm_agents.schemas import LLMHealth

log = get_logger(__name__)


def _headers(cfg: LLMConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}


def check_health(cfg: LLMConfig, detect_guided: bool = False) -> LLMHealth:
    """Probe ``GET {base_url}/models``; never raises."""
    if not cfg.base_url:
        return LLMHealth(configured=False, reachable=False, error="LLM_BASE_URL not set")
    base = cfg.base_url.rstrip("/")
    t0 = time.perf_counter()
    try:
        resp = httpx.get(f"{base}/models", headers=_headers(cfg), timeout=cfg.health_timeout_s)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return LLMHealth(configured=True, reachable=False, base_url=base, latency_ms=latency,
                             error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        ids = [m.get("id") for m in resp.json().get("data", [])]
        model_id = cfg.model if cfg.model in ids else (ids[0] if ids else None)
        guided = detect_guided_json(cfg) if detect_guided else False
        return LLMHealth(configured=True, reachable=True, base_url=base, model_id=model_id,
                         latency_ms=round(latency, 1), guided_json=guided)
    except (httpx.HTTPError, ValueError) as exc:
        return LLMHealth(configured=True, reachable=False, base_url=base,
                         error=f"{type(exc).__name__}: {exc}")


def detect_guided_json(cfg: LLMConfig) -> bool:
    """True if the backend enforces ``response_format`` JSON schemas (vLLM/SGLang do).

    The prompt deliberately does not ask for JSON: only a backend that constrains
    decoding returns a schema-valid object.
    """
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}
    body = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "Say hello in five words."}],
        "max_tokens": 30,
        "temperature": 0,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "probe", "schema": schema, "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = httpx.post(f"{cfg.base_url.rstrip('/')}/chat/completions", json=body,
                          headers=_headers(cfg), timeout=cfg.timeout_s)
        if resp.status_code != 200:
            return False
        content = resp.json()["choices"][0]["message"].get("content") or ""
        return isinstance(json.loads(content).get("ok"), bool)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        log.info("guided JSON not available: %s", exc)
        return False
