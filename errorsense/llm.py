"""LLM infrastructure — LLMConfig and LLMClient for LLM API calls."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from errorsense.models import SenseResult
from errorsense.signal import Signal
from errorsense.skill import Skill

logger = logging.getLogger("errorsense.llm")

__all__ = ["LLMConfig", "LLMClient"]

DEFAULT_BASE_URL = "https://relay.opengpu.network/v2/openai/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_PROMPT_FORMAT = (
    "{instructions}\n\n"
    "Classify the following error signal into exactly one of these labels: {labels}\n\n"
    "Signal data:\n{signal}\n\n"
    "Respond with ONLY a valid JSON object. No explanation, no markdown, no code fences.\n"
    'Format: {{"label": "<one of the labels above>", "confidence": <0.0-1.0>, "reason": "<short explanation>"}}'
)


@dataclass(frozen=True)
class LLMConfig:
    """Connection config for LLM API.

    Resolution order: explicit arg > env var > built-in default.

    Env vars: ERRORSENSE_LLM_API_KEY, ERRORSENSE_MODEL, ERRORSENSE_LLM_URL
    """

    # Defaults are empty so __post_init__ can detect "not set" and check env vars.
    # Actual defaults (DEFAULT_MODEL, DEFAULT_BASE_URL) are resolved in __post_init__.
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    timeout: float = 10.0
    max_signal_size: int = 500

    def __post_init__(self) -> None:
        if not self.api_key:
            object.__setattr__(self, "api_key", os.environ.get("ERRORSENSE_LLM_API_KEY", ""))
        if not self.model:
            object.__setattr__(self, "model", os.environ.get("ERRORSENSE_MODEL", DEFAULT_MODEL))
        if not self.base_url:
            object.__setattr__(self, "base_url", os.environ.get("ERRORSENSE_LLM_URL", DEFAULT_BASE_URL))


def _build_prompt(signal: Signal, skill: Skill, labels: list[str], config: LLMConfig) -> str:
    signal_text = json.dumps(signal.to_dict(), default=str)
    if len(signal_text) > config.max_signal_size:
        signal_text = signal_text[: config.max_signal_size] + "...(truncated)"

    template = skill.prompt_format or DEFAULT_PROMPT_FORMAT
    return template.format(
        instructions=skill.instructions,
        labels=", ".join(labels) if labels else "unknown",
        signal=signal_text,
    )


def _build_request_body(skill: Skill, prompt: str, config: LLMConfig) -> dict:
    return {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": skill.temperature,
    }


def _build_headers(config: LLMConfig) -> dict:
    # Empty api_key sends "Bearer " — relay accepts this for guest tier.
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }


def _extract_json(raw: str) -> dict | None:
    """Best-effort JSON extraction from LLM output.

    Handles thinking blocks (<think>...</think>), code fences, and
    JSON embedded in prose.
    """
    import re

    text = raw.strip()

    # Strip paired XML-like blocks (<think>, <reasoning>, <scratchpad>, etc.)
    text = re.sub(r"<(\w+)>.*?</\1>", "", text, flags=re.DOTALL).strip()

    # Strip code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()

    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first {...} block with brace matching
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _parse_response(
    data: dict,
    labels: list[str],
    skill_name: str,
    include_reason: bool = False,
) -> SenseResult | None:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        logger.warning("Skill %r: no content in LLM response: %s", skill_name, data)
        return None

    parsed = _extract_json(content)
    if parsed is None:
        logger.warning(
            "Skill %r: could not extract JSON from LLM response: %.300s",
            skill_name,
            content,
        )
        return None

    try:
        label = parsed.get("label", "")
        confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.7))))
        reason = parsed.get("reason") if include_reason else None
    except (ValueError, TypeError) as e:
        logger.warning(
            "Skill %r: invalid field values in LLM response: %s", skill_name, e
        )
        return None

    if labels and label not in labels:
        logger.warning(
            "Skill %r: LLM returned unknown label %r", skill_name, label
        )
        return None

    return SenseResult(
        label=label,
        confidence=confidence,
        skill_name=skill_name,
        reason=reason,
    )


class LLMClient:
    """HTTP client for LLM classification calls. Supports both sync and async."""

    def __init__(self, config: LLMConfig) -> None:
        if httpx is None:
            raise ImportError(
                "LLM skills require httpx. Install with: pip install errorsense[llm]"
            )

        self._config = config
        self._sync_client: Any = None
        self._async_client: Any = None
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def _get_sync_client(self) -> Any:
        with self._sync_lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(timeout=self._config.timeout)
            return self._sync_client

    async def _get_async_client(self) -> Any:
        async with self._async_lock:
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(timeout=self._config.timeout)
            return self._async_client

    def classify_sync(
        self,
        signal: Signal,
        skill: Skill,
        labels: list[str],
        include_reason: bool = False,
    ) -> SenseResult | None:
        prompt = _build_prompt(signal, skill, labels, self._config)
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"

        try:
            client = self._get_sync_client()
            resp = client.post(
                url,
                headers=_build_headers(self._config),
                json=_build_request_body(skill, prompt, self._config),
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("LLM call failed for skill %r: %s", skill.name, e)
            return None

        return _parse_response(data, labels, skill.name, include_reason)

    async def classify_async(
        self,
        signal: Signal,
        skill: Skill,
        labels: list[str],
        include_reason: bool = False,
    ) -> SenseResult | None:
        prompt = _build_prompt(signal, skill, labels, self._config)
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"

        try:
            client = await self._get_async_client()
            resp = await client.post(
                url,
                headers=_build_headers(self._config),
                json=_build_request_body(skill, prompt, self._config),
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("LLM call failed for skill %r: %s", skill.name, e)
            return None

        return _parse_response(data, labels, skill.name, include_reason)

    def close_sync(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def close_async(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    async def close(self) -> None:
        self.close_sync()
        await self.close_async()
