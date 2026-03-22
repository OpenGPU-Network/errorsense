"""LLM infrastructure — LLMConfig and LLMClient for LLM API calls."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from errorsense.models import SenseResult
from errorsense.signal import Signal
from errorsense.skill import Skill

logger = logging.getLogger("errorsense.llm")

__all__ = ["LLMConfig", "LLMClient"]

DEFAULT_BASE_URL = "https://relay.opengpu.network/v1"
DEFAULT_MODEL = "gpt-oss:120b"
DEFAULT_PROMPT_TEMPLATE = (
    "{instructions}\n\n"
    "Classify the following error signal into exactly one of these categories: {categories}\n\n"
    "Signal data:\n{signal}\n\n"
    'Reply ONLY with JSON: {{"label": "...", "confidence": 0.0, "reason": "..."}}'
)


@dataclass(frozen=True)
class LLMConfig:
    """Connection config for LLM API."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 10.0
    max_signal_size: int = 500


def _build_prompt(signal: Signal, skill: Skill, categories: list[str], config: LLMConfig) -> str:
    signal_text = json.dumps(signal.to_dict(), default=str)
    if len(signal_text) > config.max_signal_size:
        signal_text = signal_text[: config.max_signal_size] + "..."

    template = skill.prompt_template or DEFAULT_PROMPT_TEMPLATE
    return template.format(
        instructions=skill.instructions,
        categories=", ".join(categories) if categories else "unknown",
        signal=signal_text,
    )


def _build_request_body(skill: Skill, prompt: str, config: LLMConfig) -> dict:
    return {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": skill.temperature,
    }


def _build_headers(config: LLMConfig) -> dict:
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }


def _parse_response(
    data: dict,
    categories: list[str],
    skill_name: str,
    include_reason: bool = False,
) -> SenseResult | None:
    try:
        content = data["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        parsed = json.loads(content.strip())

        label = parsed.get("label", "") or parsed.get("category", "")
        confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.7))))
        reason = parsed.get("reason") if include_reason else None

        if categories and label not in categories:
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
    except (KeyError, json.JSONDecodeError, IndexError, ValueError) as e:
        logger.warning("Failed to parse LLM response for skill %r: %s", skill_name, e)
        return None


class LLMClient:
    """HTTP client for LLM classification calls. Supports both sync and async."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(
                "LLM skills require httpx. Install with: pip install errorsense[llm]"
            ) from None

        self._config = config
        self._sync_client: Any = None
        self._async_client: Any = None
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def _get_sync_client(self) -> Any:
        import httpx

        with self._sync_lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(timeout=self._config.timeout)
            return self._sync_client

    async def _get_async_client(self) -> Any:
        import httpx

        async with self._async_lock:
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(timeout=self._config.timeout)
            return self._async_client

    def classify_sync(
        self,
        signal: Signal,
        skill: Skill,
        categories: list[str],
        include_reason: bool = False,
    ) -> SenseResult | None:
        config = skill.llm if skill.llm is not None else self._config
        prompt = _build_prompt(signal, skill, categories, config)
        url = f"{config.base_url.rstrip('/')}/chat/completions"

        try:
            client = self._get_sync_client()
            resp = client.post(
                url,
                headers=_build_headers(config),
                json=_build_request_body(skill, prompt, config),
            )
            resp.raise_for_status()
            data = resp.json()
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning("LLM call failed for skill %r: %s", skill.name, e)
            return None

        return _parse_response(data, categories, skill.name, include_reason)

    async def classify_async(
        self,
        signal: Signal,
        skill: Skill,
        categories: list[str],
        include_reason: bool = False,
    ) -> SenseResult | None:
        config = skill.llm if skill.llm is not None else self._config
        prompt = _build_prompt(signal, skill, categories, config)
        url = f"{config.base_url.rstrip('/')}/chat/completions"

        try:
            client = await self._get_async_client()
            resp = await client.post(
                url,
                headers=_build_headers(config),
                json=_build_request_body(skill, prompt, config),
            )
            resp.raise_for_status()
            data = resp.json()
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning("LLM call failed for skill %r: %s", skill.name, e)
            return None

        return _parse_response(data, categories, skill.name, include_reason)

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
