"""Phase — named stage in the classification pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any

from errorsense.llm import LLMClient, LLMConfig
from errorsense.models import SenseResult
from errorsense.ruleset import Ruleset
from errorsense.signal import Signal
from errorsense.skill import Skill

logger = logging.getLogger("errorsense")

__all__ = ["Phase"]


class Phase:
    """A named stage in the classification pipeline.

    Each phase contains either rulesets (deterministic) or skills (LLM).
    Not both.
    """

    def __init__(
        self,
        name: str,
        rulesets: list[Ruleset] | None = None,
        skills: list[Skill] | None = None,
        llm: LLMConfig | None = None,
    ) -> None:
        if not name:
            raise ValueError("Phase requires a non-empty name")

        has_rulesets = rulesets is not None and len(rulesets) > 0
        has_skills = skills is not None and len(skills) > 0

        if has_rulesets and has_skills:
            raise ValueError(
                f"Phase {name!r}: cannot mix rulesets and skills. "
                "Use rulesets OR (skills + llm), not both."
            )
        if not has_rulesets and not has_skills:
            raise ValueError(
                f"Phase {name!r}: must have at least one ruleset or skill."
            )
        if has_skills and not llm:
            raise ValueError(
                f"Phase {name!r}: skills require llm=LLMConfig(...)."
            )
        if has_rulesets and llm:
            logger.warning(
                "Phase %r: llm config ignored for ruleset phase.", name
            )

        self.name = name
        self.rulesets = rulesets or []
        self.skills = skills or []
        self.llm = llm
        self.is_llm_phase = has_skills
        self._labels: list[str] = []
        self._llm_client: LLMClient | None = None

        if self.is_llm_phase and llm:
            self._llm_client = LLMClient(llm)

    def set_labels(self, labels: list[str]) -> None:
        self._labels = list(labels)

    def classify(self, signal: Signal, explain: bool = False) -> SenseResult | None:
        """Sync classification. Full pipeline — rulesets or LLM."""
        if self.is_llm_phase:
            return self._run_skills_sync(signal, explain)
        return self._run_rulesets(signal)

    async def async_classify(self, signal: Signal, explain: bool = False) -> SenseResult | None:
        """Async classification. Full pipeline — rulesets or LLM."""
        if self.is_llm_phase:
            return await self._run_skills_async(signal, explain)
        return self._run_rulesets(signal)

    def _run_rulesets(self, signal: Signal) -> SenseResult | None:
        for ruleset in self.rulesets:
            try:
                result = ruleset.classify(signal)
            except Exception as e:
                logger.warning(
                    "Phase %r: ruleset %s raised %s: %s",
                    self.name, type(ruleset).__name__, type(e).__name__, e,
                )
                continue
            if result is not None:
                return self._stamp_phase(result, type(ruleset).__name__)
        return None

    def _run_skills_sync(self, signal: Signal, explain: bool) -> SenseResult | None:
        if not self._llm_client:
            return None

        best: SenseResult | None = None
        for skill in self.skills:
            try:
                r = self._run_one_skill_sync(signal, skill, explain)
            except Exception as e:
                logger.warning("Phase %r: skill %r failed: %s", self.name, skill.name, e)
                continue
            if r is None:
                continue
            result = self._stamp_phase(r, r.skill_name)
            if best is None or result.confidence > best.confidence:
                best = result
        return best

    async def _run_skills_async(self, signal: Signal, explain: bool) -> SenseResult | None:
        if not self._llm_client:
            return None

        results = await asyncio.gather(
            *[self._run_one_skill_async(signal, skill, explain) for skill in self.skills],
            return_exceptions=True,
        )

        best: SenseResult | None = None
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Phase %r: skill failed: %s", self.name, r)
                continue
            if r is None:
                continue
            result = self._stamp_phase(r, r.skill_name)
            if best is None or result.confidence > best.confidence:
                best = result
        return best

    def _run_one_skill_sync(self, signal: Signal, skill: Skill, explain: bool) -> SenseResult | None:
        return self._llm_client.classify_sync(signal, skill, self._labels, include_reason=explain)

    async def _run_one_skill_async(self, signal: Signal, skill: Skill, explain: bool) -> SenseResult | None:
        return await self._llm_client.classify_async(signal, skill, self._labels, include_reason=explain)

    def _stamp_phase(self, result: SenseResult, skill_name: str) -> SenseResult:
        updates: dict[str, Any] = {}
        if not result.phase:
            updates["phase"] = self.name
        if not result.skill_name:
            updates["skill_name"] = skill_name
        if updates:
            return replace(result, **updates)
        return result

    def close_sync(self) -> None:
        if self._llm_client:
            self._llm_client.close_sync()

    async def close_async(self) -> None:
        if self._llm_client:
            await self._llm_client.close_async()

    async def close(self) -> None:
        if self._llm_client:
            await self._llm_client.close()
