"""ErrorSense — phase pipeline classification engine."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from errorsense.llm import LLMClient, LLMConfig
from errorsense.models import SenseResult, TrailResult, TrailingConfig
from errorsense.phase import Phase
from errorsense.ruleset import Ruleset
from errorsense.signal import Signal
from errorsense.skill import Skill

logger = logging.getLogger("errorsense")

__all__ = ["ErrorSense"]


class ErrorSense:
    """Classification engine — runs signals through a phase pipeline.

    Supports stateless classification (classify) and stateful trailing
    (trail) with per-key error history and threshold-based decisions.
    """

    def __init__(
        self,
        labels: list[str],
        # Explicit mode
        pipeline: list[Phase] | None = None,
        # Implicit mode
        rulesets: list[Ruleset] | None = None,
        skills: list[Skill] | None = None,
        llm: LLMConfig | None = None,
        # Common
        default: str = "unknown",
        trailing: TrailingConfig | None = None,
        on_classify: Callable[[Signal, SenseResult], Any] | None = None,
        on_error: Callable[[str, Exception], Any] | None = None,
    ) -> None:
        self.labels = set(labels)
        self.default = default
        self._on_classify = on_classify
        self._on_error = on_error

        if pipeline is not None:
            if rulesets is not None or skills is not None or llm is not None:
                raise ValueError(
                    "Cannot mix explicit (pipeline=) and implicit (rulesets=/skills=/llm=) modes"
                )
            self._pipeline = list(pipeline)
        else:
            self._pipeline = self._build_implicit_pipeline(rulesets, skills, llm)

        self._validate_phase_names()
        self._pipeline_names = frozenset(p.name for p in self._pipeline)
        self._validate_labels()
        self._validate_llm_api_keys()
        for phase in self._pipeline:
            phase.set_labels(list(labels))

        # Trailing state
        self._trailing = trailing
        self._reviewer_client: LLMClient | None = None
        self._reviewer_skill: Skill | None = None
        if trailing:
            self._init_trailing(trailing)

    def _init_trailing(self, config: TrailingConfig) -> None:
        if config.reviewer_llm is not None:
            self._reviewer_client = LLMClient(config.reviewer_llm)
            self._reviewer_skill = config.reviewer_skill
        self._threshold = config.threshold
        self._count_labels = set(config.count_labels or [])
        hs = config.history_size
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=hs)
        )
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._trail_lock = threading.Lock()
        self._trail_locks: dict[str, threading.Lock] = {}
        self._async_trail_lock = asyncio.Lock()
        self._async_trail_locks: dict[str, asyncio.Lock] = {}

    @property
    def pipeline(self) -> list[Phase]:
        return list(self._pipeline)

    def get_phase(self, name: str) -> Phase | None:
        for phase in self._pipeline:
            if phase.name == name:
                return phase
        return None

    def close(self) -> None:
        """Close all LLM phase clients (sync)."""
        for phase in self._pipeline:
            phase.close_sync()
        if self._reviewer_client:
            self._reviewer_client.close_sync()

    async def async_close(self) -> None:
        """Close all LLM phase clients (async)."""
        for phase in self._pipeline:
            await phase.close_async()
        if self._reviewer_client:
            await self._reviewer_client.close_async()

    async def __aenter__(self) -> ErrorSense:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.async_close()

    # -- Stateless classification --

    def classify(
        self,
        signal: Signal,
        skip: set[str] | list[str] | None = None,
        short_circuit: bool = True,
        explain: bool = False,
    ) -> list[SenseResult]:
        """Classify a signal through the phase pipeline (sync).

        Returns list of SenseResult from pipeline phases that matched.
        If nothing matched, returns [default_result].
        """
        skip_set = self._validate_skip(skip)
        results: list[SenseResult] = []

        for phase in self._pipeline:
            if phase.name in skip_set:
                continue
            try:
                result = phase.classify(signal, explain=explain)
            except Exception as e:
                logger.warning("Phase %r raised: %s", phase.name, e)
                self._notify_error(phase.name, e)
                continue

            if result is not None:
                results.append(result)
                self._notify_classify(signal, result)
                if short_circuit:
                    break

        if not results:
            results.append(self._make_default_result(signal))

        return results

    async def async_classify(
        self,
        signal: Signal,
        skip: set[str] | list[str] | None = None,
        short_circuit: bool = True,
        explain: bool = False,
    ) -> list[SenseResult]:
        """Classify a signal through the phase pipeline (async)."""
        skip_set = self._validate_skip(skip)
        results: list[SenseResult] = []

        for phase in self._pipeline:
            if phase.name in skip_set:
                continue
            try:
                result = await phase.async_classify(signal, explain=explain)
            except Exception as e:
                logger.warning("Phase %r raised: %s", phase.name, e)
                self._notify_error(phase.name, e)
                continue

            if result is not None:
                results.append(result)
                self._notify_classify(signal, result)
                if short_circuit:
                    break

        if not results:
            results.append(self._make_default_result(signal))

        return results

    # -- Stateful trailing --

    def trail(self, key: str, signal: Signal) -> TrailResult:
        """Classify + track per key (sync)."""
        if not self._trailing:
            raise RuntimeError(
                "Trailing not configured. Pass trailing=TrailingConfig(...) to ErrorSense."
            )

        with self._trail_lock:
            lock = self._trail_locks.setdefault(key, threading.Lock())
        with lock:
            result = self.classify(signal)[0]
            at_threshold = self._record_and_check(key, signal, result)
            review_result = (
                self._run_review_sync(key)
                if at_threshold and self._reviewer_client else None
            )
            return self._build_trail_result(key, result, at_threshold, review_result)

    async def async_trail(self, key: str, signal: Signal) -> TrailResult:
        """Classify + track per key (async)."""
        if not self._trailing:
            raise RuntimeError(
                "Trailing not configured. Pass trailing=TrailingConfig(...) to ErrorSense."
            )

        async with self._async_trail_lock:
            if key not in self._async_trail_locks:
                self._async_trail_locks[key] = asyncio.Lock()
            lock = self._async_trail_locks[key]
        async with lock:
            result = (await self.async_classify(signal))[0]
            at_threshold = self._record_and_check(key, signal, result)
            review_result = (
                await self._run_review_async(key)
                if at_threshold and self._reviewer_client else None
            )
            return self._build_trail_result(key, result, at_threshold, review_result)

    def review(self, key: str) -> SenseResult | None:
        """Manually review full history for a key (sync). Returns LLM verdict."""
        if not self._trailing:
            raise RuntimeError(
                "Trailing not configured. Pass trailing=TrailingConfig(...) to ErrorSense."
            )
        return self._run_review_sync(key)

    async def async_review(self, key: str) -> SenseResult | None:
        """Manually review full history for a key (async). Returns LLM verdict."""
        if not self._trailing:
            raise RuntimeError(
                "Trailing not configured. Pass trailing=TrailingConfig(...) to ErrorSense."
            )
        return self._run_review_async(key)

    def _record_and_check(self, key: str, signal: Signal, result: SenseResult) -> bool:
        entry = {
            "label": result.label,
            "confidence": result.confidence,
            "phase": result.phase,
            "skill": result.skill_name,
            "timestamp": time.time(),
            "signal_data": signal.to_dict(),
        }
        history = self._history[key]

        if len(history) == history.maxlen:
            evicted = history[0]["label"]
            if evicted in self._count_labels and self._counts[key].get(evicted, 0) > 0:
                self._counts[key][evicted] -= 1

        history.append(entry)

        if result.label in self._count_labels:
            self._counts[key][result.label] += 1

        return self._is_at_threshold(key)

    def _build_trail_result(
        self, key: str, result: SenseResult, at_threshold: bool,
        review_result: SenseResult | None,
    ) -> TrailResult:
        label = result.label
        reason = None
        if review_result:
            reason = review_result.reason
            if review_result.label != result.label:
                self._update_latest_label(key, result.label, review_result.label)
                label = review_result.label
                at_threshold = self._is_at_threshold(key)

        return TrailResult(
            label=label,
            confidence=result.confidence,
            phase=result.phase,
            skill_name=result.skill_name,
            at_threshold=at_threshold,
            reason=reason,
        )

    def _is_at_threshold(self, key: str) -> bool:
        return sum(self._counts[key].values()) >= self._threshold

    def _update_latest_label(self, key: str, old_label: str, new_label: str) -> None:
        """Update the most recent history entry's label and adjust counts."""
        history = self._history[key]
        if not history:
            return
        history[-1]["label"] = new_label

        if old_label in self._count_labels and self._counts[key].get(old_label, 0) > 0:
            self._counts[key][old_label] -= 1
        if new_label in self._count_labels:
            self._counts[key][new_label] += 1

    def _run_review_sync(self, key: str) -> SenseResult | None:
        if not self._reviewer_client:
            return None
        signal, skill = self._build_review_context(key)
        try:
            return self._reviewer_client.classify_sync(
                signal, skill, list(self.labels), include_reason=True,
            )
        except Exception as e:
            logger.warning("LLM review failed: %s", e)
            return None

    async def _run_review_async(self, key: str) -> SenseResult | None:
        if not self._reviewer_client:
            return None
        signal, skill = self._build_review_context(key)
        try:
            return await self._reviewer_client.classify_async(
                signal, skill, list(self.labels), include_reason=True,
            )
        except Exception as e:
            logger.warning("LLM review failed: %s", e)
            return None

    def _get_reviewer_skill(self) -> Skill:
        if self._reviewer_skill is None:
            self._reviewer_skill = Skill("reclassification")
        return self._reviewer_skill

    def _build_review_context(self, key: str) -> tuple[Signal, Skill]:
        history = list(self._history[key])
        summary = json.dumps(
            [{"label": e["label"], "phase": e.get("phase", ""), "signal": e.get("signal_data", {})}
             for e in history],
            default=str,
        )
        signal = Signal({
            "context": "trailing_review",
            "key": key,
            "history_summary": summary,
        })
        return signal, self._get_reviewer_skill()

    def reset(self, key: str) -> None:
        """Clear trailing history and counts for a key."""
        if not self._trailing:
            return
        with self._trail_lock:
            lock = self._trail_locks.get(key)
        if lock is not None:
            with lock:
                self._history.pop(key, None)
                self._counts.pop(key, None)
            with self._trail_lock:
                if self._trail_locks.get(key) is lock:
                    del self._trail_locks[key]
        else:
            self._history.pop(key, None)
            self._counts.pop(key, None)

    def reset_all(self) -> None:
        """Clear all trailing state."""
        if not self._trailing:
            return
        with self._trail_lock:
            self._history.clear()
            self._counts.clear()
            self._trail_locks.clear()

    # -- Internal --

    def _build_implicit_pipeline(
        self,
        rulesets: list[Ruleset] | None,
        skills: list[Skill] | None,
        llm: LLMConfig | None,
    ) -> list[Phase]:
        phases: list[Phase] = []
        if rulesets:
            phases.append(Phase("rulesets", rulesets=rulesets))
        if skills:
            if not llm:
                raise ValueError("skills= requires llm=LLMConfig(...)")
            phases.append(Phase("llm", skills=skills, llm=llm))
        if not phases:
            raise ValueError("Must provide pipeline= or at least rulesets= or skills=")
        return phases

    def _validate_phase_names(self) -> None:
        seen: set[str] = set()
        for phase in self._pipeline:
            if phase.name in seen:
                raise ValueError(f"Duplicate phase name: {phase.name!r}")
            seen.add(phase.name)

    def _validate_labels(self) -> None:
        all_labels = self.labels | {self.default}
        for phase in self._pipeline:
            for ruleset in phase.rulesets:
                bad = ruleset.referenced_labels() - all_labels
                if bad:
                    raise ValueError(
                        f"Ruleset on field {getattr(ruleset, 'field', '?')!r} maps to "
                        f"label {bad.pop()!r} not in {sorted(self.labels)}"
                    )

    def _validate_llm_api_keys(self) -> None:
        for phase in self._pipeline:
            if not phase.is_llm_phase:
                continue
            if not phase.llm:
                raise ValueError(
                    f"Phase {phase.name!r} uses LLM skills but no LLM config provided. "
                    f"Pass llm=LLMConfig() to the Phase."
                )

    def _validate_skip(self, skip: set[str] | list[str] | None) -> set[str]:
        if not skip:
            return set()
        skip_set = set(skip)
        invalid = skip_set - self._pipeline_names
        if invalid:
            raise ValueError(
                f"Unknown phase names in skip: {invalid}. "
                f"Valid phase names: {sorted(self._pipeline_names)}"
            )
        return skip_set

    def _make_default_result(self, signal: Signal) -> SenseResult:
        result = SenseResult(
            label=self.default,
            confidence=0.0,
            skill_name="default",
        )
        self._notify_classify(signal, result)
        return result

    def _notify_classify(self, signal: Signal, result: SenseResult) -> None:
        if self._on_classify:
            try:
                self._on_classify(signal, result)
            except Exception as e:
                logger.debug("on_classify callback raised: %s", e)

    def _notify_error(self, phase_name: str, error: Exception) -> None:
        if self._on_error:
            try:
                self._on_error(phase_name, error)
            except Exception as e:
                logger.debug("on_error callback raised: %s", e)
