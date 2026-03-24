from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from errorsense.llm import LLMConfig
    from errorsense.skill import Skill


@dataclass(frozen=True)
class SenseResult:
    """Result from classification — produced by rulesets or skills."""

    label: str
    confidence: float
    phase: str = ""
    skill_name: str = ""
    reason: str | None = None  # only set when explain=True, LLM phases only


@dataclass(frozen=True)
class TrailResult:
    """Result from trail() — classification + threshold state.

    If a review ran (threshold hit + review enabled), label and reason
    reflect the review's verdict. If the review changed the label,
    the history entry is updated and counts are adjusted.
    """

    label: str
    confidence: float
    phase: str
    skill_name: str
    at_threshold: bool
    reason: str | None = None  # LLM review explanation, None if no review ran


@dataclass(frozen=True)
class TrailingConfig:
    """Configuration for trailing (stateful error tracking).

    Args:
        threshold: Number of counted errors before review triggers.
        count_labels: Only these labels count toward threshold.
        history_size: Max errors kept per key (ring buffer).
        reviewer_llm: LLM config for review. Set to enable review, None to disable.
        reviewer_skill: Custom review skill. Defaults to built-in reviewer.
    """

    threshold: int = 3
    count_labels: list[str] | None = None
    history_size: int = 10
    reviewer_llm: LLMConfig | None = None
    reviewer_skill: Skill | None = None
