from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        review: Whether to LLM-review history when threshold hit.
            None = auto (True if LLM phase exists, False if not).
            True = force (raises if no LLM phase).
            False = never.
    """

    threshold: int = 3
    count_labels: list[str] | None = None
    history_size: int = 10
    review: bool | None = None
