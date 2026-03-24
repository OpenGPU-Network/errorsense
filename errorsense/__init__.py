"""ErrorSense — error classification engine."""

from errorsense.engine import ErrorSense
from errorsense.llm import LLMConfig
from errorsense.models import (
    SenseResult,
    TrailResult,
    TrailingConfig,
)
from errorsense.phase import Phase
from errorsense.ruleset import Ruleset
from errorsense.signal import Signal
from errorsense.skill import Skill

__all__ = [
    "ErrorSense",
    "Phase",
    "Ruleset",
    "Skill",
    "LLMConfig",
    "Signal",
    "SenseResult",
    "TrailResult",
    "TrailingConfig",
]

__version__ = "0.2.0"
