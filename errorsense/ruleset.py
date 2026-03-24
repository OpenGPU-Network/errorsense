"""Ruleset — deterministic (non-LLM) classification logic."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from errorsense.models import SenseResult
from errorsense.signal import Signal

logger = logging.getLogger("errorsense")

__all__ = ["Ruleset"]


def _resolve_dotted(data: Any, path: str) -> Any:
    """Resolve a dotted path like 'error.type' into nested dict access."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


class Ruleset:
    """Deterministic classification logic.

    Each ruleset does one thing: either field matching (match=) or regex
    patterns (patterns=). Not both. Subclass and override classify() for
    custom logic beyond config.
    """

    def __init__(
        self,
        field: str | None = None,
        match: dict[Any, str | None] | None = None,
        patterns: list[tuple[str, list[str]]] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        if type(self) is Ruleset:
            if not field:
                raise ValueError("Ruleset requires a 'field' parameter")
            if match is not None and patterns is not None:
                raise ValueError(
                    "Ruleset takes match= OR patterns=, not both. "
                    "Use separate rulesets in the same phase."
                )
            if match is None and patterns is None:
                raise ValueError("Ruleset requires either match= or patterns=")

        self._init_fields(field, match, patterns, case_sensitive)

    def _init_fields(
        self,
        field: str | None,
        match: dict[Any, str | None] | None,
        patterns: list[tuple[str, list[str]]] | None,
        case_sensitive: bool,
    ) -> None:
        self.field = field
        self._match = match
        self._range_keys: dict[str, str] = {}
        self._exact_keys: dict[Any, str | None] = {}
        self._compiled: list[tuple[str, list[re.Pattern[str]]]] | None = None

        if match:
            self._split_match_keys(match)
        if patterns:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._compiled = [
                (label, [re.compile(p, flags) for p in pats])
                for label, pats in patterns
            ]

    def _split_match_keys(self, match: dict[Any, str | None]) -> None:
        for key, value in match.items():
            if isinstance(key, str) and len(key) == 3 and key[0].isdigit() and key.endswith("xx"):
                if value is not None:
                    self._range_keys[key] = value
            else:
                self._exact_keys[key] = value

    def referenced_labels(self) -> set[str]:
        """Return set of label strings this ruleset can produce. Used by engine validation."""
        labels: set[str] = set()
        match = getattr(self, "_match", None)
        if match is not None:
            labels |= {v for v in match.values() if isinstance(v, str)}
        compiled = getattr(self, "_compiled", None)
        if compiled is not None:
            labels |= {label for label, _ in compiled}
        return labels

    def classify(self, signal: Signal) -> SenseResult | None:
        """Classify a signal. Override in subclass for custom logic."""
        value = self._resolve_field(signal)
        if value is None:
            return None

        if self._match is not None:
            return self._match_value(value)
        if self._compiled is not None:
            return self._match_patterns(value)
        return None

    def _resolve_field(self, signal: Signal) -> Any:
        field = self.field
        if field is None:
            return None

        if field.startswith("headers."):
            headers = signal.get("headers")
            if not hasattr(headers, "get"):
                return None
            header_name = field[len("headers."):]
            return headers.get(header_name, "")

        if field.startswith("body."):
            body = signal.get("body")
            if not isinstance(body, str):
                return None
            try:
                parsed = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Ruleset %r: failed to parse JSON body", field)
                return None
            if not isinstance(parsed, dict):
                return None
            dot_path = field[len("body."):]
            return _resolve_dotted(parsed, dot_path)

        return signal.get(field)

    def _match_value(self, value: Any) -> SenseResult | None:
        if value in self._exact_keys:
            label = self._exact_keys[value]
            if label is None:
                return None
            return SenseResult(label=label, confidence=1.0)

        if isinstance(value, int) and self._range_keys:
            range_key = f"{value // 100}xx"
            if range_key in self._range_keys:
                label = self._range_keys[range_key]
                return SenseResult(label=label, confidence=1.0)

        return None

    def _match_patterns(self, value: Any) -> SenseResult | None:
        if not isinstance(value, str):
            return None
        for label, compiled_pats in self._compiled:
            for pat in compiled_pats:
                if pat.search(value):
                    return SenseResult(label=label, confidence=0.9)
        return None
