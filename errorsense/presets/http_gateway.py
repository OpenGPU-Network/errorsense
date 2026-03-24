"""HTTP presets — client vs server error classification."""

from __future__ import annotations

from errorsense.engine import ErrorSense
from errorsense.llm import LLMConfig
from errorsense.phase import Phase
from errorsense.ruleset import Ruleset
from errorsense.skill import Skill

__all__ = ["http", "http_no_llm"]


def _ruleset_phases(extra_rulesets: list[Ruleset] | None = None) -> list[Phase]:
    """Shared ruleset phases for both http() and http_no_llm()."""
    return [
        Phase("rules", rulesets=[
            Ruleset(field="status_code", match={
                "4xx": "client", 502: "server", 503: "server", 504: "server",
            }),
            Ruleset(field="headers.content-type", patterns=[
                ("server", [r"^text/html"]),
            ]),
        ]),
        Phase("patterns", rulesets=[
            Ruleset(field="body", patterns=[
                ("server", [r"Bad Gateway", r"Service Unavailable", r"Gateway Timeout"]),
            ]),
            *(extra_rulesets or []),
        ]),
    ]


def http(
    llm: LLMConfig,
    extra_rulesets: list[Ruleset] | None = None,
) -> ErrorSense:
    """HTTP error classification with LLM: client, server, or undecided.

    Rulesets handle clear-cut cases (4xx, 502/503/504, HTML responses).
    LLM handles ambiguous errors — this is where ErrorSense earns its keep.

    Args:
        llm: LLM connection config (required).
        extra_rulesets: Additional rulesets appended to the patterns phase.
    """
    phases = _ruleset_phases(extra_rulesets)
    phases.append(Phase("llm", skills=[Skill("http_classifier")], llm=llm))

    return ErrorSense(
        labels=["client", "server", "undecided"],
        pipeline=phases,
        default="undecided",
    )


def http_no_llm(
    extra_rulesets: list[Ruleset] | None = None,
) -> ErrorSense:
    """HTTP error classification without LLM: client, server, or undecided.

    Only classifies clear-cut cases (status codes, gateway patterns).
    Ambiguous errors are "undecided".

    Args:
        extra_rulesets: Additional rulesets appended to the patterns phase.
    """
    return ErrorSense(
        labels=["client", "server", "undecided"],
        pipeline=_ruleset_phases(extra_rulesets),
        default="undecided",
    )
