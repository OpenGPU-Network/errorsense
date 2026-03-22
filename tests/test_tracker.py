"""Tests for trailing (stateful error tracking) on ErrorSense."""

import pytest

from errorsense import ErrorSense, Phase, Ruleset, Signal, TrailingConfig


def make_sense(**kwargs):
    """Helper to build an ErrorSense with trailing enabled."""
    defaults = {
        "categories": ["infra", "provider", "user"],
        "pipeline": [
            Phase("rules", rulesets=[
                Ruleset(field="status_code", match={400: "user", 401: "user", 502: "infra", 503: "infra"}),
            ]),
            Phase("patterns", rulesets=[
                Ruleset(field="body", patterns=[
                    ("user", [r"model.*not found", r"invalid"]),
                    ("infra", [r"cuda", r"connection refused"]),
                ]),
            ]),
        ],
        "default": "provider",
        "trailing": TrailingConfig(
            threshold=3,
            count_labels=["infra", "provider"],
        ),
    }
    defaults.update(kwargs)
    return ErrorSense(**defaults)


class TestTrail:
    def test_user_errors_dont_count(self):
        sense = make_sense()
        for _ in range(5):
            result = sense.trail("p1", Signal.from_http(status_code=400, body="bad request"))
            assert result.label == "user"
            assert result.at_threshold is False

    def test_infra_errors_count(self):
        sense = make_sense(trailing=TrailingConfig(threshold=3, count_labels=["infra", "provider"]))
        for i in range(2):
            result = sense.trail("p1", Signal.from_http(status_code=502, body="Bad Gateway"))
            assert result.at_threshold is False
        result = sense.trail("p1", Signal.from_http(status_code=503, body="Service Unavailable"))
        assert result.at_threshold is True

    def test_mixed_errors(self):
        sense = make_sense(trailing=TrailingConfig(threshold=3, count_labels=["infra", "provider"]))
        sense.trail("p1", Signal.from_http(status_code=400, body="bad"))
        sense.trail("p1", Signal.from_http(status_code=400, body="bad"))
        sense.trail("p1", Signal.from_http(status_code=502))
        result = sense.trail("p1", Signal.from_http(status_code=502))
        assert result.at_threshold is False
        result = sense.trail("p1", Signal.from_http(status_code=503))
        assert result.at_threshold is True

    def test_default_label_counts(self):
        sense = make_sense(trailing=TrailingConfig(threshold=2, count_labels=["infra", "provider"]))
        result = sense.trail("p1", Signal.from_http(status_code=500, body="mystery"))
        assert result.label == "provider"
        result = sense.trail("p1", Signal.from_http(status_code=500, body="mystery"))
        assert result.at_threshold is True

    def test_reset(self):
        sense = make_sense(trailing=TrailingConfig(threshold=3, count_labels=["infra", "provider"]))
        sense.trail("p1", Signal.from_http(status_code=502))
        sense.trail("p1", Signal.from_http(status_code=502))
        sense.reset("p1")
        result = sense.trail("p1", Signal.from_http(status_code=502))
        assert result.at_threshold is False

    def test_reset_all(self):
        sense = make_sense(trailing=TrailingConfig(threshold=2, count_labels=["infra", "provider"]))
        sense.trail("p1", Signal.from_http(status_code=502))
        sense.trail("p2", Signal.from_http(status_code=502))
        sense.reset_all()
        r1 = sense.trail("p1", Signal.from_http(status_code=502))
        r2 = sense.trail("p2", Signal.from_http(status_code=502))
        assert r1.at_threshold is False
        assert r2.at_threshold is False

    def test_separate_keys(self):
        sense = make_sense(trailing=TrailingConfig(threshold=2, count_labels=["infra", "provider"]))
        sense.trail("p1", Signal.from_http(status_code=502))
        sense.trail("p2", Signal.from_http(status_code=502))
        r1 = sense.trail("p1", Signal.from_http(status_code=502))
        assert r1.at_threshold is True
        r2 = sense.trail("p2", Signal.from_http(status_code=503))
        assert r2.at_threshold is True

    def test_counts_decrement_on_history_eviction(self):
        sense = make_sense(trailing=TrailingConfig(
            threshold=3, count_labels=["infra", "provider"], history_size=3,
        ))
        for _ in range(3):
            sense.trail("p1", Signal.from_http(status_code=502))
        result = sense.trail("p1", Signal.from_http(status_code=502))
        assert result.at_threshold is True

        for _ in range(3):
            result = sense.trail("p1", Signal.from_http(status_code=400, body="bad"))
        assert result.at_threshold is False
        assert result.label == "user"

    def test_phase_in_trail_result(self):
        sense = make_sense()
        result = sense.trail("p1", Signal.from_http(status_code=502))
        assert result.phase == "rules"

    def test_trail_without_config_raises(self):
        sense = ErrorSense(
            categories=["a"],
            pipeline=[Phase("p1", rulesets=[Ruleset(field="x", match={1: "a"})])],
        )
        with pytest.raises(RuntimeError, match="Trailing not configured"):
            sense.trail("key", Signal({"x": 1}))

    def test_review_true_without_llm_raises(self):
        with pytest.raises(ValueError, match="requires an LLM phase"):
            make_sense(trailing=TrailingConfig(
                threshold=3, count_labels=["infra"], review=True,
            ))

    def test_review_false_no_review(self):
        sense = make_sense(trailing=TrailingConfig(
            threshold=2, count_labels=["infra", "provider"], review=False,
        ))
        sense.trail("p1", Signal.from_http(status_code=502))
        result = sense.trail("p1", Signal.from_http(status_code=502))
        assert result.at_threshold is True
        assert result.reason is None
