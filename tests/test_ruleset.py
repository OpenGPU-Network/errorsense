"""Tests for Ruleset — field matching, ranges, headers, JSON, patterns, custom subclass."""

import json
import pytest

from errorsense import Ruleset, Signal, SenseResult


class TestFieldMatch:
    def test_exact_int_match(self):
        rs = Ruleset(field="status_code", match={400: "user", 502: "infra"})
        result = rs.classify(Signal({"status_code": 400}))
        assert result.label == "user"
        assert result.confidence == 1.0

    def test_exact_string_match(self):
        rs = Ruleset(field="level", match={"ERROR": "critical", "WARN": "warning"})
        result = rs.classify(Signal({"level": "ERROR"}))
        assert result.label == "critical"

    def test_no_match_returns_none(self):
        rs = Ruleset(field="status_code", match={400: "user"})
        result = rs.classify(Signal({"status_code": 200}))
        assert result is None

    def test_missing_field_returns_none(self):
        rs = Ruleset(field="status_code", match={400: "user"})
        result = rs.classify(Signal({"other": "stuff"}))
        assert result is None

    def test_none_value_explicit_pass(self):
        rs = Ruleset(field="content_type", match={"application/json": None, "text/html": "infra"})
        result = rs.classify(Signal({"content_type": "application/json"}))
        assert result is None


class TestRangeMatch:
    def test_4xx_range(self):
        rs = Ruleset(field="status_code", match={"4xx": "user", "5xx": "server"})
        result = rs.classify(Signal({"status_code": 404}))
        assert result.label == "user"

    def test_5xx_range(self):
        rs = Ruleset(field="status_code", match={"4xx": "user", "5xx": "server"})
        result = rs.classify(Signal({"status_code": 500}))
        assert result.label == "server"

    def test_exact_overrides_range(self):
        rs = Ruleset(field="status_code", match={"5xx": "server", 503: "infra"})
        assert rs.classify(Signal({"status_code": 503})).label == "infra"
        assert rs.classify(Signal({"status_code": 500})).label == "server"

    def test_no_range_match(self):
        rs = Ruleset(field="status_code", match={"4xx": "user"})
        result = rs.classify(Signal({"status_code": 200}))
        assert result is None


class TestHeaderMatch:
    def test_content_type_exact_match(self):
        rs = Ruleset(field="headers.content-type", match={"text/html": "infra"})
        signal = Signal.from_http(status_code=500, body="<html>", headers={"content-type": "text/html"})
        result = rs.classify(signal)
        assert result.label == "infra"

    def test_content_type_with_charset_no_match(self):
        """Exact match: 'text/html' != 'text/html; charset=utf-8'. Use patterns for prefix matching."""
        rs = Ruleset(field="headers.content-type", match={"text/html": "infra"})
        signal = Signal.from_http(status_code=500, body="<html>", headers={"content-type": "text/html; charset=utf-8"})
        result = rs.classify(signal)
        assert result is None

    def test_content_type_pattern_match(self):
        """Use patterns= for prefix matching on content-type with charset."""
        rs = Ruleset(field="headers.content-type", patterns=[("infra", [r"^text/html"])])
        signal = Signal.from_http(status_code=500, body="<html>", headers={"content-type": "text/html; charset=utf-8"})
        result = rs.classify(signal)
        assert result.label == "infra"

    def test_content_type_none_pass(self):
        rs = Ruleset(field="headers.content-type", match={"application/json": None})
        signal = Signal.from_http(status_code=500, body="{}", headers={"content-type": "application/json"})
        result = rs.classify(signal)
        assert result is None

    def test_missing_headers_returns_none(self):
        rs = Ruleset(field="headers.content-type", match={"text/html": "infra"})
        result = rs.classify(Signal({"body": "hello"}))
        assert result is None


class TestJsonBodyMatch:
    def test_dot_path_field_match(self):
        body = json.dumps({"type": "invalid_request_error"})
        rs = Ruleset(field="body.type", match={"invalid_request_error": "user"})
        result = rs.classify(Signal.from_http(status_code=500, body=body))
        assert result.label == "user"

    def test_nested_dot_path(self):
        body = json.dumps({"error": {"type": "server_error"}})
        rs = Ruleset(field="body.error.type", match={"server_error": "infra"})
        result = rs.classify(Signal.from_http(status_code=500, body=body))
        assert result.label == "infra"

    def test_invalid_json_returns_none(self):
        rs = Ruleset(field="body.type", match={"error": "infra"})
        result = rs.classify(Signal.from_http(status_code=500, body="not json"))
        assert result is None

    def test_missing_path_returns_none(self):
        body = json.dumps({"other": "field"})
        rs = Ruleset(field="body.type", match={"error": "infra"})
        result = rs.classify(Signal.from_http(status_code=500, body=body))
        assert result is None


class TestPatterns:
    def test_regex_match(self):
        rs = Ruleset(field="body", patterns=[
            ("infra", [r"cuda", r"out of memory"]),
            ("user", [r"model.*not found"]),
        ])
        result = rs.classify(Signal.from_http(status_code=500, body="CUDA OOM error"))
        assert result.label == "infra"
        assert result.confidence == 0.9

    def test_case_insensitive_by_default(self):
        rs = Ruleset(field="body", patterns=[("infra", [r"cuda"])])
        result = rs.classify(Signal.from_http(status_code=500, body="CUDA error"))
        assert result is not None

    def test_case_sensitive(self):
        rs = Ruleset(field="body", patterns=[("infra", [r"cuda"])], case_sensitive=True)
        result = rs.classify(Signal.from_http(status_code=500, body="CUDA error"))
        assert result is None
        result = rs.classify(Signal.from_http(status_code=500, body="cuda error"))
        assert result is not None

    def test_no_pattern_match(self):
        rs = Ruleset(field="body", patterns=[("infra", [r"cuda"])])
        result = rs.classify(Signal.from_http(status_code=500, body="unknown error"))
        assert result is None

    def test_json_body_pattern(self):
        body = json.dumps({"error": {"message": "CUDA out of memory"}})
        rs = Ruleset(field="body.error.message", patterns=[("infra", [r"CUDA"])])
        result = rs.classify(Signal.from_http(status_code=500, body=body))
        assert result.label == "infra"

    def test_non_string_field_returns_none(self):
        rs = Ruleset(field="status_code", patterns=[("infra", [r"5\d\d"])])
        result = rs.classify(Signal({"status_code": 500}))
        assert result is None


class TestValidation:
    def test_match_and_patterns_raises(self):
        with pytest.raises(ValueError, match="match= OR patterns="):
            Ruleset(field="body", match={"a": "b"}, patterns=[("c", [r"d"])])

    def test_neither_match_nor_patterns_raises(self):
        with pytest.raises(ValueError, match="match= or patterns="):
            Ruleset(field="body")

    def test_no_field_raises(self):
        with pytest.raises(ValueError, match="field"):
            Ruleset(match={"a": "b"})


class TestCustomSubclass:
    def test_custom_classify(self):
        class VendorBug(Ruleset):
            def classify(self, signal: Signal) -> SenseResult | None:
                if signal.get("vendor") == "acme":
                    return SenseResult(label="known_bug", confidence=1.0)
                return None

        rs = VendorBug()
        result = rs.classify(Signal({"vendor": "acme"}))
        assert result.label == "known_bug"

    def test_custom_no_match(self):
        class VendorBug(Ruleset):
            def classify(self, signal: Signal) -> SenseResult | None:
                return None

        rs = VendorBug()
        assert rs.classify(Signal({"x": 1})) is None

    def test_referenced_labels_empty_for_custom(self):
        class Custom(Ruleset):
            def classify(self, signal): return None

        assert Custom().referenced_labels() == set()
