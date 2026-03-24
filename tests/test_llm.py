"""Tests for LLM prompt building and response parsing."""

import json

from errorsense.llm import (
    DEFAULT_PROMPT_FORMAT,
    LLMConfig,
    _build_prompt,
    _extract_json,
    _parse_response,
)
from errorsense.signal import Signal
from errorsense.skill import Skill


class TestParseResponse:
    def _make_response(self, content: str) -> dict:
        return {"choices": [{"message": {"content": content}}]}

    def test_valid_json(self):
        data = self._make_response('{"label": "server", "confidence": 0.9, "reason": "OOM"}')
        result = _parse_response(data, ["client", "server"], "test_skill", include_reason=True)
        assert result is not None
        assert result.label == "server"
        assert result.confidence == 0.9
        assert result.reason == "OOM"

    def test_reason_excluded_when_not_requested(self):
        data = self._make_response('{"label": "server", "confidence": 0.9, "reason": "OOM"}')
        result = _parse_response(data, ["client", "server"], "test_skill", include_reason=False)
        assert result is not None
        assert result.reason is None

    def test_unknown_label_returns_none(self):
        data = self._make_response('{"label": "bogus", "confidence": 0.9}')
        result = _parse_response(data, ["client", "server"], "test_skill")
        assert result is None

    def test_empty_label_returns_none(self):
        data = self._make_response('{"confidence": 0.9}')
        result = _parse_response(data, ["client", "server"], "test_skill")
        assert result is None

    def test_malformed_json_returns_none(self):
        data = self._make_response("not json at all")
        result = _parse_response(data, ["client", "server"], "test_skill")
        assert result is None

    def test_code_fence_stripped(self):
        content = '```json\n{"label": "client", "confidence": 0.8}\n```'
        data = self._make_response(content)
        result = _parse_response(data, ["client", "server"], "test_skill")
        assert result is not None
        assert result.label == "client"

    def test_confidence_clamped_high(self):
        data = self._make_response('{"label": "server", "confidence": 5.0}')
        result = _parse_response(data, ["server"], "test_skill")
        assert result.confidence == 1.0

    def test_confidence_clamped_low(self):
        data = self._make_response('{"label": "server", "confidence": -1.0}')
        result = _parse_response(data, ["server"], "test_skill")
        assert result.confidence == 0.0

    def test_default_confidence(self):
        data = self._make_response('{"label": "server"}')
        result = _parse_response(data, ["server"], "test_skill")
        assert result.confidence == 0.7

    def test_missing_choices_returns_none(self):
        result = _parse_response({}, ["server"], "test_skill")
        assert result is None

    def test_empty_labels_accepts_anything(self):
        data = self._make_response('{"label": "whatever", "confidence": 0.5}')
        result = _parse_response(data, [], "test_skill")
        assert result is not None
        assert result.label == "whatever"

    def test_skill_name_set(self):
        data = self._make_response('{"label": "server", "confidence": 0.9}')
        result = _parse_response(data, ["server"], "my_skill")
        assert result.skill_name == "my_skill"


class TestBuildPrompt:
    def _make_skill(self, instructions: str = "Classify this error.", **kwargs) -> Skill:
        return Skill("test", instructions=instructions, **kwargs)

    def test_basic_format(self):
        signal = Signal({"status_code": 500})
        skill = self._make_skill()
        config = LLMConfig()
        prompt = _build_prompt(signal, skill, ["client", "server"], config)
        assert "Classify this error." in prompt
        assert "client, server" in prompt
        assert "500" in prompt

    def test_truncation(self):
        signal = Signal({"body": "x" * 1000})
        skill = self._make_skill()
        config = LLMConfig(max_signal_size=50)
        prompt = _build_prompt(signal, skill, ["a"], config)
        assert "...(truncated)" in prompt

    def test_no_truncation_when_small(self):
        signal = Signal({"x": 1})
        skill = self._make_skill()
        config = LLMConfig(max_signal_size=500)
        prompt = _build_prompt(signal, skill, ["a"], config)
        assert "(truncated)" not in prompt

    def test_custom_template(self):
        skill = self._make_skill(prompt_format="Labels: {labels}\nSignal: {signal}\n{instructions}")
        signal = Signal({"code": 1})
        config = LLMConfig()
        prompt = _build_prompt(signal, skill, ["a", "b"], config)
        assert prompt.startswith("Labels: a, b")
        assert "Classify this error." in prompt

    def test_empty_labels(self):
        signal = Signal({"x": 1})
        skill = self._make_skill()
        config = LLMConfig()
        prompt = _build_prompt(signal, skill, [], config)
        assert "unknown" in prompt

    def test_prompt_says_json_only(self):
        signal = Signal({"x": 1})
        skill = self._make_skill()
        config = LLMConfig()
        prompt = _build_prompt(signal, skill, ["a"], config)
        assert "ONLY" in prompt
        assert "No explanation" in prompt


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"label": "x"}') == {"label": "x"}

    def test_thinking_block(self):
        raw = '<think>\nLet me analyze...\n</think>\n{"label": "server", "confidence": 0.9}'
        result = _extract_json(raw)
        assert result["label"] == "server"

    def test_reasoning_block(self):
        raw = '<reasoning>The error is clearly a timeout</reasoning>\n{"label": "transient"}'
        result = _extract_json(raw)
        assert result["label"] == "transient"

    def test_multiline_xml_block(self):
        raw = '<scratchpad>step 1\nstep 2</scratchpad>\n\n{"label": "client"}'
        result = _extract_json(raw)
        assert result["label"] == "client"

    def test_code_fence(self):
        raw = '```json\n{"label": "server"}\n```'
        result = _extract_json(raw)
        assert result["label"] == "server"

    def test_json_in_prose(self):
        raw = 'Based on analysis, here is the result:\n{"label": "server", "confidence": 0.8}\nDone.'
        result = _extract_json(raw)
        assert result["label"] == "server"

    def test_no_json(self):
        assert _extract_json("This is just text with no JSON") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_xml_block_then_code_fence(self):
        raw = '<think>analyzing...</think>\n```json\n{"label": "client"}\n```'
        result = _extract_json(raw)
        assert result["label"] == "client"

    def test_unclosed_brace(self):
        assert _extract_json('{"label": "server"') is None
