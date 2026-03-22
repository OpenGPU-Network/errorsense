import pytest

from errorsense import ErrorSense, Phase, Ruleset, Signal, SenseResult


class AlwaysMatchRuleset(Ruleset):
    def __init__(self, label: str, confidence: float = 1.0):
        self._label = label
        self._confidence = confidence

    def classify(self, signal: Signal) -> SenseResult | None:
        return SenseResult(label=self._label, confidence=self._confidence)


class NeverMatchRuleset(Ruleset):
    def classify(self, signal: Signal) -> SenseResult | None:
        return None


class BrokenRuleset(Ruleset):
    def classify(self, signal: Signal) -> SenseResult | None:
        raise RuntimeError("boom")


class TestExplicitMode:
    def test_first_match_wins(self):
        engine = ErrorSense(
            categories=["a", "b"],
            phases=[Phase("p1", rulesets=[AlwaysMatchRuleset("a"), AlwaysMatchRuleset("b")])],
        )
        results = engine.classify(Signal({"x": 1}))
        assert len(results) == 1
        assert results[0].label == "a"

    def test_skip_none_results(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[NeverMatchRuleset(), AlwaysMatchRuleset("a")])],
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "a"

    def test_default_when_no_match(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[NeverMatchRuleset()])],
            default="fallback",
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "fallback"
        assert results[0].confidence == 0.0

    def test_skill_name_auto_filled(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[AlwaysMatchRuleset("a")])],
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].skill_name == "AlwaysMatchRuleset"
        assert results[0].phase == "p1"

    def test_broken_ruleset_skipped(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[BrokenRuleset(), AlwaysMatchRuleset("a")])],
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "a"

    def test_all_broken_falls_to_default(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[BrokenRuleset()])],
            default="oops",
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "oops"

    def test_multi_phase_first_catch(self):
        engine = ErrorSense(
            categories=["a", "b"],
            phases=[
                Phase("first", rulesets=[NeverMatchRuleset()]),
                Phase("second", rulesets=[AlwaysMatchRuleset("b")]),
            ],
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "b"
        assert results[0].phase == "second"

    def test_skip_phase(self):
        engine = ErrorSense(
            categories=["a", "b"],
            phases=[
                Phase("skip_me", rulesets=[AlwaysMatchRuleset("a")]),
                Phase("use_me", rulesets=[AlwaysMatchRuleset("b")]),
            ],
        )
        results = engine.classify(Signal({"x": 1}), skip=["skip_me"])
        assert results[0].label == "b"

    def test_skip_invalid_phase_raises(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[AlwaysMatchRuleset("a")])],
        )
        with pytest.raises(ValueError, match="Unknown phase"):
            engine.classify(Signal({"x": 1}), skip=["typo"])

    def test_duplicate_phase_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            ErrorSense(
                categories=["a"],
                phases=[
                    Phase("p1", rulesets=[AlwaysMatchRuleset("a")]),
                    Phase("p1", rulesets=[AlwaysMatchRuleset("a")]),
                ],
            )


class TestShortCircuit:
    def test_short_circuit_true_returns_one(self):
        engine = ErrorSense(
            categories=["a", "b"],
            phases=[
                Phase("first", rulesets=[AlwaysMatchRuleset("a")]),
                Phase("second", rulesets=[AlwaysMatchRuleset("b")]),
            ],
        )
        results = engine.classify(Signal({"x": 1}))
        assert len(results) == 1
        assert results[0].label == "a"

    def test_short_circuit_false_returns_all_matches(self):
        engine = ErrorSense(
            categories=["a", "b"],
            phases=[
                Phase("first", rulesets=[AlwaysMatchRuleset("a", confidence=0.8)]),
                Phase("second", rulesets=[AlwaysMatchRuleset("b", confidence=0.9)]),
            ],
        )
        results = engine.classify(Signal({"x": 1}), short_circuit=False)
        assert len(results) == 2
        assert results[0].label == "a"
        assert results[1].label == "b"

    def test_short_circuit_false_skips_unmatched(self):
        engine = ErrorSense(
            categories=["a", "b"],
            phases=[
                Phase("first", rulesets=[AlwaysMatchRuleset("a")]),
                Phase("second", rulesets=[NeverMatchRuleset()]),
                Phase("third", rulesets=[AlwaysMatchRuleset("b")]),
            ],
        )
        results = engine.classify(Signal({"x": 1}), short_circuit=False)
        assert len(results) == 2
        assert results[0].phase == "first"
        assert results[1].phase == "third"

    def test_short_circuit_false_no_matches_default(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[NeverMatchRuleset()])],
            default="none",
        )
        results = engine.classify(Signal({"x": 1}), short_circuit=False)
        assert len(results) == 1
        assert results[0].label == "none"


class TestImplicitMode:
    def test_rulesets_only(self):
        engine = ErrorSense(
            categories=["a"],
            rulesets=[AlwaysMatchRuleset("a")],
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "a"
        assert results[0].phase == "rulesets"

    def test_default_when_no_match(self):
        engine = ErrorSense(
            categories=["a"],
            rulesets=[NeverMatchRuleset()],
            default="none",
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "none"

    def test_cannot_mix_modes(self):
        with pytest.raises(ValueError, match="Cannot mix"):
            ErrorSense(
                categories=["a"],
                phases=[Phase("p1", rulesets=[AlwaysMatchRuleset("a")])],
                rulesets=[AlwaysMatchRuleset("a")],
            )

    def test_must_provide_something(self):
        with pytest.raises(ValueError, match="Must provide"):
            ErrorSense(categories=["a"])


class TestCallbacks:
    def test_on_classify_callback(self):
        collected = []
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[AlwaysMatchRuleset("a")])],
            on_classify=lambda sig, res: collected.append(res),
        )
        engine.classify(Signal({"x": 1}))
        assert len(collected) == 1
        assert collected[0].label == "a"

    def test_on_error_callback(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[
                Phase("broken", rulesets=[BrokenRuleset()]),
                Phase("ok", rulesets=[AlwaysMatchRuleset("a")]),
            ],
            on_error=lambda phase, err: None,
        )
        results = engine.classify(Signal({"x": 1}))
        assert results[0].label == "a"


class TestLabelValidation:
    def test_invalid_label_in_ruleset(self):
        with pytest.raises(ValueError, match="not in"):
            ErrorSense(
                categories=["a", "b"],
                phases=[Phase("p1", rulesets=[
                    Ruleset(field="x", match={1: "c"}),
                ])],
            )


class TestAsyncClassify:
    @pytest.mark.asyncio
    async def test_async_classify_first_catch(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[NeverMatchRuleset(), AlwaysMatchRuleset("a")])],
        )
        results = await engine.async_classify(Signal({"x": 1}))
        assert results[0].label == "a"

    @pytest.mark.asyncio
    async def test_async_classify_default(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[NeverMatchRuleset()])],
            default="fallback",
        )
        results = await engine.async_classify(Signal({"x": 1}))
        assert results[0].label == "fallback"

    @pytest.mark.asyncio
    async def test_async_classify_broken_skipped(self):
        engine = ErrorSense(
            categories=["a"],
            phases=[Phase("p1", rulesets=[BrokenRuleset(), AlwaysMatchRuleset("a")])],
        )
        results = await engine.async_classify(Signal({"x": 1}))
        assert results[0].label == "a"

    @pytest.mark.asyncio
    async def test_async_classify_all_phases(self):
        engine = ErrorSense(
            categories=["a", "b"],
            phases=[
                Phase("first", rulesets=[AlwaysMatchRuleset("a", confidence=0.5)]),
                Phase("second", rulesets=[AlwaysMatchRuleset("b", confidence=0.9)]),
            ],
        )
        results = await engine.async_classify(Signal({"x": 1}), short_circuit=False)
        assert len(results) == 2
        assert results[0].label == "a"
        assert results[1].label == "b"
