# ErrorSense

Error classification engine for circuit breakers, benching decisions, alert routing, and retry logic.

**The problem:** A user sends a bad request (invalid model name). The provider returns 500. Your circuit breaker counts it as a provider failure. After 3 bad user requests, the healthy provider is benched. ErrorSense classifies errors *before* the decision so only real failures count.

## Install

```bash
pip install errorsense              # core only (zero dependencies)
pip install errorsense[llm]         # + LLM classification
```

## Quick Start

```python
from errorsense import ErrorSense, Phase, Ruleset, Signal

sense = ErrorSense(
    categories=["infra", "provider", "user"],
    phases=[
        Phase("rules", rulesets=[
            Ruleset(field="status_code", match={
                "4xx": "user", 502: "infra", 503: "infra", 504: "infra",
            }),
        ]),
        Phase("patterns", rulesets=[
            Ruleset(field="body", patterns=[
                ("infra", [r"cuda", r"out of memory", r"connection refused"]),
                ("user",  [r"model.*not found", r"invalid", r"unsupported"]),
            ]),
        ]),
    ],
    default="provider",
)

signal = Signal.from_http(status_code=503, body="CUDA out of memory")
results = sense.classify(signal)
# results[0].label      -> "infra"
# results[0].confidence -> 0.9
# results[0].phase      -> "patterns"
```

## Or use a preset

```python
from errorsense.presets import http, http_no_llm
from errorsense import LLMConfig

# With LLM — handles ambiguous errors (recommended)
sense = http(llm=LLMConfig(api_key="your_api_key"))

# Without LLM — only classifies clear-cut cases
sense = http_no_llm()

results = sense.classify(Signal.from_http(status_code=400, body="bad request"))
# results[0].label -> "client"
```

## Phase Pipeline

Classification runs through named phases in order. First phase to match wins (`short_circuit=True`, the default).

**Rulesets** = deterministic logic (field matching, regex patterns):

```python
Ruleset(field="status_code", match={400: "user", 502: "infra"})
Ruleset(field="status_code", match={"4xx": "user", 503: "infra"})
Ruleset(field="headers.content-type", match={"text/html": "infra"})
Ruleset(field="body.error.type", match={"invalid_request_error": "user"})
Ruleset(field="body", patterns=[("infra", [r"cuda", r"OOM"]), ("user", [r"invalid"])])
```

**Skills** = LLM domain instructions (loaded from .md files in errorsense/skills/):

```python
from errorsense import Skill, LLMConfig

Phase("llm", skills=[
    Skill(name="my_classifier", instructions="Classify this error..."),
], llm=LLMConfig(api_key="..."))
```

95%+ of errors resolve at the ruleset level. LLM is a last resort.

## All Phases Mode

```python
# Default — stops at first match
results = sense.classify(signal)
# [SenseResult(label="infra", phase="rules", ...)]

# All phases — every phase runs
results = sense.classify(signal, short_circuit=False)
# [SenseResult(label="infra", phase="rules", ...), SenseResult(label="infra", phase="patterns", ...)]

# With LLM reasoning
results = sense.classify(signal, short_circuit=False, explain=True)
# LLM results include .reason field
```

## Tracker (Stateful Decisions)

For "should I bench this provider?" decisions:

```python
from errorsense import Tracker, Signal

tracker = Tracker(
    classifier=sense,
    threshold=3,
    count_labels=["infra", "provider"],  # user errors don't count
)

signal = Signal.from_http(status_code=502, body="Bad Gateway")
result = tracker.record("provider-1:gpt-5", signal)

if result.at_threshold:
    decision = await tracker.async_should_trip("provider-1:gpt-5")
    if decision.trip:
        bench_provider(decision)

tracker.reset("provider-1:gpt-5")
```

## Custom Rulesets

```python
from errorsense import Ruleset, Signal, SenseResult

class VendorBugRuleset(Ruleset):
    def classify(self, signal: Signal) -> SenseResult | None:
        if signal.get("vendor") == "acme" and signal.get("code") == "X99":
            return SenseResult(label="known_bug", confidence=1.0)
        return None
```

## License

MIT
