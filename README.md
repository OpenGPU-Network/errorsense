# ErrorSense

Error classification engine. Rules for the obvious, LLM for the ambiguous.

Most errors are easy to classify — a 400 is a client error, a 502 is a server error. But some aren't — a 500 with "model not found" in the body is actually a client error, not a server failure. Your rules can't catch every edge case. An LLM can.

ErrorSense runs errors through a phase pipeline: fast deterministic rulesets first, LLM only when rulesets can't decide. Most errors never hit the LLM. The ones that do get classified correctly instead of falling through as "unknown."

**Use it for:** circuit breakers, alert routing, retry logic, error dashboards; anywhere you need to know *what kind* of error happened, not just *that* it happened.

## Install

```bash
pip install errorsense              # core only (zero dependencies)
pip install errorsense[llm]         # + LLM classification
```

## Quick Start — Use a Preset

```python
from errorsense.presets import http
from errorsense import LLMConfig, Signal

sense = http(llm=LLMConfig(api_key="your_api_key"))

results = sense.classify(Signal.from_http(status_code=400, body="bad request"))
results[0].label  # "client"

results = sense.classify(Signal.from_http(status_code=502))
results[0].label  # "server"

results = sense.classify(Signal.from_http(status_code=500, body="model not found"))
results[0].label  # "client" (LLM figured it out)
```

The `http` preset gives you a 3-phase pipeline (rules → patterns → LLM) with 3 labels: `"client"`, `"server"`, `"undecided"`. Rulesets handle obvious cases instantly. LLM handles the ambiguous ones.

Don't want LLM? Use `http_no_llm()` — rulesets only, ambiguous errors come back as `"undecided"`.

## Build Your Own Pipeline

A pipeline is a list of phases. Each phase has rulesets (deterministic) or skills (LLM). You can mix both, use only rulesets, or use only skills.

```python
from errorsense import ErrorSense, Phase, Ruleset, Skill, LLMConfig, Signal

# Rulesets + LLM
sense = ErrorSense(
    labels=["transient", "permanent", "user"],
    pipeline=[
        Phase("codes", rulesets=[
            Ruleset(field="error_code", match={
                "ECONNRESET": "transient", "ETIMEOUT": "transient", "EPERM": "permanent",
            }),
        ]),
        Phase("patterns", rulesets=[
            Ruleset(field="message", patterns=[
                ("transient", [r"timeout", r"connection reset", r"retry"]),
                ("permanent", [r"corruption", r"fatal"]),
            ]),
        ]),
        Phase("llm", skills=[
            Skill("my_classifier", path="./skills/my_classifier.md"),
        ], llm=LLMConfig(api_key="your_key")),
    ],
    default="transient",
)

# Rulesets only — no LLM needed
sense = ErrorSense(
    labels=["client", "server"],
    pipeline=[
        Phase("rules", rulesets=[
            Ruleset(field="status_code", match={"4xx": "client", 502: "server"}),
        ]),
    ],
    default="server",
)

# LLM only — skip rulesets entirely
sense = ErrorSense(
    labels=["client", "server"],
    pipeline=[
        Phase("llm", skills=[
            Skill("my_classifier", path="./skills/my_classifier.md"),
        ], llm=LLMConfig(api_key="your_key")),
    ],
    default="unknown",
)
```

Phases run in order. First match wins. Rulesets are instant and free. LLM is the fallback.

## Rulesets

Each ruleset does one thing — `match=` for exact field matching or `patterns=` for regex:

```python
Ruleset(field="status_code", match={400: "client", 502: "server"})         # exact match
Ruleset(field="status_code", match={"4xx": "client", 503: "server"})       # range match
Ruleset(field="body.error.type", match={"validation_error": "client"})     # JSON dot-path
Ruleset(field="headers.content-type", patterns=[("server", [r"^text/html"])])  # regex
Ruleset(field="body", patterns=[("server", [r"OOM"]), ("client", [r"invalid"])])  # regex
```

Custom logic? Subclass:

```python
class VendorBugRuleset(Ruleset):
    def classify(self, signal: Signal) -> SenseResult | None:
        if signal.get("vendor") == "acme" and signal.get("code") == "X99":
            return SenseResult(label="known_bug", confidence=1.0)
        return None
```

## Skills

Skills are LLM instructions stored as `.md` files. Each skill teaches the LLM how to classify errors in a specific domain. Each skill triggers a separate LLM call — highest confidence result wins.

```python
# Loads from errorsense/skills/http_classifier.md (built-in)
Skill("http_classifier")

# Loads from your own file
Skill("my_classifier", path="./skills/my_classifier.md")
```

**Multiple skills in one phase:** Use this when you want multiple domain-specific opinions on the same error.

```python
Phase("llm", skills=[
    Skill("http_classifier"),    # knows HTTP error patterns
    Skill("db_classifier"),      # knows database error patterns
], llm=LLMConfig(...))
```

In sync (`classify`), skills run sequentially. In async (`async_classify`), skills run concurrently.

## All Phases Mode

```python
# Default — stops at first match
results = sense.classify(signal)

# All phases run
results = sense.classify(signal, short_circuit=False)

# With LLM reasoning
results = sense.classify(signal, explain=True)
results[0].reason  # "ECONNRESET indicates transient network failure"
```

## Trailing (Stateful Error Tracking)

Track errors per key. When a threshold is hit, optionally have an LLM review the full error history.

```python
from errorsense import LLMConfig, TrailingConfig

# With LLM review at threshold
sense = ErrorSense(
    labels=["transient", "permanent", "user"],
    pipeline=[...],
    trailing=TrailingConfig(
        threshold=3,
        count_labels=["transient", "permanent"],  # user errors don't count
        reviewer_llm=LLMConfig(),                 # enables LLM review
    ),
)

# Without LLM review (just counting)
trailing=TrailingConfig(threshold=3, count_labels=["transient", "permanent"])

# In your error handler:
result = sense.trail("service-a", signal)
result.label         # "transient"
result.at_threshold  # True (3rd counted error)
result.reason        # LLM review: "3 transient errors — all connection resets..."

# On success:
sense.reset("service-a")
```

**How it works:**
- Each `trail()` call classifies the signal normally through the pipeline
- Counted labels accumulate per key toward the threshold
- At threshold, the LLM reviews all recorded errors (if `reviewer_llm` is set)
- If the review changes the label, the history entry is corrected and the count adjusts
- `reviewer_skill=Skill(...)` lets you override the default review instructions

**Manual review anytime:**

```python
verdict = sense.review("service-a")
verdict.label   # LLM's verdict on the full history
verdict.reason  # explanation
```

## License

MIT
