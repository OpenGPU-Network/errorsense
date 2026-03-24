# ErrorSense — Design Document

Error classification engine. Rules for the obvious, LLM for the ambiguous.

---

## Problem

Every system that makes automated decisions about errors — circuit breakers, alert routing, retry logic — relies on crude signals: HTTP status codes, exception types, or static rules. This causes false positives at scale.

A user sends a bad request. The server returns 500. Your circuit breaker counts it as a server failure. After a few bad user requests, the healthy server is tripped. ErrorSense classifies errors *before* the decision so only real failures count.

**Existing tools don't solve this.** Circuit breaker libraries ([pybreaker](https://github.com/danielfm/pybreaker), [circuitbreaker](https://pypi.org/project/circuitbreaker/)) only support manual exception exclusion lists. Observability platforms (Datadog, Grafana) analyze errors after the fact, not inline at decision time.

**ErrorSense fills this gap.** A classification engine that runs errors through a pipeline of fast deterministic rulesets, falling back to LLM only when rulesets can't decide.

---

## Core Abstractions

### Signal

Immutable, dict-like container for error data. Deep-frozen at creation via `MappingProxyType`.

```python
from errorsense import Signal

Signal({"error_code": "E001", "message": "disk full"})
Signal.from_http(status_code=500, body='{"error": "model not found"}', headers={...})
Signal.from_grpc(code=14, details="unavailable")
Signal.from_exception(exc)
```

### Ruleset

Deterministic classification logic. Each ruleset does one thing: `match=` (field matching) or `patterns=` (regex). Not both.

```python
from errorsense import Ruleset

Ruleset(field="status_code", match={400: "client", 502: "server"})
Ruleset(field="status_code", match={"4xx": "client", 503: "server"})
Ruleset(field="headers.content-type", match={"text/html": "server"})
Ruleset(field="body.error.type", match={"validation_error": "client"})
Ruleset(field="body", patterns=[("server", [r"OOM"]), ("client", [r"invalid"])])
```

**Field resolution:** Plain field → `signal[field]`. `headers.*` → header lookup. `body.*` → JSON parse + dot-path. `body` (no dots) → raw string.

**Confidence:** Match = 1.0. Pattern = 0.9. Custom subclass sets its own.

**Custom logic:** Subclass and override `classify()`:

```python
class VendorBugRuleset(Ruleset):
    def classify(self, signal: Signal) -> SenseResult | None:
        if signal.get("vendor") == "acme":
            return SenseResult(label="known_bug", confidence=1.0)
        return None
```

### Skill

LLM domain instructions loaded from `.md` files. Each skill teaches the LLM how to classify errors in a specific domain. Each skill triggers a separate LLM call. Multiple skills = multiple opinions, highest confidence wins.

```python
from errorsense import Skill

Skill("http_classifier")                              # loads errorsense/skills/http_classifier.md
Skill("my_classifier", path="./skills/custom.md")     # loads from explicit path
Skill("inline", instructions="Classify this error...")  # inline (for programmatic use)
```

**LLMConfig:** Connection config for the LLM API. Set at phase level (shared) or per-skill (override).

```python
from errorsense import LLMConfig

LLMConfig(
    api_key="your_key",
    model="gpt-oss:120b",
    base_url="https://relay.opengpu.network/v1",
    timeout=10.0,
    max_signal_size=500,
)
```

### Phase

Named stage in the pipeline. Contains rulesets OR skills — not both.

```python
from errorsense import Phase

Phase("rules", rulesets=[Ruleset(...), Ruleset(...)])
Phase("llm", skills=[Skill(...)], llm=LLMConfig(...))
```

### ErrorSense Engine

Runs signals through a pipeline of phases.

```python
from errorsense import ErrorSense, Phase, Ruleset, Skill, LLMConfig

sense = ErrorSense(
    labels=["client", "server", "undecided"],
    pipeline=[
        Phase("rules", rulesets=[...]),
        Phase("patterns", rulesets=[...]),
        Phase("llm", skills=[...], llm=LLMConfig(...)),
    ],
    default="undecided",
)
```

**Implicit mode** (no Phase objects needed):

```python
sense = ErrorSense(
    labels=["client", "server"],
    rulesets=[Ruleset(...)],
    skills=[Skill(...)],
    llm=LLMConfig(...),
)
```

**Classification:**

```python
results = sense.classify(signal)                           # sync, first match
results = sense.classify(signal, short_circuit=False)      # all phases run
results = sense.classify(signal, explain=True)             # LLM includes reasoning
results = sense.classify(signal, skip=["llm"])             # skip specific phases
results = await sense.async_classify(signal)               # async (concurrent LLM skills)
```

Returns `list[SenseResult]`. Each result has `.label`, `.confidence`, `.phase`, `.skill_name`. If `explain=True`, LLM results include `.reason`.

Both `classify()` and `async_classify()` run the full pipeline including LLM. The sync version uses `httpx.Client`, async uses `httpx.AsyncClient`.

### Presets

Opinionated pre-configured pipelines.

```python
from errorsense.presets import http, http_no_llm

sense = http(llm=LLMConfig(api_key="..."))   # rulesets + LLM, "client"/"server"/"undecided"
sense = http_no_llm()                         # rulesets only
```

Skills for presets live as `.md` files in `errorsense/skills/`.

### Trailing (Stateful Error Tracking)

Track errors per key with threshold-based LLM review.

```python
from errorsense import TrailingConfig

sense = ErrorSense(
    labels=["client", "server", "undecided"],
    pipeline=[...],
    trailing=TrailingConfig(
        threshold=3,
        count_labels=["server"],
        history_size=10,
        reviewer_llm=LLMConfig(),          # enables LLM review at threshold
        reviewer_skill=Skill("custom"),     # optional, defaults to built-in reclassification.md
    ),
)

result = sense.trail("service-a", signal)
result.label         # "server"
result.at_threshold  # True
result.reason        # LLM review explanation (or None)

sense.reset("service-a")
```

**How trailing works:**

1. Each `trail()` call classifies the signal normally through the pipeline
2. If the label is in `count_labels`, increment that key's count
3. At threshold, the LLM reviews all recorded errors for that key
4. If the review changes the label, the latest history entry is corrected and counts adjust
5. `at_threshold` recalculates after any correction

**Review behavior:**
- `reviewer_llm=LLMConfig(...)`: LLM reviews error history at threshold
- `reviewer_llm=None` (default): no review, just count
- `reviewer_skill=Skill(...)`: override the default review instructions

**Manual review:** `sense.review(key)` / `await sense.async_review(key)` — LLM reviews full history anytime.

**Thread safety:** Per-key locking (sync: `threading.Lock`, async: `asyncio.Lock`) with global guard lock. Different keys don't contend.

---

## Data Models

```python
@dataclass(frozen=True)
class SenseResult:
    label: str
    confidence: float
    phase: str = ""
    skill_name: str = ""
    reason: str | None = None   # only set when explain=True or LLM review

@dataclass(frozen=True)
class TrailResult:
    label: str
    confidence: float
    phase: str
    skill_name: str
    at_threshold: bool
    reason: str | None = None   # LLM review explanation

@dataclass(frozen=True)
class TrailingConfig:
    threshold: int = 3
    count_labels: list[str] | None = None
    history_size: int = 10
    reviewer_llm: LLMConfig | None = None
    reviewer_skill: Skill | None = None
```

---

## Engine Design Details

### Pipeline Execution

```python
# First match (short_circuit=True, default)
for phase in pipeline:
    if phase.name in skip: continue
    result = phase.classify(signal)
    if result is not None:
        results.append(result)
        break

# All phases (short_circuit=False)
for phase in pipeline:
    if phase.name in skip: continue
    result = phase.classify(signal)
    if result is not None:
        results.append(result)
```

Returns `list[SenseResult]`. Empty list → default result appended.

**Within a ruleset phase:** Rulesets run in order, first match wins.

**Within an LLM phase:** Sync — skills run sequentially. Async — skills run concurrently via `asyncio.gather`. Highest confidence wins.

### Error Handling

Every `classify()` call is wrapped in try/except at the phase level. Exceptions are logged, component is skipped. Pipeline never crashes.

### Validation (at construction)

- Labels in rulesets must be in `labels` or `default`
- Phases must have rulesets OR (skills + llm)
- LLM phases must have an API key in LLMConfig
- `pipeline=` and `rulesets=/skills=` cannot be mixed
- Duplicate phase names rejected
- `skip=` validated against actual phase names
- `TrailingConfig(review=True)` requires an LLM phase

### Signal Immutability

Signals are deeply frozen at creation via `MappingProxyType`. Nested dicts become `MappingProxyType`, lists become tuples. Rulesets and skills cannot mutate signal data.

### Observability

```python
sense = ErrorSense(
    pipeline=[...],
    on_classify=lambda signal, result: log(result),
    on_error=lambda phase, error: alert(phase, error),
)
```

Callbacks are fire-and-forget — exceptions caught and logged.

---

## Library Structure

```
errorsense/
├── __init__.py             # Public API exports
├── engine.py               # ErrorSense (pipeline + trailing)
├── signal.py               # Signal (immutable container)
├── models.py               # SenseResult, TrailResult, TrailingConfig
├── ruleset.py              # Ruleset (field match, patterns, custom subclass)
├── skill.py                # Skill (loads .md files for LLM instructions)
├── llm.py                  # LLMConfig + LLMClient (sync + async HTTP)
├── phase.py                # Phase (named pipeline stage)
├── presets/
│   ├── __init__.py
│   └── http_gateway.py     # http(), http_no_llm()
└── skills/
    ├── http_classifier.md  # HTTP error classification instructions
    └── reclassification.md # Trailing review instructions
```

## Dependencies

| Scope | Dependencies |
|-------|-------------|
| Core | **Zero** — stdlib only |
| LLM skills | `httpx` |

```bash
pip install errorsense
pip install errorsense[llm]
```

---

## Competitive Landscape

| Tool | What it does | Gap ErrorSense fills |
|------|-------------|---------------------|
| **pybreaker** | Circuit breaker, manual exception list | No classification — binary fail/pass |
| **circuitbreaker** | Decorator circuit breaker | Same — binary, no error analysis |
| **Sentry / Datadog** | Error tracking + dashboards | Post-hoc, not inline at decision time |
| **PagerDuty** | Alert routing | Routes after creation, doesn't classify |

---

## Roadmap

| Version | Deliverable |
|---------|-------------|
| **v0.1** (done) | Core engine, Signal, Ruleset, Skill, Phase, ErrorSense, trailing, presets, 81 tests, PyPI publish |
| **v0.2** | Relay integration, more presets, community skill packs |
