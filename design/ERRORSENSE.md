# ErrorSense — Error Classification Engine

## Problem

Every system that makes automated decisions about errors — circuit breakers, provider benching, alert routing, retry logic — relies on crude signals: HTTP status codes, exception types, or static rules. This causes false positives at scale.

**Real-world example (OpenGPU Relay):** A user sends a bad request (invalid model name). The provider returns 500. The circuit breaker counts it as a provider failure. After 3 bad user requests, the healthy provider is benched. The ops team gets a false alarm.

This pattern repeats everywhere:
- **API gateways** bench healthy backends because user errors look like server errors
- **Service meshes** open circuit breakers on request validation failures
- **Alert systems** page oncall for transient issues mixed with real outages
- **Retry logic** retries permanently broken requests because it can't tell them apart

**Existing tools don't solve this.** Python circuit breaker libraries ([pybreaker](https://github.com/danielfm/pybreaker), [circuitbreaker](https://pypi.org/project/circuitbreaker/)) only support manual exception exclusion lists. Spring AI has an [open issue](https://github.com/spring-projects/spring-ai/issues/3857) for the same problem — unresolved. Observability platforms (Datadog, Grafana) analyze errors after the fact in dashboards, not inline at decision time.

**ErrorSense fills this gap.** A general-purpose classification engine that takes any structured input (HTTP error, gRPC failure, exception, log entry) and classifies it into user-defined categories using a phase pipeline — from fast deterministic rulesets to AI-powered skills.

---

## Core Abstractions

### Signal

A `Signal` is the generic input to ErrorSense. An immutable, dict-like container that holds any structured data about an error or event. Deep-frozen at creation — rulesets and skills get a read-only view.

```python
from errorsense import Signal

# Raw dict — classify anything
signal = Signal({
    "error_code": "E001",
    "message": "disk full",
    "source": "storage",
    "severity": 3,
})

# Convenience adapters for common protocols
signal = Signal.from_http(
    status_code=500,
    body='{"error": "model not found"}',
    headers={"content-type": "application/json"},
)

signal = Signal.from_grpc(code=14, details="unavailable")
signal = Signal.from_exception(exc)

# Dict-like access
signal["status_code"]     # 500
signal["body"]            # '{"error": "model not found"}'
signal.get("custom")      # None — no crash on missing fields
```

**Design principle:** Signals are dumb containers. All intelligence lives in rulesets and skills.

---

### Ruleset

A `Ruleset` is any deterministic (non-LLM) classification logic. Each ruleset does **one thing**: either field matching (`match=`) or regex patterns (`patterns=`). Not both — pick one per ruleset, combine multiple rulesets in a phase.

For custom logic beyond what config can express, subclass `Ruleset` and override `classify()`.

```python
from errorsense import Ruleset

# Field value matching — exact lookup
Ruleset(field="status_code", match={400: "user", 502: "infra", 503: "infra"})

# HTTP status ranges — string keys ending in "xx" are ranges, int keys are exact overrides
Ruleset(field="status_code", match={"4xx": "user", "5xx": "server", 503: "infra"})

# Header matching
Ruleset(field="headers.content-type", match={"text/html": "infra", "application/json": None})

# JSON body field matching — dot notation resolves nested JSON
Ruleset(field="body.type", match={"invalid_request_error": "user", "server_error": "infra"})
Ruleset(field="body.error.type", match={"invalid_request_error": "user"})

# Regex patterns on a string field (separate ruleset — can't mix with match=)
Ruleset(field="body", patterns=[
    ("infra", [r"cuda", r"out of memory", r"connection refused"]),
    ("user",  [r"model.*not found", r"invalid.*param", r"unsupported"]),
], case_sensitive=False)

# Regex patterns on parsed JSON string fields
Ruleset(field="body.error.message", patterns=[
    ("user", [r"model.*not found", r"exceeds.*limit"]),
    ("infra", [r"cuda", r"out of memory"]),
])
```

**Custom rulesets (subclass for logic beyond config):**

```python
class VendorBugRuleset(Ruleset):
    def classify(self, signal: Signal) -> ClassifyResult | None:
        if signal.get("vendor") == "acme" and signal.get("code") == "X99":
            return ClassifyResult(
                category="known_bug",
                confidence=1.0,
                reason="ACME vendor bug X99 — documented, not our fault",
            )
        return None  # not confident, pass to next
```

**Construction validation — fails fast:**

```python
# These raise ValueError at construction time:
Ruleset(field="body", match={...}, patterns=[...])   # Can't use both match and patterns
Ruleset(field="body")                                 # Must provide match or patterns
Ruleset(field="status_code", match={400: "typo"})     # "typo" validated against categories at engine init
```

**How field resolution works:**

| Field pattern | Strategy | On failure |
|---|---|---|
| `"status_code"` | Exact lookup: `signal[field]` | Field missing → returns None |
| `"headers.content-type"` | Reads `signal["headers"]`, substring match on value | No headers → returns None |
| `"body.error.type"` | `json.loads(signal["body"])`, resolves dot path | Invalid JSON → logs warning, returns None |
| `"body"` (no dots) | Raw `signal["body"]` as string | Missing → returns None |

Field resolution never crashes. Missing fields and parse errors return None (pass to next ruleset) with a debug log.

**How match dicts work:**

| Key type | Behavior | Example |
|---|---|---|
| `int` key | Exact match | `{400: "user"}` — matches status_code == 400 |
| `str` ending in `"xx"` | Range match (HTTP-style) | `{"4xx": "user"}` — matches 400-499 |
| Other `str` key | Exact/substring match | `{"text/html": "infra"}` |
| `None` value | Matched but explicit pass | `{"application/json": None}` — stops this ruleset, returns None |

**Confidence values:** Match rulesets return 1.0 (deterministic). Pattern rulesets return 0.9 (heuristic). Custom subclass controls its own confidence.

---

### Skill

A `Skill` is domain-specific knowledge for the LLM — like [Claude Code skills](https://docs.anthropic.com/en/docs/claude-code). Each skill teaches the LLM how to classify errors in a specific domain. Each skill triggers a separate LLM call. Multiple skills in a phase = multiple expert opinions, highest confidence wins.

`Skill` is a frozen dataclass. Two required fields: `name` and `instructions`.

```python
from errorsense import Skill

# Direct instantiation (most common)
Skill(name="relay_errors", instructions="""
You classify errors from an AI inference relay that routes requests to GPU providers.
- 502/503 from a provider = infra (server/GPU failure)
- "model not found" in error message = user (asked for a model the provider doesn't have)
- CUDA OOM / connection refused = infra
- Ambiguous errors with no clear signal = provider
""")

# Reusable skill (subclass with fixed args)
class RelayErrors(Skill):
    def __init__(self):
        super().__init__(
            name="relay_errors",
            instructions="You classify errors from an AI inference relay...",
        )
```

**Full Skill constructor:**

```python
Skill(
    name="relay_errors",        # required
    instructions="...",          # required
    prompt_template=None,        # override default prompt (optional)
    temperature=0.0,             # LLM temperature (default: 0.0 for determinism)
    llm=None,                    # per-skill LLMConfig override (optional)
)
```

**LLM connection config:**

```python
from errorsense import LLMConfig

LLMConfig(
    api_key="relay_sk_...",
    model="gpt-oss:120b",                           # default
    base_url="https://relay.opengpu.network/v1",     # default (Relay)
    timeout=10.0,
    max_signal_size=500,
)
```

LLMConfig can be set at the **phase level** (shared by all skills) or **per-skill** (overrides phase default). This lets you run different skills on different models:

```python
Phase("llm", skills=[
    Skill(name="fast_check", instructions="...", llm=LLMConfig(api_key="...", model="gpt-oss:120b")),
    Skill(name="deep_analysis", instructions="...", llm=LLMConfig(api_key="...", model="claude-sonnet-4-6")),
], llm=LLMConfig(api_key="...", model="gpt-oss:120b"))  # phase default, overridden per-skill
```

- **Default provider: Relay.** Out of the box, LLM phases point to `relay.opengpu.network` (OpenAI-compatible API). Get a Relay API key and start classifying — no separate Anthropic/OpenAI account needed.
- **Model switching:** Single param: `model="claude-sonnet-4-6"`. All Relay models work.
- **Provider switching:** `base_url` override for non-Relay providers.
- **Sync classify() skips LLM phases with a warning.** LLM requires async. The sync `classify()` logs a warning and skips LLM phases. Use `aclassify()` for LLM-powered classification.
- **Cost optimization:** LLM phases should be last. Combined with Tracker's `llm_on_threshold`, LLM calls are rare — 5-20/day for a busy service. Rulesets handle 95%+ of classifications for free.
- **Fallback:** If an LLM call fails (timeout, rate limit, API error), that skill returns `None`. Never crashes the pipeline.
- **Getting started:** Sign up at [relay.opengpu.network](https://relay.opengpu.network) with code `ERRORSENSE` for bonus classification credits.

---

### Phase

A `Phase` is a named stage in the classification pipeline. Each phase contains either **rulesets** (deterministic logic) or **skills** (LLM calls) — not both. Users name phases whatever they want. The engine doesn't care.

```python
from errorsense import Phase, Ruleset, Skill, LLMConfig

# Ruleset phase — rulesets run in order, first match wins
Phase("rules", rulesets=[
    Ruleset(field="status_code", match={"4xx": "user", 502: "infra", 503: "infra"}),
    Ruleset(field="headers.content-type", match={"text/html": "infra"}),
])

# Another ruleset phase — regex patterns are rulesets too
Phase("patterns", rulesets=[
    Ruleset(field="body", patterns=[("infra", [r"cuda"]), ("user", [r"invalid"])]),
])

# LLM phase — skills run as separate LLM calls, highest confidence wins
Phase("llm", skills=[
    Skill(name="relay_errors", instructions="..."),
], llm=LLMConfig(api_key="relay_sk_..."))
```

**Validation:** A phase must have `rulesets` OR (`skills` + `llm`). Not both. Not neither.

---

### ErrorSense Engine

The engine runs signals through a phase pipeline. Two construction modes:

**Explicit mode (power users):** Define phases yourself.

```python
from errorsense import ErrorSense, Phase, Ruleset, Skill, LLMConfig

sense = ErrorSense(
    categories=["infra", "provider", "user"],
    phases=[
        Phase("rules", rulesets=[...]),
        Phase("patterns", rulesets=[...]),
        Phase("llm", skills=[...], llm=LLMConfig(...)),
    ],
    default="provider",
)
```

**Implicit mode (simple use):** Pass rulesets and skills directly. Engine builds phases for you.

```python
sense = ErrorSense(
    categories=["infra", "provider", "user"],
    rulesets=[
        Ruleset(field="status_code", match={400: "user", 502: "infra"}),
        Ruleset(field="body", patterns=[("infra", [r"cuda"])]),
    ],
    skills=[Skill(name="relay", instructions="...")],
    llm=LLMConfig(api_key="..."),
    default="provider",
)
# Builds implicit phases: "rulesets" → "llm"
```

**Classification modes:**

```python
signal = Signal.from_http(status_code=502, body="Bad Gateway")

# First-catch (default) — stops at first phase with a match
result = sense.classify(signal)
result.category     # "infra"
result.confidence   # 1.0
result.phase        # "rules"
result.reason       # "status_code=502 -> infra"

# All-phases — every phase runs, highest confidence wins
result = sense.classify_all(signal)
result.winner       # ClassifyResult(category="infra", confidence=1.0, phase="rules")
result.phases       # {"rules": ClassifyResult(...), "patterns": ClassifyResult(...), "llm": None}

# Skip phases at runtime
result = sense.classify(signal, skip=["llm"])

# Async — needed for LLM phases
result = await sense.aclassify(signal)
result = await sense.aclassify_all(signal)
```

**Validation (fails fast at setup time):**

- Rulesets: match values must be in `categories` or `default`. Raises `ValueError` if not.
- Phases: must have `rulesets` OR (`skills` + `llm`). Not both. Not neither. Not empty lists.
- Explicit/implicit: cannot mix `phases=` with `rulesets=/skills=/llm=`. Raises `ValueError`.
- `skip=`: phase names are validated against actual phase names. Unknown names raise `ValueError`.
- Duplicate phase names raise `ValueError`.

---

### Presets

Presets are standalone functions that return a pre-configured ErrorSense with sensible phase pipelines. One preset ships with v0.2: `http_gateway` — built for our Relay v1 use case.

```python
from errorsense.preset import http_gateway

# Use directly
sense = http_gateway(categories=["infra", "provider", "user"])

# With LLM
sense = http_gateway(
    categories=["infra", "provider", "user"],
    llm_api_key="relay_sk_...",
    llm_model="gpt-oss:120b",
)

# With custom rulesets
sense = http_gateway(
    categories=["infra", "provider", "user"],
    extra_rulesets=[VendorBugRuleset()],
    llm_api_key="relay_sk_...",
)
```

The `http_gateway` preset builds 3 phases:
- **rules:** Status codes (4xx→user, 502/503/504→infra), content-type (HTML→infra), JSON field matching
- **patterns:** Body regex, JSON message field regex
- **llm:** Relay skill with HTTP error domain knowledge (only if `llm_api_key` provided)

Each preset is a plain function — no magic, no hidden state. Read the source to see exactly what rulesets and skills are included.

---

### Tracker (Optional Stateful Layer)

The `Tracker` wraps an ErrorSense engine and adds stateful error tracking with threshold-based decisions. Use it when you need to answer "should I trip/bench/alert?" rather than just "what category is this error?"

```python
from errorsense import Tracker, Signal

tracker = Tracker(
    classifier=sense,
    threshold=3,                                  # errors before tripping
    count_categories=["infra", "provider"],        # only these count toward threshold
    history_size=10,                               # ring buffer per key
    llm_on_threshold=True,                         # defer LLM until about to trip
    async_mode=True,                               # use asyncio.Lock (False = threading.Lock)
)

# Record errors with a generic string key
signal = Signal.from_http(status_code=502, body="Bad Gateway")
result = tracker.record("provider-1:gpt-5", signal)

result.classified_as       # "infra"
result.at_threshold        # True — counted categories hit the threshold

# Trip decision — sync (no LLM reclassification)
decision = tracker.should_trip("provider-1:gpt-5")

# Trip decision — async (runs LLM reclassification if llm_on_threshold enabled)
decision = await tracker.ashould_trip("provider-1:gpt-5")

decision.trip              # True/False
decision.category_counts   # {"infra": 2, "provider": 1, "user": 3}
decision.history           # last N classified errors
decision.llm_analysis      # LLM explanation (if llm_on_threshold fired)
```

**Key design decisions:**

- **Generic keys (strings):** Not tied to "provider" or "model" concepts. Use `"host:port"`, `"service:endpoint"`, whatever your domain needs.
- **Configurable count_categories:** Only specified categories count toward the threshold. User errors are tracked in history (for visibility) but don't bring you closer to tripping.
- **LLM deferral (`llm_on_threshold`):** When enabled, LLM skills are deferred during normal `record()` calls. Only invoked via `ashould_trip()` when the threshold is about to be reached. The LLM receives the full history as context for reclassification.
- **History ring buffer:** Capped at `history_size` per key. Oldest entries are evicted. Counts stay in sync with the bounded history — evicted entries have their counts decremented.
- **Reset on success:** `tracker.reset(key)` clears the error history and counts for that key. Thread-safe (acquires per-key lock before clearing).
- **Thread-safe:** Per-key locking with a global guard lock to prevent lock-creation races. `reset_all()` holds the guard lock during clearing.

---

## Data Models

```python
@dataclass(frozen=True)
class ClassifyResult:
    """Result from classification — produced by rulesets or skills."""
    category: str          # one of the user-defined categories
    confidence: float      # 0.0 - 1.0
    reason: str            # human-readable explanation
    phase: str = ""        # which phase produced this (auto-filled by engine)
    skill_name: str = ""   # which ruleset or skill within the phase

@dataclass(frozen=True)
class ClassifyAllResult:
    """Result from classify_all — every phase reports."""
    winner: ClassifyResult                        # highest confidence across all phases
    phases: dict[str, ClassifyResult | None]      # phase_name → result or None

@dataclass(frozen=True)
class TrackResult:
    """Result from Tracker.record() — classification + threshold hint."""
    classified_as: str
    confidence: float
    reason: str
    phase: str
    skill_name: str
    at_threshold: bool

@dataclass(frozen=True)
class TrackDecision:
    """Result from Tracker.should_trip() — full decision context."""
    trip: bool
    category_counts: dict[str, int]
    history: list[dict[str, Any]]
    llm_analysis: str | None = None
```

---

## Usage Examples

### API Gateway (Relay)

```python
from errorsense import ErrorSense, Phase, Ruleset, Skill, LLMConfig, Signal, Tracker

relay_sense = ErrorSense(
    categories=["infra", "provider", "user"],
    phases=[
        Phase("rules", rulesets=[
            Ruleset(field="status_code", match={
                "4xx": "user", 502: "infra", 503: "infra", 504: "infra",
            }),
            Ruleset(field="headers.content-type", match={
                "text/html": "infra", "application/json": None,
            }),
            Ruleset(field="body.type", match={
                "invalid_request_error": "user",
            }),
        ]),
        Phase("patterns", rulesets=[
            Ruleset(field="body.error", patterns=[
                ("user", [r"model.*not found"]),
            ]),
            Ruleset(field="body", patterns=[
                ("infra", [r"cuda", r"oom", r"connection refused"]),
                ("user",  [r"unsupported", r"invalid"]),
            ]),
        ]),
        Phase("llm", skills=[
            Skill(name="relay_errors", instructions="""
                You classify errors from an AI inference relay.
                502/503 = infra, model not found = user, ambiguous = provider.
            """),
        ], llm=LLMConfig(api_key=RELAY_KEY)),
    ],
    default="provider",
)

# Or just use the preset:
# from errorsense.preset import http_gateway
# relay_sense = http_gateway(categories=["infra", "provider", "user"], llm_api_key=RELAY_KEY)

tracker = Tracker(
    classifier=relay_sense,
    threshold=3,
    count_categories=["infra", "provider"],
    llm_on_threshold=True,
)

# In the worker error path:
signal = Signal.from_http(status_code=response.status, body=error_text)
result = tracker.record(f"{provider_id}:{model}", signal)

if result.at_threshold:
    decision = await tracker.ashould_trip(f"{provider_id}:{model}")
    if decision.trip:
        await send_bench_alert(provider_id, model, decision)

# On success:
tracker.reset(f"{provider_id}:{model}")
```

### Database Error Monitoring

```python
from errorsense import ErrorSense, Phase, Ruleset, Signal

db_sense = ErrorSense(
    categories=["transient", "permanent", "config"],
    phases=[
        Phase("codes", rulesets=[
            Ruleset(field="error_code", match={
                "E11000": "config",
                "E13":    "config",
                "E50":    "transient",
                "E89":    "transient",
            }),
        ]),
        Phase("patterns", rulesets=[
            Ruleset(field="message", patterns=[
                ("permanent", [r"corruption", r"checksum mismatch", r"data too large"]),
                ("transient", [r"timeout", r"connection reset", r"retry"]),
            ]),
        ]),
    ],
    default="transient",
)

signal = Signal({"error_code": "E11000", "message": "duplicate key error"})
result = db_sense.classify(signal)
# result.category == "config", result.phase == "codes"
```

### Log Triage (Custom 4-Phase Pipeline)

```python
from errorsense import ErrorSense, Phase, Ruleset, Skill, LLMConfig, Signal

log_sense = ErrorSense(
    categories=["critical", "warning", "noise"],
    phases=[
        Phase("severity", rulesets=[
            Ruleset(field="level", match={
                "FATAL": "critical",
                "ERROR": "critical",
                "WARN": "warning",
            }),
        ]),
        Phase("keywords", rulesets=[
            Ruleset(field="message", patterns=[
                ("noise", [r"health check", r"keepalive", r"debug"]),
                ("critical", [r"OOM", r"segfault", r"panic"]),
            ]),
        ]),
        Phase("source", rulesets=[
            Ruleset(field="service", match={
                "payments": "critical",
                "analytics": "noise",
            }),
        ]),
        Phase("llm", skills=[
            Skill(name="log_triage", instructions="You triage log entries..."),
        ], llm=LLMConfig(api_key="relay_sk_...")),
    ],
)

signal = Signal({"level": "ERROR", "message": "connection reset by peer", "service": "payments"})

# First-catch
result = log_sense.classify(signal)

# All phases — see what each phase thinks
result = log_sense.classify_all(signal)
# result.winner → ClassifyResult(category="critical", confidence=1.0, phase="severity")
# result.phases → {"severity": ..., "keywords": None, "source": ..., "llm": ...}
```

### Using classify_all for Observability

```python
# Run all phases and log the full picture
result = sense.classify_all(signal)

for phase_name, phase_result in result.phases.items():
    if phase_result:
        logger.info(f"Phase {phase_name}: {phase_result.category} ({phase_result.confidence})")
    else:
        logger.info(f"Phase {phase_name}: no match")

logger.info(f"Winner: {result.winner.category} from phase {result.winner.phase}")
```

---

## Engine Design Details

### Phase Execution

```python
# First-catch mode (simplified)
for phase in self.phases:
    if phase.name in skip:
        continue
    result = phase.classify(signal)  # phase handles its own rulesets/skills
    if result is not None:
        return result

return default_result

# All-phases mode (simplified)
phase_results = {}
for phase in self.phases:
    if phase.name in skip:
        continue
    result = phase.classify(signal)
    phase_results[phase.name] = result  # None if no match

# Winner: highest confidence. Ties broken by phase order (earlier phase wins).
candidates = [(r, i) for i, (name, r) in enumerate(phase_results.items()) if r]
winner = min(candidates, key=lambda x: (-x[0].confidence, x[1]))[0] if candidates else default_result
return ClassifyAllResult(winner=winner, phases=phase_results)
```

**Within a ruleset phase:** Rulesets run in order. First confident result (non-None) wins. Remaining rulesets are skipped.

**Within an LLM phase:** All skills run concurrently (separate LLM calls). Each returns a classification or None. Highest confidence wins.

### Error Handling

Rulesets and skills must never crash the pipeline. The engine wraps every `classify()` call in try/except. Exceptions are logged and the component is skipped.

### Sync classify() and LLM Phases

When `classify()` (sync) encounters an LLM phase, it logs a warning and skips it. LLM requires async I/O. Use `aclassify()` for full pipeline execution including LLM. This is deliberate — sync callers get fast deterministic results, async callers get full classification.

### Engine Public API for Phase Access

The engine exposes `phases` as a read-only property and `get_phase(name)` for retrieving specific phases. The Tracker uses this to find LLM phases for reclassification via `ashould_trip()`.

### Ordering Guide

Order rulesets within a phase from fast + specific to slow + broad:

**Typical rules phase:** Exact field match (O(1)) → HTTP status range → header check → JSON field parse

**Typical patterns phase:** JSON message regex → raw body regex

**LLM phase:** Always last. In practice, 95%+ of signals are classified by rulesets. LLM is the fallback.

### Concurrency Model

**Engine (`ErrorSense`):** Stateless and thread-safe. Multiple threads/tasks can call `classify()` concurrently.

**Tracker:** Stateful — uses per-key locks (threading.Lock or asyncio.Lock based on `async_mode`). A global guard lock prevents lock-creation races. Different keys don't contend.

### Observability Hooks

```python
sense = ErrorSense(
    categories=[...],
    phases=[...],
    on_classify=lambda signal, result: metrics.inc("errorsense", tags={"category": result.category}),
    on_error=lambda phase, error: logger.warning(f"Phase {phase} failed: {error}"),
)

tracker = Tracker(
    classifier=sense,
    on_trip=lambda key, decision: send_alert(key, decision),
    on_record=lambda key, result: logger.info(f"Error for {key}: {result.category}"),
)
```

Hooks are fire-and-forget — exceptions caught and logged, never propagated.

### Signal Immutability

Signals are deeply frozen at creation via `MappingProxyType`. Nested dicts become `MappingProxyType`, lists become tuples. Rulesets and skills cannot mutate signal data.

---

## Library Structure

```
errorsense/
├── pyproject.toml              # Package config, zero required deps
├── README.md
├── LICENSE                     # MIT
├── errorsense/
│   ├── __init__.py             # Public API: ErrorSense, Phase, Ruleset, Skill, LLMConfig, Signal, Tracker
│   ├── engine.py               # ErrorSense engine (phase pipeline runner)
│   ├── signal.py               # Signal class (immutable dict + adapters)
│   ├── tracker.py              # Tracker (stateful threshold decisions)
│   ├── models.py               # ClassifyResult, ClassifyAllResult, TrackResult, TrackDecision
│   ├── ruleset.py              # Ruleset class (field match, patterns, custom subclass)
│   ├── skill.py                # Skill class (LLM domain instructions)
│   ├── phase.py                # Phase class (named container of rulesets or skills)
│   ├── llm.py                  # LLMConfig + LLMClient (async HTTP to LLM API)
│   └── preset.py               # http_gateway()
└── tests/
    ├── test_engine.py
    ├── test_signal.py
    ├── test_tracker.py
    ├── test_integration.py
    ├── test_ruleset.py
    ├── test_skill.py
    ├── test_phase.py
    ├── test_llm.py
    ├── test_classify_all.py
    └── fixtures/
        └── http_responses.json
```

## Dependencies

| Scope | Dependencies |
|-------|-------------|
| Core (`errorsense`) | **Zero** — stdlib only (`re`, `json`, `dataclasses`, `abc`) |
| LLM skills | `httpx` (async HTTP, no heavy SDK) |

```bash
pip install errorsense              # core only (no LLM)
pip install errorsense[llm]         # core + httpx for LLM skills (Relay is default)
```

**Quickstart with LLM (3 lines):**
```python
from errorsense.preset import http_gateway

sense = http_gateway(
    categories=["infra", "provider", "user"],
    llm_api_key="relay_sk_...",
)
```

---

## Competitive Landscape

| Tool | What it does | Gap ErrorSense fills |
|------|-------------|---------------------|
| **pybreaker** | Circuit breaker with manual exception exclusion list | No classification — binary fail/pass. Can't distinguish user vs infra errors. |
| **circuitbreaker** | Decorator-based circuit breaker | Same — binary fail/pass, no error analysis. |
| **Spring AI** | Java AI framework | [Issue #3857](https://github.com/spring-projects/spring-ai/issues/3857) open for this exact problem. Unresolved. |
| **Sentry / Datadog** | Error tracking + dashboarding | Post-hoc analysis. Not inline at decision time. |
| **PagerDuty / OpsGenie** | Alert routing | Routes alerts after creation. Doesn't classify the triggering error. |

**ErrorSense is the missing layer between "an error happened" and "here's what to do about it."**

## Target Persona

- **Platform / infra engineers** building API gateways, service meshes, or worker systems
- **SRE / DevOps teams** tired of false alarms from status-code-based monitoring
- **AI/ML platform teams** running multi-provider inference (the Relay use case)
- **Anyone with a "bad user request benches the healthy server" problem**

---

## Roadmap

| Version | Scope | Deliverable |
|---------|-------|-------------|
| **v0.1** (done) | Core engine + flat skill chain | ErrorSense, Signal, Skill, SkillResult, Tracker. Flat skill chain with first-match-wins. Built-in classifiers. HTTP preset. 99 tests. |
| **v0.2** | Ruleset + Skill + Phase pipeline | Ruleset (unified non-LLM classification), Skill (LLM domain instructions), Phase (named pipeline stages), ClassifyResult, classify_all(), skip=, implicit/explicit modes, http_gateway preset. |
| **v0.3** | Relay integration | Import into `workers/direct_worker.py`. Replace `is_benchable_error()`. Deploy to staging. |
| **v1.0** | PyPI publish | Package, README, presets, `errorsense[llm]` extra. |
| *Post-1.0* | *v1.1+* | *More presets (database_monitor, log_triage), Signal adapter improvements, community skill packs, `llm_on_threshold` proper deferral.* |

---

## Relay Integration Spec (v0.3)

### Files to modify in relay-v2

| File | Change |
|------|--------|
| `workers/direct_worker.py` | Import ErrorSense + Tracker. Replace `is_benchable_error()` with `tracker.record()`. Only `"infra"` and `"provider"` categories count toward benching. |
| `config.py` | Add `ERROR_CLASSIFIER_LLM_KEY`, `ERROR_CLASSIFIER_LLM_MODEL` settings |
| `requirements.txt` | Add `errorsense[llm]` |

### Enriched bench alert payload

```json
{
    "model": "gpt-oss:120b",
    "status": "benched",
    "category_counts": {"infra": 3, "user": 2},
    "error_history": [
        {"category": "infra", "confidence": 1.0, "phase": "rules", "status": 502, "timestamp": "..."},
        {"category": "user", "confidence": 0.9, "phase": "patterns", "status": 500, "timestamp": "..."},
        {"category": "infra", "confidence": 1.0, "phase": "rules", "status": 503, "timestamp": "..."}
    ],
    "llm_analysis": "3 of 5 errors are infrastructure failures. 2 are user errors. Benching is justified.",
    "timestamp": "2026-03-22T..."
}
```

### Verification

| Test | Expected |
|------|----------|
| Send bad request (invalid model) 5x | Provider NOT benched (user errors don't count) |
| Stop a provider container | 3 requests fail with 502/503 → provider IS benched |
| Check bench alert payload | `error_history` with categories + phases present |
| LLM call frequency | Only called via `ashould_trip()`, NOT on every error |
