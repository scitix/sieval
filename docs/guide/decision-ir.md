# Decision IR

SiEval defines a provider-neutral vocabulary for typed decision questions and
answers. The shared infrastructure includes evidence, capability requirements,
request and response validation, and persisted model-call metadata.

**No built-in dialect supports decision requests yet.** OpenRouter Decisions
will be implemented separately. Native OpenAI, SystemOne and Google decision
adapters and production benchmarks are outside this change.

## Evidence and questions

`ContextInput` holds ordered evidence without assigning conversation roles or
tool-call semantics. Its initial `ContextPart` union is `TextPart | JsonPart`;
the chat `ContentPart` union is unchanged. Context evidence is separate from the
existing `ChatInput` and `CompletionInput` input kinds.
Decision requests may also carry `ChatInput` when a dialect supports that
evidence form; completion-continuation semantics are incompatible with decisions.

```python
from sieval.core.models import (
    ChoiceOption,
    ChoiceQuestion,
    ContextInput,
    DecisionParams,
    JsonPart,
    PredicateQuestion,
    Request,
    ScoreQuestion,
    TextPart,
)

request = Request(
    input=ContextInput((
        TextPart("Route this customer support ticket."),
        JsonPart.from_value({
            "description": "The item arrived damaged.",
            "order_id": "123",
            "previous_ticket": None,
        }),
    )),
    decisions=DecisionParams((
        PredicateQuestion("damaged", "Does the ticket report product damage?"),
        ChoiceQuestion(
            "team",
            "Which team should handle this ticket?",
            (ChoiceOption("support"), ChoiceOption("billing")),
        ),
        ScoreQuestion(
            "urgency",
            "How urgent is the ticket?",
            ("Routine", "Respond soon", "Respond today"),
        ),
    )),
)
```

Question IDs are stable correlation identifiers. A dialect maps native answer
identities back to them and retains the submitted question order. Choice values
are strings, unique within a question. Their submission order does not prescribe
a provider's scoring order. Score levels form an ordered, zero-based rubric.

Question instructions are text. Optional predicate and choice guidance can be
text or structured JSON. Score levels and returned legend entries can also be
text or structured JSON, but cannot be null.

`JsonPart.from_value()` accepts objects, arrays and explicit null. Standalone
context evidence must be an object or array; explicit null is meaningful for
optional guidance. An omitted guidance field (`None`) and present JSON null
(`JsonPart.from_value(None)`) are distinct. Scalars are not coerced into text or
choice values.

The `json` field is a validated storage representation. `.value` returns a fresh
structured value for native lowering. This preserves nested JSON null through
SiEval's existing serializer, which omits `None` dictionary entries. It does not
authorize a dialect to turn structured evidence into a prompt string.

## Answers and validation

Decision answers use the `Response.decisions` channel. `Response.texts` is empty;
the dialect must not invent placeholder generation strings.

```python
from sieval.core.models import (
    ChoiceAnswer,
    DecisionResult,
    PredicateAnswer,
    RefusalAnswer,
    Response,
)

response = Response(
    texts=(),
    decisions=DecisionResult((
        PredicateAnswer("damaged", probability=0.92),
        ChoiceAnswer("team", choice="support"),
        RefusalAnswer("urgency"),
    )),
)
```

| Question | Successful answer | Meaning |
| --- | --- | --- |
| Predicate | `PredicateAnswer` | Native probability that the predicate holds |
| Choice | `ChoiceAnswer` | A requested choice, with an optional distribution |
| Score | `ScoreAnswer` | Native ordinal score, with optional distribution and legend |
| Any | `RefusalAnswer` | Refusal for that question, without fabricated values |

A score can be fractional: `1.8` on a three-level rubric stays `1.8`. Confidence
is optional and is not derived from the maximum probability. Probability mass
is not normalized and scores are not reconstructed from distributions.

The shared output contract checks question identity, order and answer type;
choice membership; rubric bounds; finite numbers; probabilities and confidence
in `[0, 1]`; and requested distribution completeness. Set
`DecisionParams.require_probabilities` or `require_confidence` when the task
depends on those outputs. A per-question refusal remains valid when either is
required. A request-level block must fail the request, rather than synthesizing
per-question refusals.

The initial contract requires `sampling.n == 1` and rejects streaming. Generic
generation controls and other optional channels remain subject to the dialect's
normal capability checks and no-silent-drop request audit.

When reported, `Response.finish_reasons` contains one call-level reason for a
decision response, and `finish_reason_sources` has the same cardinality. Those
records feed the existing truncation detector and report counts without requiring
placeholder texts. A task tagged `decisions` participates in truncation and
context-overflow detection and receives `n_truncated` / `n_scored_rollouts` report
counts. It does not need a `gen` tag. Missing native finish information remains
absent; if recorded metadata is unavailable after resume, the counts are omitted
rather than reported as zero. These counters are also included in pre-run metric
declarations when the task's report metrics are known.

Tasks declaring `requires.decisions` must have a `gen` or `decisions` diagnostic
tag; construction fails if neither is present. `@sieval_task(eval_mode=EvalMode.GEN,
...)` already adds `gen`. An undecorated decision task can set
`tags = frozenset({"decisions"})` on its class.

## Task and model capabilities

Tasks declare the `decisions` capability with `DecisionsOptions`:

```python
from sieval.core.models import DecisionsOptions
from sieval.core.models.requirements import InputKind, TaskRequirements

requires = TaskRequirements(
    input=InputKind.CONTEXT,
    decisions=DecisionsOptions(
        question_types=("predicate", "choice"),
        evidence_kinds=("json",),
        require_probabilities=True,
    ),
)
```

The declared sets name required semantics, rather than an exhaustive list of
provider features. Empty sets add no form restriction. Supported values are:

- `question_types`: `predicate`, `choice`, `score`.
- `evidence_kinds`: `text`, `json` for context evidence.
- `guidance_kinds`: `text`, `json`, `null`.
- `require_probabilities`, `require_confidence`: explicit output guarantees.

Requirements are unioned across tasks sharing a model; boolean guarantees are
joined with OR, and task sources survive into reconciliation diagnostics.
Capability declarations cannot omit a task-required form. A future dialect
binding must verify these requirements against its native semantics.

The form sets are serving floors, not per-request allowlists. For example,
`question_types=("predicate",)` requires a deployment that supports predicates;
it does not prohibit a choice request when that dialect and model support it.
Each concrete request still undergoes the dialect's validation and audit.

`Task` applies probability/confidence requirements to a task-local view of its
candidate model. This view projects the flags onto every decision request before
lowering and validates the returned answers against them, even when `infer`
omits the flags. Other tasks sharing the binding are unaffected. Explicit
`DecisionParams` flags can strengthen a call's requirements. A directly used or
auxiliary model can be given the same view with
`model.with_decision_requirements(options)`.

Model configuration can require the same outputs across the entire binding:

```yaml
capabilities:
  decisions:
    require_probabilities: true
    require_confidence: true
```

These configuration flags apply to every decision call through that binding,
including calls without a `Task`. Configuration, task-local and explicit request
flags are combined with OR before native lowering and answer validation. Omitted
or false flags do not weaken a requirement from another source. Rebinding reads
the new plan's configuration while preserving the view's task-local guarantees.
Aggregated task guarantees in `capability_minimums` remain serving requirements
and do not strengthen unrelated consumers' requests.

Decision capability form sets are sorted during normalization, so changing only
their declaration order does not change a binding's fingerprint. Question and
choice submission order is retained independently of those capability sets.

Context consumers bind the canonical `Model`. Their config omits the legacy
`type: chat/gen` assertion and supplies an explicit dialect. Automatic generation
serving recipes do not apply. All eight existing dialects declare decisions
unsupported, so requests and capability declarations are rejected before I/O.

## Records and metadata

`Request` and its input containers remain runtime types. To make preprocessing
auditable and reloadable, return explicit records such as:

```python
preprocessed = {
    "evidence": JsonPart.from_value({"ticket": "damaged item", "history": None}),
    "questions": request.decisions,
}
```

Decision question and answer records are registered with `@sieval_record`.
Persist the canonical response without projecting it into legacy `ModelOutput`:

```python
from sieval.core.tasks import TaskStageOutput

response = await model.arun(request)
stage_output = TaskStageOutput(value=response)
```

A following stage reads `stage_output.value`. The runner automatically records
model-call metadata for decision responses, whether returned directly or inside
`TaskStageOutput`. It also handles lists of calls, including lists mixing decision
responses with legacy `ModelOutput`. Finish information, usage, recorded request
parameters and authoritative response provenance feed the existing profiler and
report counts without requiring a task to supply meta.

Automatic metadata uses the candidate model's `meta()` as the fallback. When the
stage calls another model or needs explicit metadata, use
`build_model_call_meta_from_response(actual_model.meta(), response)` and
`build_stage_meta(model_calls=[...])`. Metadata supplied in `TaskStageOutput.meta`
overrides automatic metadata; a configured stage hook runs after both.

Setting `model_calls` in explicit metadata or a stage hook replaces the entire
automatic call list, following the same override contract as generation tasks.
Calls are neither appended nor merged field by field, and missing fields are not
recovered from automatic metadata. Explicit records may change usage and finish
values; the runner does not compare them with the returned response.

The profiler and report counts read the replacement list. To retain usage and
finish diagnostics, include `usage`, `finish_reasons` and `finish_reason_sources`
in its records. The response metadata helper supplies the fields the provider
reported. A shortened entry containing only `model` drops the automatic usage
and finish information, which can remove token costs and make finish counts
appear zero. Changes to this shared override contract are outside this extension.

To add custom metadata while retaining automatic call records, set `extra` or
`env` and omit `model_calls`.

Unreported usage breakdowns stay absent; reported zero stays zero. These calls
use the existing append-only sharded records. Response provenance and typed
decisions also survive when `record_meta=False`.

This extension keeps the result directory layout and existing generation
semantics unchanged. `RESULT_SCHEMA_VERSION` remains 1. An unused decisions
capability does not change existing binding fingerprints or relax resume checks.
Unavailable capability diagnostics remain complete in memory. Serialized plans
keep the original IR's historical rows for fingerprint compatibility; unused
capability extensions are omitted unless declared or required. This compatibility
baseline is frozen when new capability keys are added.

## Follow-up dialect work

The OpenRouter PR must add its descriptor, executable binder, native capability
binding, lossless request lowering and answer normalization. It must verify the
HTTP envelope, request-level errors, optional native outputs, and real API
behavior independently of these shared tests.

`JsonPart` protects typed evidence and guidance records. It does not change the
existing `Response.request_params` dictionary serialization: explicit null in a
native wire snapshot needs a deliberate persistence representation in the adapter
work. The foundation tests do not establish native wire round-trip fidelity.

A production evaluation task and dataset can follow the adapter. The integration
test here uses a test dialect seam to verify disk records and usage accounting.
