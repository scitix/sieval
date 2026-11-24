# SiEval Decisions plugin

A standalone plugin for evaluating typed Decisions APIs through SiEval's
`EvalSubject` interface. Multiple benchmarks share the same OpenRouter client;
provider field translation stays outside `sieval/core`.

## Install

Requires Python 3.12+ and Git. Until SiEval releases `EvalSubject` ([#301](https://github.com/scitix/sieval/pull/301)),
the dependency is pinned to its merge commit [`607cfde`](https://github.com/scitix/sieval/commit/607cfded82fca553716b4e92215c6c423926ab83).
SiEval 0.17.0 alone predates this interface. Install from the repository root;
the command also installs the pinned SiEval dependency:

```bash
python -m pip install -e ./plugins/decisions
```

The Git dependency is for development. Before publishing this plugin to PyPI,
replace it with the minimum SiEval release containing `EvalSubject`.

## Supported contracts

| Provider | Endpoint | Shared evidence | Questions | Refusal |
| --- | --- | --- | --- | --- |
| OpenRouter | `/api/alpha/decisions` | Text, JSON, images | Predicate, choice, score | Per question |

The shared question name is `predicate`; OpenRouter calls it `noul`.
Scores use zero-based ordered levels. Native answers, distributions and confidence
are retained without renormalization or deriving confidence from probabilities.
Distribution checks allow two-decimal display rounding for OpenRouter answers.
An explicitly requested probability or confidence must be reported on a successful
answer. Refusals remain separate from negative answers.

This first implementation covers the shared text/JSON/image evidence contract.
OpenRouter boxes/assets, streaming and provider discovery are outside this
version. OpenRouter routing-specific restrictions remain server-validated.
No model capability entries or output channels are added to SiEval core.

## Run the local labelled example

From the repository root, set the OpenRouter key:

```bash
export OPENROUTER_API_KEY='your-key'
sieval run plugins/decisions/examples/openrouter.yaml --dry-run
sieval run plugins/decisions/examples/openrouter.yaml
```

The example is a small labelled smoke dataset, not a published benchmark or a
calibration claim. Replace it with your own JSONL dataset. One line is one sample
with shared input, independent questions and references:

```json
{
  "input": "The screen arrived cracked.",
  "questions": [
    {"id": "damaged", "type": "predicate", "instructions": "Is the item damaged?"},
    {
      "id": "queue",
      "type": "choice",
      "instructions": "Which queue should handle this?",
      "options": [
        {"value": "returns", "guidance": "Damaged items"},
        {"value": "billing", "guidance": "Payment problems"}
      ]
    }
  ],
  "reference": {"damaged": true, "queue": "returns"}
}
```

Text may instead be a list of typed parts: `{"type": "text", "text": "..."}`,
`{"type": "json", "data": {...}}`, or an image with `url` or base64 `data`
and `media_type`. Explicit JSON nulls remain intact on disk. Reference answers
are validated before inference and are never sent to the provider.

Set `provider: openrouter` and choose `model_name` in the task's `args`:

| Provider value | Example model | Key environment variable | Base URL |
| --- | --- | --- | --- |
| `openrouter` | `typesafe/jev-1.14` | `OPENROUTER_API_KEY` | Default provided |

Use a distinct result directory when switching models. Check model
availability in your account. No live calls are made by the test suite.

OpenRouter options are explicitly limited to `provider`, `session_id` and `user`.
Options cannot replace model, questions, input or credentials.

## Metrics and persistence

`DecisionJSONLTask` reports question accuracy, answered accuracy, refusals,
binary predicate Brier score, multiclass choice Brier score, and score mean
absolute error. Predicate decisions use `predicate_threshold` (default 0.5);
score correctness uses `score_tolerance` (default 0.5). Refusals count as incorrect
in accuracy and are excluded from probability/error metrics. Failed samples are
reported separately; question metrics cover finalized samples.

The client opens a `subject_call()` scope and uses `RecordingAsyncClient`:

- HTTP attempts and errors go to `requests.jsonl` using SiEval's retry policy.
- Request snapshots and normalized token usage become stage `model_calls`.
- The native response is preserved as `JsonPart`, alongside typed answers.
- Credentials are read from environment variables and excluded from identity.
- Endpoint, protocol version and provider options join inference identity.
- Changing model, revision or request options refuses inference reuse. Pin a
  model/revision when the service exposes one; an alias alone cannot detect a
  silent remote model update.
- Changing grading thresholds reuses recorded inference. A change confined to a
  task's postprocess or feedback method also follows stage-based resume. Shared
  helper-module changes may affect several stage identities and can require a
  fresh run under SiEval's strict comparison.

Re-run a saved evaluation with `sieval run ... --resume`. The typed records are
decorated with `@sieval_record` and reload in a fresh process without converting
decisions into generated text or requiring a core `Response` extension.

## Add another benchmark

Import the question records and `create_decisions_client()` from this package.
Your task returns that client from `subject`, builds `DecisionRequest` in
`preprocess()`, calls `client.decide()` in `infer()`, and owns its grading rules.
API adapters and call recording are reused across benchmarks.

## Validation

```bash
python -m pytest plugins/decisions/tests tests/integration/test_decisions_plugin.py -q
ruff check plugins/decisions tests/integration/test_decisions_plugin.py
ruff format --check plugins/decisions tests/integration/test_decisions_plugin.py
ty check plugins/decisions/src/sieval_decisions
```

Protocol fixtures are derived from documented response shapes and use mocked
HTTP. Real provider access and calibration on a published benchmark remain to
be validated with provider credentials and the selected dataset.

## Protocol references

- [OpenRouter Decisions guide](https://openrouter.ai/docs/guides/features/decisions)
- [OpenRouter Decisions reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
