# Inference Service Management

SiEval orchestrates inference backends (vLLM, SGLang) with recipe-driven auto-resolve.

## Quick Reference

```bash
sieval infer start /path/to/model     # auto-resolve and launch
sieval infer list                      # show running services
sieval infer show <name>               # detailed service info (includes status/phase/conditions)
sieval infer logs <name> -f            # stream engine logs
sieval infer stop <name>               # graceful shutdown
```

## Starting a Service

```bash
# Auto-resolve: detect architecture, match recipe, launch
sieval infer start /path/to/Qwen3-8B

# Explicit YAML config (recipe auto-resolved from checkpoint if omitted)
sieval infer start config.yaml

# Dry-run: print launch command without executing
sieval infer start /path/to/Qwen3-8B --dry-run

# Pass extra engine arguments after --
sieval infer start /path/to/Qwen3-8B -- --served-model-name my-model

# Detach: return immediately without waiting for ready
sieval infer start /path/to/Qwen3-8B --detach

# Serve a base checkpoint: skip the recipe's instruct-only serving params
sieval infer start /path/to/Qwen3-8B-Base --model-type gen
```

### Model type and the capability layer

A recipe splits its params into a **hardware** layer (dtype, memory,
parallelism, context) and a **capability** layer (reasoning parser, tool-call
parser, tool choice). The capability layer is selected by model type: `chat`
resolves the instruct params, `gen` resolves none — a base checkpoint has no
chat template and no tool-calling surface for those flags to act on.

In YAML mode the type comes from the config: the model's `type:` if declared,
otherwise it is inferred from the tasks pointing at that model (a PPL or CLP
task requires `gen`). `sieval run` does the same. Checkpoint mode has no config
and no task context, so `--model-type` declares it there; unset, it defaults to
`chat`. Passing `--model-type` in YAML mode is rejected rather than silently
overriding the config.

The instruct flags are inert on a base checkpoint — the engine accepts and
ignores them — so this affects what `infer_plans.yaml` records as the params
used, not whether the service starts.

## Resolving for declared GPUs

A recipe's hardware layer and its tensor and data parallelism come from the
GPUs `nvidia-smi` reports. To resolve a plan on another node, a CPU node
included, declare those GPUs instead:

```bash
export SIEVAL_TARGET_GPU='NVIDIA H100 80GB HBM3:8:81559'   # <name>:<count>:<memory MiB>
```

`<name>` and `<memory MiB>` are what
`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` prints
on a node with those GPUs, and `<count>` is how many it lists. The plan is then
the one that node resolves, so `runs[].serving` and the context limit match
too. A shorthand like `H100-80G:8` is refused: the name is recorded in a
path-only plan and the memory decides tensor parallelism, so only the reported
values reproduce it.

Declared, `sieval infer start` and `sieval run` launch an engine only on a node
with exactly those GPUs, memory included: two nodes of one model can report
different `memory.total` (ECC on or off, PCIe or SXM), and the refusal names
each field that differs, and exits 4 ([Exit codes](configuration.md#exit-codes)).
`--dry-run` checks nothing, so it runs on any node.

`infer_plans.yaml` records as `gpu` the GPUs its plans were resolved for,
declared or detected. A `sieval run --resume` declaring the same GPUs reuses a
recorded plan, rather than resolving it again, for each served model whose
`path`, `infer`, capability layer and `deterministic` are unchanged. A plan
holding an engine `api_key` or `authorization` is resolved again instead:
`infer_plans.yaml` writes those values as `<redacted>`.

## Handles and engine logs

Each launched service has a handle file under `~/.sieval/handles`:

| Launched by | Handle | Name in `infer list` / `show` / `stop` |
| --- | --- | --- |
| `sieval infer start` | `<model>.json` | `<model>` |
| `sieval run` | `run-<pid>/<model>.json` | `run-<pid>/<model>` |

So `sieval run` never collides with an `infer start` of the same model. A run
removes its handles when it ends; if it was killed, the next `infer list`
cleans them up, except a handle whose engine is still alive, which stays
listed so `infer stop` can reach it.

Engine logs:

| Launched by | Log |
| --- | --- |
| `sieval run` with a `result_dir` | `<result_dir>/logs/<model>/<role>.log` |
| otherwise | `~/.sieval/logs/<model>/<role>-<deployment_id>.log` |

Each launch prunes `~/.sieval/logs/<model>/` to the last N logs per role,
counting the new one; other models' logs and `result_dir` logs are untouched.
Set N with `--keep-logs` (on `infer start` and `run`), else `SIEVAL_KEEP_LOGS`,
else 3. `0` deletes the log when the service stops; a negative N keeps all.

A task may not be named `logs`: `<result_dir>/logs/` is reserved for these logs.

## Services a task needs

A task declares the processes it calls besides the model in `Task.services`.
Today there is one, `code-eval`, the code-execution service vendored at
`vendor/code-evaluator`, used by HumanEval, MBPP, LiveCodeBench,
Ag-LiveCodeBench-X, LiveOIBench, MultiPL-E, SciCode and QuoteBench.

When `SIEVAL_CODE_EVAL_API` is unset, `sieval run` and `sieval eval` start
the vendored copy once per run on a free port,
set the variable, and stop it when the run ends. When it is set, that endpoint
is used and nothing starts.

**A started evaluator runs model-generated code on this host**, guarded by
resource limits and disabled builtins, not by a sandbox; QuoteBench runs bash
with network access. To isolate grading, build an image from
`vendor/code-evaluator/docker/` and set `SIEVAL_CODE_EVAL_API` to it.

| Variable | Default |
| --- | --- |
| `SIEVAL_CODE_EVAL_DIR` | `vendor/code-evaluator` of the source checkout; set it when SiEval is installed from a wheel |
| `SIEVAL_CODE_EVAL_PYTHON` | SiEval's own interpreter, which needs the `code-eval` group (`pip install 'sieval[code-eval]'`) |

Graded programs run with the evaluator's host and interpreter: LiveOIBench needs
`g++` on `PATH`, MultiPL-E each language's toolchain, SciCode the `scicode`
group. LiveOIBench and MultiPL-E refuse to start when their toolchain is missing.

`sieval run` checks the copy and the interpreter before it starts any engine.
The service's handle is `run-<pid>/service-<name>.json` and its log
`<result_dir>/logs/services/<name>/server.log`, else `~/.sieval/logs/<name>/`;
a served model may not be named `services`. Only one run per process may start
a given service. SIGTERM stops a run as Ctrl+C does, and it exits 143.

Each task's `meta.json` records what graded it under `identity.feedback.services`:
`"source": "vendored"` with the `VENDORED.md` commit, a digest of the
evaluator's Python sources and its `runtime` (Python version, platform, versions
of the packages graded programs import); or `"source": "external"`, with no URL.
A change to any of these changes the `feedback` identity, so a resume reruns
`feedback` and `report`.

## YAML Infer Configuration

Models with a `path` field (and no `api_base`) or an `infer` section in the YAML config are automatically launched by `sieval run` and stopped after evaluation completes.

A top-level `api_base` turns auto-serve off. An `args.api_base` next to `path` or `infer.checkpoint` is refused before any engine starts: keep one or the other.

A run that may resume (`--resume`, or `resume` in a `runner_config`) plans each
task's resume before it starts any engine, with the same per-sample planning
the run uses, and launches one only for a model some task still has samples to
infer with. A resume that keeps every sample, or reruns only from
`postprocess`, `feedback` or `report`, starts none, so it runs on a node
without GPUs; its `infer_plans.yaml`, the inference identity and `realized_deployments`
are what a launch would record, and no engine log is written. When some sample
needs inference and `nvidia-smi` finds no GPU, the run is refused before
anything starts. A judge or other external model is called only when a sample
is graded. A recorded context overflow the resume scores from its record still
starts the engine: a task that calls the model more than once asks again.

A sample that a rerun `feedback` sends to another iteration needs the model
again, which planning cannot foresee. With no engine started it fails at
`infer`, before any request, and a later `--resume` that starts the engine
retries it.

### Losing an engine mid-run

`sieval run` watches each engine it launched while its samples run. An engine
is lost when its process exits, or fails its health check for 180 s in a row.
Every sample still needing it then fails at once: queued ones as
`engine_died_unsent`, in-flight ones as `engine_died`. Samples on other
engines carry on.

Each task still writes `report.json`, the engine is stopped, no later serving
group starts, and the run exits 6 with the engine's exit code (or failing
health check), the failures per task and its last log lines.

`sieval run --resume` retries both reasons. `engine_died_unsent` spends no
retry; `engine_died` spends one, so a sample that crashes the engine stops at
`max_retries`. After an out-of-memory exit, change the plan
(`infer.overrides`) before resuming.

An engine stopped on purpose, lost after its last sample, or not launched by
SiEval (`api_base`, `sieval infer start`) is not reported.

### Serving plans: context window and dialect

`sieval run` and `sieval infer start <yaml>` size each served model's context
window from what its tasks need, before any engine starts (`--dry-run`
included).

A task needs its `min_context` if set, else the larger of what its class
declares (`TaskRequirements.min_context`; RULER: `max_seq_length` plus any
unreserved thinking budget) and its effective `max_tokens` plus 4096 for the
prompt. Both `min_context` forms already count prompt and output and are used
as they are. A task whose prompts are longer (or known to be shorter) sets
`min_context`. A root model states what it can serve:

```yaml
models:
  qwen3-8b:
    path: /models/Qwen3-8B
    context:                 # only for a model SiEval launches
      native: 40960          # default: config.json max_position_embeddings
      max: 131072            # longest window reachable with scaling
      scaling:               # engine parameters that reach it
        json_model_override_args: '{"rope_scaling": {"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 32768}}'
tasks:
  long_math:
    class: AIME2025ZeroShotGenTask
    dataset: aime
    min_context: 38000       # optional: override the derived need
```

- Within the recipe window: nothing changes.
- Within `native`: the plan's `context_length` / `max_model_len` is raised to
  the largest such need, for every task on that model. A `max_tokens` below
  `native` raises it to `native` itself, even when the 4096 would not fit:
  such a task is never scaled, and its prompt gets `native - max_tokens`
  tokens. A longer prompt is that sample's
  [context overflow](metrics.md#always-present), as before.
- Within `max`: a plan of its own with `scaling`, in serving group
  `<model>@<window>`. Scaling changes short-context outputs, so no other task
  shares that engine.
- Otherwise, or beyond a window pinned by `infer.overrides` / `context_limit`:
  refused before launch, naming the task, its need and the model's limit. An
  unknown native window refuses any raise (set `context.native`).

An `api_base` model is checked against its `context_limit`; without one, an
overflow shows per sample.

**Resume.** The served window is in each task's inference identity, and a
resume never reruns inference. A run recorded at a window this release now
raises is refused: start fresh, or pin the recorded window with
`infer.overrides`.

Groups run one after another, each group's engines stopped before the next
start; an engine starts only for a group with tasks on it. `--serving-group
KEY` (keys as `--dry-run` lists them under `serving`) runs one group, so groups
can run as separate jobs into one `result_dir`. Each group writes
`infer_plans[.<key>].yaml` and records its key in `meta.json`
`runs[].serving.group`.

A `gen` model served by SGLang with no `dialect:` gets `sglang_native` when a
task scores input: SGLang's `/v1/completions` cannot. `/generate` has no
per-request `seed`, so an explicit one is refused; a managed engine pins
`--random-seed` instead.

A vLLM `gen` model keeps `openai_completions` unless it sets `dialect:
vllm_native`, which posts to the same `/v1/completions` with vLLM's own
fields (measured on 0.26.0):

| Need | `openai_completions` | `vllm_native` |
| --- | --- | --- |
| Input scoring | `echo`, split at `usage.prompt_tokens` | `prompt_logprobs`, checked against `prompt_token_ids` |
| `texts` of a scoring call | prompt + generation | generation only |
| Token ids | none | input and sampled (`return_token_ids`) |
| Structured output | refused | `structured_outputs` |
| Unknown options | `extra_body` | refused, except `min_p`, `repetition_penalty`, `stop_token_ids` |

It sends one non-streaming POST and offers no reasoning or tool parsers
(they act on chat requests). `suffix` and `max_tokens: 0` are refused before
I/O. Input top-k needs `--max-logprobs` at least the breadth.

### Model lifecycles

A model `sieval run` launches may bound its engine by stage barriers
([Stage barriers](configuration.md#stage-barriers-and-stop_after)):

| `lifecycle` | Engine runs |
| --- | --- |
| absent | The whole run |
| `until: S` | Until every sample has finished `S`, then frees its GPUs |
| `from: S` | From the barrier on `S` to the end |
| `from: S`, `until: T` | Between the two barriers |

A task's `grader` or `extractor` may name a `models:` entry, which `sieval
run` then launches. Here the judge takes the subject's GPUs once every sample
has answered:

```yaml
models:
  subject:
    path: /models/candidate
    lifecycle:
      until: inference
  judge:
    path: /models/judge
    weights: sha256:3f…    # the judge's feedback identity follows its weights
    lifecycle:
      from: inference
tasks:
  open_qa:
    class: OpenQAJudgedTask
    dataset: open_qa
    model: subject
    args:
      grader: judge
```

- **Stopping early.** With `stop_after: inference` the judge never starts; a
  later resume starts only the judge. A `stop_after` before a model's `until`
  just stops it with the run.
- **Fit.** Models alive at the same time get disjoint GPU blocks (indices into
  an inherited `CUDA_VISIBLE_DEVICES`, if set) and their own ports. If they do
  not fit, the run is refused before any engine starts. A model alone takes
  every GPU.
- **Lazy launch.** A model starts only if some sample still needs it: a
  resume with inference complete starts only the judge, a re-report nothing.
- **Identity.** Lifecycles are recorded in `runs[].serving.lifecycles` and in
  no identity.
- **Serving groups.** A model starting at a barrier launches once for all
  groups, which each run up to it and then resume. Needs a `result_dir`.
- `--dry-run` lists `barriers`, `phases` and `launches` (judges included).
  Malformed lifecycles are refused there.

### Input-scoring headroom

A task that scores its input (`TaskRequirements.input_scoring`: the PPL and
CLP tasks) makes SGLang compute full-vocabulary logits for every prompt token
of a prefill batch. A recipe sized for generation leaves too little memory for
them, and the engine dies under full load. So an SGLang plan serving such a
task reserves that memory, per GPU:

```text
row     = vocab_size × (dtype bytes + 12)        # three fp32 copies
budget  = 10% of GPU memory ÷ row
prefill = min(budget, longest scored prompt, 8192), rounded to 1024
reserve = prefill × row
```

`chunked_prefill_size` is lowered to `prefill`, and `mem_fraction_static` by
`reserve` as a share of GPU memory. Qwen3-30B-A3B-Base (151936 × bf16) on
H100 80GB: 3072 tokens, 6231 MiB, so 0.87 becomes 0.79.

- `infer.overrides` wins. A pinned `chunked_prefill_size` sizes the reserve;
  one whose reserve exceeds the 10% budget is refused unless
  `mem_fraction_static` is pinned too.
- A smaller cap already in the plan is kept. A share left to SGLang is not
  lowered.
- Unknown vocabulary or GPU memory, or a vLLM plan, leaves the plan unchanged.
- Scoring and generation tasks share one engine; a scaled group sizes its own.

`--dry-run` reports it as `serving.<group>.engines.<engine>.headroom`. It is
launch-only, so it is recorded in `infer_plans.yaml` and `runs[].serving`,
never in the inference identity: older result directories resume.

### Anthropic Messages

To call the native Anthropic Messages API, select the dialect explicitly and
provide an external endpoint and credential:

```yaml
models:
  claude:
    name: claude-sonnet-4-5
    dialect: anthropic_messages
    api_base: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    args:
      max_tokens: 1024
```

The dialect sends JSON or SSE requests to `POST /v1/messages`. Every request
must specify `max_tokens`; set a model-wide default under `models.<name>.args`,
as above, or override it for one task under `tasks.<name>.infer_args`.

Anthropic Messages has no per-request `seed`. Deterministic mode therefore does
not inject one for this dialect, and an explicit `seed` is rejected before any
network I/O.

Raw request passthrough is intentionally unavailable. `dialect_options` cannot
inject arbitrary Anthropic body fields or HTTP headers; in particular, there is
currently no way to send the `anthropic-beta` header. Features that require a
beta header are unsupported until they receive an explicit, audited mapping.

The key is sent in the `x-api-key` header; see
[Credentials](configuration.md#credentials).

### Google GenAI

To call the native Gemini `generateContent` REST API, select the dialect
explicitly and provide the Google AI endpoint and API key:

```yaml
models:
  gemini:
    name: gemini-3.8-flash
    type: chat
    dialect: google_genai
    api_base: https://generativelanguage.googleapis.com
    api_key_env: GEMINI_API_KEY
```

The model name may be `gemini-...` or `models/gemini-...`. An endpoint root
uses `v1beta`; an explicit `/v1` or `/v1beta` suffix selects that version, and
gateway path prefixes are preserved. The key is sent only in the
`x-goog-api-key` header (see [Credentials](configuration.md#credentials)).
Vertex AI resource paths and OAuth are not supported.

Supported: text and image input, function tools, the `google_search` and
`url_context` hosted tools, JSON output, thinking controls, logprobs,
streaming, request seeds, and opaque continuation.

A YAML binding can set reasoning and logprobs on any model. Google rejects an
unsupported thinking level or logprobs request with a 400 instead of ignoring
it, and a failed sample's `error_msg` carries Google's message (for example
`Thinking level is unsupported: THINKING_LEVEL_MINIMAL`). Logprobs also keep
their response check. The other model-dependent capabilities (tools, JSON
output, images, opaque continuation) have no measured refusal, so a YAML
binding still fails them with `model_outcome_missing`.

Rejected before any network I/O:

- per-image `detail`, assistant prefill, input scoring,
  `previous_response_id`, and `developer` messages (Google has no such role);
- on `v1`, function schemas using constructs outside Google's OpenAPI subset
  (`additionalProperties`, `$ref`, type arrays, numeric enums) — use `/v1beta`;
  on both versions, a function needs a non-empty description and an explicit
  `type: object` parameter root.

Other per-model limits (thinking budgets, candidate counts, tool combinations)
are also left to Google's 400. Google may accept `thinkingBudget` and
`includeThoughts` and ignore them. They are sent as written; a reply with no
thought tokens for a budget, or no thought parts for a summary, records
`"no_effect": ["reasoning.budget_tokens"]` or `["reasoning.summary"]` in
`request_params`. A reply cut short (`MAX_TOKENS`, `SAFETY`) records nothing.
The Gemini 3 call-id/`thoughtSignature`
contracts match the model name, so a gateway alias not starting with
`gemini-3` skips them.

On Gemini 3, Google
[advises leaving `temperature`, `top_p` and `top_k` unset](https://ai.google.dev/gemini-api/docs/latest-model)
but accepts them; SiEval forwards and records what the config sets. In #153,
`temperature: 0` and `top_p: 0.01` reduced distinct replies over 8 draws,
while `top_k: 1` had no visible effect.

Provider-only root controls such as `safetySettings`, `serviceTier`, or `store`
go through `dialect_options` (`cachedContent` and `labels` are `v1beta`-only).
They cannot override `contents`, `generationConfig`, `tools`, `model`, or
`stream`, and there is no HTTP-header passthrough.

Accounting:

- `MAX_TOKENS` and `CONTINUATION` keep the partial text and count toward
  `n_truncated`. The dialect does not replay `continuationToken`; raise
  `max_tokens` instead.
- Candidate policy stops (`SAFETY`, `RECITATION`, …) count toward
  `n_truncated`; a prompt blocked under `promptFeedback.blockReason` does not.
  Both keep the raw reason, and `finish_reason_sources` records which it was.
- `input_tokens` is `promptTokenCount + toolUsePromptTokenCount` and
  `output_tokens` is `candidatesTokenCount + thoughtsTokenCount`, so
  `reasoning_tokens` stays a breakdown of `output_tokens`.

### Google Interactions

`dialect: google_interactions` calls the native Gemini Interactions REST API
(`POST /v1beta/interactions`). Endpoint, key header (`x-goog-api-key`) and
version selection (an endpoint root is `v1beta`; a trailing `/v1` or `/v1beta`
picks that version; gateway prefixes are kept) match `google_genai`:

```yaml
models:
  gemini:
    name: gemini-3.6-flash
    type: chat
    dialect: google_interactions
    api_base: https://generativelanguage.googleapis.com
    api_key_env: GEMINI_API_KEY
```

Supported: text and image input, system/user/assistant turns, function tools
with `tool_choice` (`required` is the wire's `any`) and replayed
`function_call`/`function_result` exchanges, the `google_search` and
`url_context` hosted tools, JSON-schema structured output, streaming,
`temperature`, `top_p`, `max_tokens`, `stop`, `seed`, and `reasoning.effort` /
`reasoning.summary`. Each `function_result` carries the name of its matching
call, which the live backend requires.

Through a YAML config only plain text, streaming and `dialect_options` work
today: reasoning, tools, structured output, images and opaque continuation
need a Google model-profile outcome, which ordinary bindings do not supply, so
they fail reconciliation with `model_outcome_missing`.

Rejected before any network I/O: `previous_response_id`, logprobs, input
scoring, `n > 1`, `top_k`, penalties, `reasoning.budget_tokens` (use
`effort`), `developer` messages, assistant prefill, more than one system
message, per-image `detail`, `tools.parallel`, structured output `name` /
`strict`, a tool message with more than one `ToolResultPart`, `agent` /
`agent_config`, `background`, a `service_tier` other than `standard` or
`priority` (others can return a queued interaction), and `dialect_options`
that set an IR-owned body key.

**SiEval always sends `store: false`**, so Interactions does not store
evaluation prompts; any other `store` value is rejected. Multi-turn calls are
stateless: resend the full history, or pass the previous reply's
`ReasoningOutput.opaque_roundtrip` as `session.opaque_continuation` and send
only the new turn. The continuation carries the native steps, thought
signatures and system instruction; a conflicting system message or a tool
result without its call is rejected. No `session_id` is returned.

Accounting:

- The terminal `status` is the finish reason. `incomplete` keeps any partial
  output and counts toward `n_truncated`, but does not tell a token limit from
  filtering. `requires_action` is valid only when function tools were
  declared; every other status (`failed`, `budget_exceeded`, `queued`, …) is
  an error.
- The request identity includes the body's `stream` flag but not the route or
  gateway prefix.
- `input_tokens` is `total_input_tokens + total_tool_use_tokens` and
  `output_tokens` is `total_output_tokens + total_thought_tokens`, so
  `reasoning_tokens` stays a breakdown of `output_tokens`. Without terminal
  `usage`, a stream uses its latest cumulative usage snapshot.
- Thought summaries are concatenated verbatim into `ReasoningOutput.text`.
  `summary: auto` does not guarantee one: a signed thought without a summary
  leaves `text` unset and keeps the continuation.

## Environment Variables

Custom environment variables can be passed through the YAML config's `infer.env` section. Values are injected into the inference engine process.
