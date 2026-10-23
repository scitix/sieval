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
the one that node resolves, so the inference identity matches too. A shorthand
like `H100-80G:8` is refused: the name is recorded in a path-only plan and the
memory decides tensor parallelism, so only the reported values reproduce it.

Declared, `sieval infer start` and `sieval run` launch an engine only on a node
with exactly those GPUs, memory included: two nodes of one model can report
different `memory.total` (ECC on or off, PCIe or SXM), and the refusal names
each field that differs. `--dry-run` checks nothing, so it runs on any node.

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

## Environment Variables

Custom environment variables can be passed through the YAML config's `infer.env` section. Values are injected into the inference engine process.
