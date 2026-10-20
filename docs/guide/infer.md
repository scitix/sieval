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

When `SIEVAL_CODE_EVAL_API` is unset, `sieval run`, `sieval eval` and
`sieval leaderboard run` start the vendored copy once per run on a free port,
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

The service's handle is `run-<pid>/service-<name>.json` and its log
`<result_dir>/logs/<name>/server.log`, else `~/.sieval/logs/<name>/`. Only one
run per process may start a given service.

Each task's `meta.json` records what graded it under `identity.grader.services`:
`"source": "vendored"` with the `VENDORED.md` commit, a digest of the
evaluator's Python sources and its `runtime` (Python version, platform, versions
of the packages graded programs import); or `"source": "external"`, with no URL.
A change to any of these changes the grader identity.

## YAML Infer Configuration

Models with a `path` field (and no `api_base`) or an `infer` section in the YAML config are automatically launched by `sieval run` and stopped after evaluation completes.

### Anthropic Messages

To call the native Anthropic Messages API, select the dialect explicitly and
provide an external endpoint and credential:

```yaml
models:
  claude:
    name: claude-sonnet-4-5
    dialect: anthropic_messages
    api_base: https://api.anthropic.com/v1
    api_key: "<your Anthropic API key>"
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

The YAML loader does not expand environment-variable expressions in `api_key`.
Replace the placeholder above through your configuration/secret-management
workflow; omit `api_key` only when the target endpoint does not require
authentication.

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
    api_key: "<your Gemini API key>"
```

The model name may be `gemini-...` or `models/gemini-...`. An endpoint root
uses `v1beta`; an explicit `/v1` or `/v1beta` suffix selects that version, and
gateway path prefixes are preserved. The key is sent only in the
`x-goog-api-key` header. Vertex AI resource paths and OAuth are not supported.

Supported: text and image input, function tools, the `google_search` and
`url_context` hosted tools, JSON output, thinking controls, logprobs,
streaming, request seeds, and opaque continuation. Ordinary YAML bindings do
not yet supply Google model-profile outcomes, so model-dependent capabilities
(images, reasoning, tools, structured output, logprobs, opaque continuation)
currently fail reconciliation with `model_outcome_missing`; plain text
generation, streaming, seeds, and `dialect_options` work today.

Rejected before any network I/O:

- per-image `detail`, assistant prefill, input scoring,
  `previous_response_id`, and `developer` messages (Google has no such role);
- on `v1`, function schemas using constructs outside Google's OpenAPI subset
  (`additionalProperties`, `$ref`, type arrays, numeric enums) — use `/v1beta`;
  on both versions, a function needs a non-empty description and an explicit
  `type: object` parameter root;
- explicit `temperature`/`top_p`/`top_k` on `gemini-3…` models: Google
  [advises omitting them](https://ai.google.dev/gemini-api/docs/latest-model)
  but accepts them, so the record could claim a control with no effect.

Per-model limits Google enforces itself (thinking levels, thinking budgets,
candidate counts, tool combinations) are left to its 400. The sampling check
above and the Gemini 3 call-id/`thoughtSignature` contracts match the model
name, so a gateway alias not starting with `gemini-3` skips them.

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
