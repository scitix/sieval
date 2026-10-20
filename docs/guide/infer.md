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

## Environment Variables

Custom environment variables can be passed through the YAML config's `infer.env` section. Values are injected into the inference engine process.
