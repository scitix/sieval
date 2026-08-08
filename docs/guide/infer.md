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

## YAML Infer Configuration

Models with a `path` field (and no `api_base`) or an `infer` section in the YAML config are automatically launched by `sieval run` and stopped after evaluation completes.

## Environment Variables

Custom environment variables can be passed through the YAML config's `infer.env` section. Values are injected into the inference engine process.
