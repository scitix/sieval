# Concurrency Control

Four levels of concurrency limiting. Effective limit = `min(global, task, stage, model)`.

```python
# Global (across all tasks)
multi_runner = MultiTaskRunner(concurrency_limit=256)

# Task-level
config = TaskRunnerConfig(concurrency_limit=128)

# Stage-level
config = TaskRunnerConfig(concurrency_limits={"infer": 64, "postprocess": 32})

# Model-level
model = ChatModel("gpt-4o", concurrency_limit=32)
```

## Self-hosted SGLang engines

For a mixed workload on an SGLang engine `sieval run` launches, set a high
fixed `concurrency_limit` on the root model and let the engine schedule. On a
mixed arck-general-1k slice on 4×H100, a cap of 256 served about 4370 useful
tokens/s, a cap of 64 about 3600, and 0.15.6's KV admission about 2470. At 256
a few requests hit connect timeouts and were retried; lower the cap if many
do.

```yaml
models:
    candidate:
        concurrency_limit: 256
        infer:
            backend: sglang
            checkpoint: /models/candidate
```

## `concurrency_limit: auto`

`auto` is opt-in and being reworked (#287); prefer a fixed cap for now.

A root model on an SGLang engine `sieval run` launches can set
`concurrency_limit: auto` to admit requests by the KV cache they may hold.
Only `auto` does this: a number is a fixed cap, and no limit leaves the root
bounded by the task and stage limits alone.

```yaml
models:
    candidate:
        concurrency_limit: auto
        infer:
            backend: sglang
            checkpoint: /models/candidate
```

- **Capacity.** `max_total_num_tokens` from `/get_server_info` once the
  engine is ready, times its data-parallel ranks. The launch prints
  `KV: <tokens> x <replicas>`.
- **Size.** `n × min(context, max_tokens + 4096)`, a whole context per
  sequence when `max_tokens` is unset. One `n: 8` request holds what eight
  `n: 1` ones do; a 1024-token request holds 5120.
- **Order.** First come, first served, after the other limits. A request
  waits before it is sent, so no client timeout runs meanwhile. One larger
  than the whole pool runs alone.
- **Engine load.** While requests wait, SiEval polls `/v1/loads` once a
  second and holds them back while the engine reports queued requests or a
  KV pool at least 95% full. No answer leaves the KV budget in charge.
- Throughput only: no stage identity changes. Request records show the
  budget as `kv_tokens` under `in_flight`.

`--dry-run` accepts it before the engine exists; a model started at a barrier
reads its pool at its first request. If the engine reports no KV pool, its
requests fail naming the root. Anywhere else (external API, vLLM, a derived
model, the run's top level) `auto` is refused before launch. Per-task
`concurrency_limits` still apply.
