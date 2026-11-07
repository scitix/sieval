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

## KV Admission

An SGLang engine `sieval run` launches also admits requests by the KV cache
they may hold, unless its root model sets `concurrency_limit`.

- **Capacity.** `max_total_num_tokens` from `/get_server_info` once the
  engine is ready, times its data-parallel ranks. The launch prints
  `KV: <tokens> x <replicas>`. vLLM, external endpoints and an engine that
  does not report it get none.
- **Size.** `n × min(context, max_tokens + 4096)`, a whole context per
  sequence when `max_tokens` is unset. One `n: 8` request holds what eight
  `n: 1` ones do; a 1024-token request holds 5120.
- **Order.** First come, first served, after the other limits. A request
  waits before it is sent, so no client timeout runs meanwhile. One larger
  than the whole pool runs alone.
- Throughput only: no stage identity changes. Request records show the
  budget as `kv_tokens` under `in_flight`.

## `concurrency_limit: auto`

A root model on an SGLang engine `sieval run` launches can set
`concurrency_limit: auto`: KV admission as above, and while requests wait,
SiEval reads the engine's `/v1/loads` once a second. A waiting request is held
back while the engine reports queued requests or a KV pool at least 95% full.
An engine that does not answer leaves the KV budget in charge.

```yaml
models:
    candidate:
        concurrency_limit: auto
        infer:
            backend: sglang
            checkpoint: /models/candidate
```

Planning (`--dry-run`, and a run before it launches) accepts `auto` for such a
root before the engine exists; the pool is read at the first request through
it, once it is up, which also covers a model started at a barrier. An engine
that reports no KV pool fails that request, naming the root, before anything is
sent. Anywhere else (external API, vLLM, a derived model, the run's top level)
`auto` is refused before launch. Per-task `concurrency_limits` still apply.
