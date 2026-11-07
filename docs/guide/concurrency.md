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

An engine `sieval run` launches also admits requests by the KV cache they
may hold, unless its root model sets `concurrency_limit`. Requests are
counted by size, not number: one `n: 8` request at `max_tokens: 28672` holds
what eight `n: 1` ones do.

- **Capacity.** The engine's own KV pool once it is ready, times its data
  parallel replicas. SGLang reports it at `/get_server_info`
  (`max_total_num_tokens`); the launch prints `KV: <tokens> x <replicas>`.
  An engine that does not report it, and vLLM, get no KV admission.
- **A request's size.** `n × min(context, max_tokens + 4096)`, the 4096 being
  the prompt headroom serving sizes windows with; unset `max_tokens` counts
  the whole context.
- **Order.** First come, first served. A request that does not fit waits
  before it is sent, holding its request slot, so no client timeout runs
  while it waits. One larger than the whole pool runs alone.
- **Effect.** Short tasks keep their throughput (a 1024-token request holds
  5120 tokens); long ones stop overfilling the cache, which made SGLang
  retract running requests and queue the rest.
- The other limits still apply first. Throughput only: no stage identity
  changes. A request record names the budget `kv_tokens` under `in_flight`.
