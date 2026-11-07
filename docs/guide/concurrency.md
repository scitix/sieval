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
