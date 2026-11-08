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

## `concurrency_limit: auto`

A root model on an SGLang engine `sieval run` launches can set
`concurrency_limit: auto`. SiEval then finds the number of sequences in flight
that gives the most token throughput, and keeps adjusting it during the run.
Only `auto` does this. A number is a fixed cap, and no limit leaves the root
bounded only by the task and stage limits.

```yaml
models:
    candidate:
        concurrency_limit: auto
        infer:
            backend: sglang
            checkpoint: /models/candidate
```

The best limit depends on the model, the GPUs and the prompt and output
lengths of the task set, so it is measured, not computed:

- **Queueing is healthy.** An engine that queues some requests is saturated,
  and that is the goal. Queue length is not a signal.
- **KV usage is not throughput.** With DP attention KV usage stays low.
  For MoE, total throughput peaks at some concurrency and then falls while the
  engine still looks healthy. A reservation per request
  (`n × min(context, max_tokens + 4096)`) ignores the prompt: it is too
  pessimistic for long outputs and too optimistic for long prompts.

### How it adapts

The limit counts sequences, so one `n: 8` request counts 8. It starts at 32.
Every 5 s window while requests wait at the limit, SiEval reads the engine's
counters and measures the tokens it processed: prompt tokens it computed
(prefix-cache hits excluded) plus tokens it generated. Then:

- **Climb.** It doubles the limit while each step gives at least 5% more
  throughput. After the first stop it steps by a quarter of the limit.
- **Plateau or fall.** A step up that does not gain 5% goes back to the best
  limit so far and holds there. So does a step that loses throughput, as MoE
  does past its peak.
- **Probe.** After 6 measurements at a limit it tries a step up, and next
  time a step down (a fifth of the limit). A smaller limit that loses no more
  than 5% is kept, since it holds the same throughput with fewer sequences.
- **Back off.** Retracted requests, a request timeout or a 5xx halve the
  limit, and it does not climb back to where that happened until the next
  probe.
- **Ceiling.** The limit stays between `min` (8) and `max`. By default `max`
  is twice the requests the engine runs at once (`max_running_requests` from
  `/v1/loads`, which SGLang sizes from its KV pool). A KV pool at least 95%
  full also stops a climb.

A window where requests did not wait at the limit tells nothing about a
higher limit, so it changes nothing. Each measurement averages two windows,
after one window to settle at a new limit. A failed or slow engine read
changes nothing and fails no request.

### What it reads

`sieval run` adds `--enable-metrics` to the launch of an `auto` root. The flag
only enables reporting: it changes no numerics, plan or identity, and the
dry-run's `launch.argv` shows it. On SGLang 0.5.13:

| Read | Field | Use |
| --- | --- | --- |
| `/metrics` | `sglang:realtime_tokens_total`, modes `prefill_compute` and `decode`, summed over ranks | tokens per window |
| `/metrics` | `sglang:prompt_tokens_total` + `sglang:generation_tokens_total` | tokens, if the above is missing (counted as requests finish) |
| `/metrics` | `sglang:num_retracted_requests_total` | retractions |
| `/v1/loads?include=core` | `token_usage` (fullest rank), `max_running_requests` (sum) | KV ceiling, default `max` |

An engine that serves no counters (for example `/metrics` refused) is measured
from the usage of the answers SiEval received. That count lags, because a
request counts only when it finishes, so the engine's counters are preferred.

### Options

`auto` alone uses the defaults below. To change one, give `auto` a mapping:

```yaml
models:
    candidate:
        concurrency_limit:
            auto:
                signal: throughput
                max: 256
        infer:
            backend: sglang
            checkpoint: /models/candidate
```

| Option | Default | Meaning |
| --- | --- | --- |
| `signal` | `throughput` | `throughput`: the controller above. `kv`: KV admission as in 0.15.6 (below). `both`: the controller, plus the KV reservation as a second gate. |
| `start` | `32` | Sequences in flight at the start |
| `min` | `8` | Floor |
| `max` | engine's `max_running_requests` × 2, else 1024 | Ceiling |
| `kv_ceiling` | `true` | A KV pool at least 95% full stops a climb |
| `retraction_backoff` | `true` | Retracted requests halve the limit |
| `interval` | `5` | Seconds per window |

Timeouts and 5xx always back off.

#### `signal: kv`

This is the `auto` of 0.15.5 and 0.15.6, kept for comparison. A request
reserves `n × min(context, max_tokens + 4096)` tokens of the engine's KV pool
(`max_total_num_tokens` from `/get_server_info` times its data-parallel ranks),
a whole context per sequence when `max_tokens` is unset. Requests are admitted
first come first served, and one larger than the whole pool runs alone. While
requests wait, SiEval polls `/v1/loads` once a second and holds them back
while the engine reports queued requests or a KV pool at least 95% full.
`kv` and `both` need the pool. If the engine reports none, their requests
fail naming the root.

### What a run shows

- At launch, `KV: <tokens> x <replicas>, context <window>`.
- A run log line on every change:
  `auto candidate: limit 64 -> 128 (rising, 21,350 tok/s)`. The reasons are
  `start`, `rising`, `plateau`, `falling`, `probe up`, `probe down`,
  `same lower`, `retracted`, `failed`, `engine max`. At the end:
  `auto candidate: limit ended at 96 after 11 change(s)`.
- Request records show the limit under `in_flight.auto`:
  `{"borrowed": 61, "total": 64, "decision": "holding", "source": "engine"}`.
  `decision` is the controller's last answer, including the reasons a window
  changed nothing (`settling`, `measuring`, `holding`, `below limit`,
  `no reading`, `draining`, `kv full`, `at max`). `source` is `engine` or
  `client`. `kv` and `both` add `kv_tokens`.

`auto` changes throughput only. No stage identity changes, and a run resumes
under any `auto` option or a number.

`--dry-run` accepts `auto` before the engine exists. The controller reads
nothing until requests wait at its limit, so a model started at a barrier
needs nothing at bind time; with `kv` or `both` it reads its pool at its first
request. Anywhere else (external
API, vLLM, a derived model, the run's top level) `auto` is refused before
launch. Per-task `concurrency_limits` still apply.
