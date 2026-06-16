# Configuration

## YAML Format

SiEval uses YAML files for batch evaluation. A config defines models, datasets, tasks, and runner options.

```yaml
result_dir: "./outputs/my-eval"

runner_config:
    concurrency_limits:
        infer: 128
    max_iterations: 3
    auto_resume: false

models:
    base_model:
        name: "gpt-4o"
        type: "chat"
        args:
            max_retries: 3
            concurrency_limit: 128
            temperature: 0.0

    # Derived model — inherits base, reserves 64 from base's 128
    math_model:
        base: base_model
        args:
            concurrency_limit: 64

    # Derived model with type conversion (ChatModel -> GenModel)
    gen_model:
        base: base_model
        type: "gen"
        args:
            concurrency_limit: 32

datasets:
    gsm8k:
        class: GSM8KDataset
        path: "openai/gsm8k"

tasks:
    gsm8k_kshot_base_gen:
        class: GSM8KFewShotBaseGenTask
        dataset: gsm8k
        model: math_model
        args:
            k: 8
        # infer_args:                  # per-task inference parameter overrides
        #     max_tokens: 512
```

### Key Concepts

- **Model derivation**: `base: parent_model` inherits client, limiter, and args
- **Type conversion**: `type: "gen"` switches between ChatModel and GenModel
- **Quota allocation**: `concurrency_limit` in `args` reserves capacity from base
- **Class resolution**: built-in classes (exported by `sieval.tasks` / `sieval.datasets`) use short names; custom classes must use full module paths (`my_pkg.my_module.MyTask`)

## Task Pipeline

Each task implements a typed 5-stage pipeline:

```text
preprocess -> infer -> postprocess -> feedback -> report
                            ^            |
                            +-- iterate -+  (bounded by max_iterations)
```

| Stage | Role |
| ------- | ------ |
| **Preprocess** | raw sample -> model input |
| **Infer** | model API call (generation or perplexity) |
| **Postprocess** | extract answer from model output |
| **Feedback** | check correctness; return `(finalize, feedback)` — `False` to iterate |
| **Report** | aggregate results into final metrics |

## Model Resource Pool

Hierarchical concurrency control — derive child models from a base and allocate API quotas:

```python
from sieval.core.models import ChatModel, GenModel

base = ChatModel("gpt-4o", concurrency_limit=128)

math_model = base.with_args(concurrency_limit=64)   # reserves 64
code_model = base.with_args(concurrency_limit=32)   # reserves 32
gen_model = base.as_type(GenModel)                   # same quota, different type

# base uses remaining elastic capacity (128 - 64 - 32 = 32)
```

## Anomaly Detection

Built-in anomaly detection runs automatically after each task and saves to `anomalies.json`. Rules are filtered by task tags — custom rules can be added via `@sieval_detection_rule` (see `sieval/core/tasks/anomaly.py`).

## Sharded Persistence

Append-only sharded storage provides crash recovery via `auto_resume`, stream processing (low memory), and parallel shard writes. Custom dataclass types in `TaskContext` fields must use `@sieval_record` for proper deserialization.
