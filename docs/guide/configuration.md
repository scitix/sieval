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
        name: "gpt-3.5-turbo-instruct"
        type: "gen"
        args:
            max_retries: 3
            concurrency_limit: 128
            temperature: 0.0

    # Derived model — inherits base and adds a local cap under base's 128
    math_model:
        base: base_model
        args:
            concurrency_limit: 64

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
            n_shot: 8
        # infer_args:                  # per-task inference parameter overrides
        #     max_tokens: 512
```

### Key Concepts

- **Model derivation**: `base: parent_model` inherits request defaults and
  shares the root deployment, connection pool, and limiter hierarchy
- **Model kind**: normalized task input requirements determine one legacy kind
  per deployment root; YAML `type: "chat"` / `type: "gen"` is a checked
  compatibility assertion, and every config sharing that root must agree
- **Dialect rebinding**: the composition layer reconciles a target runtime plan
  and uses `with_dialect()` to reuse a compatible root pool; a model never
  invents or converts that plan locally
- **Quota allocation**: `concurrency_limit` in `args` adds a local cap while
  retaining the root's shared cap
- **Reserved task name**: `logs` (`<result_dir>/logs/` holds engine and service logs)
- **Services**: code tasks start the vendored evaluator unless
  `SIEVAL_CODE_EVAL_API` is set; see [Services a task needs](infer.md#services-a-task-needs)
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
from sieval.core.models import ChatModel

base = ChatModel("gpt-4o", concurrency_limit=128)

math_model = base.with_args(concurrency_limit=64)  # local cap under shared 128
code_model = base.with_args(concurrency_limit=32)  # local cap under shared 128
```

The compatibility wrappers do not expose `as_type()`. A dialect change is
composition-mediated: configuration/session code first reconciles a target
`RuntimeBindingPlan`, then calls `with_dialect()` with that plan. If the target
uses the same deployment and connection family, the rebound model retains the
same client, pool, and limiter hierarchy. An incompatible target fails before
allocation or I/O and requires a separately constructed deployment and pool.

`with_dialect()` is therefore a low-level binding operation, not a shortcut
that derives a plan from a wrapper class. `as_compat_type()` remains an internal
legacy shape adapter for an already-bound model and never changes its dialect.

## Anomaly Detection

Built-in anomaly detection runs automatically after each task and saves to `anomalies.json`. Rules are filtered by task tags — custom rules can be added via `@sieval_detection_rule` (see `sieval/core/tasks/anomaly.py`).

## Sharded Persistence

Append-only sharded storage provides crash recovery via `auto_resume`, stream processing (low memory), and parallel shard writes. Custom dataclass types in `TaskContext` fields must use `@sieval_record` for proper deserialization.

## Resume

A resume compares the stage identities in each task's `meta.json` with the ones
the new run computes:

| Differs | Resume |
| --- | --- |
| Data or prompt, for some samples | Reuses the samples that match; reruns the rest from the start |
| Inference | Refused, naming the stage |
| Grader (grading code, `grader_args()`, judges) | Reruns `postprocess`, `feedback` and `report` on the recorded inference |

A grader-only change therefore needs no hand-deleted shards: rerun with
`--resume` and the task regrades.

`--resume-from DIR` starts from another run, possibly read-only and of another
task. `DIR` is a session `result_dir`; each task copies the shards of
`DIR/<task>` into its own empty `result_dir`, then resumes as above. With a
one-task config, `DIR` may be the task directory itself. A task with no
directory under `DIR` runs fresh. Per task, `runner_config.resume_from: <task
dir>` does the same and wins over the option. The run is recorded as
`runs[].start: "from"` with `runs[].from = {dir, identity}`; `DIR` is never
written. Rerunning the same command after an interruption resumes the copy.

A directory recorded before stage identities can still be resumed from. Its
prompt is digested from the recorded `preprocess` outputs; its data and
inference are compared through the `effective_config.yaml` beside it. Without
that file they are unknown, and the resume is refused.
