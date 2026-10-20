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

A resume compares each task's recorded stage identities (`meta.json`) with this
run's:

| Differs | Resume |
| --- | --- |
| Data or prompt | Those samples rerun from the start; the rest are kept |
| Inference | Refused, naming the stage |
| Grader (grading code, `grader_args()`, judges) | `postprocess`, `feedback` and `report` rerun on the recorded inference |

So a grader-only change is a plain `--resume`. `grader_args()` counts every
constructor argument, read after `setup()`, except those in
`GRADER_ARGS_EXCLUDED`; a task extends it with arguments that cannot change a
verdict: `Task.GRADER_ARGS_EXCLUDED | {"grade_timeout"}`.

A same-directory resume reruns a sample with no recorded identity, such as one
interrupted before `preprocess`. Under `record_meta: false` no sample has one,
so every recorded sample reruns, with a warning.

A resume with another model regrades and reports on the recorded inference:
`model` stays the recorded one, the invoking model goes in `runs[].model`, and
`identity.inference.requests` is unknown. If any sample would be inferred, the
resume is refused. A refused resume leaves `meta.json` as it was.

`--resume-from DIR` starts each task from `DIR/<task>` of another run, which
may be read-only or another task's: its shards are copied into the empty
`result_dir`, then resumed as above. `DIR` is never written. A task missing
there runs fresh; with a one-task config `DIR` may be the task directory.
`runner_config.resume_from` names one task directory and wins over the option.
Rerunning the same command after an interruption resumes the copy.

A directory recorded before stage identities is compared through the session's
`effective_config.yaml`. Without one, `--resume-from` it is refused, and a
same-directory `--resume` trusts it, as before.
