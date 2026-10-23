# Configuration

## YAML Format

SiEval uses YAML files for batch evaluation. A config defines models, datasets, tasks, and runner options.

```yaml
result_dir: "./outputs/my-eval"

runner_config:
    concurrency_limits:
        infer: 128
    max_iterations: 3
    resume: false

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

## Model on the Command Line

A config without `models:` can name its model at run time:

```bash
sieval run benchmark.yaml --checkpoint /models/my-sft          # launch, then evaluate
sieval eval benchmark.yaml --api-base http://localhost:8000/v1 --model my-sft
```

The flag becomes the config's one `models:` entry, `{path: <absolute
checkpoint>}` or `{api_base: <url>}`, plus `name: <--model>`. It is keyed by
the last path segment of `--model`, else of the checkpoint, and recorded in
`effective_config.yaml`, so that file reruns and resumes with no flags.

- `--checkpoint` launches with the default infer plan; anything more needs
  `models:`. `sieval eval` has no `--checkpoint`.
- `--api-base` needs `--model` and turns auto-serve off. Its key comes from
  `OPENAI_API_KEY`.
- Both flags, or either on a config that defines `models:`, is refused. A task
  naming another model (a judge) still needs `models:`.

## Credentials

Write a key as `api_key`, or name the environment variable holding it with
`api_key_env` (the loader does not expand `${VAR}`). Either is accepted on a
root model (top level or `args`) and on an inline `grader` / `extractor`, on
every dialect. A derived model (`base:`) uses its root's key.

```yaml
models:
    candidate:
        name: org/candidate
        api_base: https://gateway.example/v1
        api_key_env: CANDIDATE_API_KEY

tasks:
    simpleqa_verified_0shot_gen:
        class: SimpleQAVerifiedZeroShotGenTask
        dataset: simpleqa
        model: candidate
        args:
            grader:
                model: gemini-3.8-flash
                dialect: google_genai
                api_base: https://generativelanguage.googleapis.com
                api_key_env: SIEVAL_JUDGE_API_KEY
```

- Setting both is refused. An unset or empty variable fails before any engine
  starts, `--dry-run` included.
- `effective_config.yaml` and `infer_plans.yaml` keep the variable name but
  write every `api_key` / `authorization` value as `<redacted>`; rerunning such
  a file is refused until the key is supplied again.
- Credentials are in no identity: rotating a key, renaming its variable or
  switching forms resumes without a rerun or regrade. `runs[].config` records
  a switch of form, not a rotated value.

With neither set, `openai_chat` / `openai_completions` fall back to
`OPENAI_API_KEY`, as the OpenAI client does.

## Client Timeout

`client_timeout` sets the HTTP client's timeout per phase, in seconds. It is
accepted where `max_retries` is (a root model, top level or `args`; an inline
`grader` / `extractor`) and is never sent to the model; a derived model
(`base:`) uses its root's. `args.timeout` is different: it is a request
argument, forwarded to the server.

```yaml
models:
    candidate:
        name: org/candidate
        api_base: https://gateway.example/v1
        client_timeout:
            connect: 30
            read: 1800
```

| Phase | Default | Bounds |
| --- | --- | --- |
| `connect` | 5 (30 for an engine `sieval run` launched) | opening a connection |
| `read` | 600 | a gap between stream chunks; the whole response when not streamed |
| `write` | 600 | sending the request |
| `pool` | 600 | waiting for a free connection |

- An omitted phase keeps its default; an unknown phase or non-positive value
  fails before any engine starts.
- A timed-out sample's `error_msg` names the phase, e.g.
  `Request timed out. (timeout phase: connect)`.
- It is in no stage identity, so changing it resumes without a rerun.

## Generation Logprobs

Ask for per-token logprobs of sampled text in a task's `infer_args`:

```yaml
tasks:
    human_eval_0shot_base_gen:
        class: HumanEvalZeroShotBaseGenTask
        dataset: human_eval
        model: base_model
        args:
            n: 8
            k: 1
        infer_args:
            return_logprobs: true   # sampled tokens and their logprobs
            top_logprobs: 20        # plus the top-20 alternatives per position
```

Each `infer` record's `ModelOutput` carries them as `logprobs_tokens`,
`logprobs` and `top_logprobs` at `n: 1`, and at `n > 1` as `choice_logprobs`:
one entry per rollout, in `texts` order, with the same three fields.

| Dialect | `n: 1` | `n > 1` |
| --- | --- | --- |
| `sglang_legacy`, `openai_completions`, `openai_chat` | yes | yes |
| `google_genai` | yes | refused |
| `openai_responses` | yes | refused (no `n`) |
| `anthropic_messages` | refused | refused |

A refused request fails before any network I/O; a reply missing a rollout's
logprobs fails that sample. `infer_args` are part of the inference identity,
so changing them on a resume reruns inference.

With `echo` (`alogprobs(text, echo=True, logprobs=N)` or `infer_args: {echo:
true, top_logprobs: N}`), each input token's top-N lands in
`ModelOutput.input_top_logprobs` (`{}` for the first); `top_logprobs` keeps
the generated tokens'. Served by `sglang_legacy` and `openai_completions`.
The reply grows by N entries per input token.

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

Built-in anomaly detection runs automatically after each task and saves per-sample detail to `anomalies.json`; the count is in `report.json`'s `sieval` block. Rules are filtered by task tags — custom rules can be added via `@sieval_detection_rule` (see `sieval/core/tasks/anomaly.py`).

`context_overflow` flags a rollout whose prompt the model under test refused as longer than its context window; the sample is scored as an empty answer instead of failing. See [metrics](metrics.md) for the rule, the `sieval.context_overflow` count and how a task opts back into a failure. In `requests.jsonl` the refused attempt keeps its HTTP status and its outcome is `context_overflow` rather than `4xx`.

## Sharded Persistence

Append-only sharded storage provides crash recovery via `resume`, stream processing (low memory), and parallel shard writes. Custom dataclass types in `TaskContext` fields must use `@sieval_record` for proper deserialization.

A task directory holds:

| File | Answers |
| --- | --- |
| `meta.json` | Version, `model`, `config` (relative path to the session's `effective_config.yaml`), stage identities, `runs[]` |
| `report.json` | The result, with the run's counts in `sieval` |
| `<iteration>/<stage>/<shard>.jsonl`, `.idx` | Each sample's records and their index — the only per-sample state |
| `progress.json`, `profile.json`, `anomalies.json` | Views, rewritten on every exit |

A run killed between appending a record and indexing it reruns that stage.

## Resume

A resume compares each task's recorded stage identities (`meta.json`) with this
run's, and reruns each sample from the earliest stage that changed:

| Differs | Resume |
| --- | --- |
| `data` (the selected samples) or `preprocess` (the prompt) | Those samples rerun from the start; the rest are kept |
| `inference` | Refused, naming the stage |
| `postprocess` (answer extraction) | `postprocess`, `feedback` and `report` rerun on the recorded inference |
| `feedback` (scoring: judges, their prompts and sampling, services) | `feedback` and `report` rerun on the recorded `postprocess` |
| `report` (metric aggregation) | `report.json` is recomputed from the recorded samples: no engine starts, no judge is called |

So a grading or report change is a plain `--resume`. A sample past iteration 0
restarts instead of rolling back to a grading stage.

Each grading stage's identity holds the code its Task method reaches, the task
class and `grader_args()`; `feedback` also holds every judge and service.
`grader_args()` counts every constructor argument, read after `setup()`, except
those in `GRADER_ARGS_EXCLUDED`; a task extends it with arguments that cannot
change a verdict: `Task.GRADER_ARGS_EXCLUDED | {"grade_timeout"}`.

Code is attributed to stages by a static walk from each stage method; code two
stages reach counts in both. A stage that reaches code by a name only known at
run time (`getattr(self, name)`, `eval`, `importlib`) counts all grading code,
and so does every stage after it; `stage_code()` names the construct.

A schema-1 `meta.json` (0.10.0 to 0.11.x) folds the grading stages into one
`grader`. If it still matches the task's code, every verdict is kept and the
block is upgraded in place; otherwise every sample regrades.

A same-directory resume reruns a sample with no recorded identity, such as one
interrupted before `preprocess`. Under `record_meta: false` no sample has one,
so every recorded sample reruns, with a warning.

`runs[].resume` counts what a resume did with each recorded sample.
`sieval.core.runners.task_resume_plan()` returns the same plan without running
anything.

A resume with another model regrades and reports on the recorded inference:
`model` stays the recorded one, the invoking model goes in `runs[].model`, and
`identity.inference.requests` is unknown. If any sample would be inferred, the
resume is refused. A refused resume leaves `meta.json` as it was.

How a run starts:

| Form | Means |
| --- | --- |
| `--resume` / `runner_config.resume: true` | Resume in place |
| `--resume DIR` | Start each task from `DIR/<task>`; with one task, `DIR` may be the task directory |
| `runner_config.resume: <dir>` | Start this task from that task directory |

`--resume` takes `DIR` only from its next token, never one starting with `-`,
so `--resume -v` and `--resume cfg.yaml` resume in place. Starting from `DIR`
copies its shards into the empty `result_dir` (`DIR` is never written; a task
missing there runs fresh); a `DIR` that is the run's own directory resumes in
place. `runs[].start` records `from` or `resume`. A YAML directory beats
`--resume`, which beats `resume: false`; `resume` stays out of `runs[].config`.

Rerunning the same command after an interruption resumes the copy, or copies
again if the run never began (`.copying` in the task directory); `--resume`
alone refuses such a directory.

A directory recorded before stage identities is compared through the session's
`effective_config.yaml`. Without one, `--resume DIR` is refused and an in-place
`--resume` trusts it.

### What a run would do

`sieval run --dry-run [--resume [DIR]] -o json` reports what the run would do,
without launching an engine or writing a result directory. It reads the
`--resume` directories in place. On a node without GPUs, declare the target ones
in `SIEVAL_TARGET_GPU` ([Resolving for declared GPUs](infer.md#resolving-for-declared-gpus)).

```json
{
  "checks": [{"name": "resume_plan", "ok": true, "detail": "1 task(s): 1 sample(s) to infer, 1 to grade"}],
  "n_errors": 0,
  "n_warnings": 0,
  "gpu": "NVIDIA H100 80GB HBM3:8:81559",
  "models": {
    "subject": {
      "plan": {"checkpoint": "/models/Qwen3-4B", "backend": "sglang", "assignments": []},
      "launch": [{"role": "full", "argv": ["sglang", "serve", "--model-path", "/models/Qwen3-4B"]}],
      "needs_inference": true
    }
  },
  "tasks": {
    "echo": {
      "model": "subject",
      "start": "resume",
      "recorded": "outputs/run/echo",
      "samples": 3,
      "resume_plan": {"kept": 2, "rereported": 0, "regraded": 0, "repostprocessed": 0, "restarted": 0, "dropped": 0},
      "new": 1,
      "to_infer": 1,
      "to_grade": 1
    }
  },
  "needs_inference": true
}
```

- `checks` are the config checks, then `resume_plan`, which fails when a task's
  resume would be refused; that task's entry then holds `error` instead.
- `models` has one entry per model `sieval run` serves: the plan as
  `infer_plans.yaml` would record it, each process the launch starts (`argv`,
  and `env` and `cwd` when set), and `needs_inference`, whether this run starts
  its engine. `gpu` is what the plans were resolved for.
- `tasks.<task>.start` is `fresh`, `resume` (stage by stage) or `reuse` (the
  recorded `report.json` stands, recomputed when only `report` changed).
  `resume_plan` counts the recorded samples as `runs[].resume` will, and is
  `null` for a fresh start; `new` counts samples with no record. `to_infer`
  counts the samples sent to the model under test, `to_grade` those run through
  a grading stage.
- `needs_inference` is whether the run starts any engine. A task that infers
  through an `api_base` model counts in its `to_infer` but starts none.

`gpu`, `models`, `tasks` and `needs_inference` are always present. When the
config checks fail, or planning raises (the `resume_plan` check then names
why), `tasks` and every `needs_inference` are `null`: unknown, not false.

Planning sets the session up as the run does. It loads every dataset,
downloading one not yet cached, and starts each vendored service a task
declares; neither writes under the result directory. A sample that `feedback`
sends to another iteration after a rerun from `feedback` is not counted, since
the plan cannot know it.

Deprecated, removed next release: `--resume-from DIR`, `runner_config.auto_resume`
and `runner_config.resume_from` (and the same-named Python fields and
arguments) map to `resume` and warn once; one that disagrees with `resume` is
refused.
