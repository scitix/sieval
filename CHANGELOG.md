# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.13.1] - 2026-10-25

A patch release: a `max_tokens`-derived context need reserves room for the
prompt, without moving a task that fits under the native window into a scaled
group. No on-disk format changes: `meta.json` `identity` stays schema 4 and
0.13.0 result directories resume, except where the served window changed
(below).

### Migrating from 0.13.0

- **A task's served window changes when its need comes from `max_tokens`, so its resume is refused.** The window is part of the inference identity, and a resume never reruns inference. On a model `sieval run` launches with a known native window, a `max_tokens` below `native` that 0.13.0 served at exactly `max_tokens`, or at the recipe window with fewer than 4096 tokens to spare, is now served at `native`. The other tasks on that model's default plan share the window, so theirs changes too. A `max_tokens` at or above `native` is served in scaled group `<model>@<max_tokens + 4096>`, where 0.13.0 served it at `max_tokens` (in `<model>@<max_tokens>`, or unscaled when the recipe window held it). Start such a run fresh, or pin the recorded window with `infer.overrides` (`context_length` / `max_model_len`). A model whose tasks all fit the recipe window with 4096 to spare, or all set `min_context` or declare `TaskRequirements.min_context`, keeps its window (#239, #240).
- **Where 4096 more no longer fits, the task is refused before launch**: a `max_tokens` above `context_limit - 4096` on an endpoint, above the plan's window minus 4096 when the native window is unknown, or above `context.max - 4096` when scaled. Set the task's `min_context` if its prompts are known to be shorter (#239).

### Fixed

- **A raised serving window leaves room for the prompt.** 0.13.0 raised a plan's window to exactly a task's `max_tokens`, so every request with a non-empty prompt was refused as longer than the context. A need derived from `max_tokens` now adds 4096 tokens for the prompt; `min_context` and a task's declared need already count it and are unchanged. A task with longer prompts sets `min_context` (#239, closes #238).
- **A `max_tokens` below the native window is served at that window, never scaled.** With the 4096 added, a task such as `max_tokens: 38000` on a model with `native: 40960` would move into a scaled (YaRN) group, and scaling changes outputs. Such a task now raises the window to `native`; its prompt gets `native - max_tokens` tokens, and a longer prompt is that sample's context overflow, as before. A `max_tokens` at or above `native` still gets `max_tokens + 4096`, in a scaled group when `context.max` and `scaling` allow (#240, refs #238).

## [0.13.0] - 2026-10-25

A minor bump: `meta.json` `identity` moves to schema 4, where `inference`
digests what the model is asked, including its context limit, and no longer
how it is served; `inference_identity` takes `context_limit` instead of
`recipe`. `RESULT_SCHEMA_VERSION` stays 1, so 0.12.x result directories
resume, but a run recorded at a context window that `sieval run` now raises is
refused on resume. Refusals get their own exit codes and a machine-readable
stderr line, and a task can declare the Python packages it needs.

### Migrating from 0.12.x

- **`meta.json` `identity` is schema 4** (#233). **A reader of `meta.json` must accept schemas 1 to 4.** `inference` drops the engine and serving recipe and adds the context limit; each `runs[]` entry records `serving` instead, never compared. A schema 1 to 3 directory resumes once by the digest it was written with, so its engine and recipe must still match, then is rewritten as schema 4.
- **`inference_identity` takes `context_limit` instead of `recipe`**; `TaskRunner`, `MultiTaskRunner.add_task` and `task_resume_plan` take `context_limit` (#233).
- **New root-model keys:** `context_limit` (an endpoint's window), `weights` (compared on resume instead of the name when both runs state it), `serving` (an endpoint's stated `engine_version`/`topology`/`hardware`; refused on a model `sieval run` launches) (#233), and `context` (`native`/`max`/`scaling`) (#236).
- **Serving groups** (#236): a task needing more than the recipe window raises it up to `native`, or, up to `max`, runs on its own scaled engine in serving group `<model>@<window>`. Groups run one after another; `--dry-run` lists them under `serving`, and `--serving-group KEY` runs one.
- **Breaking for resume:** a run recorded at a window `sieval run` now raises is refused, since the window is part of the inference identity and a resume never reruns inference. Start it fresh, or pin the recorded window with `infer.overrides` (`context_length` / `max_model_len`) (#236).
- **Refusals exit 3 (resume), 4 (hardware) or 5 (environment)**, not 1, and write one `SIEVAL_REFUSAL {json}` line per refusal to stderr (#230, #235). A scheduler that treated every non-zero exit alike now sees these codes; the line's `v` is 2.
- **`extra` at both placements is refused** (`models.<m>.extra` and `models.<m>.args.extra`, or both in an inline `grader`/`extractor` config). Keep one (#228). A 0.12.0 result directory written from a top-level `extra` ran without it, so its resume is refused; start it fresh.

### Added

- **Refusals exit with their own code.** `sieval run`, `eval` and `infer start` exit 3 on a refused resume and 4 on refused hardware (no GPU for an engine, or not the declared GPUs), and print one `SIEVAL_REFUSAL {json}` line to stderr; both used to exit 1 like a crash. `--dry-run`, and a batch failing in more than one way, still exit 1. An `nvidia-smi` timeout now fails instead of reading as "no GPU". `StageResumeError` and `ResumeVersionError` gain `task`; GPU refusals raise `LaunchGpuError` (a `RuntimeError`). See [Exit codes](docs/guide/configuration.md#exit-codes) (#230, closes #229).
- **A task declares the Python packages it needs, and a run without them is refused.** `Task.python_requires = {import name: PEP 508 requirement}` on the class, or a `python_requires_for(context)` classmethod when the need depends on the task's `args`. `sieval run` and `sieval eval` check it before any engine starts or dataset loads: a package is met when its import name is found (not imported) and, if pinned, the installed distribution satisfies the pin; extras are rejected. A missing package is an `environment` refusal: exit 5, one `SIEVAL_REFUSAL` line per task with `kind: "environment"`, `reason: "python_requires"`. `--dry-run` reports it as a new `dependencies` check, `task list`/`show` as a `python-requires` entry under `missing`, and `TaskMeta.python_requires` is in `meta/index.json`. `sieval infer start` does not check it. The `SIEVAL_REFUSAL` line's `v` moves to 2; the `resume` and `hardware` lines are otherwise unchanged. It is outside every identity, so resumes are unaffected. No built-in task declares one yet (#235, closes #234).
- **How the model is served no longer refuses a resume.** `meta.json` `identity` moves to schema 4: `inference` digests what the model is asked, now including the context limit, and drops the engine and serving recipe. Each `runs[]` entry records `serving` instead, never compared, so a run on 1 GPU resumes on 8 and a launched engine's run resumes on an equivalent endpoint. New root-model keys: `context_limit` (an endpoint's window; a launched engine's comes from its plan), `weights` (compared on resume instead of the name when both runs state it) and `serving` (`engine_version`/`topology`/`hardware` of an endpoint, recorded in `runs[].serving.stated`). **Readers of `meta.json` must accept schemas 1 to 4.** A schema 1 to 3 directory resumes once by the digest it was written with, so its engine and recipe must still match, then is rewritten as schema 4. `inference_identity` takes `context_limit` instead of `recipe`; `TaskRunner`, `MultiTaskRunner.add_task` and `task_resume_plan` take `context_limit` (#233, closes #232).
- **`sieval run` sizes serving to what the tasks need.** A task's need is `TaskRequirements.min_context` (RULER declares one) or its effective `max_tokens`, overridable with a task's `min_context`; a root model's new `context:` block states `native` / `max` / `scaling`. The recipe window is raised up to `native`; a need up to `max` gets its own scaled engine in serving group `<model>@<window>`, run after the default group; anything else, or a need beyond a window pinned by `infer.overrides` or `context_limit`, is refused before launch. `--dry-run` lists groups under `serving`, `--serving-group KEY` runs one, and `runs[].serving.group` records it. `sieval infer start <yaml>` serves the same raised window. **Breaking for resume:** a run recorded at a window now raised is refused (inference is never rerun); start fresh or pin it with `infer.overrides`. An SGLang `gen` model with no `dialect:` now defaults to `sglang_legacy` when a task scores input, instead of refusing (#236, closes #231).

### Fixed

- **A model's top-level `extra` reaches the model.** `models.<m>.extra` passed `--dry-run` and was then dropped at binding, so `Model.extra` was `{}` and the inference identity omitted it. It now binds like `args.extra` on every dialect and on derived models; `args.extra` identities are unchanged. A 0.12.0 result directory written from a top-level `extra` ran without it, so its resume is refused as an inference change; start it fresh (#228, closes #227).
- **Breaking: `extra` at both placements is refused** before any engine starts, on named models and inline `grader`/`extractor` configs. Inline configs used to accept both and silently use `args.extra`. Keep one (#228).

## [0.12.0] - 2026-10-23

A minor bump: `meta.json` `identity` moves to schema 3, with one identity per
stage, and the 0.10/0.11 deprecations are removed. `RESULT_SCHEMA_VERSION`
stays 1, so 0.10.x and 0.11.0 result directories resume, and an unchanged task
upgrades in place to schema 3 without regrading. `sieval run` now starts an
engine only for samples left to infer, and `sieval run --dry-run` reports what
the run would do, per model and per task.

### Migrating from 0.11.x

- **`meta.json` `identity` is schema 3** (#211, #218): one identity per stage, `data`, `preprocess` (was `prompt`), `inference`, `postprocess`, `feedback`, `report`. The combined `grader` is gone; judges and services move to `feedback`. `preprocess` is `{digest, modules, args, splits?, fallback?}`, the code and arguments that render a prompt, instead of a digest of the rendered prompts. Each `preprocess` record is stamped `{data, preprocess}` instead of `{data, prompt}`. **A reader of `meta.json` must accept schemas 1 to 3**: 1 from 0.10.x and 0.11.0, 2 from unreleased `main` builds only, 3 from now on. `RESULT_SCHEMA_VERSION` stays 1, so 0.10.x and 0.11.0 directories resume: an unchanged task's directory upgrades in place without regrading, after one local `preprocess` call per recorded sample. A `preprocess` change now restarts every sample of the task.
- `TaskSaver.begin_run(identity={...})` and `set_identity(**stages)` replace the per-stage keyword arguments.
- **Removed (#222):**
    - `sieval.cli.leaderboard.session`, deprecated in 0.11.0. Import from `sieval.cli.session`, and resolvers from `sieval.cli.resolution`.
    - `--resume-from DIR` is an unknown option. Use `--resume DIR`.
    - **`runner_config.auto_resume` and `runner_config.resume_from`**, top-level or per task, are refused before anything runs, naming the `resume` form to use. In 0.11.0 they warned and still worked. Write `resume: true` or `resume: <dir>`.
    - The Transport aliases (`sieval.core.models.Transport`, `OpenAIChatTransport`, `OpenAICompletionsTransport`); use the Dialect names. `SglangTransport` stays. `sieval.infer.InferConfig` is gone.
- **`SIEVAL_TARGET_GPU`** (`<name>:<count>:<memory MiB>`, as `nvidia-smi` reports, e.g. `NVIDIA H100 80GB HBM3:8:81559`) declares the GPUs a serving plan is resolved for, so a CPU node resolves the plan, and the inference identity, a GPU node would. Unset, nothing changes. Set, an engine launches only on a node with exactly those GPUs (#215).
- **`sieval run` launches engines lazily on a resume** (#216): a run that may resume plans each task first and starts only the engines some task still has samples to infer with. A resume that reruns only grading stages starts no engine and runs without GPUs. A fresh run launches as before.
- **`sieval run --dry-run` now loads every dataset** and starts each vendored service a task declares, as the run does, in order to plan each task (#217). **A dry-run on a node without the data fails the `resume_plan` check**, where 0.11.0's passed; stage the data, or mount it, before the dry-run. It still launches no engine and writes no result directory.
- **`Model.extra` carries capabilities a task may ask for**, starting with `sequence_wrappers` (#218, #221). A malformed `sequence_wrappers` template is refused when the model is built and by `--dry-run`.
- **A plugin task declares its report metric names** with `report_metrics = report_metrics(...)` (`sieval.core.tasks.report_metrics`) on its class (#225). Undeclared, `tasks.<task>.metrics` in the dry-run is `null`; a malformed declaration fails the `report_metrics` check.

### Changed

- **`preprocess` is identified by code, not by the prompts it renders** (#218): the static walk the grading stages use, the `grader_args()` it reads, and for few-shot a digest of the split it draws from. A model's `extra` reaches the request, which `inference` digests, so a wrapping and a plain model share `preprocess` and differ in `inference`. A run-time name or a file opened at run time falls back to the task's whole code and the data files beside the reader. **A preprocess change now restarts every sample of the task**, where before only the samples whose prompt changed restarted.
- **A resume reruns from the earliest stage that changed** (#210). An extraction change reruns from `postprocess`, a scoring change (judge, prompt, sampling, service, `feedback` code) from `feedback`, and a `report`-only change recomputes `report.json` from the records with no engine or judge call. Before, each of these regraded every sample. Code is attributed to stages by a static walk; a run-time lookup (`getattr(self, <expr>)`, `eval`, `importlib`) falls back to all grading code. No built-in task falls back.
- `runs[].resume` counts what a resume did with each recorded sample.
- **`sieval run` starts an engine only for a model with samples to infer** (closes #213). A run that may resume (`--resume`, or `resume` in a `runner_config`) first plans each task's resume, read-only, through the runner's own resume planning (`TaskRunner.plan_run`), and launches only the engines whose models some task still has samples to infer with. A resume that reruns only grading stages, or re-reports, starts no engine and runs on a node without GPUs; when some sample needs inference and `nvidia-smi` finds no GPU, the run is refused before anything starts. The plans, `infer_plans.yaml`, the inference identity and its request digest are what an eager launch records. A fresh run is not planned and launches as before. A sample that `feedback` sends to another iteration after a rerun from `feedback` fails at `infer` when its engine was not started, and a resume that starts it retries it.

### Added

- **`Model.extra` is the capability contract between tasks and models, starting with `sequence_wrappers`** (#218). A task asks for a capability by key and does nothing different when it is absent. `sequence_wrappers: {<seq_type>: "<format with {seq}>"}` (types `dna`, `rna`, `protein`, `mol`) gives the special tokens a model was trained with around biological sequences; a task reads it with `sequence_wrapper(self.model.extra, <type>)` in `preprocess`. A malformed template is refused when the model is built and by `--dry-run`. No built-in task injects such sequences yet.
- **`SIEVAL_TARGET_GPU` declares the GPUs a serving plan is resolved for**, as `<name>:<count>:<memory MiB>` with the name and memory `nvidia-smi` reports, e.g. `NVIDIA H100 80GB HBM3:8:81559`. Recipe resolution reads it instead of `nvidia-smi`, on any node, so a CPU node resolves the plan, and the inference identity, a node with those GPUs would. Unset, nothing changes. Declared, an engine launches only on a node with exactly those GPUs. `infer_plans.yaml` gains `gpu`, the GPUs its plans were resolved for, outside every identity; a `sieval run --resume` declaring the same GPUs reuses the recorded plan of each served model whose `path`, `infer`, capability layer and `deterministic` are unchanged, unless the plan held an engine secret, which `infer_plans.yaml` records redacted (#215, closes #212).
- `sieval.core.runners.task_resume_plan` / `plan_resume`: the per-stage resume plan, without running anything.
- `TaskRunner.plan_run` / `MultiTaskRunner.plan_run` (`RunPlan`): the runner's per-sample plan of a run, `to_infer` included, read-only; the one `sieval run` launches by.
- **`sieval run --dry-run` reports what the run would do** (closes #214). `-o json` adds `gpu`, `models` (each served model's plan as `infer_plans.yaml` would record it, each process its launch starts, and `needs_inference`, whether this run starts it), `tasks` (each task's `start`, `resume_plan` with the counts `runs[].resume` will record, `null` for a fresh start, `new`, `to_infer` and `to_grade`) and `needs_inference`, whether any engine starts. The plans come from the call `sieval run` launches by. A resume that would be refused, or planning that raises, fails the new `resume_plan` check; when the dry-run stops before planning, `tasks` and `needs_inference` are `null`. It launches no engine, writes no result directory, reads `--resume` directories in place, and runs on a node without GPUs given `SIEVAL_TARGET_GPU`. Planning loads the datasets and starts each vendored service a task declares, as the run does, so a dry-run that passes its checks costs a session setup. The existing checks are unchanged.
- `sieval run --dry-run` reports each task's `stages` (per-stage `same`/`differs`/`unknown`, and `rerun_from`, the stage a resume reruns from), its own `needs_inference`, and its `metrics` (below); `RunPlan` gains `stages` (closes #219).
- **A task's report metric names are known before it runs** (closes #223): `tasks.<task>.metrics` in the dry-run is `{names, patterns}`, or `null` when unknown, and a malformed declaration fails a new `report_metrics` check. Built-in tasks are declared in `sieval/meta/report_metrics.py`, outside every stage identity; a plugin task declares `report_metrics = report_metrics(...)` (`sieval.core.tasks.report_metrics`) on its class. Every task test checks its report against the declaration.

### Removed

- **`sieval.cli.leaderboard.session`**, deprecated in 0.11.0. Import from `sieval.cli.session` (resolvers from `sieval.cli.resolution`).
- **The resume aliases deprecated in 0.10.0.** `--resume-from DIR` is now an unknown option (use `--resume DIR`). A config setting `runner_config.auto_resume` or `runner_config.resume_from`, top-level or per task, is refused before anything runs, naming the `resume` form to use, rather than ignored, so a resume never silently becomes a fresh run. `TaskRunnerConfig(auto_resume=, resume_from=)`, `EvalSession(resume_from=)`, `arun_session(resume_from=)` / `run_session(resume_from=)` and `sieval.core.runners.fold_resume_aliases` are gone; pass `resume=True` or `resume=<dir>`.
- The transport import aliases: `sieval.core.models.Transport` (use `sieval.core.models.dialect.Dialect`), and `OpenAIChatTransport` / `OpenAICompletionsTransport` from `sieval.core.models.transports` (use `OpenAIChatDialect` / `OpenAICompletionsDialect` from `sieval.core.models.dialects`). `SglangTransport` stays.
- `sieval.infer.InferConfig`, unused since `DeploymentPlan` replaced it.

### Fixed

- **A concurrency limit admits no extra task.** `anyio` before 4.15 let a task scheduled ahead of a woken waiter take the slot too, so `concurrency_limit`, `concurrency_limits` and a model's `concurrency_limit` could run one request over their bound. SiEval now requires `anyio>=4.15` (#224).

## [0.11.0] - 2026-10-23

A minor bump: `sieval leaderboard` is removed and SiEval is scoped to an
evaluation engine, and a resume is now admitted by the result schema version
instead of the package version. No stage identity changes: `IDENTITY_SCHEMA`
and `RESULT_SCHEMA_VERSION` are both 1, every built-in task's grader digest is
the one 0.10.2 computes, and 0.10.0, 0.10.1 and 0.10.2 result directories,
local builds included, resume. `anomalies.json`'s `rules_hash` changes, since
the rule set gained `context_overflow`.

### Migrating from 0.10.x

- **`sieval leaderboard run` is now `sieval eval`**, with the same options. `sieval leaderboard report` / `list`, alignment cards and the shipped leaderboard specs are gone; compare runs from their result files and `meta.json`. A config's `alignment:` block loads with a warning and is ignored.
- **Import the engine from `sieval.cli.session`.** `sieval.cli.leaderboard.session` still works for this release, with a `DeprecationWarning`, and forwards every name.
- **Each task directory gains `requests.jsonl`**, one line per model request attempt. Set `runner_config.record_requests: false` to turn it off.
- **A prompt longer than the context window scores as wrong** and is flagged by the `context_overflow` anomaly, instead of failing the sample; it counts under `report.json` `sieval.context_overflow` and `anomalies`, not `failed`. A task declares `context_overflow = "fail"` to keep the failure.
- **The resume gate reads `meta.json` `result_schema_version`.** A directory without it maps from its `version` (0.10.0 or later is schema 1), so 0.10.x directories resume as they are. Replace `resume_version_verdict` with `resume_schema_verdict`.

### Added

- **`requests.jsonl`: every model request attempt is recorded.** One line per attempt beside a task's shards: phase, role, model, sample, attempt number, timings (`connect_s` is the wait for a pooled connection), HTTP status or exception class, provider request id, retry wait, usage, and the occupancy of every limiter it ran under. Judges and extractors are covered without task code. `profile.json` gains a per-phase, per-model `requests` summary, `report.json`'s `sieval` block its counts and limiter peaks, and each `meta.json` `runs[]` entry the run's own counts. The file is append-only, rotated (`request_log_max_bytes`, `request_log_max_files`), and read by nothing: no identity, digest or score depends on it. `record_requests: false` turns it off (#197, #201).
- **`sieval run --checkpoint <path>` and `sieval run|eval --api-base <url> --model <name>`** add the one `models:` entry a config with only `datasets:` and `tasks:` lacks, and `effective_config.yaml` records it, so a rerun or resume needs no flags. Both flags together, `--api-base` without `--model`, or either on a config that defines models is refused before anything launches; `--dry-run` reports it as the `cli_model` check (#195, #198).
- **`result_schema_version` in `meta.json`** (`RESULT_SCHEMA_VERSION`, starting at 1) versions the result directory a resume reads, separately from `identity.schema`. The package version stays in `version` and `runs[].sieval` as provenance (#202, #203).
- **`context_overflow` anomaly rule** (gen tasks), and `Model.context_overflow_output` for a task that measures its own prompt and records the overflow without sending it (#206, #207).

### Changed

- **BREAKING — SiEval is an evaluation engine.** The `leaderboard` command group is removed with `run`, `report` and `list`; `sieval eval` (`cli/eval.py`) replaces `sieval leaderboard run` and always reports `command="eval"` / `eval.dry_run`. The engine moves to `sieval.cli.session`. `card.py`, `catalog.py`, `scanner.py`, `annotation.py`, the leaderboard renderers and `leaderboards/` are deleted; `examples/leaderboard-math-sft.yaml` is now `examples/math-suite.yaml` (#194, #199).
- **BREAKING — resume compatibility is keyed on the result schema version, not the package version.** A resume refuses another schema version, naming both schema versions and both SiEval versions. A directory without the field maps from its `version`: 0.10.0 and later, local builds included, is 1; older, unparseable or `0.0.0` ones are refused as before. Dev and local builds (`0.10.2+gc18745b`) now resume across builds when the schema matches. `resume_version_verdict` is replaced by `resume_schema_verdict` (#202, #203).
- **A prompt longer than the model's context window is scored as wrong, not failed.** When the server refuses the model under test's prompt as too long (SGLang's "is longer than the model's context length" or "Requested token count exceeds the model's maximum context length", vLLM's and OpenAI's "maximum context length", a 400 or 413), the call inside `infer` returns one empty text per rollout with finish reason `context_overflow`. The task grades it like any blank answer, the `context_overflow` rule flags it, and `report.json`'s `sieval` block counts it under `anomalies` and the new `context_overflow` key, not under `failed`. It used to fail the sample on every resume, so a run with one never completed. A task declares `context_overflow = "fail"` to keep the failure, now recorded with reason `context_overflow` and never retried. Timeouts, connection errors, 5xx and a judge's own overflow still fail; a logprob call fails as `context_overflow`. A task that catches the refusal itself (Multi-IF's and SysBench's walks) is unchanged. `extraction_failure` and `empty_postprocess` leave such rollouts to the new rule. In `requests.jsonl` and the request summaries the refused attempt keeps its status and is counted under outcome `context_overflow` instead of `4xx` (#206, #207).
- **A resume scores a recorded overflow without asking again.** A directory whose samples failed with a context-overflow refusal at `infer` (as 0.10.x recorded them) resumes them from the recorded refusal when the inference identity matches: no request is sent, no retry is spent, and the sample is graded and counted as above. The recorded message must name a 400 or 413 (or no status, as a stream error does), as live: a 429 or 5xx quoting the words stays a retriable failure. A task whose `infer` calls the model under test more than once, in turn or concurrently, is inferred again instead, since the record cannot say which call was refused. Because a refusal that escapes `infer` makes the runner call `infer` once more with refusals converted, an `infer` should have no side effects beyond its model calls (#206, #207).
- **`max_retries` means the same on every connection family.** The Anthropic and Google clients used to retry only connect failures, so a 429 or 503 from such a judge failed its sample at once. They now retry as the OpenAI SDK does: 408/409/429/5xx and transport errors, `x-should-retry`, `retry-after(-ms)` within 60 s, 0.5 s doubling to 8 s with the SDK's jitter. Both families get the SDK's connection limits (1000/100; httpx's default of 100 queued requests past a higher concurrency limit) and follow redirects, and Anthropic's HTTP errors name the provider's reason, as Google's do. The retry policy is renamed `sieval-http:max-retries=`, so a binding plan made under the old meaning is refused rather than reread (#201).
- `progress.json` is written off the event loop, as `report.json` and `meta.json` are; on a network filesystem its fsync used to stall every task in a session (#196, #200).

### Fixed

- The deprecated `sieval.cli.leaderboard.session` alias forwards every global of `sieval.cli.session` and `sieval.cli.resolution`, not only its ten `__all__` names; `getattr(alias, "DETERMINISTIC_DEFAULT_SEED", None)` returned `None` and silently dropped `seed` in deterministic runs (#204, #205).

### Deprecated

- `sieval.cli.leaderboard.session`, kept for one release (#199). The resume aliases deprecated in 0.10.0 (`--resume-from`, `auto_resume`, `resume_from`) are still accepted and still warn.

### Identity

`RESULT_SCHEMA_VERSION` stays 1 although scoring an overflow is a core change that alters results: 0.10.x never finalized an overflowed sample, so a resume reuses no result the change would have made differently, and it builds that sample as a fresh run would. A reused 0.10.x `report.json` gains `context_overflow` in its `sieval` block. Every built-in task's grader digest is 0.10.2's, and inference digests are unchanged.

## [0.10.2] - 2026-10-22

No on-disk format or stage identity changes: `IDENTITY_SCHEMA` stays 1, every
built-in task's grader digest is the one 0.10.1 computes, and 0.10.0 and 0.10.1
result directories resume. Two configurations that used to run are now refused:
an evaluation that loads a staged dataset holding bytes other than its pinned
revision, and an `args.api_base` on a model `sieval run` launches.

### Added

- **Input-side top-k logprobs.** A request that scores its input with `top_logprobs > 0`, such as `alogprobs(echo=True, logprobs=N)`, returns the top-N alternatives at every echoed input position in `ModelOutput.input_top_logprobs` (IR: `InputScoringResult.top_logprobs`, aligned with `token_logprobs`). Supported on `sglang_legacy` and `openai_completions`; a missing or misaligned input top-k is refused. `top_logprobs` still holds only the generated tokens'. `alogprobs`' `logprobs` defaults to 0 with `echo` (what was sent before) and 5 without; an explicit value is now sent with `echo` too, so only a caller passing `echo=True, logprobs=N>0` sends a different request than in 0.10.1. No built-in task does (#178).
- **`gated` in the dataset registry.** `DatasetMeta.gated` (frozen wire field; absent reads as `false`, so older `index.json` files load). `hle` is gated. `sieval dataset list` gains a GATED column, `dataset show` a `Gated:` line, and `dataset download` refuses a gated source without a Hugging Face token, or the Hub's refusal, with a message naming the gate and what to set (#181, #182).
- **`bbh`, `mgsm`, `mmlu_redux` and `trivia_qa` are registered**, each revision-pinned, so `sieval dataset download` fetches them and tasks name them by bare repo id. MGSM points at `juletxara/mgsm`, since `ljvmiranda921/mgsm` is gone from the Hub. An `hf:` source may end in `#<glob>`, forwarded as `snapshot_download(allow_patterns=...)`; `trivia_qa` stages only `rc.nocontext/*` (about 60 MB of a 35 GB repo) and evaluates `validation`, because upstream `test` is unlabeled (#180, #184).

### Fixed

- **Staged datasets are checked against their pin.** `hf:` sources must carry the download record `snapshot_download` writes naming the pinned commit for every file; `url:` sources must match their declared checksum. `sieval dataset download` records each checksum it verified in `<dataset>/.sieval/download/`, and probes read that record while size and mtime match, else hash once per process. Probes never write into the data dir, so a read-only or shared `SIEVAL_DATA_DIR` works (#179, #183, #187, #190).
- A `sglang_legacy` model with an `infer:` block reaches the engine `sieval run` launched; it used to fall back to the OpenAI SDK default base and fail every request with HTTP 404 (#185, #186).
- An engine `sieval run` launched gets the placeholder key `local` when the config names none, instead of failing at client creation without `OPENAI_API_KEY` or sending whatever that variable held. An explicit key still wins, and an endpoint the run did not launch still reads the SDK's env var (#188, #191).

### Changed

- **BREAKING — a stale staged dataset is refused.** `sieval dataset download` refuses a source whose staged directory holds other bytes, with the reason and a `--force` hint, and deletes nothing. `--force` on an `hf:` source clears the repo directory before fetching, since `snapshot_download` keeps files it did not write. Readiness reports such a source as missing data (`missing[].stale` in JSON, `[stale: …]` in text). `sieval run` / `sieval eval` refuse a dataset `path:` that resolves to a registered source's staging directory holding other bytes; a `path:` outside staging loads as given. A data dir filled by hand now reads `ready: no`: re-run `sieval dataset download <name> --force` (#179, #183).
- **BREAKING — `args.api_base` next to `path` or `infer.checkpoint` is refused** before any engine starts, by `sieval run` and `--dry-run` alike. It used to be ignored by canonical dialects and to bypass the launched engine on `sglang_legacy`. Remove it to evaluate the launched engine (the #185 workaround is no longer needed), or remove `infer`/`path` to evaluate the endpoint. A top-level `api_base` still turns auto-serve off. `EvalSession` also refuses a configured `api_base` that differs from a realized deployment's (#189, #192).

## [0.10.1] - 2026-10-22

A patch release despite two breaking changes, because neither touches an
on-disk format: `meta.json`, `report.json` and the shard records read as
before, `IDENTITY_SCHEMA` stays 1, and a 0.10.0 result directory still resumes
(one retuned serving plan aside, below). What breaks is configuration (the
embedding endpoint must now be set) and the grader digest; a resume that sees a
changed grader digest regrades on the recorded inference, so no sample is
inferred again.

### Added

- **Per-rollout generation logprobs at `n > 1`.** `infer_args: {return_logprobs: true, top_logprobs: N}` used to be refused above `n: 1`; now every rollout's logprobs land in `ModelOutput.choice_logprobs`, one `RolloutLogprobs` per rollout in `texts` order, and the single-sequence fields stay `None`. Supported on `sglang_legacy` (each sample's own `meta_info`, and a reply with a sample count other than `n` is refused), `openai_completions` and `openai_chat` (streaming included). `google_genai` and `openai_responses` refuse it before any I/O. Old records decode with `choice_logprobs=None`. For dialect authors: an `OutputContract` defined outside SiEval needs a `choice_logprobs` rule, or building it raises (#171, #175).
- **`client_timeout: {connect, read, write, pool}`**, in seconds, wherever `max_retries` is accepted (a root model, top level or `args`, and an inline `grader` / `extractor`; a derived model is refused). It is a client setting: never sent in the request body and in no stage identity, so changing it resumes without a rerun or regrade. Defaults are 5 s connect and 600 s for the rest; an engine `sieval run` launched defaults to a 30 s connect. A model's `args.timeout` is still a request argument, not the client timeout (#169, #174).

### Fixed

- `qwen3-30b-a3b` SGLang `mem_fraction_static` drops from 0.90 to 0.87 on A100-80G bf16 and H100-80G bf16/fp8. At tp2×dp4, 0.90 left DP0 too little memory for SGLang's piecewise CUDA graph capture and the server OOMed at startup; that H100 configuration never came up at 0.90. The serving plan is part of the inference identity, so an earlier result for this model on these GPUs that used the recipe default (no hand-pinned value) now reads as a different inference: `--resume` refuses it, and a leaderboard shows it Pending (#168, #172).
- A request timeout's `error_msg` names the phase that timed out, e.g. `Request timed out. (timeout phase: connect)`; `error_reason` is unchanged (#169, #174).

### Changed

- **BREAKING — NL2SH-ALFA and T-Eval no longer default to an internal embedding endpoint.** NL2SH-ALFA (both readings) and T-Eval with `eval_thought=True` need both `SIEVAL_EMBED_API` and `SIEVAL_EMBED_API_KEY` set, and the task fails at construction if either is unset. T-Eval without `eval_thought` needs neither (#167).
- **BREAKING — the grader identity is keyed by file path within its code root, not by import name.** `identity.grader.modules` keys become file paths (`sieval/tasks/gsm8k_0shot_gen.py`, or `./…` for the task's own plugin package), and `identity.grader.task` becomes `<file key>:<qualname>`. Every plugin package the walk reaches is digested whole, whatever its name, namespace packages included. Within the task's own plugin package, absolute self-imports are normalised before hashing and a directory without `__init__.py` digests as having an empty one, so a renamed copy (such as sieval-plugins' `sieval_plugins__<task>__<variant>`) digests the same as the original. Grader digests rotate fleet-wide, built-in tasks included; `IDENTITY_SCHEMA` stays 1, and a resume of a 0.10.0 directory regrades on the recorded inference (#170, #173, #176).
- CI's change detection compares a push to `main` against the previous commit, so a re-run no longer fails on a stale default-branch name (#166).

## [0.10.0] - 2026-10-21

A minor bump for four breaking changes, all in on-disk or CLI-report contracts:
`manifest.json` is gone, every exit path finalizes `report.json` and
`anomalies.json` the same way, a resume compares stage identities instead of
config text, and `sieval infer` reports `engine_pid` instead of `handle_id`. The
theme is RFC #148: `meta.json` now records where a result directory came from,
and a resume reruns, regrades or refuses by which stage changed. Two native
provider dialects (Anthropic Messages, Google GenAI) and auto-started vendored
services land alongside.

### Added

- **`api_key_env: <VAR>`** reads a model's key from an environment variable, wherever `api_key` is accepted (root models and inline `grader` / `extractor`), on every dialect. Setting both is refused; an unset variable fails before any engine starts. Neither the value nor the name is in any identity, so rotating a key or switching forms resumes without a rerun (#148, #163).
- **Per-stage resume.** `--resume` (in place) and `--resume DIR` / `runner_config.resume` (start from another, possibly read-only, run of another task) compare the four stage identities: changed data or prompt reruns those samples, a changed grader regrades on the recorded inference, a changed inference or an unknown stage it would reuse is refused (#157).
- `Task.grader_args()`: task arguments folded into the grader identity, read after `setup()`; a task excludes arguments that cannot change a verdict via `GRADER_ARGS_EXCLUDED` (#157, #158).
- **Stage identities and run history in `meta.json`.** `identity` (schema 1) holds `data`, `prompt`, `inference` and `grader` digests — the inference digest folds in what each sample was actually sent, the grader digest walks the task's MRO, its static imports and plugin packages, and records each judge's model, version, sampling and prompt (never its `api_base`). `runs[]` is append-only, and `meta.json` is rewritten atomically on every run (#150, #154).
- **`dialect: anthropic_messages`** — native Anthropic Messages JSON and SSE, with explicit capability admission, request auditing, replay-safe extended-thinking continuations, and refusals lifted as scoreable terminal outcomes. `max_tokens` is required (#145).
- **`dialect: google_genai`** — native Gemini `generateContent` (`v1` / `v1beta`): streaming, function and hosted tools, thinking, logprobs, seeds, opaque continuation. Google enforces model limits itself, so a provider 400 is the error and a gateway alias skips nothing; a YAML-bound model takes reasoning and logprobs as supported, while tools, structured output, images and continuation still fail reconciliation with `model_outcome_missing` until Google model profiles exist (#147, #153, #156).
- **SiEval starts the services it vendors.** A task declares them in `Task.services`; the ten families that post to `SIEVAL_CODE_EVAL_API` declare `code-eval`. `sieval run`, `sieval eval` and `sieval leaderboard run` start each declared service whose endpoint variable is unset — once per run, on a free port, stopped when the run ends. The started evaluator grades **without a sandbox** and warns so; point the variable at a containerized endpoint to avoid it. Running it with SiEval's own interpreter needs the new `code-eval` dependency group. `identity.grader.services` records each service as `vendored` (commit, code digest, interpreter runtime) or `external` (#155).
- SIGTERM stops `sieval run` and `sieval eval` as Ctrl+C does; exit 143 (#158).
- Google `request_params` record `no_effect` for a thinking budget or summary the reply shows no sign of (#158).

### Fixed

- `effective_config.yaml` and `infer_plans.yaml` write every `api_key` / `authorization` value as `<redacted>`; rerunning such a file is refused until the key is supplied again (#163).
- A resume's `runs[].config` in `meta.json` showed a changed `api_key` by value; it now shows `<redacted>` (#163).
- A `--resume DIR` run killed before `meta.json` recorded it now resumes on the same command, instead of being refused as a run of unknown version. The copy writes `.copying` first and removes it once the run is recorded; the same `--resume DIR` copies again, and `--resume` alone refuses it (`UnfinishedCopyError`) (#162).
- An `.idx` outside a `<iteration>/<stage>/` directory named for a `TaskStage` is neither loaded nor counts as a started run (#162).
- A directory holding shards counts as started whatever `meta.json` says: a fresh run refuses it instead of adopting its samples, and a resume without a usable `meta.json` (a killed v0.6 run, including the latest `outputs/<task>/<timestamp>`, no longer skipped) is refused by the version gate (#160).
- A run stopped mid-way keeps every finished sample: the saver flushes on every exit, samples still in the storage stream are drained to disk, and a failed or cancelled flush re-queues its batch instead of dropping it (#154, #158).
- A run interrupted before every sample passed `preprocess` resumes instead of refusing on `data`; a refused resume appends no `runs[]` entry (#158).
- A fast resume that finds `report.json` from a run that never ended finalizes again instead of reusing it; refinalizing a run whose `end_run` failed writes `anomalies.json` (#154, #158).
- Atomic writes use a unique temp file with fsync, and create `report.json` and `meta.json` as `0o666 & ~umask`, not `0600` (#154, #158).
- The grader identity covers patch packages and whole plugin packages, including modules loaded with `importlib` (#154, #158).
- A resume with another model regrades the recorded inference, and is refused if it would infer any sample (#158).
- The inference requests digest keeps only the request body and a normalized stream flag, so Google's model path or a gateway prefix no longer moves it (#154).
- Google usage: `output_tokens` includes thoughts and `input_tokens` the tool-use prompt, so `reasoning_tokens` is a breakdown of its parent; Google's `error.message` is appended to the HTTP error, so a failed sample's `error_msg` names the reason (#147, #156).
- Google's request envelope is marked explicitly, not detected by a `path` key (#158).
- A bad code-eval copy or interpreter fails before any engine starts; service logs move to `logs/services/<name>/` (#158). LiveOIBench refuses an evaluator whose `/languages` lacks `cpp`, and fails a rollout that could not compile for want of `g++` instead of scoring it 0 (#155).
- `SIEVAL_JUDGE_MODEL_REF` is validated when a runner is built and unmatched roles are warned about; `tunedModels/` is stripped like `models/` (#154, #158).
- `sieval run` no longer collides with an `infer start` of the same model; every handle write goes through one locked transition, and a pending handle stays held while any of its engines is alive (#149, #155).
- The resume benchmarks seed their result directory with a real run, so the `benchmark` gate and `tests/performance/test_resume_speed.py` pass again after per-stage resume; each resume test now asserts it re-infers only the missing samples (#161).

### Changed

- **One resume option, `--resume [DIR]`**, on `sieval run`, `sieval eval` and `sieval leaderboard run`. `--resume` resumes each task in place; `--resume DIR` starts each task from `DIR/<task>`, as `--resume-from DIR` did. A `DIR` that is the run's own directory resumes in place instead of being refused. YAML: `runner_config.resume` (`true` or a task directory). Python: `TaskRunnerConfig.resume` and `resume=` on `EvalSession` / `arun_session` / `run_session`. `resume` never enters `runs[].config` (#164).
- **BREAKING — `manifest.json` is removed.** A sample's state is its latest shard record, found through `.idx`; read per-sample status from the `.idx` columns or the record's `error_*` fields (#159).
- **BREAKING — every exit path finalizes the same way.** `report.json` gains a `sieval` block (`total`, `final`, `failed`, `anomalies`) that decides a fast resume; `anomalies.json` covers every iteration and loses `summary` and its timestamped backups; `runs[].rules_hash` names the rules applied (#159).
- **BREAKING — a resume compares stage identities, not config text.** `effective_config.yaml` and `infer_plans.yaml` no longer refuse a resume and are audit records; `gate_resume_identity` and `_append_resume_note` are removed, and YAML headers keep only the first run's provenance. The version-series gate stays (#150, #157).
- **BREAKING — `sieval infer start/list/show/stop` JSON reports `engine_pid` instead of `handle_id`.** Handle files use `owner_pid`/`engine_pid` (old `pid`/`handle_id` files still load and are rewritten on the next transition). `sieval run` keeps its handles under `~/.sieval/handles/run-<pid>/`. Engine logs move to `<result_dir>/logs/<model>/<role>.log` for `sieval run` with a result_dir, else `~/.sieval/logs/<model>/<role>-<deployment_id>.log`, pruned at launch to the last N per model and role (`--keep-logs` → `SIEVAL_KEEP_LOGS` → 3; 0 deletes on stop; negative keeps all) (#149).
- **Behavior change — `n_truncated` counts every interrupted output**, not just budget stops: Google `MAX_TOKENS`/`CONTINUATION` and candidate policy stops (`SAFETY`, `RECITATION`, …), and Anthropic `refusal` and context-window stops. A new per-choice `finish_reason_sources` records which. Anthropic runs that refuse show higher `n_truncated` than before — not a regression. The rule prose changed, so `rules_hash` rotates fleet-wide to `2c440347324b1edb` (#145, #147).
- `meta.json` records `config` (the session's `effective_config.yaml`) and `runs[].id`, a run ID sortable by start time; the leaderboard scanner reads them instead of walking up the filesystem (#159).
- A failed write of `meta.json` or `report.json` now fails the task (#150). A task named `logs` is refused: `<result_dir>/logs/` is reserved (#155).
- `google_genai` no longer validates Gemini limits by model name: unsupported thinking levels and budgets get Google's own 400, Gemini 3 `candidateCount > 1` is allowed, and Gemini 3 `temperature`/`top_p`/`top_k` are forwarded and recorded (#153, #156).
- User-visible dialect and capability errors describe current behavior instead of citing PR numbers (#146).
- Agent configuration is harness-neutral: `AGENTS.md` is the instruction source of truth with a generated rule map, and skills live under `.agents/skills/` (#144).

### Deprecated

- `--resume-from`, `runner_config.auto_resume` / `resume_from`, the matching `TaskRunnerConfig` fields and the `resume_from=` argument map to `resume` and warn once. An alias that disagrees with `resume` is refused. Removed next release (#164).

### Docs

- Contributor docs say bare `ty check`, which also checks `tests/`, as CI does (#149).

## [0.9.0] - 2026-09-11

A minor bump for four breaking changes, none of which raise on an old input: a
`report.json` key unification, a config-format change where a dialect must now
be named rather than inferred, a request-audit contract that only an out-of-tree
dialect can notice, and two new `ci95_units` rules. The largest theme is
**execution-graded evaluation** — text-to-SQL, shell, C++ and multi-language
code now run and are judged by what they produce, which is what most of the new
infrastructure exists to support.

### Added

- New benchmark tasks & datasets:
    - BFCL v3 (single-turn) — `bfcl_v3_{non_live,live}_0shot_gen` and their `_fc` twins over 3,641 rows, sieval's first native-function-calling protocol. `_fc` is a new variant word: it changes which model capability is exercised, so Prompt and FC are separate tasks the way BFCL's own leaderboard reports them side by side. Multi-turn is not implemented (#143).
    - LiveOIBench — 380 of 403 informatics-olympiad problems graded in C++ with IOI-style subtask partial credit and ranked against the contests' real human contestants. The 23 interactive problems are excluded rather than scored zero, so numbers are **not** comparable to the paper's 403-problem table (#121).
    - Ag-LiveCodeBench-X — LiveCodeBench v5's stdin/stdout subset made language-agnostic, with `language` a required task argument because the benchmark measures a different thing per language (#123).
    - QuoteBench — 56 Bash tasks graded by exact final program state, shipped as **two tasks** because `raw` and `nested_shell` are two command-path contracts, and the paper's finding is that they reverse the order of 5 of 26 model pairs (#140).
    - NL2SH-ALFA — natural language to Bash, execution-graded, as an unqualified task and a `_parse` twin; the gap between the two readings reaches 32% on small models (#139).
    - MultiPL-E — HumanEval in 24 languages and MBPP in 23, with the chat task grading the model's own copy of the prefix, as upstream does deliberately (#138).
    - Spider 1.0 — sieval's first in-process SQL execution substrate, with the hardened reading carried by the unqualified task and the delta measured at 99.903% verdict parity against upstream (#124).
    - Spider 2.0-lite — all 547 questions over three engines; a credential-less run caps near 24.7%, so a per-backend breakdown always ships beside the headline (#125).
    - WikiSQL — upstream's own logical-form protocol, so no model-authored SQL is executed; the grader is anchored on upstream's shipped example predictions (#122).
    - MATH-Perturb — 279 Level-5 problems each under simple and hard perturbation, promoted to `stable` on a measured deepseek-math-7b-rl alignment run (#130).
    - SysBench — 500 Chinese 5-turn sessions, LLM-judged (#103).
    - SciTaRC — composite QA over raw LaTeX tables (#119).
- **Every metric that can carry an interval now carries its own** — `<metric>_ci95`, with a new nested `ci95_units` object saying which population each is clustered on. A headline interval now reaches 52 of 58 reports, and five new population keys ship (`n_versions`, `n_problems_wo_critical_thinking`, `n_subjects`, `n_subsets`, and `n_problems` on SimpleQA-Verified). Purely additive: `score_ci95` and `n_problems` keep their spelling and their values, verified over 3,216 paired emitter comparisons (#126).
- `ifeval` / `multi_if` / `ifbench` `_fixed` variants — seven repaired constraint checkers across the family, each with a quantified score delta (#104, #105).
- `report.json` gains `n_truncated` + `n_scored_rollouts` on every `gen` task, injected by the **runner** so a task cannot report zero by having forgotten to look. Reported as a pair, because the count alone is unreadable (#113).
- `filter` gains composite keys, derived (`callable`) keys, a `values_file`, and a `require_all` flag — each pinned into `effective_config.yaml` by a provenance digest that participates in the strict `--resume` comparison (#106).
- OpenAI Responses API dialect (#141).
- `TaskRequirements.function_tools`, so a task can demand the tools API and be rejected at construction; `ModelOutput.tool_calls` now forwards through the legacy bridge, which is what makes the declaration satisfiable (#143).
- `DEFAULT_REQUEST_TIMEOUT` is declared instead of inherited — openai's SDK and `httpx` disagree by 120x on the default, so a dialect's connection family used to decide how long a generation could take. No behavioural change today; a trap defused ahead of the first native transport (#112).
- Repeated rows record which copy they are, both in the dataset and per sample (#114).
- `iheval` records carry a per-row `key`; `extra` previously named only a cell, so two rows of one cell produced byte-identical records (#111).
- `sysbench` reports every breakdown rate's denominator, and counts what it used to drop (#110).
- `sieval infer start` exposes the readiness budget on `run` (#135).

### Fixed

- **IFBench stopped scoring a biased subset when its NLTK corpora are missing.** `nltk.download` defaults to `raise_on_error=False`, so offline the import completed as if the data were there and the absence surfaced one `LookupError` at a time — landing those samples in `fails` while `denominator_policy: judged` scored the remainder. The published number was computed over the subset whose constraints never touched NLTK (#109).
- A broken grader no longer reads as a wrong answer (#131).
- One task's crash no longer cancels the rest of the batch — a single task raising used to take every sibling runner down with it, losing even tasks that had already finished computing their reports (#117).
- `t_eval_before_calling_0shot_gen` omits an axis with no samples behind it instead of writing `null`, and reports `n_graded` / `n_parsed` so an absent axis can be read (#116).
- A missing split is reported from every dataset transform, not just `filter` (#108).
- A `--resume` no longer aborts on a field explicitly declared non-matching (#137).
- A deploy that never becomes ready says why (#134); a hardware key matches when the GPU reports no memory size (#133).
- Deterministic request seeds are dialect-aware (#118); legacy model provenance is stable (#132).
- A timeout is read off the message prefix rather than by substring match (#142).

### Changed

- **BREAKING — the LLM-judged family's grader-parse metric is unified to `n_grader_unparsed`**, replacing four spellings across five tasks. A consumer reading an old name gets a **missing key**, which reads as "not reported" rather than erroring. `judge` is now confined to upstream contract names and prose, never a record field or metric key (#103).
- **BREAKING — SysBench `ssr` changes meaning under an unchanged key.** It is now the paper's mean normalised count of *consecutive* fully-satisfied turns from a session's start; the old reading reported only the final term and ran ~20 points low. **Same key, different number** — a consumer diffing an old `report.json` against a new one sees a silent 20-point jump, not a missing key. SysBench never shipped a run, so no published sieval number is invalidated (#103).
- **BREAKING (config format) — engine identity no longer selects the request dialect.** `infer.backend: sglang` no longer activates the native `/generate` bypass. A `type: gen` config that relied on the implicit native path must declare `dialect: sglang_legacy` explicitly. Migration is guarded, not silent: the omitted-dialect case raises pre-I/O and names both options (#120).
- **BREAKING — the dialect request-audit contract is source-bound.** `consumed(path, *wire_evidence)` now **requires** evidence, `PreparedRequest.consumed_paths` / `.passthrough` are gone, and `body` is recursively frozen. Both shipped dialects were migrated; a third-party dialect implementing `prepare()` will not load unmigrated (#129).
- **BREAKING for a strict consumer of `ci95_units`** — two rules are now enforced at report-write time: every entry's value must be a population count (`n_…`), and two metrics sharing one interval must publish the same number within 0.005 points. `metric_interval` also raises on a non-count `unit` before it estimates anything, because the corrupted rate would otherwise reach disk presented as a percentage (#126).
- `vendor/code-evaluator` gains four execution sources — C++ (`liveoibench`), `agnostics`, `quotebench`, and a declarative language table backing `cpp` / `bash` / `perl` for MultiPL-E — plus its first stateful route (`POST /shell-evaluations`) and a new `GET /languages` probe, called at `setup()` before any inference so an unsupported language cannot burn a full generation budget and report `pass@1 = 0`. The three existing executors are untouched, so no existing task's grading moves.
- CI's test step cut from 4m41 to 2m07 (#107).

### Docs

- Recipe comments trimmed to what they have to carry (#136).

## [0.8.0] - 2026-08-14

A minor bump, and a wide one: the on-disk record shape, several `report.json`
keys and the infer recipe schema all changed, so **a run started under 0.7.x
cannot be resumed under 0.8.0** — start those fresh. Reported scores are
unchanged everywhere except the two places called out below.

### Added

- New benchmark tasks & datasets:
    - AGIEval v1.1 — all 21 human-exam subsets (19 MCQ + 2 math cloze, 7,272 problems), upstream's two-stage 0-shot protocol, with subset selection via `group` / `subsets` (#67).
    - IHEval — instruction-hierarchy benchmark (18,998 rows, 9 subtasks × 3 settings); headline is the *conflict* aggregate, with reference/aligned and the signed + absolute diffs reported beside it (#92).
    - Multi-IF — sieval's first multi-turn task: 4,501 three-turn conversations across 8 languages, graded by upstream's own multilingual IFEval fork (#80).
    - Inverse IFEval — 1,012 counter-intuitive instruction-following prompts across 8 challenge types, zh/en, LLM-judged (#91).
    - AdvancedIF — 1,645 expert-written prompts with human-curated rubrics. Needs an operator-staged upstream checkout (`SIEVAL_ADVANCED_IF_SRC`); the judge prompts are not redistributable here (#79).
    - ComplexConstraints — rubric-judged complex constraint following (#78).
    - GSM-Plus — every GSM8K test problem rewritten under 8 adversarial perturbations, 10,552 rows (#69).
    - GSM1k — Scale's held-out GSM8K replication, 1,205 rows, as a prompt-exact base/chat pair (#76).
    - UGMathBench — 5,061 problems × 3 randomized versions, EAcc headline, shipped as `ugmathbench_0shot_gen_fixed` (#68).
    - PlatinumBench — 5 math 0-shot tasks over the cleaned subsets; all 120 published math cells of the paper's Table 3 reproduce (#65).
    - MathArena BRUMO / SMT / CMIMC / Apex / Apex Shortlist 2025 — five more final-answer competitions, validated against upstream's published outputs (#73).
    - SciCode — research-coding, 65 problems / 288 tested sub-steps, scored through the code-eval service (#42).
- Uniform stage-output protocol — `preprocess` / `postprocess` / `feedback` return `PromptRecord` / `PredictionRecord` / `JudgementRecord` across **40 of 40** registered tasks, so any sample's prompt, prediction, ground truth and verdict read the same way without a per-task branch. `report.json` metrics verified unchanged by 8,400+ randomized differential trials (#60, #62).
- Shared sampling metrics — `pass@k`, `pass^k`, `avg@n`, `maj@k` (math only), `self_consistency`, `n_unextracted` and `n_short`, rolled out to every sampling task; `pass@k` and `pass^k` are meant to be read as a pair (#88, #89).
- `score_key` + `denominator_policy` on every task report, with a preflight gate — 21 of 49 report-bearing modules had never declared them (#93).
- `TaskMeta.reference_kind` (`value` / `procedure`) plus a `missing_reference` anomaly rule, so a judgement carrying no reference says *why* (#95).
- Run directories record which task produced them — `meta.json` gains an optional `task` block (name, dataset, eval mode, shot count, tags, status), and a new `gate_resume_identity` refuses to resume into a directory another task wrote (#57).
- Rebuilt provider-neutral model binding plane (RFC #25) — capability-based model IR, dialect and transport seams under `sieval/core/models/` (#45), with follow-ups splitting `model.py` by lifetime (#99), giving duplicated private primitives one owner (#98) and sharing auxiliary model-role resolution on `Task` (#97).
- Per-call token breakdown on disk and in `profile.json` — `reasoning_tokens`, `cached_tokens`, `accepted`/`rejected_prediction_tokens` and `reported_total_tokens`, each `int | None` so an absent key means unreported and a present `0` is a real measurement (#101).
- Infer recipes split into hardware and capability layers, so a base checkpoint no longer inherits instruct-only serving params; `sieval infer start` gains `--model-type chat|gen` (#59).
- Task names accept an optional trailing variant segment (`<task>_<N>shot_<mode>[_<variant>]`), so an upstream-faithful port and a corrected one can coexist. The unqualified name always means "what upstream measures"; `_fixed` requires a quantified score delta (#68).
- `pdm.lock` version drift is enforced, not just documented — `check_deps` fails on a version change no requirement change asked for, with a `lock-drift` pre-commit hook (#64).
- The grader's reply is persisted on every LLM-judged attempt, so `judge_unparsed` counts have an artifact behind them (#51).
- Grader spend now appears in `profile.json`; the runner recovers grader calls from the judgement record (#60).

### Fixed

- Resuming no longer raises `KeyError: 'prediction'` on samples whose extraction failed — 39 call sites across 38 tasks, plus a `check_record_key_access` preflight gate. The failure was invisible until the one moment it cost a whole run (#70).
- `theoremqa_kshot_base_gen` no longer executes model output: upstream's bare `eval()` is replaced with a bounded AST allowlist walk. Score impact measured at zero over 706 expressions (#83).
- `gsm8k_0shot_gen` / `hendrycks_math_kshot_base_gen` no longer execute model output — both of DeepSeek-Math's reachable execution paths are closed, guards shared with the UGMathBench grader. Score impact zero across 12,638 gradings (#84).
- A failing CLI command prints a result under `--output json|yaml` instead of nothing; all 15 registered commands now funnel errors through `render()`, and `sieval dataset download` gains `-o/--output` and a result payload (#85).
- `--resume` strict-match aborts say what actually differs, instead of reporting "whitespace / formatting only" while aborting; a cross-version resume now names the version (#87).
- `mmlu_kshot_clp` / `cmmlu_kshot_clp` / `drop_kshot_gen` / `openbookqa_kshot_gen` abort when the few-shot split cannot supply `n_shot` exemplars, before any inference spend, instead of silently rendering fewer (#57).
- `ugmathbench_0shot_gen_fixed` reads predictions with a case-preserving LaTeX parser — **scores move +0.2 pp**; 20 problems fixed, 0 verdicts moved right-to-wrong over 11,424 regraded rows (#90).
- LiveCodeBench uses upstream's per-case timeout rule rather than a whole-suite timeout (#66).
- Usage totals are computed as prompt + completion, and bad builder defaults are rejected at bind time rather than on first `meta()` (#100).
- `get_task_class()` resolves subpackage-hosted tasks, which previously raised `KeyError` from `sieval task show` (#57).
- Two perf gates no longer flake on a gen2 pause or the cyclic collector (#96, #102).

### Changed

- **BREAKING — infer recipe schema.** The flat `profiles[hardware][precision][framework]` bag is split into `hardware[...]` and `capabilities[instruct|base][...]`; the loader **rejects** a `profiles` key with a migration error. `resolve_profile` is replaced by `resolve_hardware_profile` + `resolve_capability_profile`. Resolved gpt-oss params keep identical content but change key **order**, and `infer_plans.yaml` is compared byte-for-byte under `--resume` — which is what makes this a minor bump (#59).
- **BREAKING — the few-shot knob is renamed `k` → `n_shot` on 13 tasks.** `k` meant both a few-shot exemplar count and a pass@k rollout count. Any YAML with `args: {k: N}` on a `*_kshot_*` task must become `args: {n_shot: N}`; the constructor raises rather than silently ignoring the old spelling. Enforced by a new `check_task_shot_knobs` preflight check (#57).
- **BREAKING — `pass@<k>` report column renamed `pass@k`, and `avg@k` renamed `avg@n`.** A run at `k=4` used to emit a `pass@4` key; the budget is now reported once as the `n` / `k` fields. `score` and `pass@1` are unchanged in name and value; a stored leaderboard row keyed on `pass@4` no longer matches (#89).
- **BREAKING — `ChatModel.as_type()` / `GenModel.as_type()` are removed.** Reconcile a target `RuntimeBindingPlan` and use `Model.with_dialect()` for cross-dialect rebinding (#45).
- **BREAKING — `sample_fraction` / `sample_seed` / `sample_by` are removed from `mmmlu_kshot_clp`.** The task's private stratified sampler is replaced by `Dataset.stratified_sample`, which gains a `fraction` budget. Migrate as a dataset `operations:` entry, not task `args:`. **The sampled rows change** — the two samplers seed independently, so only ~10% of rows overlap at `fraction=0.1` (#75).
- **BREAKING — resuming into a result directory produced by a *different* task now aborts** with `ResumeIdentityError`, where it previously handed the second task the first task's report, having evaluated nothing (#57).
- **BREAKING (on-disk records)** — result directories written by 0.7.x hold the old feedback shape and cannot be re-reported by the new `report()`. The resume version gate rejects a 0.7.x → 0.8.x resume with a version message (#60, #62).
- `anomalies.json` is backed up and regenerated on the next run of every task: the rule set grew with `extraction_failure` and `missing_reference`, which rotates `rules_hash` fleet-wide. Visible, harmless (#60, #95).
- Class resolution and model-type derivation move to a new canonical `sieval.cli.resolution`, so the infer CLI no longer imports the eval session sideways (#86).
- The unrequested `pdm.lock` drift from `d805418a` is reverted — 71 packages back on their exact pre-RULER versions, including the CUDA 12 torch tree (#63).
- `sieval/core` is over the 70% mutation bar module-wide (#81); CI now runs integration + acceptance tests (#56).

### Docs

- New `docs/guide/metrics.md` — `report.json` had no documentation at all before this. Covers every key, when each appears, the pairs that must be read together, and the `pass@<k>` → `pass@k` migration (#89).
- The record protocol is scoped to both sides of the contract (#82).

## [0.7.0] - 2026-07-30

### Added

- New benchmark tasks & datasets:
    - RULER — NVIDIA long-context benchmark: datasets + 0-shot generative tasks, with upstream-aligned `string_match_all` / `string_match_part` scoring (#11).
    - Humanity's Last Exam (HLE) — dataset + 0-shot LLM-judge task (#43).
    - BrowseComp — dataset + 0-shot LLM-autorater-graded task (#40).
    - SimpleQA Verified — dataset + 0-shot LLM-autorater-graded task; sieval's first autorater-graded benchmark, headline metric F1 (#38).
    - AA-LCR — dataset + 0-shot LLM-equality-checker-graded task (#44).
    - HMMT Nov 2025 — MathArena-aligned, validated against MathArena's published outputs (#50).
    - MATH (Hendrycks) — full dataset + DeepSeek-Math-aligned base-model task (#31).
    - C-Eval — dataset + few-shot base-model task (#15).
    - ARC-Easy / ARC-Challenge — datasets + PPL and CLP tasks (#36).
    - HellaSwag — dataset + few-shot PPL task (#34).
    - MMMLU — dataset + k-shot CLP base-model task (#30).
    - MMLU — few-shot base-model CLP task, alongside the existing 0-shot generative one (#33).
- Per-record sieval version provenance — every stage record is stamped with the producing version, and `report.json` gains a distinct, semver-sorted `sieval_versions[]` so single-version vs blended runs are visible at a glance; an unstamped scored record surfaces as the `"unknown"` sentinel (#37).

### Changed

- **`--resume` version-compatibility gate** — `--resume` now refuses to resume across an incompatible sieval version series (major post-1.0, minor under 1.0) or a dev/local-build mismatch, and is fail-closed on a missing or unreadable `meta.json`. Exact-match is checked first, so resuming your own build (including dev builds) is never blocked; within a compatible series resume stays allowed and is made auditable via the per-record provenance above — **functional** compatibility is guaranteed, not bit-identical numbers. `meta.json` is now written at run start (create-if-absent), recording the run's originating version. **Breaking (one-time transition):** a run interrupted under a pre-gate version never wrote `meta.json` and so cannot be resumed under the gate — start it fresh (#37).
- CMMLU reclassified from PPL to CLP eval mode, with the task, class, and file renamed to match; the scoring pipeline is byte-identical and scores are unchanged (#32).
- The code-evaluator is now vendored in-tree under `vendor/code-evaluator` instead of a git submodule, so a plain clone of sieval is self-contained. Adds clearer checker mismatch messages, opt-in float tolerance, and SciCode support (#41).

### Fixed

- `sieval[hle]` was uninstallable — `pyproject.toml` declared the `hle` extra but `pdm.lock` never locked the group, so installing it failed with `Requested groups not in lockfile: hle` (#55).
- Dataset subpackage exports are now attributed per module in the generated stub, matching `tasks/`; a stub that attributed an export to the wrong module previously shipped unnoticed, since the existing guards compared only export names (#54).
- SimpleQA Verified now counts pipeline failures as `NOT_ATTEMPTED` in the F1 aggregation rather than silently skewing the score (#39).
- `check_layer_imports` now enforces the cross-package half of the import policy: a relative import that escapes its own package (level ≥ 2) is flagged with the resolved absolute module offered as the fix, and the private-access carve-out is narrowed from `level > 0` to `level == 1` (#46).
- Test isolation fixtures now move a registry and its module cache as a unit behind one shared helper — clearing only one half made registration silently no-op or tripped the duplicate-name guard (#53).
- The T-Eval before-calling task now declares `deps_group="t-eval"`, so `sieval task list` reports it and the pre-run readiness check warns when the extra is missing, instead of failing later with a raw `ImportError` (#35).
- Resume test fixtures now stamp `meta.json` as a real run does, and a dataset perf test that still called the pre-#7 `Dataset.select()` was updated to `slice()` (#55).

### Docs

- README and CLAUDE.md synced with the current task/dataset registry (#35).

## [0.6.0] - 2026-07-06

### Added

- New benchmark tasks & datasets:
    - GSM8K — 8-shot base-model task (#1) and DeepSeek-Math-aligned 0-shot chat-model task (#29).
    - TheoremQA — k-shot base-model task (#3).
    - HumanEval — 0-shot base-model task (#6).
    - CMMLU — few-shot base-model task (#10).
    - MBPP — few-shot base-model task (#12).
    - IFBench — few-shot base-model task (#13).
    - LiveCodeBench — few-shot base-model code-generation task (#14).
    - OpenBookQA — k-shot generative task (#19).
    - AIME 2026 and HMMT Feb 2026 — MathArena-aligned (#16); HMMT Feb 2025 and IMO-AnswerBench (#22).
    - CLP — eval mode and naming category (#23).
- `SglangGenModel` — echoed-input logprobs via the SGLang `/generate` endpoint (#21).
- `stratified_sample` dataset op (#7).
- Dataset source integrity: pinned HF revisions and checksummed URL datasets, enforced in preflight (#8).
- `--resume` now tolerates throughput-only (scheduling) config diffs (#4).
- GitHub Actions CI pipeline and import-time dependency hardening (#2).

### Fixed

- IMO-AnswerBench: normalize during answer extraction, verbatim grader; promoted to stable (#28).
- Pass the gold answer first to `math_verify.verify` (#18, #20).
- Dataset integrity check compares on-the-wire bytes so gzipped responses are not falsely flagged (#17).

### Changed

- Renamed the `select` dataset op to `slice` (#7).
- Sanitize CI check now detects hardcoded absolute paths and scans only tracked files (#5).

## [0.5.0] - 2026-05-06

Initial public release.

### Tasks (11)

Mainstream benchmarks registered in `sieval/meta/index.json`:

- AIME 2024 / 2025 (math competitions)
- DROP (reading comprehension)
- GPQA-Diamond (graduate-level science MCQ)
- HumanEval (Python function synthesis)
- IFEval (instruction following)
- LiveCodeBench code generation (contamination-free coding)
- MATH-500 (advanced math)
- MMLU (multi-domain knowledge MCQ)
- MMLU-Pro (harder MMLU variant)
- T-Eval before-calling (tool-use planning)

### CLI

- `sieval run` / `sieval eval` — run a leaderboard YAML or single-task eval.
- `sieval infer` — start / stop / inspect local inference services (vLLM, SGLang).
- `sieval leaderboard report` — cross-run model × task score matrix.
- `sieval leaderboard list` / `run` — enumerate and execute leaderboard YAMLs.
- `sieval task list|show` / `sieval dataset list|show` — registry discovery.
- `sieval dataset download` — fetch datasets to local cache.

### Eval engine

- Async staged execution engine with sharded persistence.
- Multi-task runner for batch evaluation.
- Strict `--resume` matching (start-fresh or match-invocation, no force-overwrite).
- Bounded retries on failed samples, auto-resume across iterations.
- I/O & stage profiler; iteration / rollout level anomaly detection.
- `pass@k` for code benchmarks.
- Per-result `effective_config.yaml` and `infer_plans.yaml` for reproducibility.

### Determinism

- `deterministic: true` YAML flag + `--deterministic` CLI on `sieval run`, `sieval eval`, `sieval leaderboard run`, `sieval infer start`.
- Pins engine-level batch-invariant kernels (vLLM `VLLM_BATCH_INVARIANT=1`, SGLang `--enable-deterministic-inference`) and injects `seed=0`.
- `meta.json` records the deterministic state.

### Inference

- Local backends (vLLM, SGLang) with recipe-driven auto-resolve.
- Auto DP, unified resolve, fp8 profiles.
- Recipes: Qwen2.5 / Qwen3 / gpt-oss families with H100 / H200 profiles.
- Graceful shutdown (process-group kill prevents orphan GPU processes); STOPPING phase prevents Ready→NotReady regression during stop.

### Leaderboard

- YAML schema supports a top-level `alignment: {card: <path>}` block for user-authored TR-aligned reference cards.
- `sieval leaderboard report` auto-annotates cells with `(Δ<signed> <glyph>)` when a run's `effective_config.yaml` cites an alignment card; tolerance + IEEE-754 slack.

### Registries

- `sieval/meta/index.json` (schema v1) — task / dataset registry, auto-generated via `scripts/sync_meta_index.py`.
- `@sieval_task` / `@sieval_dataset` decorators with `TaskMeta` / `DatasetMeta` schemas.
- AST-based lazy discovery in `sieval.tasks` / `sieval.datasets`.

### Quality

- Layer-boundary import enforcement (pre-commit + preflight).
- Project-wide preflight (`scripts/check_preflight.py`): links, deps, tasks, datasets, imports, examples, meta-index sync, version.
- Tooling: `ruff`, `ty`, `mypy strict`, `pytest`.

[0.13.1]: https://github.com/scitix/sieval/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/scitix/sieval/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/scitix/sieval/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/scitix/sieval/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/scitix/sieval/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/scitix/sieval/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/scitix/sieval/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/scitix/sieval/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/scitix/sieval/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/scitix/sieval/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/scitix/sieval/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/scitix/sieval/releases/tag/v0.5.0
