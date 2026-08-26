# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[0.8.0]: https://github.com/scitix/sieval/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/scitix/sieval/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/scitix/sieval/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/scitix/sieval/releases/tag/v0.5.0
