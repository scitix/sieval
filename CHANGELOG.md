# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[0.6.0]: https://github.com/scitix/sieval/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/scitix/sieval/releases/tag/v0.5.0
