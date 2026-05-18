# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

- `sieval/meta/index.json` (schema v0.1) — task / dataset registry, auto-generated via `scripts/sync_meta_index.py`.
- `@sieval_task` / `@sieval_dataset` decorators with `TaskMeta` / `DatasetMeta` schemas.
- AST-based lazy discovery in `sieval.tasks` / `sieval.datasets`.

### Quality

- Layer-boundary import enforcement (pre-commit + preflight).
- Project-wide preflight (`scripts/check_preflight.py`): links, deps, tasks, datasets, imports, examples, meta-index sync, version.
- Tooling: `ruff`, `ty`, `mypy strict`, `pytest`.

[0.5.0]: https://github.com/scitix/sieval/releases/tag/v0.5.0
