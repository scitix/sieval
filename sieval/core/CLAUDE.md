# Core — Engine Constraints

SiEval is a **stateful, persistent, async, staged execution engine**.

## Execution Model

* Samples move **stage-by-stage** (no global stage barriers); stages may run concurrently across samples
* Iterations bounded by `max_iterations` and persistable
* Record unit = sample × iteration × stage; disk state is source of truth
* Storage: **append-only, sharded, async-flushed** — shard data is authoritative, metadata (idx/manifest) must be rebuildable
* Failed samples are per-sample and retryable (not infinite)

## Hard Prohibitions

* Do NOT convert to synchronous loop, add global stage barriers, or remove iteration semantics
* Do NOT aggregate all results in memory or replace shard storage with monolithic files
* Do NOT break resume-from-checkpoint behavior
* Do NOT import from `sieval.infer`, `sieval.tasks`, `sieval.datasets`, or `sieval.cli`

## Concurrency

Hierarchical: global (MultiTaskRunner) → task (TaskRunner) → stage → model. `effective_limit = min(all levels)`.

## Test Requirements

* Coverage ≥ 95%: `python -m pytest tests/unit/ tests/integration/ --cov -v`
    * If `--cov` dies on `pyarrow.lib.ArrowKeyError: ... Array2DExtensionType already defined`,
      that is the `pytest-cov` plugin, not your change — it reproduces on untouched modules.
      `python -m coverage run --source=sieval -m pytest <tests>` then `coverage report -m` works.
* Mutation score ≥ 70% for modified modules: `mutmut run "sieval.core.<module>.*"`
    * mutmut ≥ 3 dropped `--paths-to-mutate`; config lives in `[tool.mutmut]` in `pyproject.toml`,
      and the argument is a mutant-name glob. Run it from the **primary checkout**: from a
      worktree it resolves `sieval` through the editable install's own path and collects a
      different tree, failing before any mutant runs.
* Disk persistence tests: use fresh `TaskLoader` from disk, not `runner._contexts`
