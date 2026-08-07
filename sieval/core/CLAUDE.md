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
      and the argument is a mutant-name glob. Worktrees are fine — it mutates the cwd's tree.
    * **A module that spawns processes cannot be mutated by mutmut 3.5.** Importing
      `mutmut.__main__` (which the injected trampoline does on its first hit) runs
      `set_start_method('fork')` at import time, so it raises `RuntimeError: context has already
      been set` in any test process where the start method is already fixed. Nothing in the module
      under test causes this — `multiprocessing.get_context("spawn")` leaves the global unset.
      `core/utils/offload.py` is the current example.
* Disk persistence tests: use fresh `TaskLoader` from disk, not `runner._contexts`
