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
* Mutation score ≥ 70% for modified modules: **`mutmut run`** (the console script)
    * **Never `python -m mutmut`.** That loads `mutmut/__main__.py` as `__main__`, and the
      trampoline mutmut injects then imports it *again* as `mutmut.__main__` — a second execution
      of its top-level `set_start_method('fork')`, which raises `RuntimeError: context has already
      been set` on the first mutant hit. It fails identically for every module, so it reads like a
      broken test rather than a wrong invocation. The console script imports it once, under its
      real name, and works.
    * mutmut ≥ 3 dropped `--paths-to-mutate`; config lives in `[tool.mutmut]` in `pyproject.toml`.
      Scope a run by editing `paths_to_mutate` / `tests_dir` and restoring with
      `git checkout -- pyproject.toml`. Worktrees are fine — it mutates the cwd's tree.
    * **A module whose tests spawn processes cannot be scored to 100%.** In a *spawned* worker
      `mutmut.__main__` is not yet imported, so the trampoline re-executes it and hits the same
      `set_start_method` error; the worker dies and the code under test takes its own fallback
      path. Mutants that only differ *inside* the worker are therefore unobservable — for
      `core/utils/offload.py` that is the residual 22 (e.g. `pool.submit(func, …)` →
      `submit(None, …)`, which the inline fallback answers identically).
    * Do not read a low score as "this code is untestable" before checking. The same module went
      **38.7% → 76.3%** without a single new behaviour, purely by asserting contracts the tests had
      left implicit: that the pool is spawned rather than forked, built once and reused, sized by
      the worker count, not retried after a failed start — and that the degradation warnings name
      the cause, since a diagnostic that drops it is what makes silent fallback silent. Most of
      what looks structural is a missing assertion.
* Disk persistence tests: use fresh `TaskLoader` from disk, not `runner._contexts`
