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
      that is the `pytest-cov` plugin, not your change — it reproduces on untouched modules. Use
      `python -m coverage run --source=sieval -m pytest <tests>` + `coverage report -m`.
* Mutation score ≥ 70% for modified modules: **`mutmut run`** — the console script.
    * **Never `python -m mutmut`**: it executes `mutmut/__main__.py` twice (once as `__main__`,
      again via the injected trampoline's `import mutmut.__main__`), and the second
      `set_start_method('fork')` raises `RuntimeError: context has already been set` on the first
      mutant of *any* module. Looks like a broken test; is a wrong invocation.
    * Config lives in `[tool.mutmut]` (mutmut ≥ 3 dropped `--paths-to-mutate`). Scope a run by
      editing `paths_to_mutate` / `tests_dir`, restore with `git checkout -- pyproject.toml`.
      It must keep copying `sieval/__init__.py`, or `mutants/sieval` is not a package and every
      run dies in stats collection — enforced by `check_preflight.py --check check_mutmut_config`.
    * A module whose tests **spawn** processes cannot reach 100%: the trampoline re-executes in
      the fresh worker and hits the same error, so the code takes its fallback path and
      worker-internal mutants are unobservable.
    * A low score usually means missing assertions, not untestable code — `core/utils/offload.py`
      went 38.7% → 76.3% with no new behaviour, only by pinning contracts the tests had left
      implicit (spawn-not-fork, pool reuse, and warnings that name their cause).
* Disk persistence tests: use fresh `TaskLoader` from disk, not `runner._contexts`
