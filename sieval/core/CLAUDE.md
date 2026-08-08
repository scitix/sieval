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

* **Coverage ≥ 95%** — gated in CI (`fail_under = 95` over `sieval/core`). Locally `pytest --cov`
  dies on a pyarrow double-registration in some environments; it reproduces on untouched modules,
  so use `python -m coverage run --source=sieval -m pytest <tests>` + `coverage report -m`.
* **Mutation score ≥ 70%** for modified modules — **currently unobtainable, and not in CI.**
  `mutmut run` (never `python -m mutmut`, which double-executes its `__main__` and dies on the
  first mutant) fails during stats collection: a test that spawns a fresh interpreter re-imports
  mutmut's injected trampoline and fails on it. Scope by mutant name
  (`mutmut run "sieval.core.utils.offload.*"`), never by narrowing `paths_to_mutate`. The copy
  paths are asserted by `check_preflight.py --check check_mutmut_config` — necessary, not
  sufficient. **Do not quote a mutation score until this is fixed.**
* Disk persistence tests: use fresh `TaskLoader` from disk, not `runner._contexts`
