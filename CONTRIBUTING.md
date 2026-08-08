# Contributing to SiEval

Contributions welcome — bug fixes, new benchmarks, documentation improvements. This guide covers setup, conventions, and the PR process.

## Development Setup

```bash
git clone https://github.com/scitix/sieval.git
cd sieval
pdm install -G dev -G test
pre-commit install
```

Verify your setup:

```bash
ruff check sieval/ tests/
ruff format --check sieval/ tests/
ty check sieval/
python -m pytest tests/unit/ -v
```

## Project Architecture

```text
sieval/
├── core/       # Async staged execution engine (zero upper-layer deps)
├── infer/      # Inference service orchestration (depends on core only)
├── tasks/      # Task implementations (100+ benchmarks)
├── datasets/   # Dataset loaders
├── cli/        # CLI entry point (depends on all modules)
├── community/  # Third-party evaluation adaptations
└── probe/      # Quality verification [Planned]

tests/
├── unit/         # Mirrors sieval/ structure
├── integration/  # Mock env, CI-safe
├── acceptance/   # Release gates — alignment/ (TR reproduction records) + performance/ (perf regression)
├── performance/  # Diagnostic benchmarks (on-demand)
└── tasks/        # Manual scripts — real API, NOT in CI
```

### Layer Dependencies

```text
cli → all modules
tasks → core + datasets + community
datasets → core + community
core → zero upper-layer dependencies
infer → core only
```

Do not put task-specific logic into `core/`.

`cli/` depends on every layer, but its sub-packages may not depend on each other:
`cli/infer/` must not import `cli/leaderboard/`. Shared helpers go in a leaf both
import (`cli/resolution.py`). Enforced by `scripts/check_layer_imports.py`.

## Code Conventions

- **Formatter/Linter:** `ruff`
- **Type checker:** `ty` (primary), `mypy` strict (secondary, deferred)
- **Imports:** relative within same package, absolute across packages
- **AI-generated code:** include `AI-Generated Code - <model> (<provider>)` as the last line of the module-level docstring

## Dependencies

Re-lock with `pdm lock --update-reuse`, never bare `pdm lock` — a bare re-lock re-resolves the whole graph, so a one-package change arrives with dozens of unrelated bumps nobody asked for or reviewed.

Preflight compares your lock against its baseline and fails on version changes no requirement change asked for:

```bash
python scripts/check_preflight.py --check check_deps
```

The question it asks of each moved package is whether a `--update-reuse` re-lock could have kept the old pin. If something that did not itself move now forbids the old version — a specifier you raised, or a new dependency's own requirement — the move was forced and passes. If the old pin would still have been valid, nothing asked for the move and it fails.

So if a package genuinely needs a newer version, raise its specifier in `pyproject.toml` in the same commit: that both makes the bump reviewable and is what the check accepts. A changed `requires-python` is the exception — it re-resolves everything, so the check cannot attribute those moves and lists them as needing manual review rather than passing them silently.

## Testing

```bash
python -m pytest -v                                                           # all tests
python -m pytest tests/unit/ tests/integration/ --cov --cov-fail-under=95 -v  # with coverage
python -m pytest tests/acceptance/ -v -s                                      # release gate
ruff check sieval/ tests/ && ruff format sieval/ tests/                       # lint
ty check sieval/                                                              # type check
```

`tests/unit/` must mirror `sieval/` — e.g. `sieval/core/runners/foo.py` → `tests/unit/core/runners/test_foo.py`.

See [tests/README.md](tests/README.md) for mock infrastructure and full command reference.

## Adding a New Benchmark

1. Create dataset in `sieval/datasets/` — keep upstream field names, and cast a column's dtype
   only if the pinned revision requires it (each Task binds 1:1 to its own sample `TypedDict`, so
   uniformity with a sibling loader buys nothing)
2. Create task in `sieval/tasks/` — file naming: `<task>_<N>shot_<mode>[_<variant>].py`
   (see `sieval/tasks/CLAUDE.md`). The unqualified name tracks upstream, bugs included; a
   local correction takes a `_fixed` variant and owes a quantified score impact
3. If the reference implementation repeats sampling (`n_repeats`, `--n 4`) and your task's default
   `n` differs, record that in `reference_impl.notes` with how to match it
4. Add unit tests under `tests/unit/datasets/` and `tests/unit/tasks/` mirroring the source layout
5. Run `python scripts/sync_package_stubs.py` and `python scripts/sync_meta_index.py` to regenerate
   type stubs and the registry, then `--check` both — CI fails on a stale `meta/index.json`
6. Third-party evaluation code goes in `sieval/community/` with proper attribution

## Submitting Changes

1. Fork the repo and create a feature branch
2. Make your changes with clear, focused commits
3. Rebase onto the latest `main` before submitting
4. Open a PR — fill in all applicable sections of the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
5. Link related issues: `Closes #123`, `Refs #456`

For bugs or feature requests, use the [issue templates](.github/ISSUE_TEMPLATE/).
