# Repository Guidelines

SiEval is a **model delivery quality verification system**: eval-side knowledge constraining and verifying the entire model delivery pipeline (training → conversion → infer → evaluation).

## Architecture Principles

1. **AI-friendly** — easy for humans, with enough detail and configurability for AI.
2. **Explicit over implicit** — benefits both humans and AI.
3. **Moderate abstraction** — do not abstract ahead of time. Extract on **coupling** (sites must change together to preserve a contract), not on call count.
4. **SOLID + KISS** — single responsibility, simple and clear, easy to extend.
5. **Minimal change surface + courage to rebuild** — reuse what works; but be willing to start over when a better overall solution exists.

## Reproducibility

Precise reproducibility is a product contract, not a nicety.

- Safety guards (e.g. `--resume` strict match) ship strict-only. No `--force-*` flags, no bypass env vars, no "(future)" escape-hatch hints in errors.
- `--resume` match scope is narrowed, not bypassed: only fields that touch neither sample data nor any persisted artifact may differ across a resume — pure scheduling (concurrency, shard-I/O, write buffers), console-only progress, `auto_resume`, the non-serializable `stage_meta` hooks, and top-level `result_dir`. Everything affecting on-disk content stays strict (sampling/seeds, `max_iterations`, `shard_samples`, `record_*`, `max_retries`, `profile_*`, `detect_anomalies*`, the `progress.json` dump, and `runner_config.result_dir`, which picks the artifact directory), as does `infer_plans.yaml`.
- Recovery: "start fresh" or "match the invocation". Escape-hatch proposals must re-justify the contract, not just add a flag.

## Project Structure & Module Organization

- Core library code lives in `sieval/`.
    - `sieval/core/`: execution engine (models, runners, persistence, profiling).
    - `sieval/datasets/`: dataset wrappers/loaders.
    - `sieval/tasks/`: task implementations by domain.
    - `sieval/community/`: third-party/reference evaluation code; treat as vendor-style code and avoid edits unless explicitly required.
- Layer boundaries (orchestration → engine):

    ```text
    cli/          → orchestration layer, depends on all modules
    infer/        → can depend on core; NOT on tasks/datasets
    tasks/        → depends on core + datasets + community
    datasets/     → depends on core + community
    core/         → zero upper-layer dependencies (independently publishable)
    community/    → third-party evaluation adaptations
    ```

    - **Do not put task-specific logic into `core/`.** Each layer has its own `CLAUDE.md` with layer-specific import constraints.
- Tests are organized by purpose under `tests/`:
    - `tests/unit/`, `tests/integration/`, `tests/acceptance/`, `tests/performance/`.
    - `tests/tasks/` are manual scripts that may call real APIs (never CI).

## Build, Test, and Development Commands

- Platform: Unix (Linux, macOS) — Windows is not supported. Language: Python ≥ 3.12.
- CLI: `typer` — top-level shortcuts `sieval run`, `sieval eval`; resource groups `dataset`, `task`, `infer`, `leaderboard` (e.g. `sieval dataset list`, `sieval infer start`).
- Install deps:
    - `pdm install`
    - Dev/test setup: `pdm install -G dev -G test`
    - Task-specific extras (install on demand): `pip install -e ".[math,ifeval,drop,t-eval,molecule,genomics,beacon]"` (or a subset)
    - Lock procedures: see `.claude/rules/deps.md` (always `pdm lock --update-reuse`, never bare `pdm lock`)
- Run tests:
    - Full suite: `pdm run python -m pytest -q`
    - Unit + integration (coverage gate >=95%): `pdm run python -m pytest tests/unit tests/integration --cov --cov-fail-under=95 -q`
    - Acceptance (release gate): `pdm run python -m pytest tests/acceptance -q -s`
    - Performance (non-stress): `pdm run python -m pytest tests/performance -q -m "not stress"`
- Quality checks:
    - `pdm run ruff check`
    - `pdm run ruff format`
    - `pdm run ty check sieval/`
    - `pdm run python scripts/sync_package_stubs.py --check`
    - `pre-commit run -a`

## Architecture Guardrails

- Preserve the async staged pipeline design: no synchronous rewrite, no global stage barriers.
- Keep resumability semantics intact (`auto_resume`, per-sample failure handling, bounded iterations).
- Persistence is append-only and sharded; disk state is source of truth. Do not replace with monolithic or memory-only storage.
- Respect record-mode behavior: `record_each_stage=True` is the fail-safe default and must remain resume-compatible.

## Coding Style & Naming Conventions

- Python 3.12+, 4-space indentation, PEP 8 naming (`snake_case` functions, `PascalCase` classes).
- Keep modules focused and typed; daily type checking uses `ty` (`ty check sieval/`); `mypy` (`[tool.mypy] strict = true`) is available but secondary.
- Use Ruff for linting/formatting; avoid manual style drift.
- Prefer explicit, readable test case names and concise comments for complex parameterized cases.
- Package stubs in `sieval/tasks/__init__.pyi` and `sieval/datasets/__init__.pyi` are generated; do not hand-edit. Run `scripts/sync_package_stubs.py` after adding/removing exported tasks or datasets.
- AI-generated source files must include `AI-Generated Code - <model> (<provider>)` as the **last line** of the module-level docstring.
- Before deleting code, verify no other call sites depend on it.
- Do not use `from __future__ import annotations`.
- Import policy:
    - Datasets: use package-level imports from `sieval.datasets` for official implementations.
    - Tasks: use package-level imports from `sieval.tasks` for official built-in tasks.
    - Custom/community tasks should prefer full module paths (for example `sieval.tasks.<module>.MyTask`) to avoid naming conflicts.
    - `runners`: canonical path is `sieval.core.runners` (`TaskRunner`, `MultiTaskRunner`).
    - `session`: canonical path is `sieval.cli.leaderboard.session` (`EvalSession`, `arun_session`).
    - `validation`: canonical path is `sieval.cli.validation` (`validate_eval_config`, `run_dry_run`).
    - `resolution`: canonical path is `sieval.cli.resolution` (`resolve_task_class`, `resolve_dataset_class`, `derive_model_type`). Reaching them via `session` resolves, but is not API. `cli/infer/` must not import `cli/leaderboard/`.
    - Within the same package/subpackage: use relative imports (`from .foo import Bar`).
    - Cross-package (including parent): use absolute imports (`from sieval.core.models import ...`).
    - Imports imply public API. Cross-module `from sieval.x.y import _foo` in production code is a smell — promote the name or redesign the call site.
    - Private modules (`_*.py`) are **protected** — accessible only within their own package subtree. Same-package siblings may use `from ._x import Y`; descendants may import via absolute path into the ancestor's private module. Peer-subpackage or out-of-subtree access is forbidden. Tests are the carve-out for both rules.

## Testing Guidelines

- Always run tests after making changes. See `tests/README.md` and `.claude/rules/tests.md` for details.
- `tests/unit/` directory structure must mirror `sieval/` — e.g. `sieval/core/runners/foo.py` → `tests/unit/core/runners/test_foo.py`.
- Framework: `pytest`; async tests must use `@pytest.mark.anyio` (not `pytest-asyncio`).
- `stress` marker is defined in `pyproject.toml`; exclude by default unless intentionally profiling.
- Mutation testing uses `mutmut` (on-demand, not per-commit): mutate `sieval/core` against `tests/unit` (see `[tool.mutmut]`).
- Mutation score should stay `>=70%`; surviving mutants indicate missing assertions and should be killed by improving tests.
- Common mutation commands: `mutmut run`, `mutmut results`, `mutmut show`.
- When changing core runner/model behavior, add/adjust unit tests first, then verify with relevant integration tests.
- Prefer `tmp_path` and deterministic mocks over hardcoded paths and wall-clock sleeps.

## Commit & Pull Request Guidelines

- Always rebase onto the latest `main` before submitting a PR.
- **MUST** read `.github/PULL_REQUEST_TEMPLATE.md` before creating any PR, and fill in all applicable sections. Do not create a PR without reading the template first.
- **MUST** read the matching template from `.github/ISSUE_TEMPLATE/` before creating any issue (bug-report, feature-request, rfc, etc.), and use the template's structure. Do not create an issue without reading the template first.
- Commit messages follow Conventional Commits (enforced by `conventional-pre-commit`):
    - Examples: `feat: ...`, `fix: ...`, `refactor: ...`, `chore: ...`.
- Keep commits scoped and reviewable; include tests with code changes.
- PRs should include:
    - What changed and why.
    - Risk/regression notes.
    - Validation evidence (exact commands run and results).

## Agent Tooling

This file is the single source of truth for shared conventions, and the only
instruction file every supported agent reads. `CLAUDE.md` is a one-line import
of it. Keep it under 32 KiB: Codex truncates project docs past that byte count
silently, so content near the end simply stops applying.

- **Layer and scoped rules** — `sieval/*/CLAUDE.md` and `.claude/rules/*.md`.
  Only Claude Code loads these automatically, and the difference is not
  cosmetic: Codex reads `AGENTS.md` files on the root-to-cwd path and never
  picks up a `CLAUDE.md` below the root — not even when the working directory
  is the very folder holding it. opencode walks *upward* from the working
  directory, so it sees a layer file only while working inside that layer.
  For those two, the 16 layer and scoped rules are reachable solely through
  the generated map below, which asks the agent to read a file rather than
  putting the rule in front of it. Treat a rule as enforced only under Claude
  Code; elsewhere it is a pointer the agent has to follow.

<!-- BEGIN generated: rule-map -->

Read the matching file before editing a path it covers. Claude Code
loads these automatically; every other harness needs this map.

| When editing | Read first |
| --- | --- |
| `sieval/cli/**` | `sieval/cli/CLAUDE.md` |
| `sieval/community/**` | `sieval/community/CLAUDE.md` |
| `sieval/core/**` | `sieval/core/CLAUDE.md` |
| `sieval/datasets/**` | `sieval/datasets/CLAUDE.md` |
| `sieval/infer/**` | `sieval/infer/CLAUDE.md` |
| `sieval/tasks/**` | `sieval/tasks/CLAUDE.md` |
| `sieval/**/*.py`, `tests/**/*.py` | `.claude/rules/code-review.md` |
| `sieval/**/*.py` | `.claude/rules/core-utils.md` |
| `sieval/datasets/**/*.py` | `.claude/rules/datasets.md` |
| `pyproject.toml`, `pdm.lock` | `.claude/rules/deps.md` |
| `CLAUDE.md`, `AGENTS.md`, `sieval/**/CLAUDE.md`, `CONTRIBUTING.md`, `.claude/rules/*.md`, `scripts/check_*.py`, `.pre-commit-config.yaml` | `.claude/rules/engineering-infra.md` |
| `sieval/infer/**/*.py` | `.claude/rules/infer.md` |
| `sieval/tasks/**/*.py`, `sieval/core/tasks/**/*.py` | `.claude/rules/records.md` |
| `scripts/*.py` | `.claude/rules/scripts.md` |
| `sieval/tasks/**/*.py` | `.claude/rules/tasks.md` |
| `tests/**/*.py` | `.claude/rules/tests.md` |

<!-- END generated: rule-map -->

- **Skills** — `.agents/skills/<name>/SKILL.md` is the source of truth, with
  symlinks at `.claude/skills/<name>` and `.opencode/commands/<name>.md`.
  Never edit through a symlink.
- **Post-edit checks** — `scripts/post_edit_checks.py` holds the behavior;
  `.claude/settings.json` and `.opencode/plugins/post-edit.ts` each dispatch to
  it with the edited path as `argv[1]`. There is deliberately no Codex wiring:
  Codex delivers the hook payload as JSON on stdin rather than as an argument,
  so it needs an adapter, and none can be verified here. Run the checks by hand
  under Codex — `python scripts/post_edit_checks.py <path>`.
