# SiEval — Project-Wide Guidelines

SiEval is a **model delivery quality verification system**: eval-side knowledge constraining and verifying the entire model delivery pipeline (training → conversion → infer → evaluation).

## Architecture Principles

1. **AI-friendly** — easy for humans, with enough detail and configurability for AI.
2. **Explicit over implicit** — benefits both humans and AI.
3. **Moderate abstraction** — do not abstract ahead of time. Extract on **coupling** (sites must change together to preserve a contract), not on call count.
4. **SOLID + KISS** — single responsibility, simple and clear, easy to extend.
5. **Minimal change surface + courage to rebuild** — reuse what works; but be willing to start over when a better overall solution exists.

## Reproducibility

Precise reproducibility is a product contract, not a nicety.

* Safety guards (e.g. `--resume` strict match) ship strict-only. No `--force-*` flags, no bypass env vars, no "(future)" escape-hatch hints in errors.
* `--resume` match scope is narrowed, not bypassed: only fields that touch neither sample data nor any persisted artifact may differ across a resume — pure scheduling (concurrency, shard-I/O, write buffers) and console-only progress. Everything affecting on-disk content stays strict (sampling/seeds, `max_iterations`, `shard_samples`, `record_*`, `max_retries`, `profile_*`, `detect_anomalies*`, the `progress.json` dump), as does `infer_plans.yaml`.
* Recovery: "start fresh" or "match the invocation". Escape-hatch proposals must re-justify the contract, not just add a flag.

## Toolchain

* Platform: Unix (Linux, macOS) — Windows is not supported
* Language: Python ≥ 3.12
* Package manager: `pdm` — see `.claude/rules/deps.md` for lock procedures
* Formatting & lint: `ruff`; type checking: `ty` (primary), `mypy` (strict mode, secondary)
* Tests: `pytest`
* CLI: `typer` — top-level shortcuts `sieval run`, `sieval eval`; resource groups `dataset`, `task`, `infer`, `leaderboard` (e.g. `sieval dataset list`, `sieval infer start`)

## Layer Boundaries & Dependencies

```text
cli/          → orchestration layer, depends on all modules
infer/        → can depend on core; NOT on tasks/datasets
tasks/        → depends on core + datasets + community
datasets/     → depends on core + community
core/         → zero upper-layer dependencies (independently publishable)
community/    → third-party evaluation adaptations
```

Each layer has its own `CLAUDE.md` with layer-specific import constraints. **Do not put task-specific logic into `core/`.**

## Import Policy

* `datasets`, `tasks`: package-level imports are the official entrypoint (`from sieval.tasks import ...`)
* `runners`: canonical path is `sieval.core.runners` (`TaskRunner`, `MultiTaskRunner`).
* `session`: canonical path is `sieval.cli.leaderboard.session` (`EvalSession`, `arun_session`).
* `validation`: canonical path is `sieval.cli.validation` (`validate_eval_config`, `run_dry_run`).
* Same package: relative imports. Cross-package: absolute imports.
* Imports imply public API. Cross-module `from sieval.x.y import _foo` in production code is a smell — promote the name or redesign the call site.
* Private modules (`_*.py`) are **protected** — accessible only within their own package subtree. Same-package siblings may use `from ._x import Y`; descendants may import via absolute path into the ancestor's private module. Peer-subpackage or out-of-subtree access is forbidden. Tests are the carve-out for both rules.

## Test Rules

* Always run tests after making changes. See [tests/README.md](tests/README.md) and `.claude/rules/tests.md` for details.
* `tests/unit/` directory structure must mirror `sieval/` — e.g. `sieval/core/runners/foo.py` → `tests/unit/core/runners/test_foo.py`.

## PR / Issue Hygiene

* Always rebase onto the latest `main` before submitting a PR.
* **MUST** read `.github/PULL_REQUEST_TEMPLATE.md` before creating any PR, and fill in all applicable sections. Do not create a PR without reading the template first.
* **MUST** read the matching template from `.github/ISSUE_TEMPLATE/` before creating any issue (bug-report, feature-request, rfc, etc.), and use the template's structure. Do not create an issue without reading the template first.

## Code Conventions

* AI-generated source files must include `AI-Generated Code - <model> (<provider>)` as the **last line** of the module-level docstring.
* Before deleting code, verify no other call sites depend on it.
* Do not use `from __future__ import annotations`.
