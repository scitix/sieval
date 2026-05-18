# Infer — Inference Service Orchestration

**Recipe generator + optional launcher** — not a framework manager.

## Dependency Rules

* Can import from `sieval.core`
* Must NOT import from `sieval.tasks` or `sieval.datasets`

## Key Constraints

* **Translator vs Deployer** — translation (config → launch commands) is stateless and separated from lifecycle management (launch, status, delete, logs).
* Recipes are YAML, English only.
* Lifecycle boundary: SiEval only manages **submit → confirm ready → cleanup**.
* CLI output convention: user-facing output through `sieval.cli.output` helpers; diagnostics stay in loguru at DEBUG level (suppressed unless `--verbose`).

## Recipe Resolution

Recipe, checkpoint, and overrides fall back in priority order; when `overrides` is present it does not block startup (supports the new architecture).
