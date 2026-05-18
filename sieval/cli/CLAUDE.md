# CLI — Orchestration Layer

**User-facing entry point** — the only layer permitted to depend on all sieval packages.

## Key Constraints

* All command results go through `CommandResult` → `render()` in `output.py`.
  Do not call `log_user()` for result data. `log_user()` is only for
  progress/streaming (e.g. `infer start` wait, `infer logs`).
* Diagnostics: `logger.info/debug/warning/error()` (standard loguru).
* Call `configure_logging(verbose)` once per command entry point.
* CLI framework: `typer` with `Annotated` type hints. Async via `anyio.run()`.
* Handle `KeyboardInterrupt` gracefully (`sys.exit(130)`).
* Not every CLI command gets a programmatic counterpart — commands that orchestrate subprocesses (`sieval run`, `sieval infer <verb>`) stay CLI-only because OS-level process reaping is load-bearing for cleanup.
