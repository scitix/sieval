# CLI — Orchestration Layer

**User-facing entry point** — the only layer permitted to depend on all sieval packages.

## Key Constraints

* All command results go through `CommandResult` → `render()` in `output.py`.
  Do not call `log_user()` for result data. `log_user()` is only for
  progress/streaming (e.g. `infer start` wait, `infer logs`).
* Every command carries `@cli_command` (from `output.py`), applied *below*
  `@app.command()` so it sits inside what Typer invokes. It funnels an
  escaping exception into a failed `CommandResult`; without it a raising
  command writes zero bytes to stdout under `--output json`. Enforced by
  `TestEveryCommandIsWrapped`.
* Every command takes `-o/--output` and registers a text renderer in
  `_TEXT_RENDERERS`. Streaming-only commands (`infer logs`) are the carve-out
  and must be listed in `TestEveryCommandOffersMachineReadableOutput`.
* Raise `LookupError`/`ValueError` — not `KeyError` — for a failed lookup that
  carries an explanatory message. `KeyError.__str__` is `repr(args[0])`, so a
  sentence reaches the user double-quoted, and the CLI cannot unwrap it
  without also mangling genuine dict misses.
* Diagnostics: `logger.info/debug/warning/error()` (standard loguru).
* Call `configure_logging(verbose)` once per command entry point.
* CLI framework: `typer` with `Annotated` type hints. Async via `anyio.run()`.
* Handle `KeyboardInterrupt` gracefully (`sys.exit(130)`).
* Not every CLI command gets a programmatic counterpart — commands that orchestrate subprocesses (`sieval run`, `sieval infer <verb>`) stay CLI-only because OS-level process reaping is load-bearing for cleanup.
