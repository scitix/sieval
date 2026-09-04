# Vendored: code-evaluator

- Upstream: <https://github.com/scitix/code-evaluator>
- Vendored at commit: `e4802268f2b491c7ea3d7ed7704dd8582bc079be`
  (previously a git submodule at `submodules/code-evaluator`)

## Local patches on top of that commit

- `app/exec_py_test.py` — clearer checker messages + opt-in float tolerance
  (`CODE_EVAL_FLOAT_TOL`); from fork branch `fix/checker-messages-float-tol`
  (`cfc47d8`), plus a docstring note on `_value_close`'s type-based tolerance.
- `app/server.py`, `README.md`, `requirements/scicode.txt`,
  `docker/Dockerfile.scicode` — SciCode support: `source="scicode"` direct-run
  alias, scientific-stack pins, and a Python 3.11 image.
- `app/exec_py_test.py`, `app/server.py`, `README.md` — test-case progress
  counts on **every** source. `execute_test` additionally returns how many cases
  passed, and the response `data` gains `n_cases` / `n_passed` for both modes:
  one case per input for test-case-driven evaluation, and a single
  all-or-nothing case for a direct run (`human-eval` / `mbpp` / `scicode`, or
  `livecodebench` without `test`), where the pair is 1/1 or 1/0. Reporting it for
  direct runs is redundant with `status` by construction; it is done so a caller
  can compute a pass rate without branching on `source`. Execution behaviour is
  unchanged: the case loop still stops at the first failure, and because cases
  run in order the failing index *is* the passed count, so the count costs
  nothing. `None` means unknown (subprocess killed on timeout), not zero.
  Consumers tolerate both fields being absent, so an unpatched evaluator still
  works. Not yet upstream — land in `scitix/code-evaluator` and re-vendor.
- `app/exec_py_test.py`, `app/server.py` — **per-case timeout**
  (`timeout_per_case`). Each test case gets its own wall-clock budget, armed with
  `signal.setitimer` around the case body and around `compile_code`. That is
  where upstream arms it too: `lcb_runner/evaluation/testing_util.py` re-arms
  `signal.alarm(timeout)` inside the case loop of `grade_call_based` /
  `grade_stdio` and in `compile_code`, with `codegen_metrics(..., timeout=6)`
  supplying the default. Compilation counts because on the call-based path its
  `exec` runs the submission's module-level statements, so a hang there is inside
  no case. (The vendored `compile_code` had kept a bare `try/finally: pass` where
  upstream cancels that alarm.)

  Budgeting the total instead is a *different* rule, not a looser one: a 43-case
  suite with one 200 s case and the rest at 1 s fits inside a 258 s whole-suite
  wall and fails a 6 s-per-case one. `timeout` remains a whole-suite wall, now
  only a backstop; when a client sends just `timeout_per_case` the server derives
  it as upstream's own `(timeout_per_case + 1) * n + 5`, the shape
  `check_correctness` joins its worker at.

  It also fixes a reporting hole: a per-case timeout returns normally, so
  `n_passed` survives it, where the whole-suite wall kills the worker and loses
  the count. On a 90-rollout run every rollout missing `n_passed` was a timeout
  and no timeout carried one. `CaseTimeout` derives from `BaseException` so that
  neither the submitted code's `except Exception` nor this module's own swallows
  it; `_subprocess_target` therefore names it explicitly.

  One deliberate difference remains: upstream compares outputs off the clock,
  whereas the guard wraps call and comparison together, making it marginally
  stricter. The comparison is a line-wise `Decimal` walk — microseconds against a
  6 s budget — so splitting the guard was judged not worth threading through
  `_unsafe_execute_fn_call` / `_unsafe_execute_stdio`.

  **Expect this to LOWER a score.** Replaying 90 recorded rollouts — the stored
  submissions from an earlier run, re-graded without re-generating them — at
  6 s/case: **88 unchanged, 2 pass → fail, 0 fail → pass, net −2.22 pp.** Both
  regressions had finished *every* case inside the old wall (s11 42/42 in 114 s,
  s57 44/44 in 118 s) and own a case over 6 s. Numbers from before this landed
  are not comparable with numbers after it — re-baseline rather than expecting
  the timeouts back.

  The field is optional on the API, so an unpatched client is unaffected; sieval's
  LiveCodeBench tasks always send it, at upstream's 6 s. Not yet upstream — land
  in `scitix/code-evaluator` and re-vendor. Tests belong there rather than under
  `tests/`, which mirrors `sieval/`; thirteen written against this patch passed
  and are recoverable from `tests/unit/vendor/` at commit `7c426a69`.
- `README.md` — translated from Chinese to English, so the vendored docs match
  the rest of the repo. Content is otherwise unchanged apart from the case-count
  section above.
- `app/exec_lang.py` (new), `app/server.py`, `docker/Dockerfile.multipl-e`
  (new), `README.md` — **table-driven languages, for MultiPL-E.** Adds `cpp`,
  `bash` and `perl` to the direct-run path as rows in one declarative table
  (`ext` + build argv + run argv + budgets) behind a single executor, instead of
  a fourth, fifth and sixth copy of the write-file/spawn/wait/decode/classify
  sequence that `exec_py_code` / `exec_js` / `exec_ts` each hand-roll. Those
  three modules are **untouched**, so no existing task's grading moves; the
  `CODE_EXECUTOR_MAP` merely moved to module scope (it was rebuilt per request)
  and gains the table's rows.

  Commands follow upstream MultiPL-E's `evaluation/src/eval_<lang>.py`
  (`g++ -std=c++17`, `bash path`, `perl path`). Two of its conventions are
  carried deliberately:

  - **perl fails on `ERROR` in the output even at exit 0** (`eval_pl.py`).
    Checked only on an otherwise-passing run, so it can turn a pass into a
    failure and never the reverse. Not reachable through the shipped test
    templates — 0 of 161 `humaneval-pl` rows mention `ERROR`, and all 161 signal
    failure with `exit 1` — so it fires only on model stdout. Kept because it is
    upstream's rule, not because the data exercises it.
  - **A build gets its own wall** (60s) separate from the program's, as each
    `safe_subprocess.run` call does upstream. c++ takes upstream's own 15s for
    the program.

  Upstream's four-way failure taxonomy (`SyntaxError` / `AssertionError` /
  `ReferenceError` / `Exception`) is **not** reproduced: the response carries one
  boolean and all four buckets are the same boolean. What is preserved is the
  part a caller cannot reconstruct — build failure vs run failure — spelled in
  `msg`, reusing the vocabulary the hand-rolled modules already established
  (`failed: timeout`, `failed [exit N]: ...`).

  Memory is capped with a `ulimit -v` prologue that `exec`s the program, not
  `preexec_fn`: neither sibling module uses one, and it runs between fork and
  exec inside a threaded server, where a lock another thread holds at fork is
  held forever in the child. Verified binding rather than assumed — a 1 GiB
  allocation is refused at `memory_limit=256` and succeeds uncapped, and a
  trivial program still passes under the cap.

  **Also fixes a latent bug** on exactly this path: for an unsupported `lang`
  the direct-run branch never bound `timeout`, which the `logger.info` below it
  reads unconditionally, so the request died with `UnboundLocalError` (a 500)
  instead of returning the branch's own `not supported language: <lang>`. That is
  the path every MultiPL-E language whose toolchain is not deployed takes —
  20 of 24 today — so it would have been hit immediately. Reproduced before and
  after against the real endpoint.
- `app/server.py`, `README.md` — **`GET /languages`**, advertising the `lang`
  values a deployment accepts. A caller probes it *before* spending inference:
  without it, an unsupported language is discoverable only per sample, by which
  point every sample is generated and the report reads as a model that scored
  zero rather than an evaluator that cannot run the language. It answers for the
  source table rather than for the image (a row whose toolchain is missing is
  still listed and fails at spawn), so it means "offered", not "proven" —
  probing toolchains for real would run every compiler on each health check.

  The two entries above are **not yet upstream** — land them in
  `scitix/code-evaluator` and re-vendor. Their tests belong there rather than
  under `tests/`, which mirrors `sieval/`.
