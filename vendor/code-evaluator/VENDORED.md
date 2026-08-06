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
- `app/exec_py_test.py`, `app/server.py` — **opt-in per-case timeout**
  (`timeout_per_case`). `timeout` stays a wall for the whole suite; when
  `timeout_per_case` is also given, each case additionally gets its own
  wall-clock budget, armed with `signal.setitimer` around the case body and
  cancelled after it. This is the rule official LiveCodeBench grades by —
  `lcb_runner/evaluation/testing_util.py` re-arms `signal.alarm(timeout)` inside
  the case loop of `grade_call_based` / `grade_stdio`, with
  `codegen_metrics(..., timeout=6)` supplying the default — and it is *not* the
  same as matching the total: a 43-case suite where one case takes 200 s and the
  rest take 1 s fits inside a 258 s whole-suite wall and fails a 6 s-per-case
  one. When only `timeout_per_case` is supplied, the server derives the suite
  wall as upstream's own backstop, `(timeout_per_case + 1) * n + 5` — the shape
  `check_correctness` joins its worker at.

  `compile_code` runs on the same clock, which is also where upstream arms it.
  That matters on the call-based path, where the `exec` runs the submission's
  module-level statements: a hang there is inside no test case, so an
  execution-only budget would let it through to the whole-suite wall — the one
  outcome this patch exists to avoid. The vendored `compile_code` had kept a bare
  `try/finally: pass` where upstream cancels its alarm; the guard restores what
  that `finally` was for.

  One deliberate difference remains: upstream cancels the alarm the moment the
  submission returns and compares outputs off the clock, whereas the guard here
  wraps call and comparison together. Ours is therefore marginally stricter. The
  comparison is a line-wise `Decimal` walk, microseconds against a 6 s budget, so
  no verdict is expected to turn on it — splitting the guard was judged not worth
  threading it through `_unsafe_execute_fn_call` / `_unsafe_execute_stdio`.

  Second effect, and the reason it also improves reporting: a **per-case**
  timeout returns normally, so `n_passed` survives it. Only the whole-suite wall
  loses the count, because it kills the worker — measured on a 90-rollout
  LiveCodeBench run, every rollout missing `n_passed` was a timeout and no
  timeout carried one. `CaseTimeout` derives from `BaseException` so that neither
  the submitted code's `except Exception` nor this module's own swallows it.
  Absent the field, behaviour is byte-for-byte what it was. Not yet upstream —
  land in `scitix/code-evaluator` and re-vendor.

  Tests for the guard belong in that repo, not here — `tests/` mirrors `sieval/`,
  and a copy under this tree would be orphaned by the next re-vendor. Thirteen of
  them (seven in-process on the guard's semantics, six end-to-end through a spawned
  worker, including the compile-time budget) were written against this patch and
  passed; they are recoverable from `tests/unit/vendor/` at commit `7c426a69` to
  port upstream alongside it.

  The field stays optional on the API, so an unpatched client is unaffected — but
  sieval's own LiveCodeBench tasks now always send it, at upstream's 6 s. It is
  the rule the benchmark defines, so it is the rule they grade by; there is no
  whole-suite knob left to pick instead.

  **Expect this to LOWER a score, not raise one.** Per-case is a different rule,
  not a looser one, and it bites in both directions: a submission whose cases are
  uniformly slowish now passes where the shared wall failed it, but one that is
  fast overall with a single slow case now fails where the shared wall let it
  through. Re-grading a recorded 90-rollout lane at 6 s/case: **88 unchanged, 2
  pass → fail, 0 fail → pass, net −2.22 pp.** Both regressions had completed
  *every* case inside the old wall (s11 42/42 in 114 s, s57 44/44 in 118 s) and
  own at least one case over 6 s. LiveCodeBench numbers recorded before this
  landed are not comparable with numbers recorded after it — re-baseline rather
  than expecting the timeouts back.
- `README.md` — translated from Chinese to English, so the vendored docs match
  the rest of the repo. Content is otherwise unchanged apart from the case-count
  section above.
