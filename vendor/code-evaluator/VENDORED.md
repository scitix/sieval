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

  Second effect, and the reason it also improves reporting: a **per-case**
  timeout returns normally, so `n_passed` survives it. Only the whole-suite wall
  loses the count, because it kills the worker — measured on a 90-rollout
  LiveCodeBench run, every rollout missing `n_passed` was a timeout and no
  timeout carried one. `CaseTimeout` derives from `BaseException` so that neither
  the submitted code's `except Exception` nor this module's own swallows it.
  Absent the field, behaviour is byte-for-byte what it was. Not yet upstream —
  land in `scitix/code-evaluator` and re-vendor.
- `README.md` — translated from Chinese to English, so the vendored docs match
  the rest of the repo. Content is otherwise unchanged apart from the case-count
  section above.
