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
- `app/exec_cpp.py`, `app/server.py`, `docker/Dockerfile.cpp` — **C++ execution**,
  for `source="liveoibench"`. The first non-Python language the evaluator runs as
  a *compiled* artifact: `g++ -std=gnu++17 -Wall -O2 -pipe -static -g`, then one
  child per test case under `RLIMIT_CPU` / `RLIMIT_AS`. Both limits carry
  upstream LiveOIBench's explicit 20% buffer, and a 10 ms poller kills a child
  whose CPU time passes the buffered limit — the rule its `BatchJudge` applies,
  ported rather than reinvented, along with its output comparison (strip;
  single-number outputs compare at `rel_tol=abs_tol=1e-6`; else exact; else line
  count plus stripped per-line match).

  Three differences from upstream LiveOIBench, all documented in the module:
  compilation is bounded (`timeout`, default 60 s) where upstream's
  `subprocess.run` has no timeout; every test always runs, since subtask scoring
  needs the whole verdict vector; and no checker path exists, because the
  published dataset ships no `checkers/` directory, so upstream's own judge
  compares outputs directly on this data.

  `Sample` gains `files` (extra sources compiled alongside — `grader.cpp`,
  `{task}.h`) and `entry_filename`; the test model gains `names`, used only in
  the failure message. `ResourceMetrics` gains `case_verdicts` — one bool per
  case, in request order — because an olympiad subtask scores on its own test
  group, so `n_passed` alone cannot be attributed. All fields are optional and
  the other sources are untouched.

  The test model is still named `LiveCodeBenchTest` though two sources now share
  it; renaming it would widen the diff against upstream without changing the
  wire format.

  `Dockerfile.cpp` exists because the base image has no toolchain: g++ without
  `libstdc++-*-dev` / `libc6-dev` links nothing statically, and every submission
  would fail identically. Verified against g++ 14.2 on Debian: correct, partial,
  TLE, MLE, compile-error, float-tolerance and grader-linked submissions all
  produce the expected verdict vectors. Not yet upstream — land in
  `scitix/code-evaluator` and re-vendor.
