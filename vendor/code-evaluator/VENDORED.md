# Vendored: code-evaluator

- Upstream: <https://github.com/scitix/code-evaluator>
- Vendored at commit: `e4802268f2b491c7ea3d7ed7704dd8582bc079be`
  (previously a git submodule at `submodules/code-evaluator`)

## Local patches on top of that commit

Two kinds, and the difference is a decision rather than a status:

- **Upstream-bound** — everything marked "not yet upstream" below. These are
  divergences we would rather not own; they are meant to land in
  `scitix/code-evaluator` and come back by re-vendoring.
- **In-tree by decision** — the `quotebench` source. It is not staged for
  upstream and carries no "not yet upstream" line: sieval owns it here. Edit it
  in place; a future re-vendor of the base commit has to preserve it rather than
  expect it to have been absorbed.

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
- `app/exec_quotebench.py`, `app/server.py`, `quotebench/`,
  `docker/Dockerfile.quotebench`, `README.md` — **the `quotebench` source**
  (in-tree by decision; not staged for upstream).

  A source whose unit of work is a task id plus one command rather than a
  program plus test cases. The task builds its own filesystem fixture in Python,
  the reply runs inside it as a single `bash -c` payload, and the verdict is the
  exact final state — file bytes, argv, JSON, directory contents, or Git
  history. There is no reference command string to compare against, which is why
  none of the existing `exec_*` modules could carry it.

  `quotebench/` vendors upstream QuoteBench's `core`, `scenarios`, `shellesc`
  and `harness` byte-identically from
  [`LeonardNJU/quoteBench`](https://github.com/LeonardNJU/quoteBench/tree/693325a671e65f889e5cd9d83965db9cc3b26dc2)
  @ `693325a6`, Apache-2.0. sieval vendors the first three again under
  `sieval/community/quotebench/` and uses only the prompt-building half; the two
  copies are pinned to each other by `scenarios_digest`, echoed in every
  response, which the calling task asserts before trusting a verdict.

  `ResourceMetrics` gains four optional fields (`error_class`, `exit_code`,
  `timed_out`, `scenarios_digest`) rather than a per-source subclass — FastAPI
  filters the response against the route's declared model, so a subclass's extra
  fields would be silently stripped. The four resource numbers are reported as
  `0.0`: the payload runs in its own process tree, so the in-process monitor
  would report its own idle numbers, not the command's.

  The contract-to-transport mapping lives in `exec_quotebench.py`, not in the
  vendored package. Upstream's `public_cli.command_for_transport` accepts only
  `raw` / `native` / `nested-shell` and raises `ValueError` on `nested` — the
  spelling upstream's own released rollout dataset uses — so upstream's public
  scorer cannot read its own release. We accept the released spellings (`raw`,
  `nested`) and reject the CLI-only ones.

  **Verified at two levels, and they are gated differently** — worth stating
  plainly, because the stronger of the two is the one CI does not run:

  - *Grading core, in CI.* All 56 oracles pass, asserted by
    `tests/unit/vendor/code_evaluator/test_exec_quotebench.py` calling
    `execute_quotebench` directly.
    This runs on every push, and does **not** exercise HTTP or pydantic.
  - *Whole HTTP path, run locally.* Replaying the stored replies of upstream's
    `raw-vs-nested` arm (HF `lsamc/QuoteBench-Rollouts` @ `69957a53`) against a
    live `uvicorn app.server` reproduces the GNU verdicts upstream recorded for
    them **224/224 on `passed` and 224/224 on failure class**, across all four
    crossover cells. This is where pydantic validation and the declared response
    model are in play — but `tests/acceptance/quotebench/` skips when no server
    is reachable, so the response-model layer has no standing CI gate. Adding
    one would mean a `TestClient` test, and `fastapi` is the evaluator's
    dependency rather than sieval's, so it is not importable from `tests/unit/`.

  A protocol error (unknown task, unknown contract, missing kwargs) answers with
  `data=None`; a wrong command answers with `data` present, which is how a
  caller tells them apart.

  `GET /quotebench/digest` returns the same `scenarios_digest` every verdict
  carries, so a client can settle the handshake before it spends anything on
  inference. Read-only and executes nothing; deliberately not folded into
  `/health`, which is source-agnostic.

  The grading call goes through `asyncio.to_thread`. `execute_quotebench` is
  fully blocking — it shells out under `subprocess.run` — and `evaluate` is
  `async def`, so FastAPI runs it **on** the event loop rather than in the
  threadpool it gives a plain `def`; called directly it stalls the whole worker.
  Measured on this box, four concurrent gradings of a `sleep 3` reply while
  polling `/health` as a load balancer would:

  | | direct call | `asyncio.to_thread` |
  | --- | --- | --- |
  | wall clock for the four | 12.02 s (serialized) | 3.01 s (overlapped) |
  | `/health` polls served | 2 | 59 |
  | `/health` worst latency | 11 969 ms | 1.9 ms |

  The stall is not confined to `quotebench`: it is one shared loop, so a slow
  Bash reply also holds up LiveCodeBench and HumanEval grading on that worker.
  Only this source needed the change — `exec_js` / `exec_ts` await
  `asyncio.create_subprocess_exec`, and `exec_py_code` / `exec_py_test` already
  await `asyncio.to_thread(q.get)` over a `multiprocessing.Process` — so the
  fix is the package's own idiom rather than a new one. No verdict changes: the
  224/224 anchor replays identically through the threaded path.

  `QUOTEBENCH_EXECUTOR` selects upstream's executor (`local` default, or
  `docker`). **The image is unbuilt and unrun so far** — no container runtime was
  available where this landed — so the sieval tasks ship `experimental` until it
  has been. `Dockerfile.quotebench` pins upstream's base digest and its seven GNU
  packages, because QuoteBench scores BSD and GNU separately and the published
  crossover table is the GNU replay.
