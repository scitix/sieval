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
- `app/exec_agnostics.py` (new), `app/server.py`, `README.md` — **the Agnostics
  protocol** as `source="agnostics"`, which is what makes a non-Python language
  reachable at all. Nothing executes in this process: the request is forwarded to
  a per-language verifier container over one JSON line in, one JSON line out
  (`{code, timeout_s, test_cases}` -> `{result: "success" | "fail:*", ...}`), and
  only `"success"` is a pass. Upstream of the *protocol* is
  nuprl/Ag-LiveCodeBench-X at `b7b273ef`; sieval's client is
  `ag_livecodebench_x_0shot_gen`.

  Three deliberate choices, each of which a reviewer will want to push back on:

  * **The container command is deployment config, not a request field.** Upstream
    passes `--container-name` on its own CLI, which is safe when the harness and
    the caller are the same process. Here they are not, so a client able to name
    the image could run an arbitrary container on the evaluator host. `lang` is
    all the client sends, constrained to `[a-z0-9][a-z0-9_.+-]{0,31}` because it
    lands in an argv slot, and the command comes from
    `CODE_EVAL_AGNOSTICS_COMMAND`, defaulting to upstream's own podman
    invocation. That is the one place this deviates from upstream's shape rather
    than its behaviour.
  * **The image is pinned by digest, where upstream uses the mutable tag.** The
    verifier decides scores, so it gets the treatment a dataset revision gets:
    `_IMAGE_DIGESTS` maps each of upstream's eight published tags to a digest
    resolved from the registry on 2026-08-23 (each verified against the
    manifest's own `Docker-Content-Digest`). A language with no pinned digest is
    **refused** (`infra:unpinned-lang`) rather than floated — an unpinned
    verifier scores silently, which is the failure the table exists to prevent.
    All eight are single-platform **linux/amd64** manifests, so the pin binds the
    architecture as well; arm64 needs the override. The resolved reference is
    returned as `data.verifier_image` so the verdict's provenance reaches the run
    record, and is `None` under an override whose template does not contain
    `{image}` — reporting a digest that did not run would be worse than
    reporting nothing. Note the tags are **file extensions**, not language names
    (`jl`, `ml`, `f90`), which the framework repo's directory names
    (`executors/julia`) actively mislead about.
  * **`infra:<reason>` is reported instead of upstream's collapse to `"fail"`.**
    Upstream turns every harness-side failure (non-zero exit, undecodable stdout)
    into `result: "fail"` and recovers only the stdin-write case, by matching a
    stderr suffix. The split is named here, where it is known, rather than left to
    a client-side classifier over free text. **It does not change what counts as a
    pass** -- only `"success"` does, either way -- so `pass@1` is unaffected and
    only the diagnostic count differs (sieval's `n_run_errors` is therefore
    broader than upstream's `run_error_rate` numerator).
  * **One number in two roles, kept.** `timeout` is sent as both the container's
    `timeout_s` and the wall the process is held to, because upstream does that
    and widening the wall would move scores. Consequence, confirmed by running it:
    the outer wall is armed first, so the container's own `fail:timeout` is
    effectively unreachable and a timing-out submission surfaces as
    `infra:timeout`. Writing the payload keeps upstream's separate 300s budget
    (`stdin_write_timeout`), since a decoded LiveCodeBench suite is tens of MB.

  Resource stats are the podman *client* process's, not the container's -- the
  existing `monitor_process_resources` watches the pid it spawned. Reported anyway
  so `data` is never null, but do not read them as the submission's cost.

  Verified against a local stub verifier speaking the protocol (no podman on the
  dev box): `success` / `fail:wrong-output` / `fail:error` pass through verbatim,
  and `infra:timeout` / `infra:bad-lang` / `infra:no-test` all fire. Separately,
  on the default (table-driven) path: all four of `lua` / `jl` / `ml` / `f90`
  resolve to their digests and report them, `julia` and `rust` are refused as
  `infra:unpinned-lang`, and an override templated on `{lang}` reports no image.
  Not yet upstream -- land in `scitix/code-evaluator` and re-vendor; tests belong
  there rather than under `tests/`, which mirrors `sieval/`.

  **Re-pinning.** The digests are a snapshot. If upstream rebuilds an image, the
  table keeps scoring against the old one, which is the intended behaviour --
  moving it is a deliberate act that changes scores. Resolve a new digest with
  an anonymous pull token:

  ```bash
  TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:nuprl/agnostics:pull&service=ghcr.io" | jq -r .token)
  curl -sI -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    "https://ghcr.io/v2/nuprl/agnostics/manifests/lua" | grep -i docker-content-digest
  ```
