# Vendored: code-evaluator

- Upstream: <https://github.com/scitix/code-evaluator>
- Vendored at commit: `e4802268f2b491c7ea3d7ed7704dd8582bc079be`
  (previously a git submodule at `submodules/code-evaluator`)

## Two shell routes, and which one a third should copy

There are now two ways to grade a Bash command here, added independently
(NL2SH-ALFA first, `quotebench` second). They are **not** redundant, and neither
should be folded into the other — but a third shell benchmark must not invent a
shape without reading this first.

| | `POST /shell-evaluations` (NL2SH-ALFA) | `POST /evaluations`, `source="quotebench"` |
| --- | --- | --- |
| state | **stateful** — one git-committed baseline tree, reset between commands | **stateless** — a fresh fixture per attempt, built in Python |
| where the environment comes from | the **image** (five of them, `NL2SH_FS_ID` baked in) | the task's own `setup(tmp)`, any GNU userland |
| unit of work | a command **pair** (model + gold) against one tree | one task id plus one command |
| what comes back | **facts** (`ShellFacts`), the caller scores | a **verdict** plus upstream's failure class |
| why | equivalence needs an embedding model, and this service holds no credentials | the check is an exact final-state comparison, so it is decidable here |
| concurrency | `--workers 1`; the tree is shared mutable state | safe in parallel (measured: 672 executions at 32-way, no verdict moved) |

The split is forced by the benchmarks, not chosen: a route that returns facts
cannot return a verdict without credentials it does not have, and a route whose
environment is a per-attempt temp dir cannot be pinned to a baked image without
losing the property that makes it parallel. So the rule for a third one is to
pick the row it matches rather than to add a column:

- graded by **what the filesystem became**, with a prepared tree it cannot build
  itself → extend the shell route;
- graded by an **exact, decidable check** over a fixture the task constructs →
  add a `source` on `/evaluations`.

They are also not co-deployable: `quotebench` needs `/tmp`, which the NL2SH
images delete (see that entry below), so the two never share an image.

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

  The **wall clocks themselves diverge from upstream**, in both directions, and
  the tasks' `reference_impl.notes` enumerate it. Upstream gives every step a
  flat 15s — each `eval_<lang>.py` call is a bare `safe_subprocess.run` taking
  its default, the build included. Here an interpreted row keeps the service's
  own 3s run default rather than tripling every other benchmark's wall to match
  one, and a build gets 60s. So a submission needing between 3s and 15s to run
  fails here and passes upstream, and a compile upstream abandons at 15s
  completes here. The run wall has measured headroom rather than an assumed one:
  a pure-bash prime sieve to n=20000 finishes in 0.6s.

  A **timed-out program is killed by process group** — spawned with
  `start_new_session=True` and reaped with `os.killpg(..., SIGKILL)` — which is
  what upstream's own `safe_subprocess` does, and for the reason its comment
  gives ("Without this line, test_fork_once fails"). Killing only the direct
  child leaves the wall unenforced whenever the submission forks: measured
  before the fix, `bash` running a program that spawns a 30s sleeper returned
  after **30.1s against a 3s budget**, and the sleeper outlived the kill. After
  it, 3.1s and no survivor. The reap is itself bounded (`KILL_GRACE_SECONDS`),
  so a process wedged in uninterruptible sleep cannot re-introduce the unbounded
  wait. `exec_js` / `exec_ts` have the same shape and are **not** fixed here —
  they are untouched by this patch on purpose, and their fix belongs in its own
  change.

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
- `app/server.py`, `app/exec_lang.py`, `README.md` — **`GET /languages`**,
  advertising the `lang` values a deployment accepts. A caller probes it
  *before* spending inference: without it, an unsupported language is
  discoverable only per sample, by which point every sample is generated and the
  report reads as a model that scored zero rather than an evaluator that cannot
  run the language.

  It answers for the **image**, not the source table: `toolchain_present`
  resolves each row's entry command on `PATH` (the *build* command where there
  is one — a compiled row's `run` names the compiler's output, not anything
  installed), and an absent one is withheld. `POST /evaluations` applies the
  same test, so a missing toolchain reads as ``not supported language: cpp (row
  exists but `g++` is not on PATH in this image)`` rather than a per-sample
  `FileNotFoundError` that looks like the model's fault. Without it the guard
  only moved the silent-zeros failure from "language not in the table" to "row
  present, compiler absent" — reachable by deploying any of the other
  Dockerfiles with this server.

  An existence check, not an invocation: ~0.1 ms for the whole table, so nothing
  is compiled to answer a probe. A broken install still fails per sample.

  The two entries above are **not yet upstream** — land them in
  `scitix/code-evaluator` and re-vendor. Their tests belong there rather than
  under `tests/`, which mirrors `sieval/`.
- `app/exec_sh.py`, `app/server.py`, `docker/Dockerfile.nl2sh-{1..5}`,
  `docker/nl2sh/`, `tests/test_exec_sh.py`, `README.md` — **NL2SH-ALFA support**:
  a stateful shell route (`POST /shell-evaluations`) plus the five images that
  carry its prepared filesystems.

  This is the first route here that is stateful, and the first that needs a
  *specific* image rather than any Python. NL2SH-ALFA scores a Bash command by
  what it did to a prepared filesystem, so the unit of work is a command pair run
  against one git-committed baseline tree with a reset in between, and that tree
  is the image. Hence five images (four Ubuntu, one Alpine — its BusyBox
  coreutils print differently, so its 18 samples cannot share the others),
  `NL2SH_FS_ID` baked into each, a request for another id refused, and
  `--workers 1` in every `CMD`.

  The Dockerfiles are upstream's own
  (`westenfelder/InterCode-ALFA@2d3a6947 src/icalfa/assets/docker/`) verbatim,
  with only the two COPY sources reprefixed for this repo's build root, then a
  service block appended. That prefix is checked line-by-line rather than
  asserted. The service is installed under `/opt`, which upstream's
  `docker.gitignore` already excludes — so the baseline commit cannot see it and
  `git clean -fd` cannot delete it mid-run.

  Three behaviours are reproduced deliberately even though they read like
  defects, because the published numbers depend on them: upstream's `clean_cmd`
  wraps a command in double quotes and docker-py `shlex.split`s the result (which
  rewrites 37 of upstream's 300 golds and truncates one into a syntax error);
  `exec_action` resolves the argument of any model reply starting with `cd`
  against the working directory, so `cd ~` executes as `cd /~` and fails; and
  `md5deep` — which no image installs — is still what the hash command picks for
  a path with no dot. Four divergences are deliberate and documented in
  `app/exec_sh.py`: one container with a reset before each command instead of two
  peer containers; the 10 s wall applied to the gold command as well as the
  model's, reported as `gold_timed_out` so a caller can show it never bound; a
  reply starting with `cd` but carrying no `"cd "`, which upstream scores 0
  outright and this reports as a command that did not execute; and hashing on the
  second side restricted to the first side's changed paths, which leaves the
  scored set identical because upstream only ever hashes the intersection.

  Note that `/tmp` is not in upstream's `docker.gitignore` and is empty when the
  baseline is committed, so the first `git reset --hard; git clean -fd` deletes
  the directory outright and it does not come back (measured against a throwaway
  tree, not assumed). That is upstream's behaviour too and must not be "fixed" —
  adding `tmp` to the ignore file would change what `git status` reports and
  therefore what the benchmark scores. It does mean the stateless routes that use
  `tempfile` (`exec_py_code`, `exec_js`, `exec_ts`, and `quotebench`, whose
  `run_attempt` allocates a fresh `mkdtemp` per attempt) are unreliable on these
  five images; they are not served there. `quotebench` was added to that list
  when it landed after this entry — the failure is not subtle there, since a
  missing `/tmp` makes `mkdtemp` raise before any command runs, but it would
  read as a broken grader rather than as a route on the wrong image.

  The verdict is deliberately *not* computed here — it needs an embedding model
  when the outputs differ, and this service holds no model credentials.

  `tests/test_exec_sh.py` (26 tests, all passing) drives the whole protocol
  against a throwaway git repo via `NL2SH_FS_ROOT`, so reset/status/hashing/the
  quoting rewrite/the wall/the fs_id refusal are all covered with no container.
  They live in the vendored tree rather than under sieval's `tests/` (which
  mirrors `sieval/`) precisely so they travel upstream with the code. Not yet
  upstream — land in `scitix/code-evaluator` and re-vendor.
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

  **That flat model widens every other source's response**, which is the price
  of the above and is called out here because it is otherwise discovered by
  diffing artifacts. `ResourceMetrics` is shared, so a `human-eval` verdict now
  carries `"error_class": null, "exit_code": null, "timed_out": null,
  "scenarios_digest": null` alongside its own fields — and sieval's tasks
  persist it, because they bucket whatever they do not recognise into a
  catch-all (`resources = {k: v for k, v in data.items() if k not in
  ("n_cases", "n_passed")}`). Six modules do that — `human_eval_0shot_gen`,
  `human_eval_0shot_base_gen`, both `livecodebench_code_generation_*`,
  `mbpp_kshot_base_gen` and `multipl_e/_base.py` — so their rollout records gain
  four always-null keys from this commit onward. (`scicode_0shot_gen` reads
  named fields and is unaffected.) Nothing reads them and no score moves; it is
  a record-shape change, not a behavioural one. Narrowing it would mean either a per-source response model (stripped, as
  above) or `response_model_exclude_none`, which would also drop `n_cases` /
  `n_passed` — and `None` there means *unknown*, not zero. Accepted rather than
  worked around.

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
    `execute_quotebench` directly — and, beside it,
    `test_exec_quotebench_anchor.py` replays the **whole 224-execution crossover
    grid** through the same entry point, requiring agreement with upstream's
    recorded verdicts on both `passed` and failure class. Neither exercises HTTP
    or pydantic; together they take about a second.

    The grid is there because the oracle sweep is weaker than it looks: it
    covers the `raw` contract only, and never grades a real model reply. A
    wiring bug that ignored `contract` and graded every nested sample as raw
    passed the entire suite green before this was added, and lands at 158/224
    against the grid.
  - *Whole HTTP path, run locally.* The same replay against a live
    `uvicorn app.server` (HF `lsamc/QuoteBench-Rollouts` @ `69957a53`) reaches
    the same **224/224 on `passed` and 224/224 on failure class**. What this
    adds over the in-CI grid is exactly the transport: pydantic validation and
    the declared response model. `tests/acceptance/quotebench/` skips when no
    server is reachable, so **that layer alone** has no standing CI gate —
    closing it would mean a `TestClient` test, and `fastapi` is the evaluator's
    dependency rather than sieval's, so it is not importable from `tests/unit/`.
    The arm file's hash pin and the published-row recompute do run on every
    push, since neither needs a server.

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
  `docker`). `Dockerfile.quotebench` pins upstream's base digest and its seven
  GNU packages, because QuoteBench scores BSD and GNU separately and the
  published crossover table is the GNU replay. **That userland has been run**:
  the layers were replayed into a udocker/PRoot container (no Docker daemon
  available here) from the pinned base *by digest* — `debian:stable-slim` has
  since moved, so a tag pull would grade elsewhere — and the anchor reaches the
  same 224/224 inside it as on the host, which is what promoted the sieval tasks
  to `stable`. `docker build` itself stays unexercised; the reproduction recipe
  is in `tests/acceptance/quotebench/README.md`.
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
    The refusal covers the **override path too**, where it originally did not:
    a template containing `{image}` is asking this table for a digest, so an
    unpinned language raises there as well instead of substituting an empty
    string into the argv. That distinction matters because podman is the one
    runtime that needs no override — every other deployment, upstream's own
    `apptainer` included, runs the branch the guarantee used to skip. A template
    *without* `{image}` names its own image and is still left alone.
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

  Verified against the **real** `ghcr.io/nuprl/agnostics` lua verifier at its
  pinned digest, run under `udocker` (2026-09-11; the dev box has no podman, see
  the runtime note below): `success` / `fail:wrong-output` / `fail:error` come
  back verbatim from the container, `infra:timeout` / `infra:bad-lang` /
  `infra:no-test` all fire, and `data.verifier_image` carries the digest that
  ran. A hand-written correct Lua solution scored `success` on a real decoded
  LiveCodeBench suite while a deliberately wrong variant scored
  `fail:wrong-output`, which is what rules out the decode and whitespace paths.
  On the digest table: `lua` / `jl` / `ml` / `f90` resolve and report, `julia`
  and `rust` are refused as `infra:unpinned-lang` on the default path *and*
  under an `{image}` override, and an override templated on `{lang}` reports no
  image. All eight pinned digests still matched the registry on 2026-09-11.
  Not yet upstream -- land in `scitix/code-evaluator` and re-vendor; tests belong
  there rather than under `tests/`, which mirrors `sieval/`.

  **Running the verifier without podman.** `CODE_EVAL_AGNOSTICS_COMMAND` is the
  supported hook, but two things podman gives for free have to be rebuilt. A
  udocker container is a *persistent directory* and the Agnostics harness writes
  the submission to a fixed path in its cwd, so concurrent requests overwrite
  each other's code -- measured 19 of 40 wrong verdicts against one shared
  container, 0 of 40 once each invocation got its own bind-mounted workdir. And
  `udocker run` prints a banner to **stdout**, which this module json-decodes
  whole (as upstream does), so it needs `--quiet`. Point the env var at a
  wrapper that asserts the `{image}` it was handed equals the digest its
  container was built from; otherwise the reported provenance is a label rather
  than a fact.

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
- `app/exec_cpp.py`, `app/server.py` — **C++ execution**, for
  `source="liveoibench"`. The first non-Python language the evaluator runs as
  a *compiled* artifact: `g++ -std=gnu++17 -Wall -O2 -pipe -static -g`, then one
  child per test case under `RLIMIT_CPU` / `RLIMIT_AS`. Both limits carry
  upstream LiveOIBench's explicit 20% buffer, and a 10 ms poller kills a child
  whose CPU time passes the buffered limit — the rule its `BatchJudge` applies,
  ported rather than reinvented, along with its output comparison (strip;
  single-number outputs compare at `rel_tol=abs_tol=1e-6`; else exact; else line
  count plus stripped per-line match).

  Four differences from upstream LiveOIBench, all documented in the module:
  compilation is bounded (`timeout`, default 60 s) where upstream's
  `subprocess.run` has no timeout; every test always runs, since subtask scoring
  needs the whole verdict vector; no checker path exists, because the published
  dataset ships no `checkers/` directory, so upstream's own judge compares
  outputs directly on this data; and a suite with **no** test case is refused
  by name (`status=False`, `"no test cases to run"`) rather than graded. That
  last one is a deployment fault — a half-written materialized directory — and
  the two readings it otherwise gets are both wrong: an empty verdict vector
  scores every subtask at zero, and the failure message has no first failure to
  name, which raised `StopIteration` out of a coroutine.

  `Sample` gains `files` (extra sources compiled alongside — `grader.cpp`,
  `{task}.h`) and `entry_filename`; the test model gains `names`, used only in
  the failure message. `ResourceMetrics` gains `case_verdicts` — one bool per
  case, in request order — because an olympiad subtask scores on its own test
  group, so `n_passed` alone cannot be attributed. All fields are optional and
  the other sources are untouched.

  The test model is still named `LiveCodeBenchTest` though two sources now share
  it; renaming it would widen the diff against upstream without changing the
  wire format.

  **Deploy `docker/Dockerfile.multipl-e`** — there is no separate image for this
  source. The base image has no toolchain, and a `-static` link additionally
  needs `libstdc++-*-dev` and `libc6-dev`, but `apt-get install g++` already
  pulls both: Debian's `g++` metapackage depends on the versioned compiler, which
  depends on the matching `libstdc++-N-dev`, which depends on `libc6-dev`.
  Measured inside the real base rather than reasoned about — on the current
  `python:3.10-slim` (Debian 13.6), `g++` alone brings `libc6-dev 2.41` and
  `libstdc++-14-dev`, `libstdc++.a` and `libc.a` both resolve, and the judge's
  exact link (`g++ -std=gnu++17 -Wall -O2 -pipe -static -g`) builds and runs a
  `bits/stdc++.h` program.

  An earlier `Dockerfile.cpp` named those two packages explicitly and was
  otherwise byte-identical to `Dockerfile.multipl-e` minus its other toolchains.
  It was removed: it pinned `libstdc++-12-dev` while the base has since moved to
  gcc 14, so it installed a dev tree the compiler no longer uses — a second image
  to maintain that was both redundant and drifting.

  Verified against g++ 14.2 on Debian: correct, partial, TLE, MLE,
  compile-error, float-tolerance and grader-linked submissions all produce the
  expected verdict vectors. Not yet upstream — land in `scitix/code-evaluator`
  and re-vendor.

  **Not the same C++ path as `exec_lang`'s `cpp` row above**, and the two are not
  merge candidates. That one is direct-run: one program, one all-or-nothing
  verdict, budgets from the table. This one compiles against the problem's own
  grader, runs one child *per official test case* under that problem's
  `RLIMIT_CPU` / `RLIMIT_AS`, and returns the whole verdict vector — which is the
  only shape IOI subtask scoring can be computed from. They are reached by
  different `source` values and share no code.
