# Code Evaluator

## Overview

A multi-language code execution and testing service. It evaluates model-generated
code either by running it directly or by checking it against test cases.

## Supported datasets

- HumanEval (multi-language: python / javascript / typescript)
- MultiPL-E (cpp / bash / perl, plus javascript via the HumanEval path)
- LiveCodeBench (python only)
- SciCode (python only)
- NL2SH-ALFA / InterCode-ALFA (Bash) — **separate route, separate images**
- QuoteBench (bash only; fixture + final-state grading, see below)

HumanEval and SciCode run the submitted code directly; LiveCodeBench needs a
function name and compares inputs against expected outputs. A SciCode program
carries its own inlined test cases, so a run that raises nothing counts as a pass.

NL2SH-ALFA is the odd one out and has its own route (`POST /shell-evaluations`),
because it is **stateful**: it scores a Bash command by what it did to a prepared
filesystem, so the unit of work is a command *pair* run against one git-committed
baseline tree with a reset in between. That tree comes from the image, which is
upstream's, so there are five of them — one per prepared filesystem — and one
service instance hosts exactly one. See "Shell endpoint" below.

## Setup

### Python environment

HumanEval only:

```sh
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

With LiveCodeBench support:

```sh
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements/livecodebench.txt
```

With SciCode support (scientific stack, requires Python >= 3.11):

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements/scicode.txt
```

### Node.js (JS / TS)

Requires Node.js (>= 20 recommended):

```sh
npm install -g ts-node
```

### Docker

```sh
docker build -f docker/Dockerfile.python -t code-evaluator-py .
docker build -f docker/Dockerfile.javascript -t code-evaluator-js .
docker build -f docker/Dockerfile.typescript -t code-evaluator-ts .
docker build -f docker/Dockerfile.scicode -t code-evaluator-scicode .   # Python 3.11
docker build -f docker/Dockerfile.multipl-e -t code-evaluator-multipl-e . # g++ / bash / perl / node
docker build -f docker/Dockerfile.quotebench -t code-evaluator-quotebench .  # GNU userland
```

The five NL2SH images are built the same way and are **not** interchangeable —
each carries a different prepared filesystem, and `NL2SH_FS_ID` is baked in:

```sh
for fs in 1 2 3 4 5; do
  docker build -f docker/Dockerfile.nl2sh-$fs -t code-evaluator-nl2sh-$fs .
done
```

## Running the service

```sh
fastapi run app/server.py --port 11451
```

That runs it directly on the host, which is fine for a trusted local run and is
what the test suites assume. For untrusted submissions, see
[QuoteBench](#quotebench) — the service itself cannot be put on
`--network none`, because it has to be reachable to be called.

The NL2SH images run their own entrypoint (`uvicorn`, **one worker**) and expose
port 11451 each, so a full NL2SH run needs five endpoints up at once:

```sh
for fs in 1 2 3 4 5; do
  docker run -d --name nl2sh-$fs -p 1145$fs:11451 code-evaluator-nl2sh-$fs
done
```

The **caller** must address them per filesystem: pointing a whole run at one
instance grades only that instance's share and has the rest refused. sieval's
tasks read `SIEVAL_SHELL_EVAL_API` as a template and substitute `{fs_id}` per
sample, defaulting to the five ports above; a proxy dispatching on the request
body's `fs_id` works too. The refusal is the backstop, not the routing.

## API

### Health check

GET /health -> `{"status": true, "msg": "healthy"}`

### Language capabilities

GET /languages -> `{"status": true, "msg": "", "data": ["bash", "cpp", ...]}`

Which `lang` values this deployment accepts. A caller runs this **before**
spending inference: without it, an unsupported language is only discoverable
one sample at a time, by which point everything has been generated and the
report reads as a model that scored zero rather than as an evaluator that
cannot run the language.

It answers for **this image**, not the source table: a table-driven language
whose toolchain is not on `PATH` is withheld, and `POST /evaluations` refuses it
the same way (naming the missing command) instead of failing at spawn. A `PATH`
lookup, not an invocation — ~0.1 ms for the whole table. It proves the entry
point exists, not that the toolchain works.

### Evaluation endpoint

POST /evaluations

Fields:

- `uuid`
- `source`: `"human-eval"` | `"mbpp"` | `"livecodebench"` | `"scicode"` |
  `"quotebench"`
- `lang`: `python` | `javascript` | `typescript` | `cpp` | `bash` | `perl` —
  ask `GET /languages` rather than hard-coding this list; ignored by
  `quotebench`, which is Bash-only
- `code`: the code, as a string — for `quotebench`, the model's reply
- `test`: LiveCodeBench-specific test description (`fn_name` / `inputs` / `outputs`)
- `timeout`: float (optional, seconds; defaults below) — a wall for the **whole suite**
- `timeout_per_case`: float (optional, seconds) — budget each test case
  individually, as official LiveCodeBench does
- `memory_limit`: int (optional, MB; default 1024)

HumanEval (Python) example, with a custom timeout and memory limit:

```json
{
  "uuid":"h1",
  "source":"human-eval",
  "lang":"python",
  "code":"print(1+2)",
  "timeout": 5.0,
  "memory_limit": 512
}
```

LiveCodeBench example:

```json
{
  "uuid":"lc1",
  "source":"livecodebench",
  "lang":"python",
  "code":"def add(a,b): return a+b",
  "test":{"fn_name":"add","inputs":["[1,2]","[3,4]"],"outputs":["3","7"]}
}
```

Response:

```json
{
  "status": true,
  "msg": "",
  "data": {
    "avg_cpu_percent": 0.0,
    "peak_cpu_percent": 0.0,
    "avg_memory_mb": 0.0,
    "peak_memory_mb": 0.0,
    "n_cases": 2,
    "n_passed": 2
  }
}
```

On failure, `msg` carries the reason.

### QuoteBench

A different unit of work: a task id plus one command, not a program plus test
cases. The task builds its own filesystem fixture, the reply runs inside it as a
single `bash -c` payload, and the verdict is the exact final state — file bytes,
argv, JSON, directory contents, or Git history. `code` carries the reply; the
task and the transport go in `kwargs`:

```json
{
  "uuid": "qb1",
  "source": "quotebench",
  "code": "printf '%s' \"it's here\" > out.txt",
  "kwargs": {"task_id": "write-file/t1-apostrophe", "contract": "raw"}
}
```

- `kwargs.task_id` — one of the 56 frozen task ids, `"<scenario>/<instance>"`.
- `kwargs.contract` — `raw` (the reply executes verbatim) or `nested` (the reply
  is interpolated, deliberately unescaped, into an outer double-quoted
  `bash -c "R"`). These are the spellings upstream's released rollout dataset
  uses. The CLI-only spellings (`nested-shell`, `native`, `manual-json`) are
  rejected.

`status` is the verdict and `msg` is the check's reason. `data` additionally
carries:

- `error_class` — the failure taxonomy: `pass`, `shell-syntax`, `tool-error`,
  `runtime-error`, `silent-wrong`, `timeout`, `environment-invalid`.
- `exit_code`, `timed_out` — from the executed command.
- `scenarios_digest` — sha256 over the vendored `core.py`, `scenarios.py` and
  `shellesc.py`. A caller holding its own copy of the task definitions should
  assert this matches before trusting the verdict; otherwise the two copies can
  drift and every number still looks plausible.

The four resource numbers are `0.0` here: the payload runs in its own process
tree, so the in-process monitor would report its own idle numbers.

A protocol error — unknown task id, unknown contract, missing `kwargs` — returns
`status: false` with `data: null`. A command that merely did the wrong thing
returns `status: false` with `data` present, so the two are distinguishable.

`QUOTEBENCH_EXECUTOR` picks the executor. A Bash payload is as unbounded as a
Python one, and the two differ in what they isolate:

- `local` (default) — fresh temp dir per attempt, `bash -c` via argv, trimmed
  env, 15 s timeout, and **no network isolation**. Right for oracle validation
  and for replaying stored replies; not for untrusted model output on a host you
  care about.
- `docker` — upstream's `harness.exec_docker`, which runs each attempt as
  `docker run --rm --network none -v <task dir>:/work … quotebench-runner bash
  -c <cmd>`. The isolation is in that argv, not in an operator flag, so it holds
  whenever this executor is selected. Needs a reachable docker daemon and the
  `quotebench-runner` image to exist.

`docker/Dockerfile.quotebench` takes the other route: it sets
`QUOTEBENCH_EXECUTOR=local` and makes the service's own container the boundary,
since nesting a second docker layer would mean handing in a docker socket. Note
that container **cannot** run with `--network none` — it has to be reachable to
be called. To deny it egress while keeping it callable, put it on an internal
network (`docker network create --internal …`) rather than removing networking.

That userland has been exercised: replaying upstream's stored rollouts inside it
reaches the same 224/224 the host executor does, so the pin moves no verdict on
the anchor data. Where no Docker daemon is available, the same image can be
materialized with udocker — the recipe, and the reason the base must be pulled
by **digest** rather than by tag, are in
[`tests/acceptance/quotebench/README.md`](../../tests/acceptance/quotebench/README.md).
The image pins upstream's base digest and GNU tool set on purpose: QuoteBench
scores BSD and GNU userlands separately, and the published numbers are the GNU
replay.

### Case counts

Every evaluation reports `n_cases` / `n_passed`, so a caller can compute a pass
rate without branching on `source`. What counts as a "case" depends on the mode:

- **Test-case-driven** (`livecodebench` with `test`): one case per input. Cases run
  in order and stop at the first failure, so a failing submission's `n_passed` is
  the failing case's index — a real count, at no extra execution cost. It is
  **not** a full pass rate: a submission that fails case 0 and would pass all the
  rest still reports 0.
- **Direct run** (`human-eval` / `mbpp` / `scicode`, or `livecodebench` without
  `test`): the submitted program is a single all-or-nothing case, so `n_cases` is
  1 and `n_passed` is 1 or 0. That is redundant with `status` by construction; it
  is reported anyway so the field is always readable.

`n_passed: null` means the count is genuinely unknown — the subprocess was killed
on timeout, so it never reported one — and never means zero. Both fields are
`null` only when nothing ran at all (unsupported language or source), where `data`
itself is `null`.

## Example call

```sh
curl -X POST http://localhost:11451/evaluations \
  -H 'Content-Type: application/json' \
  -d '{"uuid":"demo","source":"human-eval","lang":"python","code":"print(42)","memory_limit":1024}'
```

## Resource limits and defaults

### Timeout

When a request omits `timeout`, these defaults apply:

- python / js: 3s
- typescript: 5s
- bash / perl: 3s
- cpp: 15s for the program, plus a separate fixed 60s for the compile — two
  budgets, as upstream MultiPL-E gives each step its own. Charging a slow
  compile against the program's wall would fail a correct submission on a cold
  cache; charging it nothing would let a pathological one hang.
- livecodebench: 6s + 2s * number of cases, or `(timeout_per_case + 1) * n + 5`
  when `timeout_per_case` was sent — upstream's own backstop shape

For the table-driven languages (cpp / bash / perl) the wall is enforced against
the whole **process group**: the program is spawned in its own session and a
timeout `SIGKILL`s the group, so a submission that forks cannot outlive its
budget or leave the child running after the response. This is what upstream
MultiPL-E's `safe_subprocess` does. Note that these walls are the service's own
and do not match upstream's flat 15s-per-step in either direction — see
`VENDORED.md`.

### Memory limit

When a request omits `memory_limit`, the default is 1024 MB.

- Python: limits the process address space via `resource.setrlimit`.
- Node.js (JS/TS): limits the V8 heap via `--max-old-space-size`.
- Table-driven languages (cpp / bash / perl): limits the address space with a
  `ulimit -v` shell prologue that `exec`s the program, so its exit code,
  signals and streams pass through untouched. Applied to the program only — the
  compiler is trusted toolchain cost, and a cap the kernel refuses is left
  unset rather than failing the submission. Verified binding: a 1 GiB
  allocation is refused at `memory_limit=256` and succeeds uncapped.
  An address-space cap does **not** suit a managed runtime — a JVM or Go
  program reserves far more virtual space than it commits — so a future `java` /
  `scala` / `go` row needs a cgroup RSS cap or the runtime's own heap flag.

## Shell endpoint (NL2SH-ALFA)

POST /shell-evaluations

```jsonc
{
  "uuid": "42-1234567890",   // caller's correlation id
  "fs_id": 3,                // 1..5; must be the one this instance hosts
  "command": "find /workspace -type f",   // the model's Bash command
  "gold": "find /workspace -type f",      // the graded ground truth
  "timeout": 10.0            // per command; upstream's TIMEOUT_DURATION
}
```

The response carries **execution facts, never a verdict**: both combined
outputs, both raw `git status --short` listings, per-path hash-command stdout for
added/untracked/copied paths, and per-command `*_exit_ok` / `*_timed_out` flags.
Deciding functional equivalence needs an embedding model when the outputs differ,
and this service holds no model credentials, so the caller owns the arithmetic
and the record of what decided each sample.

`status: false` is about the *request*, not the model: a wrong `fs_id`, or a
baseline that would not restore. A command that failed, hung or changed the wrong
files is `status: true` reporting exactly that.

Per request the service resets the tracked tree, runs the gold, snapshots, resets
again, runs the model's command, snapshots, and resets once more. Two things
follow, and both are load-bearing:

- **One worker.** Every request mutates the shared tree. An in-process lock
  serializes them; a second worker *process* would not see it. The images'
  `CMD` pins `--workers 1` — do not raise it.
- **One filesystem per instance.** `NL2SH_FS_ID` is baked into each image, and a
  request for a different one is refused. A misrouted sample would be scored
  against the wrong prepared tree and report a plausible zero with nothing in
  the logs, which on a 300-sample benchmark is indistinguishable from a bad model.

`NL2SH_FS_ROOT` (default `/`) exists so `tests/test_exec_sh.py` can drive the
whole protocol against a throwaway git repo with no container.

## Layout

- `app/`: service and execution logic
- `docker/`: per-language image files, plus `Dockerfile.nl2sh-{1..5}`
- `docker/nl2sh/`: upstream InterCode-ALFA filesystem setup scripts + gitignore
- `requirements/`: extra dependencies (`livecodebench.txt` / `scicode.txt`)
- `tests/`: the shell backend's tests (no container required)
- `README.md`: this document

## Note

Isolation is not hardened. Do not run untrusted or high-risk code.
