# Code Evaluator

## Overview

A multi-language code execution and testing service. It evaluates model-generated
code either by running it directly or by checking it against test cases.

## Supported datasets

- HumanEval (multi-language: python / javascript / typescript)
- LiveCodeBench (python only)
- SciCode (python only)
- Ag-LiveCodeBench-X and anything else speaking the Agnostics protocol (any
  language, via a per-language verifier container)

HumanEval and SciCode run the submitted code directly; LiveCodeBench needs a
function name and compares inputs against expected outputs. A SciCode program
carries its own inlined test cases, so a run that raises nothing counts as a pass.

`source: "agnostics"` is the odd one out: nothing runs in this process. The
request is handed to a container that owns the language toolchain, which is what
makes Lua / R / Julia / OCaml / Fortran reachable without installing five
toolchains here. See "Agnostics protocol" below.

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
```

## Running the service

```sh
fastapi run app/server.py --port 11451
```

## API

### Health check

GET /health -> `{"status": true, "msg": "healthy"}`

### Evaluation endpoint

POST /evaluations

Fields:

- `uuid`
- `source`: `"human-eval"` | `"mbpp"` | `"livecodebench"` | `"scicode"` |
  `"agnostics"`
- `lang`: `python` | `javascript` | `typescript`; for `"agnostics"` it is instead
  the **container tag** (`lua`, `julia`, `r`, `ocaml`, `fortran`, …)
- `code`: the code, as a string
- `test`: LiveCodeBench-specific test description (`fn_name` / `inputs` / `outputs`).
  `"agnostics"` requires it and uses only `inputs` / `outputs`
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

## Agnostics protocol

`source: "agnostics"` forwards the submission to a verifier container over one
JSON line in, one JSON line out:

```text
in  { "code": str, "timeout_s": int,
      "test_cases": [ { "input": str, "output": str }, ... ] }
out { "result": "success" }
    | { "result": "fail:wrong-output", "expected": str, "got": str, "stderr": str }
    | { "result": "fail:error", "exit_code": int, "stdout": str, "stderr": str }
    | { "result": "fail:timeout", "stdout": str, "stderr": str }
    | { "result": "fail:other", "stdout": str, "stderr": str }
```

Only `"success"` is a pass. Other result codes and extra fields are allowed by
the protocol, so anything unrecognized is reported as a failure, not an error.
The response `msg` is the container's `result` verbatim, or an `infra:<reason>`
code (`spawn` / `stdin` / `timeout` / `exit <n>` / `decode` / `bad-lang` /
`no-test`) when the container could not be asked — so a caller can tell "the
program is wrong" from "the harness could not run it".

The verdict covers the whole suite, so `n_cases` / `n_passed` are the
all-or-nothing pair (`1/1` or `1/0`); the suite's real size is on the request.

The command is **deployment configuration, not a request field** — a caller able
to name the image could run any container on this host. It defaults to upstream's
own invocation, with the image resolved to a **pinned digest**:

```bash
podman run --rm -i --tmpfs /ramdisk:size=512m,exec {image}
```

The verifier decides scores, so it is pinned the way a dataset revision is: a tag
can be moved under a finished leaderboard without anything on disk changing.
`lang` selects a digest from a table covering upstream's eight published tags
(`lua`, `r`, `python`, `jl`, `java`, `cpp`, `ml`, `f90` — note these are **file
extensions**, so Julia is `jl`, OCaml `ml`, Fortran `f90`). A language with no
pinned digest is **refused** with `infra:unpinned-lang` rather than falling back
to a floating tag; add it to `_IMAGE_DIGESTS`, or take responsibility via the
override. All eight are single-platform **linux/amd64** manifests, so the pin
binds the architecture too — on arm64 the command must be overridden.

The resolved reference comes back as `data.verifier_image` so a caller can record
which verifier produced a verdict. It is `None` under an override unless the
override's template actually contains `{image}`: naming a digest that did not run
is worse than reporting nothing.

Override the whole command with `CODE_EVAL_AGNOSTICS_COMMAND` for a different
runtime (upstream also supports `apptainer run --contain --writable-tmpfs <x>.sif`),
a mirrored registry, or a local test double; both `{image}` and `{lang}` are
substituted. `lang` must match `[a-z0-9][a-z0-9_.+-]{0,31}` since it lands in an
argv slot. `memory_limit` and `test.fn_name` are ignored: the container owns its
own limits, and the protocol has no call-based mode.

## Resource limits and defaults

### Timeout

When a request omits `timeout`, these defaults apply:

- python / js: 3s
- typescript: 5s
- livecodebench: 6s + 2s * number of cases, or `(timeout_per_case + 1) * n + 5`
  when `timeout_per_case` was sent — upstream's own backstop shape
- agnostics: 15s, the value upstream's README uses. It serves as **both** the
  container's own `timeout_s` and the wall the process is held to, as upstream
  sends it — so a suite whose per-case budgets sum past the wall is killed from
  outside before it can report its own `fail:timeout`. Writing the payload gets
  its own 300s budget, since a decoded suite runs to tens of MB and a slow write
  is not a slow program.

### Memory limit

When a request omits `memory_limit`, the default is 1024 MB.

- Python: limits the process address space via `resource.setrlimit`.
- Node.js (JS/TS): limits the V8 heap via `--max-old-space-size`.

## Layout

- `app/`: service and execution logic
- `docker/`: per-language image files
- `requirements/`: extra dependencies (`livecodebench.txt` / `scicode.txt`)
- `README.md`: this document

The `agnostics` source needs no image file or requirements of its own — the
container it delegates to supplies the toolchain, and only a container runtime
(`podman`, or `apptainer` for a `.sif`) has to exist on the host.

## Note

Isolation is not hardened. Do not run untrusted or high-risk code.
