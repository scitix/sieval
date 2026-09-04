# Code Evaluator

## Overview

A multi-language code execution and testing service. It evaluates model-generated
code either by running it directly or by checking it against test cases.

## Supported datasets

- HumanEval (multi-language: python / javascript / typescript)
- MultiPL-E (cpp / bash / perl, plus javascript via the HumanEval path)
- LiveCodeBench (python only)
- SciCode (python only)

HumanEval and SciCode run the submitted code directly; LiveCodeBench needs a
function name and compares inputs against expected outputs. A SciCode program
carries its own inlined test cases, so a run that raises nothing counts as a pass.

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
```

## Running the service

```sh
fastapi run app/server.py --port 11451
```

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

It answers for the source table, not for the image — a language whose toolchain
is missing from the running container is still listed, and fails at spawn. Read
it as "offered", one step short of "proven".

### Evaluation endpoint

POST /evaluations

Fields:

- `uuid`
- `source`: `"human-eval"` | `"mbpp"` | `"livecodebench"` | `"scicode"`
- `lang`: `python` | `javascript` | `typescript` | `cpp` | `bash` | `perl` —
  ask `GET /languages` rather than hard-coding this list
- `code`: the code, as a string
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

## Layout

- `app/`: service and execution logic
- `docker/`: per-language image files
- `requirements/`: extra dependencies (`livecodebench.txt` / `scicode.txt`)
- `README.md`: this document

## Note

Isolation is not hardened. Do not run untrusted or high-risk code.
