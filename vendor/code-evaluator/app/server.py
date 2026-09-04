import atexit
import os
import sys
from typing import Any, Generic, TypeVar

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from .exec_js import execute_code as exec_js
from .exec_py_code import execute_code as exec_py_code
from .exec_py_test import execute_test as exec_py_test
from .exec_sh import DEFAULT_TIMEOUT as SHELL_DEFAULT_TIMEOUT
from .exec_sh import execute_shell, hosted_fs_id
from .exec_ts import execute_code as exec_ts

# logger
logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "enqueue": True,
        }
    ],
)


# collect non-server loggers
@atexit.register
def exit_handler():
    logger.remove()


app = FastAPI()


# Generic type for response data
T = TypeVar("T")


class BasicResponse(BaseModel, Generic[T]):
    status: bool
    msg: str
    data: T | None = None


class ResourceMetrics(BaseModel):
    """The response ``data`` payload: what the run cost, and how far it got.

    Every evaluation reports case counts, so a caller can compute a pass rate
    without branching on ``source``. What a "case" is depends on the mode:

    * **Test-case-driven** (``livecodebench`` with ``test``): one case per input.
      ``n_passed`` is how many passed before the run stopped. Cases run in order
      and stop at the first failure, so for a failing submission this is the
      failing case's index -- a real count, at no extra execution cost. It is
      *not* a full pass rate: a submission that fails case 0 and would pass the
      rest still reports 0.
    * **Direct run** (``human-eval`` / ``mbpp`` / ``scicode``, or
      ``livecodebench`` without ``test``): the submitted program is one
      all-or-nothing case, so ``n_cases`` is 1 and ``n_passed`` is 1 or 0. That
      is redundant with ``status`` by construction -- it is reported anyway so
      the field is always readable.

    ``n_passed = None`` means the count is genuinely unknown (the subprocess was
    killed on timeout, so it never reported one) -- never zero. Both fields stay
    ``None`` only when nothing ran at all, i.e. an unsupported language or
    source, where ``data`` itself is ``None``.

    Kept as one flat model rather than a per-source subclass: FastAPI filters the
    response against this route's declared model, so a subclass returned from one
    branch would have its extra fields silently stripped.
    """

    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float
    n_cases: int | None = None
    n_passed: int | None = None


@app.get("/health")
async def check_health() -> BasicResponse[None]:
    return BasicResponse(status=True, msg="healthy")


class ShellFacts(BaseModel):
    """What running one command pair against a prepared tree produced.

    Facts, never a verdict. Deciding whether two Bash commands are functionally
    equivalent needs an embedding model when their outputs differ, and this
    service holds no model credentials -- so the caller owns the arithmetic and
    the record of what decided each sample.

    ``*_status`` are the raw ``git status --short`` text, not a parsed structure:
    the caller re-parses them with its own vendored copy of upstream's parser,
    and that copy is what scores. ``*_hashes`` map an added/untracked/copied
    path to the **raw stdout** of upstream's hash command, because upstream
    compares those strings rather than digests -- including when the command is
    ``md5deep``, which no image installs, so both sides carry the same failure
    text and compare equal.

    ``*_hashes`` is the *scored* set, not an inventory: upstream hashes only the
    paths both sides changed, so ``model_hashes`` covers the paths the gold also
    touched and nothing more. A path only one side changed is still reported in
    that side's ``*_status`` -- which is what part 1 reads -- and is simply never
    hashed, because part 2 would not have looked at it. Both maps are therefore
    empty whenever the two commands share no added path, which is the common
    case.

    ``*_exit_ok`` and ``*_timed_out`` describe the commands, not the request: a
    command that failed or hung is a successful evaluation reporting exactly
    that. ``gold_timed_out`` exists because bounding the gold command is this
    service's own divergence from upstream, and a caller has to be able to say
    the bound never bound.
    """

    fs_id: int
    gold_output: str
    model_output: str
    gold_status: str
    model_status: str
    gold_hashes: dict[str, str]
    model_hashes: dict[str, str]
    gold_exit_ok: bool
    model_exit_ok: bool
    gold_timed_out: bool
    model_timed_out: bool


class ShellSample(BaseModel):
    """One NL2SH-ALFA sample: the model's command and the graded gold.

    ``fs_id`` travels with the sample rather than living in this service's
    config alone, so that a sample routed to the instance hosting a *different*
    prepared tree is refused instead of scored. Both are checked: the request
    says which tree it needs, the instance says which it has.
    """

    uuid: str
    fs_id: int
    command: str
    gold: str
    timeout: float = SHELL_DEFAULT_TIMEOUT


@app.post("/shell-evaluations")
async def evaluate_shell(sample: ShellSample) -> BasicResponse[ShellFacts]:
    """Run a command pair against this instance's prepared filesystem.

    Separate from ``/evaluations`` rather than another ``source`` on it, for two
    reasons that are both structural: the payload is a command *pair* against
    shared state where that route's is one self-contained program, and the
    response is execution facts where that route's is ``ResourceMetrics``.
    FastAPI filters a response against the route's declared model, so returning
    these fields through that one would silently drop every last one of them.
    """
    ok, msg, data = await execute_shell(
        fs_id=sample.fs_id,
        command=sample.command,
        gold=sample.gold,
        timeout=sample.timeout,
    )
    logger.info(
        f"evaluate shell sample '{sample.uuid}' on fs {sample.fs_id} "
        f"(hosted: {hosted_fs_id()}), timeout: {sample.timeout}, status: {ok}"
        + (f", msg: {msg}" if msg else "")
        + (
            ""
            if data is None
            else (
                f", gold_exit_ok: {data['gold_exit_ok']}, "
                f"model_exit_ok: {data['model_exit_ok']}, "
                f"gold_timed_out: {data['gold_timed_out']}, "
                f"model_timed_out: {data['model_timed_out']}, "
                f"changed: {len(data['model_hashes'])}/{len(data['gold_hashes'])}"
            )
        )
    )
    return BasicResponse(
        status=ok, msg=msg, data=ShellFacts(**data) if data else None
    )


class LiveCodeBenchTest(BaseModel):
    inputs: list[str]
    outputs: list[str]
    fn_name: str | None = None


class Sample(BaseModel):
    uuid: str
    source: str
    code: str
    test: LiveCodeBenchTest | None = None
    lang: str = "python"
    timeout: float | None = None
    # Per-case budget, applied on top of `timeout`, which stays a whole-suite wall.
    # Opt-in: absent keeps the previous behaviour exactly. Official LiveCodeBench
    # budgets per case rather than per suite -- see `exec_py_test.execute_test`.
    timeout_per_case: float | None = None
    memory_limit: int = 1024  # MB
    kwargs: dict[str, Any] | None = None


@app.post("/evaluations")
async def evaluate(sample: Sample) -> BasicResponse[ResourceMetrics]:
    if sample.source in {"human-eval", "mbpp", "scicode"}:
        # 'human-eval' / 'mbpp' / 'scicode': run the submitted code directly.
        # scicode sends a self-contained program (inlined targets + test cases),
        # so a clean run == pass, same as human-eval.
        logger.debug(f"code to exec:\n{sample.code}")

        CODE_EXECUTOR_MAP = {
            "javascript": (exec_js, 3.0),
            "python": (exec_py_code, 3.0),
            "typescript": (exec_ts, 5.0),
        }
        if sample.lang in CODE_EXECUTOR_MAP:
            fn, default_timeout = CODE_EXECUTOR_MAP[sample.lang]
            timeout = sample.timeout if sample.timeout is not None else default_timeout
            ok, msg, stats = await fn(
                code=sample.code, timeout=timeout, memory_limit=sample.memory_limit
            )
        else:
            ok, msg = False, f"not supported language: {sample.lang}"
            stats = None

        logger.info(
            f"evaluate sample '{sample.uuid}' from '{sample.source}', "
            f"language: {sample.lang}, timeout: {timeout}, memory_limit: {sample.memory_limit}, "
            f"kwargs: {sample.kwargs}, status: {ok}, msg: {msg}, "
            f"cases: {int(ok)}/1, "
            f"avg_cpu: {stats.cpu_percent if stats else 0:.2f}%, "
            f"peak_cpu: {stats.peak_cpu_percent if stats else 0:.2f}%, "
            f"avg_memory: {stats.memory_mb if stats else 0:.2f}MB, "
            f"peak_memory: {stats.peak_memory_mb if stats else 0:.2f}MB"
        )
        return BasicResponse(
            status=ok,
            msg=msg,
            data=(
                ResourceMetrics(
                    avg_cpu_percent=stats.cpu_percent,
                    peak_cpu_percent=stats.peak_cpu_percent,
                    avg_memory_mb=stats.memory_mb,
                    peak_memory_mb=stats.peak_memory_mb,
                    # A direct run is one all-or-nothing case: the program either
                    # completed cleanly or it did not.
                    n_cases=1,
                    n_passed=int(ok),
                )
                if stats
                else None
            ),
        )
    elif sample.source == "livecodebench":
        # 'livecodebench' use tests to eval the code
        logger.debug(f"code to exec:\n{sample.code}")

        if sample.lang != "python":
            return BasicResponse(
                status=False, msg=f"not supported language: {sample.lang}", data=None
            )

        if sample.test is None:
            timeout = sample.timeout if sample.timeout is not None else 3.0
            ok, msg, stats = await exec_py_code(
                code=sample.code, timeout=timeout, memory_limit=sample.memory_limit
            )
            # No test cases: the program itself is the single all-or-nothing case.
            n_cases, n_passed = 1, int(ok)
        else:
            n_inputs = len(sample.test.inputs)
            if sample.timeout is not None:
                timeout = sample.timeout
            elif sample.timeout_per_case is not None:
                # Official LiveCodeBench's own backstop around a per-case budget:
                # `check_correctness` joins the worker at (timeout + 1) * n + 5.
                # A client needing its own HTTP deadline must predict this number, so
                # it holds a second copy -- sieval's tasks do, and send it explicitly
                # rather than take this branch. Keep the two in step.
                timeout = (sample.timeout_per_case + 1.0) * n_inputs + 5.0
            else:
                timeout = 6.0 + n_inputs * 2.0
            ok, msg, stats, n_passed = await exec_py_test(
                code=sample.code,
                inputs=sample.test.inputs,
                expect_outputs=sample.test.outputs,
                fn_name=sample.test.fn_name,
                timeout=timeout,
                memory_limit=sample.memory_limit,
                timeout_per_case=sample.timeout_per_case,
            )
            n_cases = len(sample.test.inputs)

        logger.info(
            f"evaluate sample '{sample.uuid}' from '{sample.source}', "
            f"language: {sample.lang}, timeout: {timeout}, memory_limit: {sample.memory_limit}, "
            f"kwargs: {sample.kwargs}, status: {ok}, msg: {msg}, "
            f"cases: {n_passed}/{n_cases}, "
            f"avg_cpu: {stats.cpu_percent:.2f}%, "
            f"peak_cpu: {stats.peak_cpu_percent:.2f}%, "
            f"avg_memory: {stats.memory_mb:.2f}MB, "
            f"peak_memory: {stats.peak_memory_mb:.2f}MB"
        )
        return BasicResponse(
            status=ok,
            msg=msg,
            data=ResourceMetrics(
                avg_cpu_percent=stats.cpu_percent,
                peak_cpu_percent=stats.peak_cpu_percent,
                avg_memory_mb=stats.memory_mb,
                peak_memory_mb=stats.peak_memory_mb,
                n_cases=n_cases,
                n_passed=n_passed,
            ),
        )
    else:
        logger.error(f"not supported data source: {sample.source}")
        return BasicResponse(
            status=False, msg=f"not supported data source: {sample.source}", data=None
        )
