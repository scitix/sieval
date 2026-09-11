import asyncio
import atexit
import os
import sys
import time
from functools import partial
from typing import Any, Generic, TypeVar

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from .exec_agnostics import execute_agnostics
from .exec_cpp import build_cases
from .exec_cpp import execute_tests as exec_cpp_test
from .exec_js import execute_code as exec_js
from .exec_lang import LANGUAGES, toolchain_entry, toolchain_present
from .exec_lang import execute_code as exec_lang
from .exec_py_code import execute_code as exec_py_code
from .exec_py_run import execute_run, truncate_stream
from .exec_py_test import execute_test as exec_py_test
from .exec_quotebench import execute_quotebench, scenarios_digest
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

#: Bumped whenever execution semantics change. sieval records it per tool call:
#: the resume gate compares YAMLs before any runner exists, so it cannot see this
#: service at all, and a version on the record is the only way a run that used a
#: different sandbox is identifiable after the fact.
SERVICE_VERSION = "code-runs/1"


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

    ``case_verdicts`` is the per-case detail behind ``n_passed``: one bool per
    case, in request order. Only ``liveoibench`` fills it, because only it needs
    *which* cases passed -- an olympiad subtask scores on its own test group, so
    a count cannot be attributed. ``None`` elsewhere means "not reported", which
    is why the count fields stay rather than being derived from it.

    ``case_names`` is what those verdicts belong to, positionally. It matters
    when the caller passed ``test_dir`` rather than inline cases: the server
    decided the order by listing that directory, and a caller that re-derived it
    would be keeping a second copy of the ordering rule -- the kind that agrees
    until one side changes.

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
    case_verdicts: list[bool] | None = None
    case_names: list[str] | None = None
    # quotebench only. They live on this flat model for the reason the docstring
    # above gives: a per-source subclass would have these stripped by FastAPI's
    # response filtering. `error_class` is the failure taxonomy
    # (shell-syntax / tool-error / runtime-error / silent-wrong / timeout /
    # environment-invalid), and `scenarios_digest` is what the caller asserts
    # against its own copy of the task definitions before trusting the verdict.
    error_class: str | None = None
    exit_code: int | None = None
    timed_out: bool | None = None
    scenarios_digest: str | None = None
    # The one field here that is not a cost. `agnostics` delegates the verdict to
    # a pinned container, and which digest scored a rollout is provenance the
    # caller has to be able to record -- there is nowhere else on this route to
    # put it, since the response model is flat by necessity (see above). `None`
    # everywhere else, and under a command override, where it is unknowable.
    verifier_image: str | None = None


class CodeRun(BaseModel):
    uuid: str
    lang: str = "python"
    code: str
    timeout: float = 10.0
    memory_limit: int = 1024


class CodeRunResult(BaseModel):
    """What one snippet printed, and what it cost.

    Distinct from `ResourceMetrics` on purpose: that payload answers "did the
    suite pass", this one answers "what did this print". Folding them would give
    every caller a field set where half is always null.
    """

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated: bool
    wall_s: float
    #: Reserved. Always null while execution is stateless; a future session
    #: route fills it without changing this shape.
    session_id: str | None = None
    service_version: str


# Direct-run executors, by `lang`. The three hand-rolled modules keep their own
# rows; every table-driven language contributes one automatically, so a new
# toolchain is a row in `exec_lang.LANGUAGES` and nothing here.
CODE_EXECUTOR_MAP: dict[str, tuple[Any, float]] = {
    "javascript": (exec_js, 3.0),
    "python": (exec_py_code, 3.0),
    "typescript": (exec_ts, 5.0),
} | {
    name: (partial(exec_lang, spec=spec), spec.default_timeout)
    for name, spec in LANGUAGES.items()
}


@app.get("/health")
async def check_health() -> BasicResponse[None]:
    return BasicResponse(status=True, msg="healthy")


@app.get("/languages")
async def list_languages() -> BasicResponse[list[str]]:
    """Which `lang` values this deployment can actually run, sorted.

    A capability probe, so a caller can refuse a run BEFORE spending inference
    on it. Without one, an unsupported language is only discoverable per sample,
    at which point every sample has already been generated and the report reads
    as a model that scored zero rather than as an evaluator that cannot run the
    language. Advertising the set is what makes that distinguishable.

    It answers for THIS DEPLOYMENT: a table row whose toolchain is missing from
    the image is withheld, not advertised. Otherwise the guard only moves the
    silent-zeros failure from "language not in the table" to "row present,
    compiler absent" -- the same run of zeros it exists to prevent.

    An existence check, not an invocation: a PATH lookup (~0.1 ms for the whole
    table), so nothing is compiled to answer a probe. It proves the entry point
    exists, not that the toolchain works.
    """
    return BasicResponse(
        status=True,
        msg="",
        data=sorted(
            lang
            for lang in CODE_EXECUTOR_MAP
            if lang not in LANGUAGES or toolchain_present(LANGUAGES[lang])
        ),
    )


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


@app.get("/quotebench/digest")
async def quotebench_digest() -> BasicResponse[str]:
    """The task definitions this build grades with, before anything is graded.

    A client prompts from its own vendored copy and this service grades with
    its own; a build skew between them mixes two fixture sets into one score
    that still looks plausible. `/evaluations` already returns the digest beside
    every verdict, but by then the run has spent its inference, and a client
    that refuses the verdict has nowhere to put the refusal except a failed
    sample. Reading it here is what lets the client refuse before it starts.

    Deliberately not folded into `/health`, which is source-agnostic and is what
    a load balancer polls.
    """
    return BasicResponse(status=True, msg="ok", data=scenarios_digest())


class LiveCodeBenchTest(BaseModel):
    inputs: list[str]
    outputs: list[str]
    fn_name: str | None = None
    # `liveoibench` only, and only for the failure message: its test cases have
    # official names ("boi1a") that a caller reading the log wants back. The
    # verdict-to-subtask mapping is positional, so nothing depends on it.
    names: list[str] | None = None


class Sample(BaseModel):
    uuid: str
    source: str
    code: str
    test: LiveCodeBenchTest | None = None
    lang: str = "python"
    timeout: float | None = None
    # `liveoibench` only: extra sources written beside the submission before
    # compiling -- the problem's `grader.cpp` and `{task}.h` -- and the filename
    # the submission itself is written as, which its `#include` has to match.
    files: dict[str, str] | None = None
    entry_filename: str | None = None
    # `liveoibench` only: a directory of `{name}.in` / `{name}.out` pairs this
    # evaluator can read, used instead of `test`. A problem averages ~140 MB of
    # test data, so inlining it would move the corpus over HTTP once per rollout;
    # a caller whose evaluator cannot see the volume sends `test` instead.
    test_dir: str | None = None
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

        spec = LANGUAGES.get(sample.lang)
        if sample.lang in CODE_EXECUTOR_MAP and (
            spec is None or toolchain_present(spec)
        ):
            fn, default_timeout = CODE_EXECUTOR_MAP[sample.lang]
            timeout = sample.timeout if sample.timeout is not None else default_timeout
            ok, msg, stats = await fn(
                code=sample.code, timeout=timeout, memory_limit=sample.memory_limit
            )
        else:
            # `timeout` is bound on this path too: the log line below reads it
            # unconditionally, so leaving it unset raised UnboundLocalError and
            # answered a 500 instead of this branch's own message -- and this is
            # exactly the path a language whose toolchain is not deployed takes.
            timeout = sample.timeout
            # A row whose toolchain is absent lands here rather than at a
            # `FileNotFoundError` from the spawn, so this and `GET /languages`
            # answer alike. Naming the missing command separates a deployment
            # gap from a per-sample failure that reads as the model's.
            ok, msg = False, f"not supported language: {sample.lang}"
            if spec is not None:
                msg = (
                    f"{msg} (row exists but `{toolchain_entry(spec)}` is not on "
                    f"PATH in this image)"
                )
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
    elif sample.source == "quotebench":
        # QuoteBench's unit of work is a task id plus one command, not a program
        # plus test cases: the task builds its own filesystem fixture and the
        # verdict is the exact final state. `code` carries the model's reply.
        kwargs = sample.kwargs or {}
        task_id = kwargs.get("task_id")
        contract = kwargs.get("contract")
        if not task_id or not contract:
            return BasicResponse(
                status=False,
                msg="quotebench requires kwargs.task_id and kwargs.contract",
                data=None,
            )
        # `local` runs bash in a fresh temp dir with a trimmed env and NO
        # network isolation. `docker` runs each attempt through upstream's
        # `harness.exec_docker`, whose argv carries `--network none` itself, and
        # needs a reachable docker daemon plus its `quotebench-runner` image.
        # Which one is safe depends on where this service is deployed, so the
        # choice is the operator's -- see the README's QuoteBench section.
        executor = os.getenv("QUOTEBENCH_EXECUTOR", "local")
        try:
            # Off the loop: `execute_quotebench` is fully blocking (it shells
            # out under `subprocess.run`, bounded at 15 s), and this endpoint is
            # `async def`, so FastAPI runs it ON the loop rather than in the
            # threadpool it gives a plain `def`. Called directly, one slow reply
            # stalls `/health` and every other source's grading on this worker,
            # and the concurrent replies never overlap.
            #
            # The other sources never had the problem: js/ts await
            # `asyncio.create_subprocess_exec`, and the two python ones already
            # await `asyncio.to_thread(q.get)` over a `multiprocessing.Process`.
            # So this is the package's own idiom, not a new one.
            ok, reason, error_class, exit_code, timed_out = await asyncio.to_thread(
                execute_quotebench,
                task_id=str(task_id),
                contract=str(contract),
                reply=sample.code,
                executor=executor,
            )
        except (KeyError, ValueError) as exc:
            # A protocol error, not a wrong command. `data=None` is how this
            # service already distinguishes "nothing ran" from a real verdict.
            #
            # `str()` on a KeyError is the *repr* of its argument, so a message
            # written as a sentence comes back wrapped in quotes. Read the
            # argument directly instead, so both raisers reach `msg` spelled the
            # way they were written.
            detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            logger.error(f"quotebench sample '{sample.uuid}' rejected: {detail}")
            return BasicResponse(status=False, msg=str(detail), data=None)

        logger.info(
            f"evaluate sample '{sample.uuid}' from 'quotebench', "
            f"task: {task_id}, contract: {contract}, executor: {executor}, "
            f"status: {ok}, class: {error_class}, exit: {exit_code}, "
            f"msg: {reason}"
        )
        return BasicResponse(
            status=ok,
            msg=reason,
            data=ResourceMetrics(
                # Not measured: the payload is a shell command in its own
                # process tree, so the service's in-process resource monitor
                # would report its own idle numbers rather than the command's.
                avg_cpu_percent=0.0,
                peak_cpu_percent=0.0,
                avg_memory_mb=0.0,
                peak_memory_mb=0.0,
                # One all-or-nothing case, as the direct-run sources report.
                n_cases=1,
                n_passed=int(ok),
                error_class=error_class,
                exit_code=exit_code,
                timed_out=timed_out,
                scenarios_digest=scenarios_digest(),
            ),
        )
    elif sample.source == "agnostics":
        # 'agnostics': the Agnostics protocol (nuprl/Ag-LiveCodeBench-X) -- one
        # JSON line to a per-language verifier container, one JSON line back.
        # `lang` names the container, not an executor in this process, which is
        # what makes Lua / R / Julia / OCaml / Fortran reachable at all.
        logger.debug(f"code to exec:\n{sample.code}")

        if sample.test is None:
            # Unlike 'livecodebench', there is no direct-run reading: the
            # protocol has no shape that carries a submission with no suite.
            msg = "infra:no-test: source 'agnostics' requires a test suite"
            logger.error(msg)
            return BasicResponse(status=False, msg=msg, data=None)

        # Upstream's `--timeout-seconds` is required and its README uses 15.
        timeout = sample.timeout if sample.timeout is not None else 15.0
        # `test.fn_name` is ignored on purpose: this set is entirely
        # stdin/stdout, and the protocol has no call-based mode to route it to.
        # `memory_limit` likewise -- the container owns its own limits, and they
        # belong in the command template rather than in a request field.
        ok, msg, stats, verifier_image = await execute_agnostics(
            code=sample.code,
            inputs=sample.test.inputs,
            expect_outputs=sample.test.outputs,
            lang=sample.lang,
            timeout=timeout,
        )

        logger.info(
            f"evaluate sample '{sample.uuid}' from '{sample.source}', "
            f"language: {sample.lang}, timeout: {timeout}, "
            f"cases: {len(sample.test.inputs)}, kwargs: {sample.kwargs}, "
            f"image: {verifier_image}, "
            f"status: {ok}, msg: {msg}, "
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
                # The verifier returns one verdict for the whole suite, so this
                # is the all-or-nothing pair the direct-run modes report. The
                # suite's real size is on the request, not in this count.
                n_cases=1,
                n_passed=int(ok),
                verifier_image=verifier_image,
            ),
        )
    elif sample.source == "liveoibench":
        # 'liveoibench': compile the C++ submission once, then run every official
        # test case. Unlike the livecodebench path this reports per-case verdicts
        # and never stops early -- an olympiad subtask is scored on its own test
        # group, so a submission that fails case 0 and passes the rest still
        # earns points.
        if sample.lang != "cpp":
            return BasicResponse(
                status=False, msg=f"not supported language: {sample.lang}", data=None
            )
        if sample.test is None and not sample.test_dir:
            return BasicResponse(
                status=False,
                msg="'liveoibench' requires either 'test' or 'test_dir'",
                data=None,
            )
        if (
            sample.test is not None
            and not sample.test_dir
            and len(sample.test.inputs) != len(sample.test.outputs)
        ):
            return BasicResponse(
                status=False,
                msg=(
                    f"test inputs/outputs length mismatch: "
                    f"{len(sample.test.inputs)} != {len(sample.test.outputs)}"
                ),
                data=None,
            )

        try:
            cases = build_cases(
                inputs=sample.test.inputs if sample.test else None,
                expect_outputs=sample.test.outputs if sample.test else None,
                names=sample.test.names if sample.test else None,
                test_dir=sample.test_dir,
            )
        except (FileNotFoundError, OSError) as e:
            # An unreadable `test_dir` is a deployment problem (the evaluator
            # cannot see the test volume), not a verdict on the submission.
            logger.error(f"cannot read tests for sample '{sample.uuid}': {e}")
            return BasicResponse(status=False, msg=f"cannot read tests: {e}", data=None)

        # The problem's own time limit, in seconds, per case; upstream buffers it
        # by 20% inside the runner. `timeout` bounds compilation instead -- the
        # one step with no per-case budget.
        timeout_per_case = (
            sample.timeout_per_case if sample.timeout_per_case is not None else 1.0
        )
        compile_timeout = sample.timeout if sample.timeout is not None else 60.0

        ok, msg, stats, n_passed, verdicts = await exec_cpp_test(
            code=sample.code,
            cases=cases,
            entry_filename=sample.entry_filename or "solution.cpp",
            files=sample.files,
            timeout_per_case=timeout_per_case,
            memory_limit=sample.memory_limit,
            compile_timeout=compile_timeout,
        )
        n_cases = len(cases)

        logger.info(
            f"evaluate sample '{sample.uuid}' from '{sample.source}', "
            f"language: {sample.lang}, timeout_per_case: {timeout_per_case}, "
            f"compile_timeout: {compile_timeout}, memory_limit: {sample.memory_limit}, "
            f"kwargs: {sample.kwargs}, status: {ok}, msg: {msg}, "
            f"cases: {n_passed}/{n_cases}, "
            f"avg_memory: {stats.memory_mb:.2f}MB, "
            f"peak_memory: {stats.peak_memory_mb:.2f}MB"
        )
        return BasicResponse(
            status=ok,
            msg=msg,
            data=ResourceMetrics(
                # Both CPU fields stay 0.0 here by construction: the per-case
                # poller measures each child's CPU *seconds* to enforce
                # RLIMIT_CPU, which is not this field's whole-service percentage.
                avg_cpu_percent=stats.cpu_percent,
                peak_cpu_percent=stats.peak_cpu_percent,
                avg_memory_mb=stats.memory_mb,
                peak_memory_mb=stats.peak_memory_mb,
                n_cases=n_cases,
                n_passed=n_passed,
                case_verdicts=verdicts,
                case_names=[case.name for case in cases],
            ),
        )
    else:
        logger.error(f"not supported data source: {sample.source}")
        return BasicResponse(
            status=False, msg=f"not supported data source: {sample.source}", data=None
        )


@app.post("/code-runs")
async def code_run(sample: CodeRun) -> BasicResponse[CodeRunResult]:
    """Run a Python snippet and return what it printed.

    Separate from ``/evaluations`` on purpose: every ``source`` there answers
    "did this submission pass", and a caller scores the result. This route
    grades nothing -- it returns stdout to a caller mid-generation, and the
    model consumes it, not a scorer. `n_cases` / `n_passed` would be meaningless
    on a response like that, so it gets its own response model instead of a
    widened ``ResourceMetrics``.
    """
    if sample.lang != "python":
        return BasicResponse(
            status=False, msg=f"/code-runs supports python only, got {sample.lang!r}"
        )
    started = time.perf_counter()
    exit_code, out, err, timed_out, _stats = await execute_run(
        code=sample.code, timeout=sample.timeout, memory_limit=sample.memory_limit
    )
    out, out_cut = truncate_stream(out)
    err, err_cut = truncate_stream(err)
    ok = exit_code == 0 and not timed_out
    logger.info(
        f"code-run '{sample.uuid}': ok={ok} exit={exit_code} "
        f"timed_out={timed_out} stdout={len(out)}c stderr={len(err)}c"
    )
    return BasicResponse(
        status=ok,
        msg="" if ok else (err[:200] or "run failed"),
        data=CodeRunResult(
            stdout=out,
            stderr=err,
            exit_code=exit_code,
            timed_out=timed_out,
            truncated=out_cut or err_cut,
            wall_s=time.perf_counter() - started,
            service_version=SERVICE_VERSION,
        ),
    )
