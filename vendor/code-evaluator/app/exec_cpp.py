"""C++ compile-and-run path, for LiveOIBench-style olympiad judging.

Mirrors the reference judge of
https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12/src/judges/batch_judge.py
(``BatchJudge.run_test_case`` / ``BaseJudge.compile_cpp``), which is the only
place a LiveOIBench verdict is defined:

* compile with ``g++ -std=gnu++17 -Wall -O2 -pipe -static -g``, the problem's
  ``grader.cpp`` first on the command line when it has one;
* per test case, ``RLIMIT_CPU = ceil(time_limit * 1.2)`` and
  ``RLIMIT_AS = ceil(memory_limit_mb * 1.2) MB`` -- upstream's explicit 20%
  buffer -- with a poller that kills the process once its CPU time passes the
  same buffered limit;
* a non-zero exit is a failure; otherwise stdout is compared against the
  expected output by ``_compare_outputs`` below.

Two deliberate differences, both bounds rather than rules:

* **Compilation is bounded** (``compile_timeout``, default 60 s). Upstream calls
  ``subprocess.run`` with no timeout, so a submission whose template expansion
  never terminates wedges the judge. Nothing legitimate approaches the bound --
  a solution that cannot compile in a minute at ``-O2`` also cannot be measured.
* **Every test runs.** Upstream's ``stop_on_failure`` mode short-circuits a
  subtask at its first failure; here the full verdict vector always comes back,
  because subtask scoring needs it and a truncated vector silently scores every
  partially-correct submission as zero. Upstream's default is also to run them
  all.

Unlike the LiveCodeBench path, this one executes a *compiled binary* rather than
interpreting a string, so the submission never shares an address space with the
evaluator.
"""

import asyncio
import math
import os
import re
import resource
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

import psutil
from loguru import logger

from .resource_monitor import ResourceStats

COMPILE_COMMAND = ["g++", "-std=gnu++17", "-Wall", "-O2", "-pipe", "-static", "-g"]

# Upstream's per-test wall clock: `process.communicate(input=..., timeout=120)`.
# The real budget is RLIMIT_CPU; this only catches a process that is not burning
# CPU at all (blocked on a read, sleeping).
WALL_TIMEOUT = 120.0

# Upstream's `cpu_time_limit *= 1.2` / `memory_limit_mb *= 1.2`.
LIMIT_BUFFER = 1.2

_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_LEADING_NUMBER_RE = re.compile(r"^[+\-]?(\d+(\.\d*)?|\.\d+)([eE][+\-]?\d+)?")


@dataclass
class CaseResult:
    correct: bool
    detail: str
    cpu_time: float
    memory_mb: float


@dataclass
class TestCase:
    """One test case, either inline or on disk.

    A LiveOIBench problem averages ~140 MB of test data, so a run that shipped
    every case inline would move the whole 33.5 GB corpus over HTTP once per
    rollout. When the evaluator can see the materialized test tree, the caller
    sends the directory instead and each case is read only while it runs.
    """

    name: str
    input_data: str | None = None
    expected: str | None = None
    input_path: str | None = None
    expected_path: str | None = None

    def read_input(self) -> str:
        if self.input_path is not None:
            with open(self.input_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        return self.input_data or ""

    def read_expected(self) -> str:
        if self.expected_path is not None:
            with open(self.expected_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        return self.expected or ""


def read_test_dir(test_dir: str) -> list[TestCase]:
    """Collect ``{name}.in`` / ``{name}.out`` pairs from a materialized problem.

    Upstream (``Problem.get_test_inputs`` / ``get_test_outputs``) sorts the two
    globs independently and pairs them by position, which silently misaligns
    every later case if one ``.out`` is missing. Pairing by stem instead makes
    that a loud error; on well-formed data the two agree, since sorting by name
    *is* sorting by stem.
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test directory not found: {test_dir}")

    entries = os.listdir(test_dir)
    inputs = sorted(name[: -len(".in")] for name in entries if name.endswith(".in"))
    outputs = {name[: -len(".out")] for name in entries if name.endswith(".out")}

    missing = [stem for stem in inputs if stem not in outputs]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test input(s) in {test_dir} have no matching .out "
            f"(e.g. {missing[0]!r})"
        )
    return [
        TestCase(
            name=stem,
            input_path=os.path.join(test_dir, f"{stem}.in"),
            expected_path=os.path.join(test_dir, f"{stem}.out"),
        )
        for stem in inputs
    ]


def _is_single_number(text: str) -> bool:
    """Upstream ``BatchJudge._is_single_number``."""
    lines = text.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != 1:
        return False
    return bool(_NUMBER_RE.fullmatch(lines[0].strip()))


def _compare_numbers(gold: str, output: str) -> bool:
    """Upstream ``BatchJudge._compare_numbers``."""
    m_gold = _LEADING_NUMBER_RE.match(gold.strip())
    m_out = _LEADING_NUMBER_RE.match(output.strip())
    if m_gold and m_out:
        return math.isclose(
            float(m_gold.group(0)), float(m_out.group(0)), rel_tol=1e-6, abs_tol=1e-6
        )
    return False


def _compare_outputs(gold_output: str, stdout_decoded: str) -> bool:
    """Upstream ``BatchJudge._compare_outputs``.

    Both sides stripped; a single number on both sides compares with tolerance;
    then exact match; then line count plus a stripped per-line match.
    """
    gold_output = gold_output.strip()
    stdout_decoded = stdout_decoded.strip()

    if _is_single_number(gold_output) and _is_single_number(stdout_decoded):
        return _compare_numbers(gold_output, stdout_decoded)

    if gold_output == stdout_decoded:
        return True

    gold_lines = gold_output.splitlines()
    output_lines = stdout_decoded.splitlines()
    if len(gold_lines) != len(output_lines):
        return False
    for gold_line, output_line in zip(gold_lines, output_lines):
        if gold_line.strip() != output_line.strip():
            return False
    return True


def _set_limits(cpu_time_limit: int, memory_limit_bytes: int):
    """Upstream ``set_limits``, applied in the child before exec."""
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def _monitor_process(proc: psutil.Process, time_limit: float, monitor_data: dict, interval=0.01):
    """Upstream ``monitor_process``: sample CPU/RSS, kill once CPU passes the limit."""
    while proc.is_running():
        try:
            cpu_time = proc.cpu_times().user + proc.cpu_times().system
            mem_usage = proc.memory_info().rss / (1024 * 1024)
            monitor_data["max_cpu_time"] = max(monitor_data["max_cpu_time"], cpu_time)
            monitor_data["max_memory"] = max(monitor_data["max_memory"], mem_usage)
            if cpu_time > time_limit:
                proc.kill()
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(interval)


def _compile(
    workdir: str,
    entry_filename: str,
    grader_filename: str | None,
    compile_timeout: float,
) -> tuple[bool, str]:
    """Upstream ``BaseJudge.compile_cpp``, run inside the submission's workdir.

    ``cwd=workdir`` is what makes ``#include "task.h"`` resolve, the same way the
    upstream judge's copied-together working directory does.
    """
    sources = [grader_filename, entry_filename] if grader_filename else [entry_filename]
    command = [*COMPILE_COMMAND, "-o", "solution", *sources]
    try:
        result = subprocess.run(
            command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=compile_timeout,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return False, e.stderr or e.stdout or "compilation failed"
    except subprocess.TimeoutExpired:
        return False, f"compilation timed out after {compile_timeout}s"
    except FileNotFoundError:
        return False, "g++ not found: this evaluator image has no C++ toolchain"
    return True, result.stdout


def _run_one_test(
    workdir: str,
    case: TestCase,
    cpu_time_limit: float,
    memory_limit_mb: float,
) -> CaseResult:
    """One test case, with upstream's buffered limits and comparison."""
    input_data = case.read_input()
    # Upstream buffers both limits by 20% before applying them.
    cpu_time_limit = cpu_time_limit * LIMIT_BUFFER
    memory_limit_mb = memory_limit_mb * LIMIT_BUFFER

    executable = os.path.join(workdir, "solution")

    def preexec():
        _set_limits(
            int(math.ceil(cpu_time_limit)),
            int(math.ceil(memory_limit_mb * 1024 * 1024)),
        )

    process = subprocess.Popen(
        [executable],
        cwd=workdir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=preexec,
    )

    monitor_data = {"max_cpu_time": 0.0, "max_memory": 0.0}
    monitor_thread = None
    try:
        proc = psutil.Process(process.pid)
        monitor_thread = threading.Thread(
            target=_monitor_process, args=(proc, cpu_time_limit, monitor_data), daemon=True
        )
        monitor_thread.start()
    except psutil.NoSuchProcess:
        # Exited before we could attach; the limits in the child still applied.
        pass

    try:
        stdout_data, stderr_data = process.communicate(
            input=input_data.encode(), timeout=WALL_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_data, stderr_data = process.communicate()
    if monitor_thread is not None:
        monitor_thread.join()

    cpu_time = monitor_data["max_cpu_time"]
    memory_mb = monitor_data["max_memory"]

    if process.returncode == 0:
        stdout_decoded = (stdout_data.decode(errors="replace") if stdout_data else "").strip()
        correct = _compare_outputs(case.read_expected(), stdout_decoded)
        return CaseResult(correct, "" if correct else "wrong answer", cpu_time, memory_mb)

    # A non-zero exit is a failure regardless of cause; the cause is reported
    # only as free text, because upstream's judge does not classify it either --
    # `ResultType` exists in its tree but `BatchJudge` never assigns one.
    if cpu_time >= cpu_time_limit:
        detail = f"time limit exceeded ({cpu_time:.2f}s of {cpu_time_limit:.2f}s)"
    elif process.returncode == -9:
        detail = f"killed (peak rss {memory_mb:.0f}MB of {memory_limit_mb:.0f}MB)"
    else:
        stderr_text = (stderr_data.decode(errors="replace") if stderr_data else "").strip()
        detail = f"runtime error [exit {process.returncode}]: {stderr_text[:200]}"
    return CaseResult(False, detail, cpu_time, memory_mb)


def build_cases(
    inputs: list[str] | None,
    expect_outputs: list[str] | None,
    names: list[str] | None = None,
    test_dir: str | None = None,
) -> list[TestCase]:
    """Normalize the two ways a caller can supply tests into one case list."""
    if test_dir:
        return read_test_dir(test_dir)
    inputs = inputs or []
    expect_outputs = expect_outputs or []
    return [
        TestCase(
            name=names[i] if names and i < len(names) else f"#{i}",
            input_data=input_data,
            expected=expected,
        )
        for i, (input_data, expected) in enumerate(zip(inputs, expect_outputs))
    ]


async def execute_tests(
    code: str,
    cases: list[TestCase],
    entry_filename: str = "solution.cpp",
    files: dict[str, str] | None = None,
    timeout_per_case: float = 1.0,
    memory_limit: int = 1024,
    compile_timeout: float = 60.0,
    max_workers: int = 4,
) -> tuple[bool, str, ResourceStats, int, list[bool]]:
    """Compile *code* and run it against every case.

    Returns ``(all_passed, msg, stats, n_passed, verdicts)``. ``verdicts`` is one
    bool per test **in case order** -- the caller maps them onto subtasks by
    index, so the list is always as long as *cases*, including on a compilation
    error (all ``False``).

    *files* are extra sources written beside the submission: the problem's
    ``grader.cpp`` and ``{task}.h``. A ``grader.cpp`` among them is compiled with
    the submission, which is how upstream grades the 41 problems that ship one.

    *max_workers* only bounds host load: verdicts are decided by each child's own
    CPU-time and address-space limits, so they do not depend on how many run at
    once.
    """
    files = dict(files or {})
    stats = ResourceStats()

    with tempfile.TemporaryDirectory(prefix="liveoibench-") as workdir:
        for filename, contents in files.items():
            # Flat by construction: upstream's bundles are `grader.cpp` / `task.h`.
            with open(os.path.join(workdir, os.path.basename(filename)), "w") as f:
                f.write(contents)
        with open(os.path.join(workdir, entry_filename), "w") as f:
            f.write(code)

        grader_filename = "grader.cpp" if "grader.cpp" in files else None
        compiled, compile_msg = await asyncio.to_thread(
            _compile, workdir, entry_filename, grader_filename, compile_timeout
        )
        if not compiled:
            logger.debug(f"compilation failed:\n{compile_msg}")
            return (
                False,
                f"compilation error: {compile_msg.strip()[:500]}",
                stats,
                0,
                [False] * len(cases),
            )

        semaphore = asyncio.Semaphore(max(1, max_workers))

        async def run_case(case: TestCase) -> CaseResult:
            async with semaphore:
                return await asyncio.to_thread(
                    _run_one_test, workdir, case, timeout_per_case, memory_limit
                )

        results: list[CaseResult] = list(
            await asyncio.gather(*(run_case(case) for case in cases))
        )

    verdicts = [result.correct for result in results]
    n_passed = sum(verdicts)
    all_passed = n_passed == len(verdicts) and bool(verdicts)

    if results:
        stats.peak_memory_mb = max(result.memory_mb for result in results)
        stats.memory_mb = sum(result.memory_mb for result in results) / len(results)

    if all_passed:
        msg = f"passed {n_passed}/{len(verdicts)} test cases"
    else:
        first_failure = next(i for i, result in enumerate(results) if not result.correct)
        msg = (
            f"passed {n_passed}/{len(verdicts)} test cases; "
            f"first failure at test {cases[first_failure].name}: "
            f"{results[first_failure].detail}"
        )

    return all_passed, msg, stats, n_passed, verdicts
