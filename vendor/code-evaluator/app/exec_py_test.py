import ast
import asyncio
import json
import multiprocessing
import os
import sys
from decimal import Decimal
from io import StringIO
from types import ModuleType
from typing import Callable
from unittest.mock import mock_open, patch

from loguru import logger

from .exec_py_code import reliability_guard
from .resource_monitor import ResourceStats, monitor_process_resources
from .utils import kill_proc

# Float-comparison tolerance is OFF by default to match official LiveCodeBench, which
# compares Decimals with exact `==` (deliberately, to avoid false positives such as
# isclose(50000000000000000, 50000000000000001) == True). Opt in by setting the env var
# CODE_EVAL_FLOAT_TOL, e.g. "1e-6". When enabled, tolerance is applied ONLY to genuinely
# fractional expected values; integer expected values always require exact equality, so a
# large-integer answer can never be loosened.
_FLOAT_TOL = (
    Decimal(os.environ["CODE_EVAL_FLOAT_TOL"])
    if os.environ.get("CODE_EVAL_FLOAT_TOL")
    else None
)


def _decimals_match(out_decimals: list, exp_decimals: list) -> bool:
    """Compare two decimal token lists. Exact by default; abs-or-rel tolerance for
    fractional expected values only when CODE_EVAL_FLOAT_TOL is set."""
    if len(out_decimals) != len(exp_decimals):
        return False
    for a, b in zip(out_decimals, exp_decimals):
        if a == b:
            continue
        if _FLOAT_TOL is not None and b != b.to_integral_value():
            diff = abs(a - b)
            if diff <= _FLOAT_TOL or diff <= _FLOAT_TOL * abs(b):
                continue
        return False
    return True


def _value_close(a, b, tol) -> bool:
    """Recursive tolerant compare for function-call outputs (leetcode-style). Tolerance
    applies only to fractional expected floats; integer expected values stay exact.

    Note: "integer expected" here is keyed off Python *type* (``isinstance(b, int)``),
    so a float ``1.0`` is treated as fractional and gets tolerance -- unlike the stdio
    path (``_decimals_match``), which keys off *value* (``b.to_integral_value()``) and
    treats ``1.0`` as exact. Behavior inherited from vendored patch ``cfc47d8``
    (fork branch ``fix/checker-messages-float-tol``; see VENDORED.md)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(b, int):
            return a == b
        return abs(a - b) <= tol or abs(a - b) <= tol * abs(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_value_close(x, y, tol) for x, y in zip(a, b))
    return a == b


async def execute_test(
    code: str,
    inputs: list[str],
    expect_outputs: list[str],
    fn_name: str | None = None,
    timeout: float = 3.0,
    memory_limit: int | None = None,
) -> tuple[bool, str, ResourceStats]:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.SimpleQueue()
    p = ctx.Process(
        target=_subprocess_target,
        args=(q, code, inputs, expect_outputs, fn_name, memory_limit),
    )
    p.start()

    # Start resource monitoring
    if p.pid is not None:
        stats, stop_event = await monitor_process_resources(p.pid)
    else:
        logger.warning("Process started but pid is None, skipping resource monitoring")
        stats = ResourceStats()
        stop_event = asyncio.Event()
        stop_event.set()  # Already "stopped" since we never started

    try:
        ok, msg = await asyncio.wait_for(asyncio.to_thread(q.get), timeout=timeout)
        return ok, msg, stats
    except asyncio.TimeoutError:
        if p.is_alive():
            reason = f"subprocess timeout: {timeout}s"
        else:
            reason = "no result from subprocess"
    except Exception as e:
        reason = f"[{type(e).__name__}] {e}"
    finally:
        # Stop monitoring
        stop_event.set()
        await asyncio.sleep(0.1)  # Give monitor time to finish

        kill_proc(p)
        try:
            q.close()
        except Exception:
            pass
    return False, f"failed: {reason}", stats


def _subprocess_target(
    q: multiprocessing.Queue,
    code: str,
    inputs: list[str],
    expect_outputs: list[str],
    fn_name: str | None,
    memory_limit: int | None,
):
    try:
        ok, msg = _unsafe_execute(code, inputs, expect_outputs, fn_name, memory_limit)
        q.put((ok, msg))
    except Exception as e:
        q.put((False, f"failed: [{type(e).__name__}] {e}"))


def _unsafe_execute(
    code: str,
    inputs: list[str],
    expect_outputs: list[str],
    fn_name: str | None,
    memory_limit: int | None,
) -> tuple[bool, str]:
    if len(inputs) != len(expect_outputs):
        return False, "failed: number of inputs and outputs mismatch"

    # Disable functionalities that can make destructive changes to the test.
    # memory_limit is in MB, convert to bytes
    limit_bytes = int(memory_limit * 1024 * 1024) if memory_limit else None
    reliability_guard(maximum_memory_bytes=limit_bytes)

    if fn_name is not None:
        code_to_compile = import_string + "\n\n" + code
        compiled_sol = compile_code(code_to_compile)
        if compiled_sol is None:
            return False, "failed: compile error"
        fn = get_function(compiled_sol, fn_name)
        if fn is None:
            return False, "failed: no function defined"
    else:
        code_to_compile = clean_if_name(code)
        code_to_compile = make_function(code_to_compile)
        compiled_sol = compile_code(code_to_compile)
        if compiled_sol is None:
            return False, "failed: compile error"
        fn = get_function(compiled_sol, "wrapped_function")
        if fn is None:
            return False, "failed: no function defined"

    for single_input, single_output in zip(inputs, expect_outputs):
        if fn_name is not None:
            ok, msg = _unsafe_execute_fn_call(fn, single_input, single_output)
        else:
            ok, msg = _unsafe_execute_stdio(fn, single_input, single_output)
        if not ok:
            return False, f"failed: {msg}"
    return True, ""


def _unsafe_execute_fn_call(
    fn: Callable, single_input: str, expect_output: str
) -> tuple[bool, str]:
    try:
        args = [json.loads(line) for line in single_input.split("\n")]
        exp_outputs = json.loads(expect_output)

        outputs = fn(*args)
        # don't penalize model if it produces tuples instead of lists
        # ground truth sequences are not tuples
        if isinstance(outputs, tuple):
            outputs = list(outputs)

        if outputs != exp_outputs and not (
            _FLOAT_TOL is not None and _value_close(outputs, exp_outputs, float(_FLOAT_TOL))
        ):
            return False, f"output {outputs} != expect {exp_outputs}"
        return True, ""
    except Exception as e:
        return False, f"[{type(e).__name__}] {e}"


def _unsafe_execute_stdio(
    method: Callable, single_input: str, expect_output: str
) -> tuple[bool, str]:
    with Capturing() as captured_output:
        try:
            call_method(method, single_input)
        except Exception as e:
            return False, f"[{type(e).__name__}] {e}"

    output = captured_output[0]
    stripped_output_lines = get_stripped_lines(output)
    stripped_expect_outputs_lines = get_stripped_lines(expect_output)

    if len(stripped_output_lines) != len(stripped_expect_outputs_lines):
        return False, "output line count mismatch"

    for out_line, exp_line in zip(stripped_output_lines, stripped_expect_outputs_lines):
        if out_line == exp_line:
            continue

        ok_out, out_decimals = convert_line_to_decimals(out_line)
        ok_exp, exp_decimals = convert_line_to_decimals(exp_line)
        # A non-numeric line that didn't exactly match is a plain output mismatch.
        # (Exact string comparison already ran above — this is NOT a checker limitation
        # on strings; the produced answer simply differs from the expected one.)
        if not ok_out or not ok_exp:
            return False, f"output mismatch: got {out_line!r} expected {exp_line!r}"
        if not _decimals_match(out_decimals, exp_decimals):
            return False, f"numeric mismatch: got {out_line!r} expected {exp_line!r}"

    return True, ""


# adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py
import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# Custom mock for sys.stdin that supports buffer attribute
class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        # Delegate other attributes to StringIO
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")  # Convert to bytes

    def read(self, *args):
        # Return as byte strings that can be split
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
                )

    except Exception:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)
            + "\n"
            + ast.unparse(function_ast)
        )
        return main_code
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # Create custom stdin mock with buffer support
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)  # Use our custom mock instead of StringIO
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return


def compile_code(code: str):
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        pass

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    # you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]
