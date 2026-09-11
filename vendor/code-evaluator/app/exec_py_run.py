"""Execute a Python snippet and return what it printed.

The sibling of ``exec_py_code``: same isolation primitives, opposite output
contract. ``exec_py_code`` swallows stdout because its callers only need to know
whether the program raised; a code-interpreter caller needs the text itself,
which is the whole point of the call.

AI-Generated Code - Claude Sonnet 5 (Anthropic)
"""

import asyncio
import contextlib
import io
import multiprocessing
import traceback

from loguru import logger

from .exec_py_code import create_tempdir, reliability_guard
from .resource_monitor import ResourceStats, monitor_process_resources
from .utils import kill_proc

#: Per-stream cap. Two jobs: an unbounded stdout is fed straight back into the
#: model's context (so it is a prompt-injection and a context-budget surface),
#: and it lands verbatim in a shard record. Callers are told when it bit.
MAX_STREAM_CHARS = 8192


def truncate_stream(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_STREAM_CHARS:
        return text, False
    return text[:MAX_STREAM_CHARS], True


def _subprocess_target(q, code: str, memory_limit: int | None) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    exit_code = 0
    with create_tempdir():
        import os
        import shutil

        rmtree, rmdir, chdir = shutil.rmtree, os.rmdir, os.chdir
        limit_bytes = int(memory_limit * 1024 * 1024) if memory_limit else None
        reliability_guard(maximum_memory_bytes=limit_bytes)
        try:
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                # WARNING: this executes untrusted model-generated code. It runs
                # under `reliability_guard` in a spawned process with an rlimit,
                # inside a temporary cwd, and the service is expected to be
                # containerised. See exec_py_code.py's fuller notice.
                exec(code, {"__name__": "__main__"})
        except BaseException:
            # The traceback goes to the caller as stderr rather than to a log:
            # it is the diagnostic the model needs in order to fix its own code,
            # and it is a RESULT of the run, not a failure of the service.
            stderr.write(traceback.format_exc())
            exit_code = 1
        finally:
            shutil.rmtree, os.rmdir, os.chdir = rmtree, rmdir, chdir
    q.put((exit_code, stdout.getvalue(), stderr.getvalue()))


async def execute_run(
    code: str, timeout: float = 10.0, memory_limit: int | None = 1024
) -> tuple[int | None, str, str, bool, ResourceStats]:
    """Run *code*; return ``(exit_code, stdout, stderr, timed_out, stats)``.

    ``exit_code`` is ``None`` when the process was killed rather than returning,
    which is the one case where neither stream can be trusted to be complete.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.SimpleQueue()
    p = ctx.Process(target=_subprocess_target, args=(q, code, memory_limit))
    p.start()

    if p.pid is not None:
        stats, stop_event = await monitor_process_resources(p.pid)
    else:
        logger.warning("process started but pid is None; skipping monitoring")
        stats, stop_event = ResourceStats(), asyncio.Event()
        stop_event.set()

    try:
        exit_code, out, err = await asyncio.wait_for(
            asyncio.to_thread(q.get), timeout=timeout
        )
        return exit_code, out, err, False, stats
    except asyncio.TimeoutError:
        return None, "", f"timed out after {timeout}s", True, stats
    except Exception as exc:
        return None, "", f"[{type(exc).__name__}] {exc}", False, stats
    finally:
        stop_event.set()
        await asyncio.sleep(0.1)
        kill_proc(p)
        with contextlib.suppress(Exception):
            q.close()
