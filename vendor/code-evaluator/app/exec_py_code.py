import asyncio
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import tempfile

from loguru import logger

from .resource_monitor import ResourceStats, monitor_process_resources
from .utils import kill_proc


async def execute_code(
    code: str, timeout: float = 3.0, memory_limit: int | None = None
) -> tuple[bool, str, ResourceStats]:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.SimpleQueue()
    p = ctx.Process(target=_subprocess_target, args=(q, code, memory_limit))
    p.start()

    # Start resource monitoring (only if pid is available)
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
            logger.debug("failed to close multiprocessing queue")
    return False, f"failed: {reason}", stats


def _subprocess_target(q: multiprocessing.Queue, code: str, memory_limit: int | None):
    try:
        ok, msg = _unsafe_execute(code, memory_limit)
        q.put((ok, msg))
    except Exception as e:
        q.put((False, f"failed: [{type(e).__name__}] {e}"))


# adapted from https://github.com/openai/human-eval/blob/6d43fb980f9fee3c892a914eda09951f772ad10d/human_eval/execution.py
def _unsafe_execute(code: str, memory_limit: int | None) -> tuple[bool, str]:
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        # Convert MB to bytes
        limit_bytes = int(memory_limit * 1024 * 1024) if memory_limit else None
        reliability_guard(maximum_memory_bytes=limit_bytes)

        try:
            exec_globals = {}
            with swallow_io():
                # WARNING
                # This program exists to execute untrusted model-generated code. Although
                # it is highly unlikely that model-generated code will do something overtly
                # malicious in response to this test suite, model-generated code may act
                # destructively due to a lack of model capability or alignment.
                # Users are strongly encouraged to sandbox this evaluation suite so that it
                # does not perform destructive actions on their host or network. For more
                # information on how OpenAI sandboxes its code, see the accompanying paper.
                # Once you have read this disclaimer and taken appropriate precautions,
                # uncomment the following line and proceed at your own risk:
                exec(code, exec_globals)
            return True, ""
        except BaseException as e:
            return False, f"failed: [{type(e).__name__}] {e}"
        finally:
            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: int | None = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        def _safe_setrlimit(res, limit):
            try:
                _, hard = resource.getrlimit(res)
                # If there is a hard limit, we cannot exceed it
                if hard != resource.RLIM_INFINITY:
                    limit = min(limit, hard)
                resource.setrlimit(res, (limit, limit))
            except (ValueError, OSError):
                pass

        _safe_setrlimit(resource.RLIMIT_AS, maximum_memory_bytes)
        _safe_setrlimit(resource.RLIMIT_DATA, maximum_memory_bytes)

        if not platform.uname().system == "Darwin":
            _safe_setrlimit(resource.RLIMIT_STACK, maximum_memory_bytes)

    def _disabled(name: str):
        def _f(*_a, **_k):
            logger.debug(f"disabled function: {name}")

        return _f

    faulthandler.disable()

    import builtins

    builtins.exit = _disabled("exit")
    builtins.quit = _disabled("quit")

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    _os_block = [
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
    ]
    for _n in _os_block:
        if hasattr(os, _n):
            setattr(os, _n, _disabled(f"os.{_n}"))

    import shutil

    for _n in ["rmtree", "move", "chown"]:
        if hasattr(shutil, _n):
            setattr(shutil, _n, _disabled(f"shutil.{_n}"))

    import subprocess

    if hasattr(subprocess, "Popen"):
        subprocess.Popen = _disabled("subprocess.Popen")

    __builtins__["help"] = _disabled("help")

    import sys

    _sys_block = [
        "ipdb",
        "joblib",
        "resource",
        "psutil",
        "tkinter",
    ]
    for _m in _sys_block:
        sys.modules[_m] = None  # type: ignore[assignment]
