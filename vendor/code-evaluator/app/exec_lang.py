"""Table-driven execution for compiled and interpreted languages.

One executor, one row per language. ``exec_py_code`` / ``exec_js`` / ``exec_ts``
each hand-roll the same write-file, spawn, wait, decode, classify sequence; a
fourth, fifth and sixth copy of it would be six places to fix the next bug in
the resource-monitor handshake. What actually differs between languages is data
-- a file extension, a build command, a run command, a budget -- so it is held
as data here and the sequence is written once.

Adding a language is a :data:`LANGUAGES` row plus its toolchain in the image.
No new module, and nothing in ``server.py`` to edit.

The classification is deliberately coarser than upstream MultiPL-E's, which
sorts a failure into ``SyntaxError`` / ``AssertionError`` / ``ReferenceError`` /
``Exception``. This service's response carries one boolean, and every one of
those buckets is the same boolean: the distinction is diagnostic, not scoring.
What survives is the part a caller cannot reconstruct -- whether the failure was
the BUILD or the RUN -- because it is spelled in ``msg``.

Messages reuse the vocabulary the hand-rolled modules already established
(``failed: timeout``, ``failed [exit N]: ...``), so a caller that greps for one
does not have to learn a second spelling for the same thing.
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .resource_monitor import ResourceStats, monitor_process_resources


@dataclass(frozen=True)
class LanguageSpec:
    """How to build and run one language's program.

    ``build`` and ``run`` are argv templates: ``{path}`` is the source file and
    ``{stem}`` the same path without its extension, which is where a compiler is
    told to put its output. A language with no build step leaves ``build`` None.

    Two budgets rather than one, because upstream gives each step its own
    ``safe_subprocess.run`` call with its own wall: charging a slow compile
    against the program's budget would fail a correct program on a cold cache,
    and charging the compile nothing would let a pathological one hang. Only
    ``timeout`` (the run) is caller-supplied; ``build_timeout`` is fixed here,
    since it is a property of the toolchain rather than of the submission.

    ``fail_on_output`` carries a language whose test harness reports failure by
    PRINTING rather than by exiting non-zero -- upstream's perl rule. Empty for
    every other row, and it is checked only on an otherwise-passing run, so it
    can turn a pass into a failure and never the reverse.
    """

    ext: str
    run: tuple[str, ...]
    build: tuple[str, ...] | None = None
    default_timeout: float = 3.0
    build_timeout: float = 60.0
    fail_on_output: tuple[str, ...] = field(default=())


# Commands follow upstream MultiPL-E's `evaluation/src/eval_<lang>.py`, so a
# program that passes here is one that passes there:
#   cpp   -- eval_cpp.py: `g++ ... -std=c++17`, build failure is its own status
#   bash  -- eval_sh.py:  `bash path`, exit code only
#   perl  -- eval_pl.py:  `perl path`, exit code AND "ERROR" in the output
# Interpreted rows keep the 3s default the service already uses for js/python;
# c++ gets upstream's own 15s, since `safe_subprocess.run` defaults to that and
# a compiled binary that needs longer is not one MultiPL-E waits for either.
LANGUAGES: dict[str, LanguageSpec] = {
    "cpp": LanguageSpec(
        ext=".cpp",
        build=("g++", "{path}", "-o", "{stem}", "-std=c++17"),
        run=("{stem}",),
        default_timeout=15.0,
    ),
    "bash": LanguageSpec(ext=".sh", run=("bash", "{path}")),
    "perl": LanguageSpec(
        ext=".pl",
        run=("perl", "{path}"),
        # eval_pl.py fails a run whose output mentions ERROR even at exit 0.
        fail_on_output=("ERROR",),
    ),
}


def _argv(template: tuple[str, ...], path: Path) -> list[str]:
    stem = str(path.with_suffix(""))
    return [part.format(path=str(path), stem=stem) for part in template]


def _capped(argv: list[str], memory_limit: int | None) -> list[str]:
    """Wrap *argv* so the program runs under an address-space cap, or return it.

    A shell prologue rather than ``preexec_fn``, which neither sibling module
    uses and which is unsafe here for the reason it is unsafe anywhere: it runs
    between fork and exec inside a THREADED server, where any lock another
    thread holds at fork is held forever in the child. js caps via
    ``NODE_OPTIONS`` and python via ``setrlimit`` inside its own forked child,
    so nothing in this service pays that risk today and this row should not be
    the first.

    ``exec`` replaces the shell, so the program's own exit code, signals and
    streams pass through untouched -- the shell is gone by the time it runs. A
    ``ulimit`` the kernel refuses is left to the shell's own reporting rather
    than aborting: a cap that cannot be set is the platform's answer, and
    failing the submission over it would score a model on the host's limits.
    """
    if not memory_limit:
        return argv
    kb = int(memory_limit) * 1024
    return ["/bin/sh", "-c", f'ulimit -v {kb} 2>/dev/null; exec "$0" "$@"', *argv]


async def _spawn(
    argv: list[str], *, cwd: str, timeout: float, memory_limit: int | None, monitor: bool
) -> tuple[int | None, str, ResourceStats, bool]:
    """Run *argv* to completion. Returns (exit code, stderr, stats, timed_out).

    An exit code of None means the process was killed on the timeout, so it
    never reported one -- the same "unknown, not zero" convention the response's
    ``n_passed`` uses.
    """
    stats = ResourceStats()
    proc = await asyncio.create_subprocess_exec(
        *_capped(argv, memory_limit),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=os.environ.copy(),
    )

    if monitor and proc.pid is not None:
        stats, stop_event = await monitor_process_resources(proc.pid)
    else:
        stop_event = asyncio.Event()
        stop_event.set()

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        # Both streams: upstream's perl rule reads stdout as well as stderr, and
        # a build error is on stderr. Joined so one scan covers both.
        output = f"{stdout.decode(errors='replace')}\n{stderr.decode(errors='replace')}"
        return proc.returncode, output.strip(), stats, False
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None, "", stats, True
    finally:
        stop_event.set()
        await asyncio.sleep(0.1)  # give the monitor time to finish


async def execute_code(
    code: str,
    timeout: float,
    memory_limit: int | None = None,
    *,
    spec: LanguageSpec,
) -> tuple[bool, str, ResourceStats]:
    """Build (if the language needs it) and run *code*, one all-or-nothing case.

    Matches the ``(ok, msg, stats)`` contract of the per-language modules, so
    ``server.py`` dispatches to it the same way. *spec* is keyword-only and has
    no default: a caller reaching this without naming a language is a bug, not a
    request to guess one.

    The stats returned are the RUN's, never the build's -- the build is
    toolchain cost, and reporting a compiler's memory as the submission's would
    make a c++ row incomparable with an interpreted one.
    """
    stats = ResourceStats()
    try:
        with tempfile.TemporaryDirectory() as workdir:
            # Named for the language, in its own directory: a compiler writes
            # its output next to the source, and the directory is what bounds
            # the mess a submission can make of the filesystem.
            path = Path(workdir) / f"program{spec.ext}"
            path.write_text(code, encoding="utf-8")

            if spec.build is not None:
                exit_code, output, _, timed_out = await _spawn(
                    _argv(spec.build, path),
                    cwd=workdir,
                    timeout=spec.build_timeout,
                    # The toolchain is trusted and can be memory-hungry; the cap
                    # exists for the submission, which is the next step.
                    memory_limit=None,
                    monitor=False,
                )
                if timed_out:
                    return False, "failed: build timeout", stats
                if exit_code != 0:
                    # Upstream calls this SyntaxError. Spelled as a build
                    # failure because that is what it is for an interpreted
                    # language too, where a syntax error surfaces at run time.
                    return False, f"failed [build exit {exit_code}]: {output}", stats

            exit_code, output, stats, timed_out = await _spawn(
                _argv(spec.run, path),
                cwd=workdir,
                timeout=timeout,
                memory_limit=memory_limit,
                monitor=True,
            )
            if timed_out:
                return False, "failed: timeout", stats
            if exit_code != 0:
                return False, f"failed [exit {exit_code}]: {output}", stats
            for marker in spec.fail_on_output:
                if marker in output:
                    return False, f"failed [output contains {marker!r}]: {output}", stats
            return True, output, stats
    except Exception as e:
        return False, f"failed: [{type(e).__name__}] {e}", stats
