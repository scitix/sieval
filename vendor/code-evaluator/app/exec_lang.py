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
import signal
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .resource_monitor import ResourceStats, monitor_process_resources

#: How long to wait for a killed process group to be reaped before giving up on
#: it. A SIGKILLed group is gone in microseconds; this exists so that a process
#: wedged in uninterruptible sleep (a stuck NFS read) cannot re-introduce the
#: unbounded wait the kill is here to remove.
KILL_GRACE_SECONDS = 5.0


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


# COMMANDS follow upstream MultiPL-E's `evaluation/src/eval_<lang>.py`, so a
# program that passes here is one that passes there -- modulo the budgets below,
# which are this service's and not upstream's:
#   cpp   -- eval_cpp.py: `g++ ... -std=c++17`, build failure is its own status
#   bash  -- eval_sh.py:  `bash path`, exit code only
#   perl  -- eval_pl.py:  `perl path`, exit code AND "ERROR" in the output
#
# Upstream gives every one of those a flat 15s, because each is a bare
# `safe_subprocess.run` call taking its default -- the build included. Two rows
# here differ, in opposite directions, and both are deliberate:
#   * interpreted rows keep the 3s default the service already uses for js and
#     python, rather than tripling every other benchmark's wall to match one.
#     Measured headroom rather than a guess: a pure-bash prime sieve to n=20000
#     finishes in 0.6s, two hundred times under the wall. A submission that
#     would need between 3s and 15s fails here and passes upstream.
#   * a build gets its own 60s, since charging a cold-cache compile against the
#     program's budget fails a correct program for the toolchain's slowness. The
#     other direction: a compile upstream would abandon at 15s completes here.
# Both are enumerated in the tasks' `reference_impl.notes` as divergences.
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


async def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL a timed-out process AND everything it forked.

    ``proc.kill()`` alone signals the direct child. Anything that child forked
    inherited this process's stdout and stderr, so an orphan holding those pipes
    keeps them open, and the ``proc.wait()`` that has to follow the kill blocks
    until *it* exits -- not until the budget is spent. The timeout then bounds
    nothing: the verdict is still ``failed: timeout``, but the call returns
    whenever the orphan decides to, which for a program that backgrounded a
    sleeper or started a server is far past the wall or never. Measured before
    this existed: `bash` running a program that forks a 30s sleeper returned
    after 30.1s against a 3s budget, and the sleeper outlived the kill.

    Upstream MultiPL-E's ``safe_subprocess`` does exactly this, and says why in
    a comment: "Kills the process group. Without this line, test_fork_once
    fails." The group is the session made at spawn time, so it holds the
    submission and its descendants and nothing belonging to this server.

    ``ProcessLookupError`` is the good case -- the program exited between the
    timeout firing and the signal landing -- and is not worth reporting.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=KILL_GRACE_SECONDS)
    except asyncio.TimeoutError:
        # Unreapable within the grace period. Returning leaves the child to the
        # event loop's watcher, which is strictly better than the caller
        # inheriting an unbounded wait -- the thing this function exists to stop.
        pass


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
        # Its own session, so the timeout has a process GROUP to kill rather
        # than one pid -- see `_kill_group`. Set at spawn because a group cannot
        # be established after the fact, and `_capped`'s shell prologue `exec`s
        # the program, so the leader is the submission itself either way.
        start_new_session=True,
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
        await _kill_group(proc)
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
