"""Stateful shell execution for the InterCode-ALFA (NL2SH) benchmark.

Unlike every other backend here, this one is **stateful and serialized**. The
benchmark scores a model's Bash command by comparing what it *did to a
filesystem* against what the ground-truth command did, so the unit of work is
not a self-contained program: it is a pair of commands run against one specific
git-committed baseline tree, with the tree restored in between.

That baseline is the container. Upstream (``westenfelder/InterCode-ALFA``) ships
five images -- four Ubuntu, one Alpine -- each carrying a different prepared
directory (``/testbed``, ``/system``, ``/workspace`` + ``/backup``, or nothing at
all) committed to a git repo rooted at ``/`` whose ``.gitignore`` excludes every
standard FHS directory. So ``git status --short`` at the root reports exactly the
prepared tree's changes and nothing else, and ``git reset --hard; git clean -fd``
restores it. This module reproduces that protocol; the image supplies the tree.

**One service instance hosts one filesystem.** ``NL2SH_FS_ID`` says which, and a
request naming a different one is refused rather than served against the wrong
tree -- a misrouted sample would score zero with no error anywhere, which on a
300-sample benchmark is indistinguishable from a bad model.

**Run this route with one worker.** Every request mutates the shared tree, so an
in-process lock serializes them; multiple *processes* would not see that lock and
would interleave resets with each other's commands. ``fastapi run --workers N``
is safe for the stateless routes and wrong for this one.

**Upstream's quoting is reproduced, not repaired.** ``BashEnv.clean_cmd`` wraps a
command in double quotes and passes the string to docker-py, which normalizes it
through ``shlex.split``. On upstream's own 300 golds that rewrites 37 and
truncates one into a syntax error. Building the argv as
``[entrypoint, "-c", command]`` -- the obviously-correct thing -- would disagree
with every published number, so ``_argv`` does what docker-py does instead.

**Divergences from upstream**, both deliberate:

* Upstream runs the model's command and the gold's in two *peer containers*. Here
  they run in one, with a reset before each, which preserves the property that
  matters (the gold never executes on a tree the model just touched) while
  needing one deployment instead of two. Residue in gitignored paths outlives a
  reset -- as it does upstream, whose containers persist across all 300 samples.
* Upstream's 10 s wall (``utils.TIMEOUT_DURATION``) covers only the model's
  command; the gold is unbounded. Here both are bounded, because an unbounded
  gold hangs the service rather than one sample. A gold that hits it is a
  fidelity problem and is reported (``gold_timed_out``) so the caller can say
  the bound never bound.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import asyncio
import os
import shlex
import signal

from loguru import logger

#: `bash_env.IMAGE_TO_SETTINGS`, keyed by filesystem id. Image 5 is Alpine and
#: its shell is `/bin/sh`; running its 18 samples under bash would change what
#: they execute.
FS_ENTRYPOINT = {
    1: "/bin/bash",
    2: "/bin/bash",
    3: "/bin/bash",
    4: "/bin/bash",
    5: "/bin/sh",
}

#: `bash_env.GIT_RESET_SCRIPT` / `GIT_STATUS_SCRIPT`, verbatim.
GIT_RESET_SCRIPT = "git reset --hard; git clean -fd;"
GIT_STATUS_SCRIPT = "git status --short;"

#: `utils.TIMEOUT_DURATION`.
DEFAULT_TIMEOUT = 10.0
#: `exec_action`'s observation when the wall fires. Compared as an output, so the
#: exact string is part of the metric.
TIMEOUT_OBSERVATION = "Command timed out"
#: `get_reward`'s part-2 filter. `M` is absent upstream; see the sieval-side
#: `sieval/community/intercode_alfa` for why that is preserved.
HASHED_STATUS_CODES = ("A", "??", "C")

def fs_root() -> str:
    """Root of the git-committed baseline tree.

    ``/`` in every upstream image. Read per call rather than at import, so the
    protocol can be exercised against a temporary repo without a container and
    without reimporting the module.
    """
    return os.getenv("NL2SH_FS_ROOT", "/")

#: Which of the five filesystems this instance hosts. Unset means "do not serve
#: this route", which is the right default for the shared stateless image.
def hosted_fs_id() -> int | None:
    raw = os.getenv("NL2SH_FS_ID")
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value in FS_ENTRYPOINT else None


# Every request mutates the shared tree, so they are serialized here. See the
# module docstring on why this route needs a single worker.
_TREE_LOCK = asyncio.Lock()


def _argv(entrypoint: str, command: str) -> list[str]:
    """What upstream's container actually executes, quoting quirk included."""
    return shlex.split(f'{entrypoint} -c "{command.strip()}"')


async def _run(argv: list[str], timeout: float) -> tuple[str, bool, bool]:
    """Run *argv* at the tree root; return (combined output, exit_ok, timed_out).

    stdout and stderr are merged because docker's ``exec_run`` attaches both by
    default and upstream decodes the single stream it gets. Which stream a line
    came from is not recoverable there, so it is not recoverable here either.
    """
    if not argv:
        # `shlex.split` of an empty command yields the entrypoint and `-c` only,
        # never nothing -- but an empty argv would raise from exec, so refuse it
        # as a failed command rather than as a service error.
        return "", False, False
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=fs_root(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # Its own process group, so a timeout can take the whole tree of
            # children with it. Upstream abandons the command and lets it run on
            # inside the container; a lingering process here would keep writing
            # to the tree the next sample is about to be scored against.
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        # A command whose first word is not executable. Upstream surfaces the
        # same thing as output text from the shell, not as a service failure.
        return f"Exception: {exc}", False, False
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            logger.debug("timed-out process group already gone")
        await proc.wait()
        return TIMEOUT_OBSERVATION, False, True
    return stdout.decode("utf-8", errors="replace"), proc.returncode == 0, False


async def _reset(entrypoint: str) -> None:
    output, exit_ok, timed_out = await _run(
        _argv(entrypoint, GIT_RESET_SCRIPT), DEFAULT_TIMEOUT
    )
    if not exit_ok or timed_out:
        # The baseline could not be restored, so the next command would run on a
        # tree nobody can describe. Fail the request instead of scoring against
        # it -- a wrong baseline is a wrong gold for every later sample too.
        raise RuntimeError(
            f"failed to restore the baseline tree at {fs_root()!r}: {output!r}"
        )


async def _status(entrypoint: str) -> str:
    output, exit_ok, _ = await _run(
        _argv(entrypoint, GIT_STATUS_SCRIPT), DEFAULT_TIMEOUT
    )
    if not exit_ok:
        raise RuntimeError(f"`git status` failed at {fs_root()!r}: {output!r}")
    return output


def _parse_status(status: str) -> list[tuple[str, str]]:
    """`BashEnv.parse_status`, verbatim -- used here only to pick hash targets.

    The caller re-parses the raw text with its own copy and that copy is what
    scores; this one exists because the hash targets have to be known before the
    tree is reset. They cannot disagree silently: a path the caller derives and
    this did not is absent from the hash map, and the caller raises on it.
    """
    status_lst = status.split()
    changes = []
    for i in range(0, len(status_lst), 2):
        changes.append((status_lst[i + 1], status_lst[i]))
    return changes


async def _hashes(status: str) -> dict[str, str]:
    """`get_hash_cmd` over every added/untracked/copied path, as raw stdout.

    Raw stdout rather than a digest, because upstream compares the two command
    outputs as strings. It runs the hash command *without* a shell wrapper
    (``exec_run(hash_cmd)``), so neither does this -- and ``md5deep`` is absent
    from every image, so the directory branch always fails. Identically on both
    sides, which is why it still compares equal.
    """
    out: dict[str, str] = {}
    try:
        changes = _parse_status(status)
    except IndexError:
        # An odd token count: a path with a space in it. Upstream raises here
        # too (inside `get_reward`), so there is nothing to hash and the caller
        # will raise on its own parse.
        return out
    for path, code in changes:
        if code not in HASHED_STATUS_CODES:
            continue
        command = f"md5sum {path}" if "." in path else f"md5deep -r {path}"
        output, _exit_ok, _timed_out = await _run(
            shlex.split(command), DEFAULT_TIMEOUT
        )
        out[path] = output
    return out


async def execute_shell(
    fs_id: int, command: str, gold: str, timeout: float = DEFAULT_TIMEOUT
) -> tuple[bool, str, dict | None]:
    """Run gold then model against the baseline tree; return the raw facts.

    Returns ``(status, msg, data)`` in this service's usual shape. ``status`` is
    about the *service*, never about the model: a command that fails, times out
    or changes the wrong files is a successful evaluation reporting exactly that.
    ``status=False`` means the request could not be served -- a wrong ``fs_id``,
    or a baseline that would not restore -- and the caller must not read it as a
    wrong answer.

    The verdict is deliberately not computed here. It needs an embedding model
    when the two outputs differ, and this service holds no model credentials; the
    caller owns the arithmetic and the record of it.
    """
    hosted = hosted_fs_id()
    if hosted is None:
        return (
            False,
            "this instance hosts no NL2SH filesystem: set NL2SH_FS_ID to one of "
            f"{sorted(FS_ENTRYPOINT)} in the image that carries that tree",
            None,
        )
    if fs_id != hosted:
        return (
            False,
            f"fs_id mismatch: this instance hosts filesystem {hosted}, the "
            f"request asked for {fs_id}. Route the sample to the instance built "
            "from that image -- scoring it here would compare against the wrong "
            "prepared tree and report a plausible zero.",
            None,
        )
    entrypoint = FS_ENTRYPOINT[hosted]

    async with _TREE_LOCK:
        # Gold first, on a freshly restored tree, then the model's command on
        # another freshly restored one. Order does not matter to the comparison
        # -- each runs on the baseline -- but running gold first means a model
        # command that wedges the tree cannot corrupt the reference it is about
        # to be compared against within this sample.
        await _reset(entrypoint)
        gold_output, gold_exit_ok, gold_timed_out = await _run(
            _argv(entrypoint, gold), timeout
        )
        gold_status = await _status(entrypoint)
        gold_hashes = await _hashes(gold_status)

        await _reset(entrypoint)
        model_output, model_exit_ok, model_timed_out = await _run(
            _argv(entrypoint, command), timeout
        )
        model_status = await _status(entrypoint)
        model_hashes = await _hashes(model_status)

        await _reset(entrypoint)

    if gold_timed_out:
        # Not fatal -- the caller decides -- but it means this port's added
        # bound bound, which is a fidelity event worth a log line of its own.
        logger.warning(
            "gold command hit the {}s wall on fs {}: {!r}", timeout, hosted, gold
        )

    return (
        True,
        "",
        {
            "fs_id": hosted,
            "gold_output": gold_output,
            "model_output": model_output,
            "gold_status": gold_status,
            "model_status": model_status,
            "gold_hashes": gold_hashes,
            "model_hashes": model_hashes,
            "gold_exit_ok": gold_exit_ok,
            "model_exit_ok": model_exit_ok,
            "gold_timed_out": gold_timed_out,
            "model_timed_out": model_timed_out,
        },
    )
