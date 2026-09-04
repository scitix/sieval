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

**So is upstream's ``cd`` rewriting**, which reaches only the model's command:
see ``model_action``. Both rewrites are text transforms applied before anything
executes, and both change the verdict on replies that carry them.

**``/tmp`` does not survive the first reset.** Upstream's ``docker.gitignore``
excludes 16 FHS directories but not ``tmp``, and ``/tmp`` is empty when the
baseline is committed -- so git tracks nothing there, the directory is untracked,
and the first ``git clean -fd`` removes it outright (measured, not assumed).
Upstream's containers do the same, so it is faithful and must not be "fixed": a
model command using ``mktemp`` fails identically on both sides. It does constrain
*this* service, which upstream's containers never had to host -- nothing on this
route may rely on ``tempfile``, and it does not. The stateless routes in
``exec_py_code`` / ``exec_js`` / ``exec_ts`` do, so they are unreliable on these
five images after the first sample; they are not served here, and an image that
needs both would have to add ``tmp`` to the ignore file, which changes what
``git status`` reports and therefore what the benchmark scores.

**Divergences from upstream**, all deliberate:

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
* A reply starting with ``cd`` but carrying no ``"cd "`` scores 0 for the whole
  sample upstream, raised out of ``exec_action`` before anything runs. A
  facts-only response cannot return a verdict, so this one reports that nothing
  ran; ``execute_shell`` has why that reaches the same answer.
* The second side's hashing is restricted to the first side's changed paths. The
  scored set (``diff_same``) is unchanged -- see ``_hashes`` -- but upstream
  hashes the intersection on both sides and this cannot, because the
  intersection is unknowable while the first side's tree is still standing.

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

#: `IntercodeEnv.workdir` at the moment the model's command runs. `reset_container`
#: sets it to "/" and only a `cd` action advances it, so on a single-turn benchmark
#: -- one command, then `submit` -- it is always the tree root. Upstream passes it
#: as `exec_run(..., workdir=...)`, and every one of the five images declares
#: `WORKDIR /`, so both sides run at the root.
_WORKDIR = "/"

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


def simplify_path(current: str, changed: str) -> str:
    """`BashEnv.simplify_path`, verbatim: resolve *changed* against *current*."""
    if not changed:
        return current
    if changed[0] == "/":
        current = ""

    path = []

    for segment in (current + "/" + changed).split("/"):
        if segment == "..":
            if path:
                path.pop()
        elif segment and segment != ".":
            path.append(segment)

    return "/" + "/".join(path)


def model_action(command: str) -> str:
    """`exec_action`'s ``cd`` pre-processing, applied to the MODEL's command only.

    Upstream rewrites any action starting with ``cd``: it takes everything after
    the first ``"cd "``, resolves it against the working directory with
    ``simplify_path``, and re-emits ``cd <absolute path>``. That is multi-turn
    machinery (``self.workdir`` tracks the cwd across an interactive session)
    reaching a single-turn benchmark, where the working directory is always the
    tree root -- so it degenerates to "absolutize the argument against ``/``".

    Reproduced rather than skipped, for the same reason the quoting rewrite is:
    it changes what bash receives. ``cd testbed && ls`` and ``cd ..`` come out
    equivalent at the root, but ``cd ~`` becomes ``cd /~`` and fails, where the
    un-rewritten reply would have succeeded.

    **The gold command never comes through here.** Upstream runs it as
    ``container_eval.exec_run(clean_cmd(self.gold))``, which does not touch
    ``exec_action`` -- and none of the 300 golds starts with ``cd`` anyway.

    Raises ``ValueError`` when the command starts with ``cd`` but carries no
    ``"cd "`` (a bare ``cd``, or ``cdparanoia``): upstream's ``action.index("cd ")``
    sits *outside* ``exec_action``'s try, so it escapes to ``submit_command``,
    which reports a score of 0 for the whole sample. The caller reproduces that
    as a command that did not execute -- see ``execute_shell``.
    """
    if not command.startswith("cd"):
        return command
    cd_arg = command[command.index("cd ") + 3 :].strip()
    return f"cd {simplify_path(_WORKDIR, cd_arg)}"


async def _run(
    command: str, timeout: float, entrypoint: str | None = None
) -> tuple[str, bool, bool]:
    """Run *command* at the tree root; return (combined output, exit_ok, timed_out).

    ``entrypoint`` wraps the command the way ``clean_cmd`` does; ``None`` runs it
    bare, which is how upstream issues its hash commands (``exec_run(hash_cmd)``,
    no shell). Either way the string is split the way docker-py splits it.

    **The split happens inside this function's error handling, deliberately.**
    ``shlex.split`` raises ``ValueError`` on an unbalanced quote, which an
    ordinary model reply can carry -- a command truncated mid-string is the
    common route. Upstream reaches the same raise (docker-py normalizes a string
    command through ``shlex.split`` inside ``exec_run``) and ``exec_action``
    catches it, so the sample is still *graded*, with the exception as its
    observation. Building the argv at the call site instead would let that
    ValueError escape the request and turn a wrong answer into a service
    failure, which is the one thing this route must never do.

    stdout and stderr are merged because docker's ``exec_run`` attaches both by
    default and upstream decodes the single stream it gets. Which stream a line
    came from is not recoverable there, so it is not recoverable here either.
    """
    try:
        argv = (
            _argv(entrypoint, command)
            if entrypoint is not None
            else shlex.split(command)
        )
    except ValueError as exc:
        # `exec_action`'s `except Exception as e: self.observation = f"Exception:
        # {e}"`, reproduced -- same text, same `action_executed = False`.
        return f"Exception: {exc}", False, False
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
        GIT_RESET_SCRIPT, DEFAULT_TIMEOUT, entrypoint
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
        GIT_STATUS_SCRIPT, DEFAULT_TIMEOUT, entrypoint
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


async def _hashes(status: str, restrict_to: set[str] | None = None) -> dict[str, str]:
    """`get_hash_cmd` over every added/untracked/copied path, as raw stdout.

    Raw stdout rather than a digest, because upstream compares the two command
    outputs as strings. It runs the hash command *without* a shell wrapper
    (``exec_run(hash_cmd)``), so neither does this -- and ``md5deep`` is absent
    from every image, so the directory branch always fails. Identically on both
    sides, which is why it still compares equal.

    ``restrict_to`` bounds the work. Upstream hashes only ``diff_same`` -- the
    paths *both* sides changed -- but that set is unknowable while the first
    side's tree is still standing, so the first side hashes all of its own
    candidates and the second is restricted to those. ``diff_same`` is a subset
    of the first side's hashed paths (a shared change is, by definition, one the
    first side made under a hashed code), so nothing the caller looks up goes
    missing and its ``KeyError`` contract is untouched. Without this, a reply
    that touches many paths pays a hash per path on both sides for an
    intersection that is usually empty.
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
        if restrict_to is not None and path not in restrict_to:
            continue
        command = f"md5sum {path}" if "." in path else f"md5deep -r {path}"
        output, _exit_ok, _timed_out = await _run(command, DEFAULT_TIMEOUT)
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
        # The gold does NOT go through `model_action`: upstream runs it as
        # `container_eval.exec_run(clean_cmd(self.gold))`, which never reaches
        # `exec_action` and so never has its `cd` rewritten.
        gold_output, gold_exit_ok, gold_timed_out = await _run(
            gold, timeout, entrypoint
        )
        gold_status = await _status(entrypoint)
        gold_hashes = await _hashes(gold_status)

        await _reset(entrypoint)
        try:
            action = model_action(command)
        except ValueError as exc:
            # A reply starting with `cd` but carrying no `"cd "`. Upstream lets
            # this escape to `submit_command`, which scores the whole sample 0
            # without running anything. A facts-only response cannot say "0", so
            # it says what actually happened -- nothing ran -- which reaches the
            # same verdict: the tree is untouched, so part 1 diverges from any
            # gold that changed something, and against a read-only gold part 3
            # compares this text to the gold's output. The two readings part only
            # if that comparison were to embed above threshold.
            model_output, model_exit_ok, model_timed_out = (
                f"Exception: {exc}",
                False,
                False,
            )
        else:
            model_output, model_exit_ok, model_timed_out = await _run(
                action, timeout, entrypoint
            )
        model_status = await _status(entrypoint)
        # Restricted to what the gold changed: `diff_same` cannot contain a path
        # the gold did not touch, and the gold's tree is already gone.
        model_hashes = await _hashes(model_status, restrict_to=set(gold_hashes))

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
