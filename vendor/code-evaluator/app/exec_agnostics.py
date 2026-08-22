"""Run a submission through an Agnostics verifier container.

The Agnostics protocol (nuprl/Ag-LiveCodeBench-X) is one JSON line in, one JSON
line out:

    in  { "code": str, "timeout_s": int,
          "test_cases": [ { "input": str, "output": str }, ... ] }
    out { "result": "success" }
        | { "result": "fail:wrong-output", "expected": str, "got": str, ... }
        | { "result": "fail:error", "exit_code": int, ... }
        | { "result": "fail:timeout", ... }
        | { "result": "fail:other", ... }

Only ``"success"`` counts as a pass; the protocol explicitly allows other result
codes and extra fields, so anything unrecognized is a failure rather than an
error.

The container command is **deployment configuration**, never a request field: a
caller able to name the image could run an arbitrary container on this host. The
default is upstream's own invocation, with ``{lang}`` filled from the request's
``lang``.
"""

import asyncio
import json
import os
import re
import shlex

from .resource_monitor import ResourceStats, monitor_process_resources

_COMMAND_ENV_VAR = "CODE_EVAL_AGNOSTICS_COMMAND"

# Upstream's podman invocation, verbatim apart from the `{lang}` placeholder.
# Override the whole command via the env var above for a different runtime
# (upstream also supports `apptainer run --contain --writable-tmpfs <file>.sif`),
# a mirrored registry, or a local test double.
_DEFAULT_COMMAND = (
    "podman run --rm -i --tmpfs /ramdisk:size=512m,exec "
    "ghcr.io/nuprl/agnostics:{lang}"
)

# `lang` reaches an argv slot, so it is constrained to what a container tag can
# be. Not a sandbox -- the command template is trusted and this is not -- but it
# keeps a tag from smuggling in a separator or a path.
_LANG_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9_.+-]{0,31}\Z")

# Upstream gives writing the payload its own budget, separate from the wall the
# process runs under, because a decoded LiveCodeBench suite is tens of MB and a
# slow write is not a slow program. Its `stdin_write_timeout=300` is kept.
_STDIN_WRITE_TIMEOUT = 300.0


def verifier_command(lang: str) -> list[str]:
    template = os.environ.get(_COMMAND_ENV_VAR) or _DEFAULT_COMMAND
    return [part.replace("{lang}", lang) for part in shlex.split(template)]


async def execute_agnostics(
    code: str,
    inputs: list[str],
    expect_outputs: list[str],
    lang: str,
    timeout: float,
) -> tuple[bool, str, ResourceStats]:
    """Return ``(passed, msg, stats)``.

    ``msg`` is the container's ``result`` value verbatim when the container ran
    and spoke the protocol, and an ``infra:<reason>`` code when it did not, so a
    client can separate "the program is wrong" from "the harness could not ask".
    Upstream collapses every one of the latter to ``"fail"`` and recovers only
    the stdin-write case, from a stderr suffix; the split is named here instead,
    where it is known. It does not change what counts as a pass: only
    ``"success"`` does, either way.

    ``timeout`` serves as both the container's own ``timeout_s`` and the wall
    this process is held to -- one number in both roles, as upstream sends it.
    A suite whose per-case budgets can sum past the wall is therefore killed
    from outside before it can report its own ``fail:timeout``; that tension is
    upstream's, and widening the wall here would move scores.
    """
    stats = ResourceStats()
    if not _LANG_PATTERN.match(lang):
        return False, f"infra:bad-lang: {lang!r}", stats

    payload = json.dumps(
        {
            "code": code,
            # The protocol types it as an int, and upstream passes its
            # `--timeout-seconds` straight through.
            "timeout_s": int(timeout),
            "test_cases": [
                {"input": stdin, "output": stdout}
                for stdin, stdout in zip(inputs, expect_outputs, strict=True)
            ],
        }
    ).encode("utf-8")

    try:
        proc = await asyncio.create_subprocess_exec(
            *verifier_command(lang),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return False, f"infra:spawn: [{type(e).__name__}] {e}", stats

    if proc.pid is not None:
        stats, stop_event = await monitor_process_resources(proc.pid)
    else:
        stop_event = asyncio.Event()
        stop_event.set()

    try:
        assert proc.stdin is not None
        try:
            proc.stdin.write(payload)
            await asyncio.wait_for(proc.stdin.drain(), timeout=_STDIN_WRITE_TIMEOUT)
            proc.stdin.close()
        except (asyncio.TimeoutError, ConnectionError, OSError) as e:
            proc.kill()
            await proc.wait()
            return False, f"infra:stdin: [{type(e).__name__}] {e}", stats

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, "infra:timeout", stats

        if proc.returncode != 0:
            detail = stderr.decode(errors="replace").strip()[-500:]
            return False, f"infra:exit {proc.returncode}: {detail}", stats

        try:
            verdict = json.loads(stdout)
        except json.JSONDecodeError:
            return False, "infra:decode", stats
        if not isinstance(verdict, dict):
            return False, "infra:decode", stats

        result = verdict.get("result")
        if not isinstance(result, str):
            return False, "infra:decode", stats
        return result == "success", result, stats
    finally:
        stop_event.set()
        await asyncio.sleep(0.1)  # give the monitor time to finish
