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

_REGISTRY = "ghcr.io/nuprl/agnostics"

# The verifier is scoring data, so it is pinned the way a dataset revision is:
# by digest, not by the mutable tag upstream's CLI passes. A tag can be moved
# under a finished leaderboard without anything on disk changing.
#
# Resolved from the registry on 2026-08-23 (each digest verified against the
# manifest's own `Docker-Content-Digest`). All eight are single-platform
# **linux/amd64** OCI manifests, not multi-arch indexes, so a digest pin also
# pins the architecture -- on arm64 these will not run and the command has to be
# overridden.
#
# Note the tags are **file extensions**, not language names: Julia is `jl`,
# OCaml `ml`, Fortran `f90`. The framework repo's directories spell the names out
# in full (`executors/julia`), so the tag list is the authority, not the source
# tree. Published: lua, r, python, jl, java, cpp, ml, f90 -- upstream's `c` and
# `rust` executors have no published image.
_IMAGE_DIGESTS = {
    "lua": "sha256:c1e437979ceb65a1da46165ab11515d63643ce58fcdb6a31d22fbb3a409117b0",
    "r": "sha256:cecf0478d8d090945b296c02515bd46a46107a2fd2860010f94c2a1fdf519c2b",
    "python": "sha256:c87079f74d78b9ba514d3431139c6fabb5c109321c95e5fee38a01676d375be3",
    "jl": "sha256:b4f00939f8c177818595c1f90b992250c12c159ace7a416cc4410e0c0742a03d",
    "java": "sha256:685d73167565b84f5daae1fe2b984f5ea19d60b4a2da50c4a5d7729a74bf5810",
    "cpp": "sha256:c154787ba5af5cd873d77e5fedfd4c0f5d79ac3668a9e6e958cd472932e6ceb5",
    "ml": "sha256:005b5f7c56ba1f6266b4a7c5e306bdb93cc468f2dd8663480aba6f06187715f0",
    "f90": "sha256:ac5487f649ba00c2cdb31451c296e5daf5dac7d3da7422eeaa640f2c98502d13",
}

# Upstream's podman invocation, with the image resolved to a pinned digest.
# `{image}` is the digest-pinned reference; `{lang}` is still substituted, for an
# override that wants the bare tag. Override the whole command via the env var
# above for a different runtime (upstream also supports
# `apptainer run --contain --writable-tmpfs <file>.sif`), a mirrored registry, or
# a local test double.
_DEFAULT_COMMAND = "podman run --rm -i --tmpfs /ramdisk:size=512m,exec {image}"

# `lang` reaches an argv slot, so it is constrained to what a container tag can
# be. Not a sandbox -- the command template is trusted and this is not -- but it
# keeps a tag from smuggling in a separator or a path.
_LANG_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9_.+-]{0,31}\Z")

# Upstream gives writing the payload its own budget, separate from the wall the
# process runs under, because a decoded LiveCodeBench suite is tens of MB and a
# slow write is not a slow program. Its `stdin_write_timeout=300` is kept.
_STDIN_WRITE_TIMEOUT = 300.0


def verifier_command(lang: str) -> tuple[list[str], str | None]:
    """Return ``(argv, image)``; ``image`` is the pinned reference that will run.

    ``image`` is ``None`` when the command came from the environment override,
    because an arbitrary command's image cannot be known from here -- and a
    reported value that might be wrong is worse than an absent one, since this
    string travels into the run record as the verdict's provenance.

    Raises ``KeyError`` for a language with no pinned digest, rather than falling
    back to the floating tag. An unpinned verifier is the failure this function
    exists to prevent, so a new language is added to ``_IMAGE_DIGESTS`` (or run
    through the override), never silently floated.
    """
    override = os.environ.get(_COMMAND_ENV_VAR)
    if override:
        digest = _IMAGE_DIGESTS.get(lang)
        image = f"{_REGISTRY}@{digest}" if digest else None
        argv = [
            part.replace("{image}", image or "").replace("{lang}", lang)
            for part in shlex.split(override)
        ]
        # Report the pinned image only if the override actually asked for it.
        # A template using `{lang}` (or naming its own image outright) runs
        # something this table cannot vouch for, and naming a digest that did not
        # run is worse than reporting nothing -- the value is the verdict's
        # provenance in the run record.
        return argv, (image if "{image}" in override else None)

    image = f"{_REGISTRY}@{_IMAGE_DIGESTS[lang]}"
    argv = [
        part.replace("{image}", image).replace("{lang}", lang)
        for part in shlex.split(_DEFAULT_COMMAND)
    ]
    return argv, image


async def execute_agnostics(
    code: str,
    inputs: list[str],
    expect_outputs: list[str],
    lang: str,
    timeout: float,
) -> tuple[bool, str, ResourceStats, str | None]:
    """Return ``(passed, msg, stats, image)``.

    ``msg`` is the container's ``result`` value verbatim when the container ran
    and spoke the protocol, and an ``infra:<reason>`` code when it did not, so a
    client can separate "the program is wrong" from "the harness could not ask".
    Upstream collapses every one of the latter to ``"fail"`` and recovers only
    the stdin-write case, from a stderr suffix; the split is named here instead,
    where it is known. It does not change what counts as a pass: only
    ``"success"`` does, either way.

    ``image`` is the digest-pinned reference that ran, for the caller to record
    beside the verdict; ``None`` under a command override, where it is unknowable.

    ``timeout`` serves as both the container's own ``timeout_s`` and the wall
    this process is held to -- one number in both roles, as upstream sends it.
    Note the container applies it **per test case** (its harness passes
    ``timeout_s`` straight to each ``subprocess.run``), so a multi-case suite can
    need many multiples of the wall and is killed from outside first -- which is
    why a timing-out submission surfaces as ``infra:timeout`` rather than the
    protocol's own ``fail:timeout``. That tension is upstream's, and widening the
    wall here would move scores.
    """
    stats = ResourceStats()
    if not _LANG_PATTERN.match(lang):
        return False, f"infra:bad-lang: {lang!r}", stats, None

    try:
        command, image = verifier_command(lang)
    except KeyError:
        # No pinned digest, and no override to take responsibility for one.
        # Refusing beats floating a tag: an unpinned verifier scores silently.
        return False, f"infra:unpinned-lang: {lang!r}", stats, None

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
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return False, f"infra:spawn: [{type(e).__name__}] {e}", stats, image

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
            return False, f"infra:stdin: [{type(e).__name__}] {e}", stats, image

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, "infra:timeout", stats, image

        if proc.returncode != 0:
            detail = stderr.decode(errors="replace").strip()[-500:]
            return False, f"infra:exit {proc.returncode}: {detail}", stats, image

        try:
            verdict = json.loads(stdout)
        except json.JSONDecodeError:
            return False, "infra:decode", stats, image
        if not isinstance(verdict, dict):
            return False, "infra:decode", stats, image

        result = verdict.get("result")
        if not isinstance(result, str):
            return False, "infra:decode", stats, image
        return result == "success", result, stats, image
    finally:
        stop_event.set()
        await asyncio.sleep(0.1)  # give the monitor time to finish
