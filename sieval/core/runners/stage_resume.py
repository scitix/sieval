"""Per-stage resume: a recorded task directory compared with a new run.

==========================  ====================================================
Data or prompt differ       That sample reruns from the start.
Inference differs           Refused: :class:`StageResumeError` names the stage.
Grader differs              ``postprocess``, ``feedback`` and ``report`` rerun
                            on the recorded inference.
==========================  ====================================================

A sample's data and prompt digests come from its ``preprocess`` record. For a
directory recorded before them, the prompt is digested from the ``preprocess``
output, and data and inference come from the caller's
:class:`LegacyStageMatch`. A stage neither can tell is unknown, and a resume
that would reuse it is refused. Nothing here writes to the recorded directory.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.run_meta import (
    IDENTITY_SCHEMA,
    STAGES,
    RunIdentity,
    digest,
    empty_identity,
    prompt_of,
)


class StageResumeError(RuntimeError):
    """A resume refused because a stage it would reuse differs or is unknown.

    ``stage`` is one of ``data``, ``prompt``, ``inference``.
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(message)


class LegacyStageMatch(TypedDict, total=False):
    """Whether a recorded run's persisted config shows this run's data or inference.

    Supplied by the caller that owns the persisted config, for a recorded
    directory whose ``meta.json`` and records carry no identity. ``True`` is the
    same, ``False`` different; a key left out is unknown.
    """

    data: bool
    inference: bool


def recorded_identity(meta: Mapping[str, Any] | None) -> RunIdentity | None:
    """The ``identity`` block of a recorded ``meta.json``, when it has this schema."""
    identity = (meta or {}).get("identity")
    if not isinstance(identity, dict) or identity.get("schema") != IDENTITY_SCHEMA:
        return None
    out = empty_identity()
    for stage in STAGES:
        value = identity.get(stage)
        if isinstance(value, dict):
            out[stage] = value  # type: ignore[literal-required]
    return out


def recorded_sample_digests(
    ctx: TaskContext,
) -> tuple[str | None, str | None, Literal["record", "derived"]]:
    """A recorded sample's ``(data, prompt)`` digests and where they came from.

    ``record`` when its ``preprocess`` record carries them. Otherwise
    ``derived``: the prompt is digested from the recorded ``preprocess`` output
    and the data is ``None``, since shards do not hold the raw sample.
    """
    for entry in ctx.stage_meta.get(TaskStage.PREPROCESSED.value, ()):
        if isinstance(entry, dict) and isinstance(entry.get("identity"), dict):
            identity = entry["identity"]
            data, prompt = identity.get("data"), identity.get("prompt")
            if isinstance(data, str) and isinstance(prompt, str):
                return data, prompt, "record"
    prompt: str | None = None
    if ctx.preprocess_result is not None:
        try:
            prompt = digest(prompt_of(ctx.preprocess_result))
        except (TypeError, ValueError):
            prompt = None
    return None, prompt, "derived"


def has_graded(ctx: TaskContext) -> bool:
    """Whether *ctx* holds an inference output and went on into grading."""
    return ctx.infer_result is not None and ctx.stage not in (
        TaskStage.INITIAL,
        TaskStage.PREPROCESSED,
        TaskStage.INFERRED,
    )


def iteration_dirs(root: Path) -> list[Path]:
    """The ``<iteration>/`` shard directories under *root*."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def copy_recorded_dir(src: Path, dst: Path) -> list[Path]:
    """Copy *src*'s shards into *dst* for a resume across tasks.

    Only the ``<iteration>/`` shard directories are copied, with their run
    ordinals shifted (see :func:`_copy_idx`); the new run writes ``meta.json``,
    ``report.json`` and the views. Files are copied without their mode, so a
    read-only source yields writable copies. Returns the
    directories created, for :func:`remove_copied` to undo.

    The caller has established that no run started in *dst*, so it holds no
    ``.idx``; shard directories already there are replaced.
    """
    remove_copied(iteration_dirs(dst))
    dst.mkdir(parents=True, exist_ok=True)
    tops = iteration_dirs(src)
    idx_files = [p for top in tops for p in top.rglob("*.idx")]
    shift = max((_last_run(p) for p in idx_files), default=0)
    created: list[Path] = []
    try:
        for top in tops:
            created.append(dst / top.name)
            for dirpath, _dirnames, filenames in os.walk(top):
                rel = Path(dirpath).relative_to(src)
                (dst / rel).mkdir(parents=True, exist_ok=True)
                for name in filenames:
                    if name.endswith(".jsonl"):
                        shutil.copyfile(Path(dirpath) / name, dst / rel / name)
                    elif name.endswith(".idx"):
                        _copy_idx(Path(dirpath) / name, dst / rel / name, shift)
    except BaseException:
        remove_copied(created)
        raise
    return created


#: Columns of an ``.idx`` line before the run ordinal the saver appends.
_IDX_COLUMNS_BEFORE_RUN = 8


def _run_of(parts: list[str]) -> int:
    """An ``.idx`` line's run ordinal; ``0`` for a line written before it."""
    if len(parts) > _IDX_COLUMNS_BEFORE_RUN and parts[_IDX_COLUMNS_BEFORE_RUN]:
        return int(parts[_IDX_COLUMNS_BEFORE_RUN])
    return 0


def _last_run(idx: Path) -> int:
    lines = idx.read_text(encoding="utf-8").splitlines()
    return max((_run_of(line.split("\t")) for line in lines if line), default=0)


def _copy_idx(src: Path, dst: Path, shift: int) -> None:
    """Copy an ``.idx`` file with each run ordinal lowered by *shift*.

    *shift* is the source's latest ordinal, so its records keep their order
    (a rewound sample's later record still wins) and all sit at or below
    ``0``, under every run of the new directory.
    """
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        run = _run_of(parts) - shift
        parts = parts[:_IDX_COLUMNS_BEFORE_RUN]
        parts += [""] * (_IDX_COLUMNS_BEFORE_RUN - len(parts))
        lines.append("\t".join([*parts, str(run)]) + "\n")
    dst.write_text("".join(lines), encoding="utf-8")


def remove_copied(created: list[Path]) -> None:
    """Remove the shard directories :func:`copy_recorded_dir` created."""
    for path in created:
        shutil.rmtree(path, ignore_errors=True)
