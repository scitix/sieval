"""Per-stage resume: what a recorded task directory says, compared with a new run.

A resume compares the stage identities this run computes with the ones a
recorded directory holds, and decides per sample and per stage:

==========================  ====================================================
Data or prompt differ       The sample reruns from the start; the rest are kept.
Inference differs           Refused: :class:`StageResumeError` names the stage.
Grader differs              Every kept sample reruns ``postprocess`` and
                            ``feedback`` on its recorded inference; ``report``
                            reruns.
==========================  ====================================================

A recorded sample's data and prompt digests come from its ``preprocess``
record. A directory recorded before those existed still holds each sample's
``preprocess`` output, so its prompt is digested from that. Its data and
inference are read from the persisted config by the caller, as
:class:`LegacyStageMatch`. A stage that neither source can tell is unknown, and
a resume that would reuse it is refused. Nothing here writes to the recorded
directory.

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

    Only the ``<iteration>/`` shard directories are copied: they are the
    record, and each ``.idx`` line's run ordinal is reset to ``0`` (see
    :func:`_copy_idx`). ``manifest.json`` is rebuilt from their indexes;
    ``meta.json``, ``report.json`` and the views are written by the new run.
    Files are copied without their mode, so a read-only source yields writable
    copies. Returns the directories created, for :func:`remove_copied` to undo.

    The caller has established that no run started in *dst*, so shard
    directories already there are an earlier copy that was interrupted before
    its run began; they are replaced.
    """
    remove_copied(iteration_dirs(dst))
    dst.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for top in iteration_dirs(src):
            target = dst / top.name
            created.append(target)
            for dirpath, _dirnames, filenames in os.walk(top):
                rel = Path(dirpath).relative_to(src)
                (dst / rel).mkdir(parents=True, exist_ok=True)
                for name in filenames:
                    if name.endswith(".jsonl"):
                        shutil.copyfile(Path(dirpath) / name, dst / rel / name)
                    elif name.endswith(".idx"):
                        _copy_idx(Path(dirpath) / name, dst / rel / name)
    except BaseException:
        remove_copied(created)
        raise
    return created


#: Columns of an ``.idx`` line before the run ordinal the saver appends.
_IDX_COLUMNS_BEFORE_RUN = 8


def _copy_idx(src: Path, dst: Path) -> None:
    """Copy an ``.idx`` file with every line's run ordinal set to ``0``.

    The copied records come before this directory's history, so any record
    the new run writes supersedes them, whatever run wrote them at the source.
    """
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        parts = parts[:_IDX_COLUMNS_BEFORE_RUN]
        parts += [""] * (_IDX_COLUMNS_BEFORE_RUN - len(parts))
        lines.append("\t".join([*parts, "0"]) + "\n")
    dst.write_text("".join(lines), encoding="utf-8")


def remove_copied(created: list[Path]) -> None:
    """Remove the shard directories :func:`copy_recorded_dir` created."""
    for path in created:
        shutil.rmtree(path, ignore_errors=True)
