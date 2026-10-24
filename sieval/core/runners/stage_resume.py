"""Per-stage resume: a recorded task directory compared with a new run.

A resume rolls each recorded sample back to the earliest stage whose identity
changed, and reruns from there:

==============================  ==============================================
Data differs                    That sample reruns from the start.
Preprocess differs              Every sample reruns from the start.
Inference differs               Refused: :class:`StageResumeError` names it.
Postprocess differs             ``postprocess`` onward reruns on the recorded
                                inference.
Feedback differs                ``feedback`` onward reruns on the recorded
                                ``postprocess``.
Report differs                  ``report.json`` is recomputed from the
                                recorded samples: no engine, no judge call.
==============================  ==============================================

A sample past iteration 0 restarts instead of rolling back to a grading stage:
its later inference saw the old grader's feedback.

A schema-1 ``grader`` (0.10.0 to 0.11.x) matches all grading stages or none
(:func:`~sieval.core.tasks.run_meta.legacy_grader_keys`).

Each ``preprocess`` record is stamped with the task's ``preprocess`` key. A
directory that recorded rendered prompts (identity schemas 1 and 2) is
upgraded by rendering its recorded samples once.

A sample's data digest comes from its ``preprocess`` record. For a directory
recorded before them, the prompt is digested from the ``preprocess`` output,
and data and inference come from the caller's :class:`LegacyStageMatch`. A
stage neither can tell is unknown, and a resume that would reuse it is
refused. Nothing here writes to the recorded directory.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.run_meta import (
    GRADING_STAGES,
    STAGES,
    PreprocessIdentity,
    ResumeOutcomes,
    RunIdentity,
    StageCode,
    digest,
    grader_key,
    grader_task,
    grading_identities,
    inference_identity,
    is_code_identity,
    judge_identities,
    legacy_grader_keys,
    preprocess_identity,
    preprocess_key,
    prompt_of,
    read_identity,
    split_digests,
    stage_code,
    stage_key,
)
from sieval.core.tasks.services import service_identities
from sieval.core.types import JSONValue
from sieval.core.utils.atomic import atomic_write

if TYPE_CHECKING:
    from sieval.core.tasks.task import Task


ResumeStage = Literal["data", "inference"]


class StageResumeError(RuntimeError):
    """A resume refused because a stage it would reuse differs or is unknown."""

    def __init__(
        self, stage: ResumeStage, message: str, *, task: str | None = None
    ) -> None:
        self.stage = stage
        self.task = task
        super().__init__(message)


class LegacyStageMatch(TypedDict, total=False):
    """Whether a recorded run's persisted config shows this run's data or inference.

    Supplied by the caller that owns the persisted config, for a recorded
    directory whose ``meta.json`` and records carry no identity. ``True`` is the
    same, ``False`` different; a key left out is unknown.
    """

    data: bool
    inference: bool


class RunPlan(TypedDict):
    """What a run of one task would do, decided before it does anything.

    Unlike :class:`ResumePlan`, which compares identities alone, this is the
    runner's per-sample plan (:meth:`~sieval.core.runners.runner.TaskRunner.plan_run`).
    ``start`` is ``fresh`` (nothing recorded), ``resume`` (stage by stage), or
    ``reuse`` (the recorded ``report.json`` stands, recomputed when only
    ``report`` changed). ``outcomes`` counts the recorded samples as
    ``runs[].resume`` will, ``None`` for a fresh run. ``stages`` is the
    recorded directory's :class:`ResumePlan` against this run, ``None`` for a
    fresh run; its ``rerun_from`` is the runner's own decision. ``new`` counts
    samples with no record. ``to_infer`` counts the samples the run sends to
    the model under test, ``to_grade`` those it runs through a grading stage.
    """

    start: Literal["fresh", "resume", "reuse"]
    recorded: str | None
    samples: int
    outcomes: ResumeOutcomes | None
    stages: "ResumePlan | None"
    new: int
    to_infer: int
    to_grade: int


def recorded_identity(meta: Mapping[str, Any] | None) -> RunIdentity | None:
    """A recorded ``meta.json``'s ``identity`` block as schema 2.

    A schema-1 block's grading stages are ``None`` (see :func:`recorded_grader`).
    ``None`` when there is no block of a schema this build reads.
    """
    read = read_identity((meta or {}).get("identity"))
    return read[0] if read is not None else None


def recorded_grader(meta: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """A schema-1 block's combined ``grader`` identity; ``None`` for later ones."""
    read = read_identity((meta or {}).get("identity"))
    return read[1] if read is not None else None


def earliest_grading_change(
    recorded: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    legacy_match: bool | None = None,
) -> str | None:
    """The first grading stage whose identity differs, else ``None``.

    *recorded* is the recorded block as schema 2 and *current* this run's. A
    stage unknown on either side differs. For a schema-1 block pass
    *legacy_match*, whether its combined ``grader`` key is among
    :func:`~sieval.core.tasks.run_meta.legacy_grader_keys`: ``True`` means no
    grading stage changed, ``False`` that ``postprocess`` did.
    """
    if legacy_match is not None:
        return None if legacy_match else GRADING_STAGES[0]
    for stage in GRADING_STAGES:
        mine = stage_key(current.get(stage))
        theirs = stage_key((recorded or {}).get(stage))
        if mine is None or theirs is None or mine != theirs:
            return stage
    return None


#: A stage compared between a recorded run and this one.
type StageComparison = Literal["same", "differs", "unknown"]


class ResumePlan(TypedDict):
    """What a resume of one task directory would do, from identities alone.

    ``rerun_from`` is the earliest stage a finished sample reruns from
    (``preprocess`` restarts every sample), ``None`` if none changed; data is
    decided per sample, so only compared here. ``refused`` is ``inference``
    when reusing the recorded inference would be refused. ``legacy_grader`` is
    set for a schema-1 directory: ``kept`` or ``regrade``.
    ``legacy_preprocess`` marks a schema-1 or schema-2 directory, whose
    ``preprocess`` compares ``unknown`` until its samples are re-rendered.
    """

    identity_schema: int | None
    stages: dict[str, StageComparison]
    rerun_from: str | None
    refused: str | None
    legacy_grader: Literal["kept", "regrade"] | None
    legacy_preprocess: bool


def plan_resume(
    recorded_meta: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    legacy_grader_keys: Iterable[str] = (),
) -> ResumePlan:
    """Compare a recorded ``meta.json`` with this run's identities, running nothing.

    A pure entry point for a dry run. *current* is this run's identity;
    ``data`` may be ``None`` before anything is preprocessed and then
    compares as ``unknown``. *legacy_grader_keys* is
    :func:`~sieval.core.tasks.run_meta.legacy_grader_keys` for this task,
    read only for a schema-1 directory. The runner plans per sample and is
    authoritative; this says what a resume of a finished directory does.
    """
    block = (recorded_meta or {}).get("identity")
    schema = block.get("schema") if isinstance(block, Mapping) else None
    read = read_identity(block)
    recorded: Mapping[str, Any] = read[0] if read is not None else {}
    grader = read[1] if read is not None else None
    stages: dict[str, StageComparison] = {
        stage: _compare(stage, recorded.get(stage), current.get(stage))
        for stage in STAGES
    }
    legacy: Literal["kept", "regrade"] | None = None
    legacy_match: bool | None = None
    if grader is not None:
        key = grader_key(grader)
        legacy_match = key is not None and key in set(legacy_grader_keys)
        legacy = "kept" if legacy_match else "regrade"
        # Unsplittable: all the same, or postprocess changed and the rest unknown.
        for stage in GRADING_STAGES:
            stages[stage] = "same" if legacy_match else "unknown"
        if not legacy_match:
            stages[GRADING_STAGES[0]] = "differs"
    recorded_preprocess = recorded.get("preprocess")
    return {
        "identity_schema": schema if isinstance(schema, int) else None,
        "stages": stages,
        "rerun_from": "preprocess"
        if stages["preprocess"] == "differs"
        else earliest_grading_change(recorded, current, legacy_match),
        "refused": "inference" if stages["inference"] != "same" else None,
        "legacy_grader": legacy,
        "legacy_preprocess": recorded_preprocess is not None
        and not is_code_identity(recorded_preprocess),
    }


#: The task argument naming the split few-shot exemplars are drawn from.
FEWSHOT_SPLIT_ARG = "fewshot_split"


def task_preprocess_identity(task: "Task", code: StageCode) -> PreprocessIdentity:
    """:func:`~sieval.core.tasks.run_meta.preprocess_identity` for *task*.

    *code* is ``stage_code(...)["preprocess"]``. When that code reads a
    dataset split, the one its ``fewshot_split`` argument names is digested,
    else every split but ``test`` (which ``data`` covers). Raises
    :class:`ValueError` when one cannot be.
    """
    splits: dict[str, str] | None = None
    if code.get("splits"):
        dataset_dict = task.dataset.dataset_dict
        args = task.grader_args()
        named = args.get(FEWSHOT_SPLIT_ARG)
        if FEWSHOT_SPLIT_ARG in code.get("args", ()) and named in dataset_dict:
            names = [named]
        else:
            names = [name for name in dataset_dict if name != "test"]
        splits = split_digests(
            {str(name): cast(Iterable[object], dataset_dict[name]) for name in names}
        )
        if splits is None:
            raise ValueError("a dataset split the prompt reads cannot be digested")
    return preprocess_identity(code, task.grader_args(), splits)


def task_resume_plan(
    task: "Task",
    recorded_meta: Mapping[str, Any] | None,
    *,
    runner_keys: Mapping[str, JSONValue],
    recipe: Mapping[str, JSONValue] | None = None,
) -> ResumePlan:
    """:func:`plan_resume` for *task* against a recorded ``meta.json``.

    Computes this run's inference and grading identities as
    :class:`~sieval.core.runners.runner.TaskRunner` does, so call it after
    ``task.setup()``, which may resolve an argument the grading identities
    read. *runner_keys* holds :data:`~sieval.core.tasks.run_meta.INFERENCE_RUNNER_KEYS`
    and *recipe* the serving recipe, as the runner is given them. Reads files,
    starts nothing and calls no model.
    """
    task_cls = type(task)
    judges = judge_identities(task.judge_models(), task.judge_prompts())
    services = service_identities(task_cls.services)
    args = task.grader_args()
    code = stage_code(
        task_cls, [j["role"] for j in judges], task_cls.services, list(args)
    )
    grading = grading_identities(code, judges, services, args, grader_task(task_cls))
    current: dict[str, Any] = {
        "preprocess": task_preprocess_identity(task, code["preprocess"]),
        "inference": inference_identity(
            model=task.model,
            task_args=task.inference_args(),
            runner_keys=runner_keys,
            recipe=recipe,
        ),
        **grading,
    }
    return plan_resume(
        recorded_meta,
        current,
        legacy_grader_keys(task_cls, judges, services, args),
    )


def _compare(stage: str, recorded: object, current: object) -> StageComparison:
    theirs = (
        cast(Mapping[str, Any], recorded) if isinstance(recorded, Mapping) else None
    )
    mine = cast(Mapping[str, Any], current) if isinstance(current, Mapping) else None
    if stage in GRADING_STAGES:
        a, b = stage_key(theirs), stage_key(mine)
    elif stage == "preprocess":
        # A recorded digest of rendered prompts compares as unknown.
        a, b = preprocess_key(theirs), preprocess_key(mine)
    else:
        # Inference compares as a resume refuses it: by its configuration.
        a = theirs.get("digest") if theirs is not None else None
        b = mine.get("digest") if mine is not None else None
    if not isinstance(a, str) or not isinstance(b, str):
        return "unknown"
    return "same" if a == b else "differs"


class SampleDigests(TypedDict):
    """A recorded sample's input stamps (:func:`recorded_sample_digests`).

    ``preprocess`` is the :func:`~sieval.core.tasks.run_meta.preprocess_key`
    its record carries; ``prompt`` the digest of its rendered prompt, which
    only a record written under identity schema 1 or 2 carries.
    """

    data: str | None
    preprocess: str | None
    prompt: str | None
    source: Literal["record", "derived"]


def recorded_sample_digests(ctx: TaskContext) -> SampleDigests:
    """A recorded sample's data and preprocess stamps, and where they came from.

    ``record`` when its ``preprocess`` record carries a data digest and a
    ``preprocess`` key, or, under schema 1 or 2, a ``prompt`` digest.
    Otherwise ``derived``: the prompt is digested from the recorded
    ``preprocess`` output and the data is ``None``, since shards do not hold
    the raw sample.
    """
    for entry in ctx.stage_meta.get(TaskStage.PREPROCESSED.value, ()):
        if isinstance(entry, dict) and isinstance(entry.get("identity"), dict):
            identity = entry["identity"]
            data = identity.get("data")
            preprocess, prompt = identity.get("preprocess"), identity.get("prompt")
            if isinstance(data, str) and (
                isinstance(preprocess, str) or isinstance(prompt, str)
            ):
                return {
                    "data": data,
                    "preprocess": preprocess if isinstance(preprocess, str) else None,
                    "prompt": prompt if isinstance(prompt, str) else None,
                    "source": "record",
                }
    rendered: str | None = None
    if ctx.preprocess_result is not None:
        try:
            rendered = digest(prompt_of(ctx.preprocess_result))
        except (TypeError, ValueError):
            rendered = None
    return {"data": None, "preprocess": None, "prompt": rendered, "source": "derived"}


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


#: Written before :func:`copy_recorded_dir` copies anything, removed by
#: :func:`end_copy` once ``meta.json`` records the run. Until then the shards
#: are only a copy, possibly partial, and the next copy replaces them.
COPY_MARKER = ".copying"


def copy_pending(root: Path) -> str | None:
    """The source named by *root*'s :data:`COPY_MARKER`, or ``None``."""
    try:
        return (root / COPY_MARKER).read_text(encoding="utf-8")
    except OSError:
        return None


def end_copy(root: Path) -> None:
    """Remove *root*'s :data:`COPY_MARKER`."""
    (root / COPY_MARKER).unlink(missing_ok=True)


def abandon_copy(root: Path, created: list[Path]) -> None:
    """Undo a copy into *root* no run began on. The marker stays while any
    shard directory does, so leftovers remain replaceable."""
    remove_copied(created)
    if not iteration_dirs(root):
        end_copy(root)


def copy_recorded_dir(src: Path, dst: Path) -> list[Path]:
    """Copy *src*'s shards into *dst* for a resume across tasks.

    Only the ``<iteration>/`` shard directories are copied, with their run
    ordinals shifted (see :func:`_copy_idx`); the new run writes ``meta.json``,
    ``report.json`` and the views. Files are copied without their mode, so a
    read-only source yields writable copies. Returns the
    directories created, for :func:`remove_copied` to undo.

    :data:`COPY_MARKER` is written first; the caller removes it with
    :func:`end_copy` once ``meta.json`` records the run.

    The caller has established that *dst* holds no run, only perhaps an
    unfinished copy, whose shard directories are replaced.
    """
    dst.mkdir(parents=True, exist_ok=True)
    atomic_write(dst / COPY_MARKER, str(src).encode())
    created: list[Path] = []
    try:
        remove_copied(iteration_dirs(dst))
        tops = iteration_dirs(src)
        idx_files = [p for top in tops for p in top.rglob("*.idx")]
        shift = max((_last_run(p) for p in idx_files), default=0)
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
        abandon_copy(dst, created)
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
