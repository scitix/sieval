"""Task pipeline state machine: context, stage outputs and ``meta.json`` types."""

from dataclasses import dataclass, field, replace
from typing import Any, NotRequired, Self, TypedDict

from sieval.core.models import ModelCallMeta
from sieval.core.types import JSONValue
from sieval.core.utils.serialization import obj_to_dict, sieval_record

from .consts import STAGE_TO_RESULT_FIELD, TaskAction, TaskStage
from .run_meta import RunIdentity, RunRecord


class TaskRunIdentity(TypedDict):
    """Registry identity of the task that produced a run.

    A deliberate subset of ``TaskMeta`` — see
    :func:`sieval.core.tasks.meta.get_task_run_identity` for what is left out
    and why. Fields are JSON-shaped (``eval_mode`` / ``status`` /
    ``reference_kind`` as plain ``str``) so this module needs no ``meta``
    import: ``meta`` imports ``task``, which imports this module.

    Every field but ``n_shot`` is the task's declaration, identical to its
    ``sieval/meta/index.json`` row. ``n_shot`` is the run's, so it may
    disagree with that row: the catalog says what the task advertises, this
    says what the run did.

    ``reference_kind`` (``"value"`` | ``"procedure"``) says what form the ground
    truth takes, and so why a judgement in these shards may carry no
    ``reference``. It is absent on a run written before the field shipped; read
    it with a ``"value"`` default, which is what every task declared implicitly
    until then.
    """

    name: str
    display_name: str
    dataset: str
    eval_mode: str
    n_shot: int
    tags: list[str]
    status: str
    reference_kind: str


class TaskRunMeta(TypedDict):
    """The ``meta.json`` document: where a task directory came from.

    ``version``, ``deterministic`` and ``task`` describe the run that created
    the directory and are kept across resumes; the resume gates read them.
    ``result_schema_version``
    (:data:`~sieval.core.tasks.run_meta.RESULT_SCHEMA_VERSION`) is what the
    version gate compares; ``version`` is provenance only.
    ``model``, ``config``, ``identity`` and ``runs`` are rewritten by every
    run — see :mod:`sieval.core.tasks.run_meta` for their shape. ``weights``
    names the weights ``model`` serves, when stated. ``config`` is
    the run's ``effective_config.yaml``, relative to this directory.

    Readers tolerate older files: ``result_schema_version`` absent is a run
    before 0.11.0, mapped from ``version`` and written by its next resume;
    ``deterministic`` absent is a pre-feature run, ``task`` absent is a
    pre-feature run or an undecorated ``Task``, and ``model`` / ``identity`` /
    ``runs`` absent is a run before stage identities. Nothing backfills those.
    """

    version: str
    result_schema_version: NotRequired[int]
    deterministic: NotRequired[bool]
    task: NotRequired[TaskRunIdentity]
    model: NotRequired[str]
    weights: NotRequired[str]
    config: NotRequired[str]
    identity: NotRequired[RunIdentity]
    runs: NotRequired[list[RunRecord]]


class TaskStageMeta(TypedDict, total=False):
    """Per-stage metadata recorded alongside each stage execution."""

    timestamp: float
    version: str  # producing sieval version — per-record provenance
    timing_s: float
    model_calls: list[ModelCallMeta]
    env: dict[str, JSONValue]
    extra: dict[str, JSONValue]  # for user-defined metadata
    # Set on the `preprocessed` entry: the sample's own data and prompt
    # digests (`run_meta.digest`), which a resume compares per sample.
    identity: dict[str, str | None]
    # Set on the `inferred` entry when the model under test refused a prompt as
    # longer than its context window: each refusal's message, in call order.
    context_overflow: list[str]


@sieval_record
@dataclass(frozen=True, slots=True)
class TaskStageOutput[T]:
    """Wrapper that pairs a stage return value with optional metadata."""

    value: T
    meta: TaskStageMeta | None = None


@dataclass(frozen=True, slots=True)
class TaskContext[TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback]:
    """Immutable state machine tracking one sample's journey through the pipeline.

    Each stage transition returns a **new** ``TaskContext`` via ``dataclasses.replace``.
    The runner drives transitions; tasks only read the context.

    Attributes:
        sample_id: Unique identifier for this sample (string key or integer index).
        raw_sample: The original sample from the dataset (may be ``None``).
        repeat_index: Which copy of a repeated split this sample came from (0-based),
            or ``None`` if none was recorded — normally an unrepeated split; see
            :meth:`serialize` for when absence means less than that. Read off the row
            by :func:`~sieval.core.datasets.repeat_index_of` at both seams that
            attach one: :meth:`~sieval.core.tasks.task.Task.make_context` and the
            runner's resume backfill.
            **A different axis from** ``iteration`` **and from a rollout index**:
            ``iteration`` counts feedback-loop passes over one sample, a rollout
            index counts the model's own samples within one request (``n``), and
            this counts copies of the whole split. All three can be >0 at once, and
            only this one distinguishes two rows that are otherwise identical.
        iteration: Current feedback-loop iteration (0-based).
        retry_count: Number of error retries consumed so far.
        stage: Current pipeline stage.
        stage_meta: Accumulated per-stage metadata, keyed by stage name.
        preprocess_result: Output of the preprocess stage (``None`` until completed).
        infer_result: Output of the infer stage (``None`` until completed).
        postprocess_result: Output of the postprocess stage (``None`` until completed).
        feedback_result: Output of the feedback stage (``None`` until completed).
        error_action: The :class:`TaskAction` that was executing when an error occurred.
        error_reason: Machine-readable error category (e.g. ``"timeout"``).
        error_msg: Human-readable error message.
        _is_snapshot: When ``True``, :meth:`serialize` emits only the current
            stage's result instead of the full context.
    """

    sample_id: str | int
    raw_sample: TRawSample | None = None

    repeat_index: int | None = None

    iteration: int = 0
    retry_count: int = 0

    stage: TaskStage = TaskStage.INITIAL
    stage_meta: dict[str, list[TaskStageMeta]] = field(default_factory=dict)

    preprocess_result: TPreprocessed | None = None
    infer_result: TInferred | None = None
    postprocess_result: TPostprocessed | None = None
    feedback_result: TFeedback | None = None

    error_action: TaskAction | None = None
    error_reason: str | None = None
    error_msg: str | None = None

    _is_snapshot: bool = False

    @property
    def is_snapshot(self) -> bool:
        """Whether this context serializes in snapshot (single-stage) mode."""
        return self._is_snapshot

    def is_terminal(self) -> bool:
        """Return ``True`` if the sample has reached FINAL or FAILED."""
        return self.stage in (TaskStage.FINAL, TaskStage.FAILED)

    def _append_history(
        self, target_dict: dict[str, list], key: str, value: Any
    ) -> dict[str, list]:
        """Return a copy of *target_dict* with *value* appended to the *key* list."""
        # Shallow copy the dict, only copy-on-write the specific list being modified.
        new_dict = target_dict.copy()
        current_list = new_dict.get(key, [])
        # Create a new list with the appended value
        new_dict[key] = current_list + [value]
        return new_dict

    def _transition(
        self, stage: TaskStage, meta: TaskStageMeta | None, **kwargs
    ) -> Self:
        """Create a new context transitioned to *stage* with optional metadata."""
        updates: dict[str, Any] = {"stage": stage, **kwargs}
        if meta:
            updates["stage_meta"] = self._append_history(
                self.stage_meta, stage.value, meta
            )
        return replace(self, **updates)

    def to_preprocessed(
        self, result: TPreprocessed, meta: TaskStageMeta | None = None
    ) -> Self:
        """Transition to PREPROCESSED."""
        return self._transition(TaskStage.PREPROCESSED, meta, preprocess_result=result)

    def to_inferred(self, result: TInferred, meta: TaskStageMeta | None = None) -> Self:
        """Transition to INFERRED."""
        return self._transition(TaskStage.INFERRED, meta, infer_result=result)

    def to_postprocessed(
        self, result: TPostprocessed, meta: TaskStageMeta | None = None
    ) -> Self:
        """Transition to POSTPROCESSED."""
        return self._transition(
            TaskStage.POSTPROCESSED, meta, postprocess_result=result
        )

    def to_feedback(self, result: TFeedback, meta: TaskStageMeta | None = None) -> Self:
        """Transition to FEEDBACK."""
        return self._transition(TaskStage.FEEDBACK, meta, feedback_result=result)

    def to_final(
        self,
    ) -> Self:
        """Mark the sample as successfully finalized (FINAL)."""
        return replace(self, stage=TaskStage.FINAL)

    def to_failed(self, action: TaskAction | None, reason: str, msg: str) -> Self:
        """Mark the sample as FAILED, recording the triggering action and error info."""
        return replace(
            self,
            stage=TaskStage.FAILED,
            error_action=action,  # None when iteration limit reached
            error_reason=reason,
            error_msg=msg,
        )

    def iterate(self) -> Self:
        """Bump the iteration counter and reset stage to INITIAL for another pass."""
        return replace(self, iteration=self.iteration + 1, stage=TaskStage.INITIAL)

    def next_action(self) -> TaskAction | None:
        """Return the :class:`TaskAction` that should follow the current stage.

        Returns ``None`` for FEEDBACK (handled by the runner), FINAL, and FAILED.
        """
        if self.stage == TaskStage.INITIAL:
            return TaskAction.PREPROCESS
        if self.stage == TaskStage.PREPROCESSED:
            return TaskAction.INFER
        if self.stage == TaskStage.INFERRED:
            return TaskAction.POSTPROCESS
        if self.stage == TaskStage.POSTPROCESSED:
            return TaskAction.FEEDBACK
        return None  # FEEDBACK handled by runner, FINAL/FAILED terminal

    def record_stage_meta(self, stage: TaskStage, meta: TaskStageMeta) -> Self:
        """Append *meta* to the history for *stage*, keeping the stage."""
        new_meta = self._append_history(self.stage_meta, stage.value, meta)
        return replace(self, stage_meta=new_meta)

    def make_snapshot(self) -> Self:
        """Return a copy with snapshot mode enabled (``_is_snapshot=True``)."""
        return replace(self, _is_snapshot=True)

    def serialize(
        self, store_type_metadata: bool, *, include_meta: bool = True
    ) -> dict[str, Any]:
        """Serialize the context to a plain dict.

        In snapshot mode only the current stage's result and latest metadata
        entry are included.  In full mode all stage results and the complete
        metadata history are emitted.
        """
        d: dict[str, Any] = {
            "sample_id": self.sample_id,
            "iteration": self.iteration,
            "stage": self.stage.value,
        }
        # Emitted only when the split was repeated, in both modes. Omitted rather
        # than written as null, so an archive from an unrepeated run stays
        # byte-identical to one written before this field existed.
        #
        # Presence says "this row is a copy"; absence only says no copy number was
        # recorded. A resume within one schema (`resume_schema_verdict`)
        # appends to shards whose older records predate the field, mixing a
        # single archive — so read a missing key as unknown, not as copy 0
        # (`stage_meta`'s per-record version tells them apart wherever it was
        # recorded).
        if self.repeat_index is not None:
            d["repeat_index"] = self.repeat_index
        if self._is_snapshot:
            field = STAGE_TO_RESULT_FIELD.get(self.stage)
            if field:
                v = getattr(self, field)
                if v is not None:
                    d[field] = obj_to_dict(v, store_type_metadata)
            # Snapshot meta
            if include_meta:
                meta_arr = self.stage_meta.get(self.stage.value)
                if meta_arr:
                    d["meta_last"] = obj_to_dict(meta_arr[-1], store_type_metadata)
        else:
            for f in STAGE_TO_RESULT_FIELD.values():
                v = getattr(self, f)
                if v is not None:
                    d[f] = obj_to_dict(v, store_type_metadata)
            if include_meta and self.stage_meta:
                d["stage_meta"] = obj_to_dict(self.stage_meta, store_type_metadata)

        if self.error_action:
            d["error_action"] = (
                self.error_action.value
                if hasattr(self.error_action, "value")
                else self.error_action
            )
        if self.error_reason:
            d["error_reason"] = self.error_reason
        if self.error_msg:
            d["error_msg"] = self.error_msg
        if self.retry_count:
            d["retry_count"] = self.retry_count
        return d
