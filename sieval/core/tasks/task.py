"""Abstract base class for the five-stage evaluation pipeline.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import replace
from typing import ClassVar, Literal, Protocol, cast

from sieval.core.datasets import Dataset, repeat_index_of
from sieval.core.models import Model
from sieval.core.models.requirements import (
    InputKind,
    RequirementContext,
    TaskModelRequirement,
    TaskRequirements,
)
from sieval.core.types import JSONValue

from .context import TaskContext


class _RuntimeBindingView(Protocol):
    dialect_id: str
    available_capabilities: frozenset[str]
    capability_minimums: Mapping[str, Mapping[str, JSONValue]]


class _RuntimeModelView(Protocol):
    dialect_id: str
    runtime_plan: _RuntimeBindingView | None


_SGLANG_LEGACY_CAPABILITIES = frozenset(
    {"input_scoring", "sampled_logprobs", "top_logprobs"}
)


def _dialect_input_shape(dialect_id: str) -> tuple[frozenset[str], frozenset[str]]:
    if dialect_id == "sglang_legacy":
        return frozenset({InputKind.COMPLETION.value}), frozenset({"text"})

    from sieval.core.models.dialect_registry import get_dialect_spec

    spec = get_dialect_spec(dialect_id)
    return frozenset(spec.input_kinds), frozenset(spec.input_modalities)


def _required_capabilities(requires: TaskRequirements) -> frozenset[str]:
    capabilities: set[str] = set()
    if requires.input_scoring:
        capabilities.add("input_scoring")
    if requires.sampled_logprobs:
        capabilities.add("sampled_logprobs")
    if requires.min_top_logprobs is not None:
        capabilities.add("top_logprobs")
    return frozenset(capabilities)


def _validate_runtime_capabilities(
    task_name: str,
    requires: TaskRequirements,
    runtime_plan: _RuntimeBindingView,
) -> None:
    missing = _required_capabilities(requires) - runtime_plan.available_capabilities
    if missing:
        raise ValueError(
            f"{task_name} requires unavailable runtime capabilities: "
            + ", ".join(sorted(missing))
        )


def _validate_descriptor_capabilities(
    task_name: str,
    requires: TaskRequirements,
    dialect_id: str,
) -> None:
    required = _required_capabilities(requires)
    if not required:
        return
    if dialect_id == "sglang_legacy":
        available = _SGLANG_LEGACY_CAPABILITIES
    else:
        from sieval.core.models.capabilities import Supported
        from sieval.core.models.dialect_registry import capability_decisions_for

        decisions = capability_decisions_for(dialect_id)
        available = frozenset(
            key
            for key, decision in decisions.items()
            if isinstance(decision, Supported)
        )
    missing = required - available
    if missing:
        raise ValueError(
            f"{task_name} requires unsupported dialect capabilities: "
            + ", ".join(sorted(missing))
        )


class Task[
    TRawSample,
    TPreprocessed,
    TInferred,
    TPostprocessed,
    TFeedback,
    TReport,
](ABC):
    """User-facing interface for a five-stage evaluation pipeline.

    A Task defines the logic for each stage of the pipeline:
    **preprocess** -> **infer** -> **postprocess** -> **feedback** -> **report**.
    The runner drives execution; the Task only provides per-stage logic.

    Type Parameters:
        TRawSample: Raw sample type from the dataset.
        TPreprocessed: Output of :meth:`preprocess`.
        TInferred: Output of :meth:`infer`.
        TPostprocessed: Output of :meth:`postprocess`.
        TFeedback: Payload returned by :meth:`feedback`.
        TReport: Shape returned by :meth:`report`.
            Must be :data:`~sieval.core.types.JSONValue`-compatible
            (the runner serializes it via ``obj_to_dict``).
            Ideally this would be expressed as ``TReport: JSONValue``,
            but Python 3.12 does not support type-alias bounds on
            PEP 695 generics.

    Class Attributes:
        model_type: Declares the required model kind (``"chat"`` or ``"gen"``).
            ``None`` means no specific requirement (defaults to ``"chat"`` in
            *cli/session*).
        tags: Free-form tag set describing the task (e.g. ``{"gen", "zero_shot"}``).
            Used by the anomaly-detection framework to decide which rules apply.
        requires: Provider-neutral input and scoring semantics required from the
            candidate model. The decorator projects legacy ``model_type`` onto
            its input kind while preserving schema-version-1 metadata.
    """

    model_type: ClassVar[Literal["chat", "gen"] | None] = None
    tags: ClassVar[AbstractSet[str]] = frozenset()  # override in subclasses
    requires: ClassVar[TaskRequirements] = TaskRequirements()

    def __init__(
        self, dataset: Dataset[TRawSample], model: Model, name: str | None = None
    ):
        self._dataset = dataset
        self._model = model
        self._name = name

        self._validate_model_requirements()

    @classmethod
    def model_requirements_for(
        cls, context: RequirementContext
    ) -> tuple[TaskModelRequirement, ...]:
        """Attach this task's candidate requirements to its normalized binding."""

        return cls._bind_model_requirements(context, cls.requires)

    @classmethod
    def _bind_model_requirements(
        cls,
        context: RequirementContext,
        requires: TaskRequirements,
    ) -> tuple[TaskModelRequirement, ...]:
        if not isinstance(context, RequirementContext):
            raise TypeError("context must be a RequirementContext")

        bindings = context.model_bindings
        if not bindings:
            raise ValueError(f"{cls.__name__} requires one candidate model binding")
        if len(bindings) == 1:
            role = next(iter(bindings))
        else:
            has_candidate = "candidate" in bindings
            has_model_alias = "model" in bindings
            if has_candidate and has_model_alias:
                raise ValueError(
                    "ambiguous candidate binding: both 'candidate' and legacy "
                    "'model' roles are present"
                )
            if has_candidate:
                role = "candidate"
            elif has_model_alias:
                role = "model"
            else:
                raise ValueError(
                    "multiple model bindings require an explicit 'candidate' role"
                )

        return cls._bind_role_requirement(context, role, requires)

    @classmethod
    def _bind_role_requirement(
        cls,
        context: RequirementContext,
        role: str,
        requires: TaskRequirements,
    ) -> tuple[TaskModelRequirement, ...]:
        """Attach one explicit model role to a task requirement projection."""

        if not isinstance(context, RequirementContext):
            raise TypeError("context must be a RequirementContext")
        if not isinstance(role, str) or not role:
            raise TypeError("role must be a non-empty string")
        try:
            binding = context.model_bindings[role]
        except KeyError as exc:
            article = "an" if role[0].lower() in "aeiou" else "a"
            raise ValueError(
                f"{cls.__name__} requires {article} {role!r} model binding"
            ) from exc
        meta = cls.__dict__.get("_sieval_task_meta")
        source_task = getattr(meta, "name", cls.__name__)
        return (
            TaskModelRequirement(
                role=role,
                binding=binding,
                requires=requires,
                source_task=source_task,
            ),
        )

    @classmethod
    def _resolve_role_model[T](
        cls,
        role: str,
        configured: T,
        models_by_role: Mapping[str, Model] | None,
        *,
        build: Callable[[T], Model],
    ) -> Model:
        """Resolve one auxiliary model role to the Model the task will call.

        ``models_by_role`` is the pooled path a YAML run takes; otherwise
        ``build`` turns ``configured`` into the Model, so each task keeps its
        own "you must supply one" message. Supplying both is an error rather
        than a precedence question: silently preferring either would let a run
        score against a model the config did not name.
        """

        if models_by_role is not None:
            if configured is not None:
                raise ValueError(f"{role} and models_by_role cannot both be supplied")
            try:
                return models_by_role[role]
            except KeyError as exc:
                raise ValueError(
                    f"models_by_role is missing the {role!r} model"
                ) from exc
        return build(configured)

    @classmethod
    def _bind_top_logprobs_requirements(
        cls,
        context: RequirementContext,
        *,
        default: int,
        floor: int = 1,
    ) -> tuple[TaskModelRequirement, ...]:
        """Bind a CLP task's configured request breadth from normalized args."""

        if not isinstance(context, RequirementContext):
            raise TypeError("context must be a RequirementContext")
        raw = context.task_args.get("logprobs", default)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError("task_args.logprobs must be an integer")
        if raw < 1:
            raise ValueError("task_args.logprobs must be >= 1")
        minimum = max(raw, floor)
        requires = replace(cls.requires, min_top_logprobs=minimum)
        return cls._bind_model_requirements(context, requires)

    def _validate_model_requirements(self) -> None:
        """Validate this task against the model's bound dialect and runtime plan."""

        requires = self.requires
        if not isinstance(requires, TaskRequirements):
            raise TypeError(f"{type(self).__name__}.requires must be TaskRequirements")

        view = cast("_RuntimeModelView", self._model)
        dialect_id = getattr(view, "dialect_id", None)
        if not isinstance(dialect_id, str) or not dialect_id:
            raise TypeError(
                f"{type(self).__name__} requires a model with a bound dialect_id"
            )
        runtime_plan = getattr(view, "runtime_plan", None)
        if runtime_plan is not None and runtime_plan.dialect_id != dialect_id:
            raise ValueError(
                "model runtime plan dialect does not match model.dialect_id: "
                f"{runtime_plan.dialect_id!r} != {dialect_id!r}"
            )

        input_kinds, modalities = _dialect_input_shape(dialect_id)
        if requires.input is not None and requires.input.value not in input_kinds:
            raise TypeError(
                f"{type(self).__name__} requires input={requires.input.value!r}, "
                f"but dialect {dialect_id!r} accepts {sorted(input_kinds)}"
            )
        requested_modalities = {item.value for item in requires.input_modalities}
        unsupported_modalities = requested_modalities - modalities
        if unsupported_modalities:
            raise TypeError(
                f"{type(self).__name__} requires unsupported input modalities "
                f"{sorted(unsupported_modalities)} for dialect {dialect_id!r}"
            )

        if runtime_plan is not None:
            _validate_runtime_capabilities(type(self).__name__, requires, runtime_plan)
        else:
            _validate_descriptor_capabilities(type(self).__name__, requires, dialect_id)

    @property
    def dataset(self) -> Dataset[TRawSample]:
        return self._dataset

    @property
    def model(self) -> Model:
        return self._model

    @property
    def name(self) -> str:
        """Filesystem-safe task name, derived from *name* or the class name."""
        task_name = self._name or self.__class__.__name__ or "task"
        # Sanitize for filesystem safety
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("._-") or "task"
        return safe_name

    n_shot: int = 0
    """Few-shot exemplars this task renders — what ``meta.json`` records.

    Seeded on the class by ``@sieval_task(n_shot=...)``, so a task with no
    shot-count knob is already correct and needs no code of its own. A task
    whose constructor takes one assigns ``self.n_shot`` in ``__init__``, which
    shadows the class value for that instance: the catalog says what the task
    advertises, a run directory says what the run did. ``"n_shot" in
    task.__dict__`` distinguishes the two after the fact.

    Deliberately a plain class attribute rather than a ``ClassVar`` like
    :attr:`model_type` / :attr:`tags`: those are never set per instance, while
    this is the one field a run can change — annotating it ``ClassVar`` would
    make the shadowing assignment a type error.
    """

    def make_context(
        self, sample_id: str | int, raw: TRawSample | None = None
    ) -> TaskContext[TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback]:
        """Create a :class:`TaskContext` for *sample_id*.

        If *raw* is ``None`` and *sample_id* is a valid integer index into
        the dataset's test set, the raw sample is fetched on demand.

        A ``repeat_index`` column left by :meth:`~sieval.core.datasets.Dataset.repeat`
        is read onto the context here — every context is built through this method, so
        no task opts in and a resume re-derives the same value from the same row. The
        runner's resume backfill stamps it too, through the same
        :func:`~sieval.core.datasets.repeat_index_of`.
        """
        # If raw not supplied and integer index available, attempt lazy fetch
        if (
            raw is None
            and isinstance(sample_id, int)
            and self._dataset.test_set
            and 0 <= sample_id < len(self._dataset.test_set)
        ):
            raw = cast(TRawSample, self._dataset.test_set[sample_id])
        return TaskContext(sample_id, raw, repeat_index=repeat_index_of(raw))

    @abstractmethod
    async def preprocess(
        self,
        raw: TRawSample,
        ctx: TaskContext[
            TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback
        ],
    ) -> TPreprocessed:
        """Transform a raw sample into the format expected by :meth:`infer`."""
        ...

    @abstractmethod
    async def infer(
        self,
        pre: TPreprocessed,
        ctx: TaskContext[
            TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback
        ],
    ) -> TInferred:
        """Run model inference on a preprocessed sample."""
        ...

    @abstractmethod
    async def postprocess(
        self,
        inf: TInferred,
        ctx: TaskContext[
            TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback
        ],
    ) -> TPostprocessed:
        """Extract or normalize the inference output for evaluation."""
        ...

    @abstractmethod
    async def feedback(
        self,
        post: TPostprocessed,
        ctx: TaskContext[
            TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback
        ],
    ) -> tuple[bool, TFeedback]:
        """Evaluate a postprocessed result and decide whether to finalize.

        Returns:
            A ``(finalize, payload)`` tuple.  When *finalize* is ``True`` the
            sample transitions to FINAL; when ``False`` the runner may iterate.
        """
        ...

    @abstractmethod
    async def report(
        self,
        finals: list[
            TaskContext[TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback]
        ],
        fails: list[
            TaskContext[TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback]
        ],
    ) -> TReport:
        """Aggregate finalized and failed contexts into a task-level report."""
        ...

    async def setup(self) -> None:
        """Lifecycle hook called before execution begins (no-op by default)."""
        return

    async def shutdown(self) -> None:
        """Lifecycle hook called after execution ends (no-op by default)."""
        return
