"""Abstract base class for the five-stage evaluation pipeline."""

import re
from abc import ABC, abstractmethod
from collections.abc import Set as AbstractSet
from typing import ClassVar, Literal

from sieval.core.datasets import Dataset
from sieval.core.models import Model

from .context import TaskContext


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
    """

    model_type: ClassVar[Literal["chat", "gen"] | None] = None
    tags: ClassVar[AbstractSet[str]] = frozenset()  # override in subclasses

    def __init__(
        self, dataset: Dataset[TRawSample], model: Model, name: str | None = None
    ):
        self._dataset = dataset
        self._model = model
        self._name = name

        if self.model_type is not None:
            self._validate_model_type()

    def _validate_model_type(self) -> None:
        """Raise ``TypeError`` if the model's kind does not match :attr:`model_type`."""
        from sieval.core.models import ChatModel, GenModel, SglangGenModel

        expected_type = self.model_type
        if isinstance(self._model, ChatModel):
            actual_type = "chat"
        elif isinstance(self._model, (GenModel, SglangGenModel)):
            actual_type = "gen"
        else:
            raise TypeError(
                f"{self.__class__.__name__} requires a ChatModel or GenModel, "
                f"but got {type(self._model).__name__}. "
                f"Please check your model configuration."
            )

        if expected_type != actual_type:
            raise TypeError(
                f"{self.__class__.__name__} requires model_type='{expected_type}', "
                f"but got '{actual_type}' model. "
                f"Please check your model configuration."
            )

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

    def make_context(
        self, sample_id: str | int, raw: TRawSample | None = None
    ) -> TaskContext[TRawSample, TPreprocessed, TInferred, TPostprocessed, TFeedback]:
        """Create a :class:`TaskContext` for *sample_id*.

        If *raw* is ``None`` and *sample_id* is a valid integer index into
        the dataset's test set, the raw sample is fetched on demand.
        """
        # If raw not supplied and integer index available, attempt lazy fetch
        if (
            raw is None
            and isinstance(sample_id, int)
            and self._dataset.test_set
            and 0 <= sample_id < len(self._dataset.test_set)
        ):
            raw = self._dataset.test_set[sample_id]  # type: ignore[invalid-assignment]
        return TaskContext(sample_id, raw)

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
