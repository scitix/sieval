"""Task-side model requirements and normalized model bindings.

This module contains only immutable setup-plane values.  It deliberately has
no dependency on live models, task implementations, configuration loaders, or
serving-policy reconciliation.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from sieval.core.types import JSONValue

from ._shared import copy_json_value, validate_nonempty_string


class InputKind(StrEnum):
    """Provider-neutral shape of a model request."""

    COMPLETION = "completion"
    CHAT = "chat"


class InputModality(StrEnum):
    """Input modalities represented by the shared request IR."""

    TEXT = "text"
    IMAGE = "image"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class TaskRequirements:
    """Model semantics required by one task role.

    Serving facts and engine constraints intentionally do not belong here.
    They are reconciled after task requirements have been aggregated.
    """

    input: InputKind | None = None
    input_modalities: frozenset[InputModality] = field(
        default_factory=lambda: frozenset({InputModality.TEXT})
    )
    input_scoring: bool = False
    sampled_logprobs: bool = False
    min_top_logprobs: int | None = None

    def __post_init__(self) -> None:
        if self.input is not None and not isinstance(self.input, InputKind):
            raise TypeError("input must be an InputKind or None")

        try:
            modalities = frozenset(self.input_modalities)
        except TypeError as exc:
            raise TypeError("modalities must be an iterable of InputModality") from exc
        if not modalities:
            raise ValueError("modalities must not be empty")
        if any(not isinstance(item, InputModality) for item in modalities):
            raise TypeError("modalities must contain only InputModality values")
        object.__setattr__(self, "input_modalities", modalities)

        if not isinstance(self.input_scoring, bool):
            raise TypeError("input_scoring must be a boolean")
        if not isinstance(self.sampled_logprobs, bool):
            raise TypeError("sampled_logprobs must be a boolean")

        minimum = self.min_top_logprobs
        if minimum is not None:
            if isinstance(minimum, bool) or not isinstance(minimum, int):
                raise TypeError("min_top_logprobs must be an integer")
            if minimum < 1:
                raise ValueError("min_top_logprobs must be >= 1")
            object.__setattr__(self, "sampled_logprobs", True)


@dataclass(frozen=True)
class NamedModelBinding:
    """A normalized task binding that references a named model config."""

    binding_id: str
    root_deployment_key: str
    requested_model_id: str
    config_name: str
    dialect_id: str | None = None

    def __post_init__(self) -> None:
        _validate_binding_identity(self)
        validate_nonempty_string(self.config_name, "config_name")


@dataclass(frozen=True)
class InlineModelBinding:
    """A normalized task binding carrying an inline, JSON-safe config."""

    binding_id: str
    root_deployment_key: str
    requested_model_id: str
    config: Mapping[str, JSONValue]
    dialect_id: str | None = None

    def __post_init__(self) -> None:
        _validate_binding_identity(self)
        object.__setattr__(self, "config", _copy_json_mapping(self.config, "config"))


@dataclass(frozen=True)
class ExternalModelBinding:
    """A normalized binding to an already-created runtime plan.

    Only stable identifiers and the secret-free plan fingerprint cross this
    boundary; the external model, client, and connection remain runtime state.
    """

    binding_id: str
    root_deployment_key: str
    requested_model_id: str
    runtime_plan_fingerprint: str
    dialect_id: str | None = None

    def __post_init__(self) -> None:
        _validate_binding_identity(self)
        validate_nonempty_string(
            self.runtime_plan_fingerprint, "runtime_plan_fingerprint"
        )


type NormalizedModelBinding = (
    NamedModelBinding | InlineModelBinding | ExternalModelBinding
)


@dataclass(frozen=True)
class RequirementContext:
    """Normalized, JSON-safe inputs available to task requirement hooks."""

    model_bindings: Mapping[str, NormalizedModelBinding] = field(default_factory=dict)
    task_args: Mapping[str, JSONValue] = field(default_factory=dict)
    dataset_config: Mapping[str, JSONValue] = field(default_factory=dict)
    infer_args: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bindings: dict[str, NormalizedModelBinding] = {}
        for role, binding in self.model_bindings.items():
            validate_nonempty_string(role, "model binding role")
            if not isinstance(
                binding,
                (NamedModelBinding, InlineModelBinding, ExternalModelBinding),
            ):
                raise TypeError(
                    f"model binding {role!r} must be a NormalizedModelBinding"
                )
            bindings[role] = binding
        object.__setattr__(self, "model_bindings", MappingProxyType(bindings))
        object.__setattr__(
            self, "task_args", _copy_json_mapping(self.task_args, "task_args")
        )
        object.__setattr__(
            self,
            "dataset_config",
            _copy_json_mapping(self.dataset_config, "dataset_config"),
        )
        object.__setattr__(
            self, "infer_args", _copy_json_mapping(self.infer_args, "infer_args")
        )


@dataclass(frozen=True)
class TaskModelRequirement:
    """One task role's requirements attached to its normalized binding."""

    role: str
    binding: NormalizedModelBinding
    requires: TaskRequirements
    source_task: str

    def __post_init__(self) -> None:
        validate_nonempty_string(self.role, "role")
        if not isinstance(
            self.binding,
            (NamedModelBinding, InlineModelBinding, ExternalModelBinding),
        ):
            raise TypeError("binding must be a NormalizedModelBinding")
        if not isinstance(self.requires, TaskRequirements):
            raise TypeError("requires must be TaskRequirements")
        validate_nonempty_string(self.source_task, "source_task")


@dataclass(frozen=True)
class AggregatedTaskRequirements:
    """Union of task requirements with evidence for every contributing value."""

    input: frozenset[InputKind] = frozenset()
    input_modalities: frozenset[InputModality] = frozenset()
    input_scoring: bool = False
    sampled_logprobs: bool = False
    min_top_logprobs: int | None = None
    input_sources: Mapping[InputKind, frozenset[str]] = field(default_factory=dict)
    modality_sources: Mapping[InputModality, frozenset[str]] = field(
        default_factory=dict
    )
    input_scoring_sources: frozenset[str] = frozenset()
    sampled_logprobs_sources: frozenset[str] = frozenset()
    min_top_logprobs_sources: Mapping[int, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        inputs = frozenset(self.input)
        if any(not isinstance(item, InputKind) for item in inputs):
            raise TypeError("input must contain only InputKind values")
        modalities = frozenset(self.input_modalities)
        if any(not isinstance(item, InputModality) for item in modalities):
            raise TypeError("modalities must contain only InputModality values")
        if not isinstance(self.input_scoring, bool):
            raise TypeError("input_scoring must be a boolean")
        if not isinstance(self.sampled_logprobs, bool):
            raise TypeError("sampled_logprobs must be a boolean")
        if self.min_top_logprobs is not None:
            if isinstance(self.min_top_logprobs, bool) or not isinstance(
                self.min_top_logprobs, int
            ):
                raise TypeError("min_top_logprobs must be an integer")
            if self.min_top_logprobs < 1:
                raise ValueError("min_top_logprobs must be >= 1")
            if not self.sampled_logprobs:
                raise ValueError(
                    "min_top_logprobs requires sampled_logprobs to be true"
                )

        object.__setattr__(self, "input", inputs)
        object.__setattr__(self, "input_modalities", modalities)
        object.__setattr__(
            self,
            "input_sources",
            _freeze_enum_sources(self.input_sources, InputKind, "input_sources"),
        )
        object.__setattr__(
            self,
            "modality_sources",
            _freeze_enum_sources(
                self.modality_sources, InputModality, "modality_sources"
            ),
        )
        object.__setattr__(
            self,
            "input_scoring_sources",
            _validate_and_freeze_sources(
                self.input_scoring_sources, "input_scoring_sources"
            ),
        )
        object.__setattr__(
            self,
            "sampled_logprobs_sources",
            _validate_and_freeze_sources(
                self.sampled_logprobs_sources, "sampled_logprobs_sources"
            ),
        )
        object.__setattr__(
            self,
            "min_top_logprobs_sources",
            _freeze_minimum_sources(self.min_top_logprobs_sources),
        )


def aggregate_task_requirements(
    requirements: Iterable[TaskModelRequirement],
) -> AggregatedTaskRequirements:
    """Purely union a batch of task-side requirements.

    Callers decide which bindings belong in a batch.  This function performs
    no deployment grouping or compatibility judgment; retaining conflicting
    input kinds is intentional so reconciliation can report all sources.
    """

    inputs: set[InputKind] = set()
    modalities: set[InputModality] = set()
    input_sources: dict[InputKind, set[str]] = {}
    modality_sources: dict[InputModality, set[str]] = {}
    input_scoring_sources: set[str] = set()
    sampled_logprobs_sources: set[str] = set()
    minimum_sources: dict[int, set[str]] = {}
    minimum: int | None = None

    for requirement in requirements:
        if not isinstance(requirement, TaskModelRequirement):
            raise TypeError("requirements must contain TaskModelRequirement values")
        required = requirement.requires
        source = requirement.source_task

        if required.input is not None:
            inputs.add(required.input)
            input_sources.setdefault(required.input, set()).add(source)
        for modality in required.input_modalities:
            modalities.add(modality)
            modality_sources.setdefault(modality, set()).add(source)
        if required.input_scoring:
            input_scoring_sources.add(source)
        if required.sampled_logprobs:
            sampled_logprobs_sources.add(source)
        if required.min_top_logprobs is not None:
            value = required.min_top_logprobs
            minimum = value if minimum is None else max(minimum, value)
            minimum_sources.setdefault(value, set()).add(source)

    return AggregatedTaskRequirements(
        input=frozenset(inputs),
        input_modalities=frozenset(modalities),
        input_scoring=bool(input_scoring_sources),
        sampled_logprobs=bool(sampled_logprobs_sources),
        min_top_logprobs=minimum,
        input_sources={key: frozenset(value) for key, value in input_sources.items()},
        modality_sources={
            key: frozenset(value) for key, value in modality_sources.items()
        },
        input_scoring_sources=frozenset(input_scoring_sources),
        sampled_logprobs_sources=frozenset(sampled_logprobs_sources),
        min_top_logprobs_sources={
            key: frozenset(value) for key, value in minimum_sources.items()
        },
    )


def _validate_binding_identity(binding: NormalizedModelBinding) -> None:
    validate_nonempty_string(binding.binding_id, "binding_id")
    validate_nonempty_string(binding.root_deployment_key, "root_deployment_key")
    validate_nonempty_string(binding.requested_model_id, "requested_model_id")
    if binding.dialect_id is not None:
        validate_nonempty_string(binding.dialect_id, "dialect_id")


def _copy_json_mapping(
    value: Mapping[str, JSONValue], path: str
) -> Mapping[str, JSONValue]:
    copied = copy_json_value(value, path)
    if not isinstance(copied, dict):
        raise TypeError(f"{path} must be a mapping")
    return MappingProxyType(copied)


def _validate_and_freeze_sources(sources: Iterable[str], name: str) -> frozenset[str]:
    frozen = frozenset(sources)
    for source in frozen:
        validate_nonempty_string(source, f"{name} item")
    return frozen


def _freeze_enum_sources[T: (InputKind, InputModality)](
    sources: Mapping[T, frozenset[str]],
    enum_type: type[T],
    name: str,
) -> Mapping[T, frozenset[str]]:
    copied: dict[T, frozenset[str]] = {}
    for value, names in sources.items():
        if not isinstance(value, enum_type):
            raise TypeError(f"{name} keys must be {enum_type.__name__} values")
        copied[value] = _validate_and_freeze_sources(names, f"{name}[{value.value}]")
    return MappingProxyType(copied)


def _freeze_minimum_sources(
    sources: Mapping[int, frozenset[str]],
) -> Mapping[int, frozenset[str]]:
    copied: dict[int, frozenset[str]] = {}
    for minimum, names in sources.items():
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise TypeError("min_top_logprobs_sources keys must be integers")
        if minimum < 1:
            raise ValueError("min_top_logprobs_sources keys must be >= 1")
        copied[minimum] = _validate_and_freeze_sources(
            names, f"min_top_logprobs_sources[{minimum}]"
        )
    return MappingProxyType(copied)
