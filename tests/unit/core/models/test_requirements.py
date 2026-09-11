"""Tests for task-side model requirements and normalized bindings.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import dataclasses
from typing import Any, cast

import pytest

from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    ExternalModelBinding,
    InlineModelBinding,
    InputKind,
    InputModality,
    NamedModelBinding,
    RequirementContext,
    TaskModelRequirement,
    TaskRequirements,
    aggregate_task_requirements,
)


def _binding(name: str = "primary") -> NamedModelBinding:
    return NamedModelBinding(
        binding_id=f"binding:{name}",
        root_deployment_key="deployment:base",
        requested_model_id="org/model",
        config_name=name,
        dialect_id="openai_chat",
    )


def _requirement(
    source: str,
    requires: TaskRequirements,
    *,
    role: str = "model",
) -> TaskModelRequirement:
    return TaskModelRequirement(
        role=role,
        binding=_binding(),
        requires=requires,
        source_task=source,
    )


class TestTaskRequirements:
    def test_defaults_are_text_only_and_frozen(self) -> None:
        requires = TaskRequirements()

        assert requires.input is None
        assert requires.input_modalities == {InputModality.TEXT}
        assert not requires.input_scoring
        assert not requires.sampled_logprobs
        assert requires.min_top_logprobs is None
        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, requires).input = InputKind.CHAT

    def test_modality_iterable_is_normalized_to_frozenset(self) -> None:
        source = {InputModality.TEXT, InputModality.IMAGE}
        requires = TaskRequirements(
            input=InputKind.CHAT, input_modalities=cast(Any, source)
        )

        source.clear()
        assert requires.input_modalities == {
            InputModality.TEXT,
            InputModality.IMAGE,
        }
        assert isinstance(requires.input_modalities, frozenset)

    def test_positive_top_logprob_minimum_implies_sampled_logprobs(self) -> None:
        assert TaskRequirements(min_top_logprobs=4).sampled_logprobs

    @pytest.mark.parametrize("value", [0, -1])
    def test_nonpositive_top_logprob_minimum_is_rejected(self, value: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            TaskRequirements(min_top_logprobs=value)

    @pytest.mark.parametrize("value", [True, 1.5, "2"])
    def test_noninteger_top_logprob_minimum_is_rejected(self, value: object) -> None:
        with pytest.raises(TypeError, match="must be an integer"):
            TaskRequirements(min_top_logprobs=cast(Any, value))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"input": "chat"}, "InputKind"),
            ({"input_modalities": 1}, "iterable"),
            ({"input_modalities": frozenset()}, "must not be empty"),
            ({"input_modalities": {"text"}}, "InputModality"),
            ({"input_scoring": 1}, "boolean"),
            ({"sampled_logprobs": 1}, "boolean"),
        ],
    )
    def test_invalid_requirement_values_are_rejected(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            TaskRequirements(**cast(Any, kwargs))

    def test_function_tools_defaults_off_and_requires_no_capability(self) -> None:
        from sieval.core.tasks.task import _required_capabilities

        assert TaskRequirements().function_tools is False
        assert "function_tools" not in _required_capabilities(TaskRequirements())

    def test_function_tools_flag_demands_the_capability(self) -> None:
        from sieval.core.tasks.task import _required_capabilities

        requires = TaskRequirements(function_tools=True)
        assert _required_capabilities(requires) == frozenset({"function_tools"})

    def test_function_tools_rejects_a_non_bool(self) -> None:
        with pytest.raises(TypeError, match="function_tools must be a boolean"):
            TaskRequirements(function_tools=cast(Any, "yes"))


class TestNormalizedBindings:
    def test_all_bindings_are_frozen_and_hold_only_setup_values(self) -> None:
        bindings = (
            _binding(),
            InlineModelBinding(
                binding_id="inline:grader",
                root_deployment_key="deployment:grader",
                requested_model_id="org/grader",
                config={"temperature": 0.0},
            ),
            ExternalModelBinding(
                binding_id="external:judge",
                root_deployment_key="deployment:judge",
                requested_model_id="org/judge",
                runtime_plan_fingerprint="sha256:abc",
            ),
        )

        for binding in bindings:
            assert "model" not in {item.name for item in dataclasses.fields(binding)}
            assert "client" not in {item.name for item in dataclasses.fields(binding)}
            with pytest.raises(dataclasses.FrozenInstanceError):
                cast(Any, binding).binding_id = "changed"

    def test_inline_config_is_json_checked_and_defensively_copied(self) -> None:
        source: dict[str, Any] = {"nested": {"values": [1, 2]}}
        binding = InlineModelBinding(
            binding_id="inline:primary",
            root_deployment_key="deployment:base",
            requested_model_id="org/model",
            config=source,
        )

        source["nested"]["values"].append(3)
        assert binding.config == {"nested": {"values": [1, 2]}}
        with pytest.raises(TypeError):
            cast(Any, binding.config)["other"] = True

    @pytest.mark.parametrize("bad", [{"x": object()}, {"x": float("nan")}])
    def test_inline_config_rejects_non_json_values(self, bad: dict[str, Any]) -> None:
        with pytest.raises((TypeError, ValueError)):
            InlineModelBinding(
                binding_id="inline:primary",
                root_deployment_key="deployment:base",
                requested_model_id="org/model",
                config=bad,
            )

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({1: "value"}, "mapping keys must be strings"),
            (["not", "a", "mapping"], "must be a mapping"),
        ],
    )
    def test_inline_config_rejects_invalid_mapping_shapes(
        self, config: object, message: str
    ) -> None:
        with pytest.raises(TypeError, match=message):
            InlineModelBinding(
                binding_id="inline:primary",
                root_deployment_key="deployment:base",
                requested_model_id="org/model",
                config=cast(Any, config),
            )

    @pytest.mark.parametrize(
        ("binding_id", "root_deployment_key", "requested_model_id", "message"),
        [
            ("", "root", "model", "binding_id"),
            ("id", "", "model", "root_deployment_key"),
            ("id", "root", "", "requested_model_id"),
        ],
    )
    def test_identity_fields_must_not_be_empty(
        self,
        binding_id: str,
        root_deployment_key: str,
        requested_model_id: str,
        message: str,
    ) -> None:
        with pytest.raises(TypeError, match=message):
            NamedModelBinding(
                binding_id,
                root_deployment_key,
                requested_model_id,
                "config",
            )

    def test_requirement_context_freezes_normalized_mappings(self) -> None:
        bindings = {"model": _binding()}
        task_args: dict[str, Any] = {"nested": {"value": 1}}
        context = RequirementContext(
            model_bindings=bindings,
            task_args=task_args,
            dataset_config={"split": "test"},
            infer_args={"max_tokens": 128},
        )

        bindings.clear()
        task_args["nested"]["value"] = 2
        assert context.model_bindings == {"model": _binding()}
        assert context.task_args == {"nested": {"value": 1}}
        with pytest.raises(TypeError):
            cast(Any, context.model_bindings)["other"] = _binding("other")

    def test_requirement_context_rejects_nonbinding_values(self) -> None:
        with pytest.raises(TypeError, match="NormalizedModelBinding"):
            RequirementContext(model_bindings={"model": cast(Any, object())})

    def test_requirement_record_validates_role_binding_and_source(self) -> None:
        with pytest.raises(TypeError, match="role"):
            TaskModelRequirement("", _binding(), TaskRequirements(), "task")
        with pytest.raises(TypeError, match="NormalizedModelBinding"):
            TaskModelRequirement(
                "model", cast(Any, object()), TaskRequirements(), "task"
            )
        with pytest.raises(TypeError, match="requires must be TaskRequirements"):
            TaskModelRequirement("model", _binding(), cast(Any, object()), "task")
        with pytest.raises(TypeError, match="source_task"):
            TaskModelRequirement("model", _binding(), TaskRequirements(), "")


class TestAggregation:
    def test_empty_input_has_the_union_identity(self) -> None:
        assert aggregate_task_requirements([]) == AggregatedTaskRequirements()

    def test_union_or_and_max_retain_each_source(self) -> None:
        records = [
            _requirement(
                "chat-image",
                TaskRequirements(
                    input=InputKind.CHAT,
                    input_modalities=frozenset(
                        {InputModality.TEXT, InputModality.IMAGE}
                    ),
                    sampled_logprobs=True,
                    min_top_logprobs=2,
                ),
            ),
            _requirement(
                "completion-score",
                TaskRequirements(
                    input=InputKind.COMPLETION,
                    input_scoring=True,
                    min_top_logprobs=8,
                ),
            ),
            _requirement(
                "chat-tools",
                TaskRequirements(
                    input=InputKind.CHAT,
                    input_modalities=frozenset(
                        {
                            InputModality.TEXT,
                            InputModality.TOOL_CALL,
                            InputModality.TOOL_RESULT,
                        }
                    ),
                ),
            ),
        ]

        result = aggregate_task_requirements(records)

        assert result.input == {InputKind.CHAT, InputKind.COMPLETION}
        assert result.input_modalities == {
            InputModality.TEXT,
            InputModality.IMAGE,
            InputModality.TOOL_CALL,
            InputModality.TOOL_RESULT,
        }
        assert result.input_scoring
        assert result.sampled_logprobs
        assert result.min_top_logprobs == 8
        assert result.input_sources == {
            InputKind.CHAT: {"chat-image", "chat-tools"},
            InputKind.COMPLETION: {"completion-score"},
        }
        assert result.modality_sources[InputModality.IMAGE] == {"chat-image"}
        assert result.modality_sources[InputModality.TEXT] == {
            "chat-image",
            "completion-score",
            "chat-tools",
        }
        assert result.input_scoring_sources == {"completion-score"}
        assert result.sampled_logprobs_sources == {
            "chat-image",
            "completion-score",
        }
        assert result.min_top_logprobs_sources == {
            2: {"chat-image"},
            8: {"completion-score"},
        }

    def test_result_is_order_independent(self) -> None:
        first = _requirement(
            "first", TaskRequirements(input=InputKind.CHAT, input_scoring=True)
        )
        second = _requirement(
            "second",
            TaskRequirements(input=InputKind.COMPLETION, min_top_logprobs=3),
        )

        assert aggregate_task_requirements([first, second]) == (
            aggregate_task_requirements([second, first])
        )

    def test_non_requirement_values_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="TaskModelRequirement"):
            aggregate_task_requirements(cast(Any, [TaskRequirements()]))

    def test_requirement_without_input_kind_still_contributes_modalities(self) -> None:
        result = aggregate_task_requirements(
            [_requirement("implicit-text", TaskRequirements())]
        )

        assert result.input == frozenset()
        assert result.modality_sources == {
            InputModality.TEXT: frozenset({"implicit-text"})
        }


class TestAggregatedRequirementValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"input": frozenset({"chat"})}, "InputKind"),
            ({"input_modalities": frozenset({"text"})}, "InputModality"),
            ({"input_scoring": 1}, "input_scoring must be a boolean"),
            ({"sampled_logprobs": 1}, "sampled_logprobs must be a boolean"),
            ({"min_top_logprobs": True}, "must be an integer"),
            ({"min_top_logprobs": 0}, ">= 1"),
            (
                {"min_top_logprobs": 1, "sampled_logprobs": False},
                "requires sampled_logprobs",
            ),
        ],
    )
    def test_invalid_aggregate_values_are_rejected(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            AggregatedTaskRequirements(**cast(Any, kwargs))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"input_sources": {"chat": frozenset({"task"})}}, "InputKind"),
            (
                {"modality_sources": {"text": frozenset({"task"})}},
                "InputModality",
            ),
            (
                {"min_top_logprobs_sources": {True: frozenset({"task"})}},
                "keys must be integers",
            ),
            (
                {"min_top_logprobs_sources": {0: frozenset({"task"})}},
                "keys must be >= 1",
            ),
        ],
    )
    def test_invalid_aggregate_source_indexes_are_rejected(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            AggregatedTaskRequirements(**cast(Any, kwargs))
