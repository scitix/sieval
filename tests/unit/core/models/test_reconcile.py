"""Tests for pure capability and runtime binding reconciliation.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from typing import Any, cast

import pytest

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    CapabilityIntent,
    CapabilityKey,
    HostedToolsOptions,
    ModelCapabilityEntry,
    ModelCapabilityProfile,
    ModelCapabilityStatus,
    RequestDefaults,
)
from sieval.core.models.deployment import (
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
    DeploymentPlanProjection,
    Engine,
    RouteIntent,
    ServingFacts,
)
from sieval.core.models.dialect_registry import DialectRegistryError, bind_dialect
from sieval.core.models.dialects.openai_chat import OpenAIChatDialect
from sieval.core.models.dialects.openai_completions import OpenAICompletionsDialect
from sieval.core.models.dialects.openai_responses import OpenAIResponsesDialect
from sieval.core.models.reconcile import (
    BindingCapabilityPlan,
    BindingReconcileInput,
    CannotVerify,
    CheckStage,
    Configured,
    ConnectionScope,
    DeferredCheck,
    DeploymentCapabilityPlan,
    DeploymentReconcileInput,
    ReconcileBatch,
    ReconcileDiagnostic,
    ReconcileSeverity,
    RuntimeBindingPlan,
    ServingOutcome,
    ServingRequirement,
    ServingUnsupported,
    _declaration_intent,
    _json_sequence,
    reconcile,
)
from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    InputKind,
    InputModality,
)
from sieval.core.types import JSONValue


def _serialized_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class _Connection:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


def _profile(
    **overrides: ModelCapabilityEntry,
) -> ModelCapabilityProfile:
    entries: dict[str, ModelCapabilityEntry] = {
        key: ModelCapabilityEntry(ModelCapabilityStatus.SUPPORTED, "test-profile")
        for key in CAPABILITY_KEYS
    }
    entries.update(overrides)
    return ModelCapabilityProfile(
        cast(Mapping[CapabilityKey, ModelCapabilityEntry], entries),
        authoritative=True,
    )


def _requirements(
    *,
    kind: InputKind = InputKind.CHAT,
    modalities: frozenset[InputModality] = frozenset({InputModality.TEXT}),
    input_scoring: bool = False,
    sampled_logprobs: bool = False,
    minimum: int | None = None,
) -> AggregatedTaskRequirements:
    return AggregatedTaskRequirements(
        input=frozenset({kind}),
        input_modalities=modalities,
        input_scoring=input_scoring,
        sampled_logprobs=sampled_logprobs or minimum is not None,
        min_top_logprobs=minimum,
        input_sources={kind: frozenset({"task-a"})},
        modality_sources={modality: frozenset({"task-a"}) for modality in modalities},
        input_scoring_sources=(frozenset({"task-a"}) if input_scoring else frozenset()),
        sampled_logprobs_sources=(
            frozenset({"task-a"})
            if sampled_logprobs or minimum is not None
            else frozenset()
        ),
        min_top_logprobs_sources=(
            {minimum: frozenset({"task-a"})} if minimum is not None else {}
        ),
    )


def _binding(
    *,
    binding_id: str = "binding-a",
    dialect_id: str = "openai_chat",
    requirements: AggregatedTaskRequirements | None = None,
    profile: ModelCapabilityProfile | None = None,
    declarations: Mapping[str, JSONValue] | None = None,
    request_intents: Mapping[CapabilityKey, CapabilityIntent] | None = None,
) -> BindingReconcileInput:
    if requirements is None:
        requirements = _requirements(
            kind=(
                InputKind.COMPLETION
                if dialect_id == "openai_completions"
                else InputKind.CHAT
            )
        )
    return BindingReconcileInput(
        binding_id=binding_id,
        root_deployment_key="root",
        requested_model_id="requested-model",
        dialect_id=dialect_id,
        requirements=requirements,
        model_profile=profile or _profile(),
        connection_scope=ConnectionScope("team-a", "standard", "root"),
        declarations={} if declarations is None else declarations,
        request_intents={} if request_intents is None else request_intents,
    )


def _deployment(*, realized: bool = True) -> DeploymentReconcileInput:
    plan = DeploymentPlanProjection("sha256:desired", "vllm")
    deployment = (
        Deployment(
            deployment_id="deployment-a",
            plan=plan,
            engine=Engine("vllm"),
            engine_source="deployment",
            api_base="https://models.example/v1",
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(engine_version="0.9.0"),
        )
        if realized
        else None
    )
    return DeploymentReconcileInput(
        root_deployment_key="root",
        engine_id="vllm",
        deployment=deployment,
        plan=None if realized else plan,
    )


def _batch(
    *bindings: BindingReconcileInput,
    deployment: DeploymentReconcileInput | None = None,
) -> ReconcileBatch:
    return ReconcileBatch(
        tuple(bindings),
        {"root": deployment or _deployment()},
    )


class _ConfiguredReconciler:
    def __init__(
        self,
        patch_value: JSONValue = None,
        *,
        evidence_value: str = "passed",
    ) -> None:
        self.calls: list[tuple[ServingRequirement, ...]] = []
        self.patch_value = patch_value
        self.evidence_value = evidence_value

    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        assert deployment.root_deployment_key == "root"
        self.calls.append(requirements)
        return {
            item.capability: Configured(
                launch_patch=(
                    {}
                    if self.patch_value is None
                    else {"verified-capability": self.patch_value}
                ),
                evidence={"probe": self.evidence_value},
            )
            for item in requirements
        }


class _CannotVerifyAtRequest:
    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        del deployment
        return {
            item.capability: CannotVerify(
                CheckStage.REQUEST,
                "validate_response_channel",
                "must inspect the concrete response",
            )
            for item in requirements
        }


class _UnsupportedReconciler:
    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        del deployment
        return {
            item.capability: ServingUnsupported(
                "deployment cannot supply the feature", "use another deployment"
            )
            for item in requirements
        }


class _SharedPatchReconciler:
    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        del deployment
        return {
            item.capability: Configured(
                launch_patch={"shared-correctness-knob": item.capability}
            )
            for item in requirements
        }


class _NoPatchReconciler:
    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        del deployment
        return {item.capability: Configured() for item in requirements}


def _unknown_profile(capability: CapabilityKey) -> ModelCapabilityProfile:
    return _profile(
        **{
            capability: ModelCapabilityEntry(
                ModelCapabilityStatus.UNKNOWN,
                "hosted-catalog",
                reason="endpoint support is not authoritative",
                verifier="probe_hosted_capability",
            )
        }
    )


class TestReconcileValueValidation:
    def test_diagnostic_validates_identity_and_serializes_all_context(self) -> None:
        with pytest.raises(TypeError, match="ReconcileSeverity"):
            ReconcileDiagnostic(cast(Any, "error"), "code", "message")
        with pytest.raises(ValueError, match="unknown diagnostic capability"):
            ReconcileDiagnostic(
                ReconcileSeverity.ERROR,
                "code",
                "message",
                capability=cast(Any, "typo"),
            )

        diagnostic = ReconcileDiagnostic(
            ReconcileSeverity.WARNING,
            "warning",
            "message",
            binding_id="binding",
            root_deployment_key="root",
            capability="reasoning",
            sources=("task-b", "task-a", "task-a"),
        )
        assert diagnostic.to_json_value() == {
            "severity": "warning",
            "code": "warning",
            "message": "message",
            "binding_id": "binding",
            "root_deployment_key": "root",
            "capability": "reasoning",
            "sources": ["task-a", "task-b"],
        }
        assert ReconcileDiagnostic(
            ReconcileSeverity.INFO, "info", "message"
        ).to_json_value() == {
            "severity": "info",
            "code": "info",
            "message": "message",
            "sources": [],
        }

    def test_deferred_check_and_serving_requirement_validate_symbols(self) -> None:
        with pytest.raises(ValueError, match="unknown deferred capability"):
            DeferredCheck(cast(Any, "typo"), CheckStage.SETUP, "probe", "reason")
        with pytest.raises(TypeError, match="stage must be a CheckStage"):
            DeferredCheck("reasoning", cast(Any, "setup"), "probe", "reason")
        with pytest.raises(ValueError, match="unknown serving capability"):
            ServingRequirement(cast(Any, "typo"))

        requirement = ServingRequirement(
            "reasoning",
            minimums={"levels": ["high"]},
            sources=("task",),
            verifier="probe",
            reason="must verify",
        )
        assert requirement.to_json_value() == {
            "capability": "reasoning",
            "minimums": {"levels": ["high"]},
            "sources": ["task"],
            "verifier": "probe",
            "reason": "must verify",
        }
        assert ServingRequirement("reasoning").to_json_value() == {
            "capability": "reasoning",
            "minimums": {},
            "sources": [],
        }

    def test_serving_outcome_records_reject_invalid_stages_and_strings(self) -> None:
        setup_check = DeferredCheck(
            "reasoning", CheckStage.SETUP, "probe", "setup-only"
        )
        with pytest.raises(ValueError, match="must use request stage"):
            Configured(request_checks=(setup_check,))
        with pytest.raises(TypeError, match="stage must be a CheckStage"):
            CannotVerify(cast(Any, "request"), "probe", "reason")
        with pytest.raises(TypeError, match="cannot-verify verifier"):
            CannotVerify(CheckStage.REQUEST, "", "reason")
        with pytest.raises(TypeError, match="unsupported remedy"):
            ServingUnsupported("unsupported", "")
        assert ServingUnsupported("unsupported").remedy is None

    @pytest.mark.parametrize(
        "field", ["credential_scope", "retry_policy", "quota_scope"]
    )
    def test_connection_scope_values_must_not_be_empty(self, field: str) -> None:
        values = {
            "credential_scope": "credential",
            "retry_policy": "retry",
            "quota_scope": "quota",
        }
        values[field] = ""
        with pytest.raises(TypeError, match=field):
            ConnectionScope(**values)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"requirements": object()}, "AggregatedTaskRequirements"),
            ({"model_profile": object()}, "ModelCapabilityProfile"),
            ({"connection_scope": object()}, "ConnectionScope"),
            ({"route_intent": object()}, "RouteIntent"),
        ],
    )
    def test_binding_input_rejects_wrong_record_types(
        self, changes: dict[str, object], message: str
    ) -> None:
        with pytest.raises(TypeError, match=message):
            dataclasses.replace(_binding(), **changes)

    @pytest.mark.parametrize(
        ("intents", "message"),
        [
            (
                {"typo": CapabilityIntent("reasoning", True)},
                "unknown request intent capability",
            ),
            ({"reasoning": object()}, "must be CapabilityIntent"),
            (
                {"reasoning": CapabilityIntent("top_logprobs", True)},
                "does not match",
            ),
        ],
    )
    def test_binding_input_validates_request_intent_index(
        self, intents: dict[str, object], message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            dataclasses.replace(_binding(), request_intents=cast(Any, intents))

    @pytest.mark.parametrize(
        ("declarations", "message"),
        [
            ({"bad": float("nan")}, "non-finite"),
            ({"bad": {1: "value"}}, "keys must be strings"),
            ({"bad": object()}, "non-JSON"),
        ],
    )
    def test_binding_declarations_must_be_json_safe(
        self, declarations: object, message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            dataclasses.replace(_binding(), declarations=cast(Any, declarations))

    def test_binding_declarations_accept_finite_float_json(self) -> None:
        binding = dataclasses.replace(_binding(), declarations={"finite": 1.5})
        assert binding.declarations == {"finite": 1.5}

    def test_deployment_input_validates_realized_and_prelaunch_identity(self) -> None:
        realized = _deployment()
        with pytest.raises(ValueError, match="deployment engine"):
            dataclasses.replace(realized, engine_id="sglang")
        with pytest.raises(ValueError, match="desired plan"):
            dataclasses.replace(
                realized,
                plan=DeploymentPlanProjection("sha256:other", "vllm"),
            )

        prelaunch_result = reconcile(
            _batch(_binding(), deployment=_deployment(realized=False))
        )
        prelaunch = prelaunch_result.deployment_plans["root"]
        with pytest.raises(ValueError, match="only valid for a realized"):
            dataclasses.replace(_deployment(realized=False), prelaunch_plan=prelaunch)
        with pytest.raises(ValueError, match="another root"):
            dataclasses.replace(
                realized,
                prelaunch_plan=dataclasses.replace(
                    prelaunch, root_deployment_key="other"
                ),
            )
        with pytest.raises(ValueError, match="another engine"):
            dataclasses.replace(
                realized,
                prelaunch_plan=dataclasses.replace(prelaunch, engine_id="sglang"),
            )

        unknown_prelaunch = dataclasses.replace(prelaunch, engine_id="unknown")
        refined = dataclasses.replace(realized, prelaunch_plan=unknown_prelaunch)
        assert refined.engine_id == "vllm"

        postlaunch = dataclasses.replace(realized, prelaunch_plan=prelaunch)
        serialized = postlaunch.to_json_value()
        assert "deployment_fingerprint" in serialized
        assert serialized["prelaunch_plan_fingerprint"] == prelaunch.fingerprint
        assert _deployment(realized=False).to_json_value()["plan"] is not None

    def test_prelaunch_plan_engine_must_match_reconcile_engine(self) -> None:
        with pytest.raises(ValueError, match="desired plan engine"):
            DeploymentReconcileInput(
                root_deployment_key="root",
                engine_id="sglang",
                plan=DeploymentPlanProjection("sha256:vllm-plan", "vllm"),
            )

    def test_reconcile_batch_rejects_duplicate_or_misindexed_inputs(self) -> None:
        binding = _binding()
        with pytest.raises(ValueError, match="binding ids must be unique"):
            ReconcileBatch((binding, binding), {"root": _deployment()})
        with pytest.raises(ValueError, match="mapping key does not match"):
            ReconcileBatch((binding,), {"other": _deployment()})

    def test_minimum_sequence_merge_rejects_nested_or_mapping_values(self) -> None:
        with pytest.raises(TypeError, match="must be scalar"):
            _json_sequence(cast(Any, [["nested"]]))
        with pytest.raises(TypeError, match="incompatible mappings"):
            _json_sequence(cast(Any, {"nested": True}))
        assert _json_sequence(cast(Any, ["one", 2])) == ("one", 2)
        assert _json_sequence(cast(Any, "single")) == ("single",)

    def test_hosted_tool_declaration_projects_its_minimum_domain(self) -> None:
        intent = _declaration_intent(
            "hosted_tools",
            HostedToolsOptions(("web_search",)),
            "model.capabilities",
        )
        assert intent.minimums == {"kinds": ["web_search"]}


class TestBindingJoin:
    @pytest.mark.parametrize("dialect_id", ["unknown", "vllm_native"])
    def test_unavailable_dialect_becomes_a_binding_diagnostic(
        self, dialect_id: str
    ) -> None:
        result = reconcile(_batch(_binding(dialect_id=dialect_id)))

        assert {item.code for item in result.errors} == {"dialect_unavailable"}
        assert not result.binding_plans

    def test_optional_intent_is_not_promoted_to_required(self) -> None:
        optional = CapabilityIntent(
            "reasoning",
            False,
            sources=("optional-caller",),
        )
        binding = _binding(request_intents={"reasoning": optional})

        result = reconcile(_batch(binding))

        assert result.is_valid
        plan = result.binding_plans["binding-a"]
        assert plan.intents["reasoning"].required is False
        assert "reasoning" not in plan.required_capabilities

    def test_explicit_false_does_not_conflict_with_optional_intent(self) -> None:
        binding = _binding(
            declarations={"reasoning": False},
            request_intents={
                "reasoning": CapabilityIntent(
                    "reasoning",
                    False,
                    sources=("optional-caller",),
                )
            },
        )

        result = reconcile(_batch(binding))

        assert result.is_valid
        assert not any(
            item.code == "task_capability_disabled" for item in result.diagnostics
        )
        assert (
            "reasoning" not in result.binding_plans["binding-a"].required_capabilities
        )

    def test_dynamic_request_intent_is_rejected_by_incompatible_dialect(self) -> None:
        binding = _binding(
            dialect_id="openai_completions",
            request_intents={
                "reasoning": CapabilityIntent(
                    "reasoning",
                    True,
                    sources=("tasks.eval.infer_args.reasoning_effort",),
                )
            },
        )

        result = reconcile(_batch(binding))

        diagnostic = next(
            item for item in result.errors if item.code == "dialect_unsupported"
        )
        assert diagnostic.capability == "reasoning"
        assert diagnostic.sources == ("tasks.eval.infer_args.reasoning_effort",)

    def test_dynamic_request_minimum_cannot_weaken_task_requirement(self) -> None:
        binding = _binding(
            dialect_id="openai_completions",
            requirements=_requirements(
                kind=InputKind.COMPLETION,
                sampled_logprobs=True,
                minimum=8,
            ),
            request_intents={
                "top_logprobs": CapabilityIntent(
                    "top_logprobs",
                    True,
                    minimums={"minimum": 4},
                    sources=("tasks.eval.infer_args.top_logprobs",),
                )
            },
        )

        result = reconcile(_batch(binding))

        assert {item.code for item in result.errors} == {
            "request_capability_minimum_weakened"
        }

    def test_task_disable_conflict_preserves_source(self) -> None:
        binding = _binding(
            dialect_id="openai_completions",
            requirements=_requirements(
                kind=InputKind.COMPLETION,
                input_scoring=True,
            ),
            declarations={"input_scoring": False},
        )

        result = reconcile(_batch(binding))

        diagnostic = next(
            item
            for item in result.diagnostics
            if item.code == "task_capability_disabled"
        )
        assert diagnostic.sources == ("task-a",)
        assert not result.binding_plans
        assert not result.runtime_plans

    def test_explicit_minimum_may_not_weaken_task(self) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True, minimum=8),
            declarations={"top_logprobs": {"minimum": 4}},
        )

        result = reconcile(_batch(binding))

        assert {item.code for item in result.errors} == {"capability_minimum_weakened"}

    def test_empty_declared_domain_does_not_erase_task_minimum(self) -> None:
        binding = _binding(
            requirements=_requirements(
                modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})
            ),
            declarations={"multimodal_input": True},
        )

        result = reconcile(_batch(binding), _NoPatchReconciler())

        assert result.is_valid
        assert result.binding_plans["binding-a"].capability_minimums == {
            "multimodal_input": {"modalities": ["image"]}
        }

    def test_input_kind_mismatch_fails_before_runtime_plan(self) -> None:
        binding = _binding(
            dialect_id="openai_chat",
            requirements=_requirements(kind=InputKind.COMPLETION),
        )

        result = reconcile(_batch(binding))

        assert any(item.code == "input_kind_unsupported" for item in result.errors)
        assert not result.runtime_plans

    def test_dialect_option_domain_failure_becomes_diagnostic(self) -> None:
        binding = _binding(
            declarations={"reasoning": {"budget_tokens": 1024}},
        )

        result = reconcile(_batch(binding))

        assert any(
            item.code == "dialect_config_invalid" and "budget_tokens" in item.message
            for item in result.errors
        )
        assert not result.runtime_plans

    def test_dialect_reasoning_effort_domain_failure_becomes_diagnostic(self) -> None:
        binding = _binding(
            declarations={"reasoning": {"effort": "banana"}},
        )

        result = reconcile(_batch(binding), _ConfiguredReconciler())

        assert any(
            item.code == "dialect_config_invalid" and "effort 'banana'" in item.message
            for item in result.errors
        )
        assert not result.runtime_plans

    def test_dialect_and_model_absence_are_distinct(self) -> None:
        dialect_result = reconcile(
            _batch(
                _binding(
                    requirements=_requirements(input_scoring=True),
                )
            )
        )
        model_result = reconcile(
            _batch(
                _binding(
                    dialect_id="openai_completions",
                    requirements=_requirements(
                        kind=InputKind.COMPLETION,
                        input_scoring=True,
                    ),
                    profile=_profile(
                        input_scoring=ModelCapabilityEntry(
                            ModelCapabilityStatus.UNSUPPORTED,
                            "test-profile",
                            reason="checkpoint cannot return prompt scores",
                        )
                    ),
                )
            )
        )

        assert any(item.code == "dialect_unsupported" for item in dialect_result.errors)
        assert any(item.code == "model_unsupported" for item in model_result.errors)

    def test_missing_required_model_outcome_is_not_treated_as_support(self) -> None:
        entries = dict(_profile().entries)
        del entries["sampled_logprobs"]
        profile = ModelCapabilityProfile(entries, authoritative=True)
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
            profile=profile,
        )

        result = reconcile(_batch(binding))

        assert any(item.code == "model_outcome_missing" for item in result.errors)

    def test_unrequired_missing_or_unsupported_model_outcomes_are_inert(self) -> None:
        entries = dict(_profile().entries)
        del entries["reasoning"]
        missing = reconcile(
            _batch(
                _binding(profile=ModelCapabilityProfile(entries, authoritative=True))
            )
        )
        unsupported = reconcile(
            _batch(
                _binding(
                    profile=_profile(
                        reasoning=ModelCapabilityEntry(
                            ModelCapabilityStatus.UNSUPPORTED,
                            "catalog",
                            reason="not supported",
                        )
                    )
                )
            )
        )

        assert missing.is_valid
        assert unsupported.is_valid

    def test_two_capabilities_cannot_own_the_same_request_default(self) -> None:
        shared = RequestDefaults({"shared.default": True})
        binding = _binding(
            request_intents={
                "reasoning": CapabilityIntent(
                    "reasoning", True, request_defaults=shared, sources=("reason",)
                ),
                "function_tools": CapabilityIntent(
                    "function_tools",
                    True,
                    request_defaults=shared,
                    sources=("tools",),
                ),
            }
        )

        result = reconcile(_batch(binding))

        assert any(
            item.code == "duplicate_request_default_owner" for item in result.errors
        )

    def test_declared_list_domain_cannot_omit_request_requirement(self) -> None:
        binding = _binding(
            declarations={
                "structured_output": {"formats": ["json_object"]},
            },
            request_intents={
                "structured_output": CapabilityIntent(
                    "structured_output",
                    True,
                    minimums={"formats": ["json_schema"]},
                    sources=("request",),
                )
            },
        )

        result = reconcile(_batch(binding))

        assert any(item.code == "capability_minimum_weakened" for item in result.errors)

    def test_declared_list_superset_satisfies_request_requirement(self) -> None:
        binding = _binding(
            declarations={
                "structured_output": {"formats": ["json_object", "json_schema"]},
            },
            request_intents={
                "structured_output": CapabilityIntent(
                    "structured_output",
                    True,
                    minimums={"formats": ["json_schema"]},
                    sources=("request",),
                )
            },
        )

        result = reconcile(_batch(binding), _NoPatchReconciler())

        assert result.is_valid
        assert result.binding_plans["binding-a"].capability_minimums[
            "structured_output"
        ] == {"formats": ["json_object", "json_schema"]}

    def test_unsupported_input_modality_preserves_shape_diagnostic(self) -> None:
        binding = _binding(
            dialect_id="openai_completions",
            requirements=_requirements(
                kind=InputKind.COMPLETION,
                modalities=frozenset({InputModality.TEXT, InputModality.TOOL_CALL}),
            ),
        )

        result = reconcile(_batch(binding))

        assert any(item.code == "input_modality_unsupported" for item in result.errors)

    def test_declaration_domains_project_defaults_and_minimums(self) -> None:
        binding = _binding(
            declarations={
                "function_tools": {"parallel": False},
                "structured_output": {"formats": ["json_schema"]},
                "multimodal_input": {"modalities": ["image"]},
            }
        )

        result = reconcile(_batch(binding), _NoPatchReconciler())

        assert result.is_valid
        plan = result.binding_plans["binding-a"]
        assert plan.request_defaults.values == {"tools.parallel": False}
        assert plan.capability_minimums["structured_output"] == {
            "formats": ["json_schema"]
        }
        assert plan.capability_minimums["multimodal_input"] == {"modalities": ["image"]}

    def test_one_binding_error_prevents_every_runtime_binding(self) -> None:
        valid = _binding(binding_id="valid")
        invalid = _binding(
            binding_id="invalid",
            declarations={"input_scoring": True},
        )

        result = reconcile(_batch(valid, invalid))

        assert result.errors
        assert not result.runtime_plans


class TestServingJoin:
    def test_supported_capability_still_requires_a_serving_outcome(self) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
        )

        result = reconcile(_batch(binding))

        plan = result.binding_plans["binding-a"]
        assert plan.pending_capabilities == {"sampled_logprobs"}
        assert "sampled_logprobs" not in plan.available_capabilities
        assert any(item.code == "cannot_verify" for item in result.errors)
        assert not result.runtime_plans

    def test_supported_capability_becomes_available_after_serving_outcome(
        self,
    ) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
        )

        result = reconcile(_batch(binding), _ConfiguredReconciler())

        assert result.is_valid
        runtime = result.runtime_plans["binding-a"]
        assert "sampled_logprobs" in runtime.available_capabilities

    def test_unknown_required_capability_without_reconciler_is_cannot_verify(
        self,
    ) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )

        result = reconcile(_batch(binding))

        assert result.binding_plans["binding-a"].pending_capabilities == {
            "sampled_logprobs"
        }
        deployment_plan = result.deployment_plans["root"]
        assert deployment_plan.outcome_kinds == {"sampled_logprobs": "cannot_verify"}
        assert deployment_plan.setup_checks[0].verifier == "probe_hosted_capability"
        assert any(item.code == "cannot_verify" for item in result.errors)
        assert not result.runtime_plans

    def test_managed_prelaunch_carries_setup_check_without_runtime_plan(self) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )

        result = reconcile(_batch(binding, deployment=_deployment(realized=False)))

        assert result.is_valid
        deployment_plan = result.deployment_plans["root"]
        assert deployment_plan.setup_checks[0].verifier == "probe_hosted_capability"
        assert any(
            item.code == "cannot_verify" and item.severity is ReconcileSeverity.WARNING
            for item in result.diagnostics
        )
        assert not result.runtime_plans

    def test_configured_outcome_activates_pending_capability_once_per_root(
        self,
    ) -> None:
        first = _binding(
            binding_id="binding-a",
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )
        second = dataclasses.replace(first, binding_id="binding-b")
        reconciler = _ConfiguredReconciler()

        result = reconcile(_batch(second, first), reconciler)

        assert result.is_valid
        assert len(reconciler.calls) == 1
        assert [item.capability for item in reconciler.calls[0]] == ["sampled_logprobs"]
        assert set(result.runtime_plans) == {"binding-a", "binding-b"}
        assert all(
            "sampled_logprobs" in plan.available_capabilities
            for plan in result.runtime_plans.values()
        )
        assert result.deployment_plans["root"].outcome_evidence == {
            "sampled_logprobs": {"probe": "passed"}
        }

    def test_serving_evidence_changes_verification_fingerprint(self) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )

        first = reconcile(
            _batch(binding),
            _ConfiguredReconciler(evidence_value="first"),
        )
        second = reconcile(
            _batch(binding),
            _ConfiguredReconciler(evidence_value="second"),
        )

        assert (
            first.runtime_plans["binding-a"].verification_fingerprint
            != second.runtime_plans["binding-a"].verification_fingerprint
        )

    def test_request_safe_cannot_verify_becomes_runtime_postcondition(self) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )

        result = reconcile(_batch(binding), _CannotVerifyAtRequest())

        assert result.is_valid
        assert any(
            item.code == "cannot_verify" and item.severity is ReconcileSeverity.WARNING
            for item in result.diagnostics
        )
        runtime = result.runtime_plans["binding-a"]
        assert runtime.request_checks[0].verifier == "validate_response_channel"
        assert "sampled_logprobs" in runtime.available_capabilities

    def test_serving_unsupported_blocks_runtime_plan_with_remedy(self) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )

        result = reconcile(_batch(binding), _UnsupportedReconciler())

        assert any(
            item.code == "serving_unsupported"
            and "use another deployment" in item.message
            for item in result.errors
        )
        assert not result.runtime_plans

    def test_prelaunch_plan_never_claims_runtime_binding(self) -> None:
        result = reconcile(_batch(_binding(), deployment=_deployment(realized=False)))

        assert result.is_valid
        assert result.binding_plans
        assert result.deployment_plans
        assert not result.runtime_plans

    def test_managed_prelaunch_may_derive_a_launch_patch(self) -> None:
        binding = _binding(requirements=_requirements(sampled_logprobs=True))

        result = reconcile(
            _batch(binding, deployment=_deployment(realized=False)),
            _ConfiguredReconciler("sampled_logprobs"),
        )

        assert result.is_valid
        assert result.deployment_plans["root"].launch_patch == {
            "verified-capability": "sampled_logprobs"
        }
        assert not result.runtime_plans

    def test_realized_deployment_rejects_patch_without_frozen_prelaunch(
        self,
    ) -> None:
        binding = _binding(requirements=_requirements(sampled_logprobs=True))

        result = reconcile(
            _batch(binding),
            _ConfiguredReconciler("sampled_logprobs"),
        )

        assert any(
            item.code == "unfrozen_realized_launch_patch" for item in result.errors
        )
        assert not result.runtime_plans

    def test_external_deployment_rejects_launch_patch(self) -> None:
        external = Deployment(
            deployment_id=None,
            plan=None,
            engine=Engine("vllm"),
            engine_source="config",
            api_base="https://external.example/v1",
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(),
        )
        deployment_input = DeploymentReconcileInput(
            root_deployment_key="root",
            engine_id="vllm",
            deployment=external,
        )
        binding = _binding(requirements=_requirements(sampled_logprobs=True))

        result = reconcile(
            _batch(binding, deployment=deployment_input),
            _ConfiguredReconciler("sampled_logprobs"),
        )

        assert any(item.code == "launch_patch_unavailable" for item in result.errors)
        assert not result.runtime_plans

    def test_missing_deployment_input_is_reported(self) -> None:
        result = reconcile(ReconcileBatch((_binding(),), {}))

        assert any(item.code == "deployment_input_missing" for item in result.errors)
        assert not result.runtime_plans

    def test_explicit_engine_parameter_conflict_is_not_overwritten(self) -> None:
        deployment = dataclasses.replace(
            _deployment(),
            explicit_parameters={"verified-capability": "explicit"},
        )
        binding = _binding(requirements=_requirements(sampled_logprobs=True))

        result = reconcile(
            _batch(binding, deployment=deployment),
            _ConfiguredReconciler("derived"),
        )

        assert any(
            item.code == "explicit_engine_parameter_conflict"
            and "set 'verified-capability' to \"derived\"" in item.message
            for item in result.errors
        )
        assert result.deployment_plans["root"].launch_patch == {}

    def test_serving_minimums_do_not_treat_boolean_as_integer(self) -> None:
        def binding(binding_id: str, minimum: bool | int) -> BindingReconcileInput:
            return _binding(
                binding_id=binding_id,
                requirements=_requirements(sampled_logprobs=True),
                profile=_unknown_profile("sampled_logprobs"),
                request_intents={
                    "sampled_logprobs": CapabilityIntent(
                        "sampled_logprobs",
                        True,
                        minimums={"custom": minimum},
                        sources=(f"{binding_id}.request",),
                    )
                },
            )

        reconciler = _ConfiguredReconciler()
        result = reconcile(
            _batch(binding("binding-a", True), binding("binding-b", 1)),
            reconciler,
        )

        assert result.is_valid
        assert reconciler.calls[0][0].minimums["custom"] == [1, True]

    def test_explicit_boolean_parameter_does_not_equal_safe_integer(self) -> None:
        deployment = dataclasses.replace(
            _deployment(),
            explicit_parameters={"verified-capability": True},
        )
        binding = _binding(requirements=_requirements(sampled_logprobs=True))

        result = reconcile(
            _batch(binding, deployment=deployment),
            _ConfiguredReconciler(1),
        )

        assert any(
            item.code == "explicit_engine_parameter_conflict"
            and "set 'verified-capability' to 1" in item.message
            for item in result.errors
        )

    def test_matching_explicit_engine_parameter_satisfies_patch(self) -> None:
        deployment = dataclasses.replace(
            _deployment(),
            explicit_parameters={"verified-capability": "sampled_logprobs"},
        )
        binding = _binding(requirements=_requirements(sampled_logprobs=True))

        result = reconcile(
            _batch(binding, deployment=deployment),
            _ConfiguredReconciler("sampled_logprobs"),
        )

        assert result.is_valid
        assert result.deployment_plans["root"].launch_patch == {}

    def test_capabilities_cannot_derive_contradictory_launch_values(self) -> None:
        binding = _binding(requirements=_requirements(sampled_logprobs=True, minimum=2))

        result = reconcile(_batch(binding), _SharedPatchReconciler())

        assert any(item.code == "contradictory_launch_patch" for item in result.errors)

    def test_conflicting_named_verifiers_fail_instead_of_fabricating_one(
        self,
    ) -> None:
        first = _binding(
            binding_id="binding-a",
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )
        second = dataclasses.replace(
            first,
            binding_id="binding-b",
            model_profile=_profile(
                sampled_logprobs=ModelCapabilityEntry(
                    ModelCapabilityStatus.UNKNOWN,
                    "another-catalog",
                    reason="another verifier owns this claim",
                    verifier="another_named_verifier",
                )
            ),
        )

        result = reconcile(_batch(first, second), _ConfiguredReconciler())

        assert any(
            item.code == "serving_verifier_conflict"
            and "another_named_verifier" in item.message
            and "probe_hosted_capability" in item.message
            for item in result.errors
        )
        assert not result.runtime_plans

    def test_prelaunch_rejects_ambiguous_multi_role_route(self) -> None:
        desired = DeploymentReconcileInput(
            root_deployment_key="root",
            engine_id="vllm",
            plan=DeploymentPlanProjection(
                "sha256:desired",
                "vllm",
                ("decode", "prefill"),
            ),
        )

        result = reconcile(_batch(_binding(), deployment=desired))

        assert any(
            item.code == "route_role_ambiguous_prelaunch" for item in result.errors
        )
        assert not result.runtime_plans

    def test_prelaunch_rejects_route_absent_from_desired_plan(self) -> None:
        desired = DeploymentReconcileInput(
            root_deployment_key="root",
            engine_id="vllm",
            plan=DeploymentPlanProjection(
                "sha256:desired",
                "vllm",
                ("decode", "prefill"),
            ),
        )
        binding = dataclasses.replace(
            _binding(),
            route_intent=RouteIntent("full"),
        )

        result = reconcile(_batch(binding, deployment=desired))

        assert any(
            item.code == "route_role_missing_from_plan" for item in result.errors
        )

    def test_postlaunch_reconcile_rejects_launch_patch_drift(self) -> None:
        binding = _binding(
            requirements=_requirements(sampled_logprobs=True),
            profile=_unknown_profile("sampled_logprobs"),
        )
        prelaunch = reconcile(
            _batch(binding, deployment=_deployment(realized=False)),
            _ConfiguredReconciler("before"),
        )
        realized = dataclasses.replace(
            _deployment(),
            prelaunch_plan=prelaunch.deployment_plans["root"],
        )

        result = reconcile(
            _batch(binding, deployment=realized),
            _ConfiguredReconciler("after"),
        )

        assert any(item.code == "postlaunch_patch_drift" for item in result.errors)
        assert not result.runtime_plans

    def test_postlaunch_reconcile_rejects_realized_plan_identity_drift(self) -> None:
        binding = _binding()
        prelaunch = reconcile(_batch(binding, deployment=_deployment(realized=False)))
        other_plan = DeploymentPlanProjection("sha256:other-desired-plan", "vllm")
        realized_deployment = Deployment(
            deployment_id="deployment-other",
            plan=other_plan,
            engine=Engine("vllm"),
            engine_source="deployment",
            api_base="https://models.example/v1",
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(),
        )
        postlaunch = DeploymentReconcileInput(
            root_deployment_key="root",
            engine_id="vllm",
            deployment=realized_deployment,
            prelaunch_plan=prelaunch.deployment_plans["root"],
        )

        result = reconcile(_batch(binding, deployment=postlaunch))

        assert any(item.code == "postlaunch_plan_drift" for item in result.errors)
        assert not result.runtime_plans


class TestRuntimePlanAndBinding:
    def test_external_deployment_without_plan_or_id_reconciles_and_binds(self) -> None:
        deployment = Deployment(
            deployment_id=None,
            plan=None,
            engine=Engine("vllm"),
            engine_source="config",
            api_base="https://external.example/v1",
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(),
        )
        deployment_input = DeploymentReconcileInput(
            root_deployment_key="root",
            engine_id="vllm",
            deployment=deployment,
        )

        result = reconcile(_batch(_binding(), deployment=deployment_input))

        assert result.is_valid
        runtime = result.runtime_plans["binding-a"]
        pool = ConnectionPool(_Connection(), runtime.connection_identity)
        dialect = bind_dialect(
            "openai_chat", "requested-model", deployment, pool, runtime
        )
        assert isinstance(dialect, OpenAIChatDialect)

    def test_plan_json_mappings_are_deeply_frozen(self) -> None:
        binding = _binding(requirements=_requirements(sampled_logprobs=True))
        result = reconcile(_batch(binding), _NoPatchReconciler())
        nested: dict[str, JSONValue] = {"nested": {"values": ["original"]}}
        defaults = RequestDefaults({"sampling.stop": ["original"]})
        intent = CapabilityIntent(
            "sampled_logprobs",
            True,
            minimums=nested,
            request_defaults=defaults,
        )

        binding_plan = dataclasses.replace(
            result.binding_plans["binding-a"],
            declared_capabilities=nested,
            intents={"sampled_logprobs": intent},
            request_defaults=defaults,
        )
        deployment_plan = dataclasses.replace(
            result.deployment_plans["root"],
            launch_patch=nested,
            outcome_evidence={"sampled_logprobs": nested},
        )
        runtime_plan = dataclasses.replace(
            result.runtime_plans["binding-a"],
            effective_capabilities=nested,
            capability_minimums={"sampled_logprobs": nested},
            request_defaults=defaults,
        )
        cast(dict[str, JSONValue], nested["nested"])["values"] = ["mutated"]

        for frozen in (
            binding_plan.declared_capabilities,
            deployment_plan.launch_patch,
            deployment_plan.outcome_evidence["sampled_logprobs"],
            runtime_plan.effective_capabilities,
            runtime_plan.capability_minimums["sampled_logprobs"],
        ):
            inner = cast(Mapping[str, JSONValue], frozen["nested"])
            values = cast(list[JSONValue], inner["values"])
            assert values == ["original"]
            with pytest.raises(TypeError):
                cast(Any, inner)["new"] = True
            with pytest.raises(TypeError):
                values.append("forbidden")

        frozen_defaults = (
            binding_plan.request_defaults,
            binding_plan.intents["sampled_logprobs"].request_defaults,
            runtime_plan.request_defaults,
        )
        for item in frozen_defaults:
            stops = cast(list[JSONValue], item.values["sampling.stop"])
            assert stops == ["original"]
            with pytest.raises(TypeError):
                stops.append("forbidden")

        intent_nested = cast(
            Mapping[str, JSONValue],
            binding_plan.intents["sampled_logprobs"].minimums["nested"],
        )
        intent_values = cast(list[JSONValue], intent_nested["values"])
        assert intent_values == ["original"]
        with pytest.raises(TypeError):
            intent_values.append("forbidden")

    def test_deterministic_serialization_and_fingerprints(self) -> None:
        first_binding = _binding(binding_id="a")
        second_binding = _binding(binding_id="b")

        first = reconcile(_batch(second_binding, first_binding))
        second = reconcile(_batch(first_binding, second_binding))

        assert first.to_json_value() == second.to_json_value()
        json.dumps(first.to_json_value(), sort_keys=True)
        assert all(
            plan.fingerprint.startswith("sha256:")
            and plan.verification_fingerprint.startswith("sha256:")
            for plan in first.runtime_plans.values()
        )
        runtime = first.runtime_plans["a"]
        assert runtime.declared_capabilities == {}
        assert runtime.effective_capabilities["available"] == []

    def test_plan_fingerprints_cover_the_complete_serialized_payload(self) -> None:
        result = reconcile(_batch(_binding()))
        plans = (
            result.binding_plans["binding-a"],
            result.deployment_plans["root"],
            result.runtime_plans["binding-a"],
        )

        for plan in plans:
            payload = plan.to_json_value()
            fingerprint = cast(str, payload.pop("fingerprint"))
            assert fingerprint == _serialized_fingerprint(payload)

    def test_plan_fingerprints_are_derived_and_replace_recomputes_them(self) -> None:
        result = reconcile(_batch(_binding()))
        binding = result.binding_plans["binding-a"]
        deployment = result.deployment_plans["root"]
        runtime = result.runtime_plans["binding-a"]

        changed_binding = dataclasses.replace(
            binding, requested_model_id="another-model"
        )
        changed_deployment = dataclasses.replace(
            deployment, explicit_parameters={"max-model-len": 8192}
        )
        changed_runtime = dataclasses.replace(
            runtime, deployment_fingerprint="sha256:another-deployment"
        )

        assert changed_binding.fingerprint != binding.fingerprint
        assert changed_deployment.fingerprint != deployment.fingerprint
        assert changed_runtime.fingerprint != runtime.fingerprint
        assert (
            changed_runtime.verification_fingerprint != runtime.verification_fingerprint
        )

        for plan in (binding, deployment, runtime):
            # Python 3.13 corrected dataclasses.replace() to report an invalid
            # init=False override as TypeError; 3.12 raises ValueError.
            with pytest.raises((TypeError, ValueError), match="init=False"):
                dataclasses.replace(plan, fingerprint="sha256:forged")
        with pytest.raises((TypeError, ValueError), match="init=False"):
            dataclasses.replace(runtime, verification_fingerprint="sha256:forged")

        for plan_type in (
            BindingCapabilityPlan,
            DeploymentCapabilityPlan,
            RuntimeBindingPlan,
        ):
            fingerprint_field = next(
                item
                for item in dataclasses.fields(plan_type)
                if item.name == "fingerprint"
            )
            assert not fingerprint_field.init
        verification_field = next(
            item
            for item in dataclasses.fields(RuntimeBindingPlan)
            if item.name == "verification_fingerprint"
        )
        assert not verification_field.init

    def test_fingerprint_is_stable_across_python_hash_seeds(self) -> None:
        script = textwrap.dedent(
            """
            import json

            from sieval.core.models.capabilities import (
                CAPABILITY_KEYS,
                ModelCapabilityEntry,
                ModelCapabilityProfile,
                ModelCapabilityStatus,
            )
            from sieval.core.models.reconcile import (
                BindingReconcileInput,
                ConnectionScope,
                reconcile_binding,
            )
            from sieval.core.models.requirements import (
                AggregatedTaskRequirements,
                InputKind,
                InputModality,
            )

            sources = frozenset({"task-a", "task-b"})
            requirements = AggregatedTaskRequirements(
                input=frozenset({InputKind.COMPLETION}),
                input_modalities=frozenset({InputModality.TEXT}),
                input_scoring=True,
                input_sources={InputKind.COMPLETION: sources},
                modality_sources={InputModality.TEXT: sources},
                input_scoring_sources=sources,
            )
            profile = ModelCapabilityProfile(
                {
                    key: ModelCapabilityEntry(
                        ModelCapabilityStatus.SUPPORTED,
                        "test-profile",
                    )
                    for key in CAPABILITY_KEYS
                },
                authoritative=True,
            )
            result = reconcile_binding(
                BindingReconcileInput(
                    binding_id="binding-a",
                    root_deployment_key="root",
                    requested_model_id="requested-model",
                    dialect_id="openai_completions",
                    requirements=requirements,
                    model_profile=profile,
                    connection_scope=ConnectionScope("team-a", "standard", "root"),
                )
            )
            assert result.plan is not None
            print(
                json.dumps(
                    {
                        "fingerprint": result.plan.fingerprint,
                        "sources": result.plan.intents["input_scoring"].sources,
                    },
                    sort_keys=True,
                )
            )
            """
        )
        outputs: list[dict[str, object]] = []
        for seed in ("1", "2"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            outputs.append(json.loads(completed.stdout))

        assert (
            outputs
            == [
                {
                    "fingerprint": (
                        "sha256:b42312d2fa6f4082d26d2aa591331acebb5b758236b2cb71eacfec70e870f1e3"
                    ),
                    "sources": ["task-a", "task-b"],
                }
            ]
            * 2
        )

    def test_route_resolution_failure_is_a_diagnostic(self) -> None:
        deployment_input = _deployment()
        assert deployment_input.deployment is not None
        ambiguous = dataclasses.replace(
            deployment_input,
            deployment=dataclasses.replace(
                deployment_input.deployment,
                endpoints={
                    "prefill": "https://prefill.example/v1",
                    "decode": "https://decode.example/v1",
                },
            ),
        )

        result = reconcile(_batch(_binding(), deployment=ambiguous))

        assert any(item.code == "route_resolution_failed" for item in result.errors)
        assert not result.runtime_plans

    @pytest.mark.parametrize(
        ("dialect_id", "expected_type"),
        [
            ("openai_chat", OpenAIChatDialect),
            ("openai_completions", OpenAICompletionsDialect),
            ("openai_responses", OpenAIResponsesDialect),
        ],
    )
    def test_runtime_plan_binds_only_to_exact_pool_identity(
        self, dialect_id: str, expected_type: type
    ) -> None:
        result = reconcile(_batch(_binding(dialect_id=dialect_id)))
        runtime = result.runtime_plans["binding-a"]
        deployment_input = _deployment()
        assert deployment_input.deployment is not None
        connection = _Connection()
        pool = ConnectionPool(connection, runtime.connection_identity)

        dialect = bind_dialect(
            dialect_id,
            "requested-model",
            deployment_input.deployment,
            pool,
            runtime,
        )

        assert isinstance(dialect, expected_type)

        wrong_identity = ConnectionIdentity(
            endpoint=runtime.connection_identity.endpoint,
            connection_family=runtime.connection_identity.connection_family,
            credential_scope="another-team",
            retry_policy=runtime.connection_identity.retry_policy,
            quota_scope=runtime.connection_identity.quota_scope,
        )
        wrong_pool = ConnectionPool(_Connection(), wrong_identity)
        with pytest.raises(ValueError, match="identity mismatch"):
            bind_dialect(
                dialect_id,
                "requested-model",
                deployment_input.deployment,
                wrong_pool,
                runtime,
            )

    def test_binding_rejects_plan_for_another_requested_model(self) -> None:
        result = reconcile(_batch(_binding()))
        runtime = result.runtime_plans["binding-a"]
        deployment_input = _deployment()
        assert deployment_input.deployment is not None
        pool = ConnectionPool(_Connection(), runtime.connection_identity)

        with pytest.raises(DialectRegistryError, match="requested-model mismatch"):
            bind_dialect(
                "openai_chat",
                "another-model",
                deployment_input.deployment,
                pool,
                runtime,
            )
