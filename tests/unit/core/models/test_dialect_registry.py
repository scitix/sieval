"""Tests for stable dialect descriptors and executable binder selection.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import dataclasses
import inspect
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

import sieval.core.models.dialect_registry as registry
from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    DialectCapabilityBinding,
    DialectCapabilityStatus,
    Supported,
)
from sieval.core.models.deployment import (
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
    DeploymentPlanProjection,
    Engine,
    ServingFacts,
    resolve_route,
)
from sieval.core.models.dialect_registry import (
    DIALECT_BINDERS,
    DIALECT_SPECS,
    DialectImplementationStatus,
    DialectNotImplemented,
    DialectRegistryError,
    DialectSpec,
    RequestSeedSupport,
    UnknownDialect,
    _normalized_symbols,
    _outcomes_from_decisions,
    bind_dialect,
    capability_decisions_for,
    compatibility_factory_for,
    dialect_is_bindable,
    dialect_registry_to_json,
    get_dialect_spec,
)
from sieval.core.models.dialects.openai_chat import OpenAIChatDialect
from sieval.core.models.dialects.openai_completions import OpenAICompletionsDialect
from sieval.core.models.dialects.openai_responses import OpenAIResponsesDialect

EXPECTED_DIALECTS = (
    "openai_chat",
    "openai_completions",
    "openai_responses",
    "anthropic_messages",
    "google_genai",
    "sglang_native",
    "vllm_native",
)


class _Connection:
    def close(self) -> None:
        pass


def _runtime_binding(
    dialect_id: str = "openai_chat",
) -> tuple[Deployment, ConnectionPool[Any], SimpleNamespace]:
    deployment = Deployment(
        deployment_id="deployment",
        plan=DeploymentPlanProjection("sha256:plan", "vllm"),
        engine=Engine("vllm"),
        engine_source="deployment",
        api_base="https://models.example/v1",
        endpoints={},
        topology=None,
        metrics_url=None,
        facts=ServingFacts(),
    )
    route = resolve_route(deployment, dialect_id, "openai_sdk")
    identity = ConnectionIdentity(
        endpoint=route.endpoint,
        connection_family=route.connection_family,
        credential_scope="test",
        retry_policy="standard",
        quota_scope="deployment",
    )
    plan = SimpleNamespace(
        dialect_id=dialect_id,
        requested_model_id="requested-model",
        deployment_fingerprint=deployment.fingerprint,
        resolved_route=route,
        connection_identity=identity,
    )
    return deployment, ConnectionPool(_Connection(), identity), plan


class TestDialectDescriptors:
    def test_complete_stable_symbol_space_is_serializable(self) -> None:
        assert tuple(DIALECT_SPECS) == EXPECTED_DIALECTS
        assert all(
            set(spec.capability_outcomes) == set(CAPABILITY_KEYS)
            for spec in DIALECT_SPECS.values()
        )

        encoded = json.dumps(
            dialect_registry_to_json(), sort_keys=True, separators=(",", ":")
        )
        assert "vllm_native" in encoded
        assert "openai_chat" in encoded
        for value in dialect_registry_to_json().values():
            assert isinstance(value, dict)
            assert "request_seed_support" in value

    def test_request_seed_support_is_required_and_explicit(self) -> None:
        field = DialectSpec.__dataclass_fields__["request_seed_support"]
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING
        assert {
            dialect_id: spec.request_seed_support
            for dialect_id, spec in DIALECT_SPECS.items()
        } == {
            "openai_chat": RequestSeedSupport.SUPPORTED,
            "openai_completions": RequestSeedSupport.SUPPORTED,
            "openai_responses": RequestSeedSupport.UNSUPPORTED,
            "anthropic_messages": RequestSeedSupport.RESERVED,
            "google_genai": RequestSeedSupport.RESERVED,
            "sglang_native": RequestSeedSupport.RESERVED,
            "vllm_native": RequestSeedSupport.RESERVED,
        }

    def test_implemented_dialects_are_active_and_bindable(self) -> None:
        active = {
            dialect_id
            for dialect_id, spec in DIALECT_SPECS.items()
            if spec.implementation_status is DialectImplementationStatus.ACTIVE
        }
        assert active == {
            "openai_chat",
            "openai_completions",
            "openai_responses",
        }
        assert set(DIALECT_BINDERS) == active
        assert all(inspect.isfunction(binder) for binder in DIALECT_BINDERS.values())

    def test_unknown_identifier_is_not_inferred(self) -> None:
        with pytest.raises(UnknownDialect, match="unknown dialect"):
            get_dialect_spec("chat")

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"dialect_id": ""}, "dialect_id"),
            ({"connection_family": ""}, "connection_family"),
            ({"implementation_status": "active"}, "implementation_status"),
            ({"request_seed_support": "supported"}, "request_seed_support"),
            (
                {"request_seed_support": RequestSeedSupport.RESERVED},
                "cannot reserve request-seed support",
            ),
            (
                {
                    "capability_outcomes": dict.fromkeys(
                        CAPABILITY_KEYS, DialectCapabilityStatus.RESERVED
                    )
                },
                "active dialect cannot reserve",
            ),
            ({"input_kinds": cast(Any, ["chat"])}, "must be a tuple"),
            ({"input_kinds": ("",)}, "non-empty strings"),
            ({"input_modalities": ("text", "text")}, "duplicates"),
        ],
    )
    def test_descriptor_rejects_invalid_metadata(
        self, changes: dict[str, object], message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            dataclasses.replace(DIALECT_SPECS["openai_chat"], **changes)

    def test_symbol_lists_are_canonicalized(self) -> None:
        assert _normalized_symbols(("tool", "image"), "modalities") == (
            "image",
            "tool",
        )

    def test_bindability_and_compatibility_factory_follow_binder_registry(
        self,
    ) -> None:
        assert dialect_is_bindable("openai_chat") is True
        assert compatibility_factory_for("openai_chat") is None
        assert dialect_is_bindable("sglang_native") is False
        assert compatibility_factory_for("sglang_native") is None

    def test_binding_key_must_match_decision_key(self) -> None:
        decisions = dict(DIALECT_BINDERS["openai_chat"].capability_decisions)
        decisions["sampled_logprobs"] = Supported(
            DialectCapabilityBinding("top_logprobs")
        )

        with pytest.raises(ValueError, match="does not match binding key"):
            _outcomes_from_decisions(decisions)

    def test_decision_row_must_be_complete_and_well_typed(self) -> None:
        incomplete = dict(DIALECT_BINDERS["openai_chat"].capability_decisions)
        incomplete.pop("reasoning")
        incomplete["typo"] = next(iter(incomplete.values()))
        with pytest.raises(ValueError, match="incomplete.*reasoning.*typo"):
            _outcomes_from_decisions(incomplete)

        invalid = dict(DIALECT_BINDERS["openai_chat"].capability_decisions)
        invalid["reasoning"] = cast(Any, object())
        with pytest.raises(TypeError, match="invalid decision"):
            _outcomes_from_decisions(invalid)

    def test_binding_request_leaf_must_exist_in_request_schema(self) -> None:
        decisions = dict(DIALECT_BINDERS["openai_chat"].capability_decisions)
        decisions["sampled_logprobs"] = Supported(
            DialectCapabilityBinding(
                "sampled_logprobs",
                request_leaves=("scoring.sampled_logprob",),
            )
        )

        with pytest.raises(ValueError, match="unknown Request leaf"):
            _outcomes_from_decisions(decisions)

    def test_binding_response_channel_must_exist_in_response_contract(self) -> None:
        decisions = dict(DIALECT_BINDERS["openai_chat"].capability_decisions)
        decisions["sampled_logprobs"] = Supported(
            DialectCapabilityBinding(
                "sampled_logprobs",
                response_channels=("logprob",),
            )
        )

        with pytest.raises(ValueError, match="unknown Response channel"):
            _outcomes_from_decisions(decisions)

    def test_request_leaf_has_only_one_capability_owner(self) -> None:
        decisions = dict(DIALECT_BINDERS["openai_chat"].capability_decisions)
        decisions["top_logprobs"] = Supported(
            DialectCapabilityBinding(
                "top_logprobs",
                request_leaves=("scoring.sampled_logprobs",),
            )
        )

        with pytest.raises(ValueError, match="owned by both"):
            _outcomes_from_decisions(decisions)

    def test_capability_decisions_are_available_only_for_active_dialects(self) -> None:
        assert set(capability_decisions_for("openai_chat")) == set(CAPABILITY_KEYS)
        assert set(capability_decisions_for("openai_responses")) == set(CAPABILITY_KEYS)
        with pytest.raises(DialectNotImplemented, match="later #25 adapter"):
            capability_decisions_for("anthropic_messages")


class TestDialectBinders:
    @pytest.mark.parametrize(
        ("dialect_id", "message"),
        [
            ("vllm_native", "explicitly deferred"),
            ("sglang_native", "legacy bypass"),
        ],
    )
    def test_reserved_dialects_fail_with_named_error(
        self, dialect_id: str, message: str
    ) -> None:
        with pytest.raises(DialectNotImplemented, match=message):
            bind_dialect(
                dialect_id,
                "model",
                cast(Any, None),
                cast(ConnectionPool[Any], None),
                cast(Any, None),
            )

    @pytest.mark.parametrize(
        ("dialect_id", "expected_type"),
        [
            ("openai_chat", OpenAIChatDialect),
            ("openai_completions", OpenAICompletionsDialect),
            ("openai_responses", OpenAIResponsesDialect),
        ],
    )
    def test_active_binders_construct_expected_dialect(
        self, dialect_id: str, expected_type: type
    ) -> None:
        binder = DIALECT_BINDERS[dialect_id]
        dialect = binder(object(), "requested-model")

        assert isinstance(dialect, expected_type)
        assert dialect.dialect_id == dialect_id
        assert dialect.connection_family == "openai_sdk"

    @pytest.mark.parametrize(
        ("dialect_id", "expected_type"),
        [
            ("openai_chat", OpenAIChatDialect),
            ("openai_completions", OpenAICompletionsDialect),
            ("openai_responses", OpenAIResponsesDialect),
        ],
    )
    def test_validated_runtime_plan_binds_active_dialect(
        self, dialect_id: str, expected_type: type
    ) -> None:
        deployment, pool, plan = _runtime_binding(dialect_id)

        dialect = bind_dialect(
            dialect_id, "requested-model", deployment, pool, cast(Any, plan)
        )

        assert isinstance(dialect, expected_type)

    @pytest.mark.parametrize(
        ("requested_model_id", "changes", "message"),
        [
            ("", {}, "requested_model_id"),
            (
                "requested-model",
                {"dialect_id": "openai_completions"},
                "dialect mismatch",
            ),
            (
                "requested-model",
                {"requested_model_id": "other"},
                "requested-model mismatch",
            ),
            (
                "requested-model",
                {"deployment_fingerprint": "sha256:other"},
                "another deployment snapshot",
            ),
        ],
    )
    def test_runtime_plan_identity_mismatches_are_rejected(
        self,
        requested_model_id: str,
        changes: dict[str, object],
        message: str,
    ) -> None:
        deployment, pool, plan = _runtime_binding()
        for name, value in changes.items():
            setattr(plan, name, value)

        with pytest.raises((ValueError, DialectRegistryError), match=message):
            bind_dialect(
                "openai_chat",
                requested_model_id,
                deployment,
                pool,
                cast(Any, plan),
            )

    def test_runtime_plan_route_family_and_fingerprint_are_checked(self) -> None:
        deployment, pool, plan = _runtime_binding()
        plan.resolved_route = dataclasses.replace(
            plan.resolved_route, connection_family="async_http_json"
        )
        with pytest.raises(DialectRegistryError, match="wrong connection family"):
            bind_dialect(
                "openai_chat", "requested-model", deployment, pool, cast(Any, plan)
            )

        deployment, pool, plan = _runtime_binding()
        plan.resolved_route = dataclasses.replace(
            plan.resolved_route, fingerprint="sha256:stale"
        )
        with pytest.raises(DialectRegistryError, match="does not match"):
            bind_dialect(
                "openai_chat", "requested-model", deployment, pool, cast(Any, plan)
            )

    @pytest.mark.parametrize(
        ("dialect_id", "family", "message"),
        [
            ("wrong", "openai_sdk", "wrong id"),
            ("openai_chat", "async_http_json", "wrong connection family"),
        ],
    )
    def test_binder_result_identity_is_checked(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dialect_id: str,
        family: str,
        message: str,
    ) -> None:
        deployment, pool, plan = _runtime_binding()

        def invalid_binder(connection: object, requested_model_id: str) -> Any:
            del connection, requested_model_id
            return SimpleNamespace(dialect_id=dialect_id, connection_family=family)

        cast(Any, invalid_binder).capability_decisions = DIALECT_BINDERS[
            "openai_chat"
        ].capability_decisions
        cast(Any, invalid_binder).compatibility_factory = None
        monkeypatch.setattr(
            registry,
            "DIALECT_BINDERS",
            {**DIALECT_BINDERS, "openai_chat": invalid_binder},
        )

        with pytest.raises(DialectRegistryError, match=message):
            bind_dialect(
                "openai_chat", "requested-model", deployment, pool, cast(Any, plan)
            )
