"""
Focused tests for non-overlapping Model behaviors.

with_args/meta branches are covered in test_model_derivation.py; the IR
primitive (arun/capabilities) in test_model_arun.py; wire lowering/lifting in
tests/unit/core/models/transports/. This file keeps unique checks: quota,
runtime concurrency paths, the legacy-kwargs request builders, and the
Response -> ModelOutput bridge.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from sieval.core.models import (
    ChatInput,
    ChatModel,
    CompletionInput,
    DialectOptions,
    GenModel,
    InputScoringResult,
    Model,
    ModelOutput,
    ReasoningOutput,
    ReasoningParams,
    Request,
    RequestDefaults,
    Response,
    SamplingParams,
    StructuredOutputParams,
    TextPart,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.models.dialect import DialectError
from sieval.core.models.model import _apply_request_defaults
from sieval.core.models.reconcile import CheckStage, DeferredCheck


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gen_model():
    return GenModel(model="test-gen", api_key="fake", concurrency_limit=16)


@pytest.fixture
def unlimited_model():
    return GenModel(model="test-unlimited", api_key="fake")


# ===================================================================
# Non-overlapping behaviors
# ===================================================================
class TestModelUnique:
    @pytest.mark.parametrize("hook", ["_agenerate_impl", "_alogprobs_impl"])
    def test_removed_subclass_hook_fails_at_class_definition(self, hook: str):
        with pytest.raises(TypeError, match=rf"removed Model hook.*{hook}"):
            type("LegacyHookModel", (Model,), {hook: object()})

    def test_derived_inherits_client(self, gen_model):
        child = gen_model.with_args(temperature=0.9)
        # Same client object
        assert child._client is gen_model._client

    def test_derived_overrides_kwargs(self, gen_model):
        child = gen_model.with_args(temperature=0.9, top_p=0.8)
        assert child._kwargs == {"temperature": 0.9, "top_p": 0.8}
        # Parent unchanged
        assert gen_model._kwargs == {}

    def test_separate_legacy_clients_have_distinct_secret_free_binding_identity(self):
        first = ChatModel(
            model="same-model",
            api_base="https://same.example/v1",
            api_key="first-secret",
        )
        second = ChatModel(
            model="same-model",
            api_base="https://same.example/v1",
            api_key="second-secret",
        )

        assert first.pool is not second.pool
        assert first.pool.identity != second.pool.identity
        assert first.runtime_plan is not None
        assert second.runtime_plan is not None
        assert first.runtime_plan.binding_id != second.runtime_plan.binding_id
        assert first.runtime_plan.fingerprint != second.runtime_plan.fingerprint
        assert "first-secret" not in repr(first.pool.identity)
        assert "second-secret" not in repr(second.pool.identity)

    def test_legacy_runtime_fingerprint_covers_every_plan_field(self, gen_model):
        plan = gen_model.runtime_plan
        assert plan is not None
        variants = (
            replace(plan, binding_id=f"{plan.binding_id}:changed"),
            replace(plan, root_deployment_key=f"{plan.root_deployment_key}:changed"),
            replace(plan, requested_model_id="changed-model"),
            replace(plan, dialect_id="changed-dialect"),
            replace(plan, declared_capabilities={"sampled_logprobs": {}}),
            replace(plan, effective_capabilities={"sampled_logprobs": {}}),
            replace(plan, available_capabilities=frozenset()),
            replace(
                plan,
                capability_minimums={"top_logprobs": {"minimum": 2}},
            ),
            replace(
                plan,
                request_defaults=RequestDefaults({"sampling.temperature": 0.25}),
            ),
            replace(plan, required_output_channels=frozenset({"reasoning"})),
            replace(
                plan,
                request_checks=(
                    DeferredCheck(
                        "sampled_logprobs",
                        CheckStage.REQUEST,
                        "validate_response_channel",
                        "regression evidence",
                    ),
                ),
            ),
            replace(plan, deployment_fingerprint="sha256:changed-deployment"),
            replace(
                plan,
                resolved_route=replace(
                    plan.resolved_route,
                    service_role="changed-role",
                ),
            ),
            replace(
                plan,
                connection_identity=replace(
                    plan.connection_identity,
                    quota_scope="changed-quota",
                ),
            ),
            replace(plan, binding_plan_fingerprint="sha256:changed-binding-plan"),
            replace(
                plan,
                deployment_plan_fingerprint="sha256:changed-deployment-plan",
            ),
        )

        fingerprints = {plan.fingerprint, *(item.fingerprint for item in variants)}
        assert len(fingerprints) == len(variants) + 1

    def test_as_compat_type_rebuilds_truthful_non_owning_wrapper(self, gen_model):
        plan = gen_model.runtime_plan
        assert plan is not None
        plain = gen_model.with_dialect(gen_model.dialect_id, plan)

        compat = plain.as_compat_type(GenModel)

        assert type(compat) is GenModel
        assert compat.pool is plain.pool
        assert compat.runtime_plan is plain.runtime_plan
        assert compat._dialect is plain._dialect
        assert compat._limiter is plain._limiter
        assert compat._parent_limiter is plain._parent_limiter
        assert compat._lifecycle_owner is None

    def test_with_dialect_preserves_pool_route_and_limiter_identity(self, gen_model):
        plan = gen_model.runtime_plan
        assert plan is not None

        rebound = gen_model.with_dialect(gen_model.dialect_id, plan)

        assert rebound.pool is gen_model.pool
        assert rebound.pool.identity is gen_model.pool.identity
        assert rebound.runtime_plan is plan
        assert rebound.runtime_plan.resolved_route == plan.resolved_route
        assert rebound._limiter is gen_model._limiter
        assert rebound._parent_limiter is gen_model._parent_limiter

    def test_with_dialect_rejects_family_or_route_drift(self, gen_model):
        plan = gen_model.runtime_plan
        assert plan is not None
        wrong_family = replace(
            plan,
            resolved_route=replace(
                plan.resolved_route,
                connection_family="async_http_json",
            ),
        )
        wrong_route = replace(
            plan,
            resolved_route=replace(
                plan.resolved_route,
                endpoint="https://other.example/v1",
            ),
        )

        with pytest.raises(ValueError, match="wrong connection family"):
            gen_model.with_dialect(gen_model.dialect_id, wrong_family)
        with pytest.raises(ValueError, match="does not match the deployment"):
            gen_model.with_dialect(gen_model.dialect_id, wrong_route)

    def test_as_compat_type_rejects_mismatched_or_derived_type(self, gen_model):
        from sieval.core.models import ChatModel

        plan = gen_model.runtime_plan
        assert plan is not None
        plain = gen_model.with_dialect(gen_model.dialect_id, plan)

        with pytest.raises(ValueError, match="requires a dialect accepting 'chat'"):
            plain.as_compat_type(ChatModel)

        class DerivedGenModel(GenModel):
            pass

        with pytest.raises(TypeError, match="exactly ChatModel or GenModel"):
            plain.as_compat_type(DerivedGenModel)

    def test_plain_constructor_and_wrapper_bind_are_rejected(self, gen_model):
        with pytest.raises(TypeError, match="use Model.bind"):
            Model()
        with pytest.raises(TypeError, match="not a wrapper bind"):
            GenModel.bind(gen_model.deployment, gen_model.pool, gen_model.runtime_plan)

    def test_initialize_rejects_dialect_and_connection_family_mismatch(self, gen_model):
        plan = gen_model.runtime_plan
        assert plan is not None
        common = {
            "deployment": gen_model.deployment,
            "pool": gen_model.pool,
            "runtime_plan": plan,
            "local_limiter": None,
            "parent_limiter": None,
            "builder_defaults": {},
            "extra": None,
            "api_base": gen_model.deployment.api_base,
            "lifecycle_owner": None,
        }

        with pytest.raises(ValueError, match="bound dialect"):
            object.__new__(Model)._initialize(
                dialect=SimpleNamespace(
                    dialect_id="openai_chat",
                    connection_family=plan.resolved_route.connection_family,
                ),
                **common,
            )
        with pytest.raises(ValueError, match="connection family"):
            object.__new__(Model)._initialize(
                dialect=SimpleNamespace(
                    dialect_id=plan.dialect_id,
                    connection_family="async_http_json",
                ),
                **common,
            )

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("root_deployment_key", "other-root", "another deployment root"),
            ("deployment_fingerprint", "sha256:other", "another deployment"),
            ("requested_model_id", "other-model", "requested_model_id"),
        ],
    )
    def test_with_dialect_rejects_runtime_plan_identity_drift(
        self, gen_model, field, value, match
    ):
        plan = gen_model.runtime_plan
        assert plan is not None
        with pytest.raises(ValueError, match=match):
            gen_model.with_dialect(
                plan.dialect_id,
                replace(plan, **{field: value}),
            )

    def test_with_dialect_rejects_target_plan_dialect_disagreement(self, gen_model):
        plan = gen_model.runtime_plan
        assert plan is not None
        with pytest.raises(ValueError, match="target dialect"):
            gen_model.with_dialect("openai_chat", plan)

    @pytest.mark.anyio
    async def test_canonical_model_rejects_wrapper_lifecycle_operations(
        self, gen_model
    ):
        plan = gen_model.runtime_plan
        assert plan is not None
        plain = Model.bind(gen_model.deployment, gen_model.pool, plan)

        with pytest.raises(RuntimeError, match="does not own"):
            await plain.aclose()
        with pytest.raises(RuntimeError, match="ConnectionPool context"):
            await plain.__aenter__()

    @pytest.mark.anyio
    async def test_legacy_wrapper_context_returns_self_and_closes(self):
        model = GenModel(model="context-model", api_key="fake")

        async with model as entered:
            assert entered is model

        assert model.pool.is_closed

    def test_canonical_model_requires_tagged_input(self, gen_model):
        plan = gen_model.runtime_plan
        assert plan is not None
        plain = Model.bind(gen_model.deployment, gen_model.pool, plan)
        prompt = CompletionInput("prompt")

        assert plain._coerce_input(prompt) is prompt
        with pytest.raises(TypeError, match="does not guess input modality"):
            plain._coerce_input("prompt")


# ===================================================================
# Response metadata
# ===================================================================
class TestModelOutputResponseMetadata:
    def test_model_output_response_metadata_defaults(self):
        """New response metadata fields default to None."""
        output = ModelOutput(
            model={"model": "test", "api_base": None, "default_params": {}},
            texts=["hello"],
        )
        assert output.response_model is None
        assert output.system_fingerprint is None

    def test_model_output_response_metadata_explicit(self):
        """New response metadata fields can be set explicitly."""
        output = ModelOutput(
            model={"model": "test", "api_base": None, "default_params": {}},
            texts=["hello"],
            response_model="gpt-4o-2024-08-06",
            system_fingerprint="fp_abc123",
        )
        assert output.response_model == "gpt-4o-2024-08-06"
        assert output.system_fingerprint == "fp_abc123"


# ===================================================================
# Quota info
# ===================================================================
class TestQuota:
    def test_total_quota_with_limit(self, gen_model):
        assert gen_model.get_total_quota() == 16

    def test_total_quota_unlimited(self, unlimited_model):
        assert unlimited_model.get_total_quota() == float("inf")

    def test_available_quota_equals_total_initially(self, gen_model):
        assert gen_model.get_available_quota() == 16

    def test_available_quota_unlimited(self, unlimited_model):
        assert unlimited_model.get_available_quota() == float("inf")

    def test_quota_info_structure(self, gen_model):
        info = gen_model.get_quota_info()
        assert info["total"] == 16
        assert info["available"] == 16
        assert info["parent"] is None
        assert info["child"]["total"] == 16

    def test_quota_info_derived(self, gen_model):
        child = gen_model.with_args(concurrency_limit=4)
        info = child.get_quota_info()
        assert info["total"] == 4
        assert info["parent"]["total"] == 16
        assert info["child"]["total"] == 4

    def test_quota_info_no_limiter(self, unlimited_model):
        info = unlimited_model.get_quota_info()
        assert info["total"] == float("inf")
        assert info["parent"] is None
        assert info["child"] is None


# ===================================================================
# agenerate / alogprobs concurrency paths
# (covers the parent_limiter branches around arun)
# ===================================================================
def _build_chat_model_for_path(path):
    from tests.conftest import MockChatModel

    if path == "parent_and_child":
        base = MockChatModel(concurrency_limit=8)
        return base.with_args(concurrency_limit=4)
    if path == "parent_only":
        base = MockChatModel(concurrency_limit=8)
        child = base.with_args()
        child._limiter = None
        child._parent_limiter = base._limiter
        return child
    if path == "own_only":
        return MockChatModel(concurrency_limit=4)
    if path == "no_limiter":
        return MockChatModel()
    raise ValueError(f"Unknown path: {path}")


def _build_gen_model_for_path(path):
    from tests.conftest import MockGenModel

    if path == "parent_and_child":
        base = MockGenModel(concurrency_limit=8)
        return base.with_args(concurrency_limit=4)
    if path == "parent_only":
        base = MockGenModel(concurrency_limit=8)
        child = base.with_args()
        child._limiter = None
        child._parent_limiter = base._limiter
        return child
    if path == "own_only":
        return MockGenModel(concurrency_limit=4)
    if path == "no_limiter":
        return MockGenModel()
    raise ValueError(f"Unknown path: {path}")


def _assert_path_shape(model, path):
    if path == "parent_and_child":
        assert model._parent_limiter is not None
        assert model._limiter is not None
        return
    if path == "parent_only":
        assert model._parent_limiter is not None
        assert model._limiter is None
        return
    if path == "own_only":
        assert model._parent_limiter is None
        assert model._limiter is not None
        return
    if path == "no_limiter":
        assert model._parent_limiter is None
        assert model._limiter is None
        return
    raise ValueError(f"Unknown path: {path}")


class TestConcurrencyPaths:
    """Exercise all four limiter combinations for agenerate and alogprobs."""

    @pytest.mark.anyio
    # (limiter_path, prompt)
    @pytest.mark.parametrize(
        "path,prompt",
        [
            ("parent_and_child", "hello"),
            ("parent_only", "hello"),
            ("own_only", "hello"),
            ("no_limiter", "hello"),
        ],
    )
    async def test_agenerate_paths(self, path, prompt):
        model = _build_chat_model_for_path(path)
        _assert_path_shape(model, path)
        result = await model.agenerate(prompt)
        assert result.texts == ["unknown"]

    @pytest.mark.anyio
    # (limiter_path, prompt)
    @pytest.mark.parametrize(
        "path,prompt",
        [
            ("parent_and_child", "A"),
            ("parent_only", "B"),
            ("own_only", "C"),
            ("no_limiter", "D"),
        ],
    )
    async def test_alogprobs_paths(self, path, prompt):
        model = _build_gen_model_for_path(path)
        _assert_path_shape(model, path)
        result = await model.alogprobs(prompt)
        assert result.texts == [""]
        assert result.logprobs is not None and len(result.logprobs) == 1
        assert result.logprobs_tokens is not None and len(result.logprobs_tokens) == 1

    @pytest.mark.anyio
    async def test_alogprobs_lowers_echo_to_score_input(self):
        """alogprobs args land on the Request the transport receives."""
        from tests.conftest import HandlerTransport, MockGenModel

        model = MockGenModel()
        await model.alogprobs("A", echo=False, max_tokens=2, logprobs=3)

        assert isinstance(model._transport, HandlerTransport)
        req = model._transport.requests[0]
        assert req.scoring.input_scoring is False
        assert req.scoring.sampled_logprobs is True
        assert req.scoring.top_logprobs == 3
        assert req.sampling.max_tokens == 2
        assert req.sampling.temperature == 0.0

    @pytest.mark.anyio
    async def test_alogprobs_echo_true_sets_score_input(self):
        from tests.conftest import HandlerTransport, MockGenModel

        model = MockGenModel()
        await model.alogprobs("A", echo=True)

        assert isinstance(model._transport, HandlerTransport)
        req = model._transport.requests[0]
        assert req.scoring.input_scoring is True
        assert req.scoring.sampled_logprobs is False
        assert req.scoring.top_logprobs == 0


# ===================================================================
# Request builders (legacy OpenAI-style kwargs -> IR)
# ===================================================================
class TestBuildGenerateRequest:
    def _model(self, **kwargs):
        return GenModel(model="m", api_key="k", **kwargs)

    def test_sampling_kwargs_map_to_sampling_params(self):
        req = self._model()._build_generate_request(
            "p",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            n=3,
        )
        sp = req.sampling
        assert sp.max_tokens == 64
        assert sp.temperature == 0.7
        assert sp.top_p == 0.9
        assert sp.seed == 42
        assert sp.frequency_penalty == 0.1
        assert sp.presence_penalty == 0.2
        assert sp.n == 3
        assert req.dialect_options is None

    def test_max_completion_tokens_aliases_max_tokens(self):
        req = self._model()._build_generate_request("p", max_completion_tokens=32)
        assert req.sampling.max_tokens == 32

    def test_aliases_may_agree_but_conflicts_fail_loud(self):
        req = self._model()._build_generate_request(
            "p", max_tokens=8, max_completion_tokens=8
        )
        assert req.sampling.max_tokens == 8

        with pytest.raises(ValueError, match="max_tokens conflicts"):
            self._model()._build_generate_request(
                "p", max_tokens=8, max_completion_tokens=32
            )
        with pytest.raises(ValueError, match="score_input conflicts"):
            self._model()._build_generate_request("p", score_input=False, echo=True)
        with pytest.raises(ValueError, match="previous_response_id conflicts"):
            self._model()._build_generate_request(
                "p", previous_response_id="first", session_id="second"
            )

    def test_stop_string_becomes_tuple(self):
        req = self._model()._build_generate_request("p", stop="\n\n")
        assert req.sampling.stop == ("\n\n",)

    def test_stop_list_becomes_tuple(self):
        req = self._model()._build_generate_request("p", stop=["\n\n", "Q:"])
        assert req.sampling.stop == ("\n\n", "Q:")

    def test_stop_tuple_becomes_tuple(self):
        """Bind time stores a tuple default unconverted, so this path sees one."""
        req = self._model()._build_generate_request("p", stop=("\n\n", "Q:"))
        assert req.sampling.stop == ("\n\n", "Q:")

    def test_top_k_kwarg_is_sampling_top_k(self):
        """`top_k` is the vLLM/sglang sampling knob, not the logprobs count."""
        req = self._model()._build_generate_request("p", top_k=40)
        assert req.sampling.top_k == 40
        assert req.scoring.top_logprobs == 0

    def test_logprobs_bool_is_chat_switch(self):
        req = self._model()._build_generate_request("p", logprobs=True)
        assert req.scoring.sampled_logprobs is True
        assert req.scoring.top_logprobs == 0

    def test_logprobs_int_is_completions_count(self):
        req = self._model()._build_generate_request("p", logprobs=5)
        assert req.scoring.sampled_logprobs is True
        assert req.scoring.top_logprobs == 5

    def test_top_logprobs_sets_top_k(self):
        req = self._model()._build_generate_request("p", logprobs=True, top_logprobs=7)
        assert req.scoring.sampled_logprobs is True
        assert req.scoring.top_logprobs == 7

    def test_model_kwargs_merge_with_call_kwargs(self):
        model = self._model(temperature=0.5, max_tokens=10)
        req = model._build_generate_request("p", max_tokens=99)
        assert req.sampling.temperature == 0.5
        assert req.sampling.max_tokens == 99  # call kwargs win

    def test_unknown_kwargs_ride_in_dialect_options(self):
        req = self._model()._build_generate_request("p", min_p=0.05, echo=True)
        assert req.scoring.input_scoring is True
        assert req.dialect_options is not None
        assert req.dialect_options.dialect_id == "openai_completions"
        assert req.dialect_options.values == {"min_p": 0.05}

    def test_stream_defaults_true_and_is_poppable(self):
        assert self._model()._build_generate_request("p").scheduling.stream is True
        assert (
            self._model()._build_generate_request("p", stream=False).scheduling.stream
            is False
        )

    def test_reasoning_effort_maps_to_reasoning_params(self):
        req = self._model()._build_generate_request("p", reasoning_effort="high")
        assert req.reasoning is not None
        assert req.reasoning.effort == "high"

    def test_reasoning_effort_rejects_non_string_instead_of_dropping_it(self):
        with pytest.raises(TypeError, match="reasoning_effort must be a string"):
            self._model()._build_generate_request("p", reasoning_effort=3)

    def test_response_format_and_tools(self):
        fmt = {"type": "json_object"}
        tools = [{"type": "function", "function": {"name": "f"}}]
        req = self._model()._build_generate_request(
            "p", response_format=fmt, tools=tools
        )
        assert req.structured_output.format == "json_object"
        assert req.tools.functions == tuple(tools)

    def test_message_input_is_materialized(self):
        from sieval.core.models import ChatModel

        model = ChatModel(model="m", api_key="k")

        def gen():
            yield {"role": "user", "content": "hi"}

        req = model._build_generate_request(gen())
        assert isinstance(req.input, ChatInput)
        assert len(req.input.messages) == 1
        assert req.input.messages[0].role == "user"
        part = req.input.messages[0].content[0]
        assert isinstance(part, TextPart)
        assert part.text == "hi"


class TestBuilderValidation:
    def _model(self):
        return GenModel(model="m", api_key="k")

    @pytest.mark.parametrize(
        "default",
        [{"a", "b"}, (item for item in ["a", "b"])],
        ids=["set", "generator"],
    )
    def test_binding_rejects_default_params_that_cannot_round_trip(self, default):
        """``default_params`` is persisted: a ``set`` reorders, a generator empties."""
        with pytest.raises(
            TypeError, match="default_params.stop must be JSON-compatible"
        ):
            GenModel(model="m", api_key="k", stop=default)

    def test_with_args_rejects_default_params_that_cannot_round_trip(self):
        model = GenModel(model="m", api_key="k")

        with pytest.raises(
            TypeError, match="default_params.stop must be JSON-compatible"
        ):
            model.with_args(stop={"a", "b"})

    def test_meta_keeps_tuple_default_params(self):
        model = GenModel(model="m", api_key="k", stop=("a", "b"))

        assert model.meta()["default_params"]["stop"] == ["a", "b"]

    @pytest.mark.parametrize(
        "stop",
        [{"a", "b"}, (item for item in ["a", "b"])],
        ids=["set", "generator"],
    )
    def test_call_time_stop_is_refused_on_the_same_terms_as_a_default(self, stop):
        """``stop`` lands in the persisted ``request_params``, so it needs an order.

        The builder pops ``stop`` before the JSON check runs, so narrowing that
        check alone left this path taking any iterable.
        """
        with pytest.raises(TypeError, match="stop must be a string, list, or tuple"):
            self._model()._build_generate_request("p", stop=stop)

    def test_n_must_be_int(self):
        with pytest.raises(TypeError, match="n must be an int"):
            self._model()._build_generate_request("p", n="3")

    def test_n_bool_rejected(self):
        with pytest.raises(TypeError, match="n must be an int"):
            self._model()._build_generate_request("p", n=True)

    def test_n_below_one_rejected(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            self._model()._build_generate_request("p", n=0)

    def test_stream_must_be_bool(self):
        with pytest.raises(TypeError, match="stream must be a bool"):
            self._model()._build_generate_request("p", stream="yes")

    def test_non_iterable_prompt_rejected(self):
        with pytest.raises(TypeError, match="prompt must be text or CompletionInput"):
            self._model()._build_generate_request(123)

    def test_binding_resource_request_kwarg_is_rejected(self):
        with pytest.raises(ValueError, match="cannot change binding resources"):
            self._model()._build_generate_request("p", api_key="secret")

    def test_alogprobs_rejects_n_gt_1(self):
        with pytest.raises(ValueError, match="alogprobs only supports n=1"):
            self._model()._build_logprobs_request(
                "p", max_tokens=1, logprobs=5, score_input=True, temperature=0.0, n=2
            )

    def test_logprobs_request_forces_logprob_fields(self):
        model = GenModel(model="m", api_key="k", logprobs=99)
        req = model._build_logprobs_request(
            "p", max_tokens=1, logprobs=5, score_input=True, temperature=0.0
        )
        # explicit alogprobs args override any _kwargs-borne logprobs config
        assert req.scoring.sampled_logprobs is False
        assert req.scoring.top_logprobs == 0
        assert req.scoring.input_scoring is True
        assert req.sampling.max_tokens == 1

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"temperature": True}, TypeError, "temperature must be a number"),
            ({"top_k": True}, TypeError, "top_k must be an integer"),
            ({"stop": ["ok", 1]}, TypeError, "only strings"),
            ({"stop": 3}, TypeError, "string, list, or tuple"),
            ({"return_logprobs": "yes"}, TypeError, "must be a bool"),
            ({"logprobs": "five"}, TypeError, "bool or integer"),
            ({"top_logprobs": True}, TypeError, "must be an integer"),
            ({"score_input": 1}, TypeError, "must be a bool"),
            ({"parallel_tool_calls": "yes"}, TypeError, "must be a bool"),
            ({"previous_response_id": 1}, TypeError, "must be a string"),
            ({"opaque_continuation": "opaque"}, TypeError, "OpaqueContinuation"),
            ({"dialect_options": {}}, TypeError, "must be DialectOptions"),
            ({"extra_body": []}, TypeError, "must be a mapping"),
            ({"extra_body": {1: "value"}}, TypeError, "keys must be strings"),
        ],
    )
    def test_request_builder_rejects_invalid_leaf_types(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            self._model()._build_generate_request("p", **kwargs)

    @pytest.mark.parametrize(
        ("tools", "match"),
        [
            ({"type": "function"}, "iterable of mappings"),
            (["not-a-mapping"], "contain mappings"),
        ],
    )
    def test_tools_reject_ambiguous_container_shapes(self, tools, match):
        with pytest.raises(TypeError, match=match):
            self._model()._build_generate_request("p", tools=tools)

    def test_tools_none_and_json_iterables_are_normalized(self):
        req = self._model()._build_generate_request(
            "p",
            tools=None,
            tool_choice=("first", 2),
        )
        assert req.tools.functions == ()
        assert req.tools.choice == ["first", 2]

    @pytest.mark.parametrize(
        ("tool_choice", "match"),
        [
            ({1: "bad-key"}, "keys must be strings"),
            (object(), "JSON-compatible"),
            ({"a", "b"}, "JSON-compatible"),
            ((item for item in ["a"]), "JSON-compatible"),
        ],
    )
    def test_tool_choice_must_be_json_compatible(self, tool_choice, match):
        with pytest.raises(TypeError, match=match):
            self._model()._build_generate_request("p", tool_choice=tool_choice)

    def test_reasoning_object_is_preserved_but_alias_conflict_is_rejected(self):
        reasoning = ReasoningParams(budget_tokens=32)
        req = self._model()._build_generate_request("p", reasoning=reasoning)
        assert req.reasoning is reasoning

        with pytest.raises(ValueError, match="cannot both be set"):
            self._model()._build_generate_request(
                "p", reasoning=reasoning, reasoning_effort="high"
            )
        with pytest.raises(TypeError, match="must be ReasoningParams"):
            self._model()._build_generate_request("p", reasoning={})

    @pytest.mark.parametrize(
        ("response_format", "error", "match"),
        [
            ("json", TypeError, "must be a mapping"),
            ({"type": "text"}, ValueError, "unsupported"),
            (
                {"type": "json_schema", "json_schema": []},
                TypeError,
                "requires a mapping",
            ),
            (
                {"type": "json_schema", "json_schema": {}},
                TypeError,
                "requires `schema`",
            ),
            (
                {
                    "type": "json_schema",
                    "json_schema": {"schema": {}, "name": 3},
                },
                TypeError,
                "name must be a string",
            ),
            (
                {
                    "type": "json_schema",
                    "json_schema": {"schema": {}, "strict": "yes"},
                },
                TypeError,
                "strict must be a bool",
            ),
        ],
    )
    def test_structured_output_rejects_invalid_legacy_shapes(
        self, response_format, error, match
    ):
        with pytest.raises(error, match=match):
            self._model()._build_generate_request("p", response_format=response_format)

    def test_structured_output_accepts_ir_value_and_json_schema(self):
        value = StructuredOutputParams(format="json_object")
        assert (
            self._model()
            ._build_generate_request("p", response_format=value)
            .structured_output
            is value
        )

        req = self._model()._build_generate_request(
            "p",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {"type": "object"},
                    "name": "answer",
                    "strict": True,
                },
            },
        )
        assert req.structured_output.schema == {"type": "object"}
        assert req.structured_output.name == "answer"
        assert req.structured_output.strict is True

    def test_suffix_requires_completion_input_and_cannot_disagree(self):
        with pytest.raises(TypeError, match="suffix must be a string"):
            self._model()._build_generate_request("p", suffix=1)

        chat = ChatModel(model="m", api_key="k")
        with pytest.raises(TypeError, match="requires CompletionInput"):
            chat._build_generate_request("p", suffix="tail")

        prompt = CompletionInput("p", suffix="first")
        with pytest.raises(ValueError, match="suffix conflicts"):
            self._model()._build_generate_request(prompt, suffix="second")

    def test_dialect_options_validate_target_resources_and_duplicates(self):
        model = self._model()
        with pytest.raises(ValueError, match="another dialect"):
            model._build_generate_request(
                "p", dialect_options=DialectOptions("openai_chat", {})
            )
        with pytest.raises(ValueError, match="binding resources"):
            model._build_generate_request(
                "p",
                dialect_options=DialectOptions(
                    "openai_completions", {"api_key": "secret"}
                ),
            )
        with pytest.raises(ValueError, match="duplicate dialect option"):
            model._build_generate_request(
                "p",
                dialect_options=DialectOptions("openai_completions", {"min_p": 0.1}),
                extra_body={"min_p": 0.2},
            )
        with pytest.raises(ValueError, match="duplicate dialect option"):
            model._build_generate_request(
                "p",
                dialect_options=DialectOptions("openai_completions", {"min_p": 0.1}),
                min_p=0.2,
            )

    @pytest.mark.parametrize("logprobs", [True, "five"])
    def test_logprobs_builder_requires_integer(self, logprobs):
        with pytest.raises(TypeError, match="logprobs must be an integer"):
            self._model()._build_logprobs_request(
                "p",
                max_tokens=1,
                logprobs=logprobs,
                score_input=True,
                temperature=0.0,
            )

    @pytest.mark.anyio
    async def test_alogprobs_legacy_bridge_rejects_missing_channels(self):
        model = self._model()

        async def no_logprobs(req):
            del req
            return Response(texts=("",))

        model.arun = no_logprobs
        with pytest.raises(RuntimeError, match="server returned none"):
            await model.alogprobs("p", echo=False)


class TestRequestDefaultsProjection:
    def _request(self, *, max_tokens=None):
        return Request(
            input=CompletionInput("p"),
            sampling=SamplingParams(max_tokens=max_tokens),
        )

    @pytest.mark.parametrize(
        ("values", "match"),
        [
            ({"sampling": 1}, "invalid request-default leaf path"),
            ({"input.text": "x"}, "cannot target"),
            ({"unknown.value": 1}, "cannot target"),
            ({"sampling.unknown": 1}, "unknown request-default leaf path"),
            ({"sampling.stop": ["ok", 1]}, "must be strings"),
        ],
    )
    def test_invalid_default_paths_and_values_fail_loud(self, values, match):
        with pytest.raises(DialectError, match=match):
            _apply_request_defaults(self._request(), RequestDefaults(values))

    def test_defaults_normalize_stop_and_do_not_override_explicit_values(self):
        projected = _apply_request_defaults(
            self._request(),
            RequestDefaults(
                {
                    "sampling.stop": ["END"],
                    "sampling.max_tokens": 8,
                }
            ),
        )
        assert projected.sampling.stop == ("END",)
        assert projected.sampling.max_tokens == 8

        explicit = _apply_request_defaults(
            self._request(max_tokens=4),
            RequestDefaults({"sampling.max_tokens": 8}),
        )
        assert explicit.sampling.max_tokens == 4


# ===================================================================
# Response -> ModelOutput bridge
# ===================================================================
class TestResponseBridge:
    def _bridge(self, resp: Response) -> ModelOutput:
        return GenModel(model="m", api_key="k")._response_to_model_output(resp)

    def test_basic_fields(self):
        out = self._bridge(
            Response(
                texts=("a", "b"),
                finish_reasons=("stop", "length"),
                usage=UsageStats(input_tokens=3, output_tokens=4, total_tokens=7),
                request_params={"max_tokens": 4},
                response_model="served-model",
                system_fingerprint="fp",
            )
        )
        assert isinstance(out, ModelOutput)
        assert out.texts == ["a", "b"]
        assert out.finish_reasons == ["stop", "length"]
        assert out.usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
        assert out.request_params == {"max_tokens": 4}
        assert out.response_model == "served-model"
        assert out.system_fingerprint == "fp"
        assert out.model["model"] == "m"

    def test_usage_absent_stays_none(self):
        """Absence != zeros: a zero-filled usage dict would silently corrupt
        the ARC echoed-logprob slice."""
        out = self._bridge(Response(texts=("t",)))
        assert out.usage is None

    def test_input_scoring_is_flattened_ahead_of_sampled_logprobs(self):
        """The legacy echo layout: prompt tokens first, then the completion."""
        resp = Response(
            texts=("t",),
            input_scoring=InputScoringResult(
                token_logprobs=(
                    TokenLogprob(token="Hello", logprob=None),
                    TokenLogprob(token=" world", logprob=-0.5),
                )
            ),
            logprobs=(TokenLogprob(token=" !", logprob=-1.5),),
        )
        out = self._bridge(resp)
        assert out.logprobs_tokens == ["Hello", " world", " !"]
        assert out.logprobs == [None, -0.5, -1.5]

    def test_logprobs_absent_stays_none(self):
        out = self._bridge(Response(texts=("t",)))
        assert out.logprobs_tokens is None
        assert out.logprobs is None
        assert out.top_logprobs is None

    def test_logprobs_present_but_empty_is_empty_list(self):
        """Anomaly detection distinguishes present-but-empty from absent."""
        out = self._bridge(Response(texts=("t",), logprobs=()))
        assert out.logprobs_tokens == []
        assert out.logprobs == []

    def test_top_logprobs_coalesce_duplicates_by_max(self):
        """Distinct token ids can normalize to identical text (sglang Ġ);
        keep the highest logprob, matching legacy CMMLU semantics."""
        resp = Response(
            texts=("t",),
            top_logprobs=(
                (
                    TopKEntry(token=" A", logprob=-2.0, token_id=1),
                    TopKEntry(token=" A", logprob=-0.5, token_id=2),
                    TopKEntry(token=" B", logprob=-1.0, token_id=3),
                ),
            ),
        )
        out = self._bridge(resp)
        assert out.top_logprobs == [{" A": -0.5, " B": -1.0}]

    def test_top_logprobs_empty_collapses_to_none(self):
        out = self._bridge(Response(texts=("t",), top_logprobs=()))
        assert out.top_logprobs is None

    def test_reasoning_text_becomes_reasoning_texts(self):
        out = self._bridge(
            Response(texts=("t",), reasoning=(ReasoningOutput(text="thinking..."),))
        )
        assert out.reasoning_texts == ["thinking..."]

    def test_empty_reasoning_stays_none(self):
        out = self._bridge(
            Response(texts=("t",), reasoning=(ReasoningOutput(text=""),))
        )
        assert out.reasoning_texts is None
