"""Shell tests: backend selector wiring for SglangGenModel.

RFC #25 moved the native /generate wire logic (URL derivation, sampling-param
translation, triple parsing, ``_normalize_token_text``, the radix-cache guard)
into ``SglangTransport``; the former ``_agenerate_impl`` / ``_alogprobs_impl``
coverage moved with it to tests/unit/core/models/transports/test_sglang.py,
and the request-builder validation (n/stream types, alogprobs n=1) lives on
``Model`` in tests/unit/core/models/test_model.py. What remains here is the
selector contract: SglangGenModel pairs the shared client (and its api_base)
with that transport, which supplies the model's capabilities.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest

from sieval.core.models import (
    DEFAULT_REQUEST_TIMEOUT,
    Capability,
    CompletionInput,
    DialectOptions,
    HostedToolSpec,
    OpaqueContinuation,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    SessionParams,
    SglangGenModel,
    ToolParams,
)
from sieval.core.models.dialect import RequestAuditError
from sieval.core.models.transports.sglang import SglangTransport


class TestDefaultTransport:
    def test_active_binder_can_replace_facade_without_editing_facade(self):
        sentinel = object()

        with patch(
            "sieval.core.models.sglang_gen_model.compatibility_factory_for",
            return_value=lambda *args, **kwargs: sentinel,
        ):
            result = SglangGenModel(model="m", api_key="local")

        assert result is sentinel

    def test_builds_sglang_transport(self):
        m = SglangGenModel(model="m", api_base="http://host:8000/v1", api_key="local")
        assert isinstance(m._transport, SglangTransport)
        assert Capability.SampledLogprobsWithTokenIds in m.capabilities

    def test_transport_bound_to_shared_client_model_and_api_base(self):
        m = SglangGenModel(model="m", api_base="http://host:8000/v1", api_key="local")
        assert isinstance(m._transport, SglangTransport)
        assert m._transport._client is m._client
        assert m._transport._model == "m"
        assert m._transport._api_base == "http://host:8000/v1"

    def test_client_declares_the_shared_request_timeout(self):
        """The other bypass owes the same declared bound as the factory.

        This facade sits outside canonical binding and builds its own client, so
        a contract asserted against ``connection_factory`` never reaches it. The
        SDK's default is the same value, which is why a drift would be silent.
        """
        client = SimpleNamespace(
            base_url="http://host:8000/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models.sglang_gen_model.AsyncOpenAI",
            return_value=client,
        ) as client_factory:
            SglangGenModel(model="m", api_base="http://host:8000/v1", api_key="local")

        client_factory.assert_called_once_with(
            base_url="http://host:8000/v1",
            api_key="local",
            max_retries=3,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )

    def test_subclass_does_not_use_registered_compatibility_factory(self):
        class DerivedSglangGenModel(SglangGenModel):
            pass

        with patch(
            "sieval.core.models.sglang_gen_model.compatibility_factory_for"
        ) as factory:
            model = DerivedSglangGenModel(model="m", api_key="local")

        factory.assert_not_called()
        assert type(model) is DerivedSglangGenModel

    def test_legacy_identity_and_input_coercion_are_explicit(self):
        model = SglangGenModel(model="m", api_key="local")
        prompt = CompletionInput("prompt")

        assert model.dialect_id == "sglang_legacy"
        assert model.runtime_plan is None
        assert model._coerce_input(prompt) is prompt
        assert model._coerce_input("prompt") == prompt
        with pytest.raises(TypeError, match="text or CompletionInput"):
            model._coerce_input(3)

    def test_private_pool_identity_is_unique_and_secret_free(self):
        first = SglangGenModel(model="m", api_key="first-secret")
        second = SglangGenModel(model="m", api_key="second-secret")

        assert first.pool.identity != second.pool.identity
        assert (
            first.pool.identity.credential_scope
            != second.pool.identity.credential_scope
        )
        assert first.pool.identity.quota_scope != second.pool.identity.quota_scope
        assert "first-secret" not in repr(first.pool.identity)
        assert "second-secret" not in repr(second.pool.identity)

    def test_legacy_facade_cannot_rebind_or_lose_its_owner(self):
        model = SglangGenModel(model="m", api_key="local")

        with pytest.raises(RuntimeError, match="cannot rebind"):
            model.with_dialect("sglang_native", object())
        model._lifecycle_owner = None
        with pytest.raises(RuntimeError, match="no lifecycle owner"):
            model._legacy_lifecycle_owner()


class _StubTransport:
    capabilities = frozenset()

    def __init__(self) -> None:
        self.calls = 0
        self.entered = anyio.Event()
        self.release = anyio.Event()

    async def arun(self, req: Request) -> Response:
        del req
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return Response(texts=("ok",))


class _ImmediateTransport:
    capabilities = frozenset()

    def __init__(self) -> None:
        self.requests: list[Request] = []

    async def arun(self, req: Request) -> Response:
        self.requests.append(req)
        return Response(texts=("ok",))


def _legacy_model(transport: object) -> tuple[SglangGenModel, AsyncMock]:
    close = AsyncMock()
    client = SimpleNamespace(base_url="http://host:8000/v1/", close=close)
    with patch(
        "sieval.core.models.sglang_gen_model.AsyncOpenAI",
        return_value=client,
    ):
        model = SglangGenModel(
            model="m",
            api_key="local",
            transport=cast(Any, transport),
        )
    return model, close


class TestLegacyRequestGate:
    @pytest.mark.anyio
    async def test_legacy_builder_rejects_dropped_fields_before_pool_acquire(
        self,
    ) -> None:
        transport = _ImmediateTransport()
        model, _ = _legacy_model(transport)

        with (
            patch.object(model.pool, "acquire") as acquire,
            pytest.raises(RequestAuditError) as exc_info,
        ):
            await model.agenerate(
                "prompt",
                reasoning_effort="high",
                session_id="response-1",
                suffix="tail",
            )

        detail = str(exc_info.value)
        assert "input.completion.suffix" in detail
        assert "reasoning.effort" in detail
        assert "session.previous_response_id" in detail
        acquire.assert_not_called()
        assert transport.requests == []
        await model.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "path"),
        [
            (
                Request(input=CompletionInput("prompt", suffix="tail")),
                "input.completion.suffix",
            ),
            (
                Request(
                    input=CompletionInput("prompt"),
                    reasoning=ReasoningParams(effort="high"),
                ),
                "reasoning.effort",
            ),
            (
                Request(
                    input=CompletionInput("prompt"),
                    reasoning=ReasoningParams(budget_tokens=32),
                ),
                "reasoning.budget_tokens",
            ),
            (
                Request(
                    input=CompletionInput("prompt"),
                    tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
                ),
                "tools.hosted",
            ),
            (
                Request(
                    input=CompletionInput("prompt"),
                    session=SessionParams(previous_response_id="response-1"),
                ),
                "session.previous_response_id",
            ),
            (
                Request(
                    input=CompletionInput("prompt"),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation(
                            "sglang_legacy", "continuation"
                        )
                    ),
                ),
                "session.opaque_continuation",
            ),
        ],
    )
    async def test_unsupported_leaf_is_rejected_before_transport(
        self, req: Request, path: str
    ) -> None:
        transport = _ImmediateTransport()
        model, close = _legacy_model(transport)

        with pytest.raises(RequestAuditError, match=path):
            await model.arun(req)

        assert transport.requests == []
        await model.aclose()
        close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_request_seed_is_an_explicit_engine_level_noop(self) -> None:
        transport = _ImmediateTransport()
        model, _ = _legacy_model(transport)

        output = await model.agenerate("prompt", seed=7)

        assert output.texts == ["ok"]
        assert len(transport.requests) == 1
        assert transport.requests[0].sampling.seed == 7
        await model.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (DialectOptions("openai_chat", {"min_p": 0.1}), "target"),
            (
                DialectOptions("sglang_legacy", {"unknown_option": True}),
                "dialect_options.unknown_option",
            ),
            (
                DialectOptions("sglang_legacy", {"min_p": None}),
                "null values are not lowered",
            ),
        ],
    )
    async def test_dialect_options_are_strict(
        self, options: DialectOptions, message: str
    ) -> None:
        transport = _ImmediateTransport()
        model, _ = _legacy_model(transport)
        req = Request(input=CompletionInput("prompt"), dialect_options=options)

        with pytest.raises(RequestAuditError, match=message):
            await model.arun(req)

        assert transport.requests == []
        await model.aclose()

    @pytest.mark.anyio
    async def test_canonical_owned_dialect_option_is_always_rejected(self) -> None:
        transport = _ImmediateTransport()
        model, _ = _legacy_model(transport)
        req = Request(
            input=CompletionInput("prompt"),
            dialect_options=DialectOptions("sglang_legacy", {"max_tokens": 3}),
        )

        with pytest.raises(
            RequestAuditError,
            match=r"dialect_options\.max_tokens.*sampling\.max_tokens",
        ):
            await model.arun(req)

        assert transport.requests == []
        await model.aclose()

    @pytest.mark.anyio
    async def test_prefill_aliases_cannot_both_be_supplied(self) -> None:
        transport = _ImmediateTransport()
        model, _ = _legacy_model(transport)
        req = Request(
            input=CompletionInput("prompt"),
            dialect_options=DialectOptions(
                "sglang_legacy",
                {"prefill": "first", "prefix": "second"},
            ),
        )

        with pytest.raises(
            RequestAuditError,
            match=r"dialect_options\.prefix.*dialect_options\.prefill",
        ):
            await model.arun(req)

        assert transport.requests == []
        await model.aclose()

    @pytest.mark.anyio
    async def test_lowered_leaves_and_stream_noop_reach_transport(self) -> None:
        transport = _ImmediateTransport()
        model, _ = _legacy_model(transport)
        req = Request(
            input=CompletionInput("prompt"),
            sampling=SamplingParams(
                max_tokens=3,
                temperature=0.2,
                top_p=0.9,
                top_k=8,
                stop=("END",),
                frequency_penalty=0.1,
                presence_penalty=0.2,
                n=2,
            ),
            scheduling=SchedulingParams(stream=True),
            dialect_options=DialectOptions(
                "sglang_legacy", {"min_p": 0.05, "prefill": "prefix"}
            ),
        )
        scoring_req = Request(
            input=CompletionInput("score"),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
        )

        response = await model.arun(req)
        await model.arun(scoring_req)

        assert response.texts == ("ok",)
        assert transport.requests == [req, scoring_req]
        await model.aclose()


class TestLegacyLifecycle:
    @pytest.mark.anyio
    async def test_context_manager_returns_self_and_closes_pool(self):
        client = SimpleNamespace(
            base_url="http://host:8000/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models.sglang_gen_model.AsyncOpenAI",
            return_value=client,
        ):
            model = SglangGenModel(model="m", api_key="local")

        async with model as entered:
            assert entered is model

        assert model.pool.is_closed
        client.close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_close_via_child_invalidates_all_siblings_and_closes_once(self):
        client = SimpleNamespace(
            base_url="http://host:8000/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models.sglang_gen_model.AsyncOpenAI",
            return_value=client,
        ):
            root = SglangGenModel(model="m", api_key="local")
            child = root.with_args(temperature=0.5)
            sibling = root.with_args(top_p=0.9)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(child.aclose)
            task_group.start_soon(root.aclose)
            task_group.start_soon(sibling.aclose)

        assert root.pool.is_closing is True
        assert root.pool.is_closed is True
        client.close.assert_awaited_once()

        for model in (root, child, sibling):
            with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
                await model.__aenter__()
            with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
                await model.arun(Request(input=CompletionInput("prompt")))

    @pytest.mark.anyio
    async def test_close_drains_admitted_request_before_closing_client(self):
        transport = _StubTransport()
        client = SimpleNamespace(
            base_url="http://host:8000/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models.sglang_gen_model.AsyncOpenAI",
            return_value=client,
        ):
            root = SglangGenModel(
                model="m",
                api_key="local",
                transport=transport,  # type: ignore[arg-type]
            )
        child = root.with_args(temperature=0.5)

        async def close_and_observe() -> None:
            await child.aclose()
            client.close.assert_awaited_once()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                root.arun,
                Request(input=CompletionInput("prompt")),
            )
            await transport.entered.wait()
            task_group.start_soon(close_and_observe)
            while not root.pool.is_closing:
                await cast(Any, anyio.lowlevel).checkpoint()

            client.close.assert_not_awaited()
            with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
                await child.arun(Request(input=CompletionInput("new")))
            transport.release.set()

        assert transport.calls == 1
        assert root.pool.is_closed is True
