"""Tests for the IR primitive on Model: arun, capabilities, assert_capability,
and the alogprobs echo gate.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import dataclasses
from typing import Any, cast

import pytest

from sieval.core.models import (
    Capability,
    CapabilityError,
    ChatInput,
    ChatMessage,
    ChatModel,
    CompletionInput,
    GenModel,
    OpaqueContinuation,
    ReasoningOutput,
    Request,
    Response,
    ScoringParams,
    SessionParams,
    TextPart,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.models.dialect import DialectError, OutputContractError
from sieval.core.models.reconcile import CheckStage, DeferredCheck


def _chat_input(text: str) -> ChatInput:
    return ChatInput((ChatMessage("user", (TextPart(text),)),))


class TestCapabilities:
    def test_chat_model_capabilities_from_transport(self):
        m = ChatModel(model="c", api_key="k")
        assert Capability.Chat in m.capabilities
        assert Capability.InputScoring not in m.capabilities

    def test_gen_model_has_input_scoring(self):
        m = GenModel(model="g", api_key="k")
        assert Capability.InputScoring in m.capabilities
        assert Capability.Completion in m.capabilities

    def test_assert_capability_passes_when_present(self):
        m = GenModel(model="g", api_key="k")
        m.assert_capability(Capability.Completion, Capability.InputScoring)

    def test_assert_capability_raises_when_missing(self):
        m = ChatModel(model="c", api_key="k")
        with pytest.raises(CapabilityError, match="InputScoring"):
            m.assert_capability(Capability.InputScoring)


class TestArun:
    def test_handler_transport_uses_registered_connection_family(self):
        from tests.conftest import HandlerTransport

        async def handler(req: Request) -> Response:
            del req
            return Response(texts=("ok",))

        native = HandlerTransport(handler, "anthropic_messages")
        assert native.connection_family == "async_http_json"

        with pytest.raises(ValueError, match="unknown dialect"):
            HandlerTransport(handler, "unregistered")

    @pytest.mark.anyio
    async def test_arun_requires_request_instance(self):
        model = GenModel(model="g", api_key="k")

        with pytest.raises(TypeError, match="arun requires Request"):
            await model.arun(cast(Any, "prompt"))

    @pytest.mark.anyio
    async def test_arun_delegates_to_transport(self):
        from tests.conftest import HandlerTransport

        resp = Response(
            texts=("hi",), usage=UsageStats(output_tokens=1, total_tokens=1)
        )

        async def handler(req: Request) -> Response:
            assert req.input == _chat_input("prompt")
            return resp

        stub = HandlerTransport(handler, "openai_chat")
        m = ChatModel(model="c", api_key="k", transport=stub)
        out = await m.arun(Request(input=_chat_input("prompt")))
        assert out.texts == resp.texts
        assert out.provenance is not None
        assert len(stub.requests) == 1
        assert stub.requests[0].input == _chat_input("prompt")

    @pytest.mark.anyio
    async def test_injected_transport_supplies_capabilities(self):
        from tests.conftest import HandlerTransport

        async def handler(req: Request) -> Response:
            del req
            return Response(texts=())

        stub = HandlerTransport(handler, "openai_chat")
        m = ChatModel(model="c", api_key="k", transport=stub)
        # Capability evidence comes from the reconciled runtime plan, not from
        # an injected execution double.
        assert Capability.Chat in m.capabilities
        assert Capability.InputScoring not in m.capabilities

    @pytest.mark.anyio
    async def test_aggregated_scoring_plan_does_not_pollute_plain_generation(self):
        from tests.conftest import HandlerTransport

        async def handler(req: Request) -> Response:
            if req.scoring.top_logprobs > 0:
                token = TokenLogprob("x", -0.1)
                return Response(
                    texts=("ok",),
                    logprobs=(token,),
                    top_logprobs=((TopKEntry("x", -0.1),),),
                )
            return Response(texts=("ok",))

        stub = HandlerTransport(handler, "openai_completions")
        model = GenModel(model="g", api_key="k", transport=stub)
        model._runtime_plan = dataclasses.replace(
            model._runtime_plan,
            capability_minimums={"top_logprobs": {"minimum": 3}},
            required_output_channels=frozenset(
                {"input_scoring", "logprobs", "top_logprobs"}
            ),
            request_checks=(
                DeferredCheck(
                    "input_scoring",
                    CheckStage.REQUEST,
                    "validate_response_channel",
                    "aggregated input-scoring requirement",
                ),
                DeferredCheck(
                    "top_logprobs",
                    CheckStage.REQUEST,
                    "validate_response_channel",
                    "aggregated top-logprobs requirement",
                ),
            ),
        )

        ordinary = await model.arun(Request(input=CompletionInput("prompt")))
        assert ordinary.texts == ("ok",)

        scored = await model.arun(
            Request(
                input=CompletionInput("prompt"),
                scoring=ScoringParams(
                    sampled_logprobs=True,
                    top_logprobs=3,
                ),
            )
        )
        assert scored.top_logprobs == ((TopKEntry("x", -0.1),),)

    @pytest.mark.anyio
    async def test_inactive_unknown_request_check_does_not_block_plain_generation(
        self,
    ):
        from tests.conftest import HandlerTransport

        async def handler(req: Request) -> Response:
            del req
            return Response(texts=("ok",))

        stub = HandlerTransport(handler, "openai_completions")
        model = GenModel(model="g", api_key="k", transport=stub)
        model._runtime_plan = dataclasses.replace(
            model._runtime_plan,
            request_checks=(
                DeferredCheck(
                    "input_scoring",
                    CheckStage.REQUEST,
                    "future_input_scoring_verifier",
                    "only relevant to scoring calls",
                ),
            ),
        )

        await model.arun(Request(input=CompletionInput("prompt")))
        assert len(stub.requests) == 1

        with pytest.raises(DialectError, match="unknown request-check verifier"):
            await model.arun(
                Request(
                    input=CompletionInput("prompt"),
                    scoring=ScoringParams(input_scoring=True),
                )
            )
        assert len(stub.requests) == 1

    @pytest.mark.anyio
    async def test_opaque_continuation_requires_roundtrip_payload(self):
        from tests.conftest import HandlerTransport

        async def handler(req: Request) -> Response:
            del req
            return Response(texts=("ok",), reasoning=(ReasoningOutput(),))

        stub = HandlerTransport(handler, "openai_chat")
        model = ChatModel(model="c", api_key="k", transport=stub)
        model._runtime_plan = dataclasses.replace(
            model._runtime_plan,
            available_capabilities=(
                model._runtime_plan.available_capabilities | {"opaque_continuation"}
            ),
        )
        req = Request(
            input=_chat_input("prompt"),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("openai_chat", "previous")
            ),
        )

        with pytest.raises(OutputContractError, match="opaque round-trip payload"):
            await model.arun(req)
        assert len(stub.requests) == 1

    @pytest.mark.anyio
    async def test_opaque_continuation_accepts_roundtrip_payload(self):
        from tests.conftest import HandlerTransport

        async def handler(req: Request) -> Response:
            del req
            return Response(
                texts=("ok",),
                reasoning=(ReasoningOutput(opaque_roundtrip="next"),),
            )

        stub = HandlerTransport(handler, "openai_chat")
        model = ChatModel(model="c", api_key="k", transport=stub)
        model._runtime_plan = dataclasses.replace(
            model._runtime_plan,
            available_capabilities=(
                model._runtime_plan.available_capabilities | {"opaque_continuation"}
            ),
        )
        req = Request(
            input=_chat_input("prompt"),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("openai_chat", "previous")
            ),
        )

        response = await model.arun(req)

        assert response.reasoning == (ReasoningOutput(opaque_roundtrip="next"),)
        assert len(stub.requests) == 1

    def test_request_check_dispatch_rejects_unknown_verifier_and_channel(self):
        model = GenModel(model="g", api_key="k")
        request = Request(input=CompletionInput("prompt"))

        model._runtime_plan = dataclasses.replace(
            model._runtime_plan,
            request_checks=(
                DeferredCheck(
                    "input_scoring",
                    CheckStage.REQUEST,
                    "future_verifier",
                    "not registered",
                ),
            ),
        )
        with pytest.raises(DialectError, match="unknown request-check verifier"):
            model._run_request_checks(
                request,
                Response(texts=("ok",)),
                frozenset({"input_scoring"}),
            )

        model._runtime_plan = dataclasses.replace(
            model._runtime_plan,
            request_checks=(
                DeferredCheck(
                    "fim",
                    CheckStage.REQUEST,
                    "validate_response_channel",
                    "no response channel",
                ),
            ),
        )
        with pytest.raises(DialectError, match="has no channel"):
            model._run_request_checks(
                request,
                Response(texts=("ok",)),
                frozenset({"fim"}),
            )

    def test_request_check_requires_present_channel_and_accepts_available_candidates(
        self,
    ):
        model = GenModel(model="g", api_key="k")
        request = Request(
            input=CompletionInput("prompt"),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
        )
        check = DeferredCheck(
            "top_logprobs",
            CheckStage.REQUEST,
            "validate_response_channel",
            "must be present",
        )
        model._runtime_plan = dataclasses.replace(
            model._runtime_plan,
            capability_minimums={"top_logprobs": {"minimum": 2}},
            request_checks=(check,),
        )

        with pytest.raises(DialectError, match="top_logprobs.*absent"):
            model._run_request_checks(
                request,
                Response(texts=("ok",)),
                frozenset({"top_logprobs"}),
            )

        response = Response(
            texts=("ok",),
            top_logprobs=(
                (
                    TopKEntry("a", -0.1),
                    TopKEntry("b", -0.2),
                ),
            ),
        )
        model._run_request_checks(
            request,
            response,
            frozenset({"top_logprobs"}),
        )

        # A response cannot prove whether a short position was provider
        # truncation or constrained decoding with fewer available candidates.
        # Consumers such as CLP tasks validate the concrete alternatives they
        # need; this generic postcondition only guarantees channel presence.
        short_response = Response(
            texts=("ok",),
            top_logprobs=((TopKEntry("a", -0.1),),),
        )
        model._run_request_checks(
            request,
            short_response,
            frozenset({"top_logprobs"}),
        )


class TestAlogprobsEchoGate:
    @pytest.mark.anyio
    async def test_echo_true_on_chat_raises_capability_error(self):
        """The historical bug: echo=True on chat was silently ignored."""
        from tests.conftest import MockChatModel

        m = MockChatModel()
        with pytest.raises(CapabilityError, match="InputScoring"):
            await m.alogprobs("prompt", echo=True)

    @pytest.mark.anyio
    async def test_echo_true_on_gen_works(self):
        from tests.conftest import MockGenModel

        m = MockGenModel()
        out = await m.alogprobs("prompt", echo=True)
        assert out.logprobs is not None

    @pytest.mark.anyio
    async def test_echo_false_on_chat_skips_the_gate(self):
        from tests.conftest import HandlerTransport, MockChatModel

        m = MockChatModel()
        # echo=False must skip the InputScoring gate and reach the transport.
        # The chat stub serves no logprob channel, so the post-arun contract
        # check fires — proving the gate was bypassed and the request ran.
        with pytest.raises(OutputContractError, match="logprobs.*absent"):
            await m.alogprobs("prompt", echo=False)
        assert isinstance(m._transport, HandlerTransport)
        assert m._transport.requests[0].scoring.input_scoring is False
