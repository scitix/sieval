"""Tests for the ``SglangGenModel`` compatibility wrapper.

The wrapper binds ``sglang_native`` over a private ``async_http_json`` pool,
the same way ``GenModel`` binds ``openai_completions``.  The wire lowering and
lifting are covered by ``tests/unit/core/models/dialects/test_sglang_native.py``;
these tests cover the binding, the request gate it inherits from ``Model``, the
pool lifecycle, and the legacy ``agenerate``/``alogprobs`` surface end to end
over an ``httpx.MockTransport``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import patch

import anyio
import httpx
import pytest

from sieval.core.models import (
    DEFAULT_REQUEST_TIMEOUT,
    Capability,
    CompletionInput,
    DialectOptions,
    HostedToolSpec,
    ModelOutput,
    OpaqueContinuation,
    ReasoningParams,
    Request,
    Response,
    RolloutLogprobs,
    SessionParams,
    SglangGenModel,
    ToolParams,
)
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.dialect import DialectError, RequestAuditError
from sieval.core.models.dialects.sglang_native import SglangNativeDialect
from sieval.core.utils.serialization import dict_to_obj, obj_to_dict

_BASE = "http://host:30000/v1"


class _Server:
    def __init__(self, *replies: object) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.replies.pop(0))

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def _model(server: Callable[[httpx.Request], httpx.Response], **kwargs: Any):
    """Build the wrapper with its HTTP client answered by *server*."""

    mock = httpx.MockTransport(server)
    with patch(
        "sieval.core.models.connection_factory.httpx.AsyncHTTPTransport",
        return_value=mock,
    ):
        return SglangGenModel(model="m", api_base=_BASE, api_key="local", **kwargs)


def _meta(**overrides: Any) -> dict:
    meta = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "cached_tokens": 0,
        "finish_reason": {"type": "stop"},
    }
    meta.update(overrides)
    return meta


class TestBinding:
    def test_binds_sglang_native_with_a_real_runtime_plan(self):
        model = SglangGenModel(model="m", api_base=_BASE, api_key="local")

        assert model.dialect_id == "sglang_native"
        assert model.runtime_plan is not None
        assert model.runtime_plan.dialect_id == "sglang_native"
        assert model.provenance_plan is not None
        assert model.deployment.engine.engine_id == "sglang"
        assert isinstance(model._dialect, SglangNativeDialect)
        assert isinstance(model.pool.connection, AsyncHTTPJSONConnection)
        assert model.pool.identity.connection_family == "async_http_json"
        assert Capability.InputScoring in model.capabilities

    def test_persisted_provenance_is_stable_across_instances(self):
        first = SglangGenModel(model="m", api_base=_BASE, api_key="first-secret")
        second = SglangGenModel(model="m", api_base=_BASE, api_key="second-secret")

        # Runtime pools never share quota; the persisted evidence is the same.
        assert first.pool.identity != second.pool.identity
        assert first.provenance_plan == second.provenance_plan
        assert "first-secret" not in repr(first.pool.identity)
        assert "first-secret" not in repr(first.provenance_plan)

    @pytest.mark.anyio
    async def test_reads_no_openai_api_key_from_the_environment(self, monkeypatch):
        # The AsyncOpenAI client this replaced sent $OPENAI_API_KEY when
        # api_key was None; async_http_json has no environment fallback.
        monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
        server = _Server({"text": "out", "meta_info": _meta()})
        mock = httpx.MockTransport(server)
        with patch(
            "sieval.core.models.connection_factory.httpx.AsyncHTTPTransport",
            return_value=mock,
        ):
            model = SglangGenModel(model="m", api_base=_BASE)

        await model.agenerate("hi")

        assert len(server.requests) == 1
        assert "Authorization" not in server.requests[0].headers

    @pytest.mark.parametrize("api_base", [None, "", "/"])
    def test_requires_an_explicit_api_base(self, api_base):
        with pytest.raises(ValueError, match="requires an explicit api_base"):
            SglangGenModel(model="m", api_base=api_base)

    def test_client_declares_the_shared_request_timeout(self):
        model = SglangGenModel(model="m", api_base=_BASE)

        connection = model.pool.connection
        assert isinstance(connection, AsyncHTTPJSONConnection)
        assert connection.client.timeout == DEFAULT_REQUEST_TIMEOUT

    def test_client_resolves_a_client_timeout_mapping(self):
        model = SglangGenModel(
            model="m", api_base=_BASE, client_timeout=cast(Any, {"connect": 30})
        )

        connection = model.pool.connection
        assert isinstance(connection, AsyncHTTPJSONConnection)
        assert connection.client.timeout == httpx.Timeout(600.0, connect=30.0)

    def test_input_coercion_is_completion_only(self):
        model = SglangGenModel(model="m", api_base=_BASE)
        prompt = CompletionInput("prompt")

        assert model._coerce_input(prompt) is prompt
        assert model._coerce_input("prompt") == prompt
        with pytest.raises(TypeError, match="text or CompletionInput"):
            model._coerce_input(3)

    def test_transport_hook_still_overrides_the_dialect(self):
        base = SglangGenModel(model="m", api_base=_BASE)
        replacement = SglangNativeDialect(base.pool.connection, "m")

        model = SglangGenModel(model="m", api_base=_BASE, transport=replacement)

        assert model._dialect is replacement


class TestRequestGate:
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
                            "sglang_native", "continuation"
                        )
                    ),
                ),
                "session.opaque_continuation",
            ),
        ],
    )
    async def test_unsupported_leaf_is_refused_before_io(self, req, path):
        server = _Server()
        model = _model(server)

        with pytest.raises(DialectError, match=path):
            await model.arun(req)

        assert server.requests == []
        await model.aclose()

    @pytest.mark.anyio
    async def test_legacy_builder_refuses_dropped_fields_before_io(self):
        server = _Server()
        model = _model(server)

        with pytest.raises(DialectError):
            await model.agenerate("prompt", reasoning_effort="high")

        assert server.requests == []
        await model.aclose()

    @pytest.mark.anyio
    async def test_a_seed_is_refused_before_io(self):
        server = _Server({"text": "ok", "meta_info": _meta()})
        model = _model(server)

        with pytest.raises(RequestAuditError, match="no per-request seed"):
            await model.agenerate("prompt", seed=7)

        assert server.requests == []
        await model.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (DialectOptions("openai_chat", {"min_p": 0.1}), "target"),
            (DialectOptions("sglang_native", {"unknown": True}), "unknown SGLang"),
            (DialectOptions("sglang_native", {"prefill": "A:"}), "no prefill"),
            (
                DialectOptions("sglang_native", {"max_tokens": 3}),
                r"sampling\.max_tokens",
            ),
        ],
    )
    async def test_dialect_options_are_strict(self, options, message):
        server = _Server()
        model = _model(server)
        req = Request(input=CompletionInput("prompt"), dialect_options=options)

        with pytest.raises((DialectError, RequestAuditError), match=message):
            await model.arun(req)

        assert server.requests == []
        await model.aclose()

    @pytest.mark.anyio
    async def test_a_sampling_option_is_forwarded(self):
        server = _Server({"text": "ok", "meta_info": _meta()})
        model = _model(server)

        await model.agenerate("prompt", extra_body={"min_p": 0.05})

        assert server.body["sampling_params"] == {"min_p": 0.05}
        await model.aclose()


class TestLifecycle:
    @pytest.mark.anyio
    async def test_context_manager_returns_self_and_closes_pool(self):
        model = SglangGenModel(model="m", api_base=_BASE)

        async with model as entered:
            assert entered is model

        assert model.pool.is_closed

    @pytest.mark.anyio
    async def test_close_via_child_invalidates_every_sibling(self):
        root = SglangGenModel(model="m", api_base=_BASE)
        child = root.with_args(temperature=0.5)
        sibling = root.with_args(top_p=0.9)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(child.aclose)
            task_group.start_soon(root.aclose)
            task_group.start_soon(sibling.aclose)

        assert root.pool.is_closed is True
        for model in (root, child, sibling):
            with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
                await model.arun(Request(input=CompletionInput("prompt")))

    @pytest.mark.anyio
    async def test_close_drains_an_admitted_request(self):
        entered = anyio.Event()
        release = anyio.Event()
        calls = 0

        class _Blocking(SglangNativeDialect):
            async def execute(self, prepared: Any) -> Response:
                nonlocal calls
                calls += 1
                entered.set()
                await release.wait()
                return Response(texts=("ok",))

        base = SglangGenModel(model="m", api_base=_BASE)
        root = SglangGenModel(
            model="m",
            api_base=_BASE,
            transport=_Blocking(base.pool.connection, "m"),
        )
        child = root.with_args(temperature=0.5)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(root.arun, Request(input=CompletionInput("p")))
            await entered.wait()
            task_group.start_soon(child.aclose)
            while not root.pool.is_closing:
                await cast(Any, anyio.lowlevel).checkpoint()
            assert not root.pool.is_closed
            with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
                await child.arun(Request(input=CompletionInput("new")))
            release.set()

        assert calls == 1
        assert root.pool.is_closed is True
        await base.aclose()


class TestLegacySurface:
    @pytest.mark.anyio
    async def test_alogprobs_echo_returns_input_top_logprobs(self):
        meta = _meta(
            prompt_tokens=2,
            finish_reason={"type": "length"},
            input_token_logprobs=[[None, 1, "Hi"], [-0.4, 2, "Ġthere"]],
            input_top_logprobs=[None, [[-0.4, 2, "Ġthere"], [-1.1, 3, "Ġall"]]],
            output_token_logprobs=[[-0.2, 4, "!"]],
            output_top_logprobs=[[[-0.2, 4, "!"], [-2.5, 5, "."]]],
        )
        server = _Server({"text": "", "meta_info": meta})
        model = _model(server)

        output = await model.alogprobs("Hi there", echo=True, logprobs=2)

        assert str(server.requests[0].url) == "http://host:30000/generate"
        assert server.body["logprob_start_len"] == 0
        assert server.body["top_logprobs_num"] == 2
        assert server.body["sampling_params"] == {
            "max_new_tokens": 1,
            "temperature": 0.0,
        }
        # The input segment leads, as before; the generated token follows.
        assert output.logprobs_tokens == ["Hi", " there", "!"]
        assert output.input_top_logprobs == [{}, {" there": -0.4, " all": -1.1}]
        assert output.top_logprobs == [{"!": -0.2, ".": -2.5}]
        await model.aclose()

    @pytest.mark.anyio
    async def test_alogprobs_echo_without_breadth_scores_the_input(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "a"], [-0.5, 2, "Ġb"]],
            output_token_logprobs=[[-0.1, 3, "c"]],
        )
        server = _Server({"text": "c", "meta_info": meta})
        model = _model(server)

        output = await model.alogprobs("a b", echo=True)

        assert output.logprobs_tokens == ["a", " b", "c"]
        assert output.logprobs == [None, -0.5, -0.1]
        await model.aclose()

    @pytest.mark.anyio
    async def test_agenerate_persists_one_logprob_sequence_per_rollout(self):
        replies = [
            {
                "text": text,
                "meta_info": _meta(
                    prompt_tokens=3,
                    output_token_logprobs=[[logprob, index, text]],
                    output_top_logprobs=[[[logprob, index, text]]],
                ),
            }
            for index, (text, logprob) in enumerate([("a", -0.1), ("b", -0.7)])
        ]
        server = _Server(replies)
        # The derivation a task's ``infer_args`` applies to the candidate.
        model = _model(server).with_args(return_logprobs=True, top_logprobs=20)

        output = await model.agenerate("prompt", n=2)

        assert output.texts == ["a", "b"]
        assert output.logprobs is None
        assert output.choice_logprobs == [
            RolloutLogprobs(
                logprobs_tokens=["a"], logprobs=[-0.1], top_logprobs=[{"a": -0.1}]
            ),
            RolloutLogprobs(
                logprobs_tokens=["b"], logprobs=[-0.7], top_logprobs=[{"b": -0.7}]
            ),
        ]
        assert server.body["top_logprobs_num"] == 20
        assert server.body["sampling_params"]["n"] == 2
        assert output.model["model"] == "m"

        restored = dict_to_obj(obj_to_dict(output, add_type=True), {})
        assert isinstance(restored, ModelOutput)
        assert restored.choice_logprobs == output.choice_logprobs
        await model.aclose()

    @pytest.mark.anyio
    async def test_provenance_records_the_native_dialect(self):
        server = _Server({"text": "ok", "meta_info": _meta()})
        model = _model(server)

        output = await model.agenerate("prompt")

        provenance = output.model.get("provenance")
        assert provenance is not None
        assert provenance.dialect_id == "sglang_native"
        assert provenance.engine_id == "sglang"
        # /generate reports no model id, and none is fabricated.
        assert provenance.model_identity.provider_reported_model_id is None
        await model.aclose()
