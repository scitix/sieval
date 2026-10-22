"""Context overflow: recognition, and what a model call returns inside a scope.

Real clients over an in-process wire, so each transport's actual error shape
(the OpenAI SDK's ``BadRequestError``, SGLang's native ``/generate`` 400) is
what gets classified.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Mapping
from typing import Any

import httpx
import openai
import pytest

from sieval.core.models import ChatModel, GenModel, SglangGenModel
from sieval.core.models.overflow import (
    CONTEXT_OVERFLOW,
    ContextOverflowError,
    OverflowScope,
    ReplayAmbiguous,
    context_overflow_message,
    is_recorded_context_overflow,
    overflow_scope,
)
from sieval.core.models.request_log import RecordingOpenAIClient
from sieval.core.types import JSONValue

#: SGLang's two refusals, as `TokenizerManager._validate_one_request` words them
#: for a 30B model served at a 40,960-token window.
SGLANG_INPUT = (
    "The input (41503 tokens) is longer than the model's context length (40960 tokens)."
)
SGLANG_TOTAL = (
    "Requested token count exceeds the model's maximum context length of 40960 "
    "tokens. You requested a total of 49152 tokens: 16384 tokens from the input "
    "messages and 32768 tokens for the completion. Please reduce the number of "
    "tokens in the input messages or the completion to fit within the limit."
)
VLLM = (
    "This model's maximum context length is 32768 tokens. However, you requested "
    "40000 tokens (7232 in the messages, 32768 in the completion). Please reduce "
    "the length of the messages or completion."
)


def _error(status: int, message: str) -> httpx.Response:
    # SGLang's `_create_error_response` and vLLM/OpenAI's error body.
    return httpx.Response(status, json={"error": {"message": message}})


def _chat_ok(text: str = "ok") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


class _Wire:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    handler = _Wire()
    real = openai.AsyncOpenAI

    def client(**kwargs: Any) -> openai.AsyncOpenAI:
        kwargs["http_client"] = RecordingOpenAIClient(
            timeout=kwargs["timeout"], transport=httpx.MockTransport(handler)
        )
        return real(**kwargs)

    monkeypatch.setattr("sieval.core.models._legacy_binding.AsyncOpenAI", client)
    monkeypatch.setattr("sieval.core.models.sglang_gen_model.AsyncOpenAI", client)
    return handler


def _chat() -> ChatModel:
    return ChatModel(
        model="m", api_base="http://engine/v1", api_key="k", max_retries=0, stream=False
    )


def _sglang() -> SglangGenModel:
    return SglangGenModel(
        model="m", api_base="http://engine/v1", api_key="k", max_retries=0
    )


def _completions() -> GenModel:
    return GenModel(
        model="m", api_base="http://engine/v1", api_key="k", max_retries=0, stream=False
    )


class TestRecognition:
    @pytest.mark.parametrize("message", [SGLANG_INPUT, SGLANG_TOTAL, VLLM])
    @pytest.mark.parametrize("status", [400, 413])
    def test_a_refusal_of_the_prompt_is_an_overflow(self, status, message):
        response = _error(status, message)
        response.request = httpx.Request("POST", "http://engine/v1/x")
        exc = httpx.HTTPStatusError("bad", request=response.request, response=response)
        assert context_overflow_message(exc) is not None

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_the_same_words_under_another_status_are_not(self, status):
        response = _error(status, SGLANG_TOTAL)
        response.request = httpx.Request("POST", "http://engine/v1/x")
        exc = httpx.HTTPStatusError("bad", request=response.request, response=response)
        assert context_overflow_message(exc) is None

    def test_a_transport_failure_is_never_an_overflow(self):
        request = httpx.Request("POST", "http://engine/v1/x")
        assert (
            context_overflow_message(httpx.ReadTimeout(VLLM, request=request)) is None
        )
        assert context_overflow_message(httpx.ConnectError(VLLM)) is None
        assert context_overflow_message(TimeoutError(VLLM)) is None

    def test_another_400_is_not(self):
        response = _error(400, "temperature must be non-negative")
        response.request = httpx.Request("POST", "http://engine/v1/x")
        exc = httpx.HTTPStatusError("bad", request=response.request, response=response)
        assert context_overflow_message(exc) is None

    def test_a_streamed_error_without_status_matches_on_its_text(self):
        # The SDK raises a bare APIError for an error event inside a 200 stream.
        exc = openai.APIError(
            SGLANG_INPUT, request=httpx.Request("POST", "http://e"), body=None
        )
        assert context_overflow_message(exc) == SGLANG_INPUT

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("exception::BadRequestError", True),
            ("exception::RuntimeError", True),
            ("exception::APITimeoutError", False),
            ("exception::InternalServerError", False),
            ("non_retriable", False),
            ("retry_limit", False),
        ],
    )
    def test_a_recorded_failure_is_read_by_name_and_message(self, reason, expected):
        message = f"Error code: 400 - {{'error': {{'message': {SGLANG_TOTAL!r}}}}}"
        assert is_recorded_context_overflow(reason, message) is expected


class TestOutsideAScope:
    @pytest.mark.anyio
    async def test_openai_chat_raises_the_sdk_error_unchanged(self, wire):
        wire.responses = [_error(400, VLLM)]
        with pytest.raises(openai.BadRequestError):
            await _chat().agenerate("hi")

    @pytest.mark.anyio
    async def test_sglang_raises_the_sdk_error_unchanged(self, wire):
        wire.responses = [_error(400, SGLANG_INPUT)]
        with pytest.raises(openai.BadRequestError):
            await _sglang().agenerate("hi")

    @pytest.mark.anyio
    async def test_context_overflow_output_raises_outside_a_scope(self, wire):
        with pytest.raises(ContextOverflowError, match="too long"):
            _chat().context_overflow_output("hi", "prompt is too long")
        assert wire.requests == []


@pytest.mark.parametrize(
    ("build", "path", "message"),
    [
        (_chat, "/v1/chat/completions", VLLM),
        (_completions, "/v1/completions", VLLM),
        (_sglang, "/generate", SGLANG_INPUT),
        (_sglang, "/generate", SGLANG_TOTAL),
    ],
)
class TestEachTransport:
    @pytest.mark.anyio
    async def test_observing_raises_and_remembers(self, wire, build, path, message):
        wire.responses = [_error(400, message)]
        model = build()
        scope = OverflowScope(model)
        with overflow_scope(scope), pytest.raises(openai.BadRequestError) as caught:
            await model.agenerate("hi")
        assert wire.requests[0].url.path == path
        escaped = scope.escaping(caught.value)
        assert escaped is not None
        assert message in escaped

    @pytest.mark.anyio
    async def test_converting_returns_an_empty_answer(self, wire, build, path, message):
        wire.responses = [_error(400, message)]
        model = build()
        scope = OverflowScope(model, convert=True)
        with overflow_scope(scope):
            output = await model.agenerate("hi", n=3, max_tokens=32768)
        assert output.texts == ["", "", ""]
        assert output.finish_reasons == [CONTEXT_OVERFLOW] * 3
        assert output.request_params is not None
        assert output.request_params[CONTEXT_OVERFLOW] is True
        assert output.request_params["max_tokens"] == 32768
        assert output.usage is None
        assert len(scope.messages) == 1 and message in scope.messages[0]


class TestOtherErrorsStillRaise:
    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [400, 404, 500, 503])
    async def test_inside_a_converting_scope(self, wire, status):
        wire.responses = [_error(status, "something else broke")]
        model = _chat()
        with (
            overflow_scope(OverflowScope(model, convert=True)),
            pytest.raises(openai.APIStatusError),
        ):
            await model.agenerate("hi")

    @pytest.mark.anyio
    async def test_a_5xx_with_overflow_words(self, wire):
        wire.responses = [_error(500, VLLM)]
        model = _chat()
        with (
            overflow_scope(OverflowScope(model, convert=True)),
            pytest.raises(openai.InternalServerError),
        ):
            await model.agenerate("hi")

    @pytest.mark.anyio
    async def test_a_judge_overflow_raises(self, wire):
        wire.responses = [_error(400, VLLM)]
        subject, judge = _chat(), _chat()
        with (
            overflow_scope(OverflowScope(subject, convert=True)),
            pytest.raises(openai.BadRequestError),
        ):
            await judge.agenerate("judge this")

    @pytest.mark.anyio
    async def test_alogprobs_fails_as_an_overflow(self, wire):
        wire.responses = [_error(400, SGLANG_INPUT)]
        model = _sglang()
        with pytest.raises(ContextOverflowError) as caught:
            await model.alogprobs("hi")
        assert isinstance(caught.value.__cause__, openai.BadRequestError)


class TestReplay:
    @pytest.mark.anyio
    async def test_known_calls_are_answered_without_a_request(self, wire):
        wire.responses = [_chat_ok("turn 1"), _error(400, VLLM)]
        model = _chat()
        observed = OverflowScope(model)
        with overflow_scope(observed), pytest.raises(openai.BadRequestError):
            await model.agenerate("turn 1")
            await model.agenerate("turn 2")
        assert len(wire.requests) == 2

        scope = OverflowScope(model, convert=True, known=observed.calls)
        with overflow_scope(scope):
            first = await model.agenerate("turn 1")
            second = await model.agenerate("turn 2")
        assert len(wire.requests) == 2, "nothing is asked twice"
        assert first.texts == ["turn 1"]
        assert second.finish_reasons == [CONTEXT_OVERFLOW]

    @pytest.mark.anyio
    async def test_a_changed_request_ends_the_replay(self, wire):
        wire.responses = [_chat_ok("a"), _chat_ok("b")]
        model = _chat()
        observed = OverflowScope(model)
        with overflow_scope(observed):
            await model.agenerate("one")
        scope = OverflowScope(model, convert=True, known=observed.calls)
        with overflow_scope(scope):
            output = await model.agenerate("another")
        assert output.texts == ["b"]
        assert len(wire.requests) == 2

    @pytest.mark.anyio
    async def test_a_recorded_refusal_answers_the_first_call(self, wire):
        model = _chat()
        scope = OverflowScope(model, convert=True, replay=VLLM)
        with overflow_scope(scope):
            output = await model.agenerate("hi")
        assert output.finish_reasons == [CONTEXT_OVERFLOW]
        assert scope.messages == [VLLM]
        assert wire.requests == []

    @pytest.mark.anyio
    async def test_a_second_call_after_a_recorded_refusal_is_ambiguous(self, wire):
        model = _chat()
        scope = OverflowScope(model, convert=True, replay=VLLM)
        with overflow_scope(scope):
            await model.agenerate("hi")
            with pytest.raises(ReplayAmbiguous):
                await model.agenerate("again")
        assert scope.ambiguous
        assert wire.requests == []

    @pytest.mark.anyio
    async def test_a_task_measured_overflow_is_recorded_without_a_request(self, wire):
        model = _chat()
        scope = OverflowScope(model, convert=True)
        with overflow_scope(scope):
            output = model.context_overflow_output("hi", "prompt is too long", n=2)
        assert output.texts == ["", ""]
        assert scope.messages == ["prompt is too long"]
        assert wire.requests == []


class TestRequestRecord:
    @pytest.mark.anyio
    async def test_the_refused_attempt_is_labelled_context_overflow(self, wire):
        from sieval.core.models.request_log import StageScope, stage_scope

        lines: list[dict] = []

        class Sink:
            def role_of(self, model_meta: Mapping[str, object]) -> str:
                return "subject"

            def record(self, line: dict[str, JSONValue]) -> None:
                lines.append(json.loads(json.dumps(line)))

        scope = StageScope(
            sink=Sink(),
            task="t",
            run="r",
            stage="infer",
            sample_id=0,
            iteration=0,
            repeat_index=None,
            limiters={},
            run_start=0.0,
        )
        wire.responses = [_error(400, SGLANG_TOTAL), _error(503, "busy")]
        model = _sglang()
        with stage_scope(scope):
            with overflow_scope(OverflowScope(model, convert=True)):
                await model.agenerate("hi")
            with pytest.raises(openai.InternalServerError):
                await model.agenerate("hi")
        overflow, busy = lines
        assert overflow["status"] == 400
        assert overflow["outcome"] == CONTEXT_OVERFLOW
        assert overflow["error"] == "exception::BadRequestError"
        assert busy["status"] == 503
        assert busy["outcome"] == "5xx"
