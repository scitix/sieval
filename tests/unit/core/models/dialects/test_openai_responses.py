"""Contract tests for the OpenAI Responses API dialect.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response as SDKResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseTextDeltaEvent,
)

from sieval.core.models.capabilities import (
    Capability,
    HostedToolsOptions,
    MultimodalInputOptions,
    ReasoningOptions,
    Supported,
    TopLogprobsOptions,
)
from sieval.core.models.deployment import BINDING_RESOURCE_KEYS
from sieval.core.models.dialect import (
    DialectError,
    OutputContractError,
    PreparedRequest,
    Rejected,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
    validate_runtime_binding_plan,
)
from sieval.core.models.dialects import openai_responses as openai_responses_module
from sieval.core.models.dialects.openai_responses import (
    CAPABILITY_DECISIONS,
    OUTPUT_CONTRACT,
    OpenAIResponsesDialect,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    CompletionInput,
    DialectOptions,
    FunctionToolCall,
    HostedToolSpec,
    ImagePart,
    OpaqueContinuation,
    ReasoningOutput,
    ReasoningParams,
    Request,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    ServerToolUse,
    SessionParams,
    StructuredOutputParams,
    TextPart,
    TokenLogprob,
    ToolCallPart,
    ToolParams,
    ToolResultPart,
    TopKEntry,
    UsageStats,
)
from sieval.core.types import JSONValue


class _AsyncItems:
    def __init__(self, items: Iterable[object]):
        self._items = iter(items)

    def __aiter__(self) -> "_AsyncItems":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _chat(*messages: ChatMessage) -> ChatInput:
    if not messages:
        messages = (ChatMessage("user", (TextPart("hello"),)),)
    return ChatInput(messages)


def _text(
    value: str = "ok",
    *,
    annotations: list[object] | None = None,
    logprobs: list[object] | None = None,
) -> object:
    return SimpleNamespace(
        type="output_text",
        text=value,
        annotations=[] if annotations is None else annotations,
        logprobs=logprobs,
    )


def _message(*content: object, status: object = "completed") -> object:
    if not content:
        content = (_text(),)
    return SimpleNamespace(
        type="message",
        id="msg_1",
        role="assistant",
        status=status,
        content=list(content),
    )


def _reasoning(
    item_id: str = "rs_1",
    *,
    summary: str | None = "summary",
    content: str | None = None,
    encrypted: str | None = "opaque",
    status: object = "completed",
) -> object:
    summaries = (
        [] if summary is None else [SimpleNamespace(type="summary_text", text=summary)]
    )
    contents = (
        None
        if content is None
        else [SimpleNamespace(type="reasoning_text", text=content)]
    )
    return SimpleNamespace(
        type="reasoning",
        id=item_id,
        summary=summaries,
        content=contents,
        encrypted_content=encrypted,
        status=status,
    )


def _usage(
    input_tokens: object = 2,
    output_tokens: object = 3,
    total_tokens: object = 5,
    *,
    cached_tokens: object = 1,
    reasoning_tokens: object = 2,
) -> object:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=SimpleNamespace(
            cached_tokens=cached_tokens,
            cache_write_tokens=17,
        ),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


def _response(
    *output: object,
    status: str = "completed",
    response_id: object = "resp_1",
    incomplete_reason: object | None = None,
    error: object | None = None,
    usage: object | None = None,
    model: object | None = "provider/model",
    reasoning_effort: object | None = "high",
) -> SimpleNamespace:
    if not output:
        output = (_message(),)
    details = (
        None if incomplete_reason is None else SimpleNamespace(reason=incomplete_reason)
    )
    return SimpleNamespace(
        id=response_id,
        status=status,
        output=list(output),
        incomplete_details=details,
        error=error,
        usage=usage,
        model=model,
        reasoning=SimpleNamespace(effort=reasoning_effort),
    )


def _dialect(
    response: object | None = None,
    *,
    requested_model_id: str = "requested/model",
) -> tuple[OpenAIResponsesDialect, AsyncMock]:
    client = MagicMock()
    create = AsyncMock(return_value=_response() if response is None else response)
    client.responses.create = create
    return OpenAIResponsesDialect(client, requested_model_id), create


def _prepare_for_audit(
    dialect: OpenAIResponsesDialect,
    request: Request,
) -> tuple[RequestAudit, PreparedRequest]:
    audit = RequestAudit(active_request_leaves(request))
    dialect.validate_request(request, audit, SimpleNamespace())
    audit.raise_rejections()
    return audit, dialect.prepare(request, audit)


def _awaited_kwargs(create: AsyncMock) -> dict[str, Any]:
    create.assert_awaited()
    call = create.await_args
    assert call is not None
    return dict(call.kwargs)


def _logprob() -> object:
    return SimpleNamespace(
        token="A",
        bytes=[65],
        logprob=-0.1,
        top_logprobs=[SimpleNamespace(token="B", bytes=[66], logprob=-2.0)],
    )


def _sdk_response_payload() -> dict[str, object]:
    """Minimal response accepted by the repository's locked OpenAI SDK."""

    return {
        "id": "resp_sdk",
        "object": "response",
        "created_at": 1.0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": "provider/model",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_sdk",
                "summary": [{"type": "summary_text", "text": "why"}],
                "content": None,
                "encrypted_content": "encrypted",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_sdk",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "sdk output",
                        "annotations": [],
                        "logprobs": None,
                    }
                ],
            },
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "background": False,
        "conversation": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": None,
        "reasoning": {"effort": "high", "summary": "detailed"},
        "safety_identifier": None,
        "service_tier": "default",
        "text": {"format": {"type": "text"}},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": None,
        "user": None,
    }


def _sdk_completed_event(
    *,
    sequence_number: int = 1,
    response: dict[str, object] | None = None,
) -> ResponseCompletedEvent:
    return ResponseCompletedEvent.model_validate(
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": _sdk_response_payload() if response is None else response,
        }
    )


def _sdk_incomplete_response_payload(reason: str) -> dict[str, object]:
    response = _sdk_response_payload()
    response["status"] = "incomplete"
    response["incomplete_details"] = {"reason": reason}
    return response


def _sdk_failed_event() -> ResponseFailedEvent:
    response = _sdk_response_payload()
    response["status"] = "failed"
    response["error"] = {"code": "server_error", "message": "boom"}
    response["output"] = []
    return ResponseFailedEvent.model_validate(
        {
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        }
    )


def _sdk_error_event() -> ResponseErrorEvent:
    return ResponseErrorEvent.model_validate(
        {
            "type": "error",
            "sequence_number": 1,
            "code": "server_error",
            "message": "bad stream",
            "param": None,
        }
    )


def _opaque_reasoning_continuation(**overrides: object) -> str:
    item: dict[str, object] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque",
    }
    item.update(overrides)
    return json.dumps({"version": 2, "items": [item]}, separators=(",", ":"))


def _sdk_web_search_response(status: str) -> SDKResponse:
    raw = _sdk_response_payload()
    raw["output"] = [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": status,
            "action": {
                "type": "open_page",
                "url": "https://example.test/source",
            },
        }
    ]
    return SDKResponse.model_validate(raw)


_RESPONSES_IR_OWNED_OPTIONS = (
    "echo",
    "extra_body",
    "frequency_penalty",
    "function_call",
    "functions",
    "include",
    "input",
    "instructions",
    "logprobs",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
    "messages",
    "model",
    "n",
    "opaque_continuation",
    "parallel_tool_calls",
    "presence_penalty",
    "previous_response_id",
    "prompt",
    "reasoning",
    "reasoning_effort",
    "response_format",
    "return_logprobs",
    "score_input",
    "seed",
    "server_tools",
    "session_id",
    "stop",
    "stream",
    "stream_options",
    "suffix",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p",
    "web_search_options",
)


class TestWireTranslation:
    @pytest.mark.anyio
    async def test_mapping_response_and_string_tool_result_need_no_sdk_shim(
        self,
    ) -> None:
        raw: Mapping[str, object] = {
            "id": "resp_mapping",
            "status": "completed",
            "error": None,
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "mapped",
                            "annotations": [],
                            "logprobs": None,
                        }
                    ],
                }
            ],
            "usage": None,
            "model": None,
            "reasoning": None,
        }
        dialect, create = _dialect(raw)

        response = await dialect.arun(
            Request(
                input=_chat(
                    ChatMessage(
                        "tool",
                        (ToolResultPart("call_1", "plain result"),),
                    )
                )
            )
        )

        assert response.texts == ("mapped",)
        assert _awaited_kwargs(create)["input"] == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "plain result",
            }
        ]

    @pytest.mark.anyio
    async def test_url_image_and_call_only_history_preserve_their_wire_shapes(
        self,
    ) -> None:
        dialect, create = _dialect()

        await dialect.arun(
            Request(
                input=_chat(
                    ChatMessage(
                        "user",
                        (
                            ImagePart(
                                url="https://example.test/image.png",
                                detail="low",
                            ),
                        ),
                    ),
                    ChatMessage(
                        "assistant",
                        (ToolCallPart("call_1", "lookup", "{}"),),
                    ),
                )
            )
        )

        assert _awaited_kwargs(create)["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://example.test/image.png",
                        "detail": "low",
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
        ]

    @pytest.mark.anyio
    async def test_real_sdk_transmits_current_original_image_detail(self) -> None:
        requests: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            requests.append(payload)
            return httpx.Response(200, json=_sdk_response_payload(), request=request)

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        try:
            dialect = OpenAIResponsesDialect(client, "requested/model")
            await dialect.arun(
                Request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (
                                ImagePart(
                                    url="https://example.test/image.png",
                                    detail="original",
                                ),
                            ),
                        )
                    )
                )
            )
        finally:
            await client.close()

        assert requests[0]["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://example.test/image.png",
                        "detail": "original",
                    }
                ],
            }
        ]
        assert "include" not in requests[0]

    @pytest.mark.anyio
    async def test_interleaved_assistant_content_preserves_ir_order(self) -> None:
        dialect, create = _dialect()

        await dialect.arun(
            Request(
                input=_chat(
                    ChatMessage(
                        "assistant",
                        (
                            ToolCallPart("call_1", "first", {}),
                            TextPart("middle"),
                            ToolCallPart("call_2", "second", {"value": 2}),
                            TextPart("tail"),
                        ),
                    )
                )
            )
        )

        assert _awaited_kwargs(create)["input"] == [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "first",
                "arguments": "{}",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "middle",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "second",
                "arguments": '{"value":2}',
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "tail",
            },
        ]

    @pytest.mark.anyio
    async def test_complete_request_preserves_responses_wire_shape(self) -> None:
        raw = _sdk_response_payload()
        raw["id"] = "resp_1"
        raw["output"] = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "summary"}],
                "content": None,
                "encrypted_content": "opaque",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"answer":42}',
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": 0,
                                "end_index": 13,
                                "url": "https://example.test/source",
                                "title": "Source",
                            }
                        ],
                        "logprobs": [
                            {
                                "token": "A",
                                "bytes": [65],
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {
                                        "token": "B",
                                        "bytes": [66],
                                        "logprob": -2.0,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_out",
                "name": "weather",
                "arguments": '{"city":"Paris"}',
                "status": "completed",
            },
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "weather Paris",
                    "sources": [
                        {
                            "type": "url",
                            "url": "https://example.test/source",
                        }
                    ],
                },
            },
        ]
        raw["usage"] = {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "input_tokens_details": {
                "cached_tokens": 1,
                "cache_write_tokens": 17,
            },
            "output_tokens_details": {"reasoning_tokens": 2},
        }
        dialect, create = _dialect(SDKResponse.model_validate(raw))
        function: Mapping[str, JSONValue] = {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
        request = Request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        TextPart("question"),
                        ImagePart(data="YWJj", media_type="image/png", detail="high"),
                    ),
                ),
                ChatMessage(
                    "assistant",
                    (
                        TextPart("checking"),
                        ToolCallPart("call_1", "weather", {"city": "Paris"}),
                    ),
                ),
                ChatMessage(
                    "tool",
                    (ToolResultPart("call_1", {"temperature": 20}),),
                ),
            ),
            sampling=SamplingParams(max_tokens=64, temperature=0.2, top_p=0.9),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
            reasoning=ReasoningParams(effort="high", summary="detailed"),
            tools=ToolParams(
                functions=(function,),
                choice={"type": "function", "function": {"name": "weather"}},
                parallel=False,
                hosted=(
                    HostedToolSpec(
                        "web_search_preview", {"search_context_size": "medium"}
                    ),
                ),
            ),
            structured_output=StructuredOutputParams(
                format="json_schema",
                schema={"type": "object"},
                name="answer",
                strict=True,
            ),
            session=SessionParams(previous_response_id="resp_previous"),
            dialect_options=DialectOptions(
                "openai_responses", {"store": True, "service_tier": "flex"}
            ),
        )

        response = await dialect.arun(request)

        create.assert_awaited_once_with(
            model="requested/model",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "question"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,YWJj",
                            "detail": "high",
                        },
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "checking",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"temperature":20}',
                },
            ],
            stream=False,
            max_output_tokens=64,
            temperature=0.2,
            top_p=0.9,
            top_logprobs=2,
            reasoning={"effort": "high", "summary": "detailed"},
            tools=[
                {
                    "type": "function",
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
                {
                    "type": "web_search_preview",
                    "search_context_size": "medium",
                },
            ],
            tool_choice={"type": "function", "name": "weather"},
            parallel_tool_calls=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
            previous_response_id="resp_previous",
            include=[
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
                "web_search_call.action.sources",
            ],
            extra_body={"store": True, "service_tier": "flex"},
        )
        assert response.texts == ('{"answer":42}',)
        assert response.finish_reasons == ("completed",)
        assert response.reasoning is not None
        assert response.reasoning[0] is not None
        assert response.reasoning == (
            ReasoningOutput(
                text="summary",
                opaque_roundtrip=response.reasoning[0].opaque_roundtrip,
                thinking_tokens=2,
                effort_used="high",
            ),
        )
        assert response.tool_calls == (
            FunctionToolCall("call_out", "weather", '{"city":"Paris"}'),
        )
        assert response.server_tool_uses == (
            ServerToolUse(
                tool_type="web_search",
                tool_use_id="ws_1",
                input={"type": "search", "query": "weather Paris"},
                result=[
                    {
                        "type": "url",
                        "url": "https://example.test/source",
                    }
                ],
            ),
        )
        assert response.logprobs == (TokenLogprob("A", -0.1),)
        assert response.top_logprobs == ((TopKEntry("B", -2.0),),)
        assert response.structured_output is not None
        assert response.structured_output.value == {"answer": 42}
        assert response.session_id == "resp_1"
        assert response.usage == UsageStats(
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
            reasoning_tokens=2,
            cached_tokens=1,
        )
        assert response.citations is not None
        assert response.citations[0].url == "https://example.test/source"
        assert response.citations[0].title == "Source"
        assert response.request_params == {
            key: value
            for key, value in _awaited_kwargs(create).items()
            if key not in {"model", "input"}
        }

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
    async def test_persisted_evidence_snapshots_the_final_pre_await_body(
        self, stream: bool
    ) -> None:
        raw = (
            _AsyncItems([_sdk_completed_event()])
            if stream
            else _response(_reasoning(), _message())
        )
        dialect, create = _dialect(raw)
        original = {"mode": "sent"}
        sent: dict[str, Any] = {}

        async def mutate_after_capture(**kwargs: Any) -> object:
            sent.update(deepcopy(kwargs))
            original["mode"] = "changed after response"
            kwargs["extra_body"]["vendor"]["mode"] = "changed inside SDK"
            kwargs["input"][0]["content"] = "changed inside SDK"
            return raw

        create.side_effect = mutate_after_capture
        response = await dialect.arun(
            Request(
                input=_chat(),
                scheduling=SchedulingParams(stream=stream),
                dialect_options=DialectOptions(
                    "openai_responses",
                    {"vendor": original},
                ),
            )
        )

        assert sent["extra_body"] == {"vendor": {"mode": "sent"}}
        assert sent["input"] == [
            {"type": "message", "role": "user", "content": "hello"}
        ]
        assert response.request_params is not None
        assert response.request_params["extra_body"] == {"vendor": {"mode": "sent"}}
        assert response.reasoning is not None
        reasoning = response.reasoning[0]
        assert reasoning is not None
        assert reasoning.opaque_roundtrip is not None
        envelope = json.loads(reasoning.opaque_roundtrip)
        assert envelope["items"][0] == {
            "type": "message",
            "role": "user",
            "content": "hello",
        }

    @pytest.mark.anyio
    async def test_real_sdk_expands_extra_body_into_http_json(self) -> None:
        requests: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            requests.append(payload)
            response_payload = _sdk_response_payload()
            response_payload["store"] = False
            return httpx.Response(200, json=response_payload)

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        try:
            dialect = OpenAIResponsesDialect(client, "requested/model")
            response = await dialect.arun(
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses",
                        {"store": False, "service_tier": "flex"},
                    ),
                )
            )
        finally:
            await client.close()

        assert response.texts == ("sdk output",)
        assert response.session_id is None
        assert len(requests) == 1
        wire = requests[0]
        assert wire["model"] == "requested/model"
        assert wire["store"] is False
        assert wire["service_tier"] == "flex"
        assert "extra_body" not in wire
        assert wire["input"] == [
            {"type": "message", "role": "user", "content": "hello"}
        ]

    @pytest.mark.anyio
    async def test_real_sdk_preserves_forward_web_search_status(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            response_payload = _sdk_response_payload()
            response_payload["output"] = [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "incomplete",
                    "action": {
                        "type": "open_page",
                        "url": "https://example.test/source",
                    },
                }
            ]
            return httpx.Response(200, json=response_payload, request=request)

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        try:
            dialect = OpenAIResponsesDialect(client, "requested/model")
            response = await dialect.arun(
                Request(
                    input=_chat(),
                    tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
                )
            )
        finally:
            await client.close()

        assert response.server_tool_uses is not None
        assert response.server_tool_uses[0].error_code == "incomplete"

    @pytest.mark.anyio
    async def test_real_sdk_replays_complete_stateless_history(self) -> None:
        requests: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            requests.append(payload)
            response_payload = _sdk_response_payload()
            response_payload["id"] = f"resp_{len(requests)}"
            response_payload["store"] = False
            response_output = cast(list[dict[str, Any]], response_payload["output"])
            message_content = cast(list[dict[str, Any]], response_output[1]["content"])
            message_content[0]["logprobs"] = [
                {
                    "bytes": [65],
                    "logprob": -0.1,
                    "token": "A",
                    "top_logprobs": [],
                }
            ]
            return httpx.Response(200, json=response_payload)

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        try:
            dialect = OpenAIResponsesDialect(client, "requested/model")
            first = await dialect.arun(
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses", {"store": False}
                    ),
                )
            )
            assert first.reasoning is not None
            first_reasoning = first.reasoning[0]
            assert first_reasoning is not None
            assert first_reasoning.opaque_roundtrip is not None

            await dialect.arun(
                Request(
                    input=_chat(ChatMessage("user", (TextPart("next"),))),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation(
                            "openai_responses",
                            first_reasoning.opaque_roundtrip,
                        )
                    ),
                    dialect_options=DialectOptions(
                        "openai_responses", {"store": False}
                    ),
                )
            )
        finally:
            await client.close()

        assert len(requests) == 2
        assert [item["type"] for item in requests[1]["input"]] == [
            "message",
            "reasoning",
            "message",
            "message",
        ]
        assert requests[1]["input"][0]["content"] == "hello"
        assert requests[1]["input"][3]["content"] == "next"
        assert requests[1]["input"][2]["content"][0]["logprobs"] == [
            {
                "bytes": [65],
                "logprob": -0.1,
                "token": "A",
                "top_logprobs": [],
            }
        ]

    @pytest.mark.anyio
    async def test_json_object_format_and_flat_function_tool_are_lowered(self) -> None:
        dialect, create = _dialect(_response(_message(_text("{}"))))

        await dialect.arun(
            Request(
                input=_chat(),
                tools=ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "name": "lookup",
                            "parameters": None,
                        },
                    )
                ),
                structured_output=StructuredOutputParams(format="json_object"),
            )
        )

        kwargs = _awaited_kwargs(create)
        assert kwargs["tools"] == [
            {"type": "function", "name": "lookup", "parameters": None}
        ]
        assert kwargs["text"] == {"format": {"type": "json_object"}}

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("auto", "auto"),
            (
                {"type": "function", "name": "lookup"},
                {"type": "function", "name": "lookup"},
            ),
        ],
    )
    async def test_supported_tool_choice_shapes_are_lowered(
        self, choice: JSONValue, expected: JSONValue
    ) -> None:
        dialect, create = _dialect()

        await dialect.arun(
            Request(
                input=_chat(),
                tools=ToolParams(
                    functions=({"type": "function", "name": "lookup"},),
                    choice=choice,
                ),
            )
        )

        assert _awaited_kwargs(create)["tool_choice"] == expected

    @pytest.mark.anyio
    async def test_sdk_web_search_open_page_action_is_lifted(self) -> None:
        dialect, create = _dialect(_sdk_web_search_response("completed"))

        response = await dialect.arun(
            Request(
                input=_chat(),
                tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
            )
        )

        assert response.server_tool_uses == (
            ServerToolUse(
                tool_type="web_search",
                tool_use_id="ws_1",
                input={
                    "type": "open_page",
                    "url": "https://example.test/source",
                },
            ),
        )
        assert _awaited_kwargs(create)["include"] == ["web_search_call.action.sources"]

    @pytest.mark.anyio
    async def test_sdk_nonterminal_web_search_status_is_rejected(self) -> None:
        dialect, _ = _dialect(_sdk_web_search_response("searching"))

        with pytest.raises(OutputContractError, match="searching"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
                )
            )

    @pytest.mark.anyio
    async def test_sdk_failed_web_search_status_is_preserved(self) -> None:
        dialect, _ = _dialect(_sdk_web_search_response("failed"))

        response = await dialect.arun(
            Request(
                input=_chat(),
                tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
            )
        )

        assert response.server_tool_uses == (
            ServerToolUse(
                tool_type="web_search",
                tool_use_id="ws_1",
                input={
                    "type": "open_page",
                    "url": "https://example.test/source",
                },
                error_code="failed",
            ),
        )

    @pytest.mark.anyio
    async def test_incomplete_web_search_status_is_preserved(self) -> None:
        raw = _sdk_response_payload()
        raw["output"] = [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "incomplete",
                "action": {
                    "type": "open_page",
                    "url": "https://example.test/source",
                },
            }
        ]
        dialect, _ = _dialect(raw)

        response = await dialect.arun(
            Request(
                input=_chat(),
                tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
            )
        )

        assert response.server_tool_uses == (
            ServerToolUse(
                tool_type="web_search",
                tool_use_id="ws_1",
                input={
                    "type": "open_page",
                    "url": "https://example.test/source",
                },
                error_code="incomplete",
            ),
        )

    @pytest.mark.anyio
    async def test_json_schema_uses_defaults_only_for_absent_optional_fields(
        self,
    ) -> None:
        dialect, create = _dialect(_response(_message(_text("{}"))))

        await dialect.arun(
            Request(
                input=_chat(),
                structured_output=StructuredOutputParams(
                    format="json_schema",
                    schema={"type": "object"},
                ),
            )
        )

        assert _awaited_kwargs(create)["text"] == {
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": {"type": "object"},
            }
        }

    @pytest.mark.anyio
    async def test_reasoning_summary_none_and_empty_options_are_noops(self) -> None:
        dialect, create = _dialect()

        await dialect.arun(
            Request(
                input=_chat(),
                reasoning=ReasoningParams(summary="none"),
                dialect_options=DialectOptions("openai_responses", {}),
            )
        )

        kwargs = _awaited_kwargs(create)
        assert "reasoning" not in kwargs
        assert "include" not in kwargs
        assert "extra_body" not in kwargs

    @pytest.mark.anyio
    async def test_plain_request_does_not_request_encrypted_reasoning(self) -> None:
        dialect, create = _dialect()

        await dialect.arun(Request(input=_chat()))

        assert "include" not in _awaited_kwargs(create)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reasoning",
        [
            pytest.param(ReasoningParams(effort="high"), id="effort"),
            pytest.param(ReasoningParams(summary="auto"), id="summary"),
        ],
    )
    async def test_active_reasoning_requests_encrypted_content(
        self, reasoning: ReasoningParams
    ) -> None:
        dialect, create = _dialect(_response(_reasoning(), _message()))

        await dialect.arun(Request(input=_chat(), reasoning=reasoning))

        assert _awaited_kwargs(create)["include"] == ["reasoning.encrypted_content"]

    @pytest.mark.anyio
    async def test_reasoning_effort_none_does_not_require_a_reasoning_item(
        self,
    ) -> None:
        dialect, create = _dialect(_response(_message()))

        response = await dialect.arun(
            Request(input=_chat(), reasoning=ReasoningParams(effort="none"))
        )

        assert response.reasoning is None
        kwargs = _awaited_kwargs(create)
        assert kwargs["reasoning"] == {"effort": "none"}
        assert "include" not in kwargs

    @pytest.mark.anyio
    async def test_opaque_history_is_complete_ordered_and_cumulative(self) -> None:
        dialect, create = _dialect(_response(_reasoning(), _message()))
        first = await dialect.arun(
            Request(input=_chat(), reasoning=ReasoningParams(effort="high"))
        )
        assert first.reasoning is not None
        opaque = first.reasoning[0]
        assert opaque is not None and opaque.opaque_roundtrip is not None

        create.reset_mock()
        create.return_value = _response(
            _reasoning("rs_2", encrypted="opaque-2"),
            _message(_text("second")),
        )
        second = await dialect.arun(
            Request(
                input=_chat(ChatMessage("user", (TextPart("next"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "openai_responses", opaque.opaque_roundtrip
                    )
                ),
            )
        )

        kwargs = _awaited_kwargs(create)
        assert kwargs["input"] == [
            {"type": "message", "role": "user", "content": "hello"},
            {
                "encrypted_content": "opaque",
                "id": "rs_1",
                "status": "completed",
                "summary": [{"text": "summary", "type": "summary_text"}],
                "type": "reasoning",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "ok",
                        "annotations": [],
                        "logprobs": None,
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": "next",
            },
        ]
        assert kwargs["include"] == ["reasoning.encrypted_content"]

        assert second.reasoning is not None
        second_reasoning = second.reasoning[0]
        assert second_reasoning is not None
        assert second_reasoning.opaque_roundtrip is not None
        create.reset_mock()
        create.return_value = _response(
            _reasoning("rs_3", encrypted="opaque-3"),
            _message(_text("third")),
        )
        await dialect.arun(
            Request(
                input=_chat(ChatMessage("user", (TextPart("again"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "openai_responses",
                        second_reasoning.opaque_roundtrip,
                    )
                ),
            )
        )

        third_input = _awaited_kwargs(create)["input"]
        assert [item["type"] for item in third_input] == [
            "message",
            "reasoning",
            "message",
            "message",
            "reasoning",
            "message",
            "message",
        ]
        assert third_input[0]["content"] == "hello"
        assert third_input[3]["content"] == "next"
        assert third_input[6] == {
            "type": "message",
            "role": "user",
            "content": "again",
        }

    @pytest.mark.anyio
    async def test_opaque_history_preserves_interleaved_function_calls(self) -> None:
        function_call = SimpleNamespace(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="lookup",
            arguments='{"q":"x"}',
            status="completed",
        )
        dialect, create = _dialect(
            _response(
                _reasoning("rs_1", encrypted="opaque-1"),
                SimpleNamespace(
                    type="web_search_call",
                    id="ws_1",
                    status="completed",
                    action={"type": "search", "query": "x"},
                ),
                function_call,
                _reasoning("rs_2", encrypted="opaque-2"),
            )
        )
        first = await dialect.arun(Request(input=_chat()))
        assert first.reasoning is not None
        first_reasoning = first.reasoning[0]
        assert first_reasoning is not None
        assert first_reasoning.opaque_roundtrip is not None

        create.reset_mock()
        create.return_value = _response(
            _reasoning("rs_3", encrypted="opaque-3"),
            _message(),
        )
        await dialect.arun(
            Request(
                input=_chat(
                    ChatMessage(
                        "tool",
                        (ToolResultPart("call_1", {"value": 1}),),
                    )
                ),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "openai_responses",
                        first_reasoning.opaque_roundtrip,
                    )
                ),
            )
        )

        wire = _awaited_kwargs(create)["input"]
        assert [item["type"] for item in wire] == [
            "message",
            "reasoning",
            "web_search_call",
            "function_call",
            "reasoning",
            "function_call_output",
        ]
        assert wire[3]["call_id"] == "call_1"
        assert wire[5] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"value":1}',
        }

    @pytest.mark.anyio
    async def test_continuation_survives_a_turn_without_new_reasoning(self) -> None:
        dialect, create = _dialect(_response(_reasoning(), _message()))
        first = await dialect.arun(Request(input=_chat()))
        assert first.reasoning is not None
        first_reasoning = first.reasoning[0]
        assert first_reasoning is not None
        assert first_reasoning.opaque_roundtrip is not None

        create.reset_mock()
        create.return_value = _response(_message(_text("no new reasoning")))
        second = await dialect.arun(
            Request(
                input=_chat(ChatMessage("user", (TextPart("next"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "openai_responses",
                        first_reasoning.opaque_roundtrip,
                    )
                ),
            )
        )

        assert second.reasoning is not None
        continuation = second.reasoning[0]
        assert continuation is not None
        assert continuation.text is None
        assert continuation.thinking_tokens == 0
        assert continuation.opaque_roundtrip is not None
        envelope = json.loads(continuation.opaque_roundtrip)
        assert [item["type"] for item in envelope["items"]] == [
            "message",
            "reasoning",
            "message",
            "message",
            "message",
        ]

    @pytest.mark.anyio
    async def test_previous_response_chain_does_not_claim_standalone_history(
        self,
    ) -> None:
        dialect, _ = _dialect(_response(_reasoning(), _message()))

        response = await dialect.arun(
            Request(
                input=_chat(),
                session=SessionParams(previous_response_id="resp_previous"),
            )
        )

        assert response.reasoning is not None
        assert response.reasoning[0] is not None
        assert response.reasoning[0].opaque_roundtrip is None

    @pytest.mark.anyio
    async def test_sampled_logprobs_without_alternatives_requests_zero(self) -> None:
        dialect, create = _dialect(_response(_message(_text(logprobs=[_logprob()]))))

        response = await dialect.arun(
            Request(
                input=_chat(),
                scoring=ScoringParams(sampled_logprobs=True),
            )
        )

        kwargs = _awaited_kwargs(create)
        assert kwargs["top_logprobs"] == 0
        assert kwargs["include"] == ["message.output_text.logprobs"]
        assert response.logprobs is not None
        assert response.top_logprobs is None


class TestWireEvidence:
    @pytest.mark.anyio
    async def test_input_lowering_drift_is_rejected_before_sdk_io(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialect, create = _dialect()
        original = openai_responses_module._message_to_input_items

        def corrupt_role(message: ChatMessage) -> list[dict[str, JSONValue]]:
            items = original(message)
            items[0]["role"] = "system"
            return items

        monkeypatch.setattr(
            openai_responses_module,
            "_message_to_input_items",
            corrupt_role,
        )

        with pytest.raises(RequestAuditError, match="body.input"):
            await dialect.arun(Request(input=_chat()))

        create.assert_not_awaited()

    def test_wire_verifiers_reject_every_semantic_mutation(self) -> None:
        dialect, _ = _dialect()
        request = Request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        TextPart("look"),
                        ImagePart(
                            data="YWJj",
                            media_type="image/png",
                            detail="high",
                        ),
                    ),
                ),
                ChatMessage(
                    "assistant",
                    (ToolCallPart("call_1", "lookup", {"city": "Paris"}),),
                ),
                ChatMessage(
                    "tool",
                    (ToolResultPart("call_1", {"ok": True}),),
                ),
            ),
            sampling=SamplingParams(
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
            ),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
            reasoning=ReasoningParams(effort="high", summary="detailed"),
            tools=ToolParams(
                functions=(
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a city",
                            "parameters": {"type": "object"},
                            "strict": True,
                        },
                    },
                ),
                choice={"type": "function", "function": {"name": "lookup"}},
                parallel=False,
                hosted=(
                    HostedToolSpec(
                        "web_search",
                        {"search_context_size": "medium"},
                    ),
                ),
            ),
            structured_output=StructuredOutputParams(
                format="json_schema",
                schema={"type": "object"},
                name="answer",
                strict=True,
            ),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation(
                    "openai_responses",
                    _opaque_reasoning_continuation(),
                )
            ),
            scheduling=SchedulingParams(stream=True),
            dialect_options=DialectOptions(
                "openai_responses",
                {"vendor": {"mode": "strict"}},
            ),
        )
        audit, prepared = _prepare_for_audit(dialect, request)
        audit.finish(prepared)

        mutations: tuple[tuple[str, Callable[[dict[str, Any]], object]], ...] = (
            (
                "opaque continuation",
                lambda body: body["input"][0].__setitem__(
                    "encrypted_content", "changed"
                ),
            ),
            (
                "message role",
                lambda body: body["input"][1].__setitem__("role", "system"),
            ),
            (
                "message text",
                lambda body: body["input"][1]["content"][0].__setitem__(
                    "text", "changed"
                ),
            ),
            (
                "image URL",
                lambda body: body["input"][1]["content"][1].__setitem__(
                    "image_url", "data:image/png;base64,changed"
                ),
            ),
            (
                "image detail",
                lambda body: body["input"][1]["content"][1].__setitem__(
                    "detail", "low"
                ),
            ),
            (
                "historical tool-call arguments",
                lambda body: body["input"][2].__setitem__(
                    "arguments", '{"city":"London"}'
                ),
            ),
            (
                "historical tool-result output",
                lambda body: body["input"][3].__setitem__("output", '{"ok":false}'),
            ),
            ("input order", lambda body: body["input"].reverse()),
            (
                "max output tokens",
                lambda body: body.__setitem__("max_output_tokens", 32),
            ),
            ("temperature", lambda body: body.__setitem__("temperature", 0.4)),
            ("top p", lambda body: body.__setitem__("top_p", 0.8)),
            ("top logprobs", lambda body: body.__setitem__("top_logprobs", 1)),
            (
                "logprobs include",
                lambda body: body["include"].remove("message.output_text.logprobs"),
            ),
            (
                "reasoning effort",
                lambda body: body["reasoning"].__setitem__("effort", "low"),
            ),
            (
                "reasoning summary",
                lambda body: body["reasoning"].__setitem__("summary", "concise"),
            ),
            (
                "function tool",
                lambda body: body["tools"][0].__setitem__("name", "changed"),
            ),
            (
                "hosted tool",
                lambda body: body["tools"][1].__setitem__("search_context_size", "low"),
            ),
            ("tool order", lambda body: body["tools"].reverse()),
            (
                "hosted-tool include",
                lambda body: body["include"].remove("web_search_call.action.sources"),
            ),
            (
                "tool choice",
                lambda body: body["tool_choice"].__setitem__("name", "changed"),
            ),
            (
                "parallel tools",
                lambda body: body.__setitem__("parallel_tool_calls", True),
            ),
            (
                "structured format",
                lambda body: body["text"]["format"].__setitem__("type", "json_object"),
            ),
            (
                "structured schema",
                lambda body: body["text"]["format"]["schema"].__setitem__(
                    "type", "array"
                ),
            ),
            (
                "structured name",
                lambda body: body["text"]["format"].__setitem__("name", "changed"),
            ),
            (
                "structured strict",
                lambda body: body["text"]["format"].__setitem__("strict", False),
            ),
            ("stream", lambda body: body.__setitem__("stream", False)),
            (
                "dialect passthrough",
                lambda body: body["extra_body"]["vendor"].__setitem__(
                    "mode", "changed"
                ),
            ),
        )

        survived: list[str] = []
        for label, mutate in mutations:
            body = cast(dict[str, Any], prepared.thaw_body())
            mutate(body)
            try:
                audit.finish(replace(prepared, body=body))
            except RequestAuditError:
                continue
            survived.append(label)
        assert not survived, f"wire mutations the audit accepted: {survived}"


class TestPreflightRejections:
    def test_capability_row_and_config_validators_are_concrete(self) -> None:
        assert set(CAPABILITY_DECISIONS) == {
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
            "reasoning",
            "function_tools",
            "hosted_tools",
            "structured_output",
            "stateful_session",
            "opaque_continuation",
            "multimodal_input",
            "prefill",
            "fim",
        }
        assert (
            Capability.ServerTools
            in OpenAIResponsesDialect(MagicMock(), "model").capabilities
        )

        top = CAPABILITY_DECISIONS["top_logprobs"]
        reasoning = CAPABILITY_DECISIONS["reasoning"]
        hosted = CAPABILITY_DECISIONS["hosted_tools"]
        multimodal = CAPABILITY_DECISIONS["multimodal_input"]
        assert isinstance(top, Supported)
        assert isinstance(reasoning, Supported)
        assert isinstance(hosted, Supported)
        assert isinstance(multimodal, Supported)
        assert "input.modality.image.detail" in multimodal.binding.request_leaves
        top.binding.validate_config(TopLogprobsOptions(minimum=20))
        with pytest.raises(ValueError, match="at most 20"):
            top.binding.validate_config(TopLogprobsOptions(minimum=21))
        with pytest.raises(ValueError, match="budget_tokens"):
            reasoning.binding.validate_config(ReasoningOptions(budget_tokens=32))
        reasoning.binding.validate_config(ReasoningOptions(effort="max"))
        with pytest.raises(ValueError, match="effort"):
            reasoning.binding.validate_config(ReasoningOptions(effort="ultra"))
        hosted.binding.validate_config(HostedToolsOptions(kinds=("web_search",)))
        with pytest.raises(ValueError, match="unsupported kinds"):
            hosted.binding.validate_config(HostedToolsOptions(kinds=("file_search",)))
        multimodal.binding.validate_config(
            MultimodalInputOptions(modalities=("image",))
        )
        invalid_multimodal = object.__new__(MultimodalInputOptions)
        object.__setattr__(invalid_multimodal, "modalities", ("audio",))
        with pytest.raises(ValueError, match="does not support modalities"):
            multimodal.binding.validate_config(invalid_multimodal)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model_request", "message"),
        [
            (Request(input=CompletionInput("prompt")), "requires ChatInput"),
            (
                Request(input=_chat(), sampling=SamplingParams(top_k=10)),
                "no equivalent request field",
            ),
            (
                Request(input=_chat(), sampling=SamplingParams(stop=("END",))),
                "no equivalent request field",
            ),
            (
                Request(input=_chat(), sampling=SamplingParams(seed=7)),
                "no equivalent request field",
            ),
            (
                Request(input=_chat(), sampling=SamplingParams(frequency_penalty=0.2)),
                "no equivalent request field",
            ),
            (
                Request(input=_chat(), sampling=SamplingParams(presence_penalty=0.2)),
                "no equivalent request field",
            ),
            (
                Request(input=_chat(), sampling=SamplingParams(n=2)),
                "no equivalent request field",
            ),
            (
                Request(input=_chat(), scoring=ScoringParams(input_scoring=True)),
                "unavailable capability 'input_scoring'",
            ),
            (
                Request(
                    input=_chat(),
                    scoring=ScoringParams(sampled_logprobs=True, top_logprobs=21),
                ),
                "at most 20",
            ),
            (
                Request(input=_chat(), reasoning=ReasoningParams(budget_tokens=32)),
                "token-budget",
            ),
            (
                Request(input=_chat(), reasoning=ReasoningParams(effort="ultra")),
                "unsupported OpenAI reasoning effort",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(hosted=(HostedToolSpec("file_search"),)),
                ),
                "web-search hosted tools",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (
                                ImagePart(
                                    url="https://example.test/image.png",
                                    media_type="image/png",
                                ),
                            ),
                        )
                    )
                ),
                "no media-type field",
            ),
            (
                Request(input=_chat(ChatMessage("user", (ImagePart(data="YWJj"),)))),
                "explicit image media type",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (ImagePart(data="YWJj", media_type="image/tiff"),),
                        )
                    )
                ),
                "unsupported Responses image media type",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (ImagePart(data="YWJj", media_type=""),),
                        )
                    )
                ),
                "unsupported Responses image media type",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (
                                ImagePart(
                                    data="YWJj",
                                    media_type="image/png",
                                    detail="",
                                ),
                            ),
                        )
                    )
                ),
                "unsupported Responses image detail",
            ),
            (
                Request(input=_chat(ChatMessage("user", (ImagePart(url=""),)))),
                "image URLs must not be empty",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (ImagePart(data="", media_type="image/png"),),
                        )
                    )
                ),
                "inline images require non-empty base64 data",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage(
                            "tool",
                            (ToolResultPart("call", "failed", is_error=True),),
                        )
                    )
                ),
                "cannot transmit is_error",
            ),
            (
                Request(
                    input=_chat(ChatMessage("user", (TextPart("hello"),), name="named"))
                ),
                "no name field",
            ),
            (
                Request(
                    input=_chat(),
                    structured_output=StructuredOutputParams(
                        format="json_schema",
                        schema={},
                        name="",
                    ),
                ),
                "schema names must not be empty",
            ),
            (
                Request(
                    input=_chat(),
                    structured_output=StructuredOutputParams(
                        format="json_object",
                        schema={"type": "object"},
                    ),
                ),
                "schema, name, and strict require format='json_schema'",
            ),
            (
                Request(
                    input=_chat(),
                    session=SessionParams(
                        previous_response_id="resp_1",
                        opaque_continuation=OpaqueContinuation(
                            "openai_responses", '{"version":1,"items":[]}'
                        ),
                    ),
                ),
                "mutually exclusive",
            ),
            (
                Request(
                    input=_chat(),
                    session=SessionParams(previous_response_id="resp_1"),
                    dialect_options=DialectOptions(
                        "openai_responses", {"store": False}
                    ),
                ),
                "store=false is stateless",
            ),
            (
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses", {"conversation": "conv_123"}
                    ),
                ),
                "conversation state is not yet modeled",
            ),
            (
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses", {"instructions": "ignore the input"}
                    ),
                ),
                "must be expressed through provider-neutral",
            ),
            (
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses", {"max_output_tokens": 1}
                    ),
                ),
                "canonical provider-neutral field",
            ),
            (
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses", {"background": True}
                    ),
                ),
                "background Responses cannot complete",
            ),
            (
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses", {"store": "false"}
                    ),
                ),
                "store must be a boolean",
            ),
            (
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions("openai_responses", {"": 1}),
                ),
                "dialect option keys must be non-empty",
            ),
        ],
    )
    async def test_every_request_rejection_happens_before_io(
        self, model_request: Request, message: str
    ) -> None:
        dialect, create = _dialect()

        with pytest.raises(
            (DialectError, RequestAuditError, ValueError), match=message
        ):
            await dialect.arun(model_request)

        create.assert_not_awaited()

    def test_empty_option_key_is_rejected_during_validation(self) -> None:
        """Pin *which* layer rejects, not merely that something does.

        ``finish`` is a backstop with identical wording, so matching only the
        message still passes with this dialect's check deleted.
        """

        dialect, _ = _dialect()
        model_request = Request(
            input=_chat(),
            dialect_options=DialectOptions("openai_responses", {"": 1}),
        )
        audit = RequestAudit(active_request_leaves(model_request))

        dialect.validate_request(model_request, audit, SimpleNamespace())

        decision = audit.decisions["dialect_options."]
        assert isinstance(decision, Rejected)
        assert "non-empty" in decision.reason

    @pytest.mark.anyio
    @pytest.mark.parametrize("option_key", _RESPONSES_IR_OWNED_OPTIONS)
    async def test_every_ir_owned_wire_key_rejects_alternate_authority(
        self, option_key: str
    ) -> None:
        dialect, create = _dialect()

        with pytest.raises(RequestAuditError, match="dialect_options"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses",
                        {option_key: None},
                    ),
                )
            )

        create.assert_not_awaited()

    def test_ir_owned_wire_key_oracle_matches_the_dialect_policy(self) -> None:
        assert set(openai_responses_module._IR_OWNED_BODY_KEYS) == (
            set(BINDING_RESOURCE_KEYS) | set(_RESPONSES_IR_OWNED_OPTIONS)
        )

    @pytest.mark.parametrize(
        "available_capabilities",
        [
            pytest.param(frozenset(), id="stateful-session-disabled"),
            pytest.param(
                frozenset({"stateful_session"}),
                id="stateful-session-available",
            ),
        ],
    )
    def test_conversation_option_cannot_bypass_stateful_session_gate(
        self, available_capabilities: frozenset[str]
    ) -> None:
        dialect, create = _dialect()
        request = Request(
            input=_chat(),
            dialect_options=DialectOptions(
                "openai_responses", {"conversation": "conv_123"}
            ),
        )
        plan = SimpleNamespace(
            dialect_id="openai_responses",
            available_capabilities=available_capabilities,
            capability_minimums={},
            required_output_channels=frozenset(),
        )

        validate_runtime_binding_plan(plan, request)
        audit = RequestAudit(active_request_leaves(request))
        dialect.validate_request(request, audit, plan)

        with pytest.raises(
            RequestAuditError,
            match=r"dialect_options\.conversation.*not yet modeled",
        ):
            audit.raise_rejections()
        create.assert_not_awaited()

    def test_empty_requested_model_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requested_model_id must not be empty"):
            OpenAIResponsesDialect(MagicMock(), "")

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model_request", "message"),
        [
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(functions=({"type": "custom", "name": "x"},)),
                ),
                "require type='function'",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(
                        functions=({"type": "function", "name": "x", "unknown": True},)
                    ),
                ),
                "unsupported function tool field",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(functions=({"type": "function"},)),
                ),
                "function tool name must be a non-empty string",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(
                        functions=(
                            {"type": "function", "function": {"name": "x"}, "x": 1},
                        )
                    ),
                ),
                "invalid Chat-shaped",
            ),
            (
                Request(input=_chat(), tools=ToolParams(choice="sometimes")),
                "unsupported Responses tool choice",
            ),
            (
                Request(input=_chat(), tools=ToolParams(choice=1)),
                "tool choice must be a string or mapping",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(choice={"type": "web_search"}),
                ),
                "only function-specific",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(
                        choice={
                            "type": "function",
                            "function": {"name": "x"},
                            "unknown": True,
                        }
                    ),
                ),
                "invalid Chat-shaped function tool choice",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(
                        choice={"type": "function", "name": "x", "unknown": True}
                    ),
                ),
                "invalid Responses function tool choice",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(choice={"type": "function", "name": ""}),
                ),
                "function tool choice requires a non-empty name",
            ),
            (
                Request(
                    input=_chat(),
                    tools=ToolParams(
                        hosted=(HostedToolSpec("web_search", {"type": "other"}),)
                    ),
                ),
                "cannot override",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage("user", (ToolCallPart("call", "tool", {}),))
                    )
                ),
                "assistant message",
            ),
            (
                Request(input=_chat(ChatMessage("tool", (TextPart("x"),)))),
                "requires one result",
            ),
            (
                Request(
                    input=_chat(
                        ChatMessage(
                            "tool",
                            (
                                ToolResultPart("call", "result"),
                                TextPart("extra"),
                            ),
                        )
                    )
                ),
                "must contain exactly one result",
            ),
        ],
    )
    async def test_structural_lowering_errors_happen_before_io(
        self, model_request: Request, message: str
    ) -> None:
        dialect, create = _dialect()

        with pytest.raises(DialectError, match=message):
            await dialect.arun(model_request)

        create.assert_not_awaited()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "opaque",
        [
            "not-json",
            "[]",
            '{"version":2,"items":[]}',
            (
                '{"version":true,"items":[{"type":"reasoning","id":"rs_1",'
                '"summary":[],"encrypted_content":"opaque"}]}'
            ),
            '{"version":1,"items":[]}',
            '{"version":2,"items":[{"type":"unknown"}]}',
            (
                '{"version":2,"items":[{"type":"message","role":"user",'
                '"content":"hello"}]}'
            ),
            '{"version":2,"items":[{"type":"reasoning"}]}',
            ('{"version":2,"items":[{"type":"reasoning","id":"rs_1","summary":[]}]}'),
            (
                '{"version":2,"items":[{"type":"reasoning","id":"rs_1",'
                '"summary":[],"encrypted_content":"opaque"}],"extra":true}'
            ),
            (
                '{"version":2,"version":2,"items":[{"type":"reasoning",'
                '"id":"rs_1","summary":[],"encrypted_content":"opaque"}]}'
            ),
            (
                '{"version":2,"items":[{"type":"reasoning","id":"rs_1",'
                '"id":"rs_2","summary":[],"encrypted_content":"opaque"}]}'
            ),
            pytest.param(
                _opaque_reasoning_continuation(unknown=None),
                id="reasoning-item-unknown-null-field",
            ),
            pytest.param(
                _opaque_reasoning_continuation(summary="bad"),
                id="reasoning-summary-not-list",
            ),
            pytest.param(
                _opaque_reasoning_continuation(
                    summary=[{"type": "reasoning_text", "text": "bad"}]
                ),
                id="reasoning-summary-wrong-type",
            ),
            pytest.param(
                _opaque_reasoning_continuation(
                    summary=[{"type": "summary_text", "text": "ok", "extra": None}]
                ),
                id="reasoning-summary-extra-field",
            ),
            pytest.param(
                _opaque_reasoning_continuation(
                    summary=[{"type": "summary_text", "text": 1}]
                ),
                id="reasoning-summary-text-not-string",
            ),
            pytest.param(
                _opaque_reasoning_continuation(content="bad"),
                id="reasoning-content-not-list",
            ),
            pytest.param(
                _opaque_reasoning_continuation(
                    content=[{"type": "summary_text", "text": "bad"}]
                ),
                id="reasoning-content-wrong-type",
            ),
            pytest.param(
                _opaque_reasoning_continuation(
                    content=[{"type": "reasoning_text", "text": "ok", "extra": None}]
                ),
                id="reasoning-content-extra-field",
            ),
            pytest.param(
                _opaque_reasoning_continuation(status=1),
                id="reasoning-status-not-string",
            ),
            pytest.param(
                _opaque_reasoning_continuation(status="queued"),
                id="reasoning-status-not-terminal-or-in-progress",
            ),
            pytest.param(
                _opaque_reasoning_continuation(encrypted_content=1),
                id="reasoning-encrypted-content-not-string",
            ),
            pytest.param(
                _opaque_reasoning_continuation(encrypted_content=""),
                id="reasoning-encrypted-content-empty",
            ),
        ],
    )
    async def test_invalid_opaque_continuation_fails_before_io(
        self, opaque: str
    ) -> None:
        dialect, create = _dialect()

        with pytest.raises(DialectError, match="opaque continuation"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation(
                            "openai_responses", opaque
                        )
                    ),
                )
            )

        create.assert_not_awaited()

    def test_prepare_rejects_mismatched_audit(self) -> None:
        dialect, _ = _dialect()
        req = Request(input=_chat())
        wrong = RequestAudit(active_request_leaves(Request(CompletionInput("x"))))

        with pytest.raises(DialectError, match="audit does not match"):
            dialect.prepare(req, wrong)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "prepared",
        [
            PreparedRequest("other", {}),
            PreparedRequest("responses.create", {}),
        ],
    )
    async def test_execute_rejects_invalid_prepared_requests(
        self, prepared: PreparedRequest
    ) -> None:
        dialect, create = _dialect()

        with pytest.raises(DialectError, match="operation|context"):
            await dialect.execute(prepared)

        create.assert_not_awaited()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda body: body.pop("model"), "invalid model"),
            (lambda body: body.pop("stream"), "invalid stream flag"),
            (lambda body: body.__setitem__("stream", "false"), "stream flag"),
            (lambda body: body.__setitem__("input", "hello"), "input body"),
            (lambda body: body.__setitem__("input", [1]), "input item"),
            (lambda body: body.__setitem__("extra_body", []), "extra_body"),
            (
                lambda body: body["extra_body"].__setitem__("store", "false"),
                "store must be a boolean",
            ),
        ],
    )
    async def test_execute_validates_prepared_body_before_sdk_io(
        self,
        mutate: Callable[[dict[str, Any]], object],
        message: str,
    ) -> None:
        dialect, create = _dialect()
        _, prepared = _prepare_for_audit(
            dialect,
            Request(
                input=_chat(),
                dialect_options=DialectOptions(
                    "openai_responses",
                    {"store": False},
                ),
            ),
        )
        body = cast(dict[str, Any], prepared.thaw_body())
        mutate(body)

        with pytest.raises(DialectError, match=message):
            await dialect.execute(replace(prepared, body=body))

        create.assert_not_awaited()


class TestResponseLifting:
    @pytest.mark.parametrize(
        "channel",
        ["server_tool_uses", "citations"],
    )
    def test_custom_tuple_output_validators_reject_wrong_item_types(
        self, channel: str
    ) -> None:
        with pytest.raises(OutputContractError, match=rf"{channel} channel"):
            OUTPUT_CONTRACT.rules[channel].validator((object(),))

    @pytest.mark.anyio
    async def test_locked_openai_sdk_response_shape_is_lifted_directly(self) -> None:
        raw = _sdk_response_payload()
        sdk_response = SDKResponse.model_validate(raw)
        dialect, _ = _dialect(sdk_response)

        response = await dialect.arun(Request(input=_chat()))

        assert response.texts == ("sdk output",)
        assert response.session_id == "resp_sdk"
        assert response.reasoning is not None
        assert response.reasoning[0] is not None
        assert response.reasoning[0].text == "why"

    @pytest.mark.anyio
    async def test_reported_store_false_suppresses_session_id(self) -> None:
        raw = _response(response_id="resp_zdr")
        raw.store = False
        dialect, _ = _dialect(raw)

        response = await dialect.arun(Request(input=_chat()))

        assert response.session_id is None

    @pytest.mark.anyio
    async def test_reported_store_must_be_boolean(self) -> None:
        raw = _response()
        raw.store = "false"
        dialect, _ = _dialect(raw)

        with pytest.raises(
            OutputContractError, match="response.store must be a boolean"
        ):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("options", "expected_session_id"),
        [
            pytest.param(None, "resp_1", id="implicit-store-true"),
            pytest.param({"store": False}, None, id="explicit-store-false"),
            pytest.param({"store": True}, "resp_1", id="explicit-store-true"),
        ],
    )
    async def test_explicit_null_store_is_treated_as_unreported(
        self,
        options: dict[str, JSONValue] | None,
        expected_session_id: str | None,
    ) -> None:
        """A partially compatible endpoint may echo ``store`` as a JSON null.

        ``None`` is not the ``_MISSING`` sentinel, so without the null check
        every reply from such an endpoint fails the boolean guard.
        """

        raw = _response()
        raw.store = None
        dialect, _ = _dialect(raw)

        response = await dialect.arun(
            Request(
                input=_chat(),
                dialect_options=(
                    None
                    if options is None
                    else DialectOptions("openai_responses", options)
                ),
            )
        )

        assert response.session_id == expected_session_id

    @pytest.mark.anyio
    async def test_store_false_falls_back_to_request_when_response_omits_it(
        self,
    ) -> None:
        dialect, _ = _dialect(_response(response_id="resp_stateless"))

        response = await dialect.arun(
            Request(
                input=_chat(),
                dialect_options=DialectOptions("openai_responses", {"store": False}),
            )
        )

        assert response.session_id is None

    @pytest.mark.anyio
    async def test_store_false_returns_replayable_history_not_a_session_id(
        self,
    ) -> None:
        dialect, _ = _dialect(_response(_reasoning(), _message()))

        response = await dialect.arun(
            Request(
                input=_chat(),
                dialect_options=DialectOptions("openai_responses", {"store": False}),
            )
        )

        assert response.session_id is None
        assert response.reasoning is not None
        reasoning = response.reasoning[0]
        assert reasoning is not None
        assert reasoning.opaque_roundtrip is not None
        envelope = json.loads(reasoning.opaque_roundtrip)
        assert envelope["version"] == 2
        assert [item["type"] for item in envelope["items"]] == [
            "message",
            "reasoning",
            "message",
        ]

    @pytest.mark.anyio
    async def test_reported_store_must_match_explicit_request(self) -> None:
        raw = _response()
        raw.store = True
        dialect, _ = _dialect(raw)

        with pytest.raises(OutputContractError, match="store contradicts"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    dialect_options=DialectOptions(
                        "openai_responses", {"store": False}
                    ),
                )
            )

    @pytest.mark.anyio
    async def test_incomplete_reason_is_preserved_verbatim(self) -> None:
        dialect, _ = _dialect(
            _response(status="incomplete", incomplete_reason="max_output_tokens")
        )

        response = await dialect.arun(Request(input=_chat()))

        assert response.finish_reasons == ("max_output_tokens",)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "item",
        [
            pytest.param(
                SimpleNamespace(
                    type="message",
                    id="msg_1",
                    role="assistant",
                    status="in_progress",
                    content=[_text()],
                ),
                id="message",
            ),
            pytest.param(
                SimpleNamespace(
                    type="reasoning",
                    id="rs_1",
                    summary=[],
                    content=None,
                    encrypted_content="opaque",
                    status="in_progress",
                ),
                id="reasoning",
            ),
            pytest.param(
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="lookup",
                    arguments="{}",
                    status="in_progress",
                ),
                id="function-call",
            ),
            pytest.param(
                SimpleNamespace(
                    type="web_search_call",
                    id="ws_1",
                    status="in_progress",
                    action={"type": "search", "query": "test"},
                ),
                id="web-search",
            ),
        ],
    )
    async def test_terminal_response_rejects_in_progress_output_items(
        self, item: object
    ) -> None:
        dialect, _ = _dialect(_response(item))

        with pytest.raises(OutputContractError, match="remains in_progress"):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "item",
        [
            pytest.param(_message(status=1), id="message-non-string"),
            pytest.param(_message(status="queued"), id="message-unknown"),
            pytest.param(
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="lookup",
                    arguments="{}",
                    status=1,
                ),
                id="function-call-non-string",
            ),
            pytest.param(
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="lookup",
                    arguments="{}",
                    status="queued",
                ),
                id="function-call-unknown",
            ),
            pytest.param(_reasoning(status=1), id="reasoning-non-string"),
            pytest.param(_reasoning(status="queued"), id="reasoning-unknown"),
        ],
    )
    async def test_terminal_response_rejects_invalid_output_item_status(
        self, item: object
    ) -> None:
        dialect, _ = _dialect(_response(item))

        with pytest.raises(OutputContractError, match="status .* is not terminal"):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "item",
        [
            pytest.param(_message(status="incomplete"), id="message"),
            pytest.param(_reasoning(status="incomplete"), id="reasoning"),
            pytest.param(
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="lookup",
                    arguments="{}",
                    status="incomplete",
                ),
                id="function-call",
            ),
        ],
    )
    async def test_incomplete_response_accepts_incomplete_output_item_status(
        self, item: object
    ) -> None:
        dialect, _ = _dialect(
            _response(
                item,
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        )

        response = await dialect.arun(Request(input=_chat()))

        assert response.finish_reasons == ("max_output_tokens",)

    @pytest.mark.anyio
    async def test_incomplete_reasoning_status_is_preserved_for_replay(self) -> None:
        dialect, _ = _dialect(
            _response(
                _reasoning(status="incomplete"),
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        )

        response = await dialect.arun(Request(input=_chat()))

        assert response.reasoning is not None
        reasoning = response.reasoning[0]
        assert reasoning is not None
        assert reasoning.opaque_roundtrip is not None
        envelope = json.loads(reasoning.opaque_roundtrip)
        assert envelope["items"][-1]["status"] == "incomplete"

    @pytest.mark.anyio
    async def test_requested_logprobs_cover_every_nonempty_text_part(self) -> None:
        dialect, _ = _dialect(
            _response(
                _message(
                    _text("A", logprobs=None),
                    _text("B", logprobs=[_logprob()]),
                )
            )
        )

        with pytest.raises(OutputContractError, match="logprobs is absent"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    scoring=ScoringParams(sampled_logprobs=True),
                )
            )

    @pytest.mark.anyio
    async def test_tool_only_reply_has_empty_requested_logprob_channels(self) -> None:
        dialect, _ = _dialect(
            _response(
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="lookup",
                    arguments="{}",
                )
            )
        )

        response = await dialect.arun(
            Request(
                input=_chat(),
                scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
            )
        )

        assert response.logprobs == ()
        assert response.top_logprobs == ()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reasoning_items",
        [
            (_reasoning(summary=None, content="raw reasoning"),),
            (_reasoning(summary=""),),
            (
                _reasoning("rs_raw", summary=None, content="raw reasoning"),
                _reasoning("rs_empty", summary=""),
            ),
        ],
    )
    async def test_requested_reasoning_summary_requires_one_nonempty_summary(
        self, reasoning_items: tuple[object, ...]
    ) -> None:
        dialect, _ = _dialect(_response(*reasoning_items, _message()))

        with pytest.raises(OutputContractError, match="requested visible reasoning"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    reasoning=ReasoningParams(summary="detailed"),
                )
            )

    @pytest.mark.anyio
    async def test_requested_reasoning_summary_is_response_level(self) -> None:
        dialect, _ = _dialect(
            _response(
                _reasoning("rs_first", summary="first summary"),
                _reasoning("rs_raw", summary=None, content="raw reasoning"),
                _reasoning("rs_empty", summary=""),
                _reasoning("rs_second", summary="second summary"),
                _message(),
            )
        )

        response = await dialect.arun(
            Request(
                input=_chat(),
                reasoning=ReasoningParams(summary="detailed"),
            )
        )

        assert response.reasoning is not None
        assert response.reasoning[0] is not None
        assert response.reasoning[0].text == "first summary\nsecond summary"

    @pytest.mark.anyio
    async def test_missing_encrypted_reasoning_is_not_advertised_as_opaque(
        self,
    ) -> None:
        dialect, _ = _dialect(_response(_reasoning(encrypted=None), _message()))

        response = await dialect.arun(Request(input=_chat()))

        assert response.reasoning is not None
        assert response.reasoning[0] is not None
        assert response.reasoning[0].opaque_roundtrip is None

    @pytest.mark.anyio
    async def test_reported_usage_total_is_preserved_only_when_it_differs(self) -> None:
        dialect, _ = _dialect(_response(usage=_usage(total_tokens=9)))

        response = await dialect.arun(Request(input=_chat()))

        assert response.usage is not None
        assert response.usage.total_tokens == 5
        assert response.usage.reported_total_tokens == 9

    @pytest.mark.anyio
    async def test_invalid_optional_usage_details_are_ignored(self) -> None:
        dialect, _ = _dialect(
            _response(usage=_usage(cached_tokens=True, reasoning_tokens=-1))
        )

        response = await dialect.arun(Request(input=_chat()))

        assert response.usage is not None
        assert response.usage.cached_tokens is None
        assert response.usage.reasoning_tokens is None

    @pytest.mark.anyio
    async def test_reasoning_without_visible_text_remains_opaque_only(self) -> None:
        dialect, _ = _dialect(
            _response(_reasoning(summary=None, content=None), _message())
        )

        response = await dialect.arun(Request(input=_chat()))

        assert response.reasoning is not None
        assert response.reasoning[0] is not None
        assert response.reasoning[0].text is None
        assert response.reasoning[0].opaque_roundtrip is not None

    @pytest.mark.anyio
    async def test_empty_requested_logprobs_fail_for_nonempty_text(self) -> None:
        dialect, _ = _dialect(_response(_message(_text("A", logprobs=[]))))

        with pytest.raises(OutputContractError, match="logprobs is empty"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    scoring=ScoringParams(sampled_logprobs=True),
                )
            )

    @pytest.mark.anyio
    async def test_structured_output_rejects_non_json_text(self) -> None:
        dialect, _ = _dialect(_response(_message(_text("not-json"))))

        with pytest.raises(
            OutputContractError, match="structured output is not valid JSON"
        ):
            await dialect.arun(
                Request(
                    input=_chat(),
                    structured_output=StructuredOutputParams(format="json_object"),
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "text",
        [
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":-Infinity}',
            '{"value":1,"value":2}',
        ],
    )
    async def test_structured_output_rejects_nonstandard_json(self, text: str) -> None:
        dialect, _ = _dialect(_response(_message(_text(text))))

        with pytest.raises(
            OutputContractError, match="structured output is not valid JSON"
        ):
            await dialect.arun(
                Request(
                    input=_chat(),
                    structured_output=StructuredOutputParams(format="json_object"),
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            (
                _response(
                    status="failed",
                    error=SimpleNamespace(code="server_error", message="boom"),
                ),
                "server_error.*boom",
            ),
            (_response(status="cancelled"), "unsupported terminal status"),
            (
                _response(status="completed", error=SimpleNamespace(code="x")),
                "completed.*error",
            ),
            (
                _response(status="completed", incomplete_reason="max_output_tokens"),
                "completed.*incomplete_details",
            ),
            (
                _response(
                    status="incomplete",
                    incomplete_reason="max_output_tokens",
                    error=SimpleNamespace(code="x"),
                ),
                "incomplete.*error",
            ),
            (
                _response(status="incomplete", incomplete_reason=None),
                "incomplete_details.reason",
            ),
            (
                _response(status="failed", error=None),
                "failed.*omitted.*error",
            ),
            (
                _response(
                    status="failed",
                    incomplete_reason="max_output_tokens",
                    error=SimpleNamespace(code="x", message="boom"),
                ),
                "failed.*incomplete_details",
            ),
            (_response(response_id=None), "response.id"),
            (
                SimpleNamespace(
                    id="resp",
                    status="completed",
                    output="bad",
                    error=None,
                    usage=None,
                    model=None,
                    reasoning=None,
                ),
                "response.output must be a sequence",
            ),
            (
                _response(_message(SimpleNamespace(type="refusal", refusal="no"))),
                "refusal has no faithful",
            ),
            (
                _response(
                    _message(
                        SimpleNamespace(
                            type="output_text",
                            text=1,
                            annotations=[],
                            logprobs=None,
                        )
                    )
                ),
                "text must be a string",
            ),
            (_response(model=1), "response.model must be a string or None"),
            (
                _response(
                    _message(
                        _text(
                            logprobs=[
                                SimpleNamespace(
                                    token="A", logprob=True, top_logprobs=[]
                                )
                            ]
                        )
                    )
                ),
                "logprob must be numeric",
            ),
            (
                _response(SimpleNamespace(type="image_generation_call")),
                "has no IR mapping",
            ),
            (
                _response(
                    _message(SimpleNamespace(type="output_audio", audio="ignored"))
                ),
                "unsupported type 'output_audio'",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="function_call",
                        call_id="call_1",
                        name="lookup",
                        arguments={},
                    )
                ),
                "arguments must be a string",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="web_search_call",
                        id="ws_1",
                        status="completed",
                        action="search",
                    )
                ),
                "action must be an object",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="web_search_call",
                        id="ws_1",
                        status="completed",
                        action={"confidence": float("nan")},
                    )
                ),
                "contains a non-finite float",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="web_search_call",
                        id="ws_1",
                        status="completed",
                        action={1: "bad"},
                    )
                ),
                "contains a non-string key",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="web_search_call",
                        id="ws_1",
                        status="completed",
                        action=object(),
                    )
                ),
                "contains unsupported value object",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="reasoning",
                        id="rs_1",
                        summary="bad",
                        content=None,
                        encrypted_content="opaque",
                    )
                ),
                "summary must be a list",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="reasoning",
                        id="rs_1",
                        summary=[SimpleNamespace(type="other", text="bad")],
                        content=None,
                        encrypted_content="opaque",
                    )
                ),
                r"summary\[0\]\.type must be 'summary_text'",
            ),
            (
                _response(
                    SimpleNamespace(
                        type="reasoning",
                        id="rs_1",
                        summary=[],
                        content=[SimpleNamespace(type="other", text="bad")],
                        encrypted_content="opaque",
                    )
                ),
                r"content\[0\]\.type must be 'reasoning_text'",
            ),
            (
                _response(
                    _message(
                        _text(
                            annotations=[
                                SimpleNamespace(type="file_citation", file_id="file")
                            ]
                        )
                    )
                ),
                "no lossless IR mapping",
            ),
            (
                _response(
                    _message(
                        _text(
                            logprobs=[
                                SimpleNamespace(
                                    token="A", logprob=float("nan"), top_logprobs=[]
                                )
                            ]
                        )
                    )
                ),
                "must be finite",
            ),
            (
                _response(usage=_usage(input_tokens=True)),
                "usage.input_tokens",
            ),
        ],
    )
    async def test_malformed_or_unrepresentable_replies_fail_loudly(
        self, raw: object, message: str
    ) -> None:
        dialect, _ = _dialect(raw)

        with pytest.raises(OutputContractError, match=message):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    async def test_reasoning_content_falls_back_when_summary_is_absent(self) -> None:
        dialect, _ = _dialect(
            _response(_reasoning(summary=None, content="visible"), _message())
        )

        response = await dialect.arun(
            Request(input=_chat(), reasoning=ReasoningParams(effort="low"))
        )

        assert response.reasoning is not None
        assert response.reasoning[0] is not None
        assert response.reasoning[0].text == "visible"

    @pytest.mark.anyio
    async def test_missing_requested_reasoning_summary_fails_loudly(self) -> None:
        dialect, _ = _dialect(_response())

        with pytest.raises(OutputContractError, match="requested visible reasoning"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    reasoning=ReasoningParams(summary="detailed"),
                )
            )


class TestStreaming:
    @pytest.mark.anyio
    async def test_stream_and_nonstream_build_identical_opaque_history(self) -> None:
        raw = _sdk_response_payload()
        nonstream_dialect, _ = _dialect(SDKResponse.model_validate(raw))
        streamed_dialect, _ = _dialect(
            _AsyncItems([_sdk_completed_event(response=raw)])
        )

        nonstream = await nonstream_dialect.arun(Request(input=_chat()))
        streamed = await streamed_dialect.arun(
            Request(input=_chat(), scheduling=SchedulingParams(stream=True))
        )

        assert nonstream.reasoning is not None
        assert streamed.reasoning is not None
        assert nonstream.reasoning[0] is not None
        assert streamed.reasoning[0] is not None
        assert (
            nonstream.reasoning[0].opaque_roundtrip
            == streamed.reasoning[0].opaque_roundtrip
        )

    @pytest.mark.anyio
    async def test_real_sdk_sse_stream_is_lifted(self) -> None:
        delta = ResponseTextDeltaEvent.model_validate(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_sdk",
                "output_index": 1,
                "content_index": 0,
                "delta": "ignored",
                "logprobs": [],
                "sequence_number": 1,
            }
        )
        terminal = _sdk_completed_event(sequence_number=2)
        body = (
            f"event: {delta.type}\ndata: {delta.model_dump_json()}\n\n"
            f"event: {terminal.type}\ndata: {terminal.model_dump_json()}\n\n"
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["stream"] is True
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        try:
            dialect = OpenAIResponsesDialect(client, "requested/model")
            response = await dialect.arun(
                Request(input=_chat(), scheduling=SchedulingParams(stream=True))
            )
        finally:
            await client.close()

        assert response.texts == ("sdk output",)
        assert response.finish_reasons == ("completed",)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("event", "error_match", "finish_reasons"),
        [
            (
                ResponseIncompleteEvent.model_validate(
                    {
                        "type": "response.incomplete",
                        "sequence_number": 1,
                        "response": _sdk_incomplete_response_payload(
                            "max_output_tokens"
                        ),
                    }
                ),
                None,
                ("max_output_tokens",),
            ),
            (_sdk_failed_event(), "server_error.*boom", None),
            (_sdk_error_event(), "bad stream", None),
        ],
    )
    async def test_real_sdk_terminal_sse_variants_are_lifted(
        self,
        event: ResponseIncompleteEvent | ResponseFailedEvent | ResponseErrorEvent,
        error_match: str | None,
        finish_reasons: tuple[str, ...] | None,
    ) -> None:
        body = f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        try:
            dialect = OpenAIResponsesDialect(client, "requested/model")
            request = Request(input=_chat(), scheduling=SchedulingParams(stream=True))
            if error_match is not None:
                with pytest.raises(OutputContractError, match=error_match):
                    await dialect.arun(request)
            else:
                response = await dialect.arun(request)
                assert response.finish_reasons == finish_reasons
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_sdk_incomplete_terminal_uses_terminal_response(self) -> None:
        terminal = ResponseIncompleteEvent.model_validate(
            {
                "type": "response.incomplete",
                "sequence_number": 1,
                "response": _sdk_incomplete_response_payload("content_filter"),
            }
        )
        dialect, create = _dialect(_AsyncItems([terminal]))

        response = await dialect.arun(
            Request(input=_chat(), scheduling=SchedulingParams(stream=True))
        )

        assert response.finish_reasons == ("content_filter",)
        assert _awaited_kwargs(create)["stream"] is True

    @pytest.mark.anyio
    async def test_sdk_failed_terminal_reports_response_error(self) -> None:
        dialect, _ = _dialect(_AsyncItems([_sdk_failed_event()]))

        with pytest.raises(OutputContractError, match="server_error.*boom"):
            await dialect.arun(
                Request(input=_chat(), scheduling=SchedulingParams(stream=True))
            )

    @pytest.mark.anyio
    async def test_sdk_error_event_reports_stream_error(self) -> None:
        dialect, _ = _dialect(_AsyncItems([_sdk_error_event()]))

        with pytest.raises(OutputContractError, match="bad stream"):
            await dialect.arun(
                Request(input=_chat(), scheduling=SchedulingParams(stream=True))
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("events", "message"),
        [
            (None, "not asynchronously iterable"),
            ([], "omitted its terminal"),
            (
                [
                    _sdk_completed_event(sequence_number=1),
                    _sdk_completed_event(sequence_number=2),
                ],
                "two terminal events",
            ),
            (
                [
                    _sdk_completed_event(
                        response=_sdk_incomplete_response_payload("max_output_tokens")
                    )
                ],
                "carried status",
            ),
            (
                [
                    SimpleNamespace(type="response.completed", response=None),
                    _sdk_completed_event(sequence_number=2),
                ],
                "omitted its response",
            ),
        ],
    )
    async def test_invalid_streams_fail_loudly(
        self, events: list[object] | None, message: str
    ) -> None:
        stream: object = object() if events is None else _AsyncItems(events)
        dialect, _ = _dialect(stream)

        with pytest.raises(OutputContractError, match=message):
            await dialect.arun(
                Request(input=_chat(), scheduling=SchedulingParams(stream=True))
            )
