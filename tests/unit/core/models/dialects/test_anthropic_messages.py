"""Tests for the native Anthropic Messages dialect.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest

import sieval.core.models.dialects.anthropic_messages as anthropic_messages_module
from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    DialectCapabilityStatus,
    MultimodalInputOptions,
    ReasoningOptions,
    StructuredOutputOptions,
    Supported,
    Unsupported,
)
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.dialect import (
    DialectError,
    OutputContractError,
    PreparedRequest,
    Rejected,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
)
from sieval.core.models.dialect_registry import DIALECT_SPECS
from sieval.core.models.dialects.anthropic_messages import (
    CAPABILITY_DECISIONS,
    AnthropicMessagesDialect,
    _AnthropicInputVerifier,
    _AnthropicToolChoiceVerifier,
    _AnthropicToolsVerifier,
    _decode_continuation,
    _finish_stream_block,
    _LegacyPlan,
    _lower_chat_input,
    _part_to_wire,
    _ReasoningSummaryVerifier,
    _tool_history_state,
    _validate_history_block,
    _validate_replay_system,
    _validate_replay_tools,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    CompletionInput,
    DialectOptions,
    HostedToolSpec,
    ImagePart,
    OpaqueContinuation,
    ReasoningParams,
    Request,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    SessionParams,
    StructuredOutputParams,
    TextPart,
    ToolCallPart,
    ToolParams,
    ToolResultPart,
)


def _chat(*messages: ChatMessage) -> ChatInput:
    return ChatInput(messages or (ChatMessage("user", (TextPart("Hello"),)),))


def _request(**changes: object) -> Request:
    values: dict[str, object] = {
        "input": _chat(),
        "sampling": SamplingParams(max_tokens=128),
    }
    values.update(changes)
    return Request(**cast(Any, values))


def _response(
    content: list[dict[str, object]] | None = None,
    *,
    stop_reason: object = "end_turn",
    usage: object | None = None,
    model: object = "claude-sonnet-4-6",
) -> dict[str, object]:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": content if content is not None else [{"type": "text", "text": "Hi"}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 7,
            "output_tokens": 3,
        },
    }


def _connection(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    credential: str | None = "anthropic-secret",
) -> AsyncHTTPJSONConnection:
    return AsyncHTTPJSONConnection(
        httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1/",
            transport=httpx.MockTransport(handler),
        ),
        credential,
    )


async def _run(
    handler: Callable[[httpx.Request], httpx.Response],
    request: Request | None = None,
    *,
    credential: str | None = "anthropic-secret",
):
    connection = _connection(handler, credential=credential)
    try:
        dialect = AnthropicMessagesDialect(connection, "claude-sonnet-4-6")
        return await dialect.arun(request or _request())
    finally:
        await connection.aclose()


def _sse(*events: Mapping[str, object]) -> bytes:
    lines: list[str] = []
    for event in events:
        event_type = cast(str, event["type"])
        lines.extend(
            (
                f"event: {event_type}",
                "data: " + json.dumps(event, separators=(",", ":")),
                "",
            )
        )
    return "\n".join(lines).encode()


def _stream_start(*, usage: Mapping[str, object] | None = None) -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            **_response(content=[], stop_reason=None, usage=usage),
            "stop_sequence": None,
        },
    }


def _stream_delta(
    *,
    stop_reason: str = "end_turn",
    usage: Mapping[str, object] | None = None,
    stop_details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    delta: dict[str, object] = {
        "stop_reason": stop_reason,
        "stop_sequence": None,
    }
    if stop_details is not None:
        delta["stop_details"] = stop_details
    return {
        "type": "message_delta",
        "delta": delta,
        "usage": usage if usage is not None else {"output_tokens": 3},
    }


class _UnbufferedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._content


class TestCapabilities:
    def test_decision_row_is_complete_and_logprobs_are_explicitly_absent(self) -> None:
        assert set(CAPABILITY_DECISIONS) == set(CAPABILITY_KEYS)
        assert isinstance(CAPABILITY_DECISIONS["input_scoring"], Unsupported)
        assert isinstance(CAPABILITY_DECISIONS["sampled_logprobs"], Unsupported)
        assert isinstance(CAPABILITY_DECISIONS["top_logprobs"], Unsupported)
        assert "does not return" in CAPABILITY_DECISIONS["sampled_logprobs"].reason
        hosted = CAPABILITY_DECISIONS["hosted_tools"]
        assert isinstance(hosted, Unsupported)
        assert hosted.reason == (
            "anthropic_messages does not support Anthropic server-hosted tools"
        )

    def test_supported_capabilities_are_first_class_registry_outcomes(self) -> None:
        spec = DIALECT_SPECS["anthropic_messages"]

        assert spec.implementation_status.value == "active"
        assert spec.request_seed_support.value == "unsupported"
        for key in (
            "reasoning",
            "function_tools",
            "structured_output",
            "opaque_continuation",
            "multimodal_input",
            "prefill",
        ):
            assert isinstance(CAPABILITY_DECISIONS[key], Supported)
            assert spec.capability_outcomes[key] is DialectCapabilityStatus.SUPPORTED

    @pytest.mark.parametrize(
        ("key", "options", "message"),
        [
            ("reasoning", ReasoningOptions(effort="minimal"), "effort"),
            ("reasoning", ReasoningOptions(budget_tokens=100), "1024"),
            ("reasoning", ReasoningOptions(summary="detailed"), "summary"),
            (
                "structured_output",
                StructuredOutputOptions(("json_object",)),
                "json_schema",
            ),
            (
                "multimodal_input",
                cast(Any, MultimodalInputOptions.__new__(MultimodalInputOptions)),
                "modalities",
            ),
        ],
    )
    def test_capability_domains_are_dialect_validated(
        self, key: str, options: object, message: str
    ) -> None:
        decision = cast(Supported, CAPABILITY_DECISIONS[cast(Any, key)])
        if key == "multimodal_input":
            object.__setattr__(options, "modalities", ("audio",))

        with pytest.raises((TypeError, ValueError), match=message):
            decision.binding.validate_config(cast(Any, options))


class TestLoweringAndLift:
    @pytest.mark.anyio
    async def test_native_nonstream_request_uses_headers_system_and_final_body(
        self,
    ) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["headers"] = dict(request.headers)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage("system", (TextPart("Be concise."),)),
                ChatMessage("user", (TextPart("Hello"),)),
            ),
            sampling=SamplingParams(
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                stop=("END",),
            ),
        )
        result = await _run(handler, req)

        assert seen["url"] == "https://api.anthropic.com/v1/messages"
        headers = cast(dict[str, str], seen["headers"])
        assert headers["x-api-key"] == "anthropic-secret"
        assert headers["anthropic-version"] == "2023-06-01"
        assert seen["body"] == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
            "stream": False,
            "system": [{"type": "text", "text": "Be concise."}],
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
        }
        assert result.texts == ("Hi",)
        assert result.finish_reasons == ("end_turn",)
        assert result.response_model == "claude-sonnet-4-6"
        assert result.request_params == {
            "max_tokens": 64,
            "stream": False,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
        }

    @pytest.mark.anyio
    async def test_missing_credential_does_not_fabricate_auth_header(self) -> None:
        seen_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.update(request.headers)
            return httpx.Response(200, json=_response())

        await _run(handler, credential=None)

        assert "x-api-key" not in seen_headers
        assert seen_headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.anyio
    async def test_usage_includes_cache_partitions_and_reasoning_breakdown(
        self,
    ) -> None:
        usage = {
            "input_tokens": 5,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 11,
            "output_tokens": 13,
            "output_tokens_details": {"thinking_tokens": 3},
            "service_tier": "standard",
        }

        result = await _run(lambda _: httpx.Response(200, json=_response(usage=usage)))

        assert result.usage is not None
        assert result.usage.input_tokens == 23
        assert result.usage.output_tokens == 13
        assert result.usage.total_tokens == 36
        assert result.usage.cached_tokens == 11
        assert result.usage.reasoning_tokens == 3

    @pytest.mark.anyio
    async def test_context_window_limit_reason_is_preserved_verbatim(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(stop_reason="model_context_window_exceeded"),
            )
        )

        assert result.finish_reasons == ("model_context_window_exceeded",)

    @pytest.mark.anyio
    async def test_zero_max_tokens_is_preserved_for_cache_warming(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response(
                    [],
                    stop_reason="max_tokens",
                    usage={"input_tokens": 7, "output_tokens": 0},
                ),
            )

        result = await _run(
            handler,
            _request(sampling=SamplingParams(max_tokens=0)),
        )

        assert seen["max_tokens"] == 0
        assert result.texts == ("",)
        assert result.usage is not None
        assert result.usage.output_tokens == 0

    @pytest.mark.anyio
    async def test_image_and_tool_history_lower_without_reordering(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        TextPart("look"),
                        ImagePart(
                            data="aGVsbG8=",
                            media_type="image/png",
                        ),
                    ),
                ),
                ChatMessage(
                    "assistant",
                    (ToolCallPart("toolu_1", "weather", {"city": "Paris"}),),
                ),
                ChatMessage(
                    "tool",
                    (ToolResultPart("toolu_1", {"temp": 20}, is_error=True),),
                ),
            )
        )

        await _run(handler, req)

        assert seen["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "weather",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": '{"temp":20}',
                        "is_error": True,
                    }
                ],
            },
        ]

    @pytest.mark.anyio
    async def test_supported_prefill_lowers_final_assistant_message(self) -> None:
        decision = CAPABILITY_DECISIONS["prefill"]
        assert isinstance(decision, Supported)

        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage("user", (TextPart("Complete this"),)),
                    ChatMessage("assistant", (TextPart("Partial answer"),)),
                )
            ),
        )

        assert cast(list[object], seen["messages"])[-1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "Partial answer"}],
        }

    @pytest.mark.anyio
    async def test_functions_choice_parallel_and_tool_use_lift(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response(
                    [
                        {"type": "text", "text": "Checking"},
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "weather",
                            "input": {"city": "Tokyo"},
                        },
                    ],
                    stop_reason="tool_use",
                ),
            )

        req = _request(
            tools=ToolParams(
                functions=(
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                            "strict": True,
                        },
                    },
                ),
                choice={"type": "function", "function": {"name": "weather"}},
                parallel=False,
            )
        )

        result = await _run(handler, req)

        assert seen["tools"] == [
            {
                "name": "weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                "strict": True,
            }
        ]
        assert seen["tool_choice"] == {
            "type": "tool",
            "name": "weather",
            "disable_parallel_tool_use": True,
        }
        assert result.finish_reasons == ("tool_use",)
        assert result.tool_calls is not None
        assert result.tool_calls[0].call_id == "toolu_2"
        assert result.tool_calls[0].arguments == {"city": "Tokyo"}

    @pytest.mark.anyio
    async def test_reasoning_budget_summary_and_opaque_history(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "thinking",
                            "thinking": "I should answer briefly.",
                            "signature": "signed-state",
                        },
                        {"type": "text", "text": "42"},
                    ],
                    usage={
                        "input_tokens": 8,
                        "output_tokens": 9,
                        "output_tokens_details": {"thinking_tokens": 6},
                    },
                ),
            )

        req = _request(
            sampling=SamplingParams(max_tokens=4096),
            reasoning=ReasoningParams(budget_tokens=2048, summary="auto"),
        )
        result = await _run(handler, req)

        assert seen["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
            "display": "summarized",
        }
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == "I should answer briefly."
        assert result.reasoning[0].thinking_tokens == 6
        opaque = result.reasoning[0].opaque_roundtrip
        assert opaque is not None
        bundle = json.loads(opaque)
        assert bundle["messages"][-1]["content"][0]["signature"] == "signed-state"

    @pytest.mark.anyio
    async def test_opaque_continuation_replays_complete_history_untouched(self) -> None:
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "thinking",
                            "thinking": "secret reasoning",
                            "signature": "sig-1",
                        },
                        {
                            "type": "text",
                            "text": "first answer",
                            "citations": [],
                        },
                    ]
                ),
            ),
            _request(
                input=_chat(
                    ChatMessage("system", (TextPart("Stable system"),)),
                    ChatMessage("user", (TextPart("first question"),)),
                ),
                sampling=SamplingParams(max_tokens=4096),
                reasoning=ReasoningParams(budget_tokens=2048),
            ),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        seen: dict[str, object] = {}

        def second_handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "redacted_thinking",
                            "data": "encrypted-next",
                        },
                        {"type": "text", "text": "second answer"},
                    ]
                ),
            )

        second = await _run(
            second_handler,
            _request(
                input=_chat(ChatMessage("user", (TextPart("follow up"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
                ),
            ),
        )

        assert seen["system"] == [{"type": "text", "text": "Stable system"}]
        assert cast(list[object], seen["messages"])[-2] == {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "secret reasoning",
                    "signature": "sig-1",
                },
                {"type": "text", "text": "first answer", "citations": []},
            ],
        }
        assert cast(list[object], seen["messages"])[-1] == {
            "role": "user",
            "content": [{"type": "text", "text": "follow up"}],
        }
        assert second.reasoning is not None
        assert second.reasoning[0] is not None
        assert second.reasoning[0].opaque_roundtrip is not None

    @pytest.mark.anyio
    async def test_continuation_remains_replayable_without_new_thinking_block(
        self,
    ) -> None:
        opaque = json.dumps(
            {
                "version": 2,
                "system": [],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "prior",
                                "signature": "prior-signature",
                            },
                            {"type": "text", "text": "prior answer"},
                        ],
                    }
                ],
                "tools": None,
                "open_turn": False,
                "reasoning": {"thinking": None, "effort": None},
            }
        )
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response([{"type": "text", "text": "next answer"}]),
            ),
            _request(
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
                )
            ),
        )

        assert result.reasoning is not None and result.reasoning[0] is not None
        next_opaque = cast(str, result.reasoning[0].opaque_roundtrip)
        next_history = json.loads(next_opaque)["messages"]
        assert next_history[-1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "next answer"}],
        }

    @pytest.mark.anyio
    async def test_json_schema_output_uses_output_config_and_parses_null(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response([{"type": "text", "text": "null"}]),
            )

        req = _request(
            structured_output=StructuredOutputParams(
                format="json_schema",
                schema={"type": ["object", "null"]},
                strict=True,
            )
        )
        result = await _run(handler, req)

        assert seen["output_config"] == {
            "format": {
                "type": "json_schema",
                "schema": {"type": ["object", "null"]},
            }
        }
        assert result.structured_output is not None
        assert result.structured_output.value is None

    @pytest.mark.anyio
    async def test_effort_and_schema_share_one_output_config(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response([{"type": "text", "text": '{"ok":true}'}]),
            )

        await _run(
            handler,
            _request(
                reasoning=ReasoningParams(effort="medium"),
                structured_output=StructuredOutputParams(
                    format="json_schema",
                    schema={"type": "object"},
                ),
            ),
        )

        assert seen["output_config"] == {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        }

    @pytest.mark.anyio
    async def test_summary_none_explicitly_requests_omitted_adaptive_thinking(
        self,
    ) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_response())

        await _run(handler, _request(reasoning=ReasoningParams(summary="none")))

        assert seen["thinking"] == {"type": "adaptive", "display": "omitted"}

    @pytest.mark.anyio
    async def test_open_tool_turn_requires_exact_prior_reasoning_configuration(
        self,
    ) -> None:
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "thinking",
                            "thinking": "choose a tool",
                            "signature": "signed-tool-turn",
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"key": "x"},
                        },
                    ],
                    stop_reason="tool_use",
                ),
            ),
            _request(
                sampling=SamplingParams(max_tokens=2048),
                reasoning=ReasoningParams(budget_tokens=1024, summary="none"),
            ),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        bundle = json.loads(opaque)
        assert bundle["open_turn"] is True
        assert bundle["reasoning"] == {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,
                "display": "omitted",
            },
            "effort": None,
        }

        calls = 0

        def rejected_handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_response())

        with pytest.raises(RequestAuditError, match="reuse the exact thinking"):
            await _run(
                rejected_handler,
                _request(
                    sampling=SamplingParams(max_tokens=2048),
                    input=_chat(
                        ChatMessage("tool", (ToolResultPart("toolu_1", {"value": 1}),))
                    ),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation(
                            "anthropic_messages", opaque
                        )
                    ),
                ),
            )
        assert calls == 0

        seen: dict[str, object] = {}

        def accepted_handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_response())

        await _run(
            accepted_handler,
            _request(
                sampling=SamplingParams(max_tokens=2048),
                input=_chat(
                    ChatMessage("tool", (ToolResultPart("toolu_1", {"value": 1}),))
                ),
                reasoning=ReasoningParams(budget_tokens=1024, summary="none"),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
                ),
            ),
        )
        assert seen["thinking"] == {
            "type": "enabled",
            "budget_tokens": 1024,
            "display": "omitted",
        }

    @pytest.mark.anyio
    async def test_open_tool_turn_preserves_reasoning_config_without_new_thinking(
        self,
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                    stop_reason="tool_use",
                ),
            ),
            _request(reasoning=ReasoningParams(summary="none")),
        )

        assert result.reasoning is not None and result.reasoning[0] is not None
        bundle = json.loads(cast(str, result.reasoning[0].opaque_roundtrip))
        assert bundle["open_turn"] is True
        assert bundle["reasoning"] == {
            "thinking": {"type": "adaptive", "display": "omitted"},
            "effort": None,
        }

    @pytest.mark.anyio
    async def test_continuation_requires_exact_final_wire_tools(self) -> None:
        functions = (
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object"},
            },
            {
                "type": "function",
                "name": "calculate",
                "parameters": {"type": "object"},
            },
        )
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                    stop_reason="tool_use",
                ),
            ),
            _request(
                reasoning=ReasoningParams(summary="none"),
                tools=ToolParams(functions=functions),
            ),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        assert json.loads(opaque)["tools"] == [
            {
                "name": "lookup",
                "description": "Look up a value",
                "input_schema": {"type": "object"},
            },
            {
                "name": "calculate",
                "input_schema": {"type": "object"},
            },
        ]

        follow_up = _request(
            input=_chat(
                ChatMessage("tool", (ToolResultPart("toolu_1", {"value": 1}),))
            ),
            reasoning=ReasoningParams(summary="none"),
            tools=ToolParams(functions=functions),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
            ),
        )
        await _run(
            lambda _: httpx.Response(200, json=_response()),
            follow_up,
        )

        changed = dict(functions[0])
        changed["description"] = "Changed"
        mismatches = (
            ToolParams(),
            ToolParams(functions=(changed, functions[1])),
            ToolParams(functions=tuple(reversed(functions))),
        )
        for tools in mismatches:
            calls = 0

            def rejected_handler(_: httpx.Request) -> httpx.Response:
                nonlocal calls
                calls += 1
                return httpx.Response(200, json=_response())

            with pytest.raises(RequestAuditError, match="must exactly match"):
                await _run(
                    rejected_handler,
                    _request(
                        input=follow_up.input,
                        reasoning=follow_up.reasoning,
                        tools=tools,
                        session=follow_up.session,
                    ),
                )
            assert calls == 0


class TestStreaming:
    @pytest.mark.anyio
    async def test_streaming_reconstructs_text_thinking_signature_and_tool_json(
        self,
    ) -> None:
        content = _sse(
            _stream_start(usage={"input_tokens": 7, "output_tokens": 1}),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "think"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "answer"},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_stream",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"x"'},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": ":1}"},
            },
            {"type": "content_block_stop", "index": 2},
            _stream_delta(
                stop_reason="tool_use",
                usage={
                    "output_tokens": 12,
                    "output_tokens_details": {"thinking_tokens": 4},
                },
            ),
            {"type": "message_stop"},
        )

        result = await _run(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            ),
            _request(
                sampling=SamplingParams(max_tokens=4096),
                reasoning=ReasoningParams(budget_tokens=2048, summary="auto"),
                scheduling=SchedulingParams(stream=True),
            ),
        )

        assert result.texts == ("answer",)
        assert result.finish_reasons == ("tool_use",)
        assert result.reasoning is not None and result.reasoning[0] is not None
        assert result.reasoning[0].text == "think"
        assert result.reasoning[0].thinking_tokens == 4
        assert result.tool_calls is not None
        assert result.tool_calls[0].arguments == {"x": 1}
        assert result.usage is not None
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 12

    @pytest.mark.anyio
    async def test_streaming_refusal_is_a_scoreable_terminal_outcome(self) -> None:
        content = _sse(
            _stream_start(),
            _stream_delta(
                stop_reason="refusal",
                stop_details={
                    "type": "refusal",
                    "category": "safety",
                    "explanation": "Request refused",
                },
            ),
            {"type": "message_stop"},
        )

        result = await _run(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            ),
            _request(
                reasoning=ReasoningParams(summary="auto"),
                scheduling=SchedulingParams(stream=True),
            ),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("refusal",)
        assert result.reasoning is None

    @pytest.mark.anyio
    async def test_streaming_refusal_discards_unfinished_partial_output(self) -> None:
        content = _sse(
            _stream_start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
            _stream_delta(
                stop_reason="refusal",
                stop_details={"type": "refusal"},
            ),
            {"type": "message_stop"},
        )

        result = await _run(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("refusal",)

    @pytest.mark.anyio
    async def test_streaming_refusal_discards_completed_partial_text(self) -> None:
        content = _sse(
            _stream_start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "completed partial"},
            },
            {"type": "content_block_stop", "index": 0},
            _stream_delta(
                stop_reason="refusal",
                stop_details={"type": "refusal"},
            ),
            {"type": "message_stop"},
        )

        result = await _run(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("refusal",)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("events", "message"),
        [
            ((_stream_delta(), {"type": "message_stop"}), "preceded message_start"),
            ((_stream_start(), {"type": "message_stop"}), "omitted message_delta"),
            ((_stream_start(), _stream_delta()), "omitted message_stop"),
            (
                (
                    _stream_start(),
                    {"type": "error", "error": {"message": "overloaded"}},
                ),
                "overloaded",
            ),
            (
                (
                    _stream_start(),
                    _stream_delta(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {"type": "message_stop"},
                ),
                "content after message_delta",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    _stream_delta(),
                    {"type": "message_stop"},
                ),
                "preceded content_block_stop",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "text_delta",
                            "text": "x",
                            "future": True,
                        },
                    },
                ),
                "unsupported field",
            ),
            (({"type": "ping", "future": True},), "unsupported field"),
            (
                (
                    _stream_start(usage={"input_tokens": 7, "output_tokens": 1}),
                    _stream_delta(usage={}),
                    {"type": "message_stop"},
                ),
                "message_delta.usage.output_tokens",
            ),
            (
                (
                    _stream_start(usage={"input_tokens": 7, "output_tokens": 1}),
                    _stream_delta(usage={"output_tokens": None}),
                    {"type": "message_stop"},
                ),
                "message_delta.usage.output_tokens",
            ),
        ],
    )
    async def test_malformed_streams_fail_loudly(
        self, events: tuple[Mapping[str, object], ...], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse(*events),
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )


class TestPreIOGuards:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "message"),
        [
            (Request(_chat()), "requires sampling.max_tokens"),
            (
                _request(input=_chat(ChatMessage("developer", (TextPart("hidden"),)))),
                "developer-role",
            ),
            (
                _request(input=_chat(ChatMessage("system", (TextPart("rules"),)))),
                "at least one conversational message",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart("u"),)),
                        ChatMessage("system", (TextPart("late"),)),
                    )
                ),
                "must precede",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart(cast(Any, 7)),)),
                    )
                ),
                r"content\[0\]\.text must be a string",
            ),
            (
                _request(
                    input=_chat(ChatMessage("assistant", (ImagePart(url="https://x"),)))
                ),
                "assistant messages",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (ImagePart(data="x", media_type=None),))
                    )
                ),
                "require media_type",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, seed=1)),
                "per-request seed",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, n=2)),
                "one choice",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, temperature=1.1)),
                "between 0 and 1",
            ),
            (
                _request(
                    sampling=SamplingParams(max_tokens=2048),
                    reasoning=ReasoningParams(budget_tokens=2048),
                ),
                "max_tokens must exceed reasoning.budget_tokens",
            ),
            (
                _request(
                    sampling=SamplingParams(max_tokens=2048, temperature=0.5),
                    reasoning=ReasoningParams(budget_tokens=1024),
                ),
                "temperature is incompatible",
            ),
            (
                _request(
                    sampling=SamplingParams(max_tokens=2048, top_p=0.9),
                    reasoning=ReasoningParams(budget_tokens=1024),
                ),
                "between 0.95 and 1",
            ),
            (
                _request(
                    sampling=SamplingParams(max_tokens=2048, top_k=20),
                    reasoning=ReasoningParams(budget_tokens=1024),
                ),
                "top_k is incompatible",
            ),
            (
                _request(reasoning=ReasoningParams(summary="detailed")),
                "only summary",
            ),
            (
                _request(tools=ToolParams(choice="required")),
                "require function tools",
            ),
            (
                _request(
                    sampling=SamplingParams(max_tokens=2048),
                    reasoning=ReasoningParams(budget_tokens=1024),
                    tools=ToolParams(
                        functions=(
                            {
                                "type": "function",
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            },
                        ),
                        choice="required",
                    ),
                ),
                "manual Anthropic thinking supports only",
            ),
            (
                _request(tools=ToolParams(hosted=(HostedToolSpec("web_search"),))),
                "unavailable capability|not supported by anthropic_messages",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(format="json_object")
                ),
                "requires json_schema",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema",
                        schema={"type": "object"},
                        name="named",
                    )
                ),
                "no schema name",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart("complete"),)),
                        ChatMessage("assistant", (TextPart("{"),)),
                    ),
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={"type": "object"}
                    ),
                ),
                "incompatible with assistant prefill",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart("continue"),)),
                        ChatMessage("assistant", (TextPart("partial"),)),
                    ),
                    sampling=SamplingParams(max_tokens=2048),
                    reasoning=ReasoningParams(budget_tokens=1024),
                ),
                "thinking is incompatible with assistant prefill",
            ),
            (
                _request(session=SessionParams(previous_response_id="msg_previous")),
                "unavailable capability|no previous-response-id",
            ),
            (
                _request(
                    dialect_options=DialectOptions(
                        "anthropic_messages", {"service_tier": "auto"}
                    )
                ),
                "does not support raw Anthropic request passthrough",
            ),
            (
                _request(
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation(
                            "anthropic_messages", '{"version":1}'
                        )
                    )
                ),
                "invalid envelope",
            ),
        ],
    )
    async def test_invalid_requests_make_zero_http_calls(
        self, req: Request, message: str
    ) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_response())

        with pytest.raises(
            (DialectError, RequestAuditError, ValueError), match=message
        ):
            await _run(handler, req)

        assert calls == 0

    @pytest.mark.anyio
    async def test_continuation_cannot_be_combined_with_new_system_authority(
        self,
    ) -> None:
        opaque = json.dumps(
            {
                "version": 2,
                "system": [],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "x",
                                "signature": "sig",
                            }
                        ],
                    }
                ],
                "tools": None,
                "open_turn": False,
                "reasoning": {"thinking": None, "effort": None},
            }
        )
        req = _request(
            input=_chat(
                ChatMessage("system", (TextPart("new"),)),
                ChatMessage("user", (TextPart("next"),)),
            ),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
            ),
        )

        with pytest.raises(RequestAuditError, match="authority ambiguous"):
            await _run(
                lambda _: pytest.fail("HTTP must not run"),
                req,
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tools", "message"),
        [
            ([], "null or non-empty"),
            (["not-an-object"], "tools are malformed"),
            (
                [{"name": "lookup", "input_schema": {}, "future": True}],
                "unsupported fields",
            ),
        ],
    )
    async def test_malformed_continuation_tools_fail_before_http(
        self, tools: object, message: str
    ) -> None:
        opaque = json.dumps(
            {
                "version": 2,
                "system": [],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "prior",
                                "signature": "signature",
                            }
                        ],
                    }
                ],
                "tools": tools,
                "open_turn": False,
                "reasoning": {"thinking": None, "effort": None},
            }
        )

        with pytest.raises(DialectError, match=message):
            await _run(
                lambda _: pytest.fail("HTTP must not run"),
                _request(
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation(
                            "anthropic_messages", opaque
                        )
                    )
                ),
            )


class TestResponseGuards:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("response", "message"),
        [
            ({**_response(), "type": "completion"}, "response.type"),
            ({**_response(), "role": "user"}, "response.role"),
            ({**_response(), "model": None}, "response.model"),
            ({**_response(), "stop_reason": "unknown"}, "unsupported"),
            ({**_response(), "future_semantic": {}}, "unsupported field"),
            (
                {**_response(), "container": {"id": "container_1"}},
                "container has no shared-IR mapping",
            ),
            (
                {**_response(), "stop_reason": "refusal"},
                "response.stop_details must be an object",
            ),
            (
                {
                    **_response(),
                    "stop_reason": "refusal",
                    "stop_details": {"type": "future"},
                },
                "stop_details.type must be 'refusal'",
            ),
            (
                {
                    **_response(),
                    "stop_reason": "refusal",
                    "stop_details": {"type": "refusal", "category": 1},
                },
                "stop_details.category must be a string or null",
            ),
            (
                {
                    **_response(),
                    "stop_reason": "refusal",
                    "stop_details": {"type": "refusal", "future": True},
                },
                "stop_details has unsupported field",
            ),
            (
                {
                    **_response(),
                    "stop_details": {"type": "refusal"},
                },
                "stop_details must be null unless stop_reason is 'refusal'",
            ),
            (
                {**_response(), "stop_sequence": "unexpected"},
                "must be null",
            ),
            ({**_response(), "usage": None}, "response.usage"),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "server_tool_use": {
                            "web_fetch_requests": 0,
                            "web_search_requests": 1,
                        },
                    }
                ),
                "server-tool usage",
            ),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_creation_input_tokens": 3,
                        "cache_creation": {
                            "ephemeral_1h_input_tokens": 1,
                            "ephemeral_5m_input_tokens": 1,
                        },
                    }
                ),
                "breakdown disagrees",
            ),
            (
                _response([{"type": "server_tool_use", "id": "x"}]),
                "no shared-IR mapping",
            ),
            (
                _response(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                            "caller": None,
                        }
                    ],
                    stop_reason="tool_use",
                ),
                "delegated/toolset",
            ),
            (
                _response(
                    [
                        {
                            "type": "thinking",
                            "thinking": "x",
                            "signature": "",
                        }
                    ]
                ),
                "signature",
            ),
            (
                _response(
                    [
                        {
                            "type": "text",
                            "text": "x",
                            "citations": [{"type": "page_location"}],
                        }
                    ]
                ),
                "citations",
            ),
        ],
    )
    async def test_malformed_native_response_fails_lift(
        self, response: Mapping[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=response))

    @pytest.mark.anyio
    async def test_refusal_is_a_scoreable_terminal_outcome(self) -> None:
        req = _request(
            reasoning=ReasoningParams(summary="auto"),
            structured_output=StructuredOutputParams(
                format="json_schema", schema={"type": "object"}
            ),
        )
        response = {
            **_response(
                [
                    {"type": "thinking", "thinking": "partial", "signature": "sig"},
                    {"type": "text", "text": "partial answer"},
                ],
                stop_reason="refusal",
            ),
            "stop_details": {
                "type": "refusal",
                "category": "reasoning_extraction",
                "explanation": None,
            },
        }

        result = await _run(lambda _: httpx.Response(200, json=response), req)

        assert result.texts == ("",)
        assert result.finish_reasons == ("refusal",)
        assert result.reasoning is None
        assert result.structured_output is None
        assert result.tool_calls is None
        assert result.usage is not None

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ([{"type": "quantum_widget"}], "has no shared-IR mapping"),
            ([{"type": "text", "text": 5}], "must be a string"),
            (["not an object"], "must be an object"),
        ],
    )
    async def test_refusal_content_blocks_are_still_validated(
        self, content: list[object], message: str
    ) -> None:
        """Discarding a refusal's content does not mean skipping its checks."""
        response = {
            **_response(cast(Any, content), stop_reason="refusal"),
            "stop_details": {"type": "refusal"},
        }

        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=response))

    @pytest.mark.anyio
    async def test_adaptive_summary_allows_a_turn_without_thinking(self) -> None:
        req = _request(reasoning=ReasoningParams(summary="auto"))

        result = await _run(
            lambda _: httpx.Response(200, json=_response()),
            req,
        )

        assert result.texts == ("Hi",)
        assert result.reasoning == (None,)

    @pytest.mark.anyio
    async def test_low_effort_adaptive_summary_allows_no_thinking(self) -> None:
        req = _request(reasoning=ReasoningParams(effort="low", summary="auto"))

        result = await _run(
            lambda _: httpx.Response(200, json=_response()),
            req,
        )

        assert result.reasoning == (None,)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "stop_reason", ["max_tokens", "model_context_window_exceeded"]
    )
    async def test_truncated_adaptive_summary_allows_no_thinking(
        self, stop_reason: str
    ) -> None:
        req = _request(reasoning=ReasoningParams(summary="auto"))

        result = await _run(
            lambda _: httpx.Response(200, json=_response(stop_reason=stop_reason)),
            req,
        )

        assert result.finish_reasons == (stop_reason,)
        assert result.reasoning == (None,)

    @pytest.mark.anyio
    async def test_truncated_adaptive_summary_allows_incomplete_thinking(self) -> None:
        req = _request(reasoning=ReasoningParams(summary="auto"))

        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [{"type": "thinking", "thinking": "   ", "signature": "sig"}],
                    stop_reason="max_tokens",
                ),
            ),
            req,
        )

        assert result.finish_reasons == ("max_tokens",)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == "   "

    @pytest.mark.anyio
    async def test_adaptive_summary_accepts_redacted_thinking(self) -> None:
        req = _request(reasoning=ReasoningParams(summary="auto"))

        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "answer"},
                    ]
                ),
            ),
            req,
        )

        assert result.texts == ("answer",)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text is None
        assert result.reasoning[0].opaque_roundtrip is not None

    @pytest.mark.anyio
    async def test_manual_thinking_requires_replayable_thinking_state(self) -> None:
        req = _request(
            sampling=SamplingParams(max_tokens=2048),
            reasoning=ReasoningParams(budget_tokens=1024),
        )

        with pytest.raises(OutputContractError, match="manual-thinking state"):
            await _run(lambda _: httpx.Response(200, json=_response()), req)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (
                _response(stop_reason="tool_use"),
                "requires a tool_use content block",
            ),
            (
                _response(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ]
                ),
                "requires stop_reason='tool_use'",
            ),
            (
                _response(stop_reason="pause_turn"),
                "pause_turn.*unsupported",
            ),
        ],
    )
    async def test_stop_reason_and_content_state_must_agree(
        self, response: Mapping[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=response))

    @pytest.mark.anyio
    async def test_structured_output_rejects_non_json_text(self) -> None:
        req = _request(
            structured_output=StructuredOutputParams(
                format="json_schema", schema={"type": "object"}
            )
        )

        with pytest.raises(OutputContractError, match="not valid JSON"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response([{"type": "text", "text": "not-json"}]),
                ),
                req,
            )

    @pytest.mark.anyio
    async def test_http_error_is_not_lifted_as_success(self) -> None:
        with pytest.raises(httpx.HTTPStatusError):
            await _run(
                lambda _: httpx.Response(
                    401,
                    json={"type": "error", "error": {"message": "bad key"}},
                )
            )

    @pytest.mark.anyio
    async def test_stream_http_error_is_raised_before_sse_lift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lift = AsyncMock(side_effect=AssertionError("SSE lift must not run"))
        monkeypatch.setattr(
            anthropic_messages_module,
            "_terminal_stream_message",
            lift,
        )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _run(
                lambda _: httpx.Response(
                    429,
                    stream=_UnbufferedAsyncByteStream(
                        b'{"type":"error","error":{"type":"overloaded_error"}}'
                    ),
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

        assert exc_info.value.response.status_code == 429
        assert "overloaded_error" in exc_info.value.response.text
        lift.assert_not_awaited()


class TestAuditAndLowLevelContract:
    def test_low_level_system_lowering_rejects_non_text_content(self) -> None:
        input_ = _chat(
            ChatMessage("system", (ImagePart(url="https://example.com/image.png"),)),
            ChatMessage("user", (TextPart("Hello"),)),
        )

        with pytest.raises(DialectError, match="system accepts text content only"):
            _lower_chat_input(input_)

    def test_low_level_text_lowering_rejects_non_string_content(self) -> None:
        with pytest.raises(DialectError, match="text content must be a string"):
            _part_to_wire(TextPart(cast(Any, 7)))

    def test_budget_conflict_is_attributed_to_sampling_max_tokens(self) -> None:
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        dialect = AnthropicMessagesDialect(connection, "claude")
        req = _request(
            sampling=SamplingParams(max_tokens=2048),
            reasoning=ReasoningParams(budget_tokens=2048),
        )
        audit = RequestAudit(active_request_leaves(req))

        dialect.validate_request(req, audit, _LegacyPlan())

        decision = audit.decisions["sampling.max_tokens"]
        assert isinstance(decision, Rejected)
        assert decision.reason == (
            "Anthropic max_tokens must exceed reasoning.budget_tokens"
        )
        assert "reasoning.budget_tokens" not in audit.decisions

    def test_mutating_lowered_input_is_detected_by_independent_verifier(self) -> None:
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        dialect = AnthropicMessagesDialect(connection, "claude")
        req = _request()
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, cast(Any, object()))
        prepared = dialect.prepare(req, audit)
        body = prepared.thaw_body()
        messages = cast(list[dict[str, object]], body["messages"])
        cast(list[dict[str, object]], messages[0]["content"])[0]["text"] = "changed"
        mutated = PreparedRequest(prepared.operation, body, prepared.context)

        with pytest.raises(RequestAuditError, match="body.messages changed"):
            audit.finish(mutated)

    def test_mutating_lowered_tools_is_detected_by_independent_verifier(self) -> None:
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        dialect = AnthropicMessagesDialect(connection, "claude")
        req = _request(
            tools=ToolParams(
                functions=(
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                )
            )
        )
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, cast(Any, object()))
        prepared = dialect.prepare(req, audit)
        body = prepared.thaw_body()
        tools = cast(list[dict[str, object]], body["tools"])
        tools[0]["name"] = "changed"

        with pytest.raises(RequestAuditError, match=r"body.tools\[0\] changed"):
            audit.finish(PreparedRequest(prepared.operation, body, prepared.context))

    def test_mutating_lowered_tool_choice_is_detected_by_independent_verifier(
        self,
    ) -> None:
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        dialect = AnthropicMessagesDialect(connection, "claude")
        req = _request(
            tools=ToolParams(
                functions=(
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                ),
                choice="auto",
                parallel=False,
            )
        )
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, cast(Any, object()))
        prepared = dialect.prepare(req, audit)
        body = prepared.thaw_body()
        choice = cast(dict[str, object], body["tool_choice"])
        choice["disable_parallel_tool_use"] = False

        with pytest.raises(RequestAuditError, match="body.tool_choice changed"):
            audit.finish(PreparedRequest(prepared.operation, body, prepared.context))

    @pytest.mark.anyio
    async def test_request_params_are_derived_from_actual_prepared_body(self) -> None:
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        try:
            dialect = AnthropicMessagesDialect(connection, "claude")
            req = _request(sampling=SamplingParams(max_tokens=10, temperature=0.2))
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, object()))
            prepared = dialect.prepare(req, audit)
            body = prepared.thaw_body()
            body["temperature"] = 0.9
            final = PreparedRequest(prepared.operation, body, prepared.context)

            response = await dialect.execute(final)

            assert response.request_params is not None
            assert response.request_params["temperature"] == 0.9
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_opaque_bundle_is_credential_stable_and_tool_sensitive(self) -> None:
        async def opaque_for(
            credential: str,
            description: str,
        ) -> str:
            connection = _connection(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "type": "redacted_thinking",
                                "data": "stable-opaque-state",
                            },
                            {"type": "text", "text": "answer"},
                        ]
                    ),
                ),
                credential=credential,
            )
            try:
                dialect = AnthropicMessagesDialect(connection, "claude")
                response = await dialect.arun(
                    _request(
                        reasoning=ReasoningParams(summary="none"),
                        tools=ToolParams(
                            functions=(
                                {
                                    "type": "function",
                                    "name": "lookup",
                                    "description": description,
                                    "parameters": {"type": "object"},
                                },
                            )
                        ),
                    )
                )
                assert response.reasoning is not None
                assert response.reasoning[0] is not None
                return cast(str, response.reasoning[0].opaque_roundtrip)
            finally:
                await connection.aclose()

        first = await opaque_for("SECRET-A", "Stable description")
        second = await opaque_for("SECRET-B", "Stable description")
        changed = await opaque_for("SECRET-A", "Changed description")

        assert first == second
        assert first != changed
        assert "SECRET-A" not in first
        assert "SECRET-B" not in second

    @pytest.mark.anyio
    async def test_execute_rejects_wrong_operation_and_context_before_http(
        self,
    ) -> None:
        connection = _connection(lambda _: pytest.fail("HTTP must not run"))
        try:
            dialect = AnthropicMessagesDialect(connection, "claude")
            with pytest.raises(DialectError, match="unexpected Anthropic operation"):
                await dialect.execute(
                    PreparedRequest("responses.create", {"stream": False})
                )
            with pytest.raises(DialectError, match="invalid context"):
                await dialect.execute(
                    PreparedRequest("messages.create", {"stream": False})
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("model", [None, 7, "", "other-model"])
    async def test_execute_rejects_invalid_prepared_model_before_http(
        self, model: object
    ) -> None:
        connection = _connection(lambda _: pytest.fail("HTTP must not run"))
        try:
            dialect = AnthropicMessagesDialect(connection, "claude")
            req = _request()
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, _LegacyPlan())
            audit.raise_rejections()
            prepared = dialect.prepare(req, audit)
            audit.finish(prepared)
            body = prepared.thaw_body()
            if model is None:
                body.pop("model")
            else:
                body["model"] = cast(Any, model)

            with pytest.raises(DialectError, match="prepared body.*model"):
                await dialect.execute(
                    PreparedRequest(prepared.operation, body, prepared.context)
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_execute_revalidates_continuation_tools_after_audit(self) -> None:
        functions = (
            {
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        )
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                    stop_reason="tool_use",
                ),
            ),
            _request(
                reasoning=ReasoningParams(summary="none"),
                tools=ToolParams(functions=functions),
            ),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        req = _request(
            input=_chat(ChatMessage("tool", (ToolResultPart("toolu_1", "ok"),))),
            reasoning=ReasoningParams(summary="none"),
            tools=ToolParams(functions=functions),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
            ),
        )
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_response())

        connection = _connection(handler)
        try:
            dialect = AnthropicMessagesDialect(connection, "claude")
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, object()))
            prepared = dialect.prepare(req, audit)
            audit.finish(prepared)
            body = prepared.thaw_body()
            cast(list[dict[str, object]], body["tools"])[0]["name"] = "changed"

            with pytest.raises(DialectError, match="must exactly match"):
                await dialect.execute(
                    PreparedRequest(prepared.operation, body, prepared.context)
                )
        finally:
            await connection.aclose()

        assert calls == 0

    def test_constructor_requires_right_connection_and_model(self) -> None:
        with pytest.raises(TypeError, match="AsyncHTTPJSONConnection"):
            AnthropicMessagesDialect(cast(Any, object()), "claude")
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        with pytest.raises(ValueError, match="requested_model_id"):
            AnthropicMessagesDialect(connection, "")


class TestProviderContractRegressions:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("input_", "message"),
        [
            (
                _chat(
                    ChatMessage(
                        "user",
                        (TextPart("result follows"), ToolResultPart("toolu_1", "ok")),
                    )
                ),
                "must precede",
            ),
            (
                _chat(
                    ChatMessage(
                        "assistant",
                        (ToolCallPart("toolu_1", "lookup", {}),),
                    ),
                    ChatMessage("tool", (ToolResultPart("wrong", "ok"),)),
                ),
                "must exactly match",
            ),
            (
                _chat(
                    ChatMessage(
                        "assistant",
                        (
                            ToolCallPart("toolu_1", "lookup", {}),
                            ToolCallPart("toolu_1", "lookup", {}),
                        ),
                    ),
                    ChatMessage("tool", (ToolResultPart("toolu_1", "ok"),)),
                ),
                "duplicates a tool_use id",
            ),
        ],
    )
    async def test_invalid_tool_history_is_rejected_before_http(
        self, input_: ChatInput, message: str
    ) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_response())

        with pytest.raises(RequestAuditError, match=message):
            await _run(handler, _request(input=input_))

        assert calls == 0

    @pytest.mark.anyio
    async def test_continuation_rejects_a_second_result_for_answered_call(self) -> None:
        opaque = json.dumps(
            {
                "version": 2,
                "system": [],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "call once",
                                "signature": "sig",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "lookup",
                                "input": {},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "done",
                                "is_error": False,
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "finished"}],
                    },
                ],
                "tools": None,
                "open_turn": False,
                "reasoning": {"thinking": None, "effort": None},
            }
        )
        req = _request(
            input=_chat(ChatMessage("tool", (ToolResultPart("toolu_1", "again"),))),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
            ),
        )

        with pytest.raises(RequestAuditError, match="do not immediately follow"):
            await _run(lambda _: pytest.fail("HTTP must not run"), req)

    @pytest.mark.anyio
    async def test_empty_end_turn_emits_a_reusable_continuation(self) -> None:
        opaque = json.dumps(
            {
                "version": 2,
                "system": [],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Use the tool"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "call once",
                                "signature": "sig",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "lookup",
                                "input": {},
                            },
                        ],
                    },
                ],
                "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
                "open_turn": True,
                "reasoning": {"thinking": None, "effort": None},
            }
        )
        functions = (
            {
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        )
        first = await _run(
            lambda _: httpx.Response(200, json=_response(content=[])),
            _request(
                input=_chat(ChatMessage("tool", (ToolResultPart("toolu_1", "done"),))),
                tools=ToolParams(functions=functions),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
                ),
            ),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        continued = cast(str, first.reasoning[0].opaque_roundtrip)
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_response())

        second = await _run(
            handler,
            _request(
                input=_chat(ChatMessage("user", (TextPart("Continue"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "anthropic_messages", continued
                    )
                ),
            ),
        )

        assert second.texts == ("Hi",)
        messages = cast(list[dict[str, object]], seen["messages"])
        assert all(message["content"] != [] for message in messages)
        assert [message["role"] for message in messages[-2:]] == ["user", "user"]

    @pytest.mark.anyio
    async def test_closed_continuation_may_use_new_tool_definitions(self) -> None:
        opaque = json.dumps(
            {
                "version": 2,
                "system": [],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "finished turn",
                                "signature": "sig",
                            },
                            {"type": "text", "text": "done"},
                        ],
                    }
                ],
                "tools": [{"name": "old", "input_schema": {"type": "object"}}],
                "open_turn": False,
                "reasoning": {"thinking": None, "effort": None},
            }
        )
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
                ),
                tools=ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "name": "new",
                            "parameters": {"type": "object"},
                        },
                    )
                ),
            ),
        )

        assert cast(list[dict[str, object]], seen["tools"])[0]["name"] == "new"

    @pytest.mark.anyio
    async def test_manual_thinking_response_must_start_with_thinking(self) -> None:
        req = _request(
            sampling=SamplingParams(max_tokens=2048),
            reasoning=ReasoningParams(budget_tokens=1024),
        )

        with pytest.raises(OutputContractError, match="must begin"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {"type": "text", "text": "premature"},
                            {
                                "type": "thinking",
                                "thinking": "late",
                                "signature": "sig",
                            },
                        ]
                    ),
                ),
                req,
            )

    @pytest.mark.anyio
    async def test_visible_summary_rejects_whitespace_only_thinking(self) -> None:
        req = _request(reasoning=ReasoningParams(summary="auto"))

        with pytest.raises(OutputContractError, match="visible thinking summary"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "type": "thinking",
                                "thinking": "   ",
                                "signature": "sig",
                            },
                            {"type": "text", "text": "answer"},
                        ]
                    ),
                ),
                req,
            )

    @pytest.mark.anyio
    async def test_response_rejects_duplicate_tool_use_ids(self) -> None:
        with pytest.raises(OutputContractError, match="duplicated a tool_use id"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "one",
                                "input": {},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "two",
                                "input": {},
                            },
                        ],
                        stop_reason="tool_use",
                    ),
                )
            )

    @pytest.mark.anyio
    async def test_response_rejects_tool_use_id_reused_from_prior_turn(self) -> None:
        opaque = json.dumps(
            {
                "version": 2,
                "system": [],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "call once",
                                "signature": "sig",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "lookup",
                                "input": {},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "done",
                                "is_error": False,
                            }
                        ],
                    },
                ],
                "tools": None,
                "open_turn": False,
                "reasoning": {"thinking": None, "effort": None},
            }
        )
        req = _request(
            tools=ToolParams(
                functions=(
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                )
            ),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("anthropic_messages", opaque)
            ),
        )

        with pytest.raises(OutputContractError, match="duplicates a tool_use id"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "lookup",
                                "input": {},
                            }
                        ],
                        stop_reason="tool_use",
                    ),
                ),
                req,
            )

    @pytest.mark.anyio
    async def test_pause_turn_fails_as_unrepresentable(self) -> None:
        with pytest.raises(OutputContractError, match="pause_turn.*unsupported"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(stop_reason="pause_turn"),
                )
            )

    @pytest.mark.anyio
    async def test_cache_creation_breakdown_supplies_missing_total(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    usage={
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "cache_creation": {
                            "ephemeral_1h_input_tokens": 5,
                            "ephemeral_5m_input_tokens": 7,
                        },
                    }
                ),
            )
        )

        assert result.usage is not None
        assert result.usage.input_tokens == 14
        assert result.usage.total_tokens == 17

    @pytest.mark.anyio
    async def test_reasoning_breakdown_cannot_exceed_inclusive_output(self) -> None:
        with pytest.raises(OutputContractError, match="exceeds inclusive"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        usage={
                            "input_tokens": 2,
                            "output_tokens": 3,
                            "output_tokens_details": {"thinking_tokens": 4},
                        }
                    ),
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "body",
        [
            b'{"id":"first","id":"second","type":"message"}',
            b'{"id":"msg","type":"message","role":"assistant",'
            b'"content":[],"model":"claude","stop_reason":"end_turn",'
            b'"stop_sequence":null,"usage":{"input_tokens":NaN,'
            b'"output_tokens":1}}',
        ],
    )
    async def test_native_response_json_must_be_unambiguous(self, body: bytes) -> None:
        with pytest.raises(OutputContractError, match="not valid JSON"):
            await _run(lambda _: httpx.Response(200, content=body))

    @pytest.mark.anyio
    async def test_structured_output_rejects_duplicate_json_keys(self) -> None:
        req = _request(
            structured_output=StructuredOutputParams(
                format="json_schema", schema={"type": "object"}
            )
        )

        with pytest.raises(OutputContractError, match="not valid JSON"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response([{"type": "text", "text": '{"value":1,"value":2}'}]),
                ),
                req,
            )

    @pytest.mark.anyio
    async def test_continuation_rejects_duplicate_envelope_keys(self) -> None:
        opaque = (
            '{"version":2,"version":2,"system":[],"messages":[],'
            '"tools":null,"open_turn":false,'
            '"reasoning":{"thinking":null,"effort":null}}'
        )

        with pytest.raises(DialectError, match="not valid JSON"):
            await _run(
                lambda _: pytest.fail("HTTP must not run"),
                _request(
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation(
                            "anthropic_messages", opaque
                        )
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_streamed_tool_json_rejects_duplicate_keys(self) -> None:
        payload = _sse(
            _stream_start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"x":1,"x":2}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            _stream_delta(stop_reason="tool_use"),
            {"type": "message_stop"},
        )

        with pytest.raises(OutputContractError, match="input is not valid JSON"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=payload,
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )


class TestReachableInputBoundaries:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "message"),
        [
            (_request(input=CompletionInput("prompt")), "requires ChatInput"),
            (
                _request(input=_chat(ChatMessage("user", ()))),
                "empty content",
            ),
            (
                _request(
                    input=_chat(ChatMessage("user", (TextPart("x"),), name="alice"))
                ),
                "no message name",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("system", (cast(Any, ImagePart(url="x")),)),
                        ChatMessage("user", (TextPart("x"),)),
                    )
                ),
                "system accepts text",
            ),
            (
                _request(input=_chat(ChatMessage("tool", (TextPart("x"),)))),
                "tool-role messages",
            ),
            (
                _request(
                    input=_chat(ChatMessage("user", (ToolCallPart("id", "tool", {}),)))
                ),
                "tool_use blocks require assistant",
            ),
            (
                _request(
                    input=_chat(ChatMessage("assistant", (ToolResultPart("id", "x"),)))
                ),
                "assistant messages",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("assistant", (ToolCallPart("", "tool", {}),))
                    )
                ),
                "non-empty id and name",
            ),
            (
                _request(
                    input=_chat(ChatMessage("assistant", (ToolCallPart("id", "", {}),)))
                ),
                "non-empty id and name",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("assistant", (ToolCallPart("id", "tool", [1]),))
                    )
                ),
                "JSON object",
            ),
            (
                _request(input=_chat(ChatMessage("tool", (ToolResultPart("", "x"),)))),
                "non-empty tool_use_id",
            ),
            (
                _request(input=_chat(ChatMessage("user", (ImagePart(url=""),)))),
                "must not be empty",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (ImagePart(url="x", detail="high"),))
                    )
                ),
                "image detail",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage(
                            "user", (ImagePart(url="x", media_type="image/png"),)
                        )
                    )
                ),
                "URL image sources",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage(
                            "user", (ImagePart(data="x", media_type="image/bmp"),)
                        )
                    )
                ),
                "base64 images require one of",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=-1)),
                "max_tokens must be non-negative",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, top_p=-0.1)),
                "top_p must be between",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, top_k=-1)),
                "top_k must be non-negative",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, stop=("",))),
                "stop sequences must not be empty",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, frequency_penalty=0.1)),
                "no equivalent penalty",
            ),
            (
                _request(sampling=SamplingParams(max_tokens=10, presence_penalty=0.1)),
                "no equivalent penalty",
            ),
            (
                _request(reasoning=ReasoningParams(effort="minimal")),
                "unsupported Anthropic reasoning effort",
            ),
            (
                _request(
                    sampling=SamplingParams(max_tokens=2048),
                    reasoning=ReasoningParams(budget_tokens=512),
                ),
                "budget must be at least 1024",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={}, strict=False
                    )
                ),
                "always schema constrained",
            ),
            (
                _request(dialect_options=DialectOptions("anthropic_messages", {"": 1})),
                "keys must be non-empty",
            ),
            (
                _request(
                    dialect_options=DialectOptions(
                        "anthropic_messages", {"max_tokens": 1}
                    )
                ),
                "canonical provider-neutral field",
            ),
        ],
    )
    async def test_invalid_request_leaf_is_rejected_before_http(
        self, req: Request, message: str
    ) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_response())

        with pytest.raises(
            (DialectError, RequestAuditError, ValueError), match=message
        ):
            await _run(handler, req)

        assert calls == 0

    @pytest.mark.parametrize(
        "scoring",
        [
            ScoringParams(input_scoring=True),
            ScoringParams(sampled_logprobs=True),
            ScoringParams(sampled_logprobs=True, top_logprobs=2),
        ],
    )
    def test_low_level_dialect_validation_also_rejects_logprob_requests(
        self, scoring: ScoringParams
    ) -> None:
        connection = _connection(lambda _: pytest.fail("HTTP must not run"))
        dialect = AnthropicMessagesDialect(connection, "claude")
        req = _request(scoring=scoring)
        audit = RequestAudit(active_request_leaves(req))

        dialect.validate_request(req, audit, _LegacyPlan())

        with pytest.raises(RequestAuditError, match="does not provide logprobs"):
            audit.raise_rejections()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tools", "message"),
        [
            (
                ToolParams(functions=({"type": "hosted", "name": "x"},)),
                "require type='function'",
            ),
            (
                ToolParams(functions=({"type": "function", "function": "wrong"},)),
                "invalid Chat-shaped",
            ),
            (
                ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "name": "x",
                            "parameters": {},
                            "future": True,
                        },
                    )
                ),
                "unsupported function tool field",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "", "parameters": {}},)
                ),
                "non-empty string",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": []},)
                ),
                "parameters must be a JSON object",
            ),
            (
                ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "name": "x",
                            "parameters": {},
                            "description": 1,
                        },
                    )
                ),
                "description must be a string",
            ),
            (
                ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "name": "x",
                            "parameters": {},
                            "strict": "yes",
                        },
                    )
                ),
                "strict must be a boolean",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice="sometimes",
                ),
                "unsupported Anthropic tool choice",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice=1,
                ),
                "string or mapping",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice={"type": "hosted", "name": "x"},
                ),
                "only function-specific",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice={"type": "function", "function": "wrong"},
                ),
                "invalid Chat-shaped function tool choice",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice={
                        "type": "function",
                        "function": {"name": "x", "future": True},
                    },
                ),
                "requires only a name",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice={"type": "function", "name": "x", "future": True},
                ),
                "invalid function tool choice",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice={"type": "function", "name": ""},
                ),
                "non-empty name",
            ),
            (
                ToolParams(
                    functions=({"type": "function", "name": "x", "parameters": {}},),
                    choice="none",
                    parallel=True,
                ),
                "meaningless with tool_choice='none'",
            ),
        ],
    )
    async def test_invalid_tool_shape_is_rejected_before_http(
        self, tools: ToolParams, message: str
    ) -> None:
        with pytest.raises(RequestAuditError, match=message):
            await _run(
                lambda _: pytest.fail("HTTP must not run"),
                _request(tools=tools),
            )


class TestOpaqueContinuationValidation:
    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "version": 2,
            "system": [],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "prior",
                            "signature": "sig",
                        }
                    ],
                }
            ],
            "tools": None,
            "open_turn": False,
            "reasoning": {"thinking": None, "effort": None},
        }

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda value: value.update(version=3), "version is unsupported"),
            (lambda value: value.update(system={}), "history must be lists"),
            (lambda value: value.update(messages={}), "history must be lists"),
            (lambda value: value.update(open_turn=1), "open_turn must be a boolean"),
            (
                lambda value: value.update(reasoning=None),
                "reasoning state is malformed",
            ),
            (
                lambda value: value.update(reasoning={"thinking": None}),
                "reasoning state is malformed",
            ),
            (
                lambda value: value.update(
                    reasoning={"thinking": None, "effort": "minimal"}
                ),
                "effort is invalid",
            ),
            (
                lambda value: value.update(reasoning={"thinking": [], "effort": None}),
                "thinking state must be an object",
            ),
            (
                lambda value: value.update(
                    reasoning={
                        "thinking": {"type": "enabled", "budget_tokens": 2048, "x": 1},
                        "effort": None,
                    }
                ),
                "thinking state is malformed",
            ),
            (
                lambda value: value.update(
                    reasoning={
                        "thinking": {"type": "enabled", "budget_tokens": 1},
                        "effort": None,
                    }
                ),
                "thinking budget is invalid",
            ),
            (
                lambda value: value.update(
                    reasoning={
                        "thinking": {"type": "adaptive", "future": True},
                        "effort": None,
                    }
                ),
                "thinking state is malformed",
            ),
            (
                lambda value: value.update(
                    reasoning={"thinking": {"type": "off"}, "effort": None}
                ),
                "thinking type is invalid",
            ),
            (
                lambda value: value.update(
                    reasoning={
                        "thinking": {"type": "adaptive", "display": "verbose"},
                        "effort": None,
                    }
                ),
                "thinking display is invalid",
            ),
            (
                lambda value: value.update(messages=["wrong"]),
                "continuation is malformed",
            ),
            (
                lambda value: value.update(open_turn=True),
                "open_turn disagrees",
            ),
            (
                lambda value: value.update(messages=[]),
                "contains no replayable",
            ),
        ],
    )
    def test_invalid_envelope_is_rejected(
        self, mutate: Callable[[dict[str, object]], None], message: str
    ) -> None:
        payload = self._payload()
        mutate(payload)

        with pytest.raises(DialectError, match=message):
            _decode_continuation(json.dumps(payload))

    @pytest.mark.parametrize(
        ("block", "role", "message"),
        [
            ({"type": "text", "text": 1}, "user", "canonical text"),
            (
                {"type": "text", "text": "x", "citations": [{}]},
                "assistant",
                "cannot be replayed",
            ),
            (
                {"type": "image", "source": {"type": "url", "url": "x"}},
                "assistant",
                "requires user role",
            ),
            (
                {"type": "image", "source": "wrong"},
                "user",
                "canonical image",
            ),
            (
                {
                    "type": "image",
                    "source": {"type": "url", "url": "x", "future": True},
                },
                "user",
                "canonical URL source",
            ),
            (
                {"type": "image", "source": {"type": "url", "url": ""}},
                "user",
                "non-empty string",
            ),
            (
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/bmp",
                        "data": "x",
                    },
                },
                "user",
                "media_type is unsupported",
            ),
            (
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": ""},
                },
                "user",
                "non-empty string",
            ),
            (
                {"type": "image", "source": {"type": "future"}},
                "user",
                "source.type is unsupported",
            ),
            (
                {"type": "tool_use", "id": "id", "name": "x", "input": {}},
                "user",
                "requires assistant role",
            ),
            (
                {
                    "type": "tool_use",
                    "id": "id",
                    "name": "x",
                    "input": {},
                    "future": True,
                },
                "assistant",
                "canonical tool_use",
            ),
            (
                {"type": "tool_use", "id": "id", "name": "x", "input": []},
                "assistant",
                "input must be an object",
            ),
            (
                {
                    "type": "tool_result",
                    "tool_use_id": "id",
                    "content": "x",
                    "is_error": False,
                },
                "assistant",
                "requires user role",
            ),
            (
                {
                    "type": "tool_result",
                    "tool_use_id": "id",
                    "content": 1,
                    "is_error": False,
                },
                "user",
                "content must be a string",
            ),
            (
                {
                    "type": "tool_result",
                    "tool_use_id": "id",
                    "content": "x",
                    "is_error": "no",
                },
                "user",
                "is_error must be a boolean",
            ),
            (
                {"type": "thinking", "thinking": "x", "signature": "sig"},
                "user",
                "canonical thinking",
            ),
            (
                {"type": "thinking", "thinking": 1, "signature": "sig"},
                "assistant",
                "thinking must be a string",
            ),
            (
                {"type": "redacted_thinking", "data": "opaque"},
                "user",
                "canonical redacted-thinking",
            ),
            (
                {"type": "future"},
                "assistant",
                "unsupported Anthropic block type",
            ),
        ],
    )
    def test_lossy_history_block_is_rejected(
        self, block: Mapping[str, object], role: str, message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            _validate_history_block(cast(Any, block), "history[0]", role=role)

    @pytest.mark.parametrize(
        ("tool", "message"),
        [
            ({"name": "x", "input_schema": {}, "future": True}, "unsupported"),
            ({"name": "", "input_schema": {}}, "name must"),
            ({"name": "x", "input_schema": []}, "input_schema must"),
            ({"name": "x", "input_schema": {}, "description": 1}, "description"),
            ({"name": "x", "input_schema": {}, "strict": "yes"}, "strict"),
        ],
    )
    def test_lossy_replay_tool_is_rejected(
        self, tool: Mapping[str, object], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            _validate_replay_tools([cast(Any, tool)])

    @pytest.mark.parametrize(
        "system",
        [
            [{"type": "future", "text": "x"}],
            [{"type": "text", "text": 1}],
        ],
    )
    def test_lossy_replay_system_is_rejected(
        self, system: list[Mapping[str, object]]
    ) -> None:
        with pytest.raises(DialectError, match="opaque.system"):
            _validate_replay_system(cast(Any, system))

    @pytest.mark.parametrize(
        ("messages", "message"),
        [
            ([{"role": "user", "content": [], "future": True}], "unsupported fields"),
            ([{"role": "tool", "content": []}], "role is invalid"),
            ([{"role": "user", "content": []}], "content must be non-empty"),
            ([{"role": "user", "content": ["wrong"]}], "must be an object"),
        ],
    )
    def test_lossy_replay_message_envelope_is_rejected(
        self, messages: list[Mapping[str, object]], message: str
    ) -> None:
        payload = self._payload()
        payload["messages"] = messages

        with pytest.raises(DialectError, match=message):
            _decode_continuation(json.dumps(payload))

    @pytest.mark.parametrize(
        ("messages", "message"),
        [
            ([{"role": "user", "content": "wrong"}], "content must be a list"),
            ([{"role": "user", "content": ["wrong"]}], "must be an object"),
            (
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "id"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "later"}],
                    },
                ],
                "must immediately return",
            ),
            (
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_result", "tool_use_id": "id"}],
                    }
                ],
                "require user role",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "id"},
                            {"type": "tool_result", "tool_use_id": "id"},
                        ],
                    }
                ],
                "duplicates a tool_result id",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": [{"type": "tool_use", "id": "id"}],
                    }
                ],
                "require assistant role",
            ),
        ],
    )
    def test_tool_history_state_rejects_invalid_protocol(
        self, messages: list[Mapping[str, object]], message: str
    ) -> None:
        rejection, pending = _tool_history_state(cast(Any, messages))

        assert rejection is not None and message in rejection
        assert pending == frozenset()


class TestWireVerifierNegativeEvidence:
    def test_input_verifier_detects_system_presence_and_content_drift(self) -> None:
        without_system = _AnthropicInputVerifier(_chat(), None)
        assert (
            without_system.verify(
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
                    ],
                    "system": [],
                }
            )
            == "body.system was added"
        )

        source = _chat(
            ChatMessage("system", (TextPart("rules"),)),
            ChatMessage("user", (TextPart("Hello"),)),
        )
        verifier = _AnthropicInputVerifier(source, None)
        assert verifier.verify({"messages": []}) == "body.system is missing"
        assert (
            verifier.verify(
                {
                    "system": [{"type": "text", "text": "changed"}],
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
                    ],
                }
            )
            == "body.system changed"
        )

    def test_tool_and_reasoning_verifiers_reject_wrong_wire_shape(self) -> None:
        source = ({"type": "function", "name": "lookup", "parameters": {}},)
        tools = _AnthropicToolsVerifier(source)
        assert tools.verify({"tools": []}) == "body.tools count changed"

        choice = _AnthropicToolChoiceVerifier(None, True)
        assert choice.verify({"tool_choice": None}) == "body.tool_choice changed"

        summary = _ReasoningSummaryVerifier("auto", False)
        assert summary.verify({}) == "body.thinking is missing"
        assert (
            summary.verify({"thinking": {"type": "adaptive", "display": "omitted"}})
            == "body.thinking.display changed"
        )
        assert (
            summary.verify({"thinking": {"type": "enabled", "display": "summarized"}})
            == "body.thinking.type changed"
        )


class TestAdditionalProviderResponseBoundaries:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("response", "message"),
        [
            ({**_response(), "id": ""}, "response.id"),
            ({**_response(), "content": {}}, "response.content must be a list"),
            (
                _response([{"type": "text", "text": "x", "future": True}]),
                "unsupported text fields",
            ),
            (_response([{"type": "text", "text": 1}]), "text must be a string"),
            (
                _response(
                    [{"type": "thinking", "thinking": "x", "signature": "sig", "x": 1}]
                ),
                "unsupported thinking fields",
            ),
            (
                _response([{"type": "thinking", "thinking": 1, "signature": "sig"}]),
                "thinking must be a string",
            ),
            (
                _response([{"type": "redacted_thinking", "data": "x", "future": True}]),
                "unsupported redacted-thinking fields",
            ),
            (
                _response(
                    [
                        {
                            "type": "tool_use",
                            "id": "id",
                            "name": "x",
                            "input": {},
                            "future": True,
                        }
                    ],
                    stop_reason="tool_use",
                ),
                "unsupported tool_use fields",
            ),
            (
                _response(
                    [{"type": "tool_use", "id": "id", "name": "x", "input": []}],
                    stop_reason="tool_use",
                ),
                "input must be an object",
            ),
            (
                {**_response(stop_reason="stop_sequence"), "stop_sequence": None},
                "response.stop_sequence",
            ),
            (
                _response(usage={"input_tokens": 1, "output_tokens": 1, "future": 1}),
                "unsupported field",
            ),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_creation": {"ephemeral_1h_input_tokens": 1},
                    }
                ),
                "cache_creation has unsupported fields",
            ),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "server_tool_use": {"web_search_requests": 0},
                    }
                ),
                "server_tool_use has unsupported fields",
            ),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "service_tier": "free",
                    }
                ),
                "service_tier is invalid",
            ),
            (
                _response(
                    usage={"input_tokens": 1, "output_tokens": 1, "inference_geo": 1}
                ),
                "inference_geo must be a string",
            ),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "output_tokens_details": {"thinking_tokens": 0, "future": 0},
                    }
                ),
                "output_tokens_details has unsupported fields",
            ),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "output_tokens_details": "not-an-object",
                    }
                ),
                "output_tokens_details must be an object",
            ),
            (
                _response(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "output_tokens_details": [],
                    }
                ),
                "output_tokens_details must be an object",
            ),
        ],
    )
    async def test_malformed_response_shape_fails_loudly(
        self, response: Mapping[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=response))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("events", "message"),
        [
            (({"type": "future"},), "unsupported Anthropic stream event"),
            (
                (_stream_start(), _stream_start()),
                "two message_start",
            ),
            (
                (
                    {
                        **_stream_start(),
                        "message": _response(
                            content=[{"type": "text", "text": "x"}], stop_reason=None
                        ),
                    },
                ),
                "content must be empty",
            ),
            (
                (
                    {
                        **_stream_start(),
                        "message": _response(content=[], stop_reason="end_turn"),
                    },
                ),
                "stop_reason must be null",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                "duplicated content block",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "x"},
                    },
                ),
                "inactive content block",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "signature_delta", "signature": "a"},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "signature_delta", "signature": "b"},
                    },
                ),
                "duplicated the signature delta",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "signature_delta", "signature": "a"},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "thinking_delta", "thinking": "late"},
                    },
                ),
                "followed the terminal signature delta",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": 1},
                    },
                ),
                "text must be a string",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "thinking_delta", "thinking": 1},
                    },
                ),
                "thinking must be a string",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "signature_delta", "signature": 1},
                    },
                ),
                "signature must be a string",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "id",
                            "name": "x",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": 1},
                    },
                ),
                "partial_json must be a string",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "thinking_delta", "thinking": "x"},
                    },
                ),
                "does not match block type",
            ),
            (
                (
                    _stream_start(),
                    {"type": "content_block_stop", "index": 0},
                ),
                "stop targets inactive",
            ),
            (
                (
                    _stream_start(),
                    _stream_delta(),
                    _stream_delta(),
                ),
                "two message_delta",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "container": {},
                        },
                        "usage": {"output_tokens": 1},
                    },
                ),
                "container has no shared-IR mapping",
            ),
            (
                (
                    _stream_start(),
                    _stream_delta(stop_reason="refusal"),
                    {"type": "message_stop"},
                ),
                "response.stop_details must be an object",
            ),
            (
                (
                    _stream_start(),
                    _stream_delta(
                        stop_reason="refusal",
                        stop_details={"type": "future"},
                    ),
                    {"type": "message_stop"},
                ),
                "stop_details.type must be 'refusal'",
            ),
            (
                (
                    _stream_start(),
                    _stream_delta(
                        stop_details={"type": "refusal"},
                    ),
                    {"type": "message_stop"},
                ),
                "stop_details must be null unless stop_reason is 'refusal'",
            ),
            (
                (
                    _stream_start(),
                    _stream_delta(),
                    {"type": "message_stop"},
                    {"type": "ping"},
                ),
                "data after message_stop",
            ),
            (
                ({"type": "ping"},),
                "omitted message_start",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": "x"},
                    },
                    {"type": "message_stop"},
                ),
                "omitted message_delta",
            ),
            (
                (
                    _stream_start(),
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "text", "text": "x"},
                    },
                    {"type": "content_block_stop", "index": 1},
                    _stream_delta(),
                    {"type": "message_stop"},
                ),
                "indexes are not dense",
            ),
        ],
    )
    async def test_additional_malformed_streams_fail_loudly(
        self, events: tuple[Mapping[str, object], ...], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=_sse(*events),
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.parametrize(
        ("block", "message"),
        [
            ({"type": "tool_use", "id": "id", "name": "x"}, "input is absent"),
            (
                {
                    "type": "tool_use",
                    "id": "id",
                    "name": "x",
                    "input": {"x": 1},
                    "_partial_json": '{"y":2}',
                },
                "must start as an empty object",
            ),
            (
                {
                    "type": "tool_use",
                    "id": "id",
                    "name": "x",
                    "input": {},
                    "_partial_json": "[]",
                },
                "input must be an object",
            ),
        ],
    )
    def test_stream_tool_block_finish_rejects_ambiguous_state(
        self, block: dict[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            _finish_stream_block(cast(Any, block), "content_block[0]")
