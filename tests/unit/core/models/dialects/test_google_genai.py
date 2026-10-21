"""Tests for the native Google Gemini ``generateContent`` dialect.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
import re
from collections.abc import AsyncIterator, Callable, Mapping
from types import SimpleNamespace
from typing import Any, Literal, cast

import httpx
import pytest

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    DialectCapabilityStatus,
    HostedToolsOptions,
    ReasoningOptions,
    Supported,
    TopLogprobsOptions,
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
from sieval.core.models.dialects import google_genai
from sieval.core.models.dialects._google_genai_wire import GOOGLE_WIRE_CONTRACTS
from sieval.core.models.dialects.google_genai import (
    CAPABILITY_DECISIONS,
    GoogleGenAIDialect,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    ChatRole,
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
    values: dict[str, object] = {"input": _chat()}
    values.update(changes)
    return Request(**cast(Any, values))


def _response(
    parts: list[dict[str, object]] | None = None,
    *,
    finish_reason: object = "STOP",
    usage: object | None = None,
    model_version: object = "gemini-2.5-flash",
    candidate_extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "index": 0,
        "content": {
            "role": "model",
            "parts": parts if parts is not None else [{"text": "Hi"}],
        },
        "finishReason": finish_reason,
    }
    if candidate_extra:
        candidate.update(candidate_extra)
    return {
        "candidates": [candidate],
        "usageMetadata": usage
        if usage is not None
        else {
            "promptTokenCount": 4,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
            "thoughtsTokenCount": 1,
            "cachedContentTokenCount": 2,
        },
        "modelVersion": model_version,
    }


def _connection(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base_url: str = "https://generativelanguage.googleapis.com/",
    credential: str | None = "google-secret",
) -> AsyncHTTPJSONConnection:
    return AsyncHTTPJSONConnection(
        httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler)),
        credential,
    )


async def _run(
    handler: Callable[[httpx.Request], httpx.Response],
    request: Request | None = None,
    *,
    model: str = "gemini-2.5-flash",
    base_url: str = "https://generativelanguage.googleapis.com/",
    credential: str | None = "google-secret",
):
    connection = _connection(handler, base_url=base_url, credential=credential)
    try:
        return await GoogleGenAIDialect(connection, model).arun(request or _request())
    finally:
        await connection.aclose()


def _sse(*chunks: Mapping[str, object]) -> bytes:
    lines: list[str] = []
    for chunk in chunks:
        payload = json.dumps(chunk, separators=(",", ":"))
        split_at = payload.find(",", len(payload) // 2)
        if split_at < 0:
            lines.extend((f"data: {payload}", ""))
        else:
            lines.extend(
                (
                    f"data: {payload[: split_at + 1]}",
                    f"data:{payload[split_at + 1 :]}",
                    "",
                )
            )
    return "\n".join(lines).encode()


def _no_io(_: httpx.Request) -> httpx.Response:
    raise AssertionError("request rejection must happen before HTTP I/O")


class TestCapabilities:
    def test_decision_row_and_registry_are_complete(self) -> None:
        assert set(CAPABILITY_DECISIONS) == set(CAPABILITY_KEYS)
        assert isinstance(CAPABILITY_DECISIONS["input_scoring"], Unsupported)
        for key in (
            "sampled_logprobs",
            "top_logprobs",
            "reasoning",
            "function_tools",
            "hosted_tools",
            "structured_output",
            "opaque_continuation",
            "multimodal_input",
        ):
            assert isinstance(CAPABILITY_DECISIONS[key], Supported)

        spec = DIALECT_SPECS["google_genai"]
        assert spec.implementation_status.value == "active"
        assert spec.request_seed_support.value == "supported"
        assert spec.capability_outcomes["reasoning"] is (
            DialectCapabilityStatus.SUPPORTED
        )

    @pytest.mark.parametrize(
        ("key", "options", "message"),
        [
            ("top_logprobs", TopLogprobsOptions(21), "20"),
            (
                "reasoning",
                ReasoningOptions(summary="detailed"),
                "reasoning summary 'none' or 'auto'",
            ),
            ("reasoning", ReasoningOptions(effort="xhigh"), "thinking level"),
            (
                "hosted_tools",
                HostedToolsOptions(("web_search",)),
                "google_search",
            ),
        ],
    )
    def test_capability_domains_are_provider_validated(
        self, key: str, options: object, message: str
    ) -> None:
        decision = cast(Supported, CAPABILITY_DECISIONS[cast(Any, key)])
        with pytest.raises(ValueError, match=message):
            decision.binding.validate_config(cast(Any, options))

    @pytest.mark.parametrize("summary", ["none", "auto"])
    def test_reasoning_config_accepts_both_summary_modes(self, summary: str) -> None:
        decision = cast(Supported, CAPABILITY_DECISIONS["reasoning"])
        decision.binding.validate_config(ReasoningOptions(summary=summary))


class TestRequestLowering:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("base_url", "model", "expected_path"),
        [
            (
                "https://generativelanguage.googleapis.com/",
                "gemini-2.5-flash",
                "/v1beta/models/gemini-2.5-flash:generateContent",
            ),
            (
                "https://generativelanguage.googleapis.com/v1beta/",
                "models/gemini-2.5-flash",
                "/v1beta/models/gemini-2.5-flash:generateContent",
            ),
            (
                "https://generativelanguage.googleapis.com/v1",
                "gemini-2.5-flash",
                "/v1/models/gemini-2.5-flash:generateContent",
            ),
            (
                "https://gateway.example/google/v1/",
                "models/gemini-2.5-flash",
                "/google/v1/models/gemini-2.5-flash:generateContent",
            ),
            (
                "https://gateway.example/v2/google/v1/",
                "gemini-2.5-flash",
                "/v2/google/v1/models/gemini-2.5-flash:generateContent",
            ),
        ],
    )
    async def test_native_request_uses_final_url_header_and_wire_body(
        self, base_url: str, model: str, expected_path: str
    ) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage("system", (TextPart("Be concise"),)),
                ChatMessage("user", (TextPart("Hello"),)),
            ),
            sampling=SamplingParams(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                max_tokens=64,
                stop=("END",),
                seed=7,
                frequency_penalty=0.1,
                presence_penalty=-0.2,
            ),
        )
        result = await _run(handler, req, model=model, base_url=base_url)

        request = cast(httpx.Request, seen["request"])
        assert request.url.path == expected_path
        assert request.headers["x-goog-api-key"] == "google-secret"
        assert seen["body"] == {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "systemInstruction": {"parts": [{"text": "Be concise"}]},
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 64,
                "stopSequences": ["END"],
                "seed": 7,
                "frequencyPenalty": 0.1,
                "presencePenalty": -0.2,
            },
        }
        assert result.texts == ("Hi",)
        assert result.finish_reasons == ("STOP",)
        assert result.request_params == {
            "envelope": "http",
            "method": "POST",
            "path": expected_path,
            "body": {
                "generationConfig": cast(dict[str, object], seen["body"])[
                    "generationConfig"
                ]
            },
        }
        assert "google-secret" not in json.dumps(result.request_params)

    @pytest.mark.anyio
    async def test_missing_credential_does_not_emit_auth_header(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "x-goog-api-key" not in request.headers
            return httpx.Response(200, json=_response())

        await _run(handler, credential=None)

    @pytest.mark.anyio
    async def test_prepared_context_is_detached_from_mutable_request_inputs(
        self,
    ) -> None:
        function_specs: list[Mapping[str, object]] = [
            {"name": "lookup", "description": "Lookup a value"}
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["tools"] == [
                {
                    "functionDeclarations": [
                        {"name": "lookup", "description": "Lookup a value"}
                    ]
                }
            ]
            return httpx.Response(
                200,
                json=_response(
                    parts=[
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "lookup",
                                "args": {},
                            }
                        }
                    ]
                ),
            )

        connection = _connection(handler)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request(
                tools=ToolParams(functions=cast(Any, function_specs)),
            )
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            audit.finish(prepared)

            function_specs.clear()

            result = await dialect.execute(prepared)
        finally:
            await connection.aclose()

        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "lookup"

    @pytest.mark.anyio
    async def test_multimodal_parts_preserve_order(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        TextPart("before"),
                        ImagePart(data="aGVsbG8=", media_type="image/png"),
                        TextPart("after"),
                        ImagePart(url="gs://bucket/image.jpg", media_type="image/jpeg"),
                    ),
                )
            )
        )
        await _run(handler, req)

        body = cast(dict[str, Any], seen["body"])
        assert body["contents"][0]["parts"] == [
            {"text": "before"},
            {"inlineData": {"data": "aGVsbG8=", "mimeType": "image/png"}},
            {"text": "after"},
            {
                "fileData": {
                    "fileUri": "gs://bucket/image.jpg",
                    "mimeType": "image/jpeg",
                }
            },
        ]

    @pytest.mark.anyio
    async def test_url_image_cannot_bypass_image_only_media_types(self) -> None:
        req = _request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        ImagePart(
                            url="gs://bucket/document.pdf", media_type="application/pdf"
                        ),
                    ),
                )
            )
        )

        with pytest.raises(DialectError, match="unsupported Google image media type"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_function_result_recovers_name_from_prior_call(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage(
                    "assistant",
                    (ToolCallPart("call_1", "weather", {"city": "Paris"}),),
                ),
                ChatMessage(
                    "tool",
                    (ToolResultPart("call_1", {"temperature": 20}),),
                ),
                ChatMessage("user", (TextPart("Continue"),)),
            )
        )
        await _run(handler, req)

        assert cast(dict[str, Any], seen["body"])["contents"] == [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "call_1",
                            "name": "weather",
                            "args": {"city": "Paris"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "call_1",
                            "name": "weather",
                            "response": {"temperature": 20},
                        }
                    }
                ],
            },
            {"role": "user", "parts": [{"text": "Continue"}]},
        ]

    @pytest.mark.anyio
    async def test_unanswered_function_call_is_rejected_before_io(self) -> None:
        req = _request(
            input=_chat(
                ChatMessage(
                    "assistant",
                    (
                        ToolCallPart("call_1", "first", {}),
                        ToolCallPart("call_2", "second", {}),
                    ),
                ),
                ChatMessage("tool", (ToolResultPart("call_1", {"ok": True}),)),
            )
        )

        with pytest.raises(DialectError, match="unanswered.*call_2"):
            await _run(_no_io, req, model="gemini-3-pro-preview")

    @pytest.mark.anyio
    async def test_function_responses_must_follow_their_call_turn(self) -> None:
        req = _request(
            input=_chat(
                ChatMessage(
                    "assistant",
                    (ToolCallPart("call_a", "first", {}),),
                ),
                ChatMessage("user", (TextPart("intervening text"),)),
                ChatMessage(
                    "assistant",
                    (ToolCallPart("call_b", "second", {}),),
                ),
                ChatMessage(
                    "tool",
                    (
                        ToolResultPart("call_a", {"ok": True}),
                        ToolResultPart("call_b", {"ok": True}),
                    ),
                ),
            )
        )

        with pytest.raises(DialectError, match="immediately follow.*call_a"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_each_function_call_turn_may_be_answered_in_order(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["contents"] = json.loads(request.content)["contents"]
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage(
                        "assistant",
                        (ToolCallPart("call_a", "first", {}),),
                    ),
                    ChatMessage("tool", (ToolResultPart("call_a", {"a": True}),)),
                    ChatMessage(
                        "assistant",
                        (ToolCallPart("call_b", "second", {}),),
                    ),
                    ChatMessage("tool", (ToolResultPart("call_b", {"b": True}),)),
                )
            ),
        )

        contents = cast(list[dict[str, Any]], seen["contents"])
        assert [content["role"] for content in contents] == [
            "model",
            "user",
            "model",
            "user",
        ]
        assert contents[1]["parts"][0]["functionResponse"]["id"] == "call_a"
        assert contents[3]["parts"][0]["functionResponse"]["id"] == "call_b"

    @pytest.mark.anyio
    async def test_error_tool_result_is_explicit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][1]["parts"][0]["functionResponse"]["response"] == {
                "error": "failed"
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage("assistant", (ToolCallPart("c", "tool", {}),)),
                    ChatMessage("tool", (ToolResultPart("c", "failed", True),)),
                )
            ),
        )

    @pytest.mark.anyio
    async def test_function_and_hosted_tools_lower_with_observable_invocations(
        self,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["tools"] == [
                {
                    "functionDeclarations": [
                        {
                            "name": "weather",
                            "description": "Get weather",
                            "parametersJsonSchema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ]
                },
                {"googleSearch": {}},
            ]
            assert body["toolConfig"] == {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["weather"],
                },
                "includeServerSideToolInvocations": True,
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
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
                            },
                        },
                    ),
                    choice={"type": "function", "name": "weather"},
                    hosted=(HostedToolSpec("google_search"),),
                )
            ),
            model="gemini-3.8-flash",
        )

    @pytest.mark.anyio
    async def test_hosted_tools_map_explicit_auto_to_validated(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["toolConfig"] == {
                "functionCallingConfig": {"mode": "VALIDATED"},
                "includeServerSideToolInvocations": True,
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    functions=({"name": "weather", "description": "Get weather"},),
                    choice="auto",
                    hosted=(HostedToolSpec("google_search"),),
                )
            ),
            model="gemini-3.8-flash",
        )

    @pytest.mark.anyio
    async def test_hosted_tool_configs_follow_google_discovery_schema(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["tools"] == [
                {
                    "googleSearch": {
                        "timeRangeFilter": {
                            "startTime": "2026-01-01T00:00:00Z",
                            "endTime": "2026-02-01T00:00:00Z",
                        },
                        "searchTypes": {
                            "webSearch": {},
                            "imageSearch": {},
                        },
                    }
                },
                {"urlContext": {}},
            ]
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    hosted=(
                        HostedToolSpec(
                            "google_search",
                            {
                                "timeRangeFilter": {
                                    "startTime": "2026-01-01T00:00:00Z",
                                    "endTime": "2026-02-01T00:00:00Z",
                                },
                                "searchTypes": {
                                    "webSearch": {},
                                    "imageSearch": {},
                                },
                            },
                        ),
                        HostedToolSpec("url_context"),
                    )
                )
            ),
        )

    @pytest.mark.anyio
    async def test_hosted_tool_time_range_accepts_equal_rfc3339_instants(self) -> None:
        interval = {
            "startTime": "2026-01-01T05:30:00.123456789+05:30",
            "endTime": "2026-01-01T00:00:00.123456789Z",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["tools"] == [{"googleSearch": {"timeRangeFilter": interval}}]
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    hosted=(
                        HostedToolSpec("google_search", {"timeRangeFilter": interval}),
                    )
                )
            ),
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tool", "message"),
        [
            (
                HostedToolSpec("url_context", {"unexpected": True}),
                "url_context config must be an empty object",
            ),
            (
                HostedToolSpec("google_search", {"whatever": 123}),
                "unsupported field.*whatever",
            ),
            (
                HostedToolSpec(
                    "google_search",
                    {"timeRangeFilter": {"startTime": "2026-01-01T00:00:00Z"}},
                ),
                "requires startTime and endTime together",
            ),
            (
                HostedToolSpec("google_search", {"timeRangeFilter": []}),
                "timeRangeFilter must be an object",
            ),
            (
                HostedToolSpec("google_search", {"timeRangeFilter": {"future": True}}),
                "timeRangeFilter has unsupported field.*future",
            ),
            (
                HostedToolSpec(
                    "google_search",
                    {"timeRangeFilter": {"startTime": "", "endTime": "end"}},
                ),
                "startTime must be a non-empty timestamp string",
            ),
            (
                HostedToolSpec(
                    "google_search",
                    {
                        "timeRangeFilter": {
                            "startTime": "banana",
                            "endTime": "apple",
                        }
                    },
                ),
                "startTime must be an RFC3339 timestamp",
            ),
            (
                HostedToolSpec(
                    "google_search",
                    {
                        "timeRangeFilter": {
                            "startTime": "2026-01-01T00:00:00",
                            "endTime": "2026-01-02T00:00:00Z",
                        }
                    },
                ),
                "startTime must be an RFC3339 timestamp",
            ),
            (
                HostedToolSpec(
                    "google_search",
                    {
                        "timeRangeFilter": {
                            "startTime": "2026-12-31T00:00:00Z",
                            "endTime": "2026-01-01T00:00:00Z",
                        }
                    },
                ),
                "startTime must not be after endTime",
            ),
            (
                HostedToolSpec("google_search", {"searchTypes": []}),
                "searchTypes must be an object",
            ),
            (
                HostedToolSpec("google_search", {"searchTypes": {"futureSearch": {}}}),
                "searchTypes has unsupported field.*futureSearch",
            ),
            (
                HostedToolSpec(
                    "google_search",
                    {"searchTypes": {"webSearch": {"unexpected": True}}},
                ),
                "webSearch must be an empty object",
            ),
        ],
    )
    async def test_invalid_hosted_tool_configs_fail_before_io(
        self, tool: HostedToolSpec, message: str
    ) -> None:
        with pytest.raises(RequestAuditError, match=message):
            await _run(
                _no_io,
                _request(tools=ToolParams(hosted=(tool,))),
            )

    @pytest.mark.anyio
    async def test_function_only_explicit_auto_remains_auto(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["toolConfig"] == {
                "functionCallingConfig": {"mode": "AUTO"}
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    functions=({"name": "weather", "description": "Get weather"},),
                    choice="auto",
                )
            ),
        )

    @pytest.mark.anyio
    async def test_safety_settings_are_audited_root_options(self) -> None:
        settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            }
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["safetySettings"] == settings
            return httpx.Response(200, json=_response())

        result = await _run(
            handler,
            _request(
                dialect_options=DialectOptions(
                    "google_genai", {"safetySettings": settings}
                )
            ),
        )
        assert result.request_params is not None
        assert result.request_params["body"] == {"safetySettings": settings}

    @pytest.mark.anyio
    async def test_provider_specific_root_controls_remain_audited_options(
        self,
    ) -> None:
        options = {
            "cachedContent": "cachedContents/example",
            "serviceTier": "PRIORITY",
            "store": False,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert {key: body[key] for key in options} == options
            return httpx.Response(200, json=_response())

        result = await _run(
            handler,
            _request(dialect_options=DialectOptions("google_genai", options)),
        )

        assert result.request_params is not None
        assert result.request_params["body"] == options


class TestAPIVersionContracts:
    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_versions_have_explicit_operations_and_final_wire_schemas(
        self, version: str, stream: bool
    ) -> None:
        # Golden bodies come from the two Discovery schemas (20260928), not
        # the dialect's version table or schema converter.
        source_schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "minLength": 1},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 7},
                    "maxItems": 3,
                },
            },
            "required": ["city"],
        }
        v1_schema = {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "minLength": "1"},
                "units": {"type": "STRING", "enum": ["celsius", "fahrenheit"]},
                "days": {
                    "type": "ARRAY",
                    "items": {"type": "INTEGER", "minimum": 1, "maximum": 7},
                    "maxItems": "3",
                },
            },
            "required": ["city"],
        }
        declaration = {
            "name": "weather",
            "description": "Get weather",
            "parameters" if version == "v1" else "parametersJsonSchema": (
                v1_schema if version == "v1" else source_schema
            ),
        }
        method = "streamGenerateContent" if stream else "generateContent"
        path = f"/gateway/{version}/models/gemini-2.5-flash:{method}"
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            assert request.url.path == path
            assert dict(request.url.params) == ({"alt": "sse"} if stream else {})
            assert json.loads(request.content) == {
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "tools": [{"functionDeclarations": [declaration]}],
            }
            payload = _response(
                usage={
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 6,
                }
            )
            if stream:
                return httpx.Response(200, content=_sse(payload))
            return httpx.Response(200, json=payload)

        connection = _connection(
            handler, base_url=f"https://example.test/gateway/{version}"
        )
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request(
                scheduling=SchedulingParams(stream=stream),
                tools=ToolParams(
                    functions=(
                        {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": source_schema,
                        },
                    )
                ),
            )
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            assert prepared.operation == (
                f"{version}/models/gemini-2.5-flash:{method}"
                + ("?alt=sse" if stream else "")
            )
            audit.finish(prepared)
            result = await dialect.execute(prepared)
            assert len(seen) == 1
            assert result.texts == ("Hi",)
            assert result.request_params is not None
            assert result.request_params["path"] == seen[0].url.path
            assert result.request_params["body"] == {
                "tools": [{"functionDeclarations": [declaration]}]
            }
            assert result.usage is not None
            assert result.usage.cached_tokens is None
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_cross_version_operation_is_rejected_by_audit_and_execute(
        self, version: str, stream: bool
    ) -> None:
        connection = _connection(_no_io, base_url=f"https://example.test/{version}")
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request(scheduling=SchedulingParams(stream=stream))
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            other = "v1beta" if version == "v1" else "v1"
            method = "streamGenerateContent" if stream else "generateContent"
            mutated = PreparedRequest(
                f"{other}/models/gemini-2.5-flash:{method}"
                + ("?alt=sse" if stream else ""),
                prepared.body,
                prepared.context,
            )
            with pytest.raises(RequestAuditError, match="operation .* does not match"):
                audit.finish(mutated)
            with pytest.raises(DialectError, match="wrong prepared operation"):
                await dialect.execute(mutated)
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    async def test_final_http_version_drift_is_rejected(
        self, version: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = _connection(_no_io, base_url=f"https://example.test/{version}")
        build_request = connection.client.build_request
        other = "v1beta" if version == "v1" else "v1"

        def drifted_request(*args: Any, **kwargs: Any) -> httpx.Request:
            result = build_request(*args, **kwargs)
            result.url = result.url.copy_with(
                path=f"/{other}/models/gemini-2.5-flash:generateContent"
            )
            return result

        monkeypatch.setattr(connection.client, "build_request", drifted_request)
        try:
            with pytest.raises(DialectError, match="final Google request URL"):
                await GoogleGenAIDialect(connection, "gemini-2.5-flash").arun(
                    _request()
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    async def test_cross_version_function_shape_fails_independent_verifier(
        self, version: str
    ) -> None:
        connection = _connection(_no_io, base_url=f"https://example.test/{version}")
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request(
                tools=ToolParams(
                    functions=(
                        {
                            "name": "f",
                            "description": "Function f",
                            "parameters": {"type": "object"},
                        },
                    )
                )
            )
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, Any], prepared.thaw_body())
            declaration = body["tools"][0]["functionDeclarations"][0]
            if version == "v1beta":
                declaration.pop("parametersJsonSchema")
                declaration["parameters"] = {"type": "OBJECT"}
            else:
                declaration.pop("parameters")
                declaration["parametersJsonSchema"] = {"type": "object"}
            mutated = PreparedRequest(prepared.operation, body, prepared.context)
            with pytest.raises(RequestAuditError, match="function tools changed"):
                audit.finish(mutated)
            with pytest.raises(DialectError, match="function|parametersJsonSchema"):
                await dialect.execute(mutated)
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "object", "additionalProperties": False},
            {"type": "object", "$defs": {"x": {"type": "string"}}},
            {"type": "object", "properties": {"x": {"$ref": "#/$defs/x"}}},
            {"type": "object", "properties": {"x": {"type": ["string", "null"]}}},
            {
                "type": "object",
                "properties": {"x": {"type": "integer", "enum": [1, 2]}},
            },
            {
                "type": "object",
                "properties": {"x": {"type": "integer", "minimum": 2**53 + 1}},
            },
        ],
    )
    async def test_v1_rejects_unrepresentable_json_schema_before_io(
        self, schema: dict[str, Any]
    ) -> None:
        req = _request(
            tools=ToolParams(
                functions=(
                    {
                        "name": "f",
                        "description": "Function f",
                        "parameters": schema,
                    },
                )
            )
        )
        connection = _connection(_no_io, base_url="https://example.test/v1")
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            with pytest.raises(RequestAuditError, match="tools.functions.*v1"):
                audit.raise_rejections()
            with pytest.raises(DialectError, match="v1"):
                dialect.prepare(req, RequestAudit(active_request_leaves(req)))
        finally:
            await connection.aclose()

        def handler(request: httpx.Request) -> httpx.Response:
            assert (
                json.loads(request.content)["tools"][0]["functionDeclarations"][0][
                    "parametersJsonSchema"
                ]
                == schema
            )
            return httpx.Response(200, json=_response())

        # These remain JSON values on v1beta; v1 must not silently strip them.
        await _run(handler, req, base_url="https://example.test/v1beta")

    @pytest.mark.anyio
    @pytest.mark.parametrize("key", ["cachedContent", "labels"])
    async def test_beta_only_root_option_is_rejected_on_v1(self, key: str) -> None:
        req = _request(dialect_options=DialectOptions("google_genai", {key: "value"}))
        with pytest.raises(RequestAuditError, match=f"{key}.*v1"):
            await _run(_no_io, req, base_url="https://example.test/v1")

    @pytest.mark.anyio
    @pytest.mark.parametrize("version_path", ["", "/v1", "/v1beta"])
    @pytest.mark.parametrize("wrapped", [False, True])
    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "string"},
            {"type": "integer"},
            {"type": "array", "items": {"type": "string"}},
            None,
            False,
            [],
            {},
            {"properties": {"x": {"type": "string"}}},
            {"type": ["object", "null"]},
            {"$ref": "#/$defs/P", "$defs": {"P": {"type": "object"}}},
        ],
    )
    async def test_function_parameters_must_describe_an_object(
        self, version_path: str, wrapped: bool, schema: object
    ) -> None:
        source: dict[str, Any] = {
            "name": "f",
            "description": "Function f",
            "parameters": schema,
        }
        if wrapped:
            source = {"type": "function", "function": source}
        req = _request(tools=ToolParams(functions=(source,)))
        connection = _connection(_no_io, base_url=f"https://example.test{version_path}")
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            with pytest.raises(RequestAuditError, match="tools.functions.*object"):
                audit.raise_rejections()
            # Bypassing validate_request must not turn an invalid schema into wire.
            with pytest.raises(DialectError, match="function parameters.*object"):
                dialect.prepare(req, RequestAudit(active_request_leaves(req)))
            with pytest.raises(RequestAuditError, match="tools.functions.*object"):
                await dialect.arun(req)
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    @pytest.mark.parametrize(
        "schema", [None, {"type": "object"}, {"type": "object", "properties": {}}]
    )
    async def test_no_argument_functions_keep_valid_wire_shapes(
        self, version: str, schema: dict[str, Any] | None
    ) -> None:
        source: dict[str, Any] = {"name": "f", "description": "Function f"}
        expected: dict[str, Any] = dict(source)
        if schema is not None:
            source["parameters"] = schema
            field = "parameters" if version == "v1" else "parametersJsonSchema"
            expected[field] = {
                **schema,
                "type": "OBJECT" if version == "v1" else "object",
            }

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["tools"] == [
                {"functionDeclarations": [expected]}
            ]
            return httpx.Response(200, json=_response())

        response = await _run(
            handler,
            _request(tools=ToolParams(functions=(source,))),
            base_url=f"https://example.test/{version}",
        )
        assert response.texts == ("Hi",)

    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    async def test_final_http_function_parameter_root_is_checked(
        self, version: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = _connection(_no_io, base_url=f"https://example.test/{version}")
        build_request = connection.client.build_request

        def drifted_request(*args: Any, **kwargs: Any) -> httpx.Request:
            request = build_request(*args, **kwargs)
            body = json.loads(request.content)
            field = "parameters" if version == "v1" else "parametersJsonSchema"
            body["tools"][0]["functionDeclarations"][0][field] = {
                "type": "STRING" if version == "v1" else "string"
            }
            return httpx.Request(request.method, request.url, json=body)

        monkeypatch.setattr(connection.client, "build_request", drifted_request)
        # Observe the final-wire validator specifically, not the later
        # source-vs-wire equality check that would also catch this mutation.
        monkeypatch.setattr(
            google_genai._FunctionsVerifier, "verify", lambda *args: None
        )
        try:
            req = _request(
                tools=ToolParams(
                    functions=(
                        {
                            "name": "f",
                            "description": "Function f",
                            "parameters": {"type": "object"},
                        },
                    )
                )
            )
            with pytest.raises(DialectError, match="function parameters.*object"):
                await GoogleGenAIDialect(connection, "gemini-2.5-flash").arun(req)
        finally:
            await connection.aclose()

    @pytest.mark.parametrize(
        "body",
        [
            {"tools": None},
            {"tools": [{"functionDeclarations": "not a list"}]},
            {"tools": [{"functionDeclarations": [1]}]},
            {"tools": [{"functionDeclarations": [{"default": object()}]}]},
            {"tools": [{"functionDeclarations": [], "googleSearch": {}}]},
            {"tools": [{"functionDeclarations": []}, {"functionDeclarations": []}]},
            {
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "f",
                                "description": "Function f",
                                "parameters": {"type": "object"},
                            }
                        ]
                    }
                ]
            },
        ],
    )
    def test_functions_verifier_rejects_invalid_direct_inputs(
        self, body: dict[str, Any]
    ) -> None:
        verifier = google_genai._FunctionsVerifier(
            (
                {
                    "name": "f",
                    "description": "Function f",
                    "parameters": {"type": "object"},
                },
            ),
            GOOGLE_WIRE_CONTRACTS["v1"],
        )
        assert verifier.verify(body) == "function tools changed"

    def test_functions_verifier_rejects_invalid_source_and_unrequested_group(
        self,
    ) -> None:
        with pytest.raises(TypeError, match="function declarations must be objects"):
            google_genai._FunctionsVerifier(
                (cast(Any, []),), GOOGLE_WIRE_CONTRACTS["v1"]
            )
        verifier = google_genai._FunctionsVerifier((), GOOGLE_WIRE_CONTRACTS["v1"])
        assert verifier.verify({}) is None
        assert verifier.verify({"tools": [{"functionDeclarations": []}]}) == (
            "function tools changed"
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("version", "extra"),
        [
            ("v1", "both_schema_fields"),
            ("v1beta", "both_schema_fields"),
            ("v1", "beta_root_field"),
            ("v1", "changed_constraint"),
            ("v1beta", "changed_constraint"),
        ],
    )
    async def test_final_serialized_function_schema_is_checked(
        self, version: str, extra: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = _connection(_no_io, base_url=f"https://example.test/{version}")
        build_request = connection.client.build_request

        def drifted_request(*args: Any, **kwargs: Any) -> httpx.Request:
            result = build_request(*args, **kwargs)
            body = json.loads(result.content)
            declaration = body["tools"][0]["functionDeclarations"][0]
            if extra == "both_schema_fields":
                declaration["parameters"] = {"type": "OBJECT"}
                declaration["parametersJsonSchema"] = {"type": "object"}
            elif extra == "beta_root_field":
                body["cachedContent"] = "cachedContents/unrequested"
            else:
                key = "parameters" if version == "v1" else "parametersJsonSchema"
                declaration[key]["properties"]["value"]["minimum"] = 3
            return httpx.Request(result.method, result.url, json=body)

        monkeypatch.setattr(connection.client, "build_request", drifted_request)
        req = _request(
            tools=ToolParams(
                functions=(
                    {
                        "name": "f",
                        "description": "Function f",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer", "minimum": 2}},
                        },
                    },
                )
            )
        )
        try:
            with pytest.raises(
                DialectError,
                match="unavailable|mutually exclusive|function tools changed",
            ):
                await GoogleGenAIDialect(connection, "gemini-2.5-flash").arun(req)
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_function_schema_encoder_cannot_supply_its_own_audit_oracle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = _connection(_no_io, base_url="https://example.test/v1")
        encode = google_genai.lower_function_schema

        def drop_required(raw: object) -> dict[str, Any]:
            result = encode(raw)
            result.pop("required", None)
            return result

        monkeypatch.setattr(google_genai, "lower_function_schema", drop_required)
        req = _request(
            tools=ToolParams(
                functions=(
                    {
                        "name": "f",
                        "description": "Function f",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                            "required": ["x"],
                        },
                    },
                )
            )
        )
        try:
            with pytest.raises(RequestAuditError, match="function tools changed"):
                await GoogleGenAIDialect(connection, "gemini-2.5-flash").arun(req)
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    async def test_schema_annotations_and_field_named_properties_are_not_rewritten(
        self, version: str
    ) -> None:
        annotation = {"type": "object", "cachedContent": "value", "minItems": 7}
        schema = {
            "type": "object",
            "properties": {"parametersJsonSchema": {"type": "string"}},
            "default": annotation,
            "example": annotation,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            key = "parameters" if version == "v1" else "parametersJsonSchema"
            actual = body["tools"][0]["functionDeclarations"][0][key]
            assert actual["example"] == annotation
            assert actual["default"] == annotation
            assert set(actual["properties"]) == {"parametersJsonSchema"}
            assert actual["properties"]["parametersJsonSchema"] == {
                "type": "STRING" if version == "v1" else "string"
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    functions=(
                        {
                            "name": "f",
                            "description": "Function f",
                            "parameters": schema,
                        },
                    )
                )
            ),
            base_url=f"https://example.test/{version}",
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_shared_response_features_work_on_each_version(
        self, version: str, stream: bool
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path.startswith(f"/{version}/models/")
            assert body["generationConfig"] == {
                "thinkingConfig": {"thinkingBudget": 128, "includeThoughts": True},
                "responseMimeType": "application/json",
                "responseJsonSchema": {"type": "object"},
            }
            assert body["tools"] == [{"googleSearch": {}}, {"urlContext": {}}]
            payload = _response(parts=[{"text": '{"ok":true}'}])
            return (
                httpx.Response(200, content=_sse(payload))
                if stream
                else httpx.Response(200, json=payload)
            )

        result = await _run(
            handler,
            _request(
                reasoning=ReasoningParams(budget_tokens=128, summary="auto"),
                structured_output=StructuredOutputParams(
                    format="json_schema", schema={"type": "object"}
                ),
                tools=ToolParams(
                    hosted=(
                        HostedToolSpec("google_search"),
                        HostedToolSpec("url_context"),
                    )
                ),
                scheduling=SchedulingParams(stream=stream),
            ),
            base_url=f"https://example.test/{version}",
        )
        assert result.structured_output is not None
        assert result.structured_output.value == {"ok": True}

    @pytest.mark.anyio
    @pytest.mark.parametrize("path", ["v2", "v1alpha", "v1bet", "v1/models/f"])
    async def test_unknown_or_resource_endpoint_is_rejected(self, path: str) -> None:
        connection = _connection(_no_io, base_url=f"https://example.test/{path}")
        try:
            with pytest.raises(ValueError, match="Google.*api_base"):
                GoogleGenAIDialect(connection, "gemini-2.5-flash")
        finally:
            await connection.aclose()


class TestPreIORejections:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "message"),
        [
            (_request(input=CompletionInput("prompt")), "ChatInput"),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart("first"),)),
                        ChatMessage("system", (TextPart("late"),)),
                    )
                ),
                "must precede",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("developer", (TextPart("higher priority"),)),
                        ChatMessage("user", (TextPart("question"),)),
                    )
                ),
                "developer message authority",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart("complete this"),)),
                        ChatMessage("assistant", (TextPart("partial"),)),
                    )
                ),
                "assistant-prefill",
            ),
            (
                _request(
                    input=_chat(ChatMessage("system", (TextPart("instructions"),)))
                ),
                "end with a user or tool-result turn",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (
                                ImagePart(
                                    data="x", media_type="image/png", detail="high"
                                ),
                            ),
                        )
                    )
                ),
                "detail",
            ),
            (
                _request(
                    input=_chat(ChatMessage("tool", (ToolResultPart("missing", {}),)))
                ),
                "no prior call",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "one", "description": "First function"},),
                        choice={"type": "function", "name": "two"},
                    )
                ),
                "undeclared",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(schema={"type": "object"})
                ),
                "requires format",
            ),
            (
                _request(
                    dialect_options=DialectOptions(
                        "google_genai", {"generationConfig": {"temperature": 2}}
                    )
                ),
                "provider-neutral",
            ),
            (
                _request(sampling=SamplingParams(stop=("1", "2", "3", "4", "5", "6"))),
                "five",
            ),
        ],
    )
    async def test_invalid_shapes_fail_before_io(
        self, req: Request, message: str
    ) -> None:
        with pytest.raises((DialectError, RequestAuditError), match=message):
            await _run(_no_io, req)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "sampling",
        [
            SamplingParams(temperature=-0.1),
            SamplingParams(temperature=2.1),
            SamplingParams(top_p=1.1),
            SamplingParams(top_k=0),
            SamplingParams(max_tokens=0),
        ],
    )
    async def test_provider_sampling_domains_fail_before_io(
        self, sampling: SamplingParams
    ) -> None:
        with pytest.raises(RequestAuditError):
            await _run(_no_io, _request(sampling=sampling))

    @pytest.mark.anyio
    @pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3.8-flash"])
    async def test_multiple_candidates_reach_the_wire(self, model: str) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["generationConfig"] == {
                "candidateCount": 2
            }
            payload = _response()
            payload["candidates"] = [
                {
                    "index": index,
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": "STOP",
                }
                for index, text in enumerate(("first", "second"))
            ]
            return httpx.Response(200, json=payload)

        result = await _run(
            handler,
            _request(sampling=SamplingParams(n=2)),
            model=model,
        )

        assert result.texts == ("first", "second")

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "req",
        [
            _request(
                sampling=SamplingParams(n=2),
                tools=ToolParams(
                    functions=({"name": "f", "description": "Function f"},)
                ),
            ),
            _request(
                sampling=SamplingParams(n=2),
                tools=ToolParams(hosted=(HostedToolSpec("google_search"),)),
            ),
            _request(
                sampling=SamplingParams(n=2),
                structured_output=StructuredOutputParams(format="json_object"),
            ),
        ],
    )
    async def test_multiple_candidates_reject_singular_channels_before_io(
        self, req: Request
    ) -> None:
        with pytest.raises(ValueError, match="singular response channels"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_multiple_candidates_reject_logprobs_before_io(self) -> None:
        # Logprobs are choice-indexed in the IR, but this lift reads one
        # sequence across candidates, so the dialect refuses them itself.
        req = _request(
            sampling=SamplingParams(n=2),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        with pytest.raises(ValueError, match="logprobs for one candidate only"):
            await _run(_no_io, req)


class TestToolContextCirculation:
    @staticmethod
    def _tools(kind: str = "google_search") -> ToolParams:
        return ToolParams(
            functions=({"name": "weather", "description": "Get weather"},),
            hosted=(HostedToolSpec(kind),),
        )

    @staticmethod
    def _parts() -> list[dict[str, Any]]:
        return [
            {
                "toolCall": {
                    "id": "search_1",
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "args": {"queries": ["weather"]},
                },
                "thoughtSignature": "server-call-signature",
            },
            {
                "toolResponse": {
                    "id": "search_1",
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "response": {"found": True},
                },
                "thoughtSignature": "server-response-signature",
            },
            {
                "functionCall": {"id": "weather_1", "name": "weather", "args": {}},
                "thoughtSignature": "function-signature",
            },
        ]

    @classmethod
    def _replay_request(
        cls, parts: list[dict[str, Any]], *, hosted: bool = True
    ) -> Request:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {"role": "user", "parts": [{"text": "Get the weather"}]},
                    {"role": "model", "parts": parts},
                ],
            }
        )
        return _request(
            input=_chat(
                ChatMessage("tool", (ToolResultPart("weather_1", {"sunny": True}),))
            ),
            tools=cls._tools() if hosted else ToolParams(),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("google_genai", opaque)
            ),
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
        ],
    )
    async def test_older_gemini_mixed_tools_are_left_to_the_provider(
        self, model: str
    ) -> None:
        # Google rejects the combination itself; no local model-family table.
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, model)
            req = _request(tools=self._tools())
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, google_genai._LegacyPlan())
            prepared = dialect.prepare(req, audit)
            tools = cast(list[dict[str, Any]], prepared.thaw_body()["tools"])
            assert "functionDeclarations" in tools[0]
            assert tools[1] == {"googleSearch": {}}
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("kind", ["google_search", "url_context"])
    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    async def test_gemini_3_mixed_tools_reach_the_actual_wire(
        self, kind: str, version: str
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert len(body["tools"]) == 2
            assert "functionDeclarations" in body["tools"][0]
            assert body["tools"][1] == {
                "googleSearch" if kind == "google_search" else "urlContext": {}
            }
            assert body["toolConfig"]["includeServerSideToolInvocations"] is True
            return httpx.Response(200, json=_response())

        result = await _run(
            handler,
            _request(tools=self._tools(kind)),
            model="models/gemini-3.8-flash",
            base_url=f"https://example.test/{version}",
        )
        assert result.texts == ("Hi",)

    @pytest.mark.anyio
    @pytest.mark.parametrize("kind", ["functions", "google_search", "url_context"])
    async def test_gemini_25_single_tool_family_is_still_accepted(
        self, kind: str
    ) -> None:
        tools = (
            ToolParams(functions=self._tools().functions)
            if kind == "functions"
            else ToolParams(hosted=(HostedToolSpec(kind),))
        )

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert len(body["tools"]) == 1
            if kind == "functions":
                assert "functionDeclarations" in body["tools"][0]
            else:
                assert body["tools"][0] == {
                    "googleSearch" if kind == "google_search" else "urlContext": {}
                }
            return httpx.Response(200, json=_response())

        assert (await _run(handler, _request(tools=tools))).texts == ("Hi",)

    @pytest.mark.anyio
    async def test_unknown_alias_does_not_inherit_an_older_gemini_constraint(
        self,
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(200, json=_response()),
            _request(tools=self._tools()),
            model="provider-model-alias",
        )
        assert result.texts == ("Hi",)

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("part_index", [0, 1])
    @pytest.mark.parametrize("signature", ["absent", None])
    async def test_gemini_3_server_parts_require_their_own_signature(
        self, stream: bool, part_index: int, signature: str | None
    ) -> None:
        parts = self._parts()
        if signature == "absent":
            parts[part_index].pop("thoughtSignature")
        else:
            parts[part_index]["thoughtSignature"] = signature
        payload = _response(parts)

        def handler(_: httpx.Request) -> httpx.Response:
            return (
                httpx.Response(200, content=_sse(payload))
                if stream
                else httpx.Response(200, json=payload)
            )

        with pytest.raises(
            OutputContractError, match=r"tool(Call|Response).*thoughtSignature"
        ):
            await _run(
                handler,
                _request(
                    tools=self._tools(), scheduling=SchedulingParams(stream=stream)
                ),
                model="gemini-3.8-flash",
            )

    @pytest.mark.anyio
    async def test_gemini_25_unsigned_server_parts_remain_accepted(self) -> None:
        parts = self._parts()[:2]
        for part in parts:
            part.pop("thoughtSignature")
        result = await _run(
            lambda _: httpx.Response(200, json=_response(parts + [{"text": "Sunny"}])),
            _request(tools=ToolParams(hosted=self._tools().hosted)),
        )
        assert result.server_tool_uses is not None
        assert result.server_tool_uses[0].result == {"found": True}

    @pytest.mark.anyio
    @pytest.mark.parametrize("invocations", [False, True])
    async def test_response_signature_requirement_uses_the_final_wire_flag(
        self, invocations: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parts = self._parts()[:2]
        for part in parts:
            part.pop("thoughtSignature")

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["toolConfig"]["includeServerSideToolInvocations"] is invocations
            return httpx.Response(200, json=_response(parts + [{"text": "Sunny"}]))

        connection = _connection(handler)
        build_request = connection.client.build_request

        def final_request(*args: Any, **kwargs: Any) -> httpx.Request:
            request = build_request(*args, **kwargs)
            body = json.loads(request.content)
            body["toolConfig"]["includeServerSideToolInvocations"] = invocations
            return httpx.Request(request.method, request.url, json=body)

        monkeypatch.setattr(connection.client, "build_request", final_request)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-3.8-flash")
            req = _request(tools=ToolParams(hosted=self._tools().hosted))
            if invocations:
                with pytest.raises(
                    OutputContractError, match="toolCall.*thoughtSignature"
                ):
                    await dialect.arun(req)
            else:
                # Direct execution observes the serialized request, not the
                # original hosted-tools leaf or prepare-time flag value.
                result = await dialect.arun(req)
                assert result.texts == ("Sunny",)
                assert result.request_params is not None
                assert result.request_params["body"] == {
                    "tools": [{"googleSearch": {}}],
                    "toolConfig": {"includeServerSideToolInvocations": False},
                }
        finally:
            await connection.aclose()

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"contents": "wrong"},
            {"contents": [None]},
            {"contents": [{"parts": None}]},
            {"contents": [{"parts": "wrong"}]},
            {"contents": [{"parts": [None]}]},
        ],
    )
    def test_server_signature_check_leaves_structural_errors_to_shape_validation(
        self, body: dict[str, Any]
    ) -> None:
        google_genai._validate_server_tool_history_signatures(body, "gemini-3.8-flash")

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_signed_server_tool_history_roundtrips_without_rewriting_parts(
        self, stream: bool
    ) -> None:
        parts = self._parts()
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            payload = _response(parts if len(seen) == 1 else [{"text": "Sunny"}])
            return (
                httpx.Response(200, content=_sse(payload))
                if stream
                else httpx.Response(200, json=payload)
            )

        connection = _connection(handler)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-3.8-flash")
            first = await dialect.arun(
                _request(
                    tools=self._tools(), scheduling=SchedulingParams(stream=stream)
                )
            )
            assert first.reasoning is not None and first.reasoning[0] is not None
            opaque = first.reasoning[0].opaque_roundtrip
            assert opaque is not None
            assert json.loads(opaque)["contents"][-1]["parts"] == parts
            second = await dialect.arun(
                _request(
                    input=_chat(
                        ChatMessage(
                            "tool", (ToolResultPart("weather_1", {"sunny": True}),)
                        )
                    ),
                    tools=self._tools(),
                    scheduling=SchedulingParams(stream=stream),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    ),
                )
            )
        finally:
            await connection.aclose()

        assert second.texts == ("Sunny",)
        assert seen[1]["contents"][1]["parts"] == parts
        assert (
            seen[1]["contents"][-1]["parts"][0]["functionResponse"]["id"] == "weather_1"
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("part_index", [0, 1])
    @pytest.mark.parametrize("hosted", [False, True])
    async def test_unsigned_server_history_is_rejected_even_if_tools_are_omitted(
        self, part_index: int, hosted: bool
    ) -> None:
        parts = self._parts()
        parts[part_index].pop("thoughtSignature")
        with pytest.raises(
            DialectError, match=r"tool(Call|Response).*thoughtSignature"
        ):
            await _run(
                _no_io,
                self._replay_request(parts, hosted=hosted),
                model="gemini-3.8-flash",
            )

    @pytest.mark.anyio
    async def test_direct_prepare_checks_server_history_signatures(self) -> None:
        connection = _connection(_no_io)
        try:
            parts = self._parts()
            parts[0].pop("thoughtSignature")
            req = self._replay_request(parts)
            dialect = GoogleGenAIDialect(connection, "gemini-3.8-flash")
            with pytest.raises(DialectError, match="toolCall.*thoughtSignature"):
                dialect.prepare(req, RequestAudit(active_request_leaves(req)))
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("part_index", [0, 1])
    async def test_final_http_server_signature_loss_is_rejected(
        self, part_index: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = _connection(_no_io)
        build_request = connection.client.build_request

        def drifted_request(*args: Any, **kwargs: Any) -> httpx.Request:
            request = build_request(*args, **kwargs)
            body = json.loads(request.content)
            body["contents"][1]["parts"][part_index].pop("thoughtSignature")
            return httpx.Request(request.method, request.url, json=body)

        monkeypatch.setattr(connection.client, "build_request", drifted_request)
        try:
            with pytest.raises(
                DialectError, match=r"tool(Call|Response).*thoughtSignature"
            ):
                await GoogleGenAIDialect(connection, "gemini-3.8-flash").arun(
                    self._replay_request(self._parts())
                )
        finally:
            await connection.aclose()


async def _assert_thinking_reaches_wire(
    model: str, reasoning: ReasoningParams, expected: dict[str, object]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        thinking = json.loads(request.content)["generationConfig"]["thinkingConfig"]
        assert thinking == expected
        return httpx.Response(
            200,
            json=_response([{"text": "thought", "thought": True}, {"text": "answer"}]),
        )

    await _run(handler, _request(reasoning=reasoning), model=model)


class TestModelSpecificRequestConstraints:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            ("gemini-2.5-pro", "high"),
            ("gemini-3.1-pro-preview", "minimal"),
            ("gemini-3.8-flash", "minimal"),
            ("gateway-alias", "minimal"),
        ],
    )
    async def test_thinking_levels_are_not_checked_per_model(
        self, model: str, effort: str
    ) -> None:
        # Google 400s an unsupported level itself; send it unchanged.
        await _assert_thinking_reaches_wire(
            model, ReasoningParams(effort=effort), {"thinkingLevel": effort.upper()}
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            ("gemini-3.6-flash", "minimal"),
            ("gemini-3.5-flash-lite", "minimal"),
            ("gemini-3-pro-preview", "low"),
            ("gemini-3-pro-preview", "high"),
            ("gemini-3.1-pro-preview", "low"),
            ("gemini-3.1-flash-lite-image", "high"),
            ("gemini-3.8-flash", "medium"),
        ],
    )
    async def test_documented_thinking_levels_reach_the_wire(
        self, model: str, effort: str
    ) -> None:
        await _assert_thinking_reaches_wire(
            model, ReasoningParams(effort=effort), {"thinkingLevel": effort.upper()}
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model", "budget"),
        [
            ("gemini-2.5-pro", 127),
            ("gemini-2.5-pro", 32769),
            ("gemini-2.5-flash", 24577),
            ("gemini-2.5-flash-lite", 511),
            ("gemini-2.5-flash-lite", 24577),
        ],
    )
    async def test_thinking_budgets_are_not_range_checked_per_model(
        self, model: str, budget: int
    ) -> None:
        # Google 400s an out-of-range budget itself; send it unchanged.
        await _assert_thinking_reaches_wire(
            model, ReasoningParams(budget_tokens=budget), {"thinkingBudget": budget}
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model", "budget"),
        [
            ("gemini-2.5-pro", 128),
            ("gemini-2.5-pro", 32768),
            ("gemini-2.5-flash", 1),
            ("gemini-2.5-flash", 24576),
            ("gemini-2.5-flash-lite", 512),
            ("gemini-2.5-flash-lite", 24576),
        ],
    )
    async def test_documented_thinking_budget_boundaries_reach_the_wire(
        self, model: str, budget: int
    ) -> None:
        await _assert_thinking_reaches_wire(
            model, ReasoningParams(budget_tokens=budget), {"thinkingBudget": budget}
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("sampling", "expected"),
        [
            (SamplingParams(temperature=0.0), {"temperature": 0.0}),
            (SamplingParams(top_p=0.8), {"topP": 0.8}),
            (SamplingParams(top_k=1), {"topK": 1}),
            (
                SamplingParams(temperature=0.0, top_p=0.5, top_k=4),
                {"temperature": 0.0, "topP": 0.5, "topK": 4},
            ),
        ],
    )
    async def test_gemini_3_forwards_sampling_controls(
        self, sampling: SamplingParams, expected: dict[str, object]
    ) -> None:
        # Google advises omitting these on Gemini 3 but accepts them, and #153
        # measured temperature and top_p changing the output. Send what is set.
        def handler(request: httpx.Request) -> httpx.Response:
            generation = json.loads(request.content)["generationConfig"]
            assert {key: generation[key] for key in expected} == expected
            return httpx.Response(200, json=_response())

        response = await _run(
            handler,
            _request(sampling=sampling, reasoning=ReasoningParams(effort="low")),
            model="gemini-3.5-flash",
        )
        body = cast(dict[str, Any], response.request_params)["body"]
        generation = cast(dict[str, Any], body)["generationConfig"]
        assert {key: generation[key] for key in expected} == expected
        assert generation["thinkingConfig"] == {"thinkingLevel": "LOW"}


class TestReasoningAndStructuredOutput:
    @pytest.mark.anyio
    async def test_visible_thoughts_and_full_history_round_trip(self) -> None:
        first_body: dict[str, Any] = {}

        def first_handler(request: httpx.Request) -> httpx.Response:
            first_body.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "text": "analysis",
                            "thought": True,
                            "thoughtSignature": "sig_1",
                        },
                        {"text": "answer"},
                    ]
                ),
            )

        first = await _run(
            first_handler,
            _request(reasoning=ReasoningParams(effort="high", summary="auto")),
            model="gemini-3.5-flash",
        )
        assert first.reasoning is not None
        assert first.reasoning[0] is not None
        assert first.reasoning[0].text == "analysis"
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)

        def second_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][:2] == [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": "analysis",
                            "thought": True,
                            "thoughtSignature": "sig_1",
                        },
                        {"text": "answer"},
                    ],
                },
            ]
            assert body["contents"][2] == {
                "role": "user",
                "parts": [{"text": "next"}],
            }
            return httpx.Response(
                200,
                json=_response([{"text": "next answer", "thoughtSignature": "sig_2"}]),
            )

        second = await _run(
            second_handler,
            _request(
                input=_chat(ChatMessage("user", (TextPart("next"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
            model="gemini-3.5-flash",
        )
        assert second.reasoning is not None
        assert second.reasoning[0] is not None
        assert second.reasoning[0].opaque_roundtrip

    @pytest.mark.anyio
    async def test_opaque_history_preserves_system_instruction(self) -> None:
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "text": "analysis",
                            "thought": True,
                            "thoughtSignature": "sig_1",
                        },
                        {"text": "answer"},
                    ]
                ),
            ),
            _request(
                input=_chat(
                    ChatMessage("system", (TextPart("Always answer as JSON"),)),
                    ChatMessage("user", (TextPart("first"),)),
                )
            ),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        envelope = json.loads(opaque)
        assert envelope["systemInstruction"] == {
            "parts": [{"text": "Always answer as JSON"}]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["systemInstruction"] == envelope["systemInstruction"]
            return httpx.Response(
                200,
                json=_response([{"text": "next", "thoughtSignature": "sig_2"}]),
            )

        await _run(
            handler,
            _request(
                input=_chat(ChatMessage("user", (TextPart("next"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
        )

    @pytest.mark.anyio
    async def test_opaque_history_rejects_conflicting_system_instruction(self) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "systemInstruction": {"parts": [{"text": "original"}]},
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"text": "answer", "thoughtSignature": "signature"}],
                    }
                ],
            }
        )
        req = _request(
            input=_chat(
                ChatMessage("system", (TextPart("replacement"),)),
                ChatMessage("user", (TextPart("next"),)),
            ),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("google_genai", opaque)
            ),
        )

        with pytest.raises(DialectError, match="systemInstruction conflicts"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("server_parts", "message"),
        [
            (
                [
                    {
                        "toolResponse": {
                            "id": "server_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "response": {"ok": True},
                        }
                    }
                ],
                "toolResponse.*has no prior toolCall",
            ),
            (
                [
                    {
                        "toolCall": {
                            "id": "server_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "args": {},
                        }
                    },
                    {
                        "toolCall": {
                            "id": "server_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "args": {},
                        }
                    },
                ],
                "duplicate Google server tool call",
            ),
            (
                [
                    {
                        "toolCall": {
                            "id": "server_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "args": {},
                        }
                    },
                    {
                        "toolResponse": {
                            "id": "server_1",
                            "toolType": "URL_CONTEXT",
                            "response": {"ok": True},
                        }
                    },
                ],
                "toolResponse type changed",
            ),
            (
                [
                    {
                        "toolCall": {
                            "id": "server_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "args": {},
                        }
                    },
                    {
                        "toolResponse": {
                            "id": "server_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "response": {"ok": True},
                        }
                    },
                    {
                        "toolResponse": {
                            "id": "server_1",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "response": {"ok": True},
                        }
                    },
                ],
                "duplicate Google server tool result",
            ),
        ],
    )
    async def test_opaque_server_tool_history_rejects_invalid_call_state_before_io(
        self, server_parts: list[dict[str, object]], message: str
    ) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": "history",
                                "thoughtSignature": "signature",
                            },
                            *server_parts,
                        ],
                    }
                ],
            }
        )

        with pytest.raises(DialectError, match=message):
            await _run(
                _no_io,
                _request(
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    )
                ),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "server_parts",
        [
            [
                {
                    "toolCall": {
                        "id": "server_1",
                        "toolType": "GOOGLE_SEARCH_WEB",
                        "args": {"q": "weather"},
                    },
                    "thoughtSignature": "signature",
                }
            ],
            [
                {
                    "toolCall": {
                        "id": "server_1",
                        "toolType": "GOOGLE_SEARCH_WEB",
                        "args": {"q": "weather"},
                    },
                    "thoughtSignature": "signature",
                },
                {
                    "toolResponse": {
                        "id": "server_1",
                        "toolType": "GOOGLE_SEARCH_WEB",
                        "response": {"ok": True},
                    }
                },
            ],
        ],
    )
    async def test_opaque_server_tool_history_accepts_open_or_answered_call(
        self, server_parts: list[dict[str, object]]
    ) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [{"role": "model", "parts": server_parts}],
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["contents"][0]["parts"] == server_parts
            return httpx.Response(200, json=_response())

        result = await _run(
            handler,
            _request(
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                )
            ),
        )

        assert result.texts == ("Hi",)

    @pytest.mark.anyio
    async def test_requested_summary_is_optional_when_thoughts_are_unavailable(
        self,
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(200, json=_response()),
            _request(reasoning=ReasoningParams(summary="auto")),
        )

        assert result.reasoning == (None,)

    @pytest.mark.anyio
    async def test_a_summary_with_no_thoughts_back_is_recorded_as_no_effect(
        self,
    ) -> None:
        """Google accepts `includeThoughts` on a model that then returns none
        (measured: gemini-3.5-flash with `thinkingBudget: 0`). Sent as written,
        not refused; the record says the control had no effect."""
        result = await _run(
            lambda _: httpx.Response(200, json=_response()),
            _request(reasoning=ReasoningParams(summary="auto")),
        )
        assert result.request_params is not None
        assert result.request_params["no_effect"] == ["reasoning.summary"]
        assert result.request_params["body"]["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": True
        }

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("budget", "thoughts", "no_effect"),
        [(128, 0, True), (128, None, True), (128, 40, False)],
    )
    async def test_a_budget_the_reply_contradicts_is_recorded_as_no_effect(
        self, budget: int, thoughts: int | None, no_effect: bool
    ) -> None:
        usage: dict[str, object] = {"promptTokenCount": 4, "candidatesTokenCount": 2}
        if thoughts is not None:
            usage["thoughtsTokenCount"] = thoughts
        result = await _run(
            lambda _: httpx.Response(200, json=_response(usage=usage)),
            _request(reasoning=ReasoningParams(budget_tokens=budget)),
        )
        assert result.request_params is not None
        recorded = result.request_params.get("no_effect", [])
        assert ("reasoning.budget_tokens" in recorded) is no_effect

    @pytest.mark.anyio
    async def test_controls_that_took_effect_record_nothing(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response([{"text": "hmm", "thought": True}, {"text": "Hi"}]),
            ),
            _request(reasoning=ReasoningParams(budget_tokens=128, summary="auto")),
        )
        assert result.request_params is not None
        assert "no_effect" not in result.request_params

    @pytest.mark.anyio
    async def test_a_truncated_reply_records_no_effect_for_nothing(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200, json=_response([{"text": ""}], finish_reason="MAX_TOKENS")
            ),
            _request(reasoning=ReasoningParams(summary="auto")),
        )
        assert result.request_params is not None
        assert "no_effect" not in result.request_params

    @pytest.mark.anyio
    async def test_requested_summary_rejects_whitespace_only_thoughts(self) -> None:
        with pytest.raises(
            OutputContractError, match="empty requested thought summary"
        ):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [{"text": " \n\t ", "thought": True}, {"text": "answer"}]
                    ),
                ),
                _request(reasoning=ReasoningParams(summary="auto")),
            )

    @pytest.mark.anyio
    async def test_truncated_reply_does_not_require_a_visible_thought_summary(
        self,
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [{"text": " \n ", "thought": True}],
                    finish_reason="MAX_TOKENS",
                ),
            ),
            _request(reasoning=ReasoningParams(summary="auto")),
        )

        assert result.finish_reasons == ("MAX_TOKENS",)
        assert result.reasoning is not None

    @pytest.mark.anyio
    async def test_empty_truncated_reply_preserves_prior_opaque_state(self) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": "prior",
                                "thoughtSignature": "signature",
                            }
                        ],
                    }
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        )

        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(parts=[], finish_reason="MAX_TOKENS"),
            ),
            _request(
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                )
            ),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("MAX_TOKENS",)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].opaque_roundtrip == opaque

    @pytest.mark.anyio
    async def test_truncated_structured_output_preserves_partial_text(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [{"text": '{"answer":'}],
                    finish_reason="MAX_TOKENS",
                ),
            ),
            _request(structured_output=StructuredOutputParams(format="json_object")),
        )

        assert result.texts == ('{"answer":',)
        assert result.finish_reasons == ("MAX_TOKENS",)
        assert result.structured_output is None

    @pytest.mark.anyio
    @pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', "not-json"])
    async def test_structured_output_requires_strict_json(self, text: str) -> None:
        with pytest.raises(OutputContractError, match="valid JSON"):
            await _run(
                lambda _: httpx.Response(200, json=_response([{"text": text}])),
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={"type": "object"}
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_structured_output_uses_response_json_schema(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["generationConfig"] == {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            }
            return httpx.Response(200, json=_response([{"text": '{"answer":42}'}]))

        result = await _run(
            handler,
            _request(
                structured_output=StructuredOutputParams(
                    format="json_schema", schema=schema
                )
            ),
        )
        assert result.structured_output is not None
        assert result.structured_output.value == {"answer": 42}


class TestResponseLifting:
    @pytest.mark.anyio
    async def test_documented_part_metadata_is_preserved_for_replay(self) -> None:
        part = {
            "text": "answer",
            "thoughtSignature": "signature",
            "partMetadata": {"source": "provider"},
        }

        result = await _run(
            lambda _: httpx.Response(200, json=_response([part])),
        )

        assert result.texts == ("answer",)
        assert result.reasoning is not None and result.reasoning[0] is not None
        opaque = json.loads(cast(str, result.reasoning[0].opaque_roundtrip))
        assert opaque["contents"][-1]["parts"] == [part]

    @pytest.mark.anyio
    async def test_blocked_prompt_surfaces_provider_reason(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json={"promptFeedback": {"blockReason": "SAFETY"}},
            ),
            _request(
                reasoning=ReasoningParams(summary="auto"),
                structured_output=StructuredOutputParams(format="json_object"),
            ),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("SAFETY",)
        assert result.finish_reason_sources == ("promptFeedback.blockReason",)
        assert result.reasoning is None
        assert result.structured_output is None

    @pytest.mark.anyio
    @pytest.mark.parametrize("finish_reason", ["PROHIBITED_CONTENT", "LANGUAGE"])
    async def test_candidate_policy_stop_is_a_scoreable_empty_outcome(
        self, finish_reason: str
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(parts=[], finish_reason=finish_reason),
            ),
            _request(reasoning=ReasoningParams(summary="auto")),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == (finish_reason,)
        assert result.finish_reason_sources == ("candidate.finishReason",)
        assert result.reasoning == (None,)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "finish_reason",
        [
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MALFORMED_RESPONSE",
        ],
    )
    async def test_model_output_failures_are_scoreable_empty_outcomes(
        self, finish_reason: str
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(parts=[], finish_reason=finish_reason),
            ),
            _request(
                reasoning=ReasoningParams(summary="auto"),
                scoring=ScoringParams(sampled_logprobs=True),
                structured_output=StructuredOutputParams(format="json_object"),
            ),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == (finish_reason,)
        assert result.reasoning == (None,)
        assert result.logprobs is None
        assert result.structured_output is None

    @pytest.mark.anyio
    async def test_continuation_is_a_scoreable_truncated_empty_outcome(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(parts=[], finish_reason="CONTINUATION"),
            ),
            _request(
                reasoning=ReasoningParams(summary="auto"),
                scoring=ScoringParams(sampled_logprobs=True),
                structured_output=StructuredOutputParams(format="json_object"),
            ),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("CONTINUATION",)
        assert result.reasoning == (None,)
        assert result.logprobs is None
        assert result.structured_output is None

    @pytest.mark.anyio
    @pytest.mark.parametrize("finish_reason", ["OTHER", "MALFORMED_FUNCTION_CALL"])
    async def test_non_normal_finish_reason_preserves_returned_text(
        self, finish_reason: str
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    parts=[{"text": "provider returned this text"}],
                    finish_reason=finish_reason,
                ),
            )
        )

        assert result.texts == ("provider returned this text",)
        assert result.finish_reasons == (finish_reason,)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("finish_reason", "message"),
        [
            ("MISSING_THOUGHT_SIGNATURE", "required thought signature"),
            ("PUP_LIMITED_DISABLED", "account is limited or disabled"),
        ],
    )
    async def test_request_or_account_failures_fail_loudly(
        self, finish_reason: str, message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(parts=[], finish_reason=finish_reason),
                )
            )

    @pytest.mark.anyio
    async def test_unspecified_prompt_block_reason_allows_candidates(self) -> None:
        payload = _response()
        payload["promptFeedback"] = {"blockReason": "BLOCK_REASON_UNSPECIFIED"}

        result = await _run(lambda _: httpx.Response(200, json=payload))

        assert result.texts == ("Hi",)

    @pytest.mark.anyio
    async def test_candidates_with_prompt_block_reason_are_contradictory(self) -> None:
        payload = _response()
        payload["promptFeedback"] = {"blockReason": "SAFETY"}

        with pytest.raises(OutputContractError, match="contradictory.*SAFETY"):
            await _run(lambda _: httpx.Response(200, json=payload))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("prompt_feedback", "message"),
        [
            ([], "promptFeedback must be an object"),
            ({"blockReason": None}, "blockReason must be a non-empty string"),
            ({"blockReason": ""}, "blockReason must be a non-empty string"),
            ({"blockReason": "   "}, "blockReason must be a non-empty string"),
            ({"blockReason": 1}, "blockReason must be a non-empty string"),
        ],
    )
    async def test_malformed_prompt_feedback_fails_loudly(
        self, prompt_feedback: object, message: str
    ) -> None:
        payload = _response()
        payload["promptFeedback"] = prompt_feedback

        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=payload))

    @pytest.mark.anyio
    async def test_multiple_text_candidates_preserve_choice_order(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "index": 0,
                            "content": {"role": "model", "parts": [{"text": "A"}]},
                            "finishReason": "STOP",
                        },
                        {
                            "index": 1,
                            "content": {"role": "model", "parts": [{"text": "B"}]},
                            "finishReason": "MAX_TOKENS",
                        },
                    ]
                },
            ),
            _request(sampling=SamplingParams(n=2)),
        )

        assert result.texts == ("A", "B")
        assert result.finish_reasons == ("STOP", "MAX_TOKENS")
        assert result.finish_reason_sources == ("candidate.finishReason",) * 2

    @pytest.mark.anyio
    async def test_streamed_candidate_filter_preserves_distinct_source(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {"index": 0, "finishReason": "SAFETY"},
                        ]
                    }
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )
        assert result.texts == ("",)
        assert result.finish_reasons == ("SAFETY",)
        assert result.finish_reason_sources == ("candidate.finishReason",)

    @pytest.mark.anyio
    async def test_multiple_candidates_reject_singular_response_metadata(self) -> None:
        with pytest.raises(OutputContractError, match="singular citation"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "A"}],
                                },
                                "finishReason": "STOP",
                                "citationMetadata": {
                                    "citationSources": [{"uri": "https://a.test"}]
                                },
                            },
                            {
                                "index": 1,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "B"}],
                                },
                                "finishReason": "STOP",
                            },
                        ]
                    },
                ),
                _request(sampling=SamplingParams(n=2)),
            )

    @pytest.mark.anyio
    async def test_older_model_missing_function_call_id_is_stable_and_replayable(
        self,
    ) -> None:
        parts = [
            {
                "functionCall": {"name": "weather", "args": {"city": "Paris"}},
                "thoughtSignature": "signature",
            }
        ]
        tools = ToolParams(
            functions=({"name": "weather", "description": "Get weather"},)
        )
        request = _request(tools=tools)
        first = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            request,
        )
        repeated = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            request,
        )

        assert first.tool_calls is not None
        call_id = first.tool_calls[0].call_id
        assert call_id.startswith("google_call_")
        assert repeated.tool_calls is not None
        assert repeated.tool_calls[0].call_id == call_id
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        assert repeated.reasoning is not None and repeated.reasoning[0] is not None
        assert repeated.reasoning[0].opaque_roundtrip == opaque
        envelope = json.loads(opaque)
        assert "id" not in envelope["contents"][-1]["parts"][0]["functionCall"]
        assert envelope["syntheticCallIds"] == [
            {
                "contentIndex": 1,
                "partIndex": 0,
                "field": "functionCall",
                "id": call_id,
            }
        ]

        changed = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            _request(
                input=_chat(ChatMessage("user", (TextPart("Different input"),))),
                tools=tools,
            ),
        )
        assert changed.tool_calls is not None
        assert changed.tool_calls[0].call_id != call_id

        def handler(http_request: httpx.Request) -> httpx.Response:
            body = json.loads(http_request.content)
            assert body["contents"][-2]["parts"] == parts
            assert body["contents"][-1]["parts"] == [
                {
                    "functionResponse": {
                        "name": "weather",
                        "response": {"temperature": 20},
                    }
                }
            ]
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage(
                        "tool",
                        (ToolResultPart(call_id, {"temperature": 20}),),
                    )
                ),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("model", ["gemini-3-pro-preview", "gemini-3.5-flash"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_gemini_3_rejects_a_missing_function_call_id(
        self, model: str, stream: bool
    ) -> None:
        tools = ToolParams(functions=({"name": "f", "description": "Function f"},))
        payload = _response([{"functionCall": {"name": "f", "args": {}}}])

        def handler(_: httpx.Request) -> httpx.Response:
            if stream:
                return httpx.Response(200, content=_sse(payload))
            return httpx.Response(200, json=payload)

        with pytest.raises(
            OutputContractError, match="response omitted the required functionCall.id"
        ):
            await _run(
                handler,
                _request(
                    tools=tools,
                    scheduling=SchedulingParams(stream=stream),
                ),
                model=model,
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize("model", ["gemini-3-pro-preview", "gemini-3.5-flash"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_gemini_3_rejects_a_missing_function_call_thought_signature(
        self, model: str, stream: bool
    ) -> None:
        tools = ToolParams(functions=({"name": "f", "description": "Function f"},))
        payload = _response(
            [{"functionCall": {"id": "call_1", "name": "f", "args": {}}}]
        )

        def handler(_: httpx.Request) -> httpx.Response:
            if stream:
                return httpx.Response(200, content=_sse(payload))
            return httpx.Response(200, json=payload)

        with pytest.raises(
            OutputContractError,
            match="function-call response omitted.*thoughtSignature",
        ):
            await _run(
                handler,
                _request(
                    tools=tools,
                    scheduling=SchedulingParams(stream=stream),
                ),
                model=model,
            )

    @pytest.mark.anyio
    async def test_gemini_3_parallel_calls_require_a_signature_only_on_the_first(
        self,
    ) -> None:
        tools = ToolParams(
            functions=(
                {"name": "first", "description": "First function"},
                {"name": "second", "description": "Second function"},
            )
        )
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "functionCall": {
                                "id": "call_a",
                                "name": "first",
                                "args": {},
                            },
                            "thoughtSignature": "signature",
                        },
                        {
                            "functionCall": {
                                "id": "call_b",
                                "name": "second",
                                "args": {},
                            }
                        },
                    ]
                ),
            ),
            _request(tools=tools),
            model="gemini-3.5-flash",
        )

        assert result.tool_calls is not None
        assert [call.call_id for call in result.tool_calls] == ["call_a", "call_b"]
        assert result.reasoning is not None and result.reasoning[0] is not None
        opaque = json.loads(cast(str, result.reasoning[0].opaque_roundtrip))
        assert opaque["contents"][-1]["parts"] == [
            {
                "functionCall": {"id": "call_a", "name": "first", "args": {}},
                "thoughtSignature": "signature",
            },
            {"functionCall": {"id": "call_b", "name": "second", "args": {}}},
        ]

    @pytest.mark.anyio
    async def test_gemini_3_function_call_signature_cannot_float_to_another_part(
        self,
    ) -> None:
        tools = ToolParams(functions=({"name": "f", "description": "Function f"},))

        with pytest.raises(
            OutputContractError,
            match="function-call response omitted.*thoughtSignature",
        ):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "text": "planning",
                                "thoughtSignature": "floating_signature",
                            },
                            {
                                "functionCall": {
                                    "id": "call_1",
                                    "name": "f",
                                    "args": {},
                                }
                            },
                        ]
                    ),
                ),
                _request(tools=tools),
                model="gemini-3.5-flash",
            )

    @pytest.mark.anyio
    async def test_gemini_3_preserves_the_exact_function_call_id(self) -> None:
        tools = ToolParams(functions=({"name": "f", "description": "Function f"},))
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "functionCall": {
                                "id": "provider_call_1",
                                "name": "f",
                                "args": {},
                            },
                            "thoughtSignature": "signature",
                        }
                    ]
                ),
            ),
            _request(tools=tools),
            model="gemini-3.5-flash",
        )

        assert first.tool_calls is not None
        assert first.tool_calls[0].call_id == "provider_call_1"
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        assert "syntheticCallIds" not in json.loads(opaque)

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][-1] == {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "provider_call_1",
                            "name": "f",
                            "response": {"ok": True},
                        }
                    }
                ],
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage(
                        "tool", (ToolResultPart("provider_call_1", {"ok": True}),)
                    )
                ),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
            model="gemini-3.5-flash",
        )

    @pytest.mark.anyio
    async def test_opaque_history_restores_answered_function_calls(self) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"id": "call_1", "name": "f"},
                                "thoughtSignature": "signature",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": "call_1",
                                    "name": "f",
                                    "response": {"ok": True},
                                }
                            }
                        ],
                    },
                ],
            }
        )
        req = _request(
            input=_chat(
                ChatMessage("tool", (ToolResultPart("call_1", {"again": True}),))
            ),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("google_genai", opaque)
            ),
        )

        with pytest.raises(DialectError, match="duplicate Google function result"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_older_model_may_answer_opaque_synthetic_function_call(
        self,
    ) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "f"},
                                "thoughtSignature": "signature",
                            }
                        ],
                    }
                ],
                "syntheticCallIds": [
                    {
                        "contentIndex": 0,
                        "partIndex": 0,
                        "field": "functionCall",
                        "id": "call_1",
                    }
                ],
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][-1] == {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "f",
                            "response": {"ok": True},
                        }
                    }
                ],
            }
            return httpx.Response(200, json=_response())

        result = await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage("tool", (ToolResultPart("call_1", {"ok": True}),))
                ),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
            model="gemini-2.5-flash",
        )

        assert result.texts == ("Hi",)

    @pytest.mark.anyio
    async def test_gemini_3_rejects_opaque_synthetic_function_call_ids(self) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "f"},
                                "thoughtSignature": "signature",
                            }
                        ],
                    }
                ],
                "syntheticCallIds": [
                    {
                        "contentIndex": 0,
                        "partIndex": 0,
                        "field": "functionCall",
                        "id": "call_1",
                    }
                ],
            }
        )

        with pytest.raises(
            DialectError, match="requires every functionCall.*provider call id"
        ):
            await _run(
                _no_io,
                _request(
                    input=_chat(
                        ChatMessage("tool", (ToolResultPart("call_1", {"ok": True}),))
                    ),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    ),
                ),
                model="gemini-3.5-flash",
            )

    @pytest.mark.anyio
    async def test_gemini_3_rejects_unsigned_canonical_current_turn_history(
        self,
    ) -> None:
        request = _request(
            input=_chat(
                ChatMessage("user", (TextPart("Call the function"),)),
                ChatMessage("assistant", (ToolCallPart("call_1", "f", {}),)),
                ChatMessage("tool", (ToolResultPart("call_1", {"ok": True}),)),
            ),
            tools=ToolParams(functions=({"name": "f", "description": "Function f"},)),
        )

        with pytest.raises(
            DialectError, match="current-turn functionCall.*thoughtSignature"
        ):
            await _run(_no_io, request, model="gemini-3.5-flash")

    @pytest.mark.anyio
    async def test_gemini_3_allows_unsigned_calls_before_the_current_turn(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][-1] == {
                "role": "user",
                "parts": [{"text": "Start a new turn"}],
            }
            return httpx.Response(200, json=_response())

        result = await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage("user", (TextPart("Call the function"),)),
                    ChatMessage("assistant", (ToolCallPart("call_1", "f", {}),)),
                    ChatMessage("tool", (ToolResultPart("call_1", {"ok": True}),)),
                    ChatMessage("user", (TextPart("Start a new turn"),)),
                ),
                tools=ToolParams(
                    functions=({"name": "f", "description": "Function f"},)
                ),
            ),
            model="gemini-3.5-flash",
        )

        assert result.texts == ("Hi",)

    @pytest.mark.anyio
    async def test_gemini_3_parallel_opaque_calls_need_only_the_first_signature(
        self,
    ) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {"role": "user", "parts": [{"text": "Call both"}]},
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"id": "call_a", "name": "first"},
                                "thoughtSignature": "signature",
                            },
                            {"functionCall": {"id": "call_b", "name": "second"}},
                        ],
                    },
                ],
            }
        )

        result = await _run(
            lambda _: httpx.Response(200, json=_response()),
            _request(
                input=_chat(
                    ChatMessage(
                        "tool",
                        (
                            ToolResultPart("call_a", {"first": True}),
                            ToolResultPart("call_b", {"second": True}),
                        ),
                    )
                ),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
            model="gemini-3.5-flash",
        )

        assert result.texts == ("Hi",)

    @pytest.mark.anyio
    async def test_gemini_3_requires_a_signature_on_each_current_turn_step(
        self,
    ) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {"role": "user", "parts": [{"text": "Call both in order"}]},
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"id": "call_a", "name": "first"},
                                "thoughtSignature": "signature_a",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": "call_a",
                                    "name": "first",
                                    "response": {"ok": True},
                                }
                            }
                        ],
                    },
                    {
                        "role": "model",
                        "parts": [{"functionCall": {"id": "call_b", "name": "second"}}],
                    },
                ],
            }
        )

        with pytest.raises(
            DialectError, match="current-turn functionCall.*thoughtSignature"
        ):
            await _run(
                _no_io,
                _request(
                    input=_chat(
                        ChatMessage("tool", (ToolResultPart("call_b", {"ok": True}),))
                    ),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    ),
                ),
                model="gemini-3.5-flash",
            )

    @pytest.mark.anyio
    async def test_opaque_function_responses_must_be_turn_local(self) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"id": "call_a", "name": "first"},
                                "thoughtSignature": "signature",
                            }
                        ],
                    },
                    {"role": "user", "parts": [{"text": "intervening"}]},
                    {
                        "role": "model",
                        "parts": [{"functionCall": {"id": "call_b", "name": "second"}}],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": "call_a",
                                    "name": "first",
                                    "response": {"ok": True},
                                }
                            },
                            {
                                "functionResponse": {
                                    "id": "call_b",
                                    "name": "second",
                                    "response": {"ok": True},
                                }
                            },
                        ],
                    },
                ],
            }
        )

        with pytest.raises(DialectError, match="immediately follow.*call_a"):
            await _run(
                _no_io,
                _request(
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_usage_logprobs_citations_grounding_and_server_tools(self) -> None:
        logprobs = {
            "chosenCandidates": [{"token": "Hi", "logProbability": -0.1, "tokenId": 7}],
            "topCandidates": [
                {
                    "candidates": [
                        {"token": "Hi", "logProbability": -0.1, "tokenId": 7},
                        {"token": "Hey", "logProbability": -1.2, "tokenId": 8},
                    ]
                }
            ],
        }
        candidate_extra = {
            "logprobsResult": logprobs,
            "citationMetadata": {"citationSources": [{"uri": "https://cite"}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://web", "title": "Web"}},
                    {
                        "image": {
                            "sourceUri": "https://image-source",
                            "title": "Image",
                        }
                    },
                ],
                "searchEntryPoint": {"renderedContent": "<div>sources</div>"},
            },
        }
        parts = [
            {
                "toolCall": {
                    "id": "server_1",
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "args": {"q": "weather"},
                }
            },
            {
                "toolResponse": {
                    "id": "server_1",
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "response": {"ok": True},
                }
            },
            {"text": "Hi"},
        ]
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(parts, candidate_extra=candidate_extra),
            ),
            _request(
                scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
                tools=ToolParams(hosted=(HostedToolSpec("google_search"),)),
            ),
        )

        assert result.logprobs is not None and result.logprobs[0].token_id == 7
        assert result.top_logprobs is not None
        assert [item.token for item in result.top_logprobs[0]] == ["Hi", "Hey"]
        assert result.citations is not None
        assert result.citations[0].url == "https://cite"
        assert result.grounding is not None
        assert [chunk.uri for chunk in result.grounding.chunks] == [
            "https://web",
            "https://image-source",
        ]
        assert result.server_tool_uses is not None
        assert result.server_tool_uses[0].tool_type == "GOOGLE_SEARCH_WEB"
        assert result.server_tool_uses[0].result == {"ok": True}
        assert result.usage is not None
        assert result.usage.input_tokens == 4
        assert result.usage.output_tokens == 3  # candidates 2 + thoughts 1
        assert result.usage.total_tokens == 7
        assert result.usage.reported_total_tokens is None  # matches computed total
        assert result.usage.reasoning_tokens == 1
        assert result.usage.cached_tokens == 2

    @pytest.mark.anyio
    async def test_missing_server_tool_ids_are_synthesized_and_paired(self) -> None:
        parts = [
            {
                "toolCall": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "args": {"q": "weather"},
                },
                "thoughtSignature": "signature",
            },
            {
                "toolResponse": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "response": {"ok": True},
                }
            },
            {"text": "sunny"},
        ]
        result = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            _request(tools=ToolParams(hosted=(HostedToolSpec("google_search"),))),
        )

        assert result.server_tool_uses is not None
        tool_id = result.server_tool_uses[0].tool_use_id
        assert tool_id.startswith("google_server_")
        assert result.server_tool_uses[0].result == {"ok": True}
        assert result.reasoning is not None and result.reasoning[0] is not None
        opaque = json.loads(cast(str, result.reasoning[0].opaque_roundtrip))
        opaque_parts = opaque["contents"][-1]["parts"]
        assert "id" not in opaque_parts[0]["toolCall"]
        assert "id" not in opaque_parts[1]["toolResponse"]
        assert opaque["syntheticCallIds"] == [
            {
                "contentIndex": 1,
                "partIndex": 0,
                "field": "toolCall",
                "id": tool_id,
            },
            {
                "contentIndex": 1,
                "partIndex": 1,
                "field": "toolResponse",
                "id": tool_id,
            },
        ]

    @pytest.mark.anyio
    async def test_usage_folds_thoughts_and_tool_use_prompt_into_parent_counts(
        self,
    ) -> None:
        """toolUsePromptTokenCount + thoughtsTokenCount are addends, not breakdowns.

        Google reports ``totalTokenCount = promptTokenCount + toolUsePromptTokenCount
        + thoughtsTokenCount + candidatesTokenCount``, but the IR expects
        ``output_tokens >= reasoning_tokens`` (reasoning is a breakdown of output).
        Folding the addends into their parents preserves profiler share semantics.
        """
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    usage={
                        "promptTokenCount": 10,
                        "toolUsePromptTokenCount": 3,
                        "thoughtsTokenCount": 5,
                        "candidatesTokenCount": 8,
                        "totalTokenCount": 26,
                    }
                ),
            )
        )
        assert result.usage is not None
        assert result.usage.input_tokens == 13  # prompt 10 + tool_use_prompt 3
        assert result.usage.output_tokens == 13  # candidates 8 + thoughts 5
        assert result.usage.reasoning_tokens == 5
        assert result.usage.total_tokens == 26
        assert result.usage.reported_total_tokens is None  # matches computed

    @pytest.mark.anyio
    async def test_usage_computes_total_without_claiming_provider_report(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    usage={"promptTokenCount": 3, "candidatesTokenCount": 4}
                ),
            )
        )
        assert result.usage is not None
        assert result.usage.total_tokens == 7
        assert result.usage.reported_total_tokens is None

    @pytest.mark.anyio
    async def test_matching_provider_total_is_not_redundantly_persisted(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    usage={
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 7,
                    }
                ),
            )
        )
        assert result.usage is not None
        assert result.usage.total_tokens == 7
        assert result.usage.reported_total_tokens is None

    @pytest.mark.anyio
    async def test_requested_top_logprobs_require_exact_cardinality(self) -> None:
        logprobs = {
            "chosenCandidates": [{"token": "Hi", "logProbability": -0.1}],
            "topCandidates": [
                {"candidates": [{"token": "Hi", "logProbability": -0.1}]}
            ],
        }
        with pytest.raises(OutputContractError, match="different number"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(candidate_extra={"logprobsResult": logprobs}),
                ),
                _request(
                    scoring=ScoringParams(
                        sampled_logprobs=True,
                        top_logprobs=2,
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_nonempty_text_requires_chosen_logprobs(self) -> None:
        logprobs = {"chosenCandidates": [], "topCandidates": []}
        with pytest.raises(OutputContractError, match="without chosen logprobs"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(candidate_extra={"logprobsResult": logprobs}),
                ),
                _request(scoring=ScoringParams(sampled_logprobs=True)),
            )

    @pytest.mark.anyio
    async def test_duplicate_provider_function_call_ids_fail_loudly(self) -> None:
        with pytest.raises(OutputContractError, match="duplicate Google function call"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {"functionCall": {"id": "same", "name": "f", "args": {}}},
                            {"functionCall": {"id": "same", "name": "g", "args": {}}},
                        ]
                    ),
                ),
                _request(
                    tools=ToolParams(
                        functions=(
                            {"name": "f", "description": "Function f"},
                            {"name": "g", "description": "Function g"},
                        )
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_provider_cannot_reuse_function_call_id_across_rounds(self) -> None:
        tools = ToolParams(functions=({"name": "f", "description": "Function f"},))
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "f",
                                "args": {},
                            },
                            "thoughtSignature": "signature_1",
                        }
                    ]
                ),
            ),
            _request(tools=tools),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)

        with pytest.raises(OutputContractError, match="duplicate Google function call"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "functionCall": {
                                    "id": "call_1",
                                    "name": "f",
                                    "args": {"again": True},
                                },
                                "thoughtSignature": "signature_2",
                            }
                        ]
                    ),
                ),
                _request(
                    input=_chat(
                        ChatMessage(
                            "tool",
                            (ToolResultPart("call_1", {"ok": True}),),
                        )
                    ),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    ),
                    tools=tools,
                ),
            )

    @pytest.mark.anyio
    async def test_provider_cannot_reuse_server_tool_id_across_rounds(self) -> None:
        tools = ToolParams(hosted=(HostedToolSpec("google_search"),))
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "toolCall": {
                                "id": "server_1",
                                "toolType": "GOOGLE_SEARCH_WEB",
                                "args": {"q": "first"},
                            },
                            "thoughtSignature": "signature_1",
                        }
                    ]
                ),
            ),
            _request(tools=tools),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)

        with pytest.raises(OutputContractError, match="duplicate Google server tool"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "toolCall": {
                                    "id": "server_1",
                                    "toolType": "GOOGLE_SEARCH_WEB",
                                    "args": {"q": "second"},
                                },
                                "thoughtSignature": "signature_2",
                            }
                        ]
                    ),
                ),
                _request(
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    ),
                    tools=tools,
                ),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (_response(finish_reason="FINISH_REASON_UNSPECIFIED"), "not terminal"),
            (_response(candidate_extra={"index": 1}), "not ordered"),
            (
                _response(
                    parts=[{"executableCode": {"language": "PYTHON", "code": "1"}}]
                ),
                "unsupported field",
            ),
            (
                _response(
                    parts=[{"functionCall": {"id": "c", "name": "f", "args": {}}}]
                ),
                "unrequested function",
            ),
        ],
    )
    async def test_unsupported_response_shapes_fail_loudly(
        self, payload: dict[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=payload))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tools", "returned_name"),
        [
            (
                ToolParams(
                    functions=({"name": "allowed", "description": "Allowed function"},)
                ),
                "other",
            ),
            (
                ToolParams(
                    functions=({"name": "allowed", "description": "Allowed function"},),
                    choice="none",
                ),
                "allowed",
            ),
            (
                ToolParams(
                    functions=(
                        {"name": "allowed", "description": "Allowed function"},
                        {"name": "other", "description": "Other function"},
                    ),
                    choice={"type": "function", "name": "allowed"},
                ),
                "other",
            ),
        ],
    )
    async def test_response_function_call_must_be_allowed_by_the_request(
        self, tools: ToolParams, returned_name: str
    ) -> None:
        with pytest.raises(OutputContractError, match="allowed function set"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "functionCall": {
                                    "id": "call_1",
                                    "name": returned_name,
                                    "args": {},
                                }
                            }
                        ]
                    ),
                ),
                _request(tools=tools),
            )

    @pytest.mark.anyio
    async def test_selected_function_call_is_accepted(self) -> None:
        tools = ToolParams(
            functions=(
                {"name": "allowed", "description": "Allowed function"},
                {"name": "other", "description": "Other function"},
            ),
            choice={"type": "function", "name": "allowed"},
        )

        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "allowed",
                                "args": {},
                            }
                        }
                    ]
                ),
            ),
            _request(tools=tools),
        )

        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "allowed"

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tool", "tool_type"),
        [
            (HostedToolSpec("google_search"), "FILE_SEARCH"),
            (HostedToolSpec("google_search"), "GOOGLE_SEARCH_IMAGE"),
            (HostedToolSpec("url_context"), "GOOGLE_SEARCH_WEB"),
        ],
    )
    async def test_response_server_tool_type_must_match_declared_hosted_tools(
        self, tool: HostedToolSpec, tool_type: str
    ) -> None:
        with pytest.raises(OutputContractError, match="allowed hosted-tool set"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "toolCall": {
                                    "id": "server_1",
                                    "toolType": tool_type,
                                    "args": {},
                                }
                            }
                        ]
                    ),
                ),
                _request(tools=ToolParams(hosted=(tool,))),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tool", "tool_type"),
        [
            (
                HostedToolSpec(
                    "google_search",
                    {"searchTypes": {"imageSearch": {}}},
                ),
                "GOOGLE_SEARCH_IMAGE",
            ),
            (HostedToolSpec("url_context"), "URL_CONTEXT"),
        ],
    )
    async def test_declared_hosted_tool_response_types_are_accepted(
        self, tool: HostedToolSpec, tool_type: str
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "toolCall": {
                                "id": "server_1",
                                "toolType": tool_type,
                                "args": {},
                            }
                        }
                    ]
                ),
            ),
            _request(tools=ToolParams(hosted=(tool,))),
        )

        assert result.server_tool_uses is not None
        assert result.server_tool_uses[0].tool_type == tool_type

    @pytest.mark.anyio
    async def test_missing_requested_logprobs_is_an_output_contract_error(self) -> None:
        with pytest.raises(OutputContractError, match="omitted requested logprobs"):
            await _run(
                lambda _: httpx.Response(200, json=_response()),
                _request(scoring=ScoringParams(sampled_logprobs=True)),
            )


class TestStreamingAndAuditEvidence:
    @pytest.mark.anyio
    async def test_streaming_uses_httpx_stream_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = _connection(
            lambda _: httpx.Response(200, content=_sse(_response()))
        )
        original_send = connection.client.send
        stream_flags: list[object] = []

        async def recording_send(
            request: httpx.Request, *, stream: bool = False
        ) -> httpx.Response:
            stream_flags.append(stream)
            return await original_send(request, stream=stream)

        monkeypatch.setattr(connection.client, "send", recording_send)
        try:
            await GoogleGenAIDialect(connection, "gemini-2.5-flash").arun(
                _request(scheduling=SchedulingParams(stream=True))
            )
        finally:
            await connection.aclose()

        assert stream_flags == [True]

    @pytest.mark.anyio
    async def test_streaming_parse_failure_closes_response(self) -> None:
        class TrackingStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield b"data: not-json\n\n"

            async def aclose(self) -> None:
                self.closed = True

        stream = TrackingStream()

        with pytest.raises(OutputContractError, match="strict JSON"):
            await _run(
                lambda _: httpx.Response(200, stream=stream),
                _request(scheduling=SchedulingParams(stream=True)),
            )

        assert stream.closed is True

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_http_error_status_fails_before_response_lifting(
        self, stream: bool
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            payload = _response()
            if stream:
                return httpx.Response(
                    400,
                    content=_sse(payload),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(400, json=payload)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _run(
                handler,
                _request(scheduling=SchedulingParams(stream=stream)),
            )

        assert "candidates" in exc_info.value.response.text

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_google_error_message_is_kept_on_the_status_error(
        self, stream: bool
    ) -> None:
        # Per-model limits are Google's to enforce, so its reason is the
        # diagnostic a failed sample records.
        reason = "Thinking level is unsupported: THINKING_LEVEL_MINIMAL"

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"code": 400, "message": reason}},
            )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _run(
                handler,
                _request(
                    reasoning=ReasoningParams(effort="minimal"),
                    scheduling=SchedulingParams(stream=stream),
                ),
                model="gemini-3.8-flash",
            )

        message = str(exc_info.value)
        assert "400 Bad Request" in message
        assert f"Google: {reason}" in message
        assert exc_info.value.response.status_code == 400

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (b"upstream unavailable", "Google: upstream unavailable"),
            (b'{"error": "plain"}', 'Google: {"error": "plain"}'),
            (("x" * 600).encode(), "Google: " + "x" * 500 + "..."),
        ],
    )
    async def test_non_standard_error_bodies_are_kept_verbatim(
        self, content: bytes, expected: str
    ) -> None:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _run(lambda _: httpx.Response(502, content=content), _request())

        assert str(exc_info.value).endswith(expected)

    @pytest.mark.anyio
    async def test_empty_error_body_keeps_the_httpx_message(self) -> None:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _run(lambda _: httpx.Response(503), _request())

        assert "Google:" not in str(exc_info.value)
        assert "503" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_rejects_absolute_operation_before_credential_leak(
        self,
    ) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request()
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            audit.finish(prepared)
            attacker_operation = (
                "https://attacker.invalid/v1beta/models/"
                "gemini-2.5-flash:generateContent"
            )

            with pytest.raises(DialectError, match="wrong prepared operation"):
                await dialect.execute(
                    PreparedRequest(
                        attacker_operation,
                        prepared.body,
                        prepared.context,
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_streaming_uses_stream_operation_and_merges_deltas(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Hel"}],
                                },
                            }
                        ],
                        "modelVersion": "gemini-2.5-flash",
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {"role": "model", "parts": [{"text": "lo"}]},
                                "finishReason": "MAX_TOKENS",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 2,
                            "candidatesTokenCount": 2,
                            "totalTokenCount": 4,
                        },
                        "modelVersion": "gemini-2.5-flash",
                    },
                ),
                headers={"content-type": "text/event-stream"},
            )

        result = await _run(
            handler,
            _request(scheduling=SchedulingParams(stream=True)),
        )
        assert seen["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:streamGenerateContent?alt=sse"
        )
        assert result.texts == ("Hello",)
        assert result.finish_reasons == ("MAX_TOKENS",)
        assert result.usage is not None
        assert result.usage.total_tokens == 4
        assert result.usage.output_tokens == 2
        assert result.request_params is not None
        assert result.request_params["query"] == {"alt": "sse"}

    @pytest.mark.anyio
    async def test_streaming_rejects_candidate_events_after_finish_reason(self) -> None:
        with pytest.raises(
            OutputContractError, match="candidate 0 after its terminal finishReason"
        ):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=_sse(
                        {
                            "candidates": [
                                {
                                    "index": 0,
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": "done"}],
                                    },
                                    "finishReason": "STOP",
                                }
                            ]
                        },
                        {
                            "candidates": [
                                {
                                    "index": 0,
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": " extra"}],
                                    },
                                }
                            ]
                        },
                    ),
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_streaming_allows_usage_only_event_after_candidate_finishes(
        self,
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "done"}],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                    {
                        "usageMetadata": {
                            "promptTokenCount": 2,
                            "candidatesTokenCount": 1,
                            "totalTokenCount": 3,
                        }
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.texts == ("done",)
        assert result.finish_reasons == ("STOP",)
        assert result.usage is not None
        assert result.usage.total_tokens == 3

    @pytest.mark.anyio
    async def test_streaming_merges_reasoning_deltas_without_crossing_boundaries(
        self,
    ) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "ana", "thought": True}],
                                },
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {
                                            "text": "lysis",
                                            "thought": True,
                                            "thoughtSignature": "signature",
                                        },
                                        {"text": "answer"},
                                    ],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(
                reasoning=ReasoningParams(summary="auto"),
                scheduling=SchedulingParams(stream=True),
            ),
        )

        assert result.texts == ("answer",)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == "analysis"
        opaque = json.loads(cast(str, result.reasoning[0].opaque_roundtrip))
        assert opaque["contents"][-1]["parts"] == [
            {
                "text": "analysis",
                "thought": True,
                "thoughtSignature": "signature",
            },
            {"text": "answer"},
        ]

    @pytest.mark.parametrize(
        ("previous", "current"),
        [
            (
                {"text": "reasoning", "thought": True},
                {"text": "answer"},
            ),
            (
                {"text": "before"},
                {"functionCall": {"id": "call_1", "name": "f", "args": {}}},
            ),
            (
                {
                    "text": "signed",
                    "thought": True,
                    "thoughtSignature": "signature",
                },
                {"text": "new thought", "thought": True},
            ),
        ],
    )
    def test_streaming_text_merge_respects_semantic_boundaries(
        self, previous: object, current: object
    ) -> None:
        assert google_genai._merge_stream_text_delta(previous, current) is None

    @pytest.mark.anyio
    async def test_streaming_blocked_prompt_surfaces_consistent_reason(self) -> None:
        feedback = {"blockReason": "SAFETY", "blockReasonMessage": "unsafe"}

        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {"promptFeedback": feedback},
                    {
                        "promptFeedback": {
                            "blockReason": "SAFETY",
                            "blockReasonMessage": "unsafe",
                        }
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("SAFETY",)
        assert result.finish_reason_sources == ("promptFeedback.blockReason",)

    @pytest.mark.anyio
    async def test_streaming_rejects_conflicting_prompt_feedback(self) -> None:
        with pytest.raises(OutputContractError, match="changed promptFeedback"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=_sse(
                        {"promptFeedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"}},
                        {"promptFeedback": {"blockReason": "SAFETY"}},
                    ),
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_streaming_does_not_persist_nonterminal_usage(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Hel"}],
                                },
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 2,
                            "candidatesTokenCount": 1,
                            "totalTokenCount": 3,
                        },
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "lo"}],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.texts == ("Hello",)
        assert result.usage is None

    @pytest.mark.anyio
    async def test_streaming_preserves_url_context_citations(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "answer"}],
                                },
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "finishReason": "STOP",
                                "urlContextMetadata": {
                                    "urlMetadata": [
                                        {"retrievedUrl": "https://example.test/source"}
                                    ]
                                },
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(
                scheduling=SchedulingParams(stream=True),
                tools=ToolParams(hosted=(HostedToolSpec("url_context"),)),
            ),
        )

        assert result.citations is not None
        assert [item.url for item in result.citations] == [
            "https://example.test/source"
        ]

    @pytest.mark.anyio
    async def test_streaming_accumulates_grounding_chunks_in_event_order(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "grounded "}],
                                },
                                "groundingMetadata": {
                                    "groundingChunks": [
                                        {
                                            "web": {
                                                "uri": "https://first.test",
                                                "title": "First",
                                            }
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "answer"}],
                                },
                                "finishReason": "STOP",
                                "groundingMetadata": {
                                    "groundingChunks": [
                                        {
                                            "retrievedContext": {
                                                "uri": "https://second.test",
                                                "title": "Second",
                                            }
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(
                scheduling=SchedulingParams(stream=True),
                tools=ToolParams(hosted=(HostedToolSpec("google_search"),)),
            ),
        )

        assert result.grounding is not None
        assert [chunk.uri for chunk in result.grounding.chunks] == [
            "https://first.test",
            "https://second.test",
        ]

    def test_streaming_grounding_metadata_preserves_all_incremental_evidence(
        self,
    ) -> None:
        search_entry_point = {"renderedContent": "<div>sources</div>"}
        retrieval_metadata = {"googleSearchDynamicRetrievalScore": 0.75}
        merged = google_genai._merge_stream_grounding_metadata(
            None,
            {
                "groundingChunks": [{"web": {"uri": "https://first.test"}}],
                "groundingSupports": [{"groundingChunkIndices": [0]}],
                "searchEntryPoint": search_entry_point,
                "retrievalMetadata": retrieval_metadata,
                "googleMapsWidgetContextToken": "maps-token",
            },
            "groundingMetadata",
        )
        merged = google_genai._merge_stream_grounding_metadata(
            merged,
            {
                "groundingChunks": [{"web": {"uri": "https://second.test"}}],
                "groundingSupports": [{"groundingChunkIndices": [1]}],
                "searchEntryPoint": {"renderedContent": "<div>sources</div>"},
                "retrievalMetadata": {"googleSearchDynamicRetrievalScore": 0.75},
                "googleMapsWidgetContextToken": "maps-token",
            },
            "groundingMetadata",
        )

        assert merged == {
            "groundingChunks": [
                {"web": {"uri": "https://first.test"}},
                {"web": {"uri": "https://second.test"}},
            ],
            "groundingSupports": [
                {"groundingChunkIndices": [0]},
                {"groundingChunkIndices": [1]},
            ],
            "searchEntryPoint": search_entry_point,
            "retrievalMetadata": retrieval_metadata,
            "googleMapsWidgetContextToken": "maps-token",
        }

    @pytest.mark.anyio
    async def test_streaming_accepts_identical_repeated_singular_metadata(
        self,
    ) -> None:
        citation_metadata = {"citationSources": [{"uri": "https://citation.test"}]}
        url_context_metadata = {
            "urlMetadata": [{"retrievedUrl": "https://context.test"}]
        }
        first_candidate_metadata = {
            "citationMetadata": citation_metadata,
            "urlContextMetadata": url_context_metadata,
        }
        second_candidate_metadata = {
            "citationMetadata": {"citationSources": [{"uri": "https://citation.test"}]},
            "urlContextMetadata": {
                "urlMetadata": [{"retrievedUrl": "https://context.test"}]
            },
        }
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "answer"}],
                                },
                                **first_candidate_metadata,
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "finishReason": "STOP",
                                **second_candidate_metadata,
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.citations is not None
        assert [item.url for item in result.citations] == [
            "https://citation.test",
            "https://context.test",
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("metadata_name", "first_value", "second_value"),
        [
            (
                "citationMetadata",
                {"citationSources": [{"uri": "https://first.test"}]},
                {"citationSources": [{"uri": "https://second.test"}]},
            ),
            (
                "urlContextMetadata",
                {"urlMetadata": [{"retrievedUrl": "https://first.test"}]},
                {"urlMetadata": [{"retrievedUrl": "https://second.test"}]},
            ),
            (
                "groundingMetadata",
                {"searchEntryPoint": {"renderedContent": "first"}},
                {"searchEntryPoint": {"renderedContent": "second"}},
            ),
        ],
    )
    async def test_streaming_rejects_conflicting_singular_metadata(
        self,
        metadata_name: str,
        first_value: Mapping[str, object],
        second_value: Mapping[str, object],
    ) -> None:
        with pytest.raises(OutputContractError, match="changed .* between events"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=_sse(
                        {
                            "candidates": [
                                {
                                    "index": 0,
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": "answer"}],
                                    },
                                    metadata_name: first_value,
                                }
                            ]
                        },
                        {
                            "candidates": [
                                {
                                    "index": 0,
                                    "finishReason": "STOP",
                                    metadata_name: second_value,
                                }
                            ]
                        },
                    ),
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize("invalid_event", ["junk", [], {}])
    async def test_streaming_rejects_non_response_data_events(
        self, invalid_event: object
    ) -> None:
        terminal = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "answer"}]},
                    "finishReason": "STOP",
                }
            ]
        }
        content = (
            f"data: {json.dumps(invalid_event)}\n\ndata: {json.dumps(terminal)}\n\n"
        ).encode()

        with pytest.raises(OutputContractError, match="non-empty response object"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=content,
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_streaming_logprobs_are_accumulated(self) -> None:
        def logprob(token: str, token_id: int) -> dict[str, object]:
            return {"token": token, "logProbability": -0.1, "tokenId": token_id}

        def handler(_: httpx.Request) -> httpx.Response:
            chunks: list[dict[str, object]] = []
            for index, token in enumerate(("A", "B")):
                candidate: dict[str, object] = {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": token}]},
                    "logprobsResult": {
                        "chosenCandidates": [logprob(token, index)],
                        "topCandidates": [{"candidates": [logprob(token, index)]}],
                    },
                }
                if index == 1:
                    candidate["finishReason"] = "STOP"
                chunks.append({"candidates": [candidate]})
            return httpx.Response(200, content=_sse(*chunks))

        result = await _run(
            handler,
            _request(
                scheduling=SchedulingParams(stream=True),
                scoring=ScoringParams(sampled_logprobs=True, top_logprobs=1),
            ),
        )
        assert result.logprobs is not None
        assert [item.token for item in result.logprobs] == ["A", "B"]

    def test_input_verifier_detects_lowering_mutation(self) -> None:
        connection = _connection(_no_io)
        dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
        req = _request(
            input=_chat(ChatMessage("user", (TextPart("before"), TextPart("after"))))
        )
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
        prepared = dialect.prepare(req, audit)
        body = prepared.thaw_body()
        body["contents"] = [{"role": "user", "parts": [{"text": "afterbefore"}]}]

        with pytest.raises(RequestAuditError, match="contents"):
            audit.finish(PreparedRequest(prepared.operation, body, prepared.context))

    def test_operation_verifier_detects_stream_routing_mutation(self) -> None:
        connection = _connection(_no_io)
        dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
        req = _request(scheduling=SchedulingParams(stream=True))
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
        prepared = dialect.prepare(req, audit)
        wrong = PreparedRequest(
            prepared.operation.replace("streamGenerateContent", "generateContent"),
            prepared.body,
            prepared.context,
        )

        with pytest.raises(RequestAuditError, match="operation"):
            audit.finish(wrong)

    @pytest.mark.anyio
    async def test_malformed_sse_fails_loudly(self) -> None:
        with pytest.raises(OutputContractError, match="malformed SSE"):
            await _run(
                lambda _: httpx.Response(200, content=b"not-sse\n\n"),
                _request(scheduling=SchedulingParams(stream=True)),
            )


class TestAdditionalRequestContracts:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "path"),
        [
            (_request(sampling=SamplingParams(seed=2**31)), "sampling.seed"),
            (_request(sampling=SamplingParams(seed=-(2**31) - 1)), "sampling.seed"),
            (_request(sampling=SamplingParams(top_k=2**31)), "sampling.top_k"),
            (
                _request(sampling=SamplingParams(max_tokens=2**31)),
                "sampling.max_tokens",
            ),
            (_request(sampling=SamplingParams(n=2**31)), "sampling.n"),
            (
                _request(reasoning=ReasoningParams(budget_tokens=2**31)),
                "reasoning.budget_tokens",
            ),
        ],
    )
    async def test_google_int32_overflow_is_rejected_before_io(
        self, req: Request, path: str
    ) -> None:
        with pytest.raises(RequestAuditError, match=re.escape(path)):
            await _run(_no_io, req)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "req",
        [
            _request(sampling=SamplingParams(seed=-(2**31))),
            _request(sampling=SamplingParams(seed=2**31 - 1)),
            _request(sampling=SamplingParams(top_k=2**31 - 1)),
            _request(sampling=SamplingParams(max_tokens=2**31 - 1)),
            _request(sampling=SamplingParams(n=2**31 - 1)),
        ],
    )
    async def test_google_int32_boundaries_pass_preflight(self, req: Request) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            audit = RequestAudit(active_request_leaves(req))

            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))

            audit.raise_rejections()
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "message"),
        [
            (
                _request(
                    input=_chat(ChatMessage(cast(Any, "alien"), (TextPart("x"),)))
                ),
                "invalid role",
            ),
            (
                _request(input=_chat(ChatMessage("user", (TextPart(cast(Any, 1)),)))),
                "text must be a string",
            ),
            (
                _request(
                    input=_chat(ChatMessage("user", (ImagePart(url=cast(Any, 1)),)))
                ),
                "url must be a string",
            ),
            (
                _request(
                    input=_chat(ChatMessage("user", (TextPart("x"),), name="named"))
                ),
                "no ChatMessage.name",
            ),
            (
                _request(input=_chat(ChatMessage("user", ()))),
                "empty message",
            ),
            (
                _request(input=_chat(ChatMessage("tool", (TextPart("x"),)))),
                "only tool results",
            ),
            (
                _request(input=_chat(ChatMessage("tool", (ImagePart(url="x"),)))),
                "instruction/tool messages are text-only",
            ),
            (
                _request(
                    input=_chat(ChatMessage("user", (ToolCallPart("call", "f", {}),)))
                ),
                "require assistant role",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("assistant", (ToolCallPart("", "f", {}),)),
                        ChatMessage("user", (TextPart("continue"),)),
                    ),
                ),
                "call_id must be a non-empty string",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage(
                            "assistant",
                            (ToolCallPart("call", "f", "not-an-object"),),
                        ),
                        ChatMessage("user", (TextPart("continue"),)),
                    )
                ),
                "arguments must be an object",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage(
                            "assistant",
                            (
                                ToolCallPart("call", "f", {}),
                                ToolCallPart("call", "f", {}),
                            ),
                        ),
                        ChatMessage("user", (TextPart("continue"),)),
                    )
                ),
                "duplicate Google function call id",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("assistant", (ToolCallPart("call", "f", {}),)),
                        ChatMessage("user", (ToolResultPart("call", {}),)),
                    )
                ),
                "require tool role",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("assistant", (ToolCallPart("call", "f", {}),)),
                        ChatMessage("tool", (ToolResultPart("call", {}),)),
                        ChatMessage("tool", (ToolResultPart("call", {}),)),
                    )
                ),
                "duplicate Google function result",
            ),
            (
                _request(input=_chat(ChatMessage("user", (ImagePart(url=""),)))),
                "URLs must not be empty",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage(
                            "user", (ImagePart(data="", media_type="image/png"),)
                        )
                    )
                ),
                "inline images require data",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"type": "function", "function": "wrong"},)
                    )
                ),
                "invalid OpenAI-shaped function declaration",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "f", "parameters": {}, "future": True},)
                    )
                ),
                "unsupported Google function declaration field",
            ),
            (
                _request(tools=ToolParams(functions=({"parameters": {}},))),
                "requires a name",
            ),
            (
                _request(tools=ToolParams(functions=({"name": "f"},))),
                "requires a non-empty description",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "bad name!", "description": "Bad"},)
                    )
                ),
                "at most 128 characters",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "f" * 129, "description": "Too long"},)
                    )
                ),
                "at most 128 characters",
            ),
            (
                _request(
                    tools=ToolParams(functions=({"name": "f", "description": 1},))
                ),
                "description must be a string",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=(
                            {
                                "name": "f",
                                "description": "Function f",
                                "parameters": [],
                            },
                        )
                    )
                ),
                "parameters must be an object",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "f", "description": "Function f"},),
                        choice="sometimes",
                    )
                ),
                "unsupported Google tool choice",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "f", "description": "Function f"},),
                        choice=1,
                    )
                ),
                "string or mapping",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "f", "description": "Function f"},),
                        choice={"type": "function", "name": ""},
                    )
                ),
                "requires a name",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "f", "description": "Function f"},),
                        choice={"type": "function", "future": True},
                    )
                ),
                "invalid Google function-specific tool choice",
            ),
            (
                _request(tools=ToolParams(parallel=True)),
                "parallel tool switch",
            ),
            (
                _request(tools=ToolParams(hosted=(HostedToolSpec("code_execution"),))),
                "unsupported Google hosted tool",
            ),
            (
                _request(scoring=ScoringParams(sampled_logprobs=True, top_logprobs=21)),
                "at most 20",
            ),
            (
                _request(reasoning=ReasoningParams(effort="xhigh")),
                "unsupported Google thinking level",
            ),
            (
                _request(reasoning=ReasoningParams(summary="detailed")),
                "supports only summary='none' or 'auto'",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={}, name="named"
                    )
                ),
                "schemas have no name",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={}, strict=True
                    )
                ),
                "no JSON-schema strict switch",
            ),
            (
                _request(dialect_options=DialectOptions("google_genai", {"": True})),
                "keys must be non-empty",
            ),
            (
                _request(
                    dialect_options=DialectOptions("google_genai", cast(Any, {1: True}))
                ),
                "option keys must be strings",
            ),
        ],
    )
    async def test_invalid_request_is_rejected_before_io(
        self, req: Request, message: str
    ) -> None:
        with pytest.raises((DialectError, RequestAuditError), match=message):
            await _run(_no_io, req)

    @pytest.mark.parametrize(
        ("req", "message"),
        [
            (_request(input=CompletionInput("prompt")), "requires ChatInput"),
            (
                _request(scoring=ScoringParams(input_scoring=True)),
                "cannot score input tokens",
            ),
            (
                _request(session=SessionParams(previous_response_id="response")),
                "not a continuation handle",
            ),
            (
                _request(
                    structured_output=cast(
                        Any,
                        StructuredOutputParams(
                            format="json_object", schema={"type": "object"}
                        ),
                    )
                ),
                "requires format='json_schema'",
            ),
        ],
    )
    def test_low_level_validation_rejects_unsupported_or_conflicting_leaves(
        self, req: Request, message: str
    ) -> None:
        connection = _connection(_no_io)
        dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
        audit = RequestAudit(active_request_leaves(req))

        dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))

        with pytest.raises(RequestAuditError, match=message):
            audit.raise_rejections()


class TestOpaqueContinuationBoundaries:
    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "version": 1,
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": "thought",
                            "thought": True,
                            "thoughtSignature": "signature",
                        }
                    ],
                }
            ],
        }

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda value: value.update(version=2), "unsupported.*version"),
            (lambda value: value.update(contents=[]), "has no contents"),
            (lambda value: value.update(contents=["wrong"]), "invalid shape"),
            (
                lambda value: value.update(contents=[{"role": "tool", "parts": [{}]}]),
                "invalid role",
            ),
            (
                lambda value: value.update(contents=[{"role": "user", "parts": []}]),
                "has no parts",
            ),
            (
                lambda value: value.update(
                    contents=[{"role": "user", "parts": ["wrong"]}]
                ),
                "is not an object",
            ),
            (
                lambda value: value.update(
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": "x", "future": True}],
                        }
                    ]
                ),
                "unsupported field",
            ),
            (
                lambda value: value.update(
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": "x", "fileData": {}}],
                        }
                    ]
                ),
                "one data field",
            ),
            (
                lambda value: value.update(
                    contents=[{"role": "user", "parts": [{"text": 1}]}]
                ),
                "text part must contain a string",
            ),
            (
                lambda value: value.update(
                    contents=[{"role": "user", "parts": [{"fileData": "x"}]}]
                ),
                "fileData must be an object",
            ),
            (
                lambda value: value.update(
                    contents=[
                        {"role": "model", "parts": [{"text": "x", "thought": "yes"}]}
                    ]
                ),
                "thought marker must be boolean",
            ),
            (
                lambda value: value.update(
                    contents=[
                        {
                            "role": "model",
                            "parts": [{"text": "x", "thoughtSignature": ""}],
                        }
                    ]
                ),
                "thoughtSignature must be a non-empty string",
            ),
            (
                lambda value: value.update(
                    contents=[{"role": "model", "parts": [{"text": "x"}]}]
                ),
                "has no thought signature",
            ),
            (
                lambda value: value.update(systemInstruction={}),
                "systemInstruction has invalid shape",
            ),
            (
                lambda value: value.update(systemInstruction={"parts": []}),
                "systemInstruction has no parts",
            ),
            (
                lambda value: value.update(systemInstruction={"parts": [{"text": 1}]}),
                "systemInstruction must be text-only",
            ),
            (lambda value: value.update(future=True), "invalid.*envelope"),
        ],
    )
    def test_malformed_opaque_continuation_is_rejected(
        self, mutate: Callable[[dict[str, object]], None], message: str
    ) -> None:
        payload = self._payload()
        mutate(payload)

        with pytest.raises(DialectError, match=message):
            google_genai._decode_opaque(json.dumps(payload))

    def test_synthetic_call_metadata_preserves_an_idless_provider_part(self) -> None:
        payload = {
            "version": 1,
            "contents": [
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "f", "args": {}}}],
                }
            ],
            "syntheticCallIds": [
                {
                    "contentIndex": 0,
                    "partIndex": 0,
                    "field": "functionCall",
                    "id": "google_call_local",
                }
            ],
        }

        _, contents, synthetic = google_genai._decode_opaque(json.dumps(payload))

        assert contents[0]["parts"] == [{"functionCall": {"name": "f", "args": {}}}]
        assert synthetic == {(0, 0, "functionCall"): "google_call_local"}

    @pytest.mark.parametrize(
        ("contents", "message"),
        [
            (
                [
                    {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "f", "description": "Function f"}}
                        ],
                    }
                ],
                "functionCall without a provider id has no synthetic",
            ),
            (
                [{"role": "model", "parts": [{"functionCall": {"id": "c"}}]}],
                "functionCall requires a non-empty name",
            ),
            (
                [
                    {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"id": "c", "name": "f"}},
                            {"functionCall": {"id": "c", "name": "f"}},
                        ],
                    }
                ],
                "duplicate Google function call id",
            ),
            (
                [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "f",
                                    "description": "Function f",
                                }
                            }
                        ],
                    }
                ],
                "functionResponse without a provider id has no synthetic",
            ),
            (
                [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": "missing",
                                    "name": "f",
                                }
                            }
                        ],
                    }
                ],
                "has no prior functionCall",
            ),
            (
                [
                    {
                        "role": "model",
                        "parts": [{"functionCall": {"id": "c", "name": "f"}}],
                    },
                    {
                        "role": "user",
                        "parts": [{"functionResponse": {"id": "c", "name": "g"}}],
                    },
                ],
                "name changed",
            ),
            (
                [{"role": "model", "parts": [{"toolCall": {"toolType": "SEARCH"}}]}],
                "toolCall without a provider id has no synthetic",
            ),
            (
                [{"role": "model", "parts": [{"toolCall": {"id": "s"}}]}],
                "toolCall requires a non-empty toolType",
            ),
            (
                [{"role": "user", "parts": [{"toolResponse": {"toolType": "SEARCH"}}]}],
                "toolResponse without a provider id has no synthetic",
            ),
            (
                [{"role": "user", "parts": [{"toolResponse": {"id": "s"}}]}],
                "toolResponse requires a non-empty toolType",
            ),
        ],
    )
    def test_opaque_call_state_rejects_ambiguous_history(
        self, contents: list[Mapping[str, object]], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            google_genai._call_state_from_contents(cast(Any, contents), {})


class TestAdditionalResponseContracts:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({"candidates": {}}, "response.candidates must be a sequence"),
            ({"candidates": []}, "expected 1 Google candidates"),
            (
                _response(candidate_extra={"content": {"role": "user", "parts": []}}),
                "content role is not model",
            ),
            (
                _response(candidate_extra={"content": "wrong"}),
                "content must be an object",
            ),
            (
                _response(candidate_extra={"content": {"role": "model", "parts": [1]}}),
                "response part must be an object",
            ),
            (
                _response(parts=[{"text": "x", "future": True}]),
                "unsupported field",
            ),
            (
                _response(parts=[{"text": "x", "thought": "yes"}]),
                "thought marker must be boolean",
            ),
            (
                _response(parts=[{"text": "x", "thoughtSignature": ""}]),
                "thoughtSignature must be a non-empty string",
            ),
            (
                _response(parts=[{"text": "x", "functionCall": {}}]),
                "unsupported shape",
            ),
            (
                _response(parts=[{"text": 1}]),
                "response text must be a string",
            ),
            (
                _response(
                    parts=[{"functionCall": "wrong"}],
                    candidate_extra=None,
                ),
                "functionCall must be an object",
            ),
            (
                _response(parts=[{"functionCall": {"name": "f", "args": []}}]),
                "functionCall.args must be an object",
            ),
            (
                _response(parts=[{"functionCall": {"name": "", "args": {}}}]),
                "functionCall.name must be a non-empty string",
            ),
            (
                _response(parts=[{"functionCall": {"id": 1, "name": "f", "args": {}}}]),
                "functionCall.id must be a string",
            ),
            (
                _response(parts=[{"toolCall": "wrong"}]),
                "toolCall must be an object",
            ),
            (
                _response(
                    parts=[{"toolCall": {"id": "s", "toolType": "SEARCH", "args": []}}]
                ),
                "toolCall.args must be an object",
            ),
            (
                _response(
                    parts=[{"toolCall": {"id": 1, "toolType": "SEARCH", "args": {}}}]
                ),
                "toolCall.id must be a string",
            ),
            (
                _response(parts=[{"toolResponse": "wrong"}]),
                "toolResponse must be an object",
            ),
            (
                _response(
                    parts=[
                        {
                            "toolResponse": {
                                "id": 1,
                                "toolType": "SEARCH",
                                "response": {},
                            }
                        }
                    ]
                ),
                "toolResponse.id must be a string",
            ),
            (
                _response(
                    parts=[
                        {
                            "toolResponse": {
                                "id": "s",
                                "toolType": "SEARCH",
                                "response": [],
                            }
                        }
                    ]
                ),
                "toolResponse.response must be an object",
            ),
            (
                _response(model_version=1),
                "response.modelVersion must be a string or None",
            ),
            (
                _response(usage=[]),
                "usageMetadata must be an object",
            ),
        ],
    )
    async def test_malformed_provider_response_fails_loudly(
        self, payload: Mapping[str, object], message: str
    ) -> None:
        # Server-tool cases need a hosted tool for the response part to be allowed.
        server_tool_case = message.startswith(("toolCall", "toolResponse"))
        request = _request(
            tools=(
                ToolParams(hosted=(HostedToolSpec("google_search"),))
                if server_tool_case
                else ToolParams(functions=({"name": "f", "description": "Function f"},))
            )
        )
        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=payload), request)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("candidate_extra", "message"),
        [
            ({"citationMetadata": []}, "citationMetadata must be an object"),
            (
                {"citationMetadata": {"citationSources": [{}]}},
                "uri is absent",
            ),
            ({"urlContextMetadata": []}, "urlContextMetadata must be an object"),
            (
                {"urlContextMetadata": {"urlMetadata": [{}]}},
                "omitted its URL",
            ),
            ({"groundingMetadata": []}, "groundingMetadata must be an object"),
            (
                {"groundingMetadata": {"groundingChunks": [{}]}},
                "is ambiguous",
            ),
            (
                {"groundingMetadata": {"groundingChunks": [{"web": {}}]}},
                "has no representable URI",
            ),
        ],
    )
    async def test_malformed_evidence_metadata_fails_loudly(
        self, candidate_extra: Mapping[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                lambda _: httpx.Response(
                    200, json=_response(candidate_extra=candidate_extra)
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("usage", "message"),
        [
            ({"promptTokenCount": -1}, "non-negative integer"),
            (
                {"promptTokenCount": 1, "candidatesTokenCount": True},
                "non-negative integer",
            ),
            (
                {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": -1,
                },
                "non-negative integer",
            ),
            (
                {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "thoughtsTokenCount": -1,
                },
                "non-negative integer",
            ),
            (
                {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "cachedContentTokenCount": -1,
                },
                "non-negative integer",
            ),
        ],
    )
    async def test_malformed_usage_fails_loudly(
        self, usage: Mapping[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=_response(usage=usage)))


class TestAdditionalStreamingContracts:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (b": keep-alive\n\n", "no response chunks"),
            (b"data: not-json\n\n", "not strict JSON"),
            (b"future: x\n\n", "unsupported SSE field"),
            (b"data\n\n", "malformed SSE line"),
            (b"data: []\n\n", "non-empty response object"),
            (b"data: {}\n\n", "non-empty response object"),
        ],
    )
    async def test_stream_parser_rejects_malformed_events(
        self, content: bytes, message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                lambda _: httpx.Response(200, content=content),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("chunks", "message"),
        [
            (
                ({"candidates": [{"index": -1}]},),
                "candidate index is invalid",
            ),
            (
                (
                    {
                        "candidates": [
                            {"index": 0, "content": {"role": "user", "parts": []}}
                        ]
                    },
                ),
                "candidate role is not model",
            ),
            (
                ({"candidates": [{"index": 0, "groundingMetadata": []}]},),
                "groundingMetadata must be an object",
            ),
            (
                ({"candidates": [{"index": 0, "logprobsResult": []}]},),
                "logprobsResult must be an object",
            ),
            (
                (
                    {"candidates": [{"index": 0, "finishReason": "STOP"}]},
                    {"modelVersion": "changed"},
                    {"modelVersion": "again"},
                ),
                "changed modelVersion",
            ),
        ],
    )
    async def test_stream_merge_rejects_invalid_state(
        self, chunks: tuple[Mapping[str, object], ...], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                lambda _: httpx.Response(200, content=_sse(*chunks)),
                _request(scheduling=SchedulingParams(stream=True)),
            )


class TestPreparedRequestAndVerifierContracts:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("extra_tool", "with_functions"),
        [
            ({"urlContext": {}}, False),
            ({"googleSearch": {}}, False),
            ({"urlContext": {}}, True),
        ],
    )
    async def test_request_audit_rejects_extra_hosted_tools(
        self, extra_tool: dict[str, object], with_functions: bool
    ) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-3.8-flash")
            req = _request(
                tools=ToolParams(
                    functions=({"name": "lookup", "description": "Lookup"},)
                    if with_functions
                    else (),
                    hosted=(HostedToolSpec("google_search"),),
                )
            )
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            audit.finish(prepared)
            body = prepared.thaw_body()
            cast(list[Any], body["tools"]).append(extra_tool)

            with pytest.raises(RequestAuditError, match="hosted tools changed"):
                audit.finish(
                    PreparedRequest(prepared.operation, body, prepared.context)
                )
        finally:
            await connection.aclose()

    @pytest.mark.parametrize(
        ("tools", "has_functions", "expected_mismatch"),
        [
            ([{"googleSearch": {}}], False, None),
            ([{"googleSearch": {}}, {"urlContext": {}}], False, "hosted tools changed"),
            (
                [{"googleSearch": {}}, {"googleSearch": {}}],
                False,
                "hosted tools changed",
            ),
            (
                [
                    {"functionDeclarations": [{"name": "unexpected"}]},
                    {"googleSearch": {}},
                ],
                False,
                "hosted tools changed",
            ),
            (
                [{"functionDeclarations": [{"name": "lookup"}]}, {"googleSearch": {}}],
                True,
                None,
            ),
            (
                [
                    {"functionDeclarations": [{"name": "lookup"}]},
                    {"googleSearch": {}},
                    {"urlContext": {}},
                ],
                True,
                "hosted tools changed",
            ),
        ],
    )
    def test_hosted_tools_verifier_requires_exact_tools(
        self,
        tools: list[dict[str, object]],
        has_functions: bool,
        expected_mismatch: str | None,
    ) -> None:
        verifier = google_genai._HostedToolsVerifier(
            (HostedToolSpec("google_search"),), has_functions=has_functions
        )
        body = {
            "tools": tools,
            "toolConfig": {"includeServerSideToolInvocations": True},
        }
        assert verifier.verify(cast(Any, body)) == expected_mismatch

    def test_verification_input_rejects_empty_image_url(self) -> None:
        request_input = _chat(ChatMessage("user", (ImagePart(url=""),)))
        with pytest.raises(DialectError, match="URLs must not be empty"):
            google_genai._verification_input_to_wire(request_input, None)

    def test_all_wire_verifiers_reject_mutated_output(self) -> None:
        assert (
            google_genai._InputVerifier(None, ()).verify(
                {"contents": [], "systemInstruction": {"parts": []}}
            )
            == "systemInstruction no longer matches system/developer messages"
        )
        assert (
            google_genai._RootOptionVerifier("serviceTier", "priority").verify({})
            == "Google dialect option 'serviceTier' changed"
        )
        assert (
            google_genai._ThinkingEffortVerifier("high").verify({})
            == "thinking level changed"
        )
        assert (
            google_genai._ReasoningSummaryVerifier().verify({})
            == "visible-thought request changed"
        )
        structured = google_genai._StructuredFormatVerifier("json_schema")
        assert structured.verify({}) == "structured output MIME type changed"
        assert (
            structured.verify(
                {"generationConfig": {"responseMimeType": "application/json"}}
            )
            == "structured output schema presence changed"
        )
        functions = google_genai._FunctionsVerifier(
            ({"name": "f", "description": "Function f"},),
            GOOGLE_WIRE_CONTRACTS["v1beta"],
        )
        assert functions.verify({}) == "function tools changed"
        hosted = google_genai._HostedToolsVerifier((HostedToolSpec("google_search"),))
        assert hosted.verify({}) == "hosted tools changed"
        assert (
            hosted.verify({"tools": [{"googleSearch": {}}]})
            == "hosted-tool invocation evidence was disabled"
        )
        assert (
            google_genai._ToolChoiceVerifier(cast(Any, 1)).verify({})
            == "source tool choice is invalid"
        )
        assert (
            google_genai._ToolChoiceVerifier("auto").verify({}) == "tool choice changed"
        )
        assert (
            google_genai._ToolChoiceVerifier(
                "auto", include_server_side_invocations=True
            ).verify(
                {
                    "toolConfig": {
                        "functionCallingConfig": {"mode": "VALIDATED"},
                        "includeServerSideToolInvocations": True,
                    }
                }
            )
            is None
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda body: body.update(contents={}), "invalid contents"),
            (lambda body: body.update(contents=["wrong"]), "invalid content items"),
            (
                lambda body: body.update(systemInstruction=[]),
                "invalid systemInstruction",
            ),
        ],
    )
    async def test_execute_rejects_mutated_prepared_body_before_io(
        self, mutation: Callable[[dict[str, object]], None], message: str
    ) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request()
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, object], prepared.thaw_body())
            mutation(body)

            with pytest.raises(DialectError, match=message):
                await dialect.execute(
                    PreparedRequest(
                        prepared.operation, cast(Any, body), prepared.context
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_execute_rejects_invalid_prepared_context_before_io(self) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            with pytest.raises(DialectError, match="invalid prepared context"):
                await dialect.execute(
                    PreparedRequest(
                        "v1beta/models/gemini-2.5-flash:generateContent", {}
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_execute_rejects_unanswered_call_in_mutated_final_body(self) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request()
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, object], prepared.thaw_body())
            body["contents"] = [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "lookup",
                                "args": {},
                            }
                        }
                    ],
                }
            ]

            with pytest.raises(DialectError, match="unanswered.*call_1"):
                await dialect.execute(
                    PreparedRequest(
                        prepared.operation, cast(Any, body), prepared.context
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_execute_rejects_globally_complete_but_delayed_call_results(
        self,
    ) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request()
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, object], prepared.thaw_body())
            body["contents"] = [
                {
                    "role": "model",
                    "parts": [{"functionCall": {"id": "call_a", "name": "first"}}],
                },
                {"role": "user", "parts": [{"text": "intervening"}]},
                {
                    "role": "model",
                    "parts": [{"functionCall": {"id": "call_b", "name": "second"}}],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call_a",
                                "name": "first",
                                "response": {"ok": True},
                            }
                        },
                        {
                            "functionResponse": {
                                "id": "call_b",
                                "name": "second",
                                "response": {"ok": True},
                            }
                        },
                    ],
                },
            ]

            with pytest.raises(DialectError, match="immediately follow.*call_a"):
                await dialect.execute(
                    PreparedRequest(
                        prepared.operation, cast(Any, body), prepared.context
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_execute_rejects_missing_current_turn_thought_signature(
        self,
    ) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-3.5-flash")
            req = _request()
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, object], prepared.thaw_body())
            body["contents"] = [
                {"role": "user", "parts": [{"text": "Call the function"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "f",
                                "args": {},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call_1",
                                "name": "f",
                                "response": {"ok": True},
                            }
                        }
                    ],
                },
            ]

            with pytest.raises(
                DialectError, match="current-turn functionCall.*thoughtSignature"
            ):
                await dialect.execute(
                    PreparedRequest(
                        prepared.operation, cast(Any, body), prepared.context
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_execute_rejects_mutated_hosted_auto_mode(self) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-3.8-flash")
            req = _request(
                tools=ToolParams(
                    functions=({"name": "lookup", "description": "Look up a value"},),
                    choice="auto",
                    hosted=(HostedToolSpec("google_search"),),
                )
            )
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, Any], prepared.thaw_body())
            body["toolConfig"]["functionCallingConfig"]["mode"] = "AUTO"

            with pytest.raises(DialectError, match="does not support.*AUTO"):
                await dialect.execute(
                    PreparedRequest(
                        prepared.operation, cast(Any, body), prepared.context
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_execute_rejects_mutated_hosted_tool_schema(self) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request(tools=ToolParams(hosted=(HostedToolSpec("google_search"),)))
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, Any], prepared.thaw_body())
            body["tools"][0]["googleSearch"] = {"unexpected": True}

            with pytest.raises(DialectError, match="unsupported field.*unexpected"):
                await dialect.execute(
                    PreparedRequest(
                        prepared.operation, cast(Any, body), prepared.context
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_request_params_come_from_final_serialized_request(self) -> None:
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request(sampling=SamplingParams(temperature=0.2))
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            body = cast(dict[str, object], prepared.thaw_body())
            generation = cast(dict[str, object], body["generationConfig"])
            generation["temperature"] = 0.9

            response = await dialect.execute(
                PreparedRequest(prepared.operation, cast(Any, body), prepared.context)
            )
        finally:
            await connection.aclose()

        assert response.request_params is not None
        assert response.request_params["body"] == {
            "generationConfig": {"temperature": 0.9}
        }


def _forced(obj: Any, **fields: object) -> Any:
    """Bypass IR construction checks to reach a dialect's own defensive guards."""

    for name, value in fields.items():
        object.__setattr__(obj, name, value)
    return obj


def _one_message(*parts: object, role: ChatRole = "user") -> ChatInput:
    return _chat(ChatMessage(role, cast(Any, tuple(parts))))


class TestHistoryShapeGuards:
    """Fail-loud guards for input the IR constructors do not themselves reject."""

    @pytest.mark.parametrize(
        ("make", "expected"),
        [
            (lambda: _forced(_chat(), messages=()), "at least one message"),
            (
                lambda: _one_message(
                    _forced(ImagePart(url="u", media_type="image/png"), data="d")
                ),
                "exactly one of url or data",
            ),
            (
                lambda: _one_message(_forced(ImagePart(url="u"), url=5)),
                ".url must be a string",
            ),
            (
                lambda: _one_message(
                    _forced(ImagePart(data="d", media_type="image/png"), data=5)
                ),
                ".data must be a string",
            ),
            (
                lambda: _one_message(_forced(ImagePart(url="u"), media_type=5)),
                ".media_type must be a string",
            ),
            (
                lambda: _one_message(_forced(ImagePart(url="u"), detail=5)),
                ".detail must be a string",
            ),
            (
                lambda: _one_message(ToolCallPart("c", "", {}), role="assistant"),
                ".name must be a non-empty string",
            ),
            (
                lambda: _one_message(ToolResultPart("", {}), role="tool"),
                ".call_id must be a non-empty string",
            ),
            (
                lambda: _one_message(
                    _forced(ToolResultPart("c", {}), is_error="yes"), role="tool"
                ),
                ".is_error must be a boolean",
            ),
            (lambda: _one_message(object()), "has unsupported type"),
        ],
    )
    def test_chat_shape_error_names_each_malformed_field(
        self, make: Callable[[], ChatInput], expected: str
    ) -> None:
        error = google_genai._chat_shape_error(make())

        assert error is not None
        assert expected in error

    @pytest.mark.parametrize("lower", ["main", "verification"])
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (ImagePart(url="u", detail="high"), "per-image detail"),
            (
                ImagePart(url="u", media_type="image/bmp"),
                "unsupported Google image media type",
            ),
            (
                ImagePart(data="", media_type="image/png"),
                "inline images require data",
            ),
            (ImagePart(data="AAAA"), "inline images require data"),
        ],
    )
    def test_image_lowering_and_its_audit_mirror_reject_the_same_inputs(
        self, lower: str, image: ImagePart, expected: str
    ) -> None:
        convert = (
            google_genai._input_to_wire
            if lower == "main"
            else google_genai._verification_input_to_wire
        )

        with pytest.raises(DialectError, match=expected):
            convert(_one_message(TextPart("look"), image), None)

    def test_empty_image_url_is_rejected_by_lowering(self) -> None:
        with pytest.raises(DialectError, match="URLs must not be empty"):
            google_genai._input_to_wire(
                _one_message(ImagePart(url="", media_type="image/png")), None
            )

    @pytest.mark.parametrize(
        ("message", "names", "wire_ids", "answered", "expected"),
        [
            (
                ChatMessage("tool", (TextPart("x"),)),
                {},
                {},
                set(),
                "may contain only tool results",
            ),
            (
                ChatMessage("tool", (ImagePart(url="u"),)),
                {},
                {},
                set(),
                "may contain only tool results",
            ),
            (
                ChatMessage("user", (ToolCallPart("c", "f", {}),)),
                {},
                {},
                set(),
                "require assistant role",
            ),
            (
                ChatMessage("assistant", (ToolCallPart("", "f", {}),)),
                {},
                {},
                set(),
                "require id and name",
            ),
            (
                ChatMessage("assistant", (ToolCallPart("c", "", {}),)),
                {},
                {},
                set(),
                "require id and name",
            ),
            (
                ChatMessage("assistant", (ToolCallPart("c", "f", {}),)),
                {"c": "f"},
                {"c": "c"},
                set(),
                "duplicate Google function call id",
            ),
            (
                ChatMessage("user", (ToolResultPart("c", {}),)),
                {"c": "f"},
                {"c": "c"},
                set(),
                "require tool role",
            ),
            (
                ChatMessage("tool", (ToolResultPart("c", {}),)),
                {},
                {},
                set(),
                "has no prior call",
            ),
            (
                ChatMessage("tool", (ToolResultPart("c", {}),)),
                {"c": "f"},
                {"c": "c"},
                {"c"},
                "duplicate Google function result",
            ),
            (
                ChatMessage("user", (cast(Any, object()),)),
                {},
                {},
                set(),
                "unsupported Google content part",
            ),
            (
                ChatMessage("user", ()),
                {},
                {},
                set(),
                "must not contain an empty message",
            ),
        ],
    )
    def test_content_parts_reject_each_malformed_message(
        self,
        message: ChatMessage,
        names: dict[str, str],
        wire_ids: dict[str, str | None],
        answered: set[str],
        expected: str,
    ) -> None:
        with pytest.raises(DialectError, match=expected):
            google_genai._content_parts(message, names, answered, wire_ids, {}, 1)

    def test_idless_function_response_requires_a_conversation_index(self) -> None:
        with pytest.raises(AssertionError, match="conversation index"):
            google_genai._content_parts(
                ChatMessage("tool", (ToolResultPart("c", {}),)),
                {"c": "f"},
                set(),
                {"c": None},
                {},
                None,
            )


def _opaque_envelope(
    synthetic: object = None,
    *,
    call: object | None = None,
    part_extra: Mapping[str, object] | None = None,
) -> str:
    part: dict[str, object] = {
        "functionCall": {"name": "f", "args": {}} if call is None else call,
        "thoughtSignature": "sig",
    }
    part.update(part_extra or {})
    envelope: dict[str, object] = {
        "version": 1,
        "contents": [
            {"role": "user", "parts": [{"text": "q"}]},
            {"role": "model", "parts": [part]},
        ],
    }
    if synthetic is not None:
        envelope["syntheticCallIds"] = synthetic
    return json.dumps(envelope)


def _synthetic(**changes: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "contentIndex": 1,
        "partIndex": 0,
        "field": "functionCall",
        "id": "local",
    }
    entry.update(changes)
    return entry


class TestOpaqueReplayGuards:
    def test_well_formed_synthetic_correlation_decodes(self) -> None:
        _, contents, synthetic = google_genai._decode_opaque(
            _opaque_envelope([_synthetic()])
        )

        assert len(contents) == 2
        assert synthetic == {(1, 0, "functionCall"): "local"}

    @pytest.mark.parametrize(
        ("synthetic", "expected"),
        [
            ({}, "must be a list"),
            ([{"contentIndex": 1}], "has invalid shape"),
            ([_synthetic(contentIndex=-1)], "non-negative integers"),
            ([_synthetic(partIndex=True)], "non-negative integers"),
            ([_synthetic(field="bogus")], "has invalid field"),
            ([_synthetic(id="")], "must be non-empty"),
            ([_synthetic(contentIndex=9)], "points outside contents"),
            ([_synthetic(field="functionResponse")], "points to the wrong part"),
            ([_synthetic(), _synthetic()], "contains a duplicate"),
        ],
    )
    def test_malformed_synthetic_correlation_is_rejected(
        self, synthetic: object, expected: str
    ) -> None:
        with pytest.raises(DialectError, match=expected):
            google_genai._decode_opaque(_opaque_envelope(synthetic))

    def test_synthetic_id_cannot_shadow_a_provider_issued_id(self) -> None:
        payload = _opaque_envelope(
            [_synthetic()], call={"id": "provider", "name": "f", "args": {}}
        )

        with pytest.raises(DialectError, match="requires a call without id"):
            google_genai._decode_opaque(payload)

    @pytest.mark.parametrize(
        "contents",
        [
            [{"role": "model", "parts": "not-a-list"}],
            [{"role": "model", "parts": [{"functionCall": "not-an-object"}]}],
        ],
    )
    def test_synthetic_validation_does_not_trust_unchecked_contents(
        self, contents: list[dict[str, object]]
    ) -> None:
        with pytest.raises(DialectError):
            google_genai._validate_synthetic_call_ids(
                [_synthetic(contentIndex=0)], cast(Any, contents)
            )

    def test_malformed_part_metadata_is_rejected_on_replay(self) -> None:
        with pytest.raises(DialectError, match="partMetadata must be an object"):
            google_genai._decode_opaque(
                _opaque_envelope(
                    [_synthetic()], part_extra={"partMetadata": "not-an-object"}
                )
            )

    @pytest.mark.parametrize(
        ("contents", "expected"),
        [
            ([{"role": "user", "parts": "x"}], "parts must be a list"),
            ([{"role": "user", "parts": ["x"]}], "must be an object"),
            (
                [{"role": "model", "parts": [{"functionCall": "x"}]}],
                "history must be an object",
            ),
            (
                [
                    {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"id": "a", "name": "f"}},
                            {"functionCall": {"id": "a", "name": "f"}},
                        ],
                    }
                ],
                "duplicates functionCall id",
            ),
        ],
    )
    def test_final_history_recovery_rejects_malformed_contents(
        self, contents: list[dict[str, object]], expected: str
    ) -> None:
        with pytest.raises(DialectError, match=expected):
            google_genai._historical_output_call_ids(cast(Any, contents), {})

    @pytest.mark.parametrize(
        ("output", "calls", "expected"),
        [
            ([], {}, "must remain an object"),
            ({"role": "model"}, {}, "must retain parts"),
            (
                {"role": "model", "parts": [{"text": "x"}]},
                {5: ("functionCall", "c")},
                "part index changed",
            ),
            (
                {"role": "model", "parts": ["x"]},
                {0: ("functionCall", "c")},
                "part changed shape",
            ),
            (
                {"role": "model", "parts": [{"functionCall": "x"}]},
                {0: ("functionCall", "c")},
                "call payload changed shape",
            ),
            (
                {
                    "role": "model",
                    "parts": [{"functionCall": {"id": "real", "name": "f"}}],
                },
                {0: ("functionCall", "c")},
                "unexpectedly gained an id",
            ),
        ],
    )
    def test_opaque_history_asserts_its_internal_invariants(
        self, output: object, calls: dict[int, tuple[str, str]], expected: str
    ) -> None:
        with pytest.raises(AssertionError, match=expected):
            google_genai._opaque_history(None, (), {}, cast(Any, output), calls)


class TestFinalBodyStructuralGuards:
    """The final-body validators run on serialized JSON, so tolerate any shape."""

    _GEMINI_3 = "gemini-3-pro-preview"
    _JUNK: list[object] = [
        "not-a-content",
        {"role": "model", "parts": "not-a-list"},
        {"role": "model", "parts": ["not-a-part"]},
    ]

    def test_call_ids_ignore_non_sequence_contents(self) -> None:
        google_genai._validate_final_function_call_ids(
            {"contents": "x"}, self._GEMINI_3
        )

    def test_call_id_check_still_reaches_a_later_malformed_call(self) -> None:
        contents = [*self._JUNK, {"role": "model", "parts": [{"functionCall": "x"}]}]

        with pytest.raises(DialectError, match="functionCall must be an object"):
            google_genai._validate_final_function_call_ids(
                cast(Any, {"contents": contents}), self._GEMINI_3
            )

    def test_signatures_ignore_non_sequence_contents(self) -> None:
        google_genai._validate_final_function_call_signatures(
            {"contents": None}, self._GEMINI_3
        )

    def test_signature_check_skips_junk_but_still_finds_an_unsigned_call(
        self,
    ) -> None:
        google_genai._validate_final_function_call_signatures(
            cast(Any, {"contents": self._JUNK}), self._GEMINI_3
        )
        contents = [
            *self._JUNK,
            {"role": "model", "parts": [{"functionCall": {"id": "1", "name": "f"}}]},
        ]

        with pytest.raises(DialectError, match="thoughtSignature"):
            google_genai._validate_final_function_call_signatures(
                cast(Any, {"contents": contents}), self._GEMINI_3
            )


_SEARCH = ToolParams(hosted=(HostedToolSpec("google_search"),))
_SEARCH_CALL: dict[str, Any] = {
    "toolCall": {"id": "s1", "toolType": "GOOGLE_SEARCH_WEB", "args": {}},
    "thoughtSignature": "sig",
}
_SEARCH_RESULT: dict[str, Any] = {
    "toolResponse": {
        "id": "s1",
        "toolType": "GOOGLE_SEARCH_WEB",
        "response": {"ok": True},
    },
    "thoughtSignature": "sig",
}


def _with_model(model: str, parts: list[dict[str, Any]], **extra: Any) -> Any:
    return _run(
        lambda _: httpx.Response(200, json=_response(parts, **extra)),
        _request(tools=_SEARCH),
        model=model,
    )


class TestServerToolResponseGuards:
    """Hosted-tool reply shapes that must fail loudly rather than be guessed at."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (
                [
                    {
                        "toolCall": {
                            "id": "s1",
                            "toolType": "CODE_EXECUTION",
                            "args": {},
                        },
                        "thoughtSignature": "sig",
                    }
                ],
                "outside the request's allowed hosted-tool set",
            ),
            (
                [_SEARCH_CALL, _SEARCH_RESULT, _SEARCH_RESULT],
                "duplicate Google server tool response id",
            ),
            (
                [
                    _SEARCH_CALL,
                    {
                        "toolResponse": {
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "response": {},
                        },
                        "thoughtSignature": "sig",
                    },
                    {
                        "toolResponse": {
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "response": {},
                        },
                        "thoughtSignature": "sig",
                    },
                ],
                "without id is ambiguous",
            ),
            (
                [
                    {
                        "toolResponse": {
                            "id": "orphan",
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "response": {},
                        },
                        "thoughtSignature": "sig",
                    }
                ],
                "without its call",
            ),
            (
                [
                    _SEARCH_CALL,
                    {
                        "toolResponse": {
                            "id": "s1",
                            "toolType": "URL_CONTEXT",
                            "response": {},
                        },
                        "thoughtSignature": "sig",
                    },
                ],
                "server tool type changed",
            ),
        ],
    )
    async def test_inconsistent_server_tool_parts_are_rejected(
        self, parts: list[dict[str, Any]], expected: str
    ) -> None:
        with pytest.raises(OutputContractError, match=expected):
            await _with_model("gemini-3-pro-preview", parts)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("part", "expected"),
        [
            (_SEARCH_CALL, "unrequested server tool call"),
            (_SEARCH_RESULT, "unrequested server tool response"),
        ],
    )
    async def test_server_tool_parts_without_a_hosted_request_are_rejected(
        self, part: dict[str, Any], expected: str
    ) -> None:
        with pytest.raises(OutputContractError, match=expected):
            await _run(
                lambda _: httpx.Response(200, json=_response([part])),
                _request(),
                model="gemini-3-pro-preview",
            )

    @pytest.mark.anyio
    async def test_candidates_cannot_each_carry_singular_grounding(self) -> None:
        grounding = {"groundingChunks": [{"web": {"uri": "https://w", "title": "t"}}]}
        candidates = [
            {
                "index": index,
                "content": {"role": "model", "parts": [{"text": "x"}]},
                "finishReason": "STOP",
                "groundingMetadata": grounding,
            }
            for index in range(2)
        ]
        # The n>1 gate fires first; the later ``len(groundings) > 1`` guard is
        # unreachable (n==1 admits one candidate, so at most one grounding).
        with pytest.raises(OutputContractError, match="cannot represent choices"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json={
                        "candidates": candidates,
                        "usageMetadata": {
                            "promptTokenCount": 1,
                            "candidatesTokenCount": 1,
                        },
                    },
                ),
                _request(sampling=SamplingParams(n=2)),
            )


class TestCandidateContentGuards:
    @pytest.mark.anyio
    async def test_candidate_without_content_is_an_empty_scoreable_reply(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json={
                    "candidates": [{"index": 0, "finishReason": "SAFETY"}],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0},
                },
            )
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("SAFETY",)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (
                {"role": "model", "parts": [{"text": "x"}], "extra": 1},
                "unsupported field",
            ),
            (
                {"role": "model", "parts": [{"text": "x", "partMetadata": "bad"}]},
                "partMetadata must be an object",
            ),
            (
                {"role": "model", "parts": [{"text": "t", "thought": True, "x": 1}]},
                "unsupported field",
            ),
            (
                {"role": "model", "parts": [{"text": 5, "thought": True}]},
                "exactly one text field",
            ),
        ],
    )
    async def test_malformed_candidate_content_is_rejected(
        self, content: dict[str, Any], expected: str
    ) -> None:
        with pytest.raises(OutputContractError, match=expected):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {"index": 0, "content": content, "finishReason": "STOP"}
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 1,
                            "candidatesTokenCount": 1,
                        },
                    },
                )
            )


class TestRequestSideGuards:
    @pytest.mark.parametrize("summary", ["none", "auto"])
    def test_reasoning_summary_none_is_a_documented_noop(
        self, summary: Literal["none", "auto"]
    ) -> None:
        req = _request(reasoning=ReasoningParams(summary=summary))
        audit = RequestAudit(active_request_leaves(req))
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
        finally:
            import asyncio

            asyncio.run(connection.aclose())

        generation_config = prepared.thaw_body().get("generationConfig", {})
        assert isinstance(generation_config, dict)
        thinking = generation_config.get("thinkingConfig", {})
        assert isinstance(thinking, dict)
        assert ("includeThoughts" in thinking) is (summary == "auto")

    def test_multimodal_config_guards_against_future_modality_widening(self) -> None:
        """The shared option type admits only ``image`` today.

        ``_validate_multimodal_config`` is the dialect's own backstop for the day
        ``_MULTIMODAL_INPUT_MODALITIES`` grows beyond what Gemini input maps, so
        it is exercised by forcing a value past the option's constructor.
        """

        from sieval.core.models.capabilities import MultimodalInputOptions

        google_genai._validate_multimodal_config(
            MultimodalInputOptions(modalities=("image",))
        )
        widened = _forced(MultimodalInputOptions(), modalities=("image", "audio"))
        with pytest.raises(ValueError, match="does not support modalities"):
            google_genai._validate_multimodal_config(widened)

    def test_empty_dialect_options_are_a_documented_noop(self) -> None:
        req = _request(dialect_options=DialectOptions("google_genai", {}))
        audit = RequestAudit(active_request_leaves(req))
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            dialect.prepare(req, audit)
        finally:
            import asyncio

            asyncio.run(connection.aclose())

        assert type(audit.decisions["dialect_options"]).__name__ == "NoOp"


class TestConstructionAndInputGuards:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("", "must not be empty"),
            ("models/", "Gemini model id"),
            ("a/b", "Gemini model id"),
        ],
    )
    def test_requested_model_id_must_name_one_gemini_model(
        self, model: str, expected: str
    ) -> None:
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        with pytest.raises(ValueError, match=expected):
            GoogleGenAIDialect(connection, model)

    def test_dialect_requires_the_provider_neutral_http_connection(self) -> None:
        with pytest.raises(TypeError, match="AsyncHTTPJSONConnection"):
            GoogleGenAIDialect(object(), "gemini-2.5-flash")

    @pytest.mark.parametrize("convert", ["main", "verification"])
    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda: _forced(_chat(), messages=()), "at least one message"),
            (
                lambda: _chat(
                    ChatMessage("user", (TextPart("x"),), name="ada"),
                ),
                "no ChatMessage.name",
            ),
            (
                lambda: _chat(
                    ChatMessage("user", (TextPart("q"),)),
                    ChatMessage("system", (TextPart("late"),)),
                ),
                "must precede conversation turns",
            ),
            (
                lambda: _chat(
                    ChatMessage("system", (ImagePart(url="u"),)),
                    ChatMessage("user", (TextPart("q"),)),
                ),
                "text-only",
            ),
        ],
    )
    def test_main_path_and_audit_mirror_reject_the_same_histories(
        self, convert: str, build: Callable[[], ChatInput], expected: str
    ) -> None:
        function = (
            google_genai._input_to_wire
            if convert == "main"
            else google_genai._verification_input_to_wire
        )

        with pytest.raises(DialectError, match=expected):
            function(build(), None)

    @pytest.mark.parametrize("convert", ["main", "verification"])
    def test_non_object_tool_result_is_wrapped_and_error_is_tagged(
        self, convert: str
    ) -> None:
        function = (
            google_genai._input_to_wire
            if convert == "main"
            else google_genai._verification_input_to_wire
        )

        def lower(result: object, *, is_error: bool = False) -> Mapping[str, object]:
            history = _chat(
                ChatMessage("user", (TextPart("q"),)),
                ChatMessage("assistant", (ToolCallPart("c", "f", {}),)),
                ChatMessage(
                    "tool", (ToolResultPart("c", cast(Any, result), is_error=is_error),)
                ),
            )
            contents = function(history, None)[1]
            last_content = contents[-1]
            assert isinstance(last_content, dict)
            parts = last_content["parts"]
            assert isinstance(parts, list)
            first_part = parts[0]
            assert isinstance(first_part, dict)
            function_response = first_part["functionResponse"]
            assert isinstance(function_response, dict)
            return function_response

        assert lower([1, 2])["response"] == {"result": [1, 2]}
        assert lower({"a": 1})["response"] == {"a": 1}
        assert lower("boom", is_error=True)["response"] == {"error": "boom"}

    def test_tool_choice_requires_declared_function_tools(self) -> None:
        req = _request(tools=ToolParams(choice="required"))
        audit = RequestAudit(active_request_leaves(req))
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")

        dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))

        decision = audit.decisions["tools.choice"]
        assert isinstance(decision, Rejected)
        assert "requires function tools" in decision.reason

    def test_unsupported_structured_output_format_is_rejected(self) -> None:
        req = _request(
            structured_output=_forced(StructuredOutputParams(), format="xml")
        )
        audit = RequestAudit(active_request_leaves(req))
        connection = _connection(lambda _: httpx.Response(200, json=_response()))
        dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")

        dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))

        decision = audit.decisions["structured_output.format"]
        assert isinstance(decision, Rejected)
        assert "unsupported Google structured output format" in decision.reason


class TestAllowedServerToolTypes:
    @staticmethod
    def _allowed(*hosted: HostedToolSpec) -> frozenset[str]:
        return google_genai._allowed_server_tool_types(
            _request(tools=ToolParams(hosted=hosted))
        )

    def test_each_hosted_kind_maps_to_its_server_tool_types(self) -> None:
        assert self._allowed(HostedToolSpec("url_context")) == {"URL_CONTEXT"}
        assert self._allowed(HostedToolSpec("google_search")) == {"GOOGLE_SEARCH_WEB"}
        assert self._allowed(
            HostedToolSpec(
                "google_search",
                {"searchTypes": {"webSearch": {}, "imageSearch": {}}},
            )
        ) == {"GOOGLE_SEARCH_WEB", "GOOGLE_SEARCH_IMAGE"}
        assert self._allowed(
            HostedToolSpec("google_search", {"searchTypes": {"imageSearch": {}}})
        ) == {"GOOGLE_SEARCH_IMAGE"}

    def test_unrecognized_kind_or_search_types_shape_admits_nothing(self) -> None:
        assert self._allowed(_forced(HostedToolSpec("google_search"), kind="x")) == (
            frozenset()
        )
        assert (
            self._allowed(HostedToolSpec("google_search", {"searchTypes": "bad"}))
            == frozenset()
        )


class TestHostedToolsVerifierShape:
    @staticmethod
    def _body(tools: object, *, invoke: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {"tools": tools}
        if invoke:
            body["toolConfig"] = {"includeServerSideToolInvocations": True}
        return body

    def _verifier(self, *, has_functions: bool = False) -> Any:
        return google_genai._HostedToolsVerifier(
            (HostedToolSpec("google_search"),), has_functions=has_functions
        )

    def test_exact_hosted_tool_list_verifies(self) -> None:
        assert self._verifier().verify(self._body([{"googleSearch": {}}])) is None

    @pytest.mark.parametrize(
        "tools",
        [
            [{"googleSearch": {}}, {"urlContext": {}}],  # extra, unrequested tool
            [{"googleSearch": {}}, {"googleSearch": {}}],  # duplicated tool
            [],
            "not-a-sequence",
        ],
    )
    def test_any_deviation_from_the_requested_hosted_tools_is_detected(
        self, tools: object
    ) -> None:
        assert self._verifier().verify(self._body(tools)) == "hosted tools changed"

    def test_function_declarations_must_lead_when_functions_are_requested(
        self,
    ) -> None:
        verifier = self._verifier(has_functions=True)
        ok = [{"functionDeclarations": [{"name": "f"}]}, {"googleSearch": {}}]

        assert verifier.verify(self._body(ok)) is None
        for tools in (
            [{"googleSearch": {}}],
            [{"googleSearch": {}}, {"functionDeclarations": []}],
            [{"functionDeclarations": [], "googleSearch": {}}],
        ):
            assert verifier.verify(self._body(tools)) == "hosted tools changed"

    def test_non_json_tool_content_never_verifies(self) -> None:
        assert (
            self._verifier().verify(self._body([{"googleSearch": {1, 2}}]))
            == "hosted tools changed"
        )

    def test_invocation_evidence_must_remain_enabled(self) -> None:
        assert (
            self._verifier().verify(self._body([{"googleSearch": {}}], invoke=False))
            == "hosted-tool invocation evidence was disabled"
        )

    def test_invalid_requested_config_is_reported(self) -> None:
        verifier = google_genai._HostedToolsVerifier(
            (_forced(HostedToolSpec("google_search"), config="bad"),)
        )

        assert verifier.verify(self._body([])) == "hosted tool config is invalid"


class TestStreamDeltaMerging:
    merge = staticmethod(google_genai._merge_stream_text_delta)

    def test_adjacent_text_deltas_join_into_one_part(self) -> None:
        assert self.merge({"text": "ab"}, {"text": "cd"}) == {"text": "abcd"}

    @pytest.mark.parametrize(
        ("previous", "current"),
        [
            ("not-a-mapping", {"text": "x"}),
            ({"text": "x"}, "not-a-mapping"),
            ({"text": 1}, {"text": "x"}),
            ({"text": "x"}, {"text": 2}),
            ({"text": "x"}, {"text": "y", "thought": True}),
            ({"text": "x", "thoughtSignature": "sig"}, {"text": "y"}),
            ({"text": "x"}, {"text": "y", "thoughtSignature": ""}),
            ({"text": "x"}, {"text": "y", "thoughtSignature": 7}),
            ({"text": "x", "functionCall": {}}, {"text": "y"}),
        ],
    )
    def test_parts_that_are_not_one_semantic_part_stay_separate(
        self, previous: object, current: object
    ) -> None:
        assert self.merge(previous, current) is None

    def test_signature_on_a_later_delta_closes_the_joined_part(self) -> None:
        assert self.merge({"text": "a"}, {"text": "b", "thoughtSignature": "sig"}) == {
            "text": "ab",
            "thoughtSignature": "sig",
        }
