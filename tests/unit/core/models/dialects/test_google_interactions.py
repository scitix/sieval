"""Tests for the Gemini Interactions API dialect.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Callable, Mapping
from typing import Any, cast

import httpx
import pytest

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    MultimodalInputOptions,
    ReasoningOptions,
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
from sieval.core.models.dialects import google_interactions
from sieval.core.models.dialects.google_interactions import (
    CAPABILITY_DECISIONS,
    GoogleInteractionsDialect,
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
from sieval.core.types import JSONValue
from sieval.core.utils.meta import is_truncated_finish_reason

_MODEL = "gemini-3.5-flash"
_WEATHER = {
    "name": "get_weather",
    "description": "Look up the weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _request(*messages: ChatMessage, **changes: object) -> Request:
    chat = ChatInput(messages or (ChatMessage("user", (TextPart("Hello"),)),))
    return Request(**cast(Any, {"input": chat, **changes}))


def _interaction(
    steps: list[dict[str, object]] | None = None,
    *,
    status: str = "completed",
    usage: object | None = None,
    **extra: object,
) -> dict[str, object]:
    return {
        "id": "interaction-1",
        "object": "interaction",
        "status": status,
        "model": _MODEL,
        "steps": steps
        if steps is not None
        else [{"type": "model_output", "content": [{"type": "text", "text": "Hi"}]}],
        "usage": usage
        if usage is not None
        else {
            "total_input_tokens": 4,
            "total_output_tokens": 2,
            "total_thought_tokens": 3,
            "total_cached_tokens": 1,
            "total_tokens": 9,
        },
        **extra,
    }


class _Recorder:
    """A mock transport that records every request and replies with *reply*."""

    def __init__(self, reply: Callable[[httpx.Request], httpx.Response]) -> None:
        self.reply = reply
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.reply(request)

    @property
    def body(self) -> dict[str, Any]:
        assert len(self.requests) == 1
        return json.loads(self.requests[0].content)


def _json_reply(payload: object, status: int = 200) -> _Recorder:
    return _Recorder(lambda _: httpx.Response(status, json=payload))


async def _run(
    recorder: _Recorder,
    request: Request | None = None,
    *,
    base_url: str = "https://generativelanguage.googleapis.com/",
    credential: str | None = "google-secret",
    model: str = _MODEL,
):
    connection = AsyncHTTPJSONConnection(
        httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(recorder)),
        credential,
    )
    try:
        dialect = GoogleInteractionsDialect(connection, model)
        return await dialect.arun(request or _request())
    finally:
        await connection.aclose()


def _sse(*events: Mapping[str, object]) -> bytes:
    lines: list[str] = []
    for event in events:
        lines.extend(
            (f"event: {event['event_type']}", f"data: {json.dumps(event)}", "")
        )
    return ("\n".join(lines) + "\n").encode()


def _stream_reply(*events: Mapping[str, object]) -> _Recorder:
    return _Recorder(
        lambda _: httpx.Response(
            200,
            content=_sse(*events),
            headers={"Content-Type": "text/event-stream"},
        )
    )


def _completed(status: str = "completed", **extra: object) -> dict[str, object]:
    return {
        "event_type": "interaction.completed",
        "interaction": {"id": "interaction-1", "status": status, **extra},
    }


def _dialect() -> GoogleInteractionsDialect:
    connection = AsyncHTTPJSONConnection(
        httpx.AsyncClient(base_url="https://generativelanguage.googleapis.com/"),
        None,
    )
    return GoogleInteractionsDialect(connection, _MODEL)


def _rejections(request: Request) -> dict[str, str]:
    audit = RequestAudit(active_request_leaves(request))
    _dialect().validate_request(request, audit, cast(Any, None))
    return {
        path: decision.reason
        for path, decision in audit.decisions.items()
        if isinstance(decision, Rejected)
    }


class TestDescriptor:
    def test_registry_marks_it_active_and_seedable(self) -> None:
        spec = DIALECT_SPECS["google_interactions"]
        assert spec.connection_family == "async_http_json"
        assert spec.implementation_status.value == "active"
        assert spec.request_seed_support.value == "supported"
        assert spec.input_kinds == ("chat",)
        # Model-family limits are left to Google's 400 (#151), but nothing was
        # measured on this endpoint, so no capability is provider-enforced yet.
        assert spec.provider_enforced_capabilities == frozenset()

    def test_eval_controls_absent_from_the_api_are_unsupported(self) -> None:
        assert set(CAPABILITY_DECISIONS) == set(CAPABILITY_KEYS)
        unsupported = {
            key
            for key, decision in CAPABILITY_DECISIONS.items()
            if isinstance(decision, Unsupported)
        }
        assert unsupported == {
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
            "hosted_tools",
            "stateful_session",
            "prefill",
            "fim",
        }
        assert all(
            isinstance(CAPABILITY_DECISIONS[key], Supported)
            for key in ("reasoning", "function_tools", "structured_output")
        )

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (ReasoningOptions(budget_tokens=10), "no thinking budget"),
            (ReasoningOptions(summary="detailed"), "'none' or 'auto'"),
            (ReasoningOptions(effort="LOW"), "unsupported Interactions thinking"),
        ],
    )
    def test_reasoning_config_refuses_controls_the_api_lacks(
        self, options: ReasoningOptions, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            google_interactions._validate_reasoning_config(options)

    def test_reasoning_config_accepts_a_level_and_summary(self) -> None:
        google_interactions._validate_reasoning_config(
            ReasoningOptions(effort="high", summary="auto")
        )

    def test_multimodal_config_admits_images_only(self) -> None:
        google_interactions._validate_multimodal_config(
            MultimodalInputOptions(modalities=("image",))
        )
        with pytest.raises(ValueError, match="audio"):
            google_interactions._validate_multimodal_config(
                MultimodalInputOptions(modalities=("image", "audio"))
            )


class TestConstruction:
    def test_requires_an_http_json_connection(self) -> None:
        with pytest.raises(TypeError, match="AsyncHTTPJSONConnection"):
            GoogleInteractionsDialect(object(), _MODEL)

    @pytest.mark.parametrize("model", ["", "models/", "tunedModels/x/y"])
    def test_rejects_a_model_id_that_is_not_a_gemini_name(self, model: str) -> None:
        connection = AsyncHTTPJSONConnection(httpx.AsyncClient(), None)
        with pytest.raises(ValueError, match="requested_model_id"):
            GoogleInteractionsDialect(connection, model)


class TestRequestBody:
    @pytest.mark.anyio
    async def test_store_false_is_always_sent(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(recorder)

        body = recorder.body
        assert body["store"] is False
        assert body["model"] == _MODEL
        assert body["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "Hello"}]}
        ]
        for key in ("previous_interaction_id", "background", "stream"):
            assert key not in body

    @pytest.mark.anyio
    async def test_route_header_and_models_prefix(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(recorder, model=f"models/{_MODEL}")

        sent = recorder.requests[0]
        assert sent.method == "POST"
        assert str(sent.url) == (
            "https://generativelanguage.googleapis.com/v1beta/interactions"
        )
        assert sent.headers["x-goog-api-key"] == "google-secret"
        assert "authorization" not in sent.headers
        assert recorder.body["model"] == _MODEL

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("https://gw.example/model-api/v1", "/model-api/v1/interactions"),
            ("https://gw.example/model-api/v1beta/", "/model-api/v1beta/interactions"),
        ],
    )
    async def test_versioned_gateway_base_keeps_its_prefix(
        self, base_url: str, expected: str
    ) -> None:
        recorder = _json_reply(_interaction())
        await _run(recorder, base_url=base_url)
        assert recorder.requests[0].url.path == expected

    @pytest.mark.anyio
    async def test_no_credential_sends_no_key_header(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(recorder, credential=None)
        assert "x-goog-api-key" not in recorder.requests[0].headers

    @pytest.mark.anyio
    async def test_sampling_and_reasoning_lower_into_generation_config(self) -> None:
        recorder = _json_reply(
            _interaction(
                [
                    {
                        "type": "thought",
                        "signature": "c2ln",
                        "summary": [{"type": "text", "text": "plan"}],
                    },
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "x"}],
                    },
                ]
            )
        )
        await _run(
            recorder,
            _request(
                sampling=SamplingParams(
                    temperature=0.0,
                    top_p=0.5,
                    max_tokens=64,
                    stop=("END",),
                    seed=7,
                ),
                reasoning=ReasoningParams(effort="low", summary="auto"),
            ),
        )
        assert recorder.body["generation_config"] == {
            "temperature": 0.0,
            "top_p": 0.5,
            "max_output_tokens": 64,
            "stop_sequences": ["END"],
            "seed": 7,
            "thinking_level": "low",
            "thinking_summaries": "auto",
        }

    @pytest.mark.anyio
    async def test_system_message_becomes_system_instruction(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(
                ChatMessage("system", (TextPart("Be terse."),)),
                ChatMessage("user", (TextPart("Hi"),)),
            ),
        )
        assert recorder.body["system_instruction"] == "Be terse."
        assert recorder.body["input"][0]["type"] == "user_input"

    @pytest.mark.anyio
    async def test_stateless_history_is_resent_as_steps(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(
                ChatMessage("user", (TextPart("Weather in Paris?"),)),
                ChatMessage(
                    "assistant",
                    (
                        TextPart("Checking."),
                        ToolCallPart("call-1", "get_weather", {"city": "Paris"}),
                    ),
                ),
                ChatMessage(
                    "tool", (ToolResultPart("call-1", {"temp": 20}, is_error=True),)
                ),
                tools=ToolParams(functions=(_WEATHER,)),
            ),
        )
        assert recorder.body["input"] == [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "Weather in Paris?"}],
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "Checking."}],
            },
            {
                "type": "function_call",
                "id": "call-1",
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_result",
                "call_id": "call-1",
                "result": {"temp": 20},
                "name": "get_weather",
                "is_error": True,
            },
        ]

    @pytest.mark.anyio
    async def test_images_lower_to_image_content(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(
                ChatMessage(
                    "user",
                    (
                        TextPart("Compare"),
                        ImagePart(url="gs://bucket/a.png", media_type="image/png"),
                        ImagePart(data="aGk=", media_type="image/jpeg"),
                    ),
                )
            ),
        )
        assert recorder.body["input"][0]["content"] == [
            {"type": "text", "text": "Compare"},
            {"type": "image", "uri": "gs://bucket/a.png", "mime_type": "image/png"},
            {"type": "image", "data": "aGk=", "mime_type": "image/jpeg"},
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("auto", "auto"),
            ("none", "none"),
            ("required", "any"),
            (
                {"type": "function", "function": {"name": "get_weather"}},
                {"allowed_tools": {"mode": "any", "tools": ["get_weather"]}},
            ),
            (
                {"type": "function", "name": "get_weather"},
                {"allowed_tools": {"mode": "any", "tools": ["get_weather"]}},
            ),
        ],
    )
    async def test_function_tools_and_choice(
        self, choice: object, expected: object
    ) -> None:
        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(
                tools=ToolParams(
                    functions=({"type": "function", "function": _WEATHER},),
                    choice=cast(Any, choice),
                )
            ),
        )
        assert recorder.body["tools"] == [{"type": "function", **_WEATHER}]
        assert recorder.body["generation_config"]["tool_choice"] == expected

    @pytest.mark.anyio
    async def test_json_schema_output_lowers_to_response_format(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        recorder = _json_reply(
            _interaction(
                [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": '{"a":1}'}],
                    }
                ]
            )
        )
        result = await _run(
            recorder,
            _request(
                structured_output=StructuredOutputParams(
                    format="json_schema", schema=schema
                )
            ),
        )
        assert recorder.body["response_format"] == {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        }
        assert result.structured_output is not None
        assert result.structured_output.value == {"a": 1}

    @pytest.mark.anyio
    async def test_json_object_output_sends_no_schema(self) -> None:
        recorder = _json_reply(
            _interaction(
                [{"type": "model_output", "content": [{"type": "text", "text": "[]"}]}]
            )
        )
        await _run(
            recorder,
            _request(structured_output=StructuredOutputParams(format="json_object")),
        )
        assert recorder.body["response_format"] == {
            "type": "text",
            "mime_type": "application/json",
        }

    @pytest.mark.anyio
    async def test_dialect_option_passes_through_at_the_root(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(
                dialect_options=DialectOptions(
                    "google_interactions", {"service_tier": "flex"}
                )
            ),
        )
        assert recorder.body["service_tier"] == "flex"

    @pytest.mark.anyio
    async def test_empty_dialect_options_are_a_noop(self) -> None:
        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(dialect_options=DialectOptions("google_interactions", {})),
        )
        assert set(recorder.body) == {"model", "input", "store"}

    @pytest.mark.anyio
    async def test_request_params_record_store_false_without_the_prompt(self) -> None:
        recorder = _json_reply(_interaction())
        result = await _run(recorder, _request(sampling=SamplingParams(seed=3)))
        assert result.request_params == {
            "envelope": "http",
            "method": "POST",
            "path": "/v1beta/interactions",
            "body": {
                "model": _MODEL,
                "store": False,
                "generation_config": {"seed": 3},
            },
        }


class TestRejections:
    @pytest.mark.parametrize(
        ("request_", "path", "message"),
        [
            (_request(sampling=SamplingParams(n=2)), "sampling.n", "candidate count"),
            (_request(sampling=SamplingParams(top_k=5)), "sampling.top_k", "top_k"),
            (
                _request(sampling=SamplingParams(frequency_penalty=0.1)),
                "sampling.frequency_penalty",
                "penalty",
            ),
            (
                _request(sampling=SamplingParams(presence_penalty=0.1)),
                "sampling.presence_penalty",
                "penalty",
            ),
            (
                _request(scoring=ScoringParams(sampled_logprobs=True)),
                "scoring.sampled_logprobs",
                "no logprobs",
            ),
            (
                _request(scoring=ScoringParams(input_scoring=True)),
                "scoring.input_scoring",
                "no logprobs",
            ),
            (
                _request(reasoning=ReasoningParams(budget_tokens=100)),
                "reasoning.budget_tokens",
                "no thinking budget",
            ),
            (
                _request(reasoning=ReasoningParams(effort="HIGH")),
                "reasoning.effort",
                "thinking level",
            ),
            (
                _request(reasoning=ReasoningParams(summary="detailed")),
                "reasoning.summary",
                "'none' or 'auto'",
            ),
            (
                _request(tools=ToolParams(functions=(_WEATHER,), parallel=True)),
                "tools.parallel",
                "parallel",
            ),
            (
                _request(tools=ToolParams(hosted=(HostedToolSpec("google_search"),))),
                "tools.hosted",
                "hosted tools",
            ),
            (
                _request(tools=ToolParams(functions=({"name": "f", "strict": True},))),
                "tools.functions",
                "strict",
            ),
            (_request(tools=ToolParams(choice="auto")), "tools.choice", "requires"),
            (
                _request(tools=ToolParams(functions=(_WEATHER,), choice="maybe")),
                "tools.choice",
                "unsupported",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=(_WEATHER,),
                        choice={"type": "function", "name": "other"},
                    )
                ),
                "tools.choice",
                "undeclared",
            ),
            (
                _request(session=SessionParams(previous_response_id="i-0")),
                "session.previous_response_id",
                "store=false",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_object", schema={"type": "object"}
                    )
                ),
                "structured_output.schema",
                "json_schema",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={"type": "object"}, name="x"
                    )
                ),
                "structured_output.name",
                "no name",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={"type": "object"}, strict=True
                    )
                ),
                "structured_output.strict",
                "strict",
            ),
            (
                _request(
                    dialect_options=DialectOptions(
                        "google_interactions", {"store": True}
                    )
                ),
                "dialect_options.store",
                "owned by google_interactions",
            ),
            (
                _request(
                    dialect_options=DialectOptions("google_interactions", {"": 1})
                ),
                "dialect_options.",
                "non-empty",
            ),
            (
                _request(ChatMessage("developer", (TextPart("x"),))),
                "input.chat",
                "developer",
            ),
            (
                _request(
                    ChatMessage("user", (TextPart("q"),)),
                    ChatMessage("assistant", (TextPart("a"),)),
                ),
                "input.chat",
                "prefill",
            ),
            (
                _request(ChatMessage("user", (TextPart("x"),), name="bob")),
                "input.chat.message.name",
                "name",
            ),
            (
                _request(
                    ChatMessage(
                        "user",
                        (ImagePart(url="u", media_type="image/png", detail="low"),),
                    )
                ),
                "input.modality.image.detail",
                "detail",
            ),
        ],
    )
    def test_refused_before_any_io(
        self, request_: Request, path: str, message: str
    ) -> None:
        reasons = _rejections(request_)
        assert path in reasons
        assert message in reasons[path]

    def test_completion_input_is_refused(self) -> None:
        reasons = _rejections(Request(CompletionInput("x", suffix="y")))
        assert "requires ChatInput" in reasons["input.completion"]
        assert "requires ChatInput" in reasons["input.completion.suffix"]

    def test_unknown_role_is_refused(self) -> None:
        message = ChatMessage(cast(Any, "narrator"), (TextPart("x"),))
        assert "invalid role" in _rejections(_request(message))["input.chat"]

    @pytest.mark.anyio
    async def test_a_refusal_sends_nothing(self) -> None:
        recorder = _json_reply(_interaction())
        with pytest.raises(RequestAuditError, match="candidate count"):
            await _run(recorder, _request(sampling=SamplingParams(n=2)))
        assert recorder.requests == []

    @pytest.mark.parametrize(
        ("messages", "message"),
        [
            (
                (
                    ChatMessage("user", (TextPart("a"),)),
                    ChatMessage("system", (TextPart("late"),)),
                    ChatMessage("user", (TextPart("b"),)),
                ),
                "precede",
            ),
            (
                (
                    ChatMessage("system", (TextPart("a"),)),
                    ChatMessage("system", (TextPart("b"),)),
                    ChatMessage("user", (TextPart("c"),)),
                ),
                "single string",
            ),
            (
                (
                    ChatMessage("system", (ImagePart(url="u"),)),
                    ChatMessage("user", (TextPart("c"),)),
                ),
                "text-only",
            ),
            ((ChatMessage("system", (TextPart("only"),)),), "must end with"),
            (
                (ChatMessage("user", (ToolCallPart("c", "f", {}),)),),
                "only assistant",
            ),
            (
                (
                    ChatMessage("user", (TextPart("q"),)),
                    ChatMessage("assistant", (ToolResultPart("c", 1),)),
                    ChatMessage("user", (TextPart("q"),)),
                ),
                "only user or tool",
            ),
            (
                (
                    ChatMessage("user", (TextPart("q"),)),
                    ChatMessage("assistant", (ImagePart(url="u"),)),
                    ChatMessage("user", (TextPart("q"),)),
                ),
                "cannot carry images",
            ),
            (
                (
                    ChatMessage("user", (TextPart("q"),)),
                    ChatMessage("assistant", (ToolCallPart("c", "f", cast(Any, [1])),)),
                    ChatMessage("tool", (ToolResultPart("c", 1),)),
                ),
                "must be an object",
            ),
            (
                (ChatMessage("user", (ImagePart(url="u", media_type="image/svg"),)),),
                "media type",
            ),
            ((ChatMessage("user", (ImagePart(url=""),)),), "must not be empty"),
            ((ChatMessage("user", (ImagePart(data="aGk="),)),), "explicit MIME"),
        ],
    )
    def test_unrepresentable_history_fails_in_prepare(
        self, messages: tuple[ChatMessage, ...], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            google_interactions._input_to_wire(ChatInput(messages), None)

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({"type": "function", "function": "x", "extra": 1}, "OpenAI-shaped"),
            ({"description": "d"}, "requires a name"),
            ({"name": "f", "description": 3}, "description must be a string"),
            ({"name": "f", "parameters": []}, "parameters must be an object"),
        ],
    )
    def test_invalid_function_declarations(
        self, raw: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            google_interactions._function_declaration(raw)

    @pytest.mark.parametrize(
        ("choice", "message"),
        [
            (3, "string or mapping"),
            ({"type": "auto"}, "function-specific"),
            ({"type": "function", "name": ""}, "requires a name"),
        ],
    )
    def test_invalid_tool_choices(self, choice: object, message: str) -> None:
        with pytest.raises(DialectError, match=message):
            google_interactions._tool_choice(cast(Any, choice), frozenset({"f"}))


class TestLift:
    @pytest.mark.anyio
    async def test_text_status_usage_and_model(self) -> None:
        result = await _run(_json_reply(_interaction()))

        assert result.texts == ("Hi",)
        assert result.finish_reasons == ("completed",)
        assert result.response_model == _MODEL
        assert result.reasoning is None
        usage = result.usage
        assert usage is not None
        # Thoughts are outside total_output_tokens, as on generateContent.
        assert (usage.input_tokens, usage.output_tokens) == (4, 5)
        assert (usage.reasoning_tokens, usage.cached_tokens) == (3, 1)
        assert usage.total_tokens == 9
        assert usage.reported_total_tokens is None

    @pytest.mark.anyio
    async def test_tool_use_tokens_count_as_input(self) -> None:
        result = await _run(
            _json_reply(
                _interaction(
                    usage={
                        "total_input_tokens": 4,
                        "total_tool_use_tokens": 6,
                        "total_output_tokens": 2,
                        "total_tokens": 99,
                    }
                )
            )
        )
        usage = result.usage
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens) == (10, 2)
        assert usage.reasoning_tokens is None
        assert usage.reported_total_tokens == 99

    @pytest.mark.anyio
    async def test_incomplete_is_recorded_and_counted_as_truncated(self) -> None:
        result = await _run(_json_reply(_interaction([], status="incomplete")))
        assert result.texts == ("",)
        assert result.finish_reasons == ("incomplete",)
        assert is_truncated_finish_reason("incomplete")

    @pytest.mark.anyio
    async def test_incomplete_leaves_structured_output_optional(self) -> None:
        result = await _run(
            _json_reply(
                _interaction(
                    [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "{"}],
                        }
                    ],
                    status="incomplete",
                )
            ),
            _request(structured_output=StructuredOutputParams(format="json_object")),
        )
        assert result.structured_output is None

    @pytest.mark.anyio
    async def test_failed_status_raises_googles_message(self) -> None:
        with pytest.raises(OutputContractError, match="quota gone"):
            await _run(
                _json_reply(
                    _interaction(
                        [],
                        status="failed",
                        errors=[{"code": "x", "message": "quota gone"}],
                    )
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", ["in_progress", "queued", "cancelled"])
    async def test_non_terminal_status_is_refused(self, status: str) -> None:
        with pytest.raises(OutputContractError, match="non-terminal"):
            await _run(_json_reply(_interaction(status=status)))

    @pytest.mark.anyio
    async def test_function_call_with_requires_action(self) -> None:
        result = await _run(
            _json_reply(
                _interaction(
                    [
                        {
                            "type": "function_call",
                            "id": "gth23981",
                            "name": "get_weather",
                            "arguments": {"city": "Boston"},
                        }
                    ],
                    status="requires_action",
                )
            ),
            _request(tools=ToolParams(functions=(_WEATHER,))),
        )
        assert result.finish_reasons == ("requires_action",)
        assert result.tool_calls is not None
        [call] = result.tool_calls
        assert (call.call_id, call.name, call.arguments) == (
            "gth23981",
            "get_weather",
            {"city": "Boston"},
        )

    @pytest.mark.anyio
    async def test_requires_action_without_a_call_is_refused(self) -> None:
        with pytest.raises(OutputContractError, match="requires_action"):
            await _run(_json_reply(_interaction(status="requires_action")))

    @pytest.mark.anyio
    async def test_undeclared_function_call_is_refused(self) -> None:
        step = {"type": "function_call", "id": "1", "name": "rm_rf", "arguments": {}}
        with pytest.raises(OutputContractError, match="undeclared"):
            await _run(
                _json_reply(_interaction([step], status="requires_action")),
                _request(tools=ToolParams(functions=(_WEATHER,))),
            )

    @pytest.mark.anyio
    async def test_url_citations_are_lifted(self) -> None:
        result = await _run(
            _json_reply(
                _interaction(
                    [
                        {
                            "type": "model_output",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "See",
                                    "annotations": [
                                        {
                                            "type": "url_citation",
                                            "url": "https://a.example",
                                            "title": "A",
                                            "start_index": 0,
                                            "end_index": 3,
                                        },
                                        {"type": "file_citation", "file_name": "f"},
                                    ],
                                }
                            ],
                        }
                    ]
                )
            )
        )
        assert result.citations is not None
        assert [(c.url, c.title) for c in result.citations] == [
            ("https://a.example", "A")
        ]

    @pytest.mark.anyio
    async def test_thought_summary_and_signature_round_trip(self) -> None:
        thought = {
            "type": "thought",
            "signature": "c2lnLTE=",
            "summary": [{"type": "text", "text": "Plan first."}],
        }
        output = {"type": "model_output", "content": [{"type": "text", "text": "4"}]}
        first = await _run(
            _json_reply(_interaction([thought, output])),
            _request(
                ChatMessage("system", (TextPart("Math."),)),
                ChatMessage("user", (TextPart("2+2?"),)),
                reasoning=ReasoningParams(summary="auto"),
            ),
        )
        assert first.reasoning is not None
        [item] = first.reasoning
        assert item is not None
        assert item.text == "Plan first."
        assert item.opaque_roundtrip is not None

        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(
                ChatMessage("user", (TextPart("And 3+3?"),)),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "google_interactions", item.opaque_roundtrip
                    )
                ),
            ),
        )
        body = recorder.body
        # The signed thought is resent verbatim, between the turns it sat in.
        assert body["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "2+2?"}]},
            thought,
            output,
            {"type": "user_input", "content": [{"type": "text", "text": "And 3+3?"}]},
        ]
        assert body["system_instruction"] == "Math."
        assert body["store"] is False

    @pytest.mark.anyio
    async def test_continuation_names_a_tool_result_from_its_recorded_call(
        self,
    ) -> None:
        call = {
            "type": "function_call",
            "id": "c-9",
            "name": "get_weather",
            "arguments": {},
        }
        opaque = json.dumps({"version": 1, "steps": [call]})
        recorder = _json_reply(_interaction())
        await _run(
            recorder,
            _request(
                ChatMessage("tool", (ToolResultPart("c-9", "sunny"),)),
                tools=ToolParams(functions=(_WEATHER,)),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "google_interactions", opaque
                    )
                ),
            ),
        )
        assert recorder.body["input"][-1] == {
            "type": "function_result",
            "call_id": "c-9",
            "result": "sunny",
            "name": "get_weather",
        }

    @pytest.mark.anyio
    async def test_empty_requested_summary_is_refused(self) -> None:
        thought = {"type": "thought", "signature": "c2ln", "summary": []}
        with pytest.raises(OutputContractError, match="empty requested thought"):
            await _run(
                _json_reply(_interaction([thought])),
                _request(reasoning=ReasoningParams(summary="auto")),
            )

    @pytest.mark.parametrize(
        ("opaque", "message"),
        [
            ("not json", "JSON"),
            ('{"steps": []}', "envelope"),
            ('{"version": 2, "steps": []}', "version"),
            ('{"version": 1, "steps": [1]}', "steps are invalid"),
            ('{"version": 1, "steps": [], "system_instruction": 3}', "string"),
        ],
    )
    def test_invalid_opaque_continuations(self, opaque: str, message: str) -> None:
        with pytest.raises(DialectError, match=message):
            google_interactions._input_to_wire(
                ChatInput((ChatMessage("user", (TextPart("x"),)),)), opaque
            )

    def test_conflicting_opaque_system_instruction_is_refused(self) -> None:
        opaque = json.dumps({"version": 1, "steps": [], "system_instruction": "A"})
        with pytest.raises(DialectError, match="conflicts"):
            google_interactions._input_to_wire(
                ChatInput(
                    (
                        ChatMessage("system", (TextPart("B"),)),
                        ChatMessage("user", (TextPart("x"),)),
                    )
                ),
                opaque,
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ([], "must be an object"),
            ({"status": ""}, "status"),
            ({"status": "completed", "steps": {}}, "must be a list"),
            ({"status": "completed", "steps": [{"type": "audio"}]}, "unsupported step"),
            (
                {"status": "completed", "steps": [{"type": "user_input"}]},
                "echoed",
            ),
            (
                {
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "image"}]}],
                },
                "unsupported type",
            ),
            (
                {
                    "status": "completed",
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": 1}],
                        }
                    ],
                },
                "text must be a string",
            ),
            (
                {
                    "status": "completed",
                    "steps": [
                        {"type": "model_output", "error": {"message": "blocked"}}
                    ],
                },
                "blocked",
            ),
            (
                {
                    "status": "completed",
                    "steps": [{"type": "thought", "signature": ""}],
                },
                "signature",
            ),
            (
                {
                    "status": "completed",
                    "steps": [
                        {"type": "thought", "summary": [{"type": "text", "text": 2}]}
                    ],
                },
                "summary text",
            ),
            (
                {
                    "status": "requires_action",
                    "steps": [
                        {
                            "type": "function_call",
                            "id": "1",
                            "name": "get_weather",
                            "arguments": [],
                        }
                    ],
                },
                "arguments must be an object",
            ),
            ({"status": "completed", "usage": []}, "usage must be an object"),
            (
                {"status": "completed", "usage": {"total_input_tokens": -1}},
                "non-negative",
            ),
            ({"status": "completed", "model": 3}, "interaction.model"),
            (
                {
                    "status": "completed",
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "x",
                                    "annotations": [{"type": "url_citation"}],
                                }
                            ],
                        }
                    ],
                },
                "url",
            ),
        ],
    )
    async def test_malformed_replies_fail_loudly(
        self, payload: object, message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                _json_reply(payload),
                _request(tools=ToolParams(functions=(_WEATHER,))),
            )

    @pytest.mark.anyio
    async def test_invalid_structured_json_is_refused(self) -> None:
        with pytest.raises(OutputContractError, match="not valid JSON"):
            await _run(
                _json_reply(_interaction()),
                _request(
                    structured_output=StructuredOutputParams(format="json_object")
                ),
            )

    @pytest.mark.anyio
    async def test_provider_error_carries_googles_message(self) -> None:
        recorder = _json_reply({"error": {"message": "model not found"}}, status=404)
        with pytest.raises(httpx.HTTPStatusError, match="model not found"):
            await _run(recorder)

    @pytest.mark.anyio
    async def test_duplicate_keys_in_the_reply_are_refused(self) -> None:
        recorder = _Recorder(
            lambda _: httpx.Response(
                200, content=b'{"status": "completed", "status": "failed"}'
            )
        )
        with pytest.raises(OutputContractError, match="strict JSON"):
            await _run(recorder)


class TestStreaming:
    @pytest.mark.anyio
    async def test_stream_merges_steps_and_terminal_usage(self) -> None:
        recorder = _stream_reply(
            {
                "event_type": "interaction.created",
                "interaction": {"id": "i", "status": "in_progress", "model": _MODEL},
            },
            {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "thought"},
            },
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "Hm"},
                },
            },
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "m."},
                },
            },
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "c2ln"},
            },
            {"event_type": "step.stop", "index": 0},
            {
                "event_type": "step.start",
                "index": 1,
                "step": {"type": "model_output"},
            },
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": "Hel"},
            },
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": "lo"},
                "metadata": {"total_usage": {"total_output_tokens": 1}},
            },
            {
                "event_type": "step.stop",
                "index": 1,
                "usage": {"total_input_tokens": 3, "total_output_tokens": 2},
            },
            {
                "event_type": "interaction.status_update",
                "interaction_id": "i",
                "status": "completed",
            },
            _completed(),
        )
        result = await _run(
            recorder,
            _request(
                scheduling=SchedulingParams(stream=True),
                reasoning=ReasoningParams(summary="auto"),
            ),
        )
        assert recorder.body["stream"] is True
        assert recorder.requests[0].headers["accept"] == "text/event-stream"
        assert result.texts == ("Hello",)
        assert result.finish_reasons == ("completed",)
        assert result.response_model == _MODEL
        assert result.reasoning is not None
        [item] = result.reasoning
        assert item is not None
        assert item.text == "Hmm."
        assert item.opaque_roundtrip is not None
        assert '"signature":"c2ln"' in item.opaque_roundtrip
        assert result.usage is not None
        assert (result.usage.input_tokens, result.usage.output_tokens) == (3, 2)

    @pytest.mark.anyio
    async def test_streamed_function_arguments_are_reassembled(self) -> None:
        recorder = _stream_reply(
            {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "function_call", "id": "c1", "name": "get_weather"},
            },
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "arguments_delta", "arguments": '{"city": '},
            },
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "arguments_delta", "arguments": '"Oslo"}'},
            },
            {"event_type": "step.stop", "index": 0},
            _completed("requires_action"),
        )
        result = await _run(
            recorder,
            _request(
                scheduling=SchedulingParams(stream=True),
                tools=ToolParams(functions=(_WEATHER,)),
            ),
        )
        assert result.tool_calls is not None
        assert result.tool_calls[0].arguments == {"city": "Oslo"}

    @pytest.mark.anyio
    async def test_stream_ignores_comments_and_multiline_data(self) -> None:
        payload = json.dumps(_completed(), indent=1).splitlines()
        body = ": keepalive\nid: 7\n" + "".join(f"data: {line}\n" for line in payload)
        recorder = _Recorder(lambda _: httpx.Response(200, content=body.encode()))
        result = await _run(
            recorder, _request(scheduling=SchedulingParams(stream=True))
        )
        assert result.texts == ("",)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("events", "message"),
        [
            ((), "ended before completion"),
            (
                ({"event_type": "error", "error": {"code": "x", "message": "boom"}},),
                "boom",
            ),
            (({"event_type": "mystery"},), "unsupported event"),
            ((_completed(), _completed()), "continued after"),
            (({"event_type": "interaction.completed"},), "carries no interaction"),
            (
                (_completed(errors=[{"message": "bad things"}]),),
                "bad things",
            ),
            (
                ({"event_type": "step.start", "index": -1, "step": {}},),
                "index is invalid",
            ),
            (
                (
                    {"event_type": "step.start", "index": 0, "step": {"type": "x"}},
                    {"event_type": "step.start", "index": 0, "step": {"type": "x"}},
                ),
                "started twice",
            ),
            (
                (
                    {"event_type": "step.start", "index": 1, "step": {"type": "x"}},
                    _completed(),
                ),
                "not contiguous",
            ),
            (({"event_type": "step.stop", "index": 0},), "unknown step"),
            (
                ({"event_type": "step.delta", "index": 0, "delta": {"type": "text"}},),
                "unknown step",
            ),
            (
                (
                    {
                        "event_type": "step.start",
                        "index": 0,
                        "step": {"type": "thought"},
                    },
                    {
                        "event_type": "step.delta",
                        "index": 0,
                        "delta": {"type": "text", "text": "x"},
                    },
                ),
                "does not fit",
            ),
            (
                (
                    {
                        "event_type": "step.start",
                        "index": 0,
                        "step": {"type": "model_output"},
                    },
                    {"event_type": "step.delta", "index": 0, "delta": {"type": "text"}},
                ),
                "must carry text",
            ),
            (
                (
                    {
                        "event_type": "step.start",
                        "index": 0,
                        "step": {"type": "thought"},
                    },
                    {
                        "event_type": "step.delta",
                        "index": 0,
                        "delta": {"type": "thought_signature", "signature": 1},
                    },
                ),
                "signature must be a string",
            ),
            (
                (
                    {
                        "event_type": "step.start",
                        "index": 0,
                        "step": {"type": "function_call", "id": "c", "name": "f"},
                    },
                    {
                        "event_type": "step.delta",
                        "index": 0,
                        "delta": {"type": "arguments_delta", "arguments": 1},
                    },
                ),
                "arguments delta must be a string",
            ),
            (
                (
                    {
                        "event_type": "step.start",
                        "index": 0,
                        "step": {"type": "function_call", "id": "c", "name": "f"},
                    },
                    {
                        "event_type": "step.delta",
                        "index": 0,
                        "delta": {"type": "arguments_delta", "arguments": "[]"},
                    },
                    _completed("requires_action"),
                ),
                "must be an object",
            ),
        ],
    )
    async def test_malformed_streams_fail_loudly(
        self, events: tuple[Mapping[str, object], ...], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                _stream_reply(*events),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (b"garbage\n\n", "malformed SSE"),
            (b"foo: bar\n\n", "unsupported SSE field"),
            (b"data: [1]\n\n", "must be an object"),
        ],
    )
    async def test_malformed_sse_framing(self, body: bytes, message: str) -> None:
        recorder = _Recorder(lambda _: httpx.Response(200, content=body))
        with pytest.raises(OutputContractError, match=message):
            await _run(recorder, _request(scheduling=SchedulingParams(stream=True)))

    @pytest.mark.anyio
    async def test_stream_http_error_carries_the_provider_message(self) -> None:
        recorder = _json_reply({"error": {"message": "slow down"}}, status=429)
        with pytest.raises(httpx.HTTPStatusError, match="slow down"):
            await _run(recorder, _request(scheduling=SchedulingParams(stream=True)))


class TestEnvelopeVerification:
    def _prepared(self, request: Request | None = None) -> PreparedRequest:
        request = request or _request()
        audit = RequestAudit(active_request_leaves(request))
        dialect = _dialect()
        dialect.validate_request(request, audit, cast(Any, None))
        prepared = dialect.prepare(request, audit)
        audit.finish(prepared)
        return prepared

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"store": True}, "store must be false"),
            ({"model": "other"}, "bound model"),
            ({"background": True}, "'background' must not be sent"),
            ({"previous_interaction_id": "i"}, "must not be sent"),
            ({"stream": True}, "stream flag"),
        ],
    )
    @pytest.mark.anyio
    async def test_tampered_body_is_refused_before_sending(
        self, change: dict[str, JSONValue], message: str
    ) -> None:
        prepared = self._prepared()
        tampered = PreparedRequest(
            prepared.operation, {**prepared.thaw_body(), **change}, prepared.context
        )
        with pytest.raises(DialectError, match=message):
            await _dialect().execute(tampered)

    @pytest.mark.anyio
    async def test_wrong_operation_and_context_are_refused(self) -> None:
        prepared = self._prepared()
        with pytest.raises(DialectError, match="operation"):
            await _dialect().execute(
                PreparedRequest("v1beta/other", prepared.body, prepared.context)
            )
        with pytest.raises(DialectError, match="invalid prepared context"):
            await _dialect().execute(
                PreparedRequest(prepared.operation, prepared.body, None)
            )

    def test_audit_catches_a_changed_input(self) -> None:
        request = _request()
        audit = RequestAudit(active_request_leaves(request))
        dialect = _dialect()
        dialect.validate_request(request, audit, cast(Any, None))
        prepared = dialect.prepare(request, audit)
        body = prepared.thaw_body()
        body["input"] = []
        with pytest.raises(RequestAuditError, match="input steps"):
            audit.finish(PreparedRequest(prepared.operation, body, prepared.context))

    @pytest.mark.parametrize(
        ("request_", "mutate", "message"),
        [
            (
                _request(tools=ToolParams(functions=(_WEATHER,))),
                lambda body: body.update(tools=[]),
                "function tools changed",
            ),
            (
                _request(tools=ToolParams(functions=(_WEATHER,), choice="auto")),
                lambda body: body["generation_config"].update(tool_choice="none"),
                "tool choice changed",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=(_WEATHER,),
                        choice={"type": "function", "name": "get_weather"},
                    )
                ),
                lambda body: body["generation_config"].update(tool_choice="any"),
                "tool choice changed",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(format="json_object")
                ),
                lambda body: body.update(response_format={"type": "text"}),
                "format changed",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(format="json_object")
                ),
                lambda body: body["response_format"].update(schema={}),
                "gained a schema",
            ),
            (
                _request(
                    dialect_options=DialectOptions(
                        "google_interactions", {"labels": {}}
                    )
                ),
                lambda body: body.update(labels={"a": "b"}),
                "dialect option 'labels' changed",
            ),
            (
                _request(
                    ChatMessage("system", (TextPart("s"),)),
                    ChatMessage("user", (TextPart("u"),)),
                ),
                lambda body: body.update(system_instruction="t"),
                "system_instruction",
            ),
        ],
    )
    def test_audit_catches_changed_lowering(
        self, request_: Request, mutate: Callable[[dict[str, Any]], None], message: str
    ) -> None:
        audit = RequestAudit(active_request_leaves(request_))
        dialect = _dialect()
        dialect.validate_request(request_, audit, cast(Any, None))
        prepared = dialect.prepare(request_, audit)
        body = prepared.thaw_body()
        mutate(body)
        with pytest.raises(RequestAuditError, match=message):
            audit.finish(PreparedRequest(prepared.operation, body, prepared.context))

    def test_prepare_refuses_completion_input_even_without_validation(self) -> None:
        request = Request(CompletionInput("x"))
        audit = RequestAudit(active_request_leaves(request))
        with pytest.raises(DialectError, match="requires ChatInput"):
            _dialect().prepare(request, audit)
