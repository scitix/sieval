"""Contract tests for the OpenAI Chat Completions dialect.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sieval.core.models.capabilities import Capability, ReasoningOptions, Supported
from sieval.core.models.dialect import (
    DialectError,
    OutputContractError,
    PreparedRequest,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
)
from sieval.core.models.dialects.openai_chat import (
    CAPABILITY_DECISIONS,
    OpenAIChatDialect,
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
    SessionParams,
    StructuredOutputParams,
    TextPart,
    TokenLogprob,
    ToolCallPart,
    ToolParams,
    ToolResultPart,
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


def _choice(
    index: object,
    content: str | None,
    *,
    finish_reason: str | None = "stop",
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[object] | None = None,
    logprobs: object | None = None,
) -> object:
    message = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        index=index,
        message=message,
        finish_reason=finish_reason,
        logprobs=logprobs,
    )


def _response(
    *choices: object,
    model: object | None = None,
    usage: object | None = None,
    fingerprint: object | None = None,
) -> object:
    if not choices:
        choices = (_choice(0, "ok"),)
    return SimpleNamespace(
        choices=list(choices),
        usage=usage,
        model=model,
        system_fingerprint=fingerprint,
    )


def _usage(prompt: object, completion: object, total: object) -> object:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _dialect(
    response: object | None = None,
    *,
    requested_model_id: str = "requested/model",
) -> tuple[OpenAIChatDialect, AsyncMock]:
    client = MagicMock()
    create = AsyncMock(return_value=_response() if response is None else response)
    client.chat.completions.create = create
    return OpenAIChatDialect(client, requested_model_id), create


def _logprob_content() -> object:
    top = SimpleNamespace(token="B", logprob=-2.0)
    item = SimpleNamespace(token="A", logprob=-0.1, top_logprobs=[top])
    return SimpleNamespace(content=[item])


def _awaited_kwargs(create: AsyncMock) -> dict[str, Any]:
    create.assert_awaited()
    call = create.await_args
    assert call is not None
    return dict(call.kwargs)


class TestWireTranslation:
    @pytest.mark.anyio
    async def test_complete_request_preserves_wire_shape(self) -> None:
        dialect, create = _dialect(
            _response(_choice(0, '{"answer":42}', logprobs=_logprob_content()))
        )
        function: Mapping[str, JSONValue] = {
            "type": "function",
            "function": {"name": "weather", "parameters": {"type": "object"}},
        }
        request = Request(
            input=_chat(ChatMessage("user", (TextPart("question"),))),
            sampling=SamplingParams(
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                stop=("END",),
                seed=7,
                frequency_penalty=0.1,
                presence_penalty=0.2,
            ),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
            reasoning=ReasoningParams(effort="high"),
            tools=ToolParams(
                functions=(function,),
                choice={"type": "function", "function": {"name": "weather"}},
                parallel=False,
            ),
            structured_output=StructuredOutputParams(
                format="json_schema",
                schema={"type": "object"},
                name="answer",
                strict=True,
            ),
            dialect_options=DialectOptions(
                "openai_chat",
                {"stop_token_ids": [1, 2], "min_p": 0.05},
            ),
        )

        response = await dialect.arun(request)

        create.assert_awaited_once_with(
            model="requested/model",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=64,
            temperature=0.2,
            top_p=0.9,
            stop=["END"],
            seed=7,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            logprobs=True,
            top_logprobs=2,
            reasoning_effort="high",
            tools=[function],
            tool_choice={"type": "function", "function": {"name": "weather"}},
            parallel_tool_calls=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            stream=False,
            extra_body={
                "top_k": 40,
                "stop_token_ids": [1, 2],
                "min_p": 0.05,
            },
        )
        assert response.request_params == {
            key: value
            for key, value in _awaited_kwargs(create).items()
            if key not in {"model", "messages"}
        }
        assert response.structured_output is not None
        assert response.structured_output.value == {"answer": 42}

    @pytest.mark.anyio
    async def test_empty_matching_dialect_options_are_an_explicit_noop(self) -> None:
        dialect, create = _dialect()

        await dialect.arun(
            Request(
                input=_chat(),
                dialect_options=DialectOptions("openai_chat", {}),
            )
        )

        assert "extra_body" not in _awaited_kwargs(create)

    @pytest.mark.anyio
    async def test_reasoning_summary_none_is_an_explicit_noop(self) -> None:
        dialect, create = _dialect()

        response = await dialect.arun(
            Request(input=_chat(), reasoning=ReasoningParams(summary="none"))
        )

        assert "reasoning" not in _awaited_kwargs(create)
        assert "reasoning_summary" not in _awaited_kwargs(create)
        assert response.reasoning is None

    @pytest.mark.anyio
    async def test_image_tool_call_and_tool_result_messages(self) -> None:
        dialect, create = _dialect()
        request = Request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        TextPart("look"),
                        ImagePart(url="https://example.test/image.png", detail="high"),
                    ),
                ),
                ChatMessage(
                    "assistant",
                    (
                        TextPart("checking"),
                        ToolCallPart("call_1", "inspect", {"z": 2, "a": 1}),
                    ),
                ),
                ChatMessage(
                    "tool",
                    (ToolResultPart("call_1", {"ok": True, "score": 3}),),
                ),
            )
        )

        await dialect.arun(request)

        assert _awaited_kwargs(create)["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.test/image.png",
                            "detail": "high",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": '{"a":1,"z":2}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok":true,"score":3}',
            },
        ]

    @pytest.mark.anyio
    async def test_tool_result_error_marker_is_rejected_before_io(self) -> None:
        dialect, create = _dialect()
        req = Request(
            input=_chat(
                ChatMessage(
                    "tool",
                    (ToolResultPart("call_1", "boom", is_error=True),),
                )
            )
        )

        with pytest.raises(
            RequestAuditError, match="cannot transmit the tool-result error state"
        ):
            await dialect.arun(req)

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_inline_image_data_is_emitted_as_base64_url(self) -> None:
        dialect, create = _dialect()

        await dialect.arun(
            Request(
                input=_chat(
                    ChatMessage(
                        "user",
                        (ImagePart(data="YWJj", media_type="image/png"),),
                    )
                )
            )
        )

        assert _awaited_kwargs(create)["messages"][0]["content"] == [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,YWJj"},
            }
        ]

    @pytest.mark.anyio
    async def test_url_image_media_type_is_rejected_before_io(self) -> None:
        dialect, create = _dialect()
        req = Request(
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
        )

        with pytest.raises(
            RequestAuditError, match="no media-type field for URL-backed images"
        ):
            await dialect.arun(req)

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_mixed_images_reject_url_media_type_before_io(self) -> None:
        dialect, create = _dialect()
        req = Request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        ImagePart(data="YWJj", media_type="image/png"),
                        ImagePart(
                            url="https://example.test/image.jpeg",
                            media_type="image/jpeg",
                        ),
                    ),
                )
            )
        )

        with pytest.raises(RequestAuditError, match="URL-backed images"):
            await dialect.arun(req)

        create.assert_not_awaited()


class TestPreflightRejections:
    def test_legacy_capability_view_matches_supported_reasoning_controls(self) -> None:
        dialect, _ = _dialect()

        assert Capability.Reasoning in dialect.capabilities
        assert Capability.ReasoningEffort in dialect.capabilities

    def test_reasoning_config_rejects_non_noop_summary(self) -> None:
        decision = CAPABILITY_DECISIONS["reasoning"]
        assert isinstance(decision, Supported)

        decision.binding.validate_config(ReasoningOptions(summary="none"))
        with pytest.raises(ValueError, match="only summary='none'"):
            decision.binding.validate_config(ReasoningOptions(summary="auto"))

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (ReasoningOptions(budget_tokens=32), "budget_tokens"),
            (ReasoningOptions(effort="ultra"), "effort"),
        ],
    )
    def test_reasoning_config_rejects_unsupported_controls(
        self, options: ReasoningOptions, message: str
    ) -> None:
        decision = CAPABILITY_DECISIONS["reasoning"]
        assert isinstance(decision, Supported)

        with pytest.raises(ValueError, match=message):
            decision.binding.validate_config(options)

    @pytest.mark.parametrize(
        "req",
        [
            Request(
                input=_chat(),
                tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
            ),
            Request(
                input=_chat(),
                session=SessionParams(previous_response_id="response-1"),
            ),
        ],
    )
    def test_dialect_rejection_is_recorded_before_prepare(self, req: Request) -> None:
        dialect, create = _dialect()
        audit = RequestAudit(active_request_leaves(req))

        dialect.validate_request(req, audit, SimpleNamespace())
        with pytest.raises(RequestAuditError):
            audit.raise_rejections()

        create.assert_not_awaited()

    @pytest.mark.parametrize("key", ["prefill", "prefix"])
    def test_prefill_option_reports_actual_protocol_limitation(self, key: str) -> None:
        req = Request(
            input=_chat(),
            dialect_options=DialectOptions("openai_chat", {key: "assistant text"}),
        )
        dialect, create = _dialect()
        audit = RequestAudit(active_request_leaves(req))

        dialect.validate_request(req, audit, SimpleNamespace())
        with pytest.raises(RequestAuditError, match="no assistant-prefill field"):
            audit.raise_rejections()

        create.assert_not_awaited()

    def test_prepare_rejects_completion_input_defensively(self) -> None:
        dialect, _ = _dialect()
        req = Request(input=CompletionInput("prompt"))

        with pytest.raises(TypeError, match="requires ChatInput"):
            dialect.prepare(req, RequestAudit(active_request_leaves(req)))

    @pytest.mark.anyio
    async def test_execute_rejects_invalid_prepared_context(self) -> None:
        dialect, create = _dialect()
        prepared = PreparedRequest(
            operation="chat.completions.create",
            body={"stream": False},
            consumed_paths=frozenset(),
            passthrough={},
        )

        with pytest.raises(TypeError, match="invalid prepared context"):
            await dialect.execute(prepared)

        create.assert_not_awaited()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "model_request",
        [
            Request(input=CompletionInput("prompt")),
            Request(input=_chat(), scoring=ScoringParams(input_scoring=True)),
            Request(input=_chat(), reasoning=ReasoningParams(budget_tokens=32)),
            Request(input=_chat(), reasoning=ReasoningParams(effort="banana")),
            Request(input=_chat(), reasoning=ReasoningParams(summary="auto")),
            Request(
                input=_chat(),
                tools=ToolParams(hosted=(HostedToolSpec("web_search"),)),
            ),
            Request(
                input=_chat(),
                session=SessionParams(previous_response_id="response-1"),
            ),
            Request(
                input=_chat(),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("openai_chat", "opaque")
                ),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions("openai_chat", {"model": "other"}),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions(
                    "openai_chat", {"authorization": "Bearer secret"}
                ),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions("openai_chat", {"top_k": 99}),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions(
                    "openai_chat", {"max_completion_tokens": 99}
                ),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions(
                    "openai_chat", {"server_tools": [{"type": "web_search"}]}
                ),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions("openai_chat", {"functions": []}),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions(
                    "openai_chat", {"function_call": "auto"}
                ),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions(
                    "openai_chat", {"web_search_options": {}}
                ),
            ),
            Request(
                input=_chat(),
                dialect_options=DialectOptions("another_dialect", {"min_p": 0.1}),
            ),
            Request(
                input=_chat(),
                scoring=ScoringParams(sampled_logprobs=False, top_logprobs=1),
            ),
            Request(
                input=_chat(),
                sampling=SamplingParams(n=2),
                structured_output=StructuredOutputParams(format="json_object"),
            ),
            Request(
                input=_chat(),
                structured_output=StructuredOutputParams(
                    format="json_object", schema={"type": "object"}
                ),
            ),
        ],
        ids=(
            "completion-input",
            "input-scoring",
            "reasoning-budget",
            "reasoning-effort-domain",
            "reasoning-summary-not-noop",
            "hosted-tool",
            "previous-response",
            "opaque-continuation",
            "canonical-model-option",
            "binding-resource-option",
            "canonical-top-k-option",
            "legacy-max-completion-tokens-option",
            "canonical-hosted-tools-option",
            "legacy-functions-option",
            "legacy-function-call-option",
            "hosted-web-search-options-option",
            "wrong-dialect-options",
            "top-logprobs-without-sampled",
            "n-with-singular-output",
            "json-object-with-schema",
        ),
    )
    async def test_every_request_rejection_happens_before_io(
        self, model_request: Request
    ) -> None:
        dialect, create = _dialect()

        with pytest.raises((DialectError, RequestAuditError, TypeError, ValueError)):
            await dialect.arun(model_request)

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_invalid_tool_result_message_fails_before_io(self) -> None:
        dialect, create = _dialect()
        request = Request(
            input=_chat(
                ChatMessage(
                    "tool",
                    (ToolResultPart("call", "result"), TextPart("extra")),
                )
            )
        )

        with pytest.raises(ValueError, match="exactly one result"):
            await dialect.arun(request)

        create.assert_not_awaited()


class TestResponseLifting:
    @pytest.mark.anyio
    async def test_valid_usage_is_lifted(self) -> None:
        dialect, _ = _dialect(_response(_choice(0, "ok"), usage=_usage(2, 3, 5)))

        response = await dialect.arun(Request(input=_chat()))

        assert response.usage == UsageStats(
            input_tokens=2, output_tokens=3, total_tokens=5
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("choices", [None, "not-a-sequence"])
    async def test_response_choices_must_be_a_complete_sequence(
        self, choices: object
    ) -> None:
        raw = SimpleNamespace(
            choices=choices,
            usage=None,
            model=None,
            system_fingerprint=None,
        )
        dialect, _ = _dialect(raw)

        with pytest.raises(OutputContractError, match="choices|omitted"):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    async def test_mapping_tool_call_with_absent_arguments_is_lifted(self) -> None:
        raw_call = {"id": "call_1", "function": {"name": "weather"}}
        dialect, _ = _dialect(_response(_choice(0, "", tool_calls=[raw_call])))

        response = await dialect.arun(Request(input=_chat()))

        assert response.tool_calls == (FunctionToolCall("call_1", "weather", ""),)

    @pytest.mark.anyio
    async def test_named_tool_call_message_without_inline_content_is_lowered(
        self,
    ) -> None:
        dialect, create = _dialect()
        request = Request(
            input=_chat(
                ChatMessage(
                    "assistant",
                    (ToolCallPart("call_1", "weather", {"city": "Paris"}),),
                    name="assistant-name",
                )
            )
        )

        await dialect.arun(request)

        message = _awaited_kwargs(create)["messages"][0]
        assert message["name"] == "assistant-name"
        assert message["content"] is None

    @pytest.mark.anyio
    async def test_json_schema_strict_flag_is_lowered(self) -> None:
        dialect, create = _dialect(_response(_choice(0, '{"answer":42}')))

        await dialect.arun(
            Request(
                input=_chat(),
                structured_output=StructuredOutputParams(
                    format="json_schema",
                    schema={"type": "object"},
                    name="answer",
                    strict=True,
                ),
            )
        )

        assert _awaited_kwargs(create)["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        }

    @pytest.mark.anyio
    async def test_stream_rejects_negative_tool_index(self) -> None:
        call = SimpleNamespace(
            index=-1,
            id="call_1",
            function=SimpleNamespace(name="weather", arguments="{}"),
        )
        dialect, _ = _dialect(_AsyncItems([_chunk(tool_calls=[call])]))

        with pytest.raises(OutputContractError, match="non-negative integer"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    tools=ToolParams(
                        functions=(
                            {"type": "function", "function": {"name": "weather"}},
                        )
                    ),
                    scheduling=SchedulingParams(stream=True),
                )
            )

    @pytest.mark.anyio
    async def test_stream_accepts_empty_delta_and_final_usage(self) -> None:
        choice = SimpleNamespace(
            index=0,
            finish_reason="stop",
            delta=None,
            logprobs=None,
        )
        chunk = SimpleNamespace(
            choices=[choice],
            usage=_usage(1, 0, 1),
            model="provider/model",
            system_fingerprint="fp_1",
        )
        dialect, _ = _dialect(_AsyncItems([chunk]))

        response = await dialect.arun(
            Request(input=_chat(), scheduling=SchedulingParams(stream=True))
        )

        assert response.usage == UsageStats(
            input_tokens=1, output_tokens=0, total_tokens=1
        )
        assert response.response_model == "provider/model"
        assert response.system_fingerprint == "fp_1"

    @pytest.mark.anyio
    @pytest.mark.parametrize("index", ["0", True, -1, 1])
    async def test_non_stream_invalid_choice_index_is_a_contract_error(
        self, index: object
    ) -> None:
        dialect, _ = _dialect(_response(_choice(index, "invalid")))

        with pytest.raises(OutputContractError, match="choice index"):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    async def test_non_stream_duplicate_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(_response(_choice(0, "first"), _choice(0, "duplicate")))

        with pytest.raises(OutputContractError, match="duplicated choice index 0"):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    async def test_non_stream_missing_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(_response(_choice(0, "first")))

        with pytest.raises(OutputContractError, match=r"omitted choice indexes \[1\]"):
            await dialect.arun(Request(input=_chat(), sampling=SamplingParams(n=2)))

    @pytest.mark.anyio
    async def test_stream_invalid_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(_AsyncItems([_chunk(index="0", content="bad")]))

        with pytest.raises(
            OutputContractError, match="choice index must be an integer"
        ):
            await dialect.arun(
                Request(input=_chat(), scheduling=SchedulingParams(stream=True))
            )

    @pytest.mark.anyio
    async def test_stream_missing_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(_AsyncItems([_chunk(index=0, content="first")]))

        with pytest.raises(OutputContractError, match=r"omitted choice indexes \[1\]"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    sampling=SamplingParams(n=2),
                    scheduling=SchedulingParams(stream=True),
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "usage",
        [
            _usage(-1, 1, 0),
            _usage(1, True, 2),
            _usage(1, 2.0, 3),
        ],
    )
    async def test_usage_fields_must_be_non_negative_integers(
        self, usage: object
    ) -> None:
        dialect, _ = _dialect(_response(_choice(0, "ok"), usage=usage))

        with pytest.raises(OutputContractError, match="non-negative integer"):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    async def test_reported_total_is_recorded_not_trusted(self) -> None:
        """A total that does not decompose is evidence, not grounds to reject.

        The reply is kept and scored on the computed total; the server's own
        figure survives beside it as the only trace that the two disagreed.
        """
        dialect, _ = _dialect(_response(_choice(0, "ok"), usage=_usage(2, 3, 99)))

        response = await dialect.arun(Request(input=_chat()))

        assert response.usage == UsageStats(
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
            reported_total_tokens=99,
        )

    @pytest.mark.anyio
    async def test_agreeing_reported_total_is_not_recorded(self) -> None:
        """Agreement carries no information, so storing it would bury the rest."""
        dialect, _ = _dialect(_response(_choice(0, "ok"), usage=_usage(2, 3, 5)))

        response = await dialect.arun(Request(input=_chat()))

        assert response.usage is not None
        assert response.usage.reported_total_tokens is None

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("logprobs", "message"),
        [
            (SimpleNamespace(content=None), "content must be a sequence"),
            (SimpleNamespace(content="A"), "content must be a sequence"),
            (
                SimpleNamespace(
                    content=[SimpleNamespace(token=1, logprob=-0.1, top_logprobs=[])]
                ),
                "token must be a string",
            ),
            (
                SimpleNamespace(
                    content=[SimpleNamespace(token="A", logprob=True, top_logprobs=[])]
                ),
                "logprob must be numeric",
            ),
            (
                SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token="A", logprob=float("nan"), top_logprobs=[]
                        )
                    ]
                ),
                "logprob must be finite",
            ),
            (
                SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token="A", logprob=-0.1, top_logprobs={"B": -2.0}
                        )
                    ]
                ),
                "top_logprobs must be a sequence",
            ),
            (
                SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token="A",
                            logprob=-0.1,
                            top_logprobs=[SimpleNamespace(token=1, logprob=-2.0)],
                        )
                    ]
                ),
                "token must be a string",
            ),
            (
                SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token="A",
                            logprob=-0.1,
                            top_logprobs=[
                                SimpleNamespace(token="B", logprob=float("inf"))
                            ],
                        )
                    ]
                ),
                "logprob must be finite",
            ),
        ],
    )
    async def test_malformed_logprob_payloads_fail_loudly(
        self, logprobs: object, message: str
    ) -> None:
        dialect, _ = _dialect(_response(_choice(0, "A", logprobs=logprobs)))

        with pytest.raises(OutputContractError, match=message):
            await dialect.arun(
                Request(
                    input=_chat(),
                    scoring=ScoringParams(sampled_logprobs=True),
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("field", "value"),
        [("model", 7), ("system_fingerprint", False)],
    )
    async def test_malformed_response_identity_fails_loudly(
        self, field: str, value: object
    ) -> None:
        response_kwargs = (
            {"model": value} if field == "model" else {"fingerprint": value}
        )
        dialect, _ = _dialect(_response(_choice(0, "ok"), **response_kwargs))

        with pytest.raises(OutputContractError, match=field):
            await dialect.arun(Request(input=_chat()))

    @pytest.mark.anyio
    async def test_reasoning_remains_choice_indexed_for_n(self) -> None:
        raw = _response(
            _choice(1, "second", reasoning_content="reason-2"),
            _choice(0, "first", reasoning="reason-1"),
        )
        dialect, create = _dialect(raw)

        response = await dialect.arun(
            Request(
                input=_chat(),
                sampling=SamplingParams(n=2),
                reasoning=ReasoningParams(effort="high"),
            )
        )

        assert _awaited_kwargs(create)["n"] == 2
        assert response.texts == ("first", "second")
        assert response.reasoning == (
            ReasoningOutput(text="reason-1"),
            ReasoningOutput(text="reason-2"),
        )

    @pytest.mark.anyio
    async def test_streamed_function_call_fragments_are_reconstructed(self) -> None:
        first_call = SimpleNamespace(
            index=0,
            id="call_",
            function=SimpleNamespace(name="wea", arguments='{"ci'),
        )
        second_call = SimpleNamespace(
            index=0,
            id="1",
            function=SimpleNamespace(name="ther", arguments='ty":"Paris"}'),
        )
        chunks = _AsyncItems(
            [
                _chunk(tool_calls=[first_call], model="provider/model"),
                _chunk(tool_calls=[second_call], finish_reason="tool_calls"),
            ]
        )
        dialect, _ = _dialect(chunks)

        response = await dialect.arun(
            Request(
                input=_chat(),
                tools=ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "function": {"name": "weather"},
                        },
                    )
                ),
                scheduling=SchedulingParams(stream=True),
            )
        )

        assert response.tool_calls == (
            FunctionToolCall("call_1", "weather", '{"city":"Paris"}'),
        )
        assert response.response_model == "provider/model"

    @pytest.mark.anyio
    async def test_tool_index_does_not_shadow_choice_index_for_stream_logprobs(
        self,
    ) -> None:
        call = SimpleNamespace(
            index=1,
            id="call_2",
            function=SimpleNamespace(name="weather", arguments="{}"),
        )
        dialect, _ = _dialect(
            _AsyncItems([_chunk(tool_calls=[call], logprobs=_logprob_content())])
        )

        response = await dialect.arun(
            Request(
                input=_chat(),
                scoring=ScoringParams(sampled_logprobs=True),
                tools=ToolParams(
                    functions=({"type": "function", "function": {"name": "weather"}},)
                ),
                scheduling=SchedulingParams(stream=True),
            )
        )

        assert response.logprobs == (TokenLogprob("A", -0.1),)
        assert response.tool_calls == (FunctionToolCall("call_2", "weather", "{}"),)

    @pytest.mark.anyio
    async def test_incomplete_streamed_function_call_is_a_contract_error(self) -> None:
        fragment = SimpleNamespace(
            index=0,
            id="call_1",
            function=SimpleNamespace(name=None, arguments="{}"),
        )
        dialect, _ = _dialect(
            _AsyncItems([_chunk(tool_calls=[fragment], finish_reason="tool_calls")])
        )

        with pytest.raises(OutputContractError, match="missing id or function name"):
            await dialect.arun(
                Request(input=_chat(), scheduling=SchedulingParams(stream=True))
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("provider_model", "expected"),
        [("provider/model-v2", "provider/model-v2"), (None, None)],
    )
    async def test_provider_model_identity_is_observed_not_forged(
        self, provider_model: str | None, expected: str | None
    ) -> None:
        dialect, _ = _dialect(
            _response(_choice(0, "ok"), model=provider_model),
            requested_model_id="requested/model",
        )

        response = await dialect.arun(Request(input=_chat()))

        assert response.response_model == expected
        if expected is not None:
            assert response.response_model != "requested/model"

    @pytest.mark.anyio
    async def test_invalid_structured_json_is_a_contract_error_after_io(self) -> None:
        dialect, create = _dialect(_response(_choice(0, "not json")))

        with pytest.raises(OutputContractError, match="not valid JSON"):
            await dialect.arun(
                Request(
                    input=_chat(),
                    structured_output=StructuredOutputParams(format="json_object"),
                )
            )

        create.assert_awaited_once()


def _chunk(
    *,
    index: object = 0,
    content: str | None = None,
    finish_reason: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[object] | None = None,
    logprobs: object | None = None,
    model: object | None = None,
    fingerprint: object | None = None,
) -> object:
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(
        index=index,
        finish_reason=finish_reason,
        delta=delta,
        logprobs=logprobs,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=None,
        model=model,
        system_fingerprint=fingerprint,
    )
