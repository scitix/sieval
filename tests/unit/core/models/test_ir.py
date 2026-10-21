"""Unit tests for the provider-neutral model Request/Response records.

AI-Generated Code - GPT-5.4 (OpenAI)
"""

import dataclasses

import pytest

from sieval.core.models.ir import (
    CapabilityEvidence,
    ChatInput,
    ChatMessage,
    ChoiceLogprobs,
    Citation,
    CompletionInput,
    DialectOptions,
    GroundingChunk,
    GroundingMetadata,
    ImagePart,
    InputScoringResult,
    ModelIdentity,
    ModelProvenance,
    OpaqueContinuation,
    ReasoningOutput,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    ServerToolUse,
    SessionParams,
    StructuredOutput,
    StructuredOutputParams,
    TextPart,
    TokenLogprob,
    ToolCallPart,
    ToolResultPart,
    TopKEntry,
    UsageStats,
    normalize_chat_input,
    response_field_contract,
)
from sieval.core.utils.serialization import (
    dict_to_obj,
    global_type_registry,
    obj_to_dict,
)


class TestRequestConstruction:
    def test_minimal_completion_request_uses_group_defaults(self):
        req = Request(input=CompletionInput("hello"))
        assert req.input == CompletionInput("hello")
        assert req.sampling == SamplingParams()
        assert req.scoring.input_scoring is False
        assert req.reasoning == ReasoningParams()
        assert req.dialect_options is None

    def test_chat_input_is_explicitly_tagged(self):
        req = Request(
            input=ChatInput(
                (ChatMessage("user", (TextPart("hi"), ImagePart(url="x"))),)
            )
        )
        assert isinstance(req.input, ChatInput)
        assert isinstance(req.input.messages[0].content[1], ImagePart)

    def test_legacy_openai_messages_normalize_once(self):
        chat = normalize_chat_input(
            [
                {"role": "system", "content": "rules"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "https://x"}},
                    ],
                },
            ]
        )
        assert chat.messages[0].content == (TextPart("rules"),)
        assert chat.messages[1].content[1] == ImagePart(url="https://x")

    def test_legacy_inline_image_preserves_media_type_and_base64_payload(self):
        chat = normalize_chat_input(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,YWJj"},
                        }
                    ],
                }
            ]
        )

        assert chat.messages[0].content == (
            ImagePart(data="YWJj", media_type="image/png"),
        )

    def test_legacy_tool_calls_reject_non_mapping_items(self):
        with pytest.raises(TypeError, match=r"tool_calls\[1\] must be a mapping"):
            normalize_chat_input(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "lookup", "arguments": "{}"},
                            },
                            "silently-dropped-before",
                        ],
                    }
                ]
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_legacy_json_values_reject_non_finite_floats(self, value):
        with pytest.raises(ValueError, match="non-finite"):
            normalize_chat_input(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": {"value": value},
                                },
                            }
                        ],
                    }
                ]
            )

    def test_completion_suffix_is_part_of_completion_modality(self):
        req = Request(input=CompletionInput("prefix", suffix="suffix"))
        assert isinstance(req.input, CompletionInput)
        assert req.input.suffix == "suffix"

    def test_dialect_options_are_tagged_and_copied(self):
        raw = {"min_p": 0.1}
        options = DialectOptions("openai_completions", raw)
        raw["min_p"] = 0.9
        assert options.values == {"min_p": 0.1}

    def test_reasoning_choices_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ReasoningParams(effort="high", budget_tokens=128)

    def test_opaque_continuation_carries_originating_dialect(self):
        session = SessionParams(
            opaque_continuation=OpaqueContinuation("openai_responses", "opaque")
        )
        assert session.opaque_continuation is not None
        assert session.opaque_continuation.dialect_id == "openai_responses"

    @pytest.mark.parametrize(
        ("dialect_id", "value", "message"),
        [
            ("", "opaque", "dialect_id"),
            ("openai_responses", "", "value"),
        ],
    )
    def test_opaque_continuation_rejects_empty_identity_or_payload(
        self, dialect_id, value, message
    ):
        with pytest.raises(ValueError, match=message):
            OpaqueContinuation(dialect_id, value)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"url": "https://image", "data": "YWJj"},
        ],
    )
    def test_image_part_requires_exactly_one_payload(self, kwargs):
        with pytest.raises(ValueError, match="exactly one"):
            ImagePart(**kwargs)

    def test_chat_input_and_dialect_options_require_nonempty_identity(self):
        with pytest.raises(ValueError, match="messages"):
            ChatInput(())
        with pytest.raises(ValueError, match="dialect_id"):
            DialectOptions("", {})

    def test_legacy_plain_string_content_part_is_normalized(self):
        chat = normalize_chat_input([{"role": "user", "content": ["plain text"]}])

        assert chat.messages[0].content == (TextPart("plain text"),)

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ([7], "content part must be a mapping"),
            ([{"type": "text", "text": 7}], "string `text`"),
            ([{"type": "image_url"}], "string URL or data"),
            (
                [{"type": "image_url", "image_url": "data:image/png,raw"}],
                "base64 encoding",
            ),
            ([{"type": "function_call", "name": "tool"}], "string id and name"),
            (
                [{"type": "tool_result", "content": "result"}],
                "string tool_call_id",
            ),
            ([{"type": "unknown"}], "unsupported chat content part"),
        ],
    )
    def test_legacy_content_parts_fail_loudly(self, content, message):
        with pytest.raises((TypeError, ValueError), match=message):
            normalize_chat_input([{"role": "user", "content": content}])

    def test_flattened_tool_call_and_result_parts_are_normalized(self):
        chat = normalize_chat_input(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "lookup",
                            "arguments": ("x", 1),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "function_result",
                            "call_id": "call-1",
                            "result": {"ok": True},
                            "is_error": True,
                        }
                    ],
                },
            ]
        )

        assert chat.messages[0].content == (ToolCallPart("call-1", "lookup", ["x", 1]),)
        assert chat.messages[1].content == (
            ToolResultPart("call-1", {"ok": True}, True),
        )

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [({1: "bad"}, "keys must be strings"), (object(), "JSON-compatible")],
    )
    def test_legacy_tool_arguments_must_be_json_compatible(self, arguments, message):
        with pytest.raises(TypeError, match=message):
            normalize_chat_input(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                ]
            )

    @pytest.mark.parametrize(
        ("message", "error"),
        [
            (7, "chat message must be a mapping"),
            ({"role": "invalid", "content": "x"}, "unsupported chat role"),
            ({"role": "user", "content": {"text": "x"}}, "string or iterable"),
            ({"role": "assistant", "tool_calls": "invalid"}, "must be an iterable"),
        ],
    )
    def test_legacy_messages_reject_invalid_shapes(self, message, error):
        with pytest.raises((TypeError, ValueError), match=error):
            normalize_chat_input([message])

    def test_existing_chat_message_and_tool_role_are_preserved(self):
        existing = ChatMessage("user", (TextPart("existing"),))
        chat = normalize_chat_input(
            [
                existing,
                {
                    "role": "tool",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                    "tool_call_id": "call-1",
                },
            ]
        )

        assert chat.messages[0] is existing
        assert chat.messages[1].content == (
            ToolResultPart(
                "call-1",
                "[{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]",
            ),
        )

    def test_reasoning_budget_and_json_schema_are_validated(self):
        with pytest.raises(ValueError, match=">= 1"):
            ReasoningParams(budget_tokens=0)
        with pytest.raises(ValueError, match="requires a schema"):
            StructuredOutputParams(format="json_schema")

    @pytest.mark.parametrize(
        ("dialect_id", "value", "message"),
        [
            (7, "opaque", "dialect_id"),
            ("openai_responses", 7, "value"),
        ],
    )
    def test_opaque_continuation_rejects_non_string_values(
        self, dialect_id, value, message
    ):
        with pytest.raises(ValueError, match=message):
            OpaqueContinuation(dialect_id, value)


class TestResponseContract:
    def test_every_root_field_declares_role_and_cardinality(self):
        contract = response_field_contract()
        assert set(contract) == {f.name for f in dataclasses.fields(Response)}
        assert contract["reasoning"] == ("channel", True)
        assert contract["input_scoring"] == ("channel", False)
        assert contract["provenance"] == ("provenance", False)
        assert contract["finish_reason_sources"] == ("provenance", True)

    def test_reasoning_must_align_with_choices(self):
        with pytest.raises(ValueError, match="reasoning must align"):
            Response(texts=("a", "b"), reasoning=(ReasoningOutput(text="one"),))

    def test_choice_logprobs_must_align_with_choices(self):
        with pytest.raises(ValueError, match="choice_logprobs must align"):
            Response(texts=("a", "b"), choice_logprobs=(ChoiceLogprobs(()),))
        assert response_field_contract()["choice_logprobs"] == ("channel", True)

    def test_finish_reasons_must_align_with_choices(self):
        with pytest.raises(ValueError, match="finish_reasons must align"):
            Response(texts=("a", "b"), finish_reasons=("stop",))

    def test_finish_reason_sources_must_align_with_reasons(self):
        with pytest.raises(ValueError, match="finish_reason_sources must align"):
            Response(texts=("a",), finish_reason_sources=("candidate.finishReason",))
        with pytest.raises(ValueError, match="finish_reason_sources must align"):
            Response(
                texts=("a", "b"),
                finish_reasons=("SAFETY", "STOP"),
                finish_reason_sources=("candidate.finishReason",),
            )

    def test_optional_usage_absence_differs_from_zero_usage(self):
        assert Response(texts=("x",)).usage is None
        assert UsageStats().total_tokens == 0

    def test_grounding_rendered_content_remains_a_real_field(self):
        grounding = GroundingMetadata((), rendered_content="<div/>")
        assert grounding.rendered_content == "<div/>"

    def test_response_detaches_request_params(self):
        params = {"temperature": 0.2}
        response = Response(texts=("x",), request_params=params)
        params["temperature"] = 0.9

        assert response.request_params == {"temperature": 0.2}

    def test_structured_output_schema_is_detached(self):
        schema = {"type": "object"}
        params = StructuredOutputParams(format="json_schema", schema=schema)
        schema["type"] = "array"

        assert params.schema == {"type": "object"}


class TestImmutability:
    @pytest.mark.parametrize(
        ("obj", "field_name", "value"),
        [
            (Request(input=CompletionInput("x")), "input", CompletionInput("y")),
            (Response(texts=("a",)), "texts", ("b",)),
            (SamplingParams(), "temperature", 0.5),
            (ReasoningParams(), "effort", "high"),
            (UsageStats(), "input_tokens", 5),
        ],
    )
    def test_frozen_assignment_raises(self, obj, field_name, value):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field_name, value)


class TestSerializationRoundTrip:
    def _round_trip(self, response: Response) -> Response:
        back = dict_to_obj(obj_to_dict(response, add_type=True), global_type_registry)
        assert isinstance(back, Response)
        return back

    def test_finish_reason_source_survives_reload_and_old_records_default(self):
        response = Response(
            texts=("",),
            finish_reasons=("SAFETY",),
            finish_reason_sources=("promptFeedback.blockReason",),
        )
        persisted = obj_to_dict(response, add_type=True)
        assert self._round_trip(response).finish_reason_sources == (
            "promptFeedback.blockReason",
        )
        persisted.pop("finish_reason_sources")
        restored = dict_to_obj(persisted, global_type_registry)
        assert isinstance(restored, Response)
        assert restored.finish_reason_sources is None

    def test_choice_indexed_reasoning_and_usage_rehydrate(self):
        back = self._round_trip(
            Response(
                texts=("answer",),
                reasoning=(ReasoningOutput(text="think", thinking_tokens=12),),
                usage=UsageStats(input_tokens=3, output_tokens=4, total_tokens=7),
            )
        )
        assert back.reasoning is not None
        reasoning = back.reasoning[0]
        assert isinstance(reasoning, ReasoningOutput)
        assert reasoning.thinking_tokens == 12
        assert isinstance(back.usage, UsageStats)

    def test_scoring_records_rehydrate(self):
        back = self._round_trip(
            Response(
                texts=("x",),
                input_scoring=InputScoringResult(
                    (TokenLogprob(token="q", logprob=-0.2, token_id=5),)
                ),
                logprobs=(TokenLogprob(token="a", logprob=-0.1),),
                top_logprobs=((TopKEntry(token="b", logprob=-1.0),),),
            )
        )
        assert isinstance(back.input_scoring, InputScoringResult)
        assert back.logprobs is not None
        assert isinstance(back.logprobs[0], TokenLogprob)
        assert back.top_logprobs is not None
        assert isinstance(back.top_logprobs[0][0], TopKEntry)

    def test_choice_logprobs_rehydrate(self):
        back = self._round_trip(
            Response(
                texts=("a", "b"),
                choice_logprobs=(
                    ChoiceLogprobs(
                        (TokenLogprob(token="a", logprob=-0.1),),
                        ((TopKEntry(token="a", logprob=-0.1),),),
                    ),
                    ChoiceLogprobs((TokenLogprob(token="b", logprob=-0.4),)),
                ),
            )
        )
        assert back.choice_logprobs is not None
        first, second = back.choice_logprobs
        assert isinstance(first, ChoiceLogprobs)
        assert isinstance(first.tokens[0], TokenLogprob)
        assert first.top_logprobs is not None
        assert isinstance(first.top_logprobs[0][0], TopKEntry)
        assert second.top_logprobs is None

    def test_grounding_citation_and_server_tool_use_rehydrate(self):
        back = self._round_trip(
            Response(
                texts=("x",),
                citations=(Citation(url="https://a"),),
                grounding=GroundingMetadata(
                    (GroundingChunk(uri="https://c"),), rendered_content="<w/>"
                ),
                server_tool_uses=(
                    ServerToolUse(
                        tool_type="web_search",
                        tool_use_id="t1",
                        input={"q": "x"},
                        result={"ok": True},
                    ),
                ),
            )
        )
        assert back.citations is not None
        assert isinstance(back.citations[0], Citation)
        assert isinstance(back.grounding, GroundingMetadata)
        assert back.grounding.rendered_content == "<w/>"
        assert back.server_tool_uses is not None
        assert isinstance(back.server_tool_uses[0], ServerToolUse)

    def test_structured_json_null_differs_from_absent(self):
        absent = self._round_trip(Response(texts=("x",)))
        present = self._round_trip(
            Response(texts=("x",), structured_output=StructuredOutput(None))
        )
        assert absent.structured_output is None
        assert isinstance(present.structured_output, StructuredOutput)
        assert present.structured_output.value is None

    def test_nested_provenance_rehydrates(self):
        provenance = ModelProvenance(
            dialect_id="openai_chat",
            engine_id="unknown",
            engine_source="unknown",
            deployment_id=None,
            model_identity=ModelIdentity("alias", "served"),
            capabilities=CapabilityEvidence(
                declared={"reasoning": {}},
                effective={"reasoning": {"effort": "high"}},
                plan_fingerprint="plan",
            ),
        )
        back = self._round_trip(Response(texts=("x",), provenance=provenance))
        assert isinstance(back.provenance, ModelProvenance)
        assert isinstance(back.provenance.model_identity, ModelIdentity)
        assert back.provenance.model_identity.requested_model_id == "alias"

    def test_old_record_without_provenance_still_rehydrates(self):
        old = {
            "texts": {"__sieval_cls__": "tuple", "items": ["old"]},
            "__sieval_mod__": Response.__module__,
            "__sieval_cls__": "Response",
        }
        back = dict_to_obj(old, global_type_registry)
        assert isinstance(back, Response)
        assert back.provenance is None
