"""Tests for the lightweight dialect request/output contracts.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from sieval.core.models.dialect import (
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PassthroughObservation,
    PreparedRequest,
    RequestAudit,
    RequestAuditError,
    active_request_capabilities,
    active_request_leaves,
    active_response_channels,
    nondefault_leaf_paths,
    request_capability,
    required_response_channels,
    validate_input_scoring,
    validate_reasoning,
    validate_request_invariants,
    validate_runtime_binding_plan,
    validate_structured_output,
    validate_tool_calls,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    CompletionInput,
    DialectOptions,
    FunctionToolCall,
    HostedToolSpec,
    ImagePart,
    InputScoringResult,
    OpaqueContinuation,
    ReasoningOutput,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    SessionParams,
    StructuredOutput,
    StructuredOutputParams,
    TextPart,
    ToolParams,
    ToolResultPart,
    normalize_chat_input,
    response_field_contract,
)
from sieval.core.types import JSONValue


@dataclass(frozen=True)
class _Plan:
    dialect_id: str = "openai_completions"
    available_capabilities: frozenset[str] = frozenset(
        {"input_scoring", "sampled_logprobs", "top_logprobs", "fim"}
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


def _all_rules(guarantee=Guarantee.BEST_EFFORT):
    return {
        name: OutputRule(guarantee)
        for name, (role, _) in response_field_contract().items()
        if role == "channel"
    }


class TestLeafDerivation:
    def test_derives_nondefault_group_leaves_and_dynamic_none(self):
        req = Request(
            input=CompletionInput("x", suffix="tail"),
            sampling=SamplingParams(temperature=0.2),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=5),
            dialect_options=DialectOptions(
                "openai_completions", {"min_p": None, "repetition_penalty": 1.1}
            ),
        )
        assert nondefault_leaf_paths(req) == frozenset(
            {
                "input.completion",
                "input.completion.suffix",
                "sampling.temperature",
                "scoring.sampled_logprobs",
                "scoring.top_logprobs",
                "dialect_options.min_p",
                "dialect_options.repetition_penalty",
            }
        )

    def test_chat_kind_and_modalities_are_visible_but_payloads_are_opaque(self):
        req = Request(
            input=normalize_chat_input(
                [ChatMessage("user", (TextPart("look"), ImagePart(url="https://x")))]
            )
        )
        leaves = active_request_leaves(req)
        assert set(leaves) == {
            "input.chat",
            "input.modality.text",
            "input.modality.image",
        }
        assert "look" not in leaves

    def test_branch_sensitive_image_media_type_is_a_separate_leaf(self):
        req = Request(
            input=ChatInput(
                (
                    ChatMessage(
                        "user",
                        (ImagePart(data="YWJj", media_type="image/png"),),
                    ),
                )
            )
        )

        assert (
            active_request_leaves(req)["input.modality.image.media_type"] == "image/png"
        )

    def test_branch_sensitive_chat_metadata_has_separate_leaves(self):
        req = Request(
            input=ChatInput(
                (
                    ChatMessage(
                        "user",
                        (ImagePart(url="https://x", detail="high"),),
                        name="speaker",
                    ),
                )
            )
        )

        leaves = active_request_leaves(req)
        assert leaves["input.chat.message.name"] == "speaker"
        assert leaves["input.modality.image.detail"] == "high"

    def test_tool_result_error_marker_is_a_separate_nondefault_leaf(self):
        normal = Request(
            input=ChatInput((ChatMessage("tool", (ToolResultPart("call", "ok"),)),))
        )
        failed = Request(
            input=ChatInput(
                (ChatMessage("tool", (ToolResultPart("call", "boom", True),)),)
            )
        )

        path = "input.modality.tool_result.is_error"
        assert path not in active_request_leaves(normal)
        assert active_request_leaves(failed)[path] is True

    def test_empty_dialect_options_are_an_explicit_leaf(self):
        req = Request(
            input=CompletionInput("x"),
            dialect_options=DialectOptions("openai_completions", {}),
        )

        assert active_request_leaves(req)["dialect_options"] == "openai_completions"

    def test_unknown_request_input_and_content_part_fail_loudly(self):
        with pytest.raises(TypeError, match="unsupported type"):
            active_request_leaves(Request(input=cast(Any, object())))

        req = Request(input=ChatInput((ChatMessage("user", cast(Any, (object(),))),)))
        with pytest.raises(TypeError, match="unclassified part"):
            active_request_leaves(req)

    @pytest.mark.parametrize(
        ("path", "capability"),
        [
            ("input.completion.suffix", "fim"),
            ("input.modality.image", "multimodal_input"),
            ("input.modality.image.detail", "multimodal_input"),
            ("input.modality.image.media_type", "multimodal_input"),
            ("scoring.input_scoring", "input_scoring"),
            ("scoring.sampled_logprobs", "sampled_logprobs"),
            ("scoring.top_logprobs", "top_logprobs"),
            ("tools.hosted", "hosted_tools"),
            ("session.previous_response_id", "stateful_session"),
            ("session.opaque_continuation", "opaque_continuation"),
            ("reasoning.effort", "reasoning"),
            ("tools.functions", "function_tools"),
            ("structured_output.format", "structured_output"),
            ("sampling.temperature", None),
        ],
    )
    def test_request_leaf_capability_projection(self, path, capability):
        assert request_capability(path) == capability

    def test_active_capabilities_and_response_channels_are_derived(self):
        req = Request(
            input=CompletionInput("x", suffix="tail"),
            scoring=ScoringParams(
                input_scoring=True,
                sampled_logprobs=True,
                top_logprobs=2,
            ),
            reasoning=ReasoningParams(summary="auto"),
            tools=ToolParams(
                functions=({"type": "function"},),
                hosted=(HostedToolSpec("web_search"),),
            ),
            structured_output=StructuredOutputParams(format="json_object"),
            session=SessionParams(previous_response_id="previous"),
        )

        assert active_request_capabilities(req) == {
            "fim",
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
            "reasoning",
            "function_tools",
            "hosted_tools",
            "structured_output",
            "stateful_session",
        }
        assert active_response_channels(req) == {
            "reasoning",
            "logprobs",
            "top_logprobs",
            "input_scoring",
            "tool_calls",
            "server_tool_uses",
            "structured_output",
            "session_id",
        }
        assert required_response_channels(req) == {
            "reasoning",
            "logprobs",
            "top_logprobs",
            "input_scoring",
            "structured_output",
            "session_id",
        }


class TestRequestAudit:
    def test_omitted_leaf_fails(self):
        audit = RequestAudit({"input.completion": "x"})
        with pytest.raises(RequestAuditError, match="unaccounted"):
            audit.finish(PreparedRequest("create", {}, frozenset(), {}))

    def test_double_accounting_fails(self):
        audit = RequestAudit({"input.completion": "x"})
        audit.consumed("input.completion")
        with pytest.raises(RequestAuditError, match="twice"):
            audit.noop("input.completion", "equivalent")

    def test_unclaimed_prepared_consumption_fails(self):
        audit = RequestAudit({"input.completion": "x"})
        audit.noop("input.completion", "empty body is semantically equivalent")
        prepared = PreparedRequest("create", {}, frozenset({"input.completion"}), {})
        with pytest.raises(RequestAuditError, match="unclaimed"):
            audit.finish(prepared)

    def test_passthrough_destination_and_value_are_verified(self):
        audit = RequestAudit({"dialect_options.min_p": 0.1})
        audit.passthrough("dialect_options.min_p", "extra_body")
        altered = PreparedRequest(
            "create",
            {},
            frozenset(),
            {"dialect_options.min_p": PassthroughObservation("extra_body", 0.2)},
        )
        with pytest.raises(RequestAuditError, match="value changed"):
            audit.finish(altered)

    def test_rejection_is_raised_before_prepare(self):
        audit = RequestAudit({"scoring.input_scoring": True})
        audit.rejected("scoring.input_scoring", "not supported")
        with pytest.raises(RequestAuditError, match="not supported"):
            audit.raise_rejections()

    def test_active_and_decisions_views_are_detached(self):
        audit = RequestAudit({"input.completion": "x"})
        active = cast(dict[str, object], audit.active)
        active["extra"] = True
        audit.consumed("input.completion")
        decisions = cast(dict[str, object], audit.decisions)
        decisions.clear()

        assert "extra" not in audit.active
        assert "input.completion" in audit.decisions

    def test_decision_must_reference_an_active_leaf(self):
        audit = RequestAudit({"input.completion": "x"})
        with pytest.raises(RequestAuditError, match="inactive leaf"):
            audit.consumed("sampling.temperature")

    def test_noop_requires_a_documented_reason(self):
        audit = RequestAudit({"input.completion": "x"})
        with pytest.raises(RequestAuditError, match="documented"):
            audit.noop("input.completion", "  ")

    def test_only_dialect_options_can_passthrough(self):
        audit = RequestAudit({"input.completion": "x"})
        with pytest.raises(RequestAuditError, match="only dialect options"):
            audit.passthrough("input.completion", "body")

    def test_consumed_leaf_requires_prepared_observation(self):
        audit = RequestAudit({"input.completion": "x"})
        audit.consumed("input.completion")
        with pytest.raises(RequestAuditError, match="unobserved"):
            audit.finish(PreparedRequest("create", {}, frozenset(), {}))

    def test_passthrough_paths_and_destinations_must_match(self):
        audit = RequestAudit({"dialect_options.min_p": 0.1})
        audit.passthrough("dialect_options.min_p", "extra_body")

        with pytest.raises(RequestAuditError, match="paths do not match"):
            audit.finish(PreparedRequest("create", {}, frozenset(), {}))

        altered = PreparedRequest(
            "create",
            {},
            frozenset(),
            {"dialect_options.min_p": PassthroughObservation("body", 0.1)},
        )
        with pytest.raises(RequestAuditError, match="destination changed"):
            audit.finish(altered)


class TestPlanAndInvariants:
    def test_unavailable_capability_is_rejected(self):
        req = Request(
            input=CompletionInput("x"),
            structured_output=StructuredOutputParams(format="json_object"),
        )
        with pytest.raises(ValueError, match="unavailable capability"):
            validate_runtime_binding_plan(_Plan(), req)

    def test_dialect_options_must_match_binding(self):
        req = Request(
            input=CompletionInput("x"),
            dialect_options=DialectOptions("openai_chat", {"min_p": 0.1}),
        )
        with pytest.raises(ValueError, match="target"):
            validate_runtime_binding_plan(_Plan(), req)

    def test_opaque_continuation_must_match_bound_dialect(self):
        req = Request(
            input=CompletionInput("x"),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("future_dialect", "state")
            ),
        )
        plan = _Plan(
            available_capabilities=_Plan().available_capabilities
            | {"opaque_continuation"}
        )

        with pytest.raises(ValueError, match="originated from 'future_dialect'"):
            validate_runtime_binding_plan(plan, req)

    def test_minimum_cannot_be_weakened_per_call(self):
        plan = _Plan(capability_minimums={"top_logprobs": {"minimum": 10}})
        req = Request(
            input=CompletionInput("x"),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=5),
        )
        with pytest.raises(ValueError, match="weakens"):
            validate_runtime_binding_plan(plan, req)

    def test_inactive_top_logprobs_is_not_forced_by_binding_minimum(self):
        plan = _Plan(capability_minimums={"top_logprobs": {"minimum": 10}})

        validate_runtime_binding_plan(plan, Request(input=CompletionInput("x")))

    def test_top_logprobs_implies_sampled_logprobs(self):
        req = Request(input=CompletionInput("x"), scoring=ScoringParams(top_logprobs=3))
        with pytest.raises(ValueError, match="requires sampled"):
            validate_request_invariants(req)

    def test_n_gt_one_allows_choice_indexed_reasoning_but_not_scoring(self):
        reasoning = Request(
            input=CompletionInput("x"),
            sampling=SamplingParams(n=2),
        )
        validate_request_invariants(reasoning)

        scoring = Request(
            input=CompletionInput("x"),
            sampling=SamplingParams(n=2),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        with pytest.raises(ValueError, match="singular.*logprobs"):
            validate_request_invariants(scoring)

    @pytest.mark.parametrize(
        ("model_request", "error", "message"),
        [
            (
                Request(
                    input=CompletionInput("x"),
                    sampling=SamplingParams(temperature=cast(Any, True)),
                ),
                TypeError,
                "temperature",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    sampling=SamplingParams(top_k=cast(Any, 1.5)),
                ),
                TypeError,
                "top_k",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    sampling=SamplingParams(stop=cast(Any, ["stop"])),
                ),
                TypeError,
                "stop",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    sampling=SamplingParams(n=cast(Any, True)),
                ),
                TypeError,
                "sampling.n",
            ),
            (
                Request(input=CompletionInput("x"), sampling=SamplingParams(n=0)),
                ValueError,
                ">= 1",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    scoring=ScoringParams(input_scoring=cast(Any, 1)),
                ),
                TypeError,
                "input_scoring",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    scoring=ScoringParams(top_logprobs=cast(Any, True)),
                ),
                TypeError,
                "top_logprobs",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    scoring=ScoringParams(top_logprobs=-1),
                ),
                ValueError,
                ">= 0",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    reasoning=ReasoningParams(effort=cast(Any, 7)),
                ),
                TypeError,
                "effort",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    reasoning=ReasoningParams(budget_tokens=cast(Any, True)),
                ),
                TypeError,
                "budget_tokens",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    reasoning=ReasoningParams(summary=cast(Any, "verbose")),
                ),
                ValueError,
                "summary",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    tools=ToolParams(parallel=cast(Any, "yes")),
                ),
                TypeError,
                "tools.parallel",
            ),
            (
                Request(
                    input=CompletionInput("x"),
                    scheduling=SchedulingParams(stream=cast(Any, 1)),
                ),
                TypeError,
                "scheduling.stream",
            ),
        ],
    )
    def test_request_invariants_reject_invalid_domains(
        self, model_request: Request, error: type[Exception], message: str
    ):
        with pytest.raises(error, match=message):
            validate_request_invariants(model_request)


class TestOutputContract:
    def test_contract_must_classify_every_root_channel(self):
        rules = _all_rules()
        rules.pop("grounding")
        with pytest.raises(ValueError, match="incomplete"):
            OutputContract(rules)

    def test_contract_rejects_extra_channel_classification(self):
        rules = _all_rules()
        rules["unknown"] = OutputRule(Guarantee.BEST_EFFORT)
        with pytest.raises(ValueError, match="extra=.*unknown"):
            OutputContract(rules)

    def test_binding_required_channel_does_not_pollute_an_inactive_call(self):
        rules = _all_rules(Guarantee.BEST_EFFORT)
        rules["input_scoring"] = OutputRule(Guarantee.PRESENT_OR_ERROR)
        contract = OutputContract(rules)
        plan = _Plan(required_output_channels=frozenset({"input_scoring"}))

        contract.validate(
            plan,
            Request(input=CompletionInput("x")),
            Response(("x",)),
        )

    def test_active_present_or_error_channel_must_exist(self):
        rules = _all_rules(Guarantee.BEST_EFFORT)
        rules["input_scoring"] = OutputRule(Guarantee.PRESENT_OR_ERROR)
        contract = OutputContract(rules)
        plan = _Plan(required_output_channels=frozenset({"input_scoring"}))
        with pytest.raises(OutputContractError, match="absent"):
            contract.validate(
                plan,
                Request(
                    input=CompletionInput("x"),
                    scoring=ScoringParams(input_scoring=True),
                ),
                Response(("x",)),
            )

    def test_required_best_effort_channel_is_not_a_guarantee(self):
        contract = OutputContract(_all_rules(Guarantee.BEST_EFFORT))
        with pytest.raises(OutputContractError, match="best_effort guarantee"):
            contract.validate(
                _Plan(),
                Request(
                    input=CompletionInput("x"),
                    scoring=ScoringParams(input_scoring=True),
                ),
                Response(("x",), input_scoring=InputScoringResult(())),
            )

    def test_never_channel_rejects_unexpected_payload(self):
        rules = _all_rules(Guarantee.BEST_EFFORT)
        rules["structured_output"] = OutputRule(Guarantee.NEVER)
        contract = OutputContract(rules)
        with pytest.raises(OutputContractError, match="promised"):
            contract.validate(
                _Plan(),
                Request(input=CompletionInput("x")),
                Response(("x",), structured_output=object()),  # type: ignore[invalid-argument-type]
            )

    def test_previous_response_id_requires_session_id_not_reasoning(self):
        req = Request(
            input=CompletionInput("x"),
            session=SessionParams(previous_response_id="previous"),
        )

        assert required_response_channels(req) == frozenset({"session_id"})

    def test_opaque_continuation_requires_reasoning_not_session_id(self):
        req = Request(
            input=CompletionInput("x"),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("future_dialect", "previous")
            ),
        )

        assert required_response_channels(req) == frozenset({"reasoning"})

    def test_opaque_continuation_requires_nonempty_payload_for_every_choice(self):
        rules = _all_rules(Guarantee.BEST_EFFORT)
        rules["reasoning"] = OutputRule(Guarantee.PRESENT_OR_ERROR)
        contract = OutputContract(rules)
        req = Request(
            input=CompletionInput("x"),
            sampling=SamplingParams(n=2),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("future_dialect", "previous")
            ),
        )

        with pytest.raises(OutputContractError, match="opaque round-trip payload"):
            contract.validate(
                _Plan(),
                req,
                Response(
                    ("first", "second"),
                    reasoning=(
                        ReasoningOutput(opaque_roundtrip="next-1"),
                        ReasoningOutput(),
                    ),
                ),
            )

        contract.validate(
            _Plan(),
            req,
            Response(
                ("first", "second"),
                reasoning=(
                    ReasoningOutput(opaque_roundtrip="next-1"),
                    ReasoningOutput(opaque_roundtrip="next-2"),
                ),
            ),
        )

    def test_opaque_continuation_rejects_absent_reasoning(self):
        rules = _all_rules(Guarantee.BEST_EFFORT)
        rules["reasoning"] = OutputRule(Guarantee.PRESENT_OR_ERROR)
        contract = OutputContract(rules)
        req = Request(
            input=CompletionInput("x"),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("future_dialect", "previous")
            ),
        )

        with pytest.raises(OutputContractError, match="required.*absent"):
            contract.validate(_Plan(), req, Response(("x",)))

    def test_response_choice_count_must_match_request(self):
        contract = OutputContract(_all_rules())
        with pytest.raises(OutputContractError, match="expected 2 choices"):
            contract.validate(
                _Plan(),
                Request(input=CompletionInput("x"), sampling=SamplingParams(n=2)),
                Response(("only-one",)),
            )

    @pytest.mark.parametrize(
        ("validator", "value", "message"),
        [
            (validate_reasoning, [], "reasoning"),
            (validate_reasoning, (object(),), "reasoning"),
            (validate_tool_calls, [], "tool_calls"),
            (validate_tool_calls, (object(),), "tool_calls"),
            (validate_input_scoring, object(), "input_scoring"),
            (validate_structured_output, object(), "structured_output"),
        ],
    )
    def test_response_channel_validators_reject_wrong_shapes(
        self, validator, value, message
    ):
        with pytest.raises(OutputContractError, match=message):
            validator(value)

    def test_response_channel_validators_accept_typed_values(self):
        validate_reasoning((None, ReasoningOutput(text="reason")))
        validate_tool_calls((FunctionToolCall("call", "tool", {}),))
        validate_input_scoring(InputScoringResult(()))
        validate_structured_output(StructuredOutput({"ok": True}))
