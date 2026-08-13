"""Tests for the split OpenAI completions dialect.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    Supported,
    Unsupported,
)
from sieval.core.models.dialect import (
    Consumed,
    DialectError,
    OutputContractError,
    Passthrough,
    PreparedRequest,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
)
from sieval.core.models.dialects.openai_completions import (
    CAPABILITY_DECISIONS,
    OUTPUT_CONTRACT,
    OpenAICompletionsDialect,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    CompletionInput,
    DialectOptions,
    Request,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    StructuredOutputParams,
    TextPart,
    TokenLogprob,
    UsageStats,
    response_field_contract,
)
from sieval.core.types import JSONValue


@dataclass
class _Plan:
    dialect_id: str = "openai_completions"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


class _AsyncItems:
    def __init__(self, items: list[object]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> "_AsyncItems":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _choice(
    *,
    text: str = "",
    finish_reason: str | None = "stop",
    tokens: object = None,
    token_logprobs: object = None,
    top_logprobs: object = None,
    index: object = 0,
) -> object:
    logprobs = None
    if tokens is not None or token_logprobs is not None or top_logprobs is not None:
        logprobs = SimpleNamespace(
            tokens=[] if tokens is None else tokens,
            token_logprobs=[] if token_logprobs is None else token_logprobs,
            top_logprobs=top_logprobs,
        )
    return SimpleNamespace(
        index=index,
        text=text,
        finish_reason=finish_reason,
        logprobs=logprobs,
    )


def _usage(prompt: int, completion: int, total: int | None = None) -> object:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
    )


def _response(
    *choices: object,
    usage: object | None = None,
    model: object | None = None,
    fingerprint: object | None = None,
) -> object:
    return SimpleNamespace(
        choices=list(choices),
        usage=usage,
        model=model,
        system_fingerprint=fingerprint,
    )


def _dialect(raw: object) -> tuple[OpenAICompletionsDialect, AsyncMock]:
    create = AsyncMock(return_value=raw)
    client = SimpleNamespace(completions=SimpleNamespace(create=create))
    return OpenAICompletionsDialect(client, "served-model"), create


def _prepare(
    dialect: OpenAICompletionsDialect, req: Request
) -> tuple[RequestAudit, PreparedRequest]:
    audit = RequestAudit(active_request_leaves(req))
    dialect.validate_request(req, audit, _Plan())
    audit.raise_rejections()
    prepared = dialect.prepare(req, audit)
    audit.finish(prepared)
    return audit, prepared


class TestDescriptor:
    def test_legacy_capability_view_includes_fim(self) -> None:
        dialect, _ = _dialect(_response(_choice(text="unused")))

        assert Capability.FIM in dialect.capabilities

    @pytest.mark.parametrize("channel", ["logprobs", "top_logprobs"])
    def test_output_channel_validator_rejects_wrong_container(
        self, channel: str
    ) -> None:
        with pytest.raises(OutputContractError, match=channel):
            OUTPUT_CONTRACT.rules[channel].validator([])

    def test_all_capabilities_have_explicit_decisions(self) -> None:
        assert tuple(CAPABILITY_DECISIONS) == CAPABILITY_KEYS
        assert {
            key
            for key, decision in CAPABILITY_DECISIONS.items()
            if isinstance(decision, Supported)
        } == {"input_scoring", "sampled_logprobs", "top_logprobs", "fim"}
        assert all(
            isinstance(decision, Supported | Unsupported)
            for decision in CAPABILITY_DECISIONS.values()
        )

    def test_output_contract_classifies_every_response_channel(self) -> None:
        channels = {
            name
            for name, (role, _) in response_field_contract().items()
            if role == "channel"
        }
        assert set(OUTPUT_CONTRACT.rules) == channels


class TestValidationAndAudit:
    def test_audit_must_match_the_request(self) -> None:
        req = Request(input=CompletionInput("prompt"))
        dialect, _ = _dialect(_response())
        audit = RequestAudit({"input.completion": "different"})

        with pytest.raises(RequestAuditError, match="audit does not match"):
            dialect.validate_request(req, audit, _Plan())

    def test_empty_model_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="model must not be empty"):
            OpenAICompletionsDialect(object(), "")

    def test_prepare_rejects_chat_input_defensively(self) -> None:
        req = Request(input=ChatInput((ChatMessage("user", (TextPart("hello"),)),)))
        dialect, _ = _dialect(_response())
        audit = RequestAudit(active_request_leaves(req))

        with pytest.raises(DialectError, match="requires CompletionInput"):
            dialect.prepare(req, audit)

    def test_empty_dialect_options_are_a_documented_noop(self) -> None:
        req = Request(
            input=CompletionInput("prompt"),
            dialect_options=DialectOptions("openai_completions", {}),
        )
        dialect, _ = _dialect(_response())
        audit, prepared = _prepare(dialect, req)

        assert "dialect_options" in audit.decisions
        assert "extra_body" not in prepared.body

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "prepared",
        [
            PreparedRequest("wrong", {}, frozenset(), {}),
            PreparedRequest("completions.create", {}, frozenset(), {}),
        ],
    )
    async def test_execute_rejects_invalid_prepared_request(
        self, prepared: PreparedRequest
    ) -> None:
        dialect, create = _dialect(_response())

        with pytest.raises(DialectError, match="unexpected|invalid context"):
            await dialect.execute(prepared)

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_stream_response_must_be_async_iterable(self) -> None:
        dialect, _ = _dialect([])
        req = Request(
            input=CompletionInput("prompt"),
            scheduling=SchedulingParams(stream=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="asynchronously iterable"):
            await dialect.execute(prepared)

    @pytest.mark.parametrize(
        "req",
        [
            Request(input=ChatInput((ChatMessage("user", (TextPart("hi"),)),))),
            Request(
                input=CompletionInput("hi"),
                structured_output=StructuredOutputParams(format="json_object"),
            ),
        ],
    )
    def test_unsupported_leaves_are_rejected_with_zero_io(self, req: Request) -> None:
        dialect, create = _dialect(_response(_choice(text="unused")))
        audit = RequestAudit(active_request_leaves(req))

        with pytest.raises(RequestAuditError, match="does not support"):
            dialect.validate_request(req, audit, _Plan())
            audit.raise_rejections()

        create.assert_not_awaited()

    def test_prepare_accounts_for_every_supported_leaf_exactly_once(self) -> None:
        req = Request(
            input=CompletionInput("prefix", suffix="suffix"),
            sampling=SamplingParams(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                max_tokens=8,
                stop=("END",),
                seed=7,
                frequency_penalty=0.1,
                presence_penalty=0.2,
            ),
            scoring=ScoringParams(
                input_scoring=True, sampled_logprobs=True, top_logprobs=2
            ),
            scheduling=SchedulingParams(stream=True),
            dialect_options=DialectOptions(
                "openai_completions", {"min_p": 0.05, "custom": None}
            ),
        )
        dialect, _ = _dialect(_response())
        audit, prepared = _prepare(dialect, req)

        consumed = {
            path
            for path, decision in audit.decisions.items()
            if isinstance(decision, Consumed)
        }
        assert consumed == {
            "input.completion",
            "input.completion.suffix",
            "sampling.temperature",
            "sampling.top_p",
            "sampling.top_k",
            "sampling.max_tokens",
            "sampling.stop",
            "sampling.seed",
            "sampling.frequency_penalty",
            "sampling.presence_penalty",
            "scoring.input_scoring",
            "scoring.sampled_logprobs",
            "scoring.top_logprobs",
            "scheduling.stream",
        }
        assert isinstance(audit.decisions["dialect_options.min_p"], Passthrough)
        assert isinstance(audit.decisions["dialect_options.custom"], Passthrough)
        audit.finish(prepared)

    @pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
    def test_dialect_extra_cannot_override_first_class_ir(self, key: str) -> None:
        req = Request(
            input=CompletionInput("x"),
            dialect_options=DialectOptions("openai_completions", {key: 99}),
        )
        dialect, create = _dialect(_response())
        audit = RequestAudit(active_request_leaves(req))

        with pytest.raises(RequestAuditError, match="first-class"):
            dialect.validate_request(req, audit, _Plan())
            audit.raise_rejections()

        create.assert_not_awaited()

    @pytest.mark.parametrize("key", ["authorization", "connection_family"])
    def test_dialect_extra_cannot_contain_binding_resources(self, key: str) -> None:
        req = Request(
            input=CompletionInput("x"),
            dialect_options=DialectOptions(
                "openai_completions", {key: "secret-or-resource"}
            ),
        )
        dialect, create = _dialect(_response())
        audit = RequestAudit(active_request_leaves(req))

        with pytest.raises(RequestAuditError, match="first-class"):
            dialect.validate_request(req, audit, _Plan())
            audit.raise_rejections()

        create.assert_not_awaited()

    @pytest.mark.parametrize("key", ["reasoning_effort", "prompt_logprobs"])
    def test_dialect_extra_cannot_bypass_registered_semantics(self, key: str) -> None:
        req = Request(
            input=CompletionInput("x"),
            dialect_options=DialectOptions("openai_completions", {key: 1}),
        )
        dialect, create = _dialect(_response())
        audit = RequestAudit(active_request_leaves(req))

        with pytest.raises(RequestAuditError, match="first-class"):
            dialect.validate_request(req, audit, _Plan())
            audit.raise_rejections()

        create.assert_not_awaited()

    @pytest.mark.parametrize("key", ["prefill", "prefix"])
    def test_prefill_option_reports_actual_protocol_limitation(self, key: str) -> None:
        req = Request(
            input=CompletionInput("x"),
            dialect_options=DialectOptions(
                "openai_completions", {key: "assistant text"}
            ),
        )
        dialect, create = _dialect(_response())
        audit = RequestAudit(active_request_leaves(req))

        dialect.validate_request(req, audit, _Plan())
        with pytest.raises(RequestAuditError, match="chat input operation"):
            audit.raise_rejections()

        create.assert_not_awaited()


class TestWirePreparation:
    @pytest.mark.anyio
    async def test_suffix_top_k_and_extras_reach_the_intended_wire_locations(
        self,
    ) -> None:
        dialect, create = _dialect(
            _response(_choice(text="middle"), usage=_usage(1, 1))
        )
        req = Request(
            input=CompletionInput("prefix", suffix="suffix"),
            sampling=SamplingParams(top_k=50, max_tokens=3),
            dialect_options=DialectOptions(
                "openai_completions", {"min_p": 0.1, "logit_bias": {"1": 2}}
            ),
        )
        _, prepared = _prepare(dialect, req)
        result = await dialect.execute(prepared)

        assert create.await_args is not None
        kwargs = create.await_args.kwargs
        assert kwargs["prompt"] == "prefix"
        assert kwargs["suffix"] == "suffix"
        assert kwargs["max_tokens"] == 3
        assert "top_k" not in {key for key in kwargs if key != "extra_body"}
        assert kwargs["extra_body"] == {
            "top_k": 50,
            "min_p": 0.1,
            "logit_bias": {"1": 2},
        }
        assert result.texts == ("middle",)
        assert result.request_params == {
            "stream": False,
            "suffix": "suffix",
            "max_tokens": 3,
            "extra_body": kwargs["extra_body"],
        }


class TestInputScoringBoundary:
    @pytest.mark.anyio
    async def test_missing_usage_is_an_error_instead_of_a_zero_split(self) -> None:
        dialect, _ = _dialect(
            _response(_choice(tokens=["p", " out"], token_logprobs=[None, -0.2]))
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(input_scoring=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="usage.*boundary"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("usage", "message"),
        [
            (_usage(0, 2), "positive"),
            (_usage(3, 0), "exceeds"),
            (_usage(1, 0), "completion-token"),
        ],
    )
    async def test_inconsistent_usage_is_an_error(
        self, usage: object, message: str
    ) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(tokens=["p", " out"], token_logprobs=[None, -0.2]),
                usage=usage,
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(input_scoring=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match=message):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_reported_total_is_ignored_in_favour_of_the_computed_one(
        self,
    ) -> None:
        """A total that does not decompose must not cost the caller the reply."""
        dialect, _ = _dialect(
            _response(
                _choice(tokens=["p", " out"], token_logprobs=[None, -0.2]),
                usage=_usage(1, 1, 9),
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(input_scoring=True),
        )
        _, prepared = _prepare(dialect, req)

        result = await dialect.execute(prepared)

        assert result.usage == UsageStats(
            input_tokens=1, output_tokens=1, total_tokens=2
        )


class TestOutputLifting:
    @pytest.mark.anyio
    async def test_usage_fields_must_be_non_negative_integers(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=True,
            completion_tokens=0,
            total_tokens=0,
        )
        dialect, _ = _dialect(_response(_choice(text="ok"), usage=usage))
        req = Request(input=CompletionInput("p"))
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="non-negative integer"):
            await dialect.execute(prepared)

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
        req = Request(input=CompletionInput("p"))
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="choices|omitted"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_input_scoring_requires_echoed_logprobs(self) -> None:
        dialect, _ = _dialect(_response(_choice(text="ok"), usage=_usage(1, 0)))
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(input_scoring=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="omitted echoed"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    @pytest.mark.parametrize("index", ["0", True, -1, 1])
    async def test_non_stream_invalid_choice_index_is_a_contract_error(
        self, index: object
    ) -> None:
        dialect, _ = _dialect(_response(_choice(text="invalid", index=index)))
        req = Request(input=CompletionInput("p"))
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="choice index"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_non_stream_duplicate_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(
            _response(_choice(text="first"), _choice(text="duplicate"))
        )
        req = Request(input=CompletionInput("p"))
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="duplicated choice index 0"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_non_stream_missing_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(_response(_choice(text="first", index=0)))
        req = Request(input=CompletionInput("p"), sampling=SamplingParams(n=2))
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match=r"omitted choice indexes \[1\]"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_stream_invalid_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(_AsyncItems([_response(_choice(index="0"))]))
        req = Request(
            input=CompletionInput("p"),
            scheduling=SchedulingParams(stream=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(
            OutputContractError, match="choice index must be an integer"
        ):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_stream_missing_choice_index_is_a_contract_error(self) -> None:
        dialect, _ = _dialect(_AsyncItems([_response(_choice(text="first", index=0))]))
        req = Request(
            input=CompletionInput("p"),
            sampling=SamplingParams(n=2),
            scheduling=SchedulingParams(stream=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match=r"omitted choice indexes \[1\]"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tokens", "token_logprobs", "message"),
        [
            ("A", [-0.1], "tokens must be a sequence"),
            ([1], [-0.1], "tokens\\[0\\] must be a string"),
            (["A"], "-0.1", "token_logprobs must be a sequence"),
            (["A"], [True], "must be numeric or None"),
            (["A"], ["bad"], "must be numeric or None"),
            (["A"], [float("nan")], "must be finite"),
            (["A"], [float("inf")], "must be finite"),
        ],
    )
    async def test_malformed_sampled_logprobs_fail_loudly(
        self, tokens: object, token_logprobs: object, message: str
    ) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(
                    text="A",
                    tokens=tokens,
                    token_logprobs=token_logprobs,
                )
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match=message):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_top_logprob_length_must_match_tokens_per_choice(self) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(
                    text="AB",
                    tokens=["A", "B"],
                    token_logprobs=[-0.1, -0.2],
                    top_logprobs=[{"A": -0.1}],
                )
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="positions are inconsistent"):
            await dialect.execute(prepared)

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
        dialect, _ = _dialect(_response(_choice(text="ok"), **response_kwargs))
        req = Request(input=CompletionInput("p"))
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match=field):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("raw_top_logprobs", "message"),
        [
            ({"A": -0.1}, "must be a sequence"),
            (["not-a-position"], "must be a mapping or None"),
            ([{1: -0.1}], "non-string token"),
            ([{"A": "bad"}], "must be numeric"),
            ([{"A": True}], "must be numeric"),
            ([{"A": float("nan")}], "must be finite"),
            ([{"A": float("inf")}], "must be finite"),
        ],
    )
    async def test_malformed_top_logprobs_fail_loudly(
        self, raw_top_logprobs: object, message: str
    ) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(
                    text="A",
                    tokens=["A"],
                    token_logprobs=[-0.1],
                    top_logprobs=raw_top_logprobs,
                )
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match=message):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_none_top_logprob_position_is_preserved_as_empty(self) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(
                    text="A",
                    tokens=["A"],
                    token_logprobs=[-0.1],
                    top_logprobs=[None],
                )
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        _, prepared = _prepare(dialect, req)

        result = await dialect.execute(prepared)

        assert result.top_logprobs == ((),)

    @pytest.mark.anyio
    async def test_absent_top_logprobs_container_remains_absent(self) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(
                    text="A",
                    tokens=["A"],
                    token_logprobs=[-0.1],
                    top_logprobs=None,
                )
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        _, prepared = _prepare(dialect, req)

        result = await dialect.execute(prepared)

        assert result.top_logprobs is None

    @pytest.mark.anyio
    async def test_misaligned_sampled_logprobs_fail_instead_of_truncating(self) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(
                    text="A",
                    tokens=["A", "B"],
                    token_logprobs=[-0.1],
                ),
                usage=_usage(1, 1),
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(sampled_logprobs=True),
        )
        _, prepared = _prepare(dialect, req)

        with pytest.raises(OutputContractError, match="counts are inconsistent"):
            await dialect.execute(prepared)

    @pytest.mark.anyio
    async def test_completion_logprobs_keep_completions_top_logprob_shape(
        self,
    ) -> None:
        dialect, _ = _dialect(
            _response(
                _choice(
                    text="A",
                    tokens=["A"],
                    token_logprobs=[-0.1],
                    top_logprobs=[{"A": -0.1, "B": -2.0}],
                ),
                usage=_usage(1, 1),
                model="provider-model",
                fingerprint="fp_1",
            )
        )
        req = Request(
            input=CompletionInput("p"),
            scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
        )
        _, prepared = _prepare(dialect, req)
        result = await dialect.execute(prepared)

        assert result.logprobs == (TokenLogprob("A", -0.1),)
        assert result.top_logprobs is not None
        assert {(item.token, item.logprob) for item in result.top_logprobs[0]} == {
            ("A", -0.1),
            ("B", -2.0),
        }
        assert result.usage == UsageStats(
            input_tokens=1, output_tokens=1, total_tokens=2
        )
        assert result.response_model == "provider-model"
        assert result.system_fingerprint == "fp_1"
        OUTPUT_CONTRACT.validate(_Plan(), req, result)

    @pytest.mark.anyio
    async def test_stream_accumulates_and_splits_echo_at_final_usage_boundary(
        self,
    ) -> None:
        chunks = [
            _response(
                _choice(
                    finish_reason=None,
                    tokens=["a", "b"],
                    token_logprobs=[None, -1.0],
                    top_logprobs=[{"a": -0.1}, {"b": -1.0}],
                ),
                model="stream-model",
            ),
            _response(
                _choice(
                    text=" c",
                    tokens=[" c"],
                    token_logprobs=[-0.3],
                    top_logprobs=[{" c": -0.3, " d": -1.3}],
                )
            ),
            _response(usage=_usage(2, 1)),
        ]
        dialect, create = _dialect(_AsyncItems(chunks))
        req = Request(
            input=CompletionInput("ab"),
            scoring=ScoringParams(
                input_scoring=True, sampled_logprobs=True, top_logprobs=2
            ),
            scheduling=SchedulingParams(stream=True),
        )
        _, prepared = _prepare(dialect, req)
        result = await dialect.execute(prepared)

        assert result.texts == (" c",)
        assert result.input_scoring is not None
        assert [item.token for item in result.input_scoring.token_logprobs] == [
            "a",
            "b",
        ]
        assert result.logprobs == (TokenLogprob(" c", -0.3),)
        assert result.top_logprobs is not None
        assert [item.token for item in result.top_logprobs[0]] == [" c", " d"]
        assert result.response_model == "stream-model"
        assert create.await_args is not None
        assert create.await_args.kwargs["stream"] is True
        assert create.await_args.kwargs["stream_options"] == {"include_usage": True}
        OUTPUT_CONTRACT.validate(_Plan(), req, result)
