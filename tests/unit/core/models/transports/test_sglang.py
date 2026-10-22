"""Tests for the legacy SGLang native ``/generate`` executor.

URL derivation, token-text normalization, triple parsing, the radix-cache
guard, and the n>1 list-response handling moved here from the legacy
SglangGenModel tests (tests/unit/core/models/test_sglang_gen_model.py).  The
RFC #25 PR-1 implementation preserves it behind the explicit
``sglang_legacy`` bypass: everything is driven through
``SglangTransport.arun(Request(...))`` with the OpenAI client's ``post`` mocked
— no real traffic.  This object is deliberately not a canonical split Dialect
before PR 5.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sieval.core.models.dialect import Dialect
from sieval.core.models.ir import (
    CompletionInput,
    DialectOptions,
    Request,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    TokenLogprob,
    TopKEntry,
    UsageStats,
    normalize_chat_input,
)
from sieval.core.models.transports.sglang import SglangTransport, _normalize_token_text


def _make_transport(payload: dict | list) -> tuple[SglangTransport, AsyncMock]:
    client = MagicMock()
    post = AsyncMock(return_value=payload)
    client.post = post
    t = SglangTransport(client=client, model="m", api_base="http://host/v1")
    return t, post


def _meta(**overrides: Any) -> dict:
    meta = {
        "prompt_tokens": 0,
        "completion_tokens": 1,
        "cached_tokens": 0,
        "finish_reason": {"type": "stop"},
    }
    meta.update(overrides)
    return meta


def _sampling(body: dict) -> dict:
    """Narrow ``body["sampling_params"]`` (typed JSONValue) to a dict for the
    type checker; the sglang lower() always nests a dict there."""
    sp = body["sampling_params"]
    assert isinstance(sp, dict)
    return sp


def _request(
    text: str,
    *,
    sampling: SamplingParams | None = None,
    input_scoring: bool = False,
    sampled_logprobs: bool = False,
    top_logprobs: int = 0,
    options: dict | None = None,
    stream: bool = False,
) -> Request:
    """Build the typed IR consumed by the one-cycle native SGLang path."""

    return Request(
        input=CompletionInput(text),
        sampling=sampling or SamplingParams(),
        scoring=ScoringParams(
            input_scoring=input_scoring,
            sampled_logprobs=sampled_logprobs,
            top_logprobs=top_logprobs,
        ),
        scheduling=SchedulingParams(stream=stream),
        dialect_options=(
            DialectOptions("sglang_legacy", options) if options is not None else None
        ),
    )


def test_legacy_executor_is_not_a_canonical_dialect() -> None:
    transport, _ = _make_transport({"text": "", "meta_info": _meta()})

    assert not isinstance(transport, Dialect)


# ===================================================================
# URL derivation (moved from the legacy SglangGenModel tests)
# ===================================================================
class TestGenerateUrl:
    def test_strips_v1_suffix(self):
        t = SglangTransport(
            client=MagicMock(), model="x", api_base="http://host:8000/v1"
        )
        assert t._generate_url() == "http://host:8000/generate"

    def test_trailing_slash_base(self):
        t = SglangTransport(
            client=MagicMock(), model="x", api_base="http://host:8000/v1/"
        )
        assert t._generate_url() == "http://host:8000/generate"

    def test_no_v1_suffix(self):
        t = SglangTransport(client=MagicMock(), model="x", api_base="http://host:8000")
        assert t._generate_url() == "http://host:8000/generate"

    def test_none_base(self):
        t = SglangTransport(client=MagicMock(), model="x", api_base=None)
        assert t._generate_url() == "/generate"


# ===================================================================
# Token text normalization (moved from the legacy SglangGenModel tests)
# ===================================================================
class TestNormalizeTokenText:
    def test_space_marker(self):
        assert _normalize_token_text("ĠA") == " A"

    def test_newline_marker(self):
        assert _normalize_token_text("Ċ") == "\n"

    def test_plain_unchanged(self):
        assert _normalize_token_text(" A") == " A"

    def test_none_text_raises(self):
        # Server without detokenization (--skip-tokenizer-init) returns no text;
        # fail loud rather than crash on None.replace or degrade to "".
        with pytest.raises(RuntimeError, match="no token text"):
            _normalize_token_text(None)


class TestLower:
    def test_max_tokens_maps_to_max_new_tokens(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", sampling=SamplingParams(max_tokens=7)))
        assert _sampling(body)["max_new_tokens"] == 7

    def test_score_input_sets_logprob_start_len_0(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", input_scoring=True, top_logprobs=5))
        assert body["logprob_start_len"] == 0
        assert body["return_logprob"] is True
        assert body["top_logprobs_num"] == 5
        assert body["return_text_in_logprobs"] is True

    def test_no_score_input_sets_logprob_start_len_minus_1(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", sampled_logprobs=True, top_logprobs=3))
        assert body["logprob_start_len"] == -1

    def test_top_k_maps_to_top_logprobs_num(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", sampled_logprobs=True, top_logprobs=4))
        assert body["top_logprobs_num"] == 4

    def test_no_logprobs_request_omits_logprob_fields(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi"))
        assert "return_logprob" not in body
        assert "logprob_start_len" not in body
        assert "max_new_tokens" not in _sampling(body)

    def test_request_seed_is_not_a_native_wire_parameter(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", sampling=SamplingParams(seed=7)))

        assert "seed" not in body
        assert "seed" not in _sampling(body)

    def test_echo_never_appears_in_wire_body(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", input_scoring=True))
        assert "echo" not in body
        assert "echo" not in _sampling(body)

    def test_prefix_maps_to_prefill(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", options={"prefill": "PRE"}))
        assert _sampling(body)["prefill"] == "PRE"

    def test_score_input_forces_min_one_new_token(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", input_scoring=True))
        assert _sampling(body)["max_new_tokens"] == 1

    def test_score_input_clamps_explicit_zero_to_one(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            _request(
                "hi",
                sampling=SamplingParams(max_tokens=0),
                input_scoring=True,
            )
        )
        assert _sampling(body)["max_new_tokens"] == 1

    def test_sampled_logprobs_clamp_explicit_zero_to_one(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            _request(
                "hi",
                sampling=SamplingParams(max_tokens=0),
                sampled_logprobs=True,
            )
        )
        assert _sampling(body)["max_new_tokens"] == 1

    def test_all_sampling_params_map(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            _request(
                "hi",
                sampling=SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    stop=("</s>",),
                    frequency_penalty=0.1,
                    presence_penalty=0.2,
                    n=3,
                ),
            )
        )
        sp = _sampling(body)
        assert sp["temperature"] == 0.7
        assert sp["top_p"] == 0.9
        assert sp["top_k"] == 40
        assert sp["stop"] == ["</s>"]
        assert sp["frequency_penalty"] == 0.1
        assert sp["presence_penalty"] == 0.2
        assert sp["n"] == 3

    def test_non_str_input_raises(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        with pytest.raises(TypeError, match="CompletionInput"):
            t._lower(
                Request(input=normalize_chat_input(({"role": "user", "content": "x"},)))
            )


class TestDialectOptions:
    """Dialect options go through the OpenAI→SGLang table or are dropped."""

    def test_known_key_mapped_into_sampling_params(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", options={"repetition_penalty": 1.1}))
        assert _sampling(body)["repetition_penalty"] == 1.1
        assert "repetition_penalty" not in body  # nested, never top-level

    def test_unknown_keys_dropped(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(_request("hi", options={"seed": 1, "custom_flag": True}))
        assert "seed" not in body
        assert "seed" not in _sampling(body)
        assert "custom_flag" not in body
        assert "custom_flag" not in _sampling(body)

    def test_explicit_ir_field_wins_over_dialect_options(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            _request(
                "hi",
                sampling=SamplingParams(max_tokens=7),
                options={"max_tokens": 99},
            )
        )
        assert _sampling(body)["max_new_tokens"] == 7


# ===================================================================
# Wire plumbing: URL, cast_to, body, provenance
# ===================================================================
class TestWirePlumbing:
    @pytest.mark.anyio
    async def test_posts_to_generate_url_with_body(self):
        t, post = _make_transport({"text": "out", "meta_info": _meta(prompt_tokens=1)})
        await t.arun(_request("hi"))
        assert post.call_args[0][0] == "http://host/generate"
        assert post.call_args[1]["cast_to"] is object
        body = post.call_args[1]["body"]
        assert body["text"] == "hi"
        assert "return_logprob" not in body

    @pytest.mark.anyio
    async def test_stream_flag_is_ignored_single_post(self):
        # Native /generate is always one POST; Request.stream never reaches
        # the wire body (pure scheduling, no content impact).
        t, post = _make_transport({"text": "x", "meta_info": _meta(prompt_tokens=1)})
        resp = await t.arun(_request("hi", stream=True))
        assert post.call_count == 1
        assert "stream" not in post.call_args[1]["body"]
        assert resp.texts == ("x",)

    @pytest.mark.anyio
    async def test_request_params_excludes_prompt_text(self):
        t, post = _make_transport(
            {"text": "x", "meta_info": _meta(prompt_tokens=1, completion_tokens=1)}
        )
        resp = await t.arun(
            _request("secret prompt", sampling=SamplingParams(temperature=0.0))
        )
        # the raw prompt is sent on the wire...
        assert post.call_args[1]["body"]["text"] == "secret prompt"
        # ...but is NOT persisted into per-call request_params.
        assert resp.request_params is not None
        assert "text" not in resp.request_params
        assert resp.request_params["sampling_params"] == {"temperature": 0.0}

    @pytest.mark.anyio
    async def test_explicit_seed_is_not_persisted_when_wire_cannot_transmit_it(self):
        t, post = _make_transport(
            {"text": "x", "meta_info": _meta(prompt_tokens=1, completion_tokens=1)}
        )

        resp = await t.arun(_request("prompt", sampling=SamplingParams(seed=7)))

        body = post.call_args.kwargs["body"]
        assert "seed" not in body
        assert "seed" not in _sampling(body)
        assert resp.request_params is not None
        assert "seed" not in resp.request_params
        sampling_params = resp.request_params["sampling_params"]
        assert isinstance(sampling_params, dict)
        assert "seed" not in sampling_params

    @pytest.mark.anyio
    async def test_absent_provider_model_id_stays_none(self):
        t, _ = _make_transport({"text": "x", "meta_info": _meta(prompt_tokens=1)})
        resp = await t.arun(_request("hi"))
        assert resp.response_model is None


# ===================================================================
# n>1: /generate returns a list of per-sample dicts
# ===================================================================
class TestNSamples:
    @pytest.mark.anyio
    async def test_list_response_yields_per_sample_texts(self):
        t, post = _make_transport(
            [
                {
                    "text": "a",
                    "meta_info": _meta(prompt_tokens=4, completion_tokens=1),
                },
                {
                    "text": "b",
                    "meta_info": _meta(
                        prompt_tokens=4,
                        completion_tokens=2,
                        finish_reason={"type": "length"},
                    ),
                },
                {
                    "text": "c",
                    "meta_info": _meta(prompt_tokens=4, completion_tokens=3),
                },
            ]
        )
        resp = await t.arun(_request("hi", sampling=SamplingParams(n=3)))
        assert resp.texts == ("a", "b", "c")
        assert resp.finish_reasons == ("stop", "length", "stop")
        # prompt tokens counted once, completions summed.
        assert resp.usage == UsageStats(
            input_tokens=4, output_tokens=6, total_tokens=10, cached_tokens=0
        )
        assert post.call_args[1]["body"]["sampling_params"]["n"] == 3

    @pytest.mark.anyio
    async def test_logprobs_are_lifted_per_sample(self):
        # Each sample carries its own meta_info; reading sample 0 only would
        # hand every rollout the first one's sequence.
        t, post = _make_transport(
            [
                {
                    "text": "a",
                    "meta_info": _meta(
                        output_token_logprobs=[[-0.1, 1, "Ġa"]],
                        output_top_logprobs=[[[-0.1, 1, "Ġa"], [-2.0, 9, "z"]]],
                    ),
                },
                {
                    "text": "bc",
                    "meta_info": _meta(
                        output_token_logprobs=[[-0.2, 2, "b"], [-0.3, 3, "c"]],
                        output_top_logprobs=[
                            [[-0.2, 2, "b"]],
                            [[-0.3, 3, "c"]],
                        ],
                    ),
                },
            ]
        )
        resp = await t.arun(
            _request(
                "x",
                sampling=SamplingParams(n=2),
                sampled_logprobs=True,
                top_logprobs=2,
            )
        )

        assert resp.logprobs is None
        assert resp.top_logprobs is None
        assert resp.choice_logprobs is not None
        first, second = resp.choice_logprobs
        assert first.tokens == (TokenLogprob(token=" a", logprob=-0.1, token_id=1),)
        assert first.top_logprobs == (
            (
                TopKEntry(token=" a", logprob=-0.1, token_id=1),
                TopKEntry(token="z", logprob=-2.0, token_id=9),
            ),
        )
        assert [item.token for item in second.tokens] == ["b", "c"]
        assert second.top_logprobs is not None
        assert len(second.top_logprobs) == 2
        body = post.call_args[1]["body"]
        assert body["return_logprob"] is True
        assert body["logprob_start_len"] == -1
        assert body["top_logprobs_num"] == 2

    @pytest.mark.anyio
    async def test_per_sample_logprobs_without_breadth_carry_no_top_k(self):
        t, _ = _make_transport(
            [
                {
                    "text": "a",
                    "meta_info": _meta(output_token_logprobs=[[-0.1, 1, "a"]]),
                },
                {"text": "", "meta_info": _meta(output_token_logprobs=[])},
            ]
        )
        resp = await t.arun(
            _request("x", sampling=SamplingParams(n=2), sampled_logprobs=True)
        )

        assert resp.choice_logprobs is not None
        assert [item.top_logprobs for item in resp.choice_logprobs] == [None, None]
        # A sample that generated nothing is an empty sequence, not an error.
        assert resp.choice_logprobs[1].tokens == ()

    @pytest.mark.anyio
    async def test_sample_missing_its_logprobs_raises(self):
        t, _ = _make_transport(
            [
                {
                    "text": "a",
                    "meta_info": _meta(output_token_logprobs=[[-0.1, 1, "a"]]),
                },
                {"text": "b", "meta_info": _meta()},
            ]
        )
        with pytest.raises(RuntimeError, match="no output_token_logprobs"):
            await t.arun(
                _request("x", sampling=SamplingParams(n=2), sampled_logprobs=True)
            )

    @pytest.mark.anyio
    async def test_sample_short_of_requested_top_k_raises(self):
        t, _ = _make_transport(
            [
                {
                    "text": "ab",
                    "meta_info": _meta(
                        output_token_logprobs=[[-0.1, 1, "a"], [-0.2, 2, "b"]],
                        output_top_logprobs=[[[-0.1, 1, "a"]]],
                    ),
                },
                {
                    "text": "c",
                    "meta_info": _meta(
                        output_token_logprobs=[[-0.3, 3, "c"]],
                        output_top_logprobs=[[[-0.3, 3, "c"]]],
                    ),
                },
            ]
        )
        with pytest.raises(RuntimeError, match="top logprobs for 1 of 2"):
            await t.arun(
                _request(
                    "x",
                    sampling=SamplingParams(n=2),
                    sampled_logprobs=True,
                    top_logprobs=1,
                )
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize("sampled_logprobs", [False, True])
    async def test_sample_count_short_of_n_raises(self, sampled_logprobs):
        # A lone dict for n=2 must not pass as one rollout with singular logprobs.
        t, _ = _make_transport(
            {
                "text": "a",
                "meta_info": _meta(output_token_logprobs=[[-0.1, 1, "a"]]),
            }
        )
        with pytest.raises(RuntimeError, match="returned 1 samples for n=2"):
            await t.arun(
                _request(
                    "x",
                    sampling=SamplingParams(n=2),
                    sampled_logprobs=sampled_logprobs,
                )
            )


class TestLift:
    @pytest.mark.anyio
    async def test_triple_populates_token_id(self):
        meta = _meta(
            prompt_tokens=1,
            output_token_logprobs=[[-0.5, 42, "hello"]],
        )
        t, _ = _make_transport({"text": "hello", "meta_info": meta})
        resp = await t.arun(_request("x", sampled_logprobs=True, top_logprobs=0))
        assert resp.logprobs is not None
        assert resp.logprobs[0].token_id == 42
        assert resp.logprobs[0].token == "hello"
        assert resp.logprobs[0].logprob == -0.5

    @pytest.mark.anyio
    async def test_normalize_token_text_called(self):
        meta = _meta(output_token_logprobs=[[-0.1, 7, "ĠA"]])  # "ĠA"
        t, _ = _make_transport({"text": "", "meta_info": meta})
        resp = await t.arun(_request("x", sampled_logprobs=True))
        assert resp.logprobs is not None
        assert resp.logprobs[0].token == " A"  # Ġ normalized to space

    @pytest.mark.anyio
    async def test_newline_marker_normalized_in_tokens(self):
        meta = _meta(output_token_logprobs=[[-0.2, 9, "Ċ"]])
        t, _ = _make_transport({"text": "\n", "meta_info": meta})
        resp = await t.arun(_request("x", sampled_logprobs=True))
        assert resp.logprobs == (TokenLogprob(token="\n", logprob=-0.2, token_id=9),)

    @pytest.mark.anyio
    async def test_input_token_logprobs_go_to_input_scoring(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "a"], [-1.2, 2, "b"]],
            output_token_logprobs=[[-0.3, 3, "c"]],
            cached_tokens=0,
        )
        t, _ = _make_transport({"text": "c", "meta_info": meta})
        resp = await t.arun(_request("ab", input_scoring=True, top_logprobs=0))
        assert resp.input_scoring is not None
        assert len(resp.input_scoring.token_logprobs) == 2
        assert resp.input_scoring.token_logprobs[0].logprob is None  # first token
        assert resp.input_scoring.token_logprobs[1].token_id == 2
        # output segment lands in logprobs, never merged with the input segment
        assert resp.logprobs is not None
        assert len(resp.logprobs) == 1
        assert resp.logprobs[0].token == "c"

    @pytest.mark.anyio
    async def test_usage_and_finish_reason(self):
        meta = _meta(
            prompt_tokens=4, completion_tokens=6, finish_reason={"type": "length"}
        )
        t, _ = _make_transport({"text": "out", "meta_info": meta})
        resp = await t.arun(_request("x"))
        assert resp.usage == UsageStats(
            input_tokens=4, output_tokens=6, total_tokens=10, cached_tokens=0
        )
        assert resp.finish_reasons == ("length",)
        assert resp.texts == ("out",)

    @pytest.mark.anyio
    async def test_absent_cached_tokens_reads_as_unreported(self):
        """An omitted key is silence; the 0 above is a measured cache miss."""
        meta = _meta(prompt_tokens=4, completion_tokens=6)
        del meta["cached_tokens"]
        t, _ = _make_transport({"text": "out", "meta_info": meta})

        resp = await t.arun(_request("x"))

        assert resp.usage is not None
        assert resp.usage.cached_tokens is None

    @pytest.mark.anyio
    async def test_cached_tokens_are_not_summed_across_samples(self):
        """They describe the shared prefix, like prompt_tokens -- read once."""
        t, _ = _make_transport(
            [
                {"text": "a", "meta_info": _meta(prompt_tokens=4, cached_tokens=3)},
                {"text": "b", "meta_info": _meta(prompt_tokens=4, cached_tokens=3)},
            ]
        )

        resp = await t.arun(_request("x", sampling=SamplingParams(n=2)))

        assert resp.usage is not None
        assert resp.usage.cached_tokens == 3

    @pytest.mark.anyio
    async def test_usage_none_when_prompt_tokens_absent(self):
        # echo=False path (no radix guard); _parse_usage returns None when the
        # server omitted the token counts — absence is not zeros.
        meta = {
            "completion_tokens": 1,
            "finish_reason": "stop",
            "output_token_logprobs": [[-0.1, 1, " A"]],
        }
        t, _ = _make_transport({"text": " A", "meta_info": meta})
        resp = await t.arun(_request("p", sampled_logprobs=True))
        assert resp.usage is None


# ===================================================================
# Top-k triple parsing
# ===================================================================
class TestTopKParsing:
    @pytest.mark.anyio
    async def test_top_k_preserves_token_id_and_normalizes_text(self):
        meta = _meta(
            output_token_logprobs=[[-0.7, 100, "ĠB"]],
            output_top_logprobs=[[[-0.7, 100, "ĠB"], [-1.2, 101, "Ċ"]]],
        )
        t, _ = _make_transport({"text": " B", "meta_info": meta})
        resp = await t.arun(_request("p", sampled_logprobs=True, top_logprobs=2))
        assert resp.top_logprobs == (
            (
                TopKEntry(token=" B", logprob=-0.7, token_id=100),
                TopKEntry(token="\n", logprob=-1.2, token_id=101),
            ),
        )

    @pytest.mark.anyio
    async def test_empty_per_token_top_entry_becomes_empty_tuple(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "Q"], [-0.2, 2, " B"]],
            input_top_logprobs=[None, [[-0.2, 2, " B"]]],
            output_token_logprobs=[[-0.3, 3, " g"]],
            output_top_logprobs=[None, [[-0.3, 3, " g"]]],
        )
        t, _ = _make_transport({"text": " g", "meta_info": meta})
        resp = await t.arun(
            _request("Q B", input_scoring=True, sampled_logprobs=True, top_logprobs=1)
        )
        assert resp.top_logprobs == (
            (),
            (TopKEntry(token=" g", logprob=-0.3, token_id=3),),
        )

    @pytest.mark.anyio
    async def test_input_top_logprobs_go_to_input_scoring(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "Q"], [-0.2, 2, "ĠB"]],
            input_top_logprobs=[None, [[-0.2, 2, "ĠB"], [-1.5, 3, "ĠC"]]],
            output_token_logprobs=[[-0.3, 4, " g"]],
            output_top_logprobs=[[[-0.3, 4, " g"], [-2.0, 5, " h"]]],
        )
        t, post = _make_transport({"text": " g", "meta_info": meta})
        resp = await t.arun(
            _request("Q B", input_scoring=True, sampled_logprobs=True, top_logprobs=2)
        )

        body = post.call_args[1]["body"]
        assert body["logprob_start_len"] == 0
        assert body["top_logprobs_num"] == 2
        assert resp.input_scoring is not None
        assert resp.input_scoring.top_logprobs == (
            (),
            (
                TopKEntry(token=" B", logprob=-0.2, token_id=2),
                TopKEntry(token=" C", logprob=-1.5, token_id=3),
            ),
        )

    @pytest.mark.anyio
    async def test_input_top_logprobs_absent_unless_requested(self):
        meta = _meta(
            prompt_tokens=1,
            input_token_logprobs=[[None, 1, "Q"]],
            input_top_logprobs=[None],
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        resp = await t.arun(_request("Q", input_scoring=True))
        assert resp.input_scoring is not None
        assert resp.input_scoring.top_logprobs is None

    @pytest.mark.anyio
    @pytest.mark.parametrize("input_top", [None, [None]])
    async def test_misaligned_input_top_logprobs_raise(self, input_top):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "Q"], [-0.2, 2, " B"]],
            input_top_logprobs=input_top,
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="input top-k logprobs"):
            await t.arun(
                _request(
                    "Q B", input_scoring=True, sampled_logprobs=True, top_logprobs=2
                )
            )

    @pytest.mark.anyio
    async def test_top_logprobs_none_when_absent(self):
        meta = _meta(output_token_logprobs=[[-0.1, 1, " A"]])
        t, _ = _make_transport({"text": " A", "meta_info": meta})
        resp = await t.arun(_request("p", sampled_logprobs=True))
        assert resp.top_logprobs is None

    @pytest.mark.anyio
    async def test_none_token_text_in_top_raises(self):
        """A top-k entry with no token text (no detokenization) fails loud."""
        meta = _meta(
            output_token_logprobs=[[-0.1, 1, " A"]],
            output_top_logprobs=[[[-0.1, 1, None]]],
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="no token text"):
            await t.arun(_request("p", sampled_logprobs=True, top_logprobs=1))


# ===================================================================
# Guards: response shape, radix cache, empty logprobs
# ===================================================================
class TestGuards:
    @pytest.mark.anyio
    async def test_radix_cache_hit_with_score_input_raises(self):
        meta = _meta(
            prompt_tokens=3,
            input_token_logprobs=[[None, 1, "a"]],  # truncated (cache hit)
            cached_tokens=2,
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="radix prefix cache"):
            await t.arun(_request("abc", input_scoring=True))

    @pytest.mark.anyio
    async def test_truncated_input_logprobs_raises(self):
        # cached_tokens reads 0, but the returned count < prompt_tokens.
        meta = _meta(
            prompt_tokens=5,
            input_token_logprobs=[[-0.1, 1, " star"]],
            cached_tokens=0,
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="partial echoed-input"):
            await t.arun(_request("abcde", input_scoring=True))

    @pytest.mark.anyio
    async def test_missing_prompt_tokens_with_score_input_raises(self):
        meta = {"completion_tokens": 1, "finish_reason": "stop"}
        meta["input_token_logprobs"] = [[None, 1, "a"]]
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="omitted prompt_tokens"):
            await t.arun(_request("a", input_scoring=True))

    @pytest.mark.anyio
    async def test_full_echoed_input_passes_guard(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "a"], [-0.1, 2, " b"]],
            cached_tokens=0,
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        resp = await t.arun(_request("ab", input_scoring=True))
        assert resp.input_scoring is not None
        assert [tl.token for tl in resp.input_scoring.token_logprobs] == ["a", " b"]

    @pytest.mark.anyio
    async def test_no_score_input_skips_radix_guard(self):
        """Output-only logprobs (CMMLU shape) — cache truncation is irrelevant."""
        meta = _meta(
            prompt_tokens=5,
            cached_tokens=4,
            output_token_logprobs=[[-0.1, 1, " A"]],
        )
        t, _ = _make_transport({"text": " A", "meta_info": meta})
        resp = await t.arun(_request("p", sampled_logprobs=True))
        assert resp.logprobs is not None
        assert resp.logprobs[0].token == " A"

    @pytest.mark.anyio
    async def test_no_logprobs_channels_raises(self):
        """Logprobs requested but every channel came back empty → fail loud."""
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        with pytest.raises(RuntimeError, match="no logprobs"):
            await t.arun(_request("x", sampled_logprobs=True))

    @pytest.mark.anyio
    async def test_empty_echoed_input_raises(self):
        # score_input with an empty echoed input passes the radix guard
        # (0 == prompt_tokens) but still yields no logprobs at all.
        meta = _meta(prompt_tokens=0, input_token_logprobs=[])
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="no logprobs"):
            await t.arun(_request("p", input_scoring=True))

    @pytest.mark.anyio
    async def test_missing_meta_info_raises(self):
        t, _ = _make_transport({"text": "hi"})
        with pytest.raises(RuntimeError, match="missing meta_info"):
            await t.arun(_request("x"))

    @pytest.mark.anyio
    async def test_list_with_non_dict_element_raises(self):
        t, _ = _make_transport(["nope"])
        with pytest.raises(RuntimeError, match="missing meta_info"):
            await t.arun(_request("x"))

    @pytest.mark.anyio
    async def test_empty_list_response_raises(self):
        t, _ = _make_transport([])
        with pytest.raises(RuntimeError, match="missing meta_info"):
            await t.arun(_request("x"))
