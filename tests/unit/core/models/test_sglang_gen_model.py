"""
Unit tests for sieval/core/models/sglang_gen_model.py.

Covers native /generate generation (_agenerate_impl) and logprob extraction
(_alogprobs_impl): URL derivation, request body, sampling-param translation,
echo→logprob_start_len, input/output token-logprob + top-logprob parsing,
token-text normalization, end-to-end extract_option_logprob / total_logprob /
CMMLU-style top-k consumption, and the n / empty-response guards. The OpenAI
client's public ``post`` is mocked — no real traffic.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from sieval.core.models.model import ModelOutput
from sieval.core.models.sglang_gen_model import (
    SglangGenModel,
    _normalize_token_text,
)
from sieval.core.utils.ppl import extract_option_logprob, total_logprob


@pytest.fixture
def model():
    return SglangGenModel(
        model="test-sglang", api_base="http://host:8000/v1", api_key="local"
    )


def _patch_post(model: SglangGenModel, payload):
    """Mock the OpenAI client's public ``post`` to return ``payload``."""
    mock_post = AsyncMock(return_value=payload)
    target: Any = model._client
    target.post = mock_post  # type: ignore[invalid-assignment]
    return mock_post


def _meta(
    input_entries=None, output_entries=None, input_top=None, output_top=None, **extra
):
    meta: dict[str, Any] = {}
    if input_entries is not None:
        meta["input_token_logprobs"] = input_entries
    if output_entries is not None:
        meta["output_token_logprobs"] = output_entries
    if input_top is not None:
        meta["input_top_logprobs"] = input_top
    if output_top is not None:
        meta["output_top_logprobs"] = output_top
    meta.update(extra)
    return meta


# ===================================================================
# URL derivation
# ===================================================================
class TestGenerateUrl:
    def test_strips_v1_suffix(self, model):
        assert model._generate_url() == "http://host:8000/generate"

    def test_trailing_slash_base(self):
        m = SglangGenModel(model="x", api_base="http://host:8000/v1/", api_key="local")
        assert m._generate_url() == "http://host:8000/generate"

    def test_no_v1_suffix(self):
        m = SglangGenModel(model="x", api_base="http://host:8000", api_key="local")
        assert m._generate_url() == "http://host:8000/generate"

    def test_none_base(self):
        m = SglangGenModel(model="x", api_key="local")
        assert m._generate_url() == "/generate"


# ===================================================================
# Token text normalization
# ===================================================================
class TestNormalizeTokenText:
    def test_space_marker(self):
        assert _normalize_token_text("ĠA") == " A"

    def test_newline_marker(self):
        assert _normalize_token_text("Ċ") == "\n"

    def test_plain_unchanged(self):
        assert _normalize_token_text(" A") == " A"


# ===================================================================
# _agenerate_impl (native /generate generation)
# ===================================================================
class TestAgenerate:
    @pytest.mark.anyio
    async def test_basic_generation(self, model):
        post = _patch_post(
            model,
            {
                "text": "hello world",
                "meta_info": _meta(prompt_tokens=5, completion_tokens=2),
            },
        )
        out = await model._agenerate_impl("hi")
        assert isinstance(out, ModelOutput)
        assert out.texts == ["hello world"]
        assert out.usage == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        # posts to /generate with the text; no return_logprob for plain generation.
        assert post.call_args[0][0] == "http://host:8000/generate"
        body = post.call_args[1]["body"]
        assert body["text"] == "hi"
        assert "return_logprob" not in body

    @pytest.mark.anyio
    async def test_non_string_prompt_raises(self, model):
        with pytest.raises(TypeError, match="requires a string"):
            await model._agenerate_impl(["not", "a", "string"])

    @pytest.mark.anyio
    async def test_sampling_param_translation(self, model):
        post = _patch_post(model, {"text": "x", "meta_info": _meta()})
        await model._agenerate_impl(
            "hi", max_tokens=64, temperature=0.7, top_p=0.9, seed=123
        )
        sp = post.call_args[1]["body"]["sampling_params"]
        assert sp["max_new_tokens"] == 64
        assert sp["temperature"] == 0.7
        assert sp["top_p"] == 0.9
        # unmapped kwargs (seed) are dropped, not forwarded to sglang.
        assert "seed" not in sp

    @pytest.mark.anyio
    async def test_n_gt_1_list_response(self, model):
        post = _patch_post(
            model,
            [
                {"text": "a", "meta_info": _meta(prompt_tokens=4, completion_tokens=1)},
                {"text": "b", "meta_info": _meta(prompt_tokens=4, completion_tokens=2)},
            ],
        )
        out = await model._agenerate_impl("hi", n=2)
        assert out.texts == ["a", "b"]
        # prompt tokens counted once, completions summed
        assert out.usage == {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}
        assert post.call_args[1]["body"]["sampling_params"]["n"] == 2

    @pytest.mark.anyio
    async def test_finish_reason_extracted(self, model):
        _patch_post(
            model,
            {"text": "x", "meta_info": _meta(finish_reason={"type": "length"})},
        )
        out = await model._agenerate_impl("hi")
        assert out.finish_reasons == ["length"]

    @pytest.mark.anyio
    async def test_n_non_int_raises(self, model):
        with pytest.raises(TypeError, match="n must be an int"):
            await model._agenerate_impl("hi", n="2")

    @pytest.mark.anyio
    async def test_n_lt_1_raises(self, model):
        with pytest.raises(ValueError, match="n must be >= 1"):
            await model._agenerate_impl("hi", n=0)


# ===================================================================
# _alogprobs_impl request body
# ===================================================================
class TestRequestBody:
    @pytest.mark.anyio
    async def test_echo_true_request_body(self, model):
        post = _patch_post(
            model, {"text": "", "meta_info": _meta(input_entries=[[-0.1, 1, " A"]])}
        )
        await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5, echo=True)
        body = post.call_args[1]["body"]
        assert body["text"] == "prompt"
        assert body["return_logprob"] is True
        assert body["logprob_start_len"] == 0
        assert body["top_logprobs_num"] == 5
        assert body["return_text_in_logprobs"] is True
        assert body["sampling_params"]["max_new_tokens"] == 1
        assert body["sampling_params"]["temperature"] == 0.0
        # routed through the public client with an absolute URL.
        assert post.call_args[0][0] == "http://host:8000/generate"
        assert post.call_args[1]["cast_to"] is object

    @pytest.mark.anyio
    async def test_echo_false_sets_start_len_minus_one(self, model):
        post = _patch_post(
            model, {"text": "", "meta_info": _meta(output_entries=[[-0.1, 1, "x"]])}
        )
        await model._alogprobs_impl("prompt", echo=False)
        assert post.call_args[1]["body"]["logprob_start_len"] == -1

    @pytest.mark.anyio
    async def test_max_tokens_floored_to_one(self, model):
        post = _patch_post(
            model, {"text": "", "meta_info": _meta(input_entries=[[-0.1, 1, " A"]])}
        )
        await model._alogprobs_impl("prompt", max_tokens=0)
        assert post.call_args[1]["body"]["sampling_params"]["max_new_tokens"] == 1


# ===================================================================
# Chosen-token logprob parsing
# ===================================================================
class TestParsing:
    @pytest.mark.anyio
    async def test_input_logprobs_to_tokens_and_logprobs(self, model):
        meta = _meta(
            input_entries=[[None, 1, "The"], [-0.5, 2, " cat"], [-0.1, 3, " A"]],
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.logprobs_tokens == ["The", " cat", " A"]
        assert out.logprobs == [None, -0.5, -0.1]

    @pytest.mark.anyio
    async def test_token_text_normalized(self, model):
        meta = _meta(input_entries=[[None, 1, "ĠThe"], [-0.1, 2, "ĠA"]])
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.logprobs_tokens == [" The", " A"]

    @pytest.mark.anyio
    async def test_array_ordering_input_then_output(self, model):
        meta = _meta(
            input_entries=[[None, 1, "Q"], [-0.2, 2, " B"]],
            output_entries=[[-0.3, 3, " gen"]],
        )
        _patch_post(model, {"text": " gen", "meta_info": meta})
        out = await model._alogprobs_impl("prompt", echo=True)
        assert out.logprobs_tokens == ["Q", " B", " gen"]
        assert out.logprobs == [None, -0.2, -0.3]

    @pytest.mark.anyio
    async def test_finish_reason_on_logprobs(self, model):
        meta = _meta(input_entries=[[-0.1, 1, " A"]], finish_reason={"type": "length"})
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.finish_reasons == ["length"]

    @pytest.mark.anyio
    async def test_usage_parsed(self, model):
        meta = _meta(
            input_entries=[[-0.1, 1, " A"]], prompt_tokens=7, completion_tokens=1
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.usage == {"input_tokens": 7, "output_tokens": 1, "total_tokens": 8}

    @pytest.mark.anyio
    async def test_usage_none_when_counts_absent(self, model):
        meta = _meta(input_entries=[[-0.1, 1, " A"]])
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.usage is None


# ===================================================================
# Top-k logprob parsing (CMMLU / MMLU-Base consumption)
# ===================================================================
class TestTopLogprobs:
    @pytest.mark.anyio
    async def test_output_top_logprobs_echo_false(self, model):
        """CMMLU shape: echo=False, first output token's top-k as {token: logprob}."""
        meta = _meta(
            output_entries=[[-0.7, 100, " A"]],
            output_top=[[[-0.7, 100, " A"], [-1.2, 101, " B"], [-3.0, 102, " C"]]],
        )
        _patch_post(model, {"text": " A", "meta_info": meta})
        out = await model._alogprobs_impl("prompt", echo=False)
        assert out.top_logprobs == [{" A": -0.7, " B": -1.2, " C": -3.0}]

    @pytest.mark.anyio
    async def test_top_logprobs_normalized_keys(self, model):
        meta = _meta(
            output_entries=[[-0.7, 100, "ĠA"]],
            output_top=[[[-0.7, 100, "ĠA"], [-1.2, 101, "ĠB"]]],
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt", echo=False)
        assert out.top_logprobs == [{" A": -0.7, " B": -1.2}]

    @pytest.mark.anyio
    async def test_top_logprobs_echo_aligns_input_first(self, model):
        """echo=True: input top-k precede output; None/empty first entry → {}."""
        meta = _meta(
            input_entries=[[None, 1, "Q"], [-0.2, 2, " B"]],
            output_entries=[[-0.3, 3, " g"]],
            input_top=[None, [[-0.2, 2, " B"], [-0.9, 9, " C"]]],
            output_top=[[[-0.3, 3, " g"]]],
        )
        _patch_post(model, {"text": " g", "meta_info": meta})
        out = await model._alogprobs_impl("prompt", echo=True)
        assert out.top_logprobs == [{}, {" B": -0.2, " C": -0.9}, {" g": -0.3}]

    @pytest.mark.anyio
    async def test_top_logprobs_none_when_absent(self, model):
        meta = _meta(input_entries=[[-0.1, 1, " A"]])
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.top_logprobs is None

    @pytest.mark.anyio
    async def test_cmmlu_style_scoring(self, model):
        """Map first output token's top-k onto A/B/C/D (echo=False, CMMLU shape)."""
        meta = _meta(
            output_entries=[[-0.7, 100, " B"]],
            output_top=[
                [[-2.0, 1, " A"], [-0.7, 2, " B"], [-3.0, 3, " C"], [-2.5, 4, " D"]]
            ],
        )
        _patch_post(model, {"text": " B", "meta_info": meta})
        out = await model._alogprobs_impl("prompt", echo=False, logprobs=100)
        scores = {tok.strip(): lp for tok, lp in (out.top_logprobs or [{}])[0].items()}
        assert max(scores, key=lambda k: scores[k]) == "B"

    @pytest.mark.anyio
    async def test_distinct_tokens_stripping_to_same_letter_both_kept(self, model):
        """Two tokens stripping to the same letter (' B' vs '\\tB') stay distinct.

        Observed live on Qwen2.5-72B: the greedy ' B' (high logprob) and a
        rare '\\tB' (very low) both strip to 'B'. They have different
        normalized text, so they must remain SEPARATE dict entries — that is
        what lets CMMLU's ``max``-over-strip recover the high logprob rather
        than clobbering it with the low one.
        """
        meta = _meta(
            output_entries=[[-0.007, 425, " B"]],
            output_top=[[[-0.007, 425, " B"], [-11.94, 12791, "\tB"]]],
        )
        _patch_post(model, {"text": " B", "meta_info": meta})
        out = await model._alogprobs_impl("prompt", echo=False, logprobs=100)
        assert out.top_logprobs == [{" B": -0.007, "\tB": -11.94}]
        # A max-over-strip consumer (CMMLU) recovers the high logprob.
        best = max(
            (lp for tok, lp in out.top_logprobs[0].items() if tok.strip() == "B")
        )
        assert best == -0.007


# ===================================================================
# End-to-end consumption by echo-based ppl utilities
# ===================================================================
class TestPplConsumption:
    @pytest.mark.anyio
    async def test_extract_option_logprob_finds_letter(self, model):
        meta = _meta(
            input_entries=[
                [None, 1, "Question:"],
                [-2.0, 2, " text"],
                [-0.7, 3, " A"],
            ],
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("Question: text A", echo=True)
        assert extract_option_logprob(out.logprobs_tokens, out.logprobs, "A") == -0.7

    @pytest.mark.anyio
    async def test_total_logprob_sums_continuation(self, model):
        meta = _meta(
            input_entries=[[None, 1, "Ctx"], [-1.0, 2, " the"], [-2.0, 3, " end"]],
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("Ctx the end", echo=True)
        total, count = total_logprob(out.logprobs_tokens, out.logprobs)
        assert total == pytest.approx(-3.0)
        assert count == 2


# ===================================================================
# Guards
# ===================================================================
class TestGuards:
    @pytest.mark.anyio
    async def test_n_gt_1_raises(self, model):
        post = _patch_post(model, {"text": "", "meta_info": _meta()})
        with pytest.raises(ValueError, match="only supports n=1"):
            await model._alogprobs_impl("prompt", n=2)
        post.assert_not_called()

    @pytest.mark.anyio
    async def test_n_non_int_raises(self, model):
        with pytest.raises(TypeError, match="n must be an int"):
            await model._alogprobs_impl("prompt", n="2")

    @pytest.mark.anyio
    async def test_n_bool_raises(self, model):
        with pytest.raises(TypeError, match="n must be an int"):
            await model._alogprobs_impl("prompt", n=True)

    @pytest.mark.anyio
    async def test_empty_response_raises(self, model):
        """No token logprobs AND no top logprobs → raise (retryable failure)."""
        _patch_post(model, {"text": "", "meta_info": _meta(input_entries=[])})
        with pytest.raises(RuntimeError, match="no logprobs"):
            await model._alogprobs_impl("prompt")

    @pytest.mark.anyio
    async def test_meta_attached(self, model):
        _patch_post(
            model, {"text": "", "meta_info": _meta(input_entries=[[-0.1, 1, " A"]])}
        )
        out = await model._alogprobs_impl("prompt")
        assert out.model["model"] == "test-sglang"
        assert out.response_model == "test-sglang"
