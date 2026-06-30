"""
Unit tests for sieval/core/models/sglang_gen_model.py.

Covers: /generate URL derivation, request body, echo→logprob_start_len,
input_token_logprobs parsing, token-text normalization, end-to-end
extract_option_logprob / total_logprob consumption, array ordering, and the
n>1 / empty-logprobs / max_tokens guards. The httpx POST is mocked — no real
traffic.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

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


def _patch_post(model: SglangGenModel, payload: dict):
    """Mock the underlying httpx POST to return ``payload`` as JSON."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    mock_post = AsyncMock(return_value=resp)
    target: Any = model._client._client
    target.post = mock_post  # type: ignore[invalid-assignment]
    return mock_post


def _meta(input_entries=None, output_entries=None, **extra):
    meta: dict[str, Any] = {}
    if input_entries is not None:
        meta["input_token_logprobs"] = input_entries
    if output_entries is not None:
        meta["output_token_logprobs"] = output_entries
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
# Request body
# ===================================================================
class TestRequestBody:
    @pytest.mark.anyio
    async def test_echo_true_request_body(self, model):
        post = _patch_post(
            model, {"text": "", "meta_info": _meta(input_entries=[[-0.1, 1, " A"]])}
        )
        await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5, echo=True)
        body = post.call_args[1]["json"]
        assert body["text"] == "prompt"
        assert body["return_logprob"] is True
        assert body["logprob_start_len"] == 0
        assert body["top_logprobs_num"] == 5
        assert body["return_text_in_logprobs"] is True
        assert body["sampling_params"]["max_new_tokens"] == 1

    @pytest.mark.anyio
    async def test_echo_false_sets_start_len_minus_one(self, model):
        post = _patch_post(
            model, {"text": "", "meta_info": _meta(output_entries=[[-0.1, 1, "x"]])}
        )
        await model._alogprobs_impl("prompt", echo=False)
        assert post.call_args[1]["json"]["logprob_start_len"] == -1

    @pytest.mark.anyio
    async def test_max_tokens_floored_to_one(self, model):
        post = _patch_post(
            model, {"text": "", "meta_info": _meta(input_entries=[[-0.1, 1, " A"]])}
        )
        await model._alogprobs_impl("prompt", max_tokens=0)
        assert post.call_args[1]["json"]["sampling_params"]["max_new_tokens"] == 1

    @pytest.mark.anyio
    async def test_url_targeted(self, model):
        post = _patch_post(
            model, {"text": "", "meta_info": _meta(input_entries=[[-0.1, 1, " A"]])}
        )
        await model._alogprobs_impl("prompt")
        assert post.call_args[0][0] == "http://host:8000/generate"


# ===================================================================
# Parsing
# ===================================================================
class TestParsing:
    @pytest.mark.anyio
    async def test_input_logprobs_to_tokens_and_logprobs(self, model):
        meta = _meta(
            input_entries=[[None, 1, "The"], [-0.5, 2, " cat"], [-0.1, 3, " A"]],
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert isinstance(out, ModelOutput)
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
        """echo=True keeps candidate tokens (output) after the input segment."""
        meta = _meta(
            input_entries=[[None, 1, "Q"], [-0.2, 2, " B"]],
            output_entries=[[-0.3, 3, " gen"]],
        )
        _patch_post(model, {"text": " gen", "meta_info": meta})
        out = await model._alogprobs_impl("prompt", echo=True)
        assert out.logprobs_tokens == ["Q", " B", " gen"]
        assert out.logprobs == [None, -0.2, -0.3]

    @pytest.mark.anyio
    async def test_usage_parsed(self, model):
        meta = _meta(
            input_entries=[[-0.1, 1, " A"]],
            prompt_tokens=7,
            completion_tokens=1,
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.usage == {
            "input_tokens": 7,
            "output_tokens": 1,
            "total_tokens": 8,
        }

    @pytest.mark.anyio
    async def test_usage_none_when_counts_absent(self, model):
        meta = _meta(input_entries=[[-0.1, 1, " A"]])
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("prompt")
        assert out.usage is None


# ===================================================================
# End-to-end consumption by ppl utilities
# ===================================================================
class TestPplConsumption:
    @pytest.mark.anyio
    async def test_extract_option_logprob_finds_letter(self, model):
        """A sequence ending in ' A' must yield the letter's logprob."""
        meta = _meta(
            input_entries=[
                [None, 1, "Question:"],
                [-2.0, 2, " text"],
                [-0.7, 3, " A"],
            ],
        )
        _patch_post(model, {"text": "", "meta_info": meta})
        out = await model._alogprobs_impl("Question: text A", echo=True)
        lp = extract_option_logprob(out.logprobs_tokens, out.logprobs, "A")
        assert lp == -0.7

    @pytest.mark.anyio
    async def test_total_logprob_sums_continuation(self, model):
        """total_logprob skips the leading None and sums the rest."""
        meta = _meta(
            input_entries=[
                [None, 1, "Ctx"],
                [-1.0, 2, " the"],
                [-2.0, 3, " end"],
            ],
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
    async def test_empty_logprobs_raises(self, model):
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
