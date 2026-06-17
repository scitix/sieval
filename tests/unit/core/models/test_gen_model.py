"""
Unit tests for sieval/core/models/gen_model.py.

Covers: _agenerate_impl (string prompt, non-string raises, streaming
accumulation, n>1 choices, usage), _alogprobs_impl (streaming logprobs,
no-logprobs raises, usage).

All OpenAI client calls are mocked — no real API traffic.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sieval.core.models.gen_model import GenModel
from sieval.core.models.model import ModelOutput


# ---------------------------------------------------------------------------
# Async streaming helpers
# ---------------------------------------------------------------------------
class _AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as e:
            raise StopAsyncIteration from e


def _make_completion_chunk(index=0, text="", finish_reason=""):
    chunk = MagicMock()
    chunk.usage = None
    choice = MagicMock()
    choice.index = index
    choice.text = text
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


def _make_usage_chunk(prompt_tokens=10, completion_tokens=5):
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    chunk.usage.total_tokens = prompt_tokens + completion_tokens
    return chunk


def _make_usage(prompt_tokens=10, completion_tokens=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    return usage


def _make_non_stream_response(
    *,
    text: str | None = "",
    finish_reason: str | None = "stop",
    usage=None,
    with_logprobs: bool = False,
    logprob: float = -0.1,
):
    resp = MagicMock()
    choice = MagicMock()
    choice.index = 0
    choice.text = text
    choice.finish_reason = finish_reason
    if with_logprobs:
        lp_obj = MagicMock()
        lp_obj.tokens = [text]
        lp_obj.token_logprobs = [logprob]
        choice.logprobs = lp_obj
    else:
        choice.logprobs = None
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_non_stream_choice(
    *,
    index: int,
    text: str,
    token_logprobs: list[float] | None,
    tokens: list[str] | None = None,
):
    choice = MagicMock()
    choice.index = index
    choice.text = text
    choice.finish_reason = "stop"
    if token_logprobs is None:
        choice.logprobs = None
    else:
        lp_obj = MagicMock()
        lp_obj.tokens = tokens or [text]
        lp_obj.token_logprobs = token_logprobs
        choice.logprobs = lp_obj
    return choice


# ---------------------------------------------------------------------------
# Concrete GenModel (delegates to parent _agenerate_impl)
# ---------------------------------------------------------------------------
@pytest.fixture
def model():
    return GenModel(model="test-gen", api_key="fake")


def _patch_create(model: GenModel, chunks):
    mock_create = AsyncMock(return_value=_AsyncIterator(chunks))
    target: Any = model._client.completions
    target.create = mock_create  # type: ignore[invalid-assignment]
    return mock_create


# ===================================================================
# _agenerate_impl
# ===================================================================
class TestGenAGenerate:
    @pytest.mark.anyio
    async def test_basic_string_prompt(self, model):
        chunks = [
            _make_completion_chunk(text="Hello"),
            _make_completion_chunk(text=" world", finish_reason="stop"),
            _make_usage_chunk(5, 2),
        ]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("What is 1+1?")
        assert isinstance(out, ModelOutput)
        assert out.texts == ["Hello world"]
        assert out.finish_reasons == ["stop"]

    @pytest.mark.anyio
    async def test_non_string_prompt_raises(self, model):
        with pytest.raises(TypeError, match="GenModel requires a string"):
            await model._agenerate_impl(["not", "a", "string"])

    @pytest.mark.anyio
    async def test_usage_captured(self, model):
        chunks = [
            _make_completion_chunk(text="ok", finish_reason="stop"),
            _make_usage_chunk(7, 3),
        ]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt")
        assert out.usage is not None
        assert out.usage["input_tokens"] == 7
        assert out.usage["output_tokens"] == 3
        assert out.usage["total_tokens"] == 10

    @pytest.mark.anyio
    async def test_no_usage_chunk(self, model):
        chunks = [_make_completion_chunk(text="hi", finish_reason="stop")]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt")
        assert out.usage is None

    @pytest.mark.anyio
    async def test_model_meta_attached(self, model):
        chunks = [_make_completion_chunk(text="x", finish_reason="stop")]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt")
        assert out.model["model"] == "test-gen"

    @pytest.mark.anyio
    async def test_request_params_captured(self, model):
        chunks = [_make_completion_chunk(text="x", finish_reason="stop")]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt", temperature=0.5)
        assert out.request_params is not None
        assert out.request_params.get("temperature") == 0.5

    @pytest.mark.anyio
    async def test_two_choices_in_single_chunk(self, model):
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [
            _make_completion_chunk(index=0, text="A1").choices[0],
            _make_completion_chunk(index=1, text="B1").choices[0],
        ]

        chunk2 = MagicMock()
        chunk2.usage = None
        chunk2.choices = [
            _make_completion_chunk(index=0, text="A2", finish_reason="stop").choices[0],
            _make_completion_chunk(index=1, text="B2", finish_reason="stop").choices[0],
        ]

        _patch_create(model, [chunk1, chunk2])
        out = await model._agenerate_impl("prompt", n=2)
        assert out.texts[0] == "A1A2"
        assert out.texts[1] == "B1B2"

    @pytest.mark.anyio
    async def test_non_stream_mode_supported(self, model):
        response = _make_non_stream_response(
            text="done",
            finish_reason="stop",
            usage=_make_usage(6, 2),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)

        assert out.texts == ["done"]
        assert out.finish_reasons == ["stop"]
        assert out.usage is not None
        assert out.usage["input_tokens"] == 6
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stream"] is False

    @pytest.mark.anyio
    async def test_non_stream_missing_finish_reason_defaults_to_empty(self, model):
        response = _make_non_stream_response(
            text="done",
            finish_reason=None,
            usage=_make_usage(3, 1),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)
        assert out.texts == ["done"]
        assert out.finish_reasons == [""]

    @pytest.mark.anyio
    async def test_non_stream_missing_text_defaults_to_empty(self, model):
        response = _make_non_stream_response(
            text=None,
            finish_reason="stop",
            usage=_make_usage(3, 1),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)
        assert out.texts == [""]
        assert out.finish_reasons == ["stop"]

    @pytest.mark.anyio
    async def test_stream_options_can_be_overridden(self, model):
        chunks = [_make_completion_chunk(text="x", finish_reason="stop")]
        mock_create = _patch_create(model, chunks)
        out = await model._agenerate_impl(
            "prompt",
            stream_options={"include_usage": False},
        )
        assert out.texts == ["x"]
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": False}

    @pytest.mark.anyio
    async def test_stream_out_of_bounds_choice_index_skipped(self, model):
        """Streaming chunks whose choice.index >= n are silently skipped."""
        # n=1 but chunk has choice.index=5 (out of bounds)
        valid_chunk = _make_completion_chunk(index=0, text="ok", finish_reason="stop")
        oob_chunk = MagicMock()
        oob_chunk.usage = None
        oob_choice = MagicMock()
        oob_choice.index = 5  # out of bounds for n=1
        oob_choice.text = "SHOULD_NOT_APPEAR"
        oob_choice.finish_reason = ""
        oob_chunk.choices = [oob_choice]

        _patch_create(model, [valid_chunk, oob_chunk])
        out = await model._agenerate_impl("prompt", n=1)
        assert out.texts == ["ok"]
        assert "SHOULD_NOT_APPEAR" not in out.texts[0]

    @pytest.mark.anyio
    async def test_non_stream_out_of_bounds_choice_index_skipped(self, model):
        """Non-streaming responses whose choice.index >= n are silently skipped."""
        resp = MagicMock()
        valid_choice = MagicMock()
        valid_choice.index = 0
        valid_choice.text = "ok"
        valid_choice.finish_reason = "stop"
        valid_choice.logprobs = None

        oob_choice = MagicMock()
        oob_choice.index = 3  # out of bounds for n=1
        oob_choice.text = "SHOULD_NOT_APPEAR"
        oob_choice.finish_reason = "stop"
        oob_choice.logprobs = None

        resp.choices = [valid_choice, oob_choice]
        resp.usage = _make_usage(5, 2)

        mock_create = AsyncMock(return_value=resp)
        target = model._client.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", n=1, stream=False)
        assert out.texts == ["ok"]
        assert "SHOULD_NOT_APPEAR" not in out.texts[0]


# ===================================================================
# _alogprobs_impl
# ===================================================================
class TestGenALogprobs:
    def _make_logprobs_chunk(
        self, token, logprob, index=0, finish_reason="", top_logprobs=None
    ):
        chunk = MagicMock()
        chunk.usage = None

        choice = MagicMock()
        choice.index = index
        choice.text = token
        choice.finish_reason = finish_reason

        lp_obj = MagicMock()
        lp_obj.tokens = [token]
        lp_obj.token_logprobs = [logprob]
        lp_obj.top_logprobs = top_logprobs
        choice.logprobs = lp_obj
        chunk.choices = [choice]
        return chunk

    def _patch_logprobs_create(self, model: GenModel, chunks):
        mock_create = AsyncMock(return_value=_AsyncIterator(chunks))
        target: Any = model._client.completions
        target.create = mock_create  # type: ignore[invalid-assignment]
        return mock_create

    @pytest.mark.anyio
    async def test_n_gt_1_raises(self, model):
        mock = self._patch_logprobs_create(
            model, [self._make_logprobs_chunk("A", -0.1, finish_reason="stop")]
        )
        with pytest.raises(ValueError, match="only supports n=1"):
            await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5, n=2)
        mock.assert_not_called()

    @pytest.mark.anyio
    async def test_logprobs_extracted(self, model):
        chunks = [
            self._make_logprobs_chunk(
                "A",
                -0.1,
                finish_reason="stop",
                top_logprobs=[{"A": -0.1, "B": -0.5}],
            ),
            self._make_logprobs_chunk("B", -0.5),
        ]
        self._patch_logprobs_create(model, chunks)
        out = await model._alogprobs_impl("prompt", max_tokens=2, logprobs=5)
        assert out.logprobs_tokens is not None
        assert "A" in out.logprobs_tokens
        assert out.logprobs is not None
        assert -0.1 in out.logprobs
        assert out.top_logprobs == [{"A": -0.1, "B": -0.5}]

    @pytest.mark.anyio
    async def test_no_logprobs_raises(self, model):
        # Chunk with logprobs=None
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.text = "x"
        choice.finish_reason = "stop"
        choice.logprobs = None
        chunk.choices = [choice]

        self._patch_logprobs_create(model, [chunk])
        with pytest.raises(RuntimeError, match="Streaming logprobs not supported"):
            await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)

    @pytest.mark.anyio
    async def test_logprobs_usage_captured(self, model):
        lp_chunk = self._make_logprobs_chunk("A", -0.1, finish_reason="stop")
        usage_chunk = _make_usage_chunk(4, 1)
        self._patch_logprobs_create(model, [lp_chunk, usage_chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.usage is not None
        assert out.usage["input_tokens"] == 4
        assert out.usage["total_tokens"] == 5

    @pytest.mark.anyio
    async def test_default_params_forwarded(self, model):
        mock = self._patch_logprobs_create(
            model, [self._make_logprobs_chunk("A", -0.1, finish_reason="stop")]
        )
        await model._alogprobs_impl("prompt", max_tokens=1, logprobs=3, echo=True)
        call_kwargs = mock.call_args[1]
        assert call_kwargs.get("logprobs") == 3
        assert call_kwargs.get("echo") is True

    @pytest.mark.anyio
    async def test_non_stream_logprobs_supported(self, model):
        response = _make_non_stream_response(
            text="A",
            finish_reason="stop",
            usage=_make_usage(4, 1),
            with_logprobs=True,
            logprob=-0.2,
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.completions
        target.create = mock_create

        out = await model._alogprobs_impl(
            "prompt",
            max_tokens=1,
            logprobs=5,
            stream=False,
        )
        assert out.logprobs_tokens == ["A"]
        assert out.logprobs == [-0.2]
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stream"] is False

    @pytest.mark.anyio
    async def test_non_stream_missing_logprobs_raises(self, model):
        response = _make_non_stream_response(
            text="A",
            finish_reason="stop",
            usage=_make_usage(4, 1),
            with_logprobs=False,
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.completions
        target.create = mock_create

        with pytest.raises(RuntimeError, match="Streaming logprobs not supported"):
            await model._alogprobs_impl(
                "prompt",
                max_tokens=1,
                logprobs=5,
                stream=False,
            )

    @pytest.mark.anyio
    async def test_stream_out_of_range_choice_index_skipped(self, model):
        """Out-of-range choice.index should be ignored when n=1."""
        oob_chunk = self._make_logprobs_chunk("BAD", -9.0, index=5)
        valid_chunk = self._make_logprobs_chunk(
            "A", -0.1, index=0, finish_reason="stop"
        )
        self._patch_logprobs_create(model, [oob_chunk, valid_chunk])

        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)

        assert out.texts == ["A"]
        assert out.logprobs_tokens == ["A"]
        assert out.logprobs == [-0.1]

    @pytest.mark.anyio
    async def test_non_stream_out_of_range_choice_index_skipped(self, model):
        """Non-stream mode should also ignore out-of-range choice.index."""
        resp = MagicMock()
        resp.choices = [
            _make_non_stream_choice(index=5, text="BAD", token_logprobs=[-9.0]),
            _make_non_stream_choice(
                index=0,
                text="ok",
                tokens=["A"],
                token_logprobs=[-0.1],
            ),
        ]
        resp.usage = _make_usage(2, 1)
        mock_create = AsyncMock(return_value=resp)
        target: Any = model._client.completions
        target.create = mock_create

        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )

        assert out.texts == ["ok"]
        assert "BAD" not in out.texts[0]
        assert out.logprobs_tokens == ["A"]
        assert out.logprobs == [-0.1]


# ===================================================================
# Response metadata capture (response_model, system_fingerprint)
# ===================================================================
class TestResponseMetadata:
    @pytest.mark.anyio
    async def test_agenerate_captures_response_model_streaming(self, model):
        """Streaming: response_model captured from first chunk."""
        chunk = _make_completion_chunk(text="ok", finish_reason="stop")
        chunk.model = "actual-model-v2"
        chunk.system_fingerprint = "fp_abc123"
        _patch_create(model, [chunk])
        out = await model._agenerate_impl("prompt")
        assert out.response_model == "actual-model-v2"
        assert out.system_fingerprint == "fp_abc123"

    @pytest.mark.anyio
    async def test_agenerate_captures_response_model_non_stream(self, model):
        """Non-streaming: response_model captured from response object."""
        response = _make_non_stream_response(
            text="ok", finish_reason="stop", usage=_make_usage(3, 1)
        )
        response.model = "actual-model-v2"
        response.system_fingerprint = "fp_xyz789"
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.completions
        target.create = mock_create
        out = await model._agenerate_impl("prompt", stream=False)
        assert out.response_model == "actual-model-v2"
        assert out.system_fingerprint == "fp_xyz789"

    @pytest.mark.anyio
    async def test_agenerate_streaming_captures_from_first_chunk_only(self, model):
        """Streaming: only the first chunk's metadata is used."""
        chunk1 = _make_completion_chunk(text="a", finish_reason="")
        chunk1.model = "model-v1"
        chunk1.system_fingerprint = "fp_first"
        chunk2 = _make_completion_chunk(text="b", finish_reason="stop")
        chunk2.model = "model-v2"
        chunk2.system_fingerprint = "fp_second"
        _patch_create(model, [chunk1, chunk2])
        out = await model._agenerate_impl("prompt")
        assert out.response_model == "model-v1"
        assert out.system_fingerprint == "fp_first"

    @pytest.mark.anyio
    async def test_agenerate_streaming_none_metadata(self, model):
        """Streaming: missing model/fingerprint attrs → None."""
        chunk = _make_completion_chunk(text="ok", finish_reason="stop")
        # Explicitly delete auto-created MagicMock attrs
        del chunk.model
        del chunk.system_fingerprint
        _patch_create(model, [chunk])
        out = await model._agenerate_impl("prompt")
        assert out.response_model is None
        assert out.system_fingerprint is None

    @pytest.mark.anyio
    async def test_alogprobs_captures_response_model_streaming(self, model):
        """Logprobs streaming: response_model captured from first chunk."""
        chunk = MagicMock()
        chunk.usage = None
        chunk.model = "logprobs-model-v1"
        chunk.system_fingerprint = "fp_lp_stream"
        choice = MagicMock()
        choice.index = 0
        choice.text = "A"
        choice.finish_reason = "stop"
        lp_obj = MagicMock()
        lp_obj.tokens = ["A"]
        lp_obj.token_logprobs = [-0.1]
        choice.logprobs = lp_obj
        chunk.choices = [choice]
        _patch_create(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.response_model == "logprobs-model-v1"
        assert out.system_fingerprint == "fp_lp_stream"

    @pytest.mark.anyio
    async def test_alogprobs_captures_response_model_non_stream(self, model):
        """Logprobs non-streaming: response_model captured from response object."""
        response = _make_non_stream_response(
            text="A",
            finish_reason="stop",
            usage=_make_usage(4, 1),
            with_logprobs=True,
            logprob=-0.2,
        )
        response.model = "logprobs-model-v2"
        response.system_fingerprint = "fp_lp_nonstream"
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.completions
        target.create = mock_create
        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.response_model == "logprobs-model-v2"
        assert out.system_fingerprint == "fp_lp_nonstream"


class TestParamValidation:
    # Parameter validation logic (n type/value, stream type) is identical between
    # ChatModel and GenModel. These tests verify GenModel's implementation.

    @pytest.mark.anyio
    async def test_invalid_n_type_raises(self, model):
        with pytest.raises(TypeError, match="n must be an int"):
            await model._agenerate_impl("prompt", n="2")

    @pytest.mark.anyio
    async def test_bool_n_rejected(self, model):
        """bool is a subclass of int in Python; the guard must catch it."""
        with pytest.raises(TypeError, match="n must be an int"):
            await model._agenerate_impl("prompt", n=True)

    @pytest.mark.anyio
    async def test_invalid_n_value_raises(self, model):
        with pytest.raises(ValueError, match="n must be >= 1"):
            await model._agenerate_impl("prompt", n=0)

    @pytest.mark.anyio
    async def test_stream_non_bool_raises(self, model):
        with pytest.raises(TypeError, match="stream must be a bool"):
            await model._agenerate_impl("prompt", stream="false")

    @pytest.mark.anyio
    async def test_logprobs_bool_n_rejected(self, model):
        with pytest.raises(TypeError, match="n must be an int"):
            await model._alogprobs_impl("prompt", n=True)

    @pytest.mark.anyio
    async def test_logprobs_invalid_n_type_raises(self, model):
        with pytest.raises(TypeError, match="n must be an int"):
            await model._alogprobs_impl("prompt", n="2")

    @pytest.mark.anyio
    async def test_logprobs_invalid_n_value_raises(self, model):
        with pytest.raises(ValueError, match="n must be >= 1"):
            await model._alogprobs_impl("prompt", n=0)

    @pytest.mark.anyio
    async def test_logprobs_stream_non_bool_raises(self, model):
        with pytest.raises(TypeError, match="stream must be a bool"):
            await model._alogprobs_impl("prompt", stream="false")


# ===================================================================
# Branch coverage: null/empty edge cases
# ===================================================================
class TestGenLogprobsNullBranches:
    """Cover logprobs branches where tokens/token_logprobs are empty or usage=None."""

    def _make_lp_chunk(self, *, tokens=None, token_logprobs=None):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.text = "x"
        choice.finish_reason = "stop"
        lp_obj = MagicMock()
        lp_obj.tokens = tokens
        lp_obj.token_logprobs = token_logprobs
        choice.logprobs = lp_obj
        chunk.choices = [choice]
        return chunk

    def _patch(self, model, chunks):
        mock = AsyncMock(return_value=_AsyncIterator(chunks))
        target: Any = model._client.completions
        target.create = mock
        return mock

    @pytest.mark.anyio
    async def test_stream_empty_tokens(self, model):
        """logprobs_obj.tokens is empty → no tokens collected."""
        chunk = self._make_lp_chunk(tokens=[], token_logprobs=[-0.1])
        self._patch(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.logprobs_tokens == []
        assert out.logprobs == [-0.1]

    @pytest.mark.anyio
    async def test_stream_empty_token_logprobs(self, model):
        """logprobs_obj.token_logprobs is empty → no logprobs collected."""
        chunk = self._make_lp_chunk(tokens=["A"], token_logprobs=[])
        self._patch(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.logprobs_tokens == ["A"]
        assert out.logprobs == []

    @pytest.mark.anyio
    async def test_stream_no_usage(self, model):
        """No usage chunk → usage stays None."""
        chunk = self._make_lp_chunk(tokens=["A"], token_logprobs=[-0.1])
        self._patch(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.usage is None

    @pytest.mark.anyio
    async def test_non_stream_empty_tokens(self, model):
        """Non-stream: logprobs_obj.tokens is empty."""
        resp = _make_non_stream_response(
            text="x",
            finish_reason="stop",
            usage=_make_usage(3, 1),
            with_logprobs=True,
        )
        resp.choices[0].logprobs.tokens = []
        resp.choices[0].logprobs.token_logprobs = [-0.1]
        mock = AsyncMock(return_value=resp)
        target: Any = model._client.completions
        target.create = mock

        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.logprobs_tokens == []
        assert out.logprobs == [-0.1]

    @pytest.mark.anyio
    async def test_non_stream_empty_token_logprobs(self, model):
        """Non-stream: logprobs_obj.token_logprobs is empty."""
        resp = _make_non_stream_response(
            text="x",
            finish_reason="stop",
            usage=_make_usage(3, 1),
            with_logprobs=True,
        )
        resp.choices[0].logprobs.tokens = ["A"]
        resp.choices[0].logprobs.token_logprobs = []
        mock = AsyncMock(return_value=resp)
        target: Any = model._client.completions
        target.create = mock

        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.logprobs_tokens == ["A"]
        assert out.logprobs == []

    @pytest.mark.anyio
    async def test_non_stream_no_usage(self, model):
        """Non-stream: resp.usage=None → usage stays None."""
        resp = _make_non_stream_response(
            text="x",
            finish_reason="stop",
            usage=None,
            with_logprobs=True,
        )
        mock = AsyncMock(return_value=resp)
        target: Any = model._client.completions
        target.create = mock

        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.usage is None
