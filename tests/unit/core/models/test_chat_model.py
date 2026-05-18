"""
Unit tests for sieval/core/models/chat_model.py.

Covers: _agenerate_impl (string prompt, message list prompt, streaming
accumulation, n>1 choices, usage, reasoning_content), _alogprobs_impl
(streaming logprobs, no-logprobs raises, usage).

All OpenAI client calls are mocked — no real API traffic.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sieval.core.models.chat_model import ChatModel
from sieval.core.models.model import ModelOutput


# ---------------------------------------------------------------------------
# Async streaming helpers
# ---------------------------------------------------------------------------
def _make_chunk(
    index: int = 0,
    content: str = "",
    finish_reason: str = "",
    usage=None,
    reasoning: str = "",
):
    """Build a minimal streaming chunk object."""
    chunk = MagicMock()
    chunk.usage = usage

    if content or finish_reason is not None or reasoning:
        choice = MagicMock()
        choice.index = index
        choice.finish_reason = finish_reason
        delta = MagicMock()
        delta.content = content or None
        # reasoning_content attribute
        reasoning_attr = reasoning or None
        delta.reasoning = None
        delta.reasoning_content = reasoning_attr
        choice.delta = delta
        chunk.choices = [choice]
    else:
        chunk.choices = []

    return chunk


def _make_usage_chunk(prompt_tokens=10, completion_tokens=5):
    """Chunk that carries usage but no choices."""
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    chunk.usage.total_tokens = prompt_tokens + completion_tokens
    return chunk


def _make_non_stream_response(
    *,
    text: str = "",
    finish_reason: str | None = "stop",
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    usage=None,
):
    resp = MagicMock()
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = finish_reason
    message = MagicMock()
    message.content = text
    message.reasoning = reasoning
    message.reasoning_content = reasoning_content
    choice.message = message
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_non_stream_choice(
    *,
    index: int = 0,
    text: str | list[object] = "",
    finish_reason: str | None = "stop",
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    logprob_items: list[tuple[str, float]] | None = None,
):
    choice = MagicMock()
    choice.index = index
    choice.finish_reason = finish_reason
    message = MagicMock()
    message.content = text
    message.reasoning = reasoning
    message.reasoning_content = reasoning_content
    choice.message = message
    if logprob_items is not None:
        logprobs_obj = MagicMock()
        logprobs_obj.content = [
            SimpleNamespace(token=token, logprob=logprob)
            for token, logprob in logprob_items
        ]
        choice.logprobs = logprobs_obj
    else:
        choice.logprobs = None
    return choice


def _make_non_stream_response_from_choices(choices: list[object], usage=None):
    resp = MagicMock()
    resp.choices = choices
    resp.usage = usage
    return resp


def _make_usage(prompt_tokens=10, completion_tokens=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    return usage


class _AsyncIterator:
    """Wraps a list into an async iterator."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as e:
            raise StopAsyncIteration from e


# ---------------------------------------------------------------------------
# Concrete ChatModel (delegates to parent _agenerate_impl)
# ---------------------------------------------------------------------------
class _TestChatModel(ChatModel):
    """ChatModel that calls the real parent _agenerate_impl / _alogprobs_impl."""

    # No override — let parent do the work so we test actual implementation.
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def model():
    return _TestChatModel(model="test-chat", api_key="fake")


def _patch_create(model: _TestChatModel, chunks):
    """Patch model._client.chat.completions.create to return async iterator."""
    mock_create = AsyncMock(return_value=_AsyncIterator(chunks))
    target: Any = model._client.chat.completions
    target.create = mock_create  # type: ignore[invalid-assignment]
    return mock_create


# ===================================================================
# _agenerate_impl — string prompt
# ===================================================================
class TestAGenerateString:
    @pytest.mark.anyio
    async def test_basic_string_prompt(self, model):
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world", finish_reason="stop"),
            _make_usage_chunk(10, 3),
        ]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("Hi")
        assert isinstance(out, ModelOutput)
        assert out.texts == ["Hello world"]
        assert out.finish_reasons == ["stop"]

    @pytest.mark.anyio
    async def test_usage_captured(self, model):
        chunks = [
            _make_chunk(content="ok", finish_reason="stop"),
            _make_usage_chunk(8, 2),
        ]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt")
        assert out.usage is not None
        assert out.usage["input_tokens"] == 8
        assert out.usage["output_tokens"] == 2
        assert out.usage["total_tokens"] == 10

    @pytest.mark.anyio
    async def test_no_usage_chunk(self, model):
        chunks = [_make_chunk(content="hi", finish_reason="stop")]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt")
        assert out.usage is None

    @pytest.mark.anyio
    async def test_model_meta_attached(self, model):
        chunks = [_make_chunk(content="x", finish_reason="stop")]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt")
        assert out.model["model"] == "test-chat"

    @pytest.mark.anyio
    async def test_request_params_captured(self, model):
        chunks = [_make_chunk(content="x", finish_reason="stop")]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt", temperature=0.7)
        assert out.request_params is not None
        assert out.request_params.get("temperature") == 0.7

    @pytest.mark.anyio
    async def test_non_stream_mode_supported(self, model):
        response = _make_non_stream_response(
            text="done",
            finish_reason="stop",
            usage=_make_usage(9, 4),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)

        assert out.texts == ["done"]
        assert out.finish_reasons == ["stop"]
        assert out.usage is not None
        assert out.usage["input_tokens"] == 9
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stream"] is False

    @pytest.mark.anyio
    async def test_non_stream_missing_finish_reason_defaults_to_empty(self, model):
        response = _make_non_stream_response(
            text="done",
            finish_reason=None,
            usage=_make_usage(2, 1),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)
        assert out.texts == ["done"]
        assert out.finish_reasons == [""]

    @pytest.mark.anyio
    async def test_stream_options_can_be_overridden(self, model):
        chunks = [_make_chunk(content="x", finish_reason="stop")]
        mock_create = _patch_create(model, chunks)
        out = await model._agenerate_impl(
            "prompt",
            stream_options={"include_usage": False},
        )
        assert out.texts == ["x"]
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": False}

    @pytest.mark.anyio
    async def test_non_stream_invalid_content_type_raises(self, model):
        response = _make_non_stream_response_from_choices(
            choices=[
                _make_non_stream_choice(
                    text=[
                        "A",
                        SimpleNamespace(text="B"),
                        SimpleNamespace(text=123),
                    ]
                )
            ],
            usage=_make_usage(3, 2),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        with pytest.raises(TypeError):
            await model._agenerate_impl("prompt", stream=False)

    @pytest.mark.anyio
    async def test_non_stream_ignores_out_of_range_choice(self, model):
        response = _make_non_stream_response_from_choices(
            choices=[_make_non_stream_choice(index=5, text="ignored")],
            usage=_make_usage(7, 1),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", n=1, stream=False)
        assert out.texts == [""]
        assert out.finish_reasons == [""]
        assert out.usage is not None
        assert out.usage["input_tokens"] == 7

    @pytest.mark.anyio
    async def test_stream_ignores_out_of_range_choice(self, model):
        chunk = _make_chunk(index=5, content="ignored", finish_reason="stop")
        _patch_create(model, [chunk, _make_usage_chunk(4, 2)])
        out = await model._agenerate_impl("prompt", n=1)
        assert out.texts == [""]
        assert out.finish_reasons == [""]
        assert out.usage is not None
        assert out.usage["total_tokens"] == 6


# ===================================================================
# _agenerate_impl — message list prompt
# ===================================================================
class TestAGenerateMessageList:
    @pytest.mark.anyio
    async def test_message_list_prompt(self, model):
        messages = [{"role": "user", "content": "hello"}]
        chunks = [_make_chunk(content="reply", finish_reason="stop")]
        mock = _patch_create(model, chunks)
        out = await model._agenerate_impl(messages)
        assert out.texts == ["reply"]
        # Verify messages forwarded
        call_kwargs = mock.call_args[1]
        assert call_kwargs["messages"] == messages


# ===================================================================
# _agenerate_impl — n > 1 choices
# ===================================================================
class TestAGenerateMultipleChoices:
    @pytest.mark.anyio
    async def test_two_choices_in_single_chunk(self, model):
        def _multi_choice_chunk(pairs, finish_reason=""):
            chunk = MagicMock()
            chunk.usage = None
            chunk.choices = []
            for index, content in pairs:
                choice = MagicMock()
                choice.index = index
                choice.finish_reason = finish_reason
                delta = MagicMock()
                delta.content = content
                delta.reasoning = None
                delta.reasoning_content = None
                choice.delta = delta
                chunk.choices.append(choice)
            return chunk

        chunks = [
            _multi_choice_chunk([(0, "A1"), (1, "B1")]),
            _multi_choice_chunk([(0, "A2"), (1, "B2")], finish_reason="stop"),
        ]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt", n=2)
        assert out.texts[0] == "A1A2"
        assert out.texts[1] == "B1B2"


# ===================================================================
# _agenerate_impl — reasoning_content
# ===================================================================
class TestAGenerateReasoning:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reasoning, reasoning_content, expected",
        [
            (None, "think", "think"),  # reasoning_content fallback
            ("primary", "secondary", "primary"),  # reasoning takes priority
        ],
        ids=["reasoning_content-fallback", "reasoning-priority"],
    )
    async def test_stream_reasoning_extraction(
        self, model, reasoning, reasoning_content, expected
    ):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "answer"
        delta.reasoning = reasoning
        delta.reasoning_content = reasoning_content
        choice.delta = delta
        chunk.choices = [choice]

        _patch_create(model, [chunk])
        out = await model._agenerate_impl("prompt")
        assert out.reasoning_texts is not None
        assert out.reasoning_texts[0] == expected

    @pytest.mark.anyio
    async def test_reasoning_accumulated_multi_chunk(self, model):
        """Reasoning tokens accumulate across multiple streaming chunks."""
        chunks = [
            _make_chunk(content="a", reasoning="think1"),
            _make_chunk(content="b", reasoning="think2", finish_reason="stop"),
        ]
        _patch_create(model, chunks)
        out = await model._agenerate_impl("prompt")
        assert out.texts == ["ab"]
        assert out.reasoning_texts is not None
        assert out.reasoning_texts[0] == "think1think2"

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reasoning, reasoning_content, expected",
        [
            ("step by step", None, "step by step"),  # reasoning field
            (None, "fallback", "fallback"),  # reasoning_content fallback
            ("primary", "secondary", "primary"),  # priority
        ],
        ids=["reasoning-field", "reasoning_content-fallback", "priority"],
    )
    async def test_non_stream_reasoning_extraction(
        self, model, reasoning, reasoning_content, expected
    ):
        """Non-streaming: reasoning extraction of chat_model.py."""
        response = _make_non_stream_response(
            text="answer",
            finish_reason="stop",
            reasoning=reasoning,
            reasoning_content=reasoning_content,
            usage=_make_usage(10, 5),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)
        assert out.texts == ["answer"]
        assert out.reasoning_texts is not None
        assert out.reasoning_texts[0] == expected


# ===================================================================
# _alogprobs_impl — reasoning in logprobs mode
# ===================================================================
class TestALogprobsReasoning:
    """Cover reasoning extraction paths in _alogprobs_impl."""

    def _make_stream_logprobs_chunk(self, reasoning, reasoning_content):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "answer"
        delta.reasoning = reasoning
        delta.reasoning_content = reasoning_content
        choice.delta = delta
        lp_item = MagicMock()
        lp_item.token = "A"
        lp_item.logprob = -0.1
        logprobs_obj = MagicMock()
        logprobs_obj.content = [lp_item]
        choice.logprobs = logprobs_obj
        chunk.choices = [choice]
        return chunk

    def _patch_logprobs_create(self, model: _TestChatModel, chunks):
        mock_create = AsyncMock(return_value=_AsyncIterator(chunks))
        target: Any = model._client.chat.completions
        target.create = mock_create  # type: ignore[invalid-assignment]
        return mock_create

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reasoning, reasoning_content, expected",
        [
            ("think", None, "think"),  # reasoning field
            (None, "fallback", "fallback"),  # reasoning_content fallback
        ],
        ids=["reasoning-field", "reasoning_content-fallback"],
    )
    async def test_stream_logprobs_reasoning_extraction(
        self, model, reasoning, reasoning_content, expected
    ):
        chunk = self._make_stream_logprobs_chunk(reasoning, reasoning_content)
        self._patch_logprobs_create(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.reasoning_texts is not None
        assert out.reasoning_texts[0] == expected

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reasoning, reasoning_content, expected",
        [
            ("thinking", None, "thinking"),  # reasoning field
            (None, "secondary", "secondary"),  # reasoning_content fallback
            ("primary", "secondary", "primary"),  # priority
        ],
        ids=["reasoning-field", "reasoning_content-fallback", "priority"],
    )
    async def test_non_stream_logprobs_reasoning_extraction(
        self, model, reasoning, reasoning_content, expected
    ):
        response = _make_non_stream_response_from_choices(
            choices=[
                _make_non_stream_choice(
                    text="answer",
                    reasoning=reasoning,
                    reasoning_content=reasoning_content,
                    logprob_items=[("A", -0.1)],
                )
            ],
            usage=_make_usage(6, 2),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.reasoning_texts is not None
        assert out.reasoning_texts[0] == expected


# ===================================================================
# _alogprobs_impl — logprobs
# ===================================================================
class TestALogprobs:
    def _make_logprobs_chunk(self, token, logprob, index=0, finish_reason=""):
        chunk = MagicMock()
        chunk.usage = None

        choice = MagicMock()
        choice.index = index
        choice.finish_reason = finish_reason
        delta = MagicMock()
        delta.content = token
        delta.reasoning = None
        delta.reasoning_content = None
        choice.delta = delta

        lp_item = MagicMock()
        lp_item.token = token
        lp_item.logprob = logprob
        logprobs_obj = MagicMock()
        logprobs_obj.content = [lp_item]
        choice.logprobs = logprobs_obj
        chunk.choices = [choice]
        return chunk

    def _patch_logprobs_create(self, model: _TestChatModel, chunks):
        mock_create = AsyncMock(return_value=_AsyncIterator(chunks))
        target: Any = model._client.chat.completions
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
            self._make_logprobs_chunk("A", -0.1, finish_reason="stop"),
            self._make_logprobs_chunk("B", -0.5),
        ]
        self._patch_logprobs_create(model, chunks)
        out = await model._alogprobs_impl("prompt", max_tokens=2, logprobs=5)
        assert out.logprobs_tokens is not None
        assert "A" in out.logprobs_tokens
        assert out.logprobs is not None
        assert -0.1 in out.logprobs

    @pytest.mark.anyio
    async def test_no_logprobs_raises(self, model):
        # Chunk with no logprobs attribute
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "x"
        delta.reasoning = None
        delta.reasoning_content = None
        choice.delta = delta
        choice.logprobs = None
        chunk.choices = [choice]

        self._patch_logprobs_create(model, [chunk])
        with pytest.raises(RuntimeError, match="Streaming logprobs not supported"):
            await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)

    @pytest.mark.anyio
    async def test_logprobs_usage_captured(self, model):
        lp_chunk = self._make_logprobs_chunk("A", -0.1, finish_reason="stop")
        usage_chunk = _make_usage_chunk(5, 1)
        self._patch_logprobs_create(model, [lp_chunk, usage_chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.usage is not None
        assert out.usage["input_tokens"] == 5

    @pytest.mark.anyio
    async def test_logprobs_reasoning_content_accumulated(self, model):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "answer"
        delta.reasoning = None
        delta.reasoning_content = "think"
        choice.delta = delta
        lp_item = MagicMock()
        lp_item.token = "A"
        lp_item.logprob = -0.1
        logprobs_obj = MagicMock()
        logprobs_obj.content = [lp_item]
        choice.logprobs = logprobs_obj
        chunk.choices = [choice]
        self._patch_logprobs_create(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.reasoning_texts is not None
        assert out.reasoning_texts[0] == "think"

    @pytest.mark.anyio
    async def test_logprobs_non_stream_supported(self, model):
        response = _make_non_stream_response_from_choices(
            choices=[
                _make_non_stream_choice(
                    text="answer",
                    logprob_items=[("A", -0.1), ("B", -0.2)],
                )
            ],
            usage=_make_usage(6, 2),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._alogprobs_impl(
            "prompt",
            max_tokens=1,
            logprobs=5,
            stream=False,
        )
        assert out.texts == ["answer"]
        assert out.logprobs_tokens == ["A", "B"]
        assert out.logprobs == [-0.1, -0.2]
        assert out.usage is not None
        assert out.usage["input_tokens"] == 6

    @pytest.mark.anyio
    async def test_logprobs_non_stream_missing_logprobs_raises(self, model):
        response = _make_non_stream_response_from_choices(
            choices=[_make_non_stream_choice(text="answer", logprob_items=None)],
            usage=_make_usage(1, 1),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        with pytest.raises(RuntimeError, match="Streaming logprobs not supported"):
            await model._alogprobs_impl(
                "prompt",
                max_tokens=1,
                logprobs=5,
                stream=False,
            )

    @pytest.mark.anyio
    async def test_logprobs_stream_out_of_range_choice_index_skipped(self, model):
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
    async def test_logprobs_non_stream_out_of_range_choice_index_skipped(self, model):
        """Non-stream mode should also ignore out-of-range choice.index."""
        response = _make_non_stream_response_from_choices(
            choices=[
                _make_non_stream_choice(
                    index=5,
                    text="BAD",
                    logprob_items=[("BAD", -9.0)],
                ),
                _make_non_stream_choice(
                    index=0,
                    text="ok",
                    logprob_items=[("A", -0.1)],
                ),
            ],
            usage=_make_usage(2, 1),
        )
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
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
        chunk = _make_chunk(content="ok", finish_reason="stop")
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
        target: Any = model._client.chat.completions
        target.create = mock_create
        out = await model._agenerate_impl("prompt", stream=False)
        assert out.response_model == "actual-model-v2"
        assert out.system_fingerprint == "fp_xyz789"

    @pytest.mark.anyio
    async def test_agenerate_streaming_captures_from_first_chunk_only(self, model):
        """Streaming: only the first chunk's metadata is used."""
        chunk1 = _make_chunk(content="a", finish_reason="")
        chunk1.model = "model-v1"
        chunk1.system_fingerprint = "fp_first"
        chunk2 = _make_chunk(content="b", finish_reason="stop")
        chunk2.model = "model-v2"
        chunk2.system_fingerprint = "fp_second"
        _patch_create(model, [chunk1, chunk2])
        out = await model._agenerate_impl("prompt")
        assert out.response_model == "model-v1"
        assert out.system_fingerprint == "fp_first"

    @pytest.mark.anyio
    async def test_agenerate_streaming_none_metadata(self, model):
        """Streaming: missing model/fingerprint attrs → None."""
        chunk = _make_chunk(content="ok", finish_reason="stop")
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
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "A"
        delta.reasoning = None
        delta.reasoning_content = None
        choice.delta = delta
        lp_item = MagicMock()
        lp_item.token = "A"
        lp_item.logprob = -0.1
        logprobs_obj = MagicMock()
        logprobs_obj.content = [lp_item]
        choice.logprobs = logprobs_obj
        chunk.choices = [choice]
        _patch_create(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.response_model == "logprobs-model-v1"
        assert out.system_fingerprint == "fp_lp_stream"

    @pytest.mark.anyio
    async def test_alogprobs_captures_response_model_non_stream(self, model):
        """Logprobs non-streaming: response_model captured from response object."""
        response = _make_non_stream_response_from_choices(
            choices=[
                _make_non_stream_choice(
                    text="A",
                    logprob_items=[("A", -0.2)],
                )
            ],
            usage=_make_usage(4, 1),
        )
        response.model = "logprobs-model-v2"
        response.system_fingerprint = "fp_lp_nonstream"
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create
        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.response_model == "logprobs-model-v2"
        assert out.system_fingerprint == "fp_lp_nonstream"


# ===================================================================
# _agenerate_impl — invalid prompt type
# ===================================================================
class TestAGenerateInvalidPrompt:
    @pytest.mark.anyio
    async def test_non_string_non_iterable_raises(self, model):
        with pytest.raises(TypeError, match="string or iterable"):
            await model._agenerate_impl(12345)


class TestParamValidation:
    @pytest.mark.anyio
    async def test_invalid_n_type_raises(self, model):
        with pytest.raises(TypeError, match="n must be an int"):
            await model._agenerate_impl("prompt", n="1")

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
            await model._alogprobs_impl("prompt", n="1")

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
class TestAGenerateNullBranches:
    """Cover streaming/non-stream branches where delta, content, or usage is None."""

    @pytest.mark.anyio
    async def test_stream_delta_none(self, model):
        """choice.delta is None → no content/reasoning accumulated."""
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        choice.delta = None
        chunk.choices = [choice]

        _patch_create(model, [chunk])
        out = await model._agenerate_impl("prompt")
        assert out.texts == [""]

    @pytest.mark.anyio
    async def test_stream_delta_content_none(self, model):
        """choice.delta.content is None → text stays empty."""
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = None
        delta.reasoning = None
        delta.reasoning_content = None
        choice.delta = delta
        chunk.choices = [choice]

        _patch_create(model, [chunk])
        out = await model._agenerate_impl("prompt")
        assert out.texts == [""]

    @pytest.mark.anyio
    async def test_non_stream_content_none(self, model):
        """Non-stream message.content is None → text stays empty."""
        response = _make_non_stream_response(text="")
        response.choices[0].message.content = None
        response.usage = _make_usage(3, 1)
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)
        assert out.texts == [""]

    @pytest.mark.anyio
    async def test_non_stream_no_usage(self, model):
        """Non-stream resp.usage is None → usage stays None."""
        response = _make_non_stream_response(text="ok", usage=None)
        mock_create = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock_create

        out = await model._agenerate_impl("prompt", stream=False)
        assert out.usage is None


class TestALogprobsNullBranches:
    """Cover logprobs-mode branches where delta, content, or usage is None."""

    def _make_lp_chunk(self, *, delta=True, content="A", logprob=-0.1):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        if delta:
            d = MagicMock()
            d.content = content
            d.reasoning = None
            d.reasoning_content = None
            choice.delta = d
        else:
            choice.delta = None
        lp_item = MagicMock()
        lp_item.token = "A"
        lp_item.logprob = logprob
        logprobs_obj = MagicMock()
        logprobs_obj.content = [lp_item]
        choice.logprobs = logprobs_obj
        chunk.choices = [choice]
        return chunk

    def _patch(self, model, chunks):
        mock = AsyncMock(return_value=_AsyncIterator(chunks))
        target: Any = model._client.chat.completions
        target.create = mock
        return mock

    @pytest.mark.anyio
    async def test_stream_delta_none(self, model):
        """Logprobs stream: delta=None → no text, but logprobs still collected."""
        chunk = self._make_lp_chunk(delta=False)
        self._patch(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.texts == [""]
        assert out.logprobs_tokens == ["A"]

    @pytest.mark.anyio
    async def test_stream_delta_content_none(self, model):
        """Logprobs stream: delta.content=None → text empty."""
        chunk = self._make_lp_chunk(content=None)
        self._patch(model, [chunk])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.texts == [""]
        assert out.logprobs_tokens == ["A"]

    @pytest.mark.anyio
    async def test_stream_empty_logprobs_content(self, model):
        """Logprobs stream: logprobs_obj.content is empty list → no tokens."""
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "x"
        delta.reasoning = None
        delta.reasoning_content = None
        choice.delta = delta
        logprobs_obj = MagicMock()
        logprobs_obj.content = []
        choice.logprobs = logprobs_obj
        chunk.choices = [choice]

        # Need a second chunk with actual logprobs so saw_logprobs=True
        chunk2 = self._make_lp_chunk()
        self._patch(model, [chunk, chunk2])
        out = await model._alogprobs_impl("prompt", max_tokens=1, logprobs=5)
        assert out.logprobs_tokens == ["A"]

    @pytest.mark.anyio
    async def test_non_stream_content_none(self, model):
        """Logprobs non-stream: message.content=None → text empty."""
        choice = _make_non_stream_choice(text="", logprob_items=[("A", -0.1)])
        choice.message.content = None
        response = _make_non_stream_response_from_choices(
            choices=[choice], usage=_make_usage(3, 1)
        )
        mock = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock

        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.texts == [""]
        assert out.logprobs_tokens == ["A"]

    @pytest.mark.anyio
    async def test_non_stream_empty_logprobs_content(self, model):
        """Logprobs non-stream: logprobs_obj.content is empty → no tokens."""
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        message = MagicMock()
        message.content = "x"
        message.reasoning = None
        message.reasoning_content = None
        choice.message = message
        logprobs_obj = MagicMock()
        logprobs_obj.content = []
        choice.logprobs = logprobs_obj

        response = _make_non_stream_response_from_choices(
            choices=[choice], usage=_make_usage(3, 1)
        )
        mock = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock

        # saw_logprobs=True (logprobs_obj is not None) but content empty
        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.logprobs_tokens == []

    @pytest.mark.anyio
    async def test_non_stream_no_usage(self, model):
        """Logprobs non-stream: resp.usage=None → usage stays None."""
        response = _make_non_stream_response_from_choices(
            choices=[_make_non_stream_choice(text="ok", logprob_items=[("A", -0.1)])],
            usage=None,
        )
        mock = AsyncMock(return_value=response)
        target: Any = model._client.chat.completions
        target.create = mock

        out = await model._alogprobs_impl(
            "prompt", max_tokens=1, logprobs=5, stream=False
        )
        assert out.usage is None
