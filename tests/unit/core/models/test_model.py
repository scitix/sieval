"""
Focused tests for non-overlapping Model behaviors.

Most with_args/as_type/meta branches are covered in test_model_derivation.py.
This file keeps only unique checks plus runtime concurrency path tests.

AI-Generated Code - GPT-5.3-Codex (OpenAI)
"""

from unittest.mock import AsyncMock

import pytest

from sieval.core.models import ChatModel, GenModel, ModelOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gen_model():
    return GenModel(model="test-gen", api_key="fake", concurrency_limit=16)


@pytest.fixture
def unlimited_model():
    return GenModel(model="test-unlimited", api_key="fake")


# ===================================================================
# Non-overlapping behaviors
# ===================================================================
class TestModelUnique:
    def test_derived_inherits_client(self, gen_model):
        child = gen_model.with_args(temperature=0.9)
        # Same client object
        assert child._client is gen_model._client

    def test_derived_overrides_kwargs(self, gen_model):
        child = gen_model.with_args(temperature=0.9, top_p=0.8)
        assert child._kwargs == {"temperature": 0.9, "top_p": 0.8}
        # Parent unchanged
        assert gen_model._kwargs == {}

    def test_as_type_preserves_kwargs(self):
        model = GenModel(model="m", api_key="k", temperature=0.5)
        chat = model.as_type(ChatModel)
        assert chat._kwargs == {"temperature": 0.5}


# ===================================================================
# Response metadata
# ===================================================================
class TestModelOutputResponseMetadata:
    def test_model_output_response_metadata_defaults(self):
        """New response metadata fields default to None."""
        output = ModelOutput(
            model={"model": "test", "api_base": None, "default_params": {}},
            texts=["hello"],
        )
        assert output.response_model is None
        assert output.system_fingerprint is None

    def test_model_output_response_metadata_explicit(self):
        """New response metadata fields can be set explicitly."""
        output = ModelOutput(
            model={"model": "test", "api_base": None, "default_params": {}},
            texts=["hello"],
            response_model="gpt-4o-2024-08-06",
            system_fingerprint="fp_abc123",
        )
        assert output.response_model == "gpt-4o-2024-08-06"
        assert output.system_fingerprint == "fp_abc123"


# ===================================================================
# Quota info
# ===================================================================
class TestQuota:
    def test_total_quota_with_limit(self, gen_model):
        assert gen_model.get_total_quota() == 16

    def test_total_quota_unlimited(self, unlimited_model):
        assert unlimited_model.get_total_quota() == float("inf")

    def test_available_quota_equals_total_initially(self, gen_model):
        assert gen_model.get_available_quota() == 16

    def test_available_quota_unlimited(self, unlimited_model):
        assert unlimited_model.get_available_quota() == float("inf")

    def test_quota_info_structure(self, gen_model):
        info = gen_model.get_quota_info()
        assert info["total"] == 16
        assert info["available"] == 16
        assert info["parent"] is None
        assert info["child"]["total"] == 16

    def test_quota_info_derived(self, gen_model):
        child = gen_model.with_args(concurrency_limit=4)
        info = child.get_quota_info()
        assert info["total"] == 4
        assert info["parent"]["total"] == 16
        assert info["child"]["total"] == 4

    def test_quota_info_no_limiter(self, unlimited_model):
        info = unlimited_model.get_quota_info()
        assert info["total"] == float("inf")
        assert info["parent"] is None
        assert info["child"] is None


# ===================================================================
# agenerate / alogprobs concurrency paths
# (covers model.py lines 206, 208-209, 223-255: parent_limiter branches)
# ===================================================================
def _build_chat_model_for_path(path):
    from tests.conftest import MockChatModel

    if path == "parent_and_child":
        base = MockChatModel(concurrency_limit=8)
        return base.with_args(concurrency_limit=4)
    if path == "parent_only":
        base = MockChatModel(concurrency_limit=8)
        child = base.with_args()
        child._limiter = None
        child._parent_limiter = base._limiter
        return child
    if path == "own_only":
        return MockChatModel(concurrency_limit=4)
    if path == "no_limiter":
        return MockChatModel()
    raise ValueError(f"Unknown path: {path}")


def _build_gen_model_for_path(path):
    from tests.conftest import MockGenModel

    if path == "parent_and_child":
        base = MockGenModel(concurrency_limit=8)
        return base.with_args(concurrency_limit=4)
    if path == "parent_only":
        base = MockGenModel(concurrency_limit=8)
        child = base.with_args()
        child._limiter = None
        child._parent_limiter = base._limiter
        return child
    if path == "own_only":
        return MockGenModel(concurrency_limit=4)
    if path == "no_limiter":
        return MockGenModel()
    raise ValueError(f"Unknown path: {path}")


def _assert_path_shape(model, path):
    if path == "parent_and_child":
        assert model._parent_limiter is not None
        assert model._limiter is not None
        return
    if path == "parent_only":
        assert model._parent_limiter is not None
        assert model._limiter is None
        return
    if path == "own_only":
        assert model._parent_limiter is None
        assert model._limiter is not None
        return
    if path == "no_limiter":
        assert model._parent_limiter is None
        assert model._limiter is None
        return
    raise ValueError(f"Unknown path: {path}")


class TestConcurrencyPaths:
    """Exercise all four limiter combinations for agenerate and alogprobs."""

    @pytest.mark.anyio
    # (limiter_path, prompt)
    @pytest.mark.parametrize(
        "path,prompt",
        [
            ("parent_and_child", "hello"),
            ("parent_only", "hello"),
            ("own_only", "hello"),
            ("no_limiter", "hello"),
        ],
    )
    async def test_agenerate_paths(self, path, prompt):
        model = _build_chat_model_for_path(path)
        _assert_path_shape(model, path)
        result = await model.agenerate(prompt)
        assert result.texts == ["unknown"]

    @pytest.mark.anyio
    # (limiter_path, prompt)
    @pytest.mark.parametrize(
        "path,prompt",
        [
            ("parent_and_child", "A"),
            ("parent_only", "B"),
            ("own_only", "C"),
            ("no_limiter", "D"),
        ],
    )
    async def test_alogprobs_paths(self, path, prompt):
        model = _build_gen_model_for_path(path)
        _assert_path_shape(model, path)
        result = await model.alogprobs(prompt)
        assert result.texts == [""]
        assert result.logprobs is not None and len(result.logprobs) == 1
        assert result.logprobs_tokens is not None and len(result.logprobs_tokens) == 1

    @pytest.mark.anyio
    async def test_alogprobs_forwards_echo_flag(self, monkeypatch):
        from tests.conftest import MockGenModel

        model = MockGenModel()
        alogprobs_impl = AsyncMock(side_effect=model._alogprobs_impl)
        monkeypatch.setattr(model, "_alogprobs_impl", alogprobs_impl)

        await model.alogprobs("A", echo=False, max_tokens=2, logprobs=3)

        assert alogprobs_impl.await_count == 1
        call = alogprobs_impl.await_args
        assert call is not None
        assert call.kwargs["echo"] is False
        assert call.kwargs["max_tokens"] == 2
        assert call.kwargs["logprobs"] == 3
