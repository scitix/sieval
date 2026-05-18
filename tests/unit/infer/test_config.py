"""Tests for sieval.infer.config — data models and enums.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from typing import Any

from sieval.infer.config import (
    InferCondition,
    InferConfig,
    InferConfigDict,
    InferHandle,
    InferMetaDict,
    InferPhase,
)


class TestInferPhase:
    """Tests for InferPhase enum."""

    def test_enum_values(self) -> None:
        assert InferPhase.PENDING.value == "pending"
        assert InferPhase.RUNNING.value == "running"
        assert InferPhase.FAILED.value == "failed"
        assert InferPhase.STOPPED.value == "stopped"
        assert InferPhase.STOPPING.value == "stopping"

    def test_enum_member_count(self) -> None:
        assert len(InferPhase) == 5

    def test_enum_from_value(self) -> None:
        assert InferPhase("pending") is InferPhase.PENDING
        assert InferPhase("running") is InferPhase.RUNNING
        assert InferPhase("stopping") is InferPhase.STOPPING


class TestInferCondition:
    """Tests for InferCondition dataclass."""

    def test_ready_true(self) -> None:
        c = InferCondition(status=True)
        assert c.status is True
        assert c.reason == ""

    def test_ready_false_with_reason(self) -> None:
        c = InferCondition(status=False, reason="health_check_failed")
        assert c.status is False
        assert c.reason == "health_check_failed"


class TestInferHandle:
    """Tests for InferHandle dataclass."""

    def test_required_fields(self) -> None:
        handle = InferHandle(
            backend="vllm",
            endpoint="http://localhost:8000",
            handle_id="pid-1234",
        )
        assert handle.backend == "vllm"
        assert handle.endpoint == "http://localhost:8000"
        assert handle.handle_id == "pid-1234"

    def test_metadata_default_factory(self) -> None:
        handle = InferHandle(backend="vllm", endpoint=None, handle_id="pid-1234")
        assert handle.metadata == {}
        assert isinstance(handle.metadata, dict)

    def test_metadata_default_not_shared(self) -> None:
        """Each instance should get its own metadata dict."""
        h1 = InferHandle(backend="vllm", endpoint=None, handle_id="1")
        h2 = InferHandle(backend="vllm", endpoint=None, handle_id="2")
        h1.metadata["key"] = "value"
        assert "key" not in h2.metadata

    def test_endpoint_none(self) -> None:
        handle = InferHandle(backend="slurm", endpoint=None, handle_id="job-5678")
        assert handle.endpoint is None

    def test_custom_metadata(self) -> None:
        meta: dict[str, Any] = {"pid": 1234, "log_file": "/tmp/vllm.log"}
        handle = InferHandle(
            backend="vllm",
            endpoint="http://localhost:8000",
            handle_id="pid-1234",
            metadata=meta,
        )
        assert handle.metadata == meta


class TestInferConfig:
    """Tests for InferConfig dataclass."""

    def test_required_fields(self) -> None:
        cfg = InferConfig(backend="vllm", checkpoint="/models/qwen3-8b")
        assert cfg.backend == "vllm"
        assert cfg.checkpoint == "/models/qwen3-8b"

    def test_params_default_factory(self) -> None:
        cfg = InferConfig(backend="vllm", checkpoint="/models/qwen3-8b")
        assert cfg.params == {}
        assert isinstance(cfg.params, dict)

    def test_metadata_default_factory(self) -> None:
        cfg = InferConfig(backend="vllm", checkpoint="/models/qwen3-8b")
        assert cfg.metadata == {}
        assert isinstance(cfg.metadata, dict)

    def test_defaults_not_shared(self) -> None:
        """Each instance should get its own default dicts."""
        c1 = InferConfig(backend="vllm", checkpoint="/a")
        c2 = InferConfig(backend="vllm", checkpoint="/b")
        c1.params["key"] = "value"
        c1.metadata["key"] = "value"
        assert "key" not in c2.params
        assert "key" not in c2.metadata

    def test_custom_params_and_metadata(self) -> None:
        cfg = InferConfig(
            backend="sglang",
            checkpoint="/models/qwen2.5-72b",
            params={"dtype": "bfloat16", "tp": 2},
            metadata={"framework": "sglang==0.3.0"},
        )
        assert cfg.params["dtype"] == "bfloat16"
        assert cfg.metadata["framework"] == "sglang==0.3.0"


class TestInferConfigDict:
    """Tests for InferConfigDict TypedDict."""

    def test_as_typed_dict(self) -> None:
        d: InferConfigDict = {
            "backend": "vllm",
            "recipe": "qwen3-8b",
            "checkpoint": "/models/qwen3-8b",
            "overrides": {"tp": 2},
        }
        assert d["backend"] == "vllm"
        assert d["recipe"] == "qwen3-8b"
        assert d["checkpoint"] == "/models/qwen3-8b"
        assert d["overrides"] == {"tp": 2}

    def test_partial_dict(self) -> None:
        """total=False means all keys are optional."""
        d: InferConfigDict = {"backend": "vllm"}
        assert d["backend"] == "vllm"
        assert "recipe" not in d

    def test_expected_keys(self) -> None:
        expected = {"backend", "recipe", "checkpoint", "overrides", "env"}
        assert set(InferConfigDict.__annotations__) == expected


class TestInferMetaDict:
    """Tests for InferMetaDict TypedDict."""

    def test_as_typed_dict(self) -> None:
        d: InferMetaDict = {
            "framework": "vllm==0.6.0",
            "dtype": "bfloat16",
            "tp": 4,
            "gpu": "A100-80G x8",
        }
        assert d["framework"] == "vllm==0.6.0"
        assert d["dtype"] == "bfloat16"
        assert d["tp"] == 4
        assert d["gpu"] == "A100-80G x8"

    def test_partial_dict(self) -> None:
        d: InferMetaDict = {"framework": "sglang==0.3.0"}
        assert d["framework"] == "sglang==0.3.0"
        assert "dtype" not in d

    def test_expected_keys(self) -> None:
        expected = {"framework", "dtype", "tp", "gpu", "image"}
        assert set(InferMetaDict.__annotations__) == expected
