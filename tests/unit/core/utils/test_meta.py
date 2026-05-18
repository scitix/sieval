"""
Tests for sieval.core.utils.meta — build_model_call_meta, build_stage_meta.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import time

from sieval.core.models.model import ModelOutput
from sieval.core.utils.meta import build_model_call_meta, build_stage_meta


class TestBuildModelCallMeta:
    def test_full_output(self, sample_model_output):
        meta = build_model_call_meta(sample_model_output)
        assert meta["model"] == sample_model_output.model
        assert meta["usage"]["input_tokens"] == 100
        assert meta["request_params"]["temperature"] == 0.7
        assert meta["finish_reasons"] == ["stop"]

    def test_minimal_output(self, sample_model_meta):
        output = ModelOutput(model=sample_model_meta, texts=["hi"])
        meta = build_model_call_meta(output)
        assert meta["model"] == sample_model_meta
        assert "usage" not in meta
        assert "request_params" not in meta
        assert "finish_reasons" not in meta

    def test_with_usage_only(self, sample_model_meta, sample_usage):
        output = ModelOutput(model=sample_model_meta, texts=["hi"], usage=sample_usage)
        meta = build_model_call_meta(output)
        assert meta["usage"]["total_tokens"] == 150
        assert "request_params" not in meta

    def test_request_params_copied(self, sample_model_meta):
        params = {"temperature": 0.5}
        output = ModelOutput(
            model=sample_model_meta, texts=["hi"], request_params=params
        )
        meta = build_model_call_meta(output)
        # Should be a copy, not the same dict
        assert meta["request_params"] == params
        assert meta["request_params"] is not params

    def test_with_response_metadata(self):
        """build_model_call_meta includes response_model and system_fingerprint."""
        output = ModelOutput(
            model={"model": "qwen", "api_base": None, "default_params": {}},
            texts=["hi"],
            response_model="Qwen/Qwen3-4B",
            system_fingerprint="fp_xyz",
        )
        meta = build_model_call_meta(output)
        assert meta["response_model"] == "Qwen/Qwen3-4B"
        assert meta["system_fingerprint"] == "fp_xyz"

    def test_without_response_metadata(self):
        """build_model_call_meta omits None response metadata."""
        output = ModelOutput(
            model={"model": "qwen", "api_base": None, "default_params": {}},
            texts=["hi"],
        )
        meta = build_model_call_meta(output)
        assert "response_model" not in meta
        assert "system_fingerprint" not in meta


class TestBuildStageMeta:
    def test_no_outputs(self):
        meta = build_stage_meta()
        assert "timestamp" in meta
        assert "model_calls" not in meta
        assert "timing_s" not in meta

    def test_with_timing(self):
        meta = build_stage_meta(timing_s=1.5)
        assert meta["timing_s"] == 1.5

    def test_single_output(self, sample_model_output):
        meta = build_stage_meta(sample_model_output)
        assert len(meta["model_calls"]) == 1
        assert meta["model_calls"][0]["model"] == sample_model_output.model

    def test_multiple_outputs(self, sample_model_meta, sample_usage):
        out1 = ModelOutput(model=sample_model_meta, texts=["a"], usage=sample_usage)
        out2 = ModelOutput(model=sample_model_meta, texts=["b"])
        meta = build_stage_meta(out1, out2)
        assert len(meta["model_calls"]) == 2
        assert "usage" in meta["model_calls"][0]
        assert "usage" not in meta["model_calls"][1]

    def test_with_extra(self, sample_model_output):
        meta = build_stage_meta(sample_model_output, extra={"custom_key": "value"})
        assert meta["extra"] == {"custom_key": "value"}

    def test_empty_extra_not_included(self):
        meta = build_stage_meta(extra={})
        assert "extra" not in meta

    def test_timestamp_is_recent(self):
        before = time.time()
        meta = build_stage_meta()
        after = time.time()
        assert "timestamp" in meta, "timestamp key missing from meta"
        assert isinstance(meta["timestamp"], (int, float)), (
            f"timestamp must be numeric, got {type(meta['timestamp'])}"
        )
        assert before <= meta["timestamp"] <= after

    def test_full_combo(self, sample_model_output):
        meta = build_stage_meta(
            sample_model_output, timing_s=2.0, extra={"note": "test"}
        )
        assert meta["timing_s"] == 2.0
        assert len(meta["model_calls"]) == 1
        assert meta["extra"]["note"] == "test"
        assert "timestamp" in meta
