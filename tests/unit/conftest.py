"""
Shared fixtures for unit tests.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.models.model import ModelMeta, ModelOutput, ModelUsage
from sieval.core.tasks.context import TaskContext, TaskStageMeta


@pytest.fixture
def sample_model_meta() -> ModelMeta:
    return {
        "model": "test-model",
        "api_base": "http://localhost:8000",
        "default_params": {"temperature": 0.7},
    }


@pytest.fixture
def sample_usage() -> ModelUsage:
    return {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


@pytest.fixture
def sample_model_output(sample_model_meta, sample_usage) -> ModelOutput:
    return ModelOutput(
        model=sample_model_meta,
        texts=["hello world"],
        finish_reasons=["stop"],
        usage=sample_usage,
        request_params={"temperature": 0.7, "max_tokens": 100},
    )


@pytest.fixture
def sample_stage_meta(sample_model_meta, sample_usage) -> TaskStageMeta:
    return {
        "timestamp": 1000.0,
        "timing_s": 1.5,
        "model_calls": [
            {
                "model": sample_model_meta,
                "usage": sample_usage,
                "request_params": {"temperature": 0.7},
                "finish_reasons": ["stop"],
            }
        ],
    }


@pytest.fixture
def base_context() -> TaskContext:
    return TaskContext(
        sample_id=0, raw_sample={"question": "What is 1+1?", "answer": "2"}
    )
