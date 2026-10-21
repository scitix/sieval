"""
Shared fixtures for unit tests.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import os

import pytest
from loguru import logger

from sieval.core.models import ModelMeta, ModelOutput, ModelUsage
from sieval.core.runners import resume_option
from sieval.core.tasks.context import TaskContext, TaskStageMeta
from sieval.core.utils.logging import configure_logging

# scripts/ tests are repo-hygiene checks (preflight, layer imports) that run
# tooling against the LIVE repo tree. Inside mutmut's mutants/ sandbox that
# tree is a partial copy, so they fail spuriously — and they exercise no
# sieval/core mutants. Skip collecting them under mutation testing only
# (mutmut sets MUTANT_UNDER_TEST for every pytest run it drives).
collect_ignore = ["scripts"] if os.environ.get("MUTANT_UNDER_TEST") else []


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


@pytest.fixture
def warnings():
    """Warning messages logged during the test; resume deprecations warn anew.

    ``configure_logging()`` first: its first call in a process removes every
    sink, so a CLI under test would otherwise drop this one.
    """
    configure_logging()
    resume_option._warned.clear()
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink)
    resume_option._warned.clear()
