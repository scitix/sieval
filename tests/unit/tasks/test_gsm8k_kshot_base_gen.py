"""Unit tests for the GSM8K k-shot base generative task.

AI-Generated Code - GPT-5-Codex (OpenAI)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.gsm8k import GSM8KDataset, GSM8KDatasetSample
from sieval.tasks.gsm8k_kshot_base_gen import (
    GSM8KFewShotBaseGenTask,
    _extract_answer,
    _extract_flexible_match,
)


class _CapturingGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=[" Work shown.\n#### 42"])

    async def _alogprobs_impl(
        self,
        prompt: str,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(answer: str = "Solution.\n#### 42") -> GSM8KDatasetSample:
    return {"question": "What is 40 + 2?", "answer": answer}


def _task() -> tuple[GSM8KFewShotBaseGenTask, _CapturingGenModel]:
    dataset = GSM8KDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(_sample())]),
                "test": HFDataset.from_list([dict(_sample())]),
            }
        )
    )
    model = _CapturingGenModel()
    return GSM8KFewShotBaseGenTask(dataset, model, k=0), model


def test_strict_and_flexible_extractors_are_distinct():
    assert _extract_answer("Therefore the answer is 42.") == ("", "none")
    assert _extract_flexible_match("Therefore the answer is 42.") == (
        "42",
        "flexible-extract",
    )
    assert _extract_answer("Final.\n#### 1,234.") == ("1234", "strict-match")


@pytest.mark.anyio
async def test_infer_does_not_forward_n():
    task, model = _task()

    await task.infer("prompt", TaskContext(sample_id=0, raw_sample=_sample()))

    assert "n" not in model.last_kwargs


@pytest.mark.anyio
async def test_feedback_and_report_include_flexible_secondary_metric():
    task, model = _task()
    raw = _sample()
    inferred = ModelOutput(
        model=model.meta(),
        texts=["No strict delimiter, but the final sentence says 42."],
    )

    post = await task.postprocess(
        inferred,
        TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred),
    )
    finalize, feedback = await task.feedback(
        post,
        TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred),
    )
    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)],
        [],
    )

    assert finalize is True
    assert feedback["correct"] is False
    assert feedback["flexible_correct"] is True
    assert report["score"] == 0.0
    assert report["exact_match"] == 0.0
    assert report["flexible_exact_match"] == 100.0
