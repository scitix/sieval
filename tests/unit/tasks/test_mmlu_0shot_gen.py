"""Unit tests for the MMLU 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.mmlu import MMLUDataset, MMLUDatasetSample
from sieval.tasks.mmlu_0shot_gen import MMLUZeroShotGenTask


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=["Answer: A"])

    async def _alogprobs_impl(
        self,
        prompt,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(subject: str = "anatomy", answer: int = 0) -> MMLUDatasetSample:
    return {
        "question": "Q?",
        "subject": subject,
        "choices": ["a", "b", "c", "d"],
        "answer": answer,
    }


def _task() -> MMLUZeroShotGenTask:
    dataset = MMLUDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    return MMLUZeroShotGenTask(dataset, _StubChatModel())


def _fb(correct: bool, subject: str = "anatomy", category: str = "other"):
    return {"correct": correct, "subject": subject, "category": category, "answer": "A"}


@pytest.mark.anyio
async def test_feedback_derives_letter_from_answer_index():
    task = _task()
    ctx = TaskContext(sample_id=0, raw_sample=_sample(subject="astronomy", answer=2))
    finalize, fb = await task.feedback("C", ctx)
    assert finalize is True
    assert fb["answer"] == "C" and fb["correct"] is True
    assert fb["subject"] == "astronomy"


@pytest.mark.anyio
async def test_report_excludes_fails_from_denominator():
    # 1 correct final + 1 pipeline failure → score 100.0 over the finalized set;
    # the fail is reported separately (matches main's mmlu_0shot_gen). A
    # len(finals)+len(fails) denominator would give 50.0.
    task = _task()
    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=_sample(), feedback_result=_fb(True))],
        [TaskContext(sample_id=1, raw_sample=_sample("anatomy", 1))],
    )
    assert report["fails"] == 1
    assert report["score"] == 100.0  # 1/1; fails excluded from the denominator
    assert report["score_other"] == 100.0
