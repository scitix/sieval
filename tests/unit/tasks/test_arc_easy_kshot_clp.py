"""
Unit tests for the ARC-Easy few-shot conditional-log-prob task (options).

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import EvalMode, TaskContext
from sieval.core.tasks.meta import get_task_meta
from sieval.datasets.arc_easy import ARCEasyDataset, ARCEasyDatasetSample
from sieval.tasks.arc_easy_kshot_clp import ARCEasyFewShotClpTask


class _TopLogprobsGenModel(GenModel):
    def __init__(self, top: dict[str, float]):
        super().__init__(model="mock-gen", api_key="fake")
        self._top = top

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])

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
        return ModelOutput(
            model=self.meta(), texts=["B"], top_logprobs=[dict(self._top)]
        )


def _sample() -> ARCEasyDatasetSample:
    return {
        "question": "Which object is hottest?",
        "choices": ["ice", "fire", "snow"],
        "answer": 1,
    }


def _task(top: dict[str, float]) -> ARCEasyFewShotClpTask:
    dataset = ARCEasyDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    return ARCEasyFewShotClpTask(dataset, _TopLogprobsGenModel(top), k=0)


@pytest.mark.anyio
async def test_preprocess_lists_options_with_letters():
    task = _task({})
    raw = _sample()

    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre == (
        "Question: Which object is hottest?\nA. ice\nB. fire\nC. snow\nAnswer:"
    )


@pytest.mark.anyio
async def test_argmax_over_option_letters():
    task = _task({" A": -3.0, " B": -0.1, " C": -2.0})  # gold index 1 -> "B"
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    _finalize, feedback = await task.feedback(post, ctx)

    assert post == 1
    assert feedback["correct"] is True


def test_task_meta_points_to_arc_easy_dataset():
    meta = get_task_meta(ARCEasyFewShotClpTask)

    assert meta.name == "arc_easy_kshot_clp"
    assert meta.dataset == "arc_easy"
    assert meta.model_type == "gen"
    assert meta.n_shot == 25
    assert meta.eval_mode == EvalMode.CLP
