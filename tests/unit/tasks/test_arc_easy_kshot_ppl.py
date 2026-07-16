"""
Unit tests for the ARC-Easy few-shot perplexity task (full text, uncond norm).

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
from sieval.tasks._arc import ARC_UNCOND_CONTEXT
from sieval.tasks.arc_easy_kshot_ppl import ARCEasyFewShotPplTask


class _ScriptedGenModel(GenModel):
    def __init__(self, scores: dict[str, tuple[float, float]]):
        super().__init__(model="mock-gen", api_key="fake")
        self._scores = scores

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
        _ = (max_tokens, logprobs, temperature, echo, kwargs)
        option = prompt.split("Answer:")[-1].strip()
        cond_lp, uncond_lp = self._scores[option]
        value = uncond_lp if prompt.startswith(ARC_UNCOND_CONTEXT) else cond_lp
        return ModelOutput(
            model=self.meta(),
            texts=[""],
            logprobs_tokens=["_", "_"],
            logprobs=[None, value],
            usage={"input_tokens": 2, "output_tokens": 0, "total_tokens": 2},
        )


def _sample() -> ARCEasyDatasetSample:
    return {
        "question": "Which object is hottest?",
        "choices": ["ice", "fire", "snow"],
        "answer": 1,
    }


def _task(scores: dict[str, tuple[float, float]]) -> ARCEasyFewShotPplTask:
    dataset = ARCEasyDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    return ARCEasyFewShotPplTask(dataset, _ScriptedGenModel(scores), k=0)


@pytest.mark.anyio
async def test_preprocess_full_text_no_letters():
    task = _task({})
    raw = _sample()

    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre == "Question: Which object is hottest?\nAnswer:"
    assert "Choices:" not in pre


@pytest.mark.anyio
async def test_unconditional_normalization_argmax():
    # "fire" (gold, index 1) wins only after unconditional subtraction.
    scores = {
        "ice": (-4.0, -1.0),  # score -3
        "fire": (-5.0, -4.0),  # score -1  (gold)
        "snow": (-6.0, -2.0),  # score -4
    }
    task = _task(scores)
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    _finalize, feedback = await task.feedback(post, ctx)

    assert post == 1
    assert feedback["correct"] is True


def test_task_meta_points_to_arc_easy_dataset():
    meta = get_task_meta(ARCEasyFewShotPplTask)

    assert meta.name == "arc_easy_kshot_ppl"
    assert meta.dataset == "arc_easy"
    assert meta.model_type == "gen"
    assert meta.n_shot == 25
    assert meta.eval_mode == EvalMode.PPL
