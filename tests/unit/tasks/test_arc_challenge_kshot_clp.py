"""
Unit tests for the ARC-Challenge few-shot conditional-log-prob task (options).

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import EvalMode, TaskContext
from sieval.core.tasks.meta import get_task_meta
from sieval.datasets.arc_challenge import (
    ARCChallengeDataset,
    ARCChallengeDatasetSample,
)
from sieval.tasks.arc_challenge_kshot_clp import ARCChallengeFewShotClpTask


class _TopLogprobsGenModel(GenModel):
    """Returns a fixed next-token top_logprobs map; records the prompt + echo."""

    def __init__(self, top: dict[str, float]):
        super().__init__(model="mock-gen", api_key="fake")
        self._top = top
        self.prompts: list[str] = []
        self.echo_flags: list[bool] = []

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
        _ = (max_tokens, logprobs, temperature, kwargs)
        self.prompts.append(prompt)
        self.echo_flags.append(echo)
        return ModelOutput(
            model=self.meta(),
            texts=["A"],
            top_logprobs=[dict(self._top)],
        )


def _train() -> list[ARCChallengeDatasetSample]:
    return [{"question": "1+1?", "choices": ["1", "2", "3"], "answer": 1}]


def _sample() -> ARCChallengeDatasetSample:
    return {
        "question": "Which material is a conductor?",
        "choices": ["copper", "rubber", "wood"],
        "answer": 0,
    }


def _task(
    top: dict[str, float], *, k: int = 0, logprobs: int = 100
) -> tuple[ARCChallengeFewShotClpTask, _TopLogprobsGenModel]:
    dataset = ARCChallengeDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(s) for s in _train()]),
                "test": HFDataset.from_list([dict(_sample())]),
            }
        )
    )
    model = _TopLogprobsGenModel(top)
    return (
        ARCChallengeFewShotClpTask(dataset, model, k=k, logprobs=logprobs),
        model,
    )


@pytest.mark.anyio
async def test_preprocess_lists_options_with_letters():
    task, _model = _task({}, k=1)
    raw = _sample()

    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre == (
        "Question: 1+1?\n"
        "A. 1\n"
        "B. 2\n"
        "C. 3\n"
        "Answer: B\n\n"  # exemplar answer is the LETTER
        "Question: Which material is a conductor?\n"
        "A. copper\n"
        "B. rubber\n"
        "C. wood\n"
        "Answer:"
    )


@pytest.mark.anyio
async def test_single_call_echo_false():
    task, model = _task({" A": -0.1, " B": -2.0, " C": -3.0})
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)

    await task.infer(pre, ctx)

    assert len(model.prompts) == 1  # one inference per sample
    assert model.echo_flags == [False]


@pytest.mark.anyio
async def test_argmax_over_option_letters():
    # Favour " A" (gold index 0); " B"/" C" lower. Token has a leading space
    # (" A") — the scorer strips it to "A".
    task, _model = _task({" A": -0.1, " B": -2.0, " C": -3.0, " the": -0.5})
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    _finalize, feedback = await task.feedback(post, ctx)
    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)], []
    )

    assert post == 0
    assert feedback["correct"] is True
    assert report == {"score": 100.0, "acc": 100.0, "fails": 0}


@pytest.mark.anyio
async def test_missing_option_letter_fails_loud():
    # top-k omits "C" (a 3-option sample needs A/B/C) → RuntimeError, not a guess.
    task, _model = _task({"A": -0.1, "B": -2.0})
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    with pytest.raises(RuntimeError, match="missing option token"):
        await task.postprocess(inf, ctx)


def test_negative_k_rejected():
    dataset = ARCChallengeDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    with pytest.raises(ValueError, match="k must be >= 0"):
        ARCChallengeFewShotClpTask(dataset, _TopLogprobsGenModel({}), k=-1)


def test_task_meta_points_to_arc_challenge_dataset():
    meta = get_task_meta(ARCChallengeFewShotClpTask)

    assert meta.name == "arc_challenge_kshot_clp"
    assert meta.dataset == "arc_challenge"
    assert meta.model_type == "gen"
    assert meta.n_shot == 25
    assert meta.eval_mode == EvalMode.CLP
