"""
Unit tests for the ARC-Challenge few-shot perplexity task (full text, uncond norm).

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
from sieval.tasks._arc import ARC_UNCOND_CONTEXT, echoed_logprob
from sieval.tasks.arc_challenge_kshot_ppl import ARCChallengeFewShotPplTask


class _ScriptedGenModel(GenModel):
    """Returns a scripted logprob per (option, conditional/unconditional).

    ``scores`` maps option text -> (conditional_total, unconditional_total). The
    echoed input is 2 tokens (first ``None`` per skip_first) whose summed logprob
    is ``value``; a 3rd, trailing GENERATED token (logprob -99) is appended and
    must be excluded by ``echoed_logprob`` (``usage.input_tokens == 2``).
    """

    def __init__(self, scores: dict[str, tuple[float, float]]):
        super().__init__(model="mock-gen", api_key="fake")
        self._scores = scores
        self.prompts: list[str] = []

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
        _ = (max_tokens, temperature, kwargs)
        assert echo is True
        assert logprobs == 0  # ppl requests no top-k (matches hellaswag sibling)
        self.prompts.append(prompt)
        option = prompt.split("Answer:")[-1].strip()
        cond_lp, uncond_lp = self._scores[option]
        value = uncond_lp if prompt.startswith(ARC_UNCOND_CONTEXT) else cond_lp
        return ModelOutput(
            model=self.meta(),
            texts=[""],
            logprobs_tokens=["_ctx", "_opt", "_generated"],
            logprobs=[None, value, -99.0],
            usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )


def _train() -> list[ARCChallengeDatasetSample]:
    return [{"question": "Sky color?", "choices": ["blue", "red"], "answer": 0}]


def _sample() -> ARCChallengeDatasetSample:
    return {
        "question": "Which material is a conductor?",
        "choices": ["copper", "rubber", "wood"],
        "answer": 0,
    }


def _task(
    scores: dict[str, tuple[float, float]], *, k: int = 0
) -> tuple[ARCChallengeFewShotPplTask, _ScriptedGenModel]:
    dataset = ARCChallengeDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(s) for s in _train()]),
                "test": HFDataset.from_list([dict(_sample())]),
            }
        )
    )
    model = _ScriptedGenModel(scores)
    return ARCChallengeFewShotPplTask(dataset, model, k=k, fewshot_seed=0), model


@pytest.mark.anyio
async def test_preprocess_full_text_no_letters_no_choices_header():
    task, _model = _task({}, k=1)
    raw = _sample()

    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre == (
        "Question: Sky color?\n"
        "Answer: blue\n\n"  # exemplar answer is the correct option TEXT, not a letter
        "Question: Which material is a conductor?\n"
        "Answer:"
    )
    assert "Choices:" not in pre
    assert "A." not in pre  # no letter labels


@pytest.mark.anyio
async def test_infer_two_calls_per_option_conditional_and_unconditional():
    scores = dict.fromkeys(_sample()["choices"], (-1.0, -1.0))
    task, model = _task(scores, k=0)
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)

    await task.infer(pre, ctx)

    assert len(model.prompts) == 6  # 3 options x (conditional + unconditional)
    for i, choice in enumerate(_sample()["choices"]):
        cond, uncond = model.prompts[2 * i], model.prompts[2 * i + 1]
        assert cond == f"{pre} {choice}"
        assert uncond == f"{ARC_UNCOND_CONTEXT} {choice}"


@pytest.mark.anyio
async def test_unconditional_normalization_flips_argmax():
    # Raw conditional favours "rubber" (-4 > -5), but after subtracting the
    # unconditional term the gold "copper" wins (-1 > -3). Verifies the task
    # actually normalizes rather than argmaxing raw conditional likelihood.
    scores = {
        "copper": (-5.0, -4.0),  # score -1  (gold, index 0)
        "rubber": (-4.0, -1.0),  # score -3
        "wood": (-6.0, -2.0),  # score -4
    }
    task, _model = _task(scores, k=0)
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    _finalize, feedback = await task.feedback(post, ctx)
    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)], []
    )

    assert post == 0  # copper, not rubber
    assert feedback["correct"] is True
    assert feedback["prediction_choice"] == "copper"
    assert report == {"score": 100.0, "acc": 100.0, "fails": 0}


def test_echoed_logprob_excludes_trailing_generated_token():
    # echo=True + max_tokens>=1 returns the echoed input followed by a generated
    # token; only the input (usage.input_tokens) is the scored sequence.
    model = _ScriptedGenModel({})
    out = ModelOutput(
        model=model.meta(),
        texts=[""],
        logprobs_tokens=["a", "b", "GEN"],
        logprobs=[None, -1.5, -99.0],
        usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )
    # first 2 tokens only (skip_first drops the None) → -1.5, NOT -1.5 + -99.0
    assert echoed_logprob(out) == -1.5


def test_negative_k_rejected():
    dataset = ARCChallengeDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    with pytest.raises(ValueError, match="k must be >= 0"):
        ARCChallengeFewShotPplTask(dataset, _ScriptedGenModel({}), k=-1)


def test_task_meta_points_to_arc_challenge_dataset():
    meta = get_task_meta(ARCChallengeFewShotPplTask)

    assert meta.name == "arc_challenge_kshot_ppl"
    assert meta.dataset == "arc_challenge"
    assert meta.model_type == "gen"
    assert meta.n_shot == 25
    assert meta.eval_mode == EvalMode.PPL
