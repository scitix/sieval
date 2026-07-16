"""Unit tests for the Hendrycks MATH (DeepSeek-Math) few-shot base task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.hendrycks_math import (
    HendrycksMathDataset,
    HendrycksMathDatasetSample,
)
from sieval.tasks.hendrycks_math_kshot_base_gen import (
    N_SHOT,
    HendrycksMathFewShotBaseGenTask,
)

_FA = "\nFinal Answer: The final answer is ${}$. I hope it is correct."


class _CapturingGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(
            model=self.meta(), texts=[f"$\\boxed{{16}}${_FA.format('16')}"]
        )

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


def _sample(
    problem: str = "What is 8 + 8?",
    solution: str = "We get $\\boxed{16}$.",
) -> HendrycksMathDatasetSample:
    return {
        "problem": problem,
        "level": "Level 1",
        "type": "Algebra",
        "solution": solution,
    }


def _task() -> tuple[HendrycksMathFewShotBaseGenTask, _CapturingGenModel]:
    dataset = HendrycksMathDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(_sample())]),
                "test": HFDataset.from_list([dict(_sample())]),
            }
        )
    )
    model = _CapturingGenModel()
    return HendrycksMathFewShotBaseGenTask(dataset, model), model


def test_default_shot_count():
    assert N_SHOT == 4


@pytest.mark.anyio
async def test_preprocess_is_deepseek_minerva_prompt():
    task, _ = _task()
    raw = _sample(problem="Find $x$.")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    # 4 baked exemplars + the query block.
    assert pre.count("Problem:\n") == 5
    # DeepSeek formatting: "Solution:\n" and rstrip => ends exactly at "Solution:".
    assert pre.endswith("Problem:\nFind $x$.\n\nSolution:")
    assert "Final Answer: The final answer is" in pre


@pytest.mark.anyio
async def test_infer_forwards_deepseek_stop_only():
    task, model = _task()
    await task.infer("prompt", TaskContext(sample_id=0, raw_sample=_sample()))
    assert model.last_kwargs == {"stop": ["\nProblem:"]}
    assert "temperature" not in model.last_kwargs
    assert "max_tokens" not in model.last_kwargs


@pytest.mark.anyio
async def test_postprocess_returns_list_via_final_answer():
    task, _ = _task()
    inf = ModelOutput(model=task.model.meta(), texts=[f"reasoning{_FA.format('16')}"])
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample()))
    assert post == ["16"]


@pytest.mark.anyio
async def test_postprocess_stops_at_next_problem_block():
    # extract_math_few_shot_cot_answer drops a hallucinated next "Problem:".
    task, _ = _task()
    text = f"reasoning{_FA.format('16')}\n\nProblem:\nNext q\n\nSolution: $99$"
    inf = ModelOutput(model=task.model.meta(), texts=[text])
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample()))
    assert post == ["16"]


@pytest.mark.anyio
async def test_feedback_scores_against_solution_via_eval_math():
    task, _ = _task()
    raw = _sample(solution="Therefore $\\boxed{16}$.")
    finalize, correct_fb = await task.feedback(
        ["16"], TaskContext(sample_id=0, raw_sample=raw)
    )
    _, wrong_fb = await task.feedback(["17"], TaskContext(sample_id=1, raw_sample=raw))

    assert finalize is True
    assert correct_fb["correct"] is True
    assert correct_fb["answer"] == ["16"]
    assert correct_fb["prediction"] == ["16"]
    assert wrong_fb["correct"] is False


@pytest.mark.anyio
async def test_feedback_percentage_equivalence():
    # math_equal's numeric layer treats 50\% as 0.5 (include_percentage),
    # independent of the (env-degraded) parse_latex symbolic layer.
    task, _ = _task()
    raw = _sample(solution="So $\\boxed{0.5}$.")
    _, fb = await task.feedback(["50\\%"], TaskContext(sample_id=0, raw_sample=raw))
    assert fb["correct"] is True


@pytest.mark.anyio
async def test_report_counts_fails_as_wrong():
    # Denominator is finals + fails (DeepSeek full-set accuracy): a pipeline
    # failure counts as wrong. With 1 correct + 1 wrong final and 1 fail, the
    # score is 1/3 = 33.3 — NOT 50.0 (which excluding fails would give).
    task, _ = _task()
    raw = _sample()
    correct = TaskContext(
        sample_id=0,
        raw_sample=raw,
        feedback_result={"correct": True, "answer": ["16"], "prediction": ["16"]},
    )
    wrong = TaskContext(
        sample_id=1,
        raw_sample=raw,
        feedback_result={"correct": False, "answer": ["16"], "prediction": ["17"]},
    )
    failed = TaskContext(sample_id=2, raw_sample=raw)  # pipeline failure, no feedback
    report = await task.report([correct, wrong], [failed])
    assert report["fails"] == 1
    assert report["score"] == pytest.approx(100 / 3)
    assert report["accuracy"] == pytest.approx(100 / 3)


@pytest.mark.anyio
async def test_report_empty_is_zero():
    task, _ = _task()
    report = await task.report([], [])
    assert report == {"score": 0.0, "fails": 0, "accuracy": 0.0}
