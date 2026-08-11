"""Unit tests for the Hendrycks MATH (DeepSeek-Math) few-shot base task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.deepseek_math import eval_math
from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import (
    NonRetriableSampleError,
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.utils.offload import GRADE_TIMEOUT
from sieval.datasets.hendrycks_math import (
    HendrycksMathDataset,
    HendrycksMathDatasetSample,
)
from sieval.tasks import hendrycks_math_kshot_base_gen as module
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
    assert pre["prompt"].count("Problem:\n") == 5
    # DeepSeek formatting: "Solution:\n" and rstrip => ends exactly at "Solution:".
    assert pre["prompt"].endswith("Problem:\nFind $x$.\n\nSolution:")
    assert "Final Answer: The final answer is" in pre["prompt"]


@pytest.mark.anyio
async def test_infer_forwards_deepseek_stop_only():
    task, model = _task()
    await task.infer(
        {"prompt": "prompt"}, TaskContext(sample_id=0, raw_sample=_sample())
    )
    # `n` rides along because it is the sampling budget rather than a decoding
    # param; `stop` is prompt-coupled and everything else stays the caller's.
    assert model.last_kwargs == {"n": 1, "stop": ["\nProblem:"]}
    assert "temperature" not in model.last_kwargs
    assert "max_tokens" not in model.last_kwargs


@pytest.mark.anyio
async def test_postprocess_returns_list_via_final_answer():
    task, _ = _task()
    inf = ModelOutput(model=task.model.meta(), texts=[f"reasoning{_FA.format('16')}"])
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample()))
    assert post["rollouts"][0]["prediction"] == ["16"]


@pytest.mark.anyio
async def test_postprocess_stops_at_next_problem_block():
    # extract_math_few_shot_cot_answer drops a hallucinated next "Problem:".
    task, _ = _task()
    text = f"reasoning{_FA.format('16')}\n\nProblem:\nNext q\n\nSolution: $99$"
    inf = ModelOutput(model=task.model.meta(), texts=[text])
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample()))
    assert post["rollouts"][0]["prediction"] == ["16"]


@pytest.mark.anyio
async def test_feedback_scores_against_solution_via_eval_math():
    task, _ = _task()
    raw = _sample(solution="Therefore $\\boxed{16}$.")
    finalize, correct_fb = await task.feedback(
        build_prediction_record([["16"]]), TaskContext(sample_id=0, raw_sample=raw)
    )
    _, wrong_fb = await task.feedback(
        build_prediction_record([["17"]]), TaskContext(sample_id=1, raw_sample=raw)
    )

    assert finalize is True
    assert correct_fb["rollouts"][0]["correct"] is True
    assert correct_fb["reference"] == ["16"]
    assert wrong_fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_an_unextractable_gold_fails_the_sample():
    """A missing gold is our defect, so it must not be scored at all.

    With no reference there is no verdict the sample could carry — `correct`
    either way is an artifact of how the miss is handled, not evidence about the
    model. The runner turns this into a `FAILED` sample, and
    `NonRetriableSampleError` is what keeps a resume from rolling it back to
    re-infer a miss that is deterministic in the row's `solution`. Scoring it
    wrong instead would charge our extraction miss to the model, silently.
    """
    task, _ = _task()
    raw = _sample(solution="No boxed answer anywhere in this solution.")

    with pytest.raises(
        NonRetriableSampleError, match="no reference answer could be extracted"
    ):
        await task.feedback(
            build_prediction_record([["16"]]), TaskContext(sample_id=7, raw_sample=raw)
        )


@pytest.mark.anyio
async def test_a_missing_prediction_scores_wrong_rather_than_failing():
    """The opposite absence, and the opposite response.

    A prediction that would not extract is the model failing to answer, which is
    a wrong answer — `extracted` records the miss and `extraction_failure`
    reports it. It must not reach `fails`, which reads as an infrastructure
    failure. Regression guard for `or ""`: against this task's *list* reference
    that fell through `is_correct` to upstream's bare `raise
    NotImplementedError`, failing the sample with an empty message.
    """
    task, _ = _task()
    raw = _sample(solution="Therefore $\\boxed{16}$.")

    finalize, fb = await task.feedback(
        build_prediction_record([None]), TaskContext(sample_id=0, raw_sample=raw)
    )

    assert finalize is True
    assert fb["reference"] == ["16"]
    assert fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_grading_is_bounded_in_a_worker_process(monkeypatch):
    """The mechanism, not the verdict — a thread offload scores identically, so
    reverting to `anyio.to_thread.run_sync` keeps every other test in this file
    passing. Why a process: criterion 2 in `core/utils/offload.py`.
    """
    seen: dict[str, object] = {}

    async def _spy(func, *args, timeout=None):
        seen.update(func=func, args=args, timeout=timeout)
        return func(*args)

    monkeypatch.setattr(module, "run_cpu_bound", _spy)

    task, _ = _task()
    raw = _sample(solution="Therefore $\\boxed{16}$.")
    _, fb = await task.feedback(
        build_prediction_record([["16"]]), TaskContext(sample_id=0, raw_sample=raw)
    )

    assert seen["func"] is eval_math
    assert seen["args"] == ({"prediction": ["16"], "answer": ["16"]},)
    assert seen["timeout"] == GRADE_TIMEOUT
    assert fb["rollouts"][0]["correct"] is True


@pytest.mark.anyio
async def test_a_grading_timeout_scores_wrong_rather_than_failing_the_sample(
    monkeypatch,
):
    # Offloading introduced a failure mode the synchronous call did not have:
    # before, a runaway `simplify` blocked; now it raises at GRADE_TIMEOUT. Left
    # to propagate, the runner turns it into a failed sample, so a slow grade
    # shows up as `fails > 0` -- which reads as infrastructure breakage and is
    # one of the signals a run is promoted on. Every sibling math grader scores
    # an ungradeable answer wrong instead; this one has to agree.
    async def _raise_timeout(_func, *_args, **_kwargs):
        raise TimeoutError("grading took too long")

    monkeypatch.setattr(module, "run_cpu_bound", _raise_timeout)

    task, _ = _task()
    raw = _sample(solution="Therefore $\\boxed{16}$.")

    finalize, fb = await task.feedback(
        build_prediction_record([["16"]]), TaskContext(sample_id=0, raw_sample=raw)
    )

    assert finalize is True
    assert fb["rollouts"][0]["correct"] is False
    assert fb["reference"] == ["16"]


@pytest.mark.anyio
async def test_feedback_percentage_equivalence():
    # math_equal's numeric layer treats 50\% as 0.5 (include_percentage),
    # independent of the (env-degraded) parse_latex symbolic layer.
    task, _ = _task()
    raw = _sample(solution="So $\\boxed{0.5}$.")
    _, fb = await task.feedback(
        build_prediction_record([["50\\%"]]), TaskContext(sample_id=0, raw_sample=raw)
    )
    assert fb["rollouts"][0]["correct"] is True


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
        feedback_result=build_judgement_record(
            ["16"], [build_rollout_judgement(0, True)]
        ),
    )
    wrong = TaskContext(
        sample_id=1,
        raw_sample=raw,
        feedback_result=build_judgement_record(
            ["16"], [build_rollout_judgement(0, False)]
        ),
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
    assert report == {
        "score": 0.0,
        "fails": 0,
        "accuracy": 0.0,
        "score_key": "accuracy",
        "denominator_policy": "requested",
        "n_unextracted": 0.0,
    }
