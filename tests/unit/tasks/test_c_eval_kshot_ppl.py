"""Unit tests for the C-Eval few-shot next-token logprob task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext, TaskStageOutput
from sieval.datasets.c_eval import CEvalDataset, CEvalDatasetSample
from sieval.tasks.c_eval_kshot_ppl import CEvalFewShotPPLTask


class _ScriptedGenModel(GenModel):
    """Returns a preset logprob for whichever letter the prompt ends with."""

    def __init__(self, letter_logprobs: dict[str, float]):
        super().__init__(model="mock-gen", api_key="fake")
        self._letter_logprobs = letter_logprobs
        self.prompts: list[str] = []

    async def _agenerate_impl(  # pragma: no cover
        self, prompt: str, **kwargs
    ) -> ModelOutput:
        raise AssertionError("ppl task must not call agenerate")

    async def _alogprobs_impl(
        self,
        prompt: str,
        *,
        max_tokens=1,
        logprobs=5,
        echo=True,
        temperature=0.0,
        **kwargs,
    ) -> ModelOutput:
        self.prompts.append(prompt)
        letter = prompt[-1]
        return ModelOutput(
            model=self.meta(),
            texts=[""],
            logprobs_tokens=[letter],
            logprobs=[self._letter_logprobs[letter]],
        )


def _sample(subject: str, answer: str, q: str = "q") -> CEvalDatasetSample:
    return {
        "question": q,
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "answer": answer,
        "subject": subject,
    }


def _fb(correct: bool, subject: str) -> dict:
    return {"correct": correct, "pred": "A", "answer": "A", "subject": subject}


def _task(model: GenModel, k: int = 0) -> CEvalFewShotPPLTask:
    dataset = CEvalDataset(
        _hf_dict=HFDatasetDict(
            {
                "dev": HFDataset.from_list([dict(_sample("law", "A"))]),
                "test": HFDataset.from_list([dict(_sample("law", "A"))]),
            }
        )
    )
    return CEvalFewShotPPLTask(dataset, model, k=k)


@pytest.mark.anyio
async def test_argmax_picks_highest_logprob_letter():
    # B has the highest conditional logprob → prediction is "B".
    model = _ScriptedGenModel({"A": -2.0, "B": -0.1, "C": -3.0, "D": -5.0})
    task = _task(model)
    raw = _sample("law", "B")
    ctx = TaskContext(sample_id=0, raw_sample=raw)

    inferred = await task.infer(raw, ctx)
    assert isinstance(inferred, TaskStageOutput)
    pred = await task.postprocess(inferred, ctx)

    assert pred == "B"
    assert len(model.prompts) == 4  # one echo call per candidate letter
    _, fb = await task.feedback(pred, ctx)
    assert fb["correct"] is True
    assert fb["subject"] == "law"


@pytest.mark.anyio
async def test_infer_does_not_generate():
    model = _ScriptedGenModel({"A": -0.1, "B": -1.0, "C": -1.0, "D": -1.0})
    task = _task(model)
    raw = _sample("law", "A")
    # _agenerate_impl raises if touched; reaching the assert proves only
    # alogprobs ran.
    inferred = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert inferred.value["A"] == -0.1


@pytest.mark.anyio
async def test_report_is_macro_average_over_subjects():
    # phys: 1/1 correct = 100%; hist: 0/2 correct = 0%.
    # Macro = mean(100, 0) = 50.0; a micro-average would be 1/3 ≈ 33.3.
    model = _ScriptedGenModel({"A": -0.1, "B": -1.0, "C": -1.0, "D": -1.0})
    task = _task(model)
    finals = [
        TaskContext(sample_id=0, feedback_result=_fb(True, "physics")),
        TaskContext(sample_id=1, feedback_result=_fb(False, "history")),
        TaskContext(sample_id=2, feedback_result=_fb(False, "history")),
    ]
    report = await task.report(finals, [])

    assert report["score"] == pytest.approx(50.0)
    assert report["macro_accuracy"] == pytest.approx(50.0)
    assert "pass@1" not in report
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_prompt_format_matches_upstream_byte_for_byte():
    # Pins the upstream evaluator_series prompt: subject header, "\nX. opt"
    # options, "\n答案：", "\n\n" exemplar separators, English subject key.
    model = _ScriptedGenModel({"A": -0.1, "B": -1.0, "C": -1.0, "D": -1.0})
    task = _task(model, k=1)  # dev exemplar: q="q", answer "A"
    raw = _sample("law", "A", q="题干")
    await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))

    expected = (
        "以下是中国关于law考试的单项选择题，请选出其中的正确答案。\n\n"
        "q\nA. a\nB. b\nC. c\nD. d\n答案：A\n\n"  # dev exemplar (with answer)
        "题干\nA. a\nB. b\nC. c\nD. d\n答案："  # test question (no answer)
        "A"  # candidate letter appended for scoring (prompts[0] == "A")
    )
    assert model.prompts[0] == expected
