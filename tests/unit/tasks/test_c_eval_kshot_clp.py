"""Unit tests for the C-Eval few-shot CLP task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.c_eval import CEvalDataset, CEvalDatasetSample
from sieval.tasks.c_eval_kshot_clp import CEvalFewShotCLPTask


class _ScriptedGenModel(GenModel):
    """Returns a fixed next-token top_logprobs map for every alogprobs call."""

    def __init__(self, top_logprobs: dict[str, float]):
        super().__init__(model="mock-gen", api_key="fake")
        self._top_logprobs = top_logprobs
        self.calls = 0
        self.prompts: list[str] = []

    async def _agenerate_impl(self, prompt, **kwargs):  # pragma: no cover
        raise AssertionError("clp task must not call agenerate")

    async def _alogprobs_impl(
        self,
        prompt,
        *,
        max_tokens=1,
        logprobs=100,
        echo=False,
        temperature=0.0,
        **kwargs,
    ) -> ModelOutput:
        self.calls += 1
        self.prompts.append(prompt)
        return ModelOutput(
            model=self.meta(), texts=[""], top_logprobs=[dict(self._top_logprobs)]
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


def _task(model: GenModel, k: int = 0) -> CEvalFewShotCLPTask:
    dataset = CEvalDataset(
        _hf_dict=HFDatasetDict(
            {
                "dev": HFDataset.from_list([dict(_sample("law", "A"))]),
                "test": HFDataset.from_list([dict(_sample("law", "A"))]),
            }
        )
    )
    return CEvalFewShotCLPTask(dataset, model, k=k)


@pytest.mark.anyio
async def test_argmax_picks_highest_logprob_letter():
    # B has the highest next-token logprob → prediction is "B", one call.
    model = _ScriptedGenModel({"A": -2.0, "B": -0.1, "C": -3.0, "D": -5.0})
    task = _task(model)
    raw = _sample("law", "B")
    ctx = TaskContext(sample_id=0, raw_sample=raw)

    inferred = await task.infer(await task.preprocess(raw, ctx), ctx)
    assert isinstance(inferred, ModelOutput)
    pred = await task.postprocess(inferred, ctx)

    assert pred == "B"
    assert model.calls == 1  # single top_logprobs call per sample
    _, fb = await task.feedback(pred, ctx)
    assert fb["correct"] is True
    assert fb["subject"] == "law"


@pytest.mark.anyio
async def test_non_option_tokens_are_ignored():
    # Non-option tokens (here with the highest logprobs) must be filtered out;
    # the argmax is taken only over A/B/C/D. Exercises the label-filter guard.
    model = _ScriptedGenModel(
        {"\n": -0.001, "的": -0.002, "A": -2.0, "B": -0.1, "C": -3.0, "D": -5.0}
    )
    task = _task(model)
    ctx = TaskContext(sample_id=0, raw_sample=_sample("law", "B"))
    inferred = await task.infer(await task.preprocess(_sample("law", "B"), ctx), ctx)
    assert await task.postprocess(inferred, ctx) == "B"


@pytest.mark.anyio
async def test_postprocess_raises_when_option_token_missing():
    # D absent from top-k → not all-present → fail loudly, not a subset argmax.
    model = _ScriptedGenModel({"A": -0.1, "B": -1.0, "C": -1.0})
    task = _task(model)
    ctx = TaskContext(sample_id=0, raw_sample=_sample("law", "A"))
    inferred = await task.infer(await task.preprocess(_sample("law", "A"), ctx), ctx)
    with pytest.raises(RuntimeError, match=r"missing option token"):
        await task.postprocess(inferred, ctx)


@pytest.mark.anyio
async def test_infer_does_not_generate():
    # _agenerate_impl raises if touched; reaching the assert proves only
    # alogprobs ran and top_logprobs are returned.
    model = _ScriptedGenModel({"A": -0.1, "B": -1.0, "C": -1.0, "D": -1.0})
    task = _task(model)
    raw = _sample("law", "A")
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    inferred = await task.infer(await task.preprocess(raw, ctx), ctx)
    assert inferred.top_logprobs is not None


@pytest.mark.anyio
async def test_report_macro_over_subjects_with_category_breakdown():
    # high_school_physics (STEM): 1/1 = 100%; high_school_history (Humanities):
    # 0/2 = 0%. Overall macro = mean(100, 0) = 50.0 (micro would be 1/3 ≈ 33.3).
    # Per-category scores are macro within category; absent categories omitted.
    model = _ScriptedGenModel({"A": -0.1, "B": -1.0, "C": -1.0, "D": -1.0})
    task = _task(model)
    finals = [
        TaskContext(sample_id=0, feedback_result=_fb(True, "high_school_physics")),
        TaskContext(sample_id=1, feedback_result=_fb(False, "high_school_history")),
        TaskContext(sample_id=2, feedback_result=_fb(False, "high_school_history")),
    ]
    report = await task.report(finals, [])

    assert report["score"] == pytest.approx(50.0)
    assert report["overall"] == pytest.approx(50.0)
    assert report["stem"] == pytest.approx(100.0)
    assert report["humanities"] == pytest.approx(0.0)
    assert "social_science" not in report  # no evaluated subjects → omitted
    assert "macro_accuracy" not in report
    assert report["fails"] == 0.0


@pytest.mark.anyio
async def test_prompt_format_matches_upstream_byte_for_byte():
    # Pins the upstream evaluator_series prompt: subject header, "\nX. opt"
    # options, "\n答案：", "\n\n" exemplar separators, English subject key.
    model = _ScriptedGenModel({"A": -0.1, "B": -1.0, "C": -1.0, "D": -1.0})
    task = _task(model, k=1)  # dev exemplar: q="q", answer "A"
    raw = _sample("law", "A", q="题干")
    prompt = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    expected = (
        "以下是中国关于law考试的单项选择题，请选出其中的正确答案。\n\n"
        "q\nA. a\nB. b\nC. c\nD. d\n答案：A\n\n"  # dev exemplar (with answer)
        "题干\nA. a\nB. b\nC. c\nD. d\n答案："  # test question (no answer)
    )
    assert prompt == expected


@pytest.mark.anyio
async def test_setup_raises_when_dev_split_absent():
    # No `dev` split + k>0 must fail early at setup, like the CMMLU sibling.
    dataset = CEvalDataset(
        _hf_dict=HFDatasetDict(
            {"test": HFDataset.from_list([dict(_sample("law", "A"))])}
        )
    )
    task = CEvalFewShotCLPTask(dataset, _ScriptedGenModel({"A": 0.0}), k=5)
    with pytest.raises(ValueError, match=r"requires a 'dev' split"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_raises_when_subject_has_too_few_dev_exemplars():
    # `dev` has 1 "law" exemplar but k=5 → _select_examples must fail loudly
    # rather than silently few-shot on fewer than k (matches the GSM8K
    # sibling's early-fail; near-unreachable since C-Eval dev is 5/subject).
    task = _task(_ScriptedGenModel({"A": 0.0}), k=5)
    with pytest.raises(ValueError, match=r"has 1 dev exemplars; need k=5"):
        await task.setup()
