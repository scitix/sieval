"""Unit tests for the MMLU few-shot base-model clp task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.mmlu import MMLUDataset, MMLUDatasetSample
from sieval.tasks.mmlu_kshot_clp import (
    CHOICES,
    MMLUFewShotCLPTask,
    _format_example,
    _format_subject,
)


class _ScriptedGenModel(GenModel):
    """Returns a top_logprobs distribution favouring ``winner``.

    ``drop`` omits a letter from the top-k to exercise the missing-option path.
    """

    def __init__(self, winner: str = "A", drop: str | None = None):
        super().__init__(model="mock-gen", api_key="fake")
        self._winner = winner
        self._drop = drop
        self.prompts: list[str] = []
        self.call_count = 0

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
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
        self.prompts.append(prompt)
        self.call_count += 1
        assert echo is False  # clp reads the next-token distribution, no echo
        dist = {
            f" {label}": (-0.1 if label == self._winner else -5.0)
            for label in CHOICES
            if label != self._drop
        }
        return ModelOutput(model=self.meta(), texts=[""], top_logprobs=[dist])


def _sample(
    subject: str = "abstract_algebra",
    question: str = "Q?",
    answer: int = 0,
    choices: list[str] | None = None,
) -> MMLUDatasetSample:
    return {
        "question": question,
        "subject": subject,
        "choices": choices or ["c0", "c1", "c2", "c3"],
        "answer": answer,
    }


def _dataset(
    dev_rows: list[MMLUDatasetSample], test_rows: list[MMLUDatasetSample]
) -> MMLUDataset:
    return MMLUDataset(
        _hf_dict=HFDatasetDict(
            {
                "dev": HFDataset.from_list([dict(r) for r in dev_rows]),
                "test": HFDataset.from_list([dict(r) for r in test_rows]),
            }
        )
    )


# --- Prompt format pinning (must match hendrycks/test evaluate_flan.py) ---


def test_format_subject_matches_reference():
    assert _format_subject("abstract_algebra") == " abstract algebra"
    assert _format_subject("anatomy") == " anatomy"


def test_format_example_with_and_without_answer():
    sample = _sample(question="What?", answer=2, choices=["w", "x", "y", "z"])
    assert (
        _format_example(sample, include_answer=False)
        == "What?\nA. w\nB. x\nC. y\nD. z\nAnswer:"
    )
    assert (
        _format_example(sample, include_answer=True)
        == "What?\nA. w\nB. x\nC. y\nD. z\nAnswer: C\n\n"
    )


@pytest.mark.anyio
async def test_header_and_prompt_are_byte_exact():
    dev = [_sample("anatomy", "S?", 0)]
    test = _sample("anatomy", "TESTQ", 1)
    task = MMLUFewShotCLPTask(_dataset(dev, [test]), _ScriptedGenModel(), k=1)
    await task.setup()

    pre = await task.preprocess(test, TaskContext(sample_id=0, raw_sample=test))
    assert pre == (
        "The following are multiple choice questions (with answers) about"
        " anatomy.\n\n"
        "S?\nA. c0\nB. c1\nC. c2\nD. c3\nAnswer: A\n\n"
        "TESTQ\nA. c0\nB. c1\nC. c2\nD. c3\nAnswer:"
    )


# --- Few-shot: first k dev examples per subject, in order ---


@pytest.mark.anyio
async def test_fewshot_is_per_subject_fixed_order():
    dev_rows = [_sample("anatomy", f"a{i}", 0) for i in range(7)] + [
        _sample("astronomy", f"s{i}", 1) for i in range(7)
    ]
    task = MMLUFewShotCLPTask(
        _dataset(dev_rows, [_sample("anatomy", "t", 0)]),
        _ScriptedGenModel(),
        k=5,
    )
    await task.setup()

    anatomy = task._select_examples("anatomy")
    astronomy = task._select_examples("astronomy")
    assert [s["question"] for s in anatomy] == ["a0", "a1", "a2", "a3", "a4"]
    assert [s["question"] for s in astronomy] == ["s0", "s1", "s2", "s3", "s4"]


@pytest.mark.anyio
async def test_preprocess_uses_only_matching_subject_shots():
    dev_rows = [_sample("anatomy", f"a{i}", 0) for i in range(5)] + [
        _sample("astronomy", f"s{i}", 1) for i in range(5)
    ]
    test = _sample("astronomy", "TESTQ", 2)
    task = MMLUFewShotCLPTask(_dataset(dev_rows, [test]), _ScriptedGenModel(), k=5)
    await task.setup()

    pre = await task.preprocess(test, TaskContext(sample_id=0, raw_sample=test))
    assert " astronomy." in pre and "anatomy" not in pre
    assert pre.count("\nAnswer:") == 6  # 5 shots + test
    assert pre.rstrip().endswith("Answer:")


# --- Scoring: single echo=False call, argmax over top_logprobs ---


@pytest.mark.anyio
async def test_infer_issues_single_call_echo_false():
    test = _sample("anatomy", "Q", 0)
    model = _ScriptedGenModel(winner="C")
    ds = _dataset([_sample("anatomy", "d", 0)], [test])
    task = MMLUFewShotCLPTask(ds, model, k=1)
    await task.setup()
    ctx = TaskContext(sample_id=0, raw_sample=test)

    pre = await task.preprocess(test, ctx)
    await task.infer(pre, ctx)
    assert model.call_count == 1  # ONE call, not 4


@pytest.mark.anyio
async def test_postprocess_argmax_and_missing_choice_raises():
    test = _sample("anatomy", "Q", 0)
    ds = _dataset([_sample("anatomy", "d", 0)], [test])
    ctx = TaskContext(sample_id=0, raw_sample=test)

    task = MMLUFewShotCLPTask(ds, _ScriptedGenModel(winner="C"), k=1)
    await task.setup()
    pre = await task.preprocess(test, ctx)
    inf = await task.infer(pre, ctx)
    assert await task.postprocess(inf, ctx) == "C"

    dropped = MMLUFewShotCLPTask(ds, _ScriptedGenModel(winner="A", drop="D"), k=1)
    await dropped.setup()
    inf2 = await dropped.infer(await dropped.preprocess(test, ctx), ctx)
    with pytest.raises(RuntimeError, match="missing option token"):
        await dropped.postprocess(inf2, ctx)


# --- Feedback + micro report (sibling-consistent with mmlu_0shot_gen) ---


@pytest.mark.anyio
async def test_feedback_and_report_micro_with_categories():
    test = _sample("abstract_algebra", "Q", 2)  # gold "C"
    task = MMLUFewShotCLPTask(
        _dataset([_sample("abstract_algebra", "d", 0)], [test]),
        _ScriptedGenModel(winner="C"),
        k=1,
    )
    await task.setup()
    ctx = TaskContext(sample_id=0, raw_sample=test)

    inf = await task.infer(await task.preprocess(test, ctx), ctx)
    post = await task.postprocess(inf, ctx)
    finalize, fb = await task.feedback(post, ctx)
    assert finalize is True
    assert fb["correct"] is True
    assert fb["answer"] == "C" and fb["prediction"] == "C"
    assert fb["subject"] == "abstract_algebra" and fb["category"] == "stem"

    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=test, feedback_result=fb)], []
    )
    assert report["score"] == 100.0
    assert report["score_stem"] == 100.0
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_feedback_marks_wrong_prediction():
    test = _sample("anatomy", "Q", 0)  # gold "A"
    task = MMLUFewShotCLPTask(
        _dataset([_sample("anatomy", "d", 0)], [test]),
        _ScriptedGenModel(winner="D"),
        k=1,
    )
    await task.setup()
    ctx = TaskContext(sample_id=0, raw_sample=test)
    inf = await task.infer(await task.preprocess(test, ctx), ctx)
    _, fb = await task.feedback(await task.postprocess(inf, ctx), ctx)
    assert fb["correct"] is False and fb["prediction"] == "D"


@pytest.mark.anyio
async def test_report_excludes_fails_from_denominator():
    # CLP-family convention (cmmlu / mmmlu): 1 correct final + 1 pipeline failure
    # → score 100.0 over the finalized set; the fail is reported separately, not
    # scored wrong. A len(finals)+len(fails) denominator would give 50.0.
    good = _sample("anatomy", "Q", 0)  # gold "A"
    failed = _sample("anatomy", "Q2", 1)
    task = MMLUFewShotCLPTask(
        _dataset([_sample("anatomy", "d", 0)], [good]),
        _ScriptedGenModel(winner="A"),
        k=1,
    )
    await task.setup()
    ctx = TaskContext(sample_id=0, raw_sample=good)
    inf = await task.infer(await task.preprocess(good, ctx), ctx)
    _, fb = await task.feedback(await task.postprocess(inf, ctx), ctx)
    assert fb["correct"] is True

    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=good, feedback_result=fb)],
        [TaskContext(sample_id=1, raw_sample=failed)],
    )
    assert report["fails"] == 1
    assert report["score"] == 100.0  # 1/1; a finals+fails denom would give 50.0
    assert report["score_other"] == 100.0  # only the final buckets; fail excluded


# --- Validation ---


def test_invalid_k_and_logprobs_rejected():
    ds = _dataset([_sample()], [_sample()])
    with pytest.raises(ValueError, match="k must be >= 0"):
        MMLUFewShotCLPTask(ds, _ScriptedGenModel(), k=-1)
    with pytest.raises(ValueError, match="logprobs must be >= 1"):
        MMLUFewShotCLPTask(ds, _ScriptedGenModel(), logprobs=0)


@pytest.mark.anyio
async def test_missing_fewshot_split_raises():
    ds = MMLUDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    task = MMLUFewShotCLPTask(ds, _ScriptedGenModel(), k=5)
    with pytest.raises(ValueError, match="dev"):
        await task.setup()
