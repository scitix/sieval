"""Shared contract for the four single-draw MCQ tasks.

GPQA-Diamond, MMLU, MMLU-Pro and OpenBookQA publish a greedy single-draw
accuracy, so none takes a ``k`` / ``n`` knob (RFC #74 wave 3 is deliberately not
implemented). What they must not do is read ``inf.texts[0]`` / ``rollouts[0]``
and drop the rest: ``agenerate`` merges ``{**model_kwargs, **kwargs}``, so a
model-level ``n`` reaches them and those draws were billed then discarded.

Both halves are asserted: every draw is extracted, graded and stored, and the
headline still scores the first alone.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import importlib

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext


class _StubChatModel(ChatModel):
    def __init__(self, texts):
        super().__init__(model="mock-chat", api_key="fake")
        self._texts = list(texts)

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=self._texts)

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _gpqa():
    module = importlib.import_module("sieval.tasks.gpqa_diamond_0shot_gen")
    ds_module = importlib.import_module("sieval.datasets.gpqa_diamond")
    row = {
        "Question": "What is 2+2?",
        "Correct Answer": "4",
        "Incorrect Answer 1": "3",
        "Incorrect Answer 2": "5",
        "Incorrect Answer 3": "6",
    }
    dataset = ds_module.GPQADiamondDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(row)])})
    )
    return module.GPQADiamondZeroShotGenTask, dataset, row


def _mmlu():
    module = importlib.import_module("sieval.tasks.mmlu_0shot_gen")
    ds_module = importlib.import_module("sieval.datasets.mmlu")
    row = {
        "question": "What is 2+2?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
        "subject": "abstract_algebra",
    }
    dataset = ds_module.MMLUDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(row)])})
    )
    return module.MMLUZeroShotGenTask, dataset, row


def _mmlu_pro():
    module = importlib.import_module("sieval.tasks.mmlu_pro_0shot_gen")
    ds_module = importlib.import_module("sieval.datasets.mmlu_pro")
    row = {
        "question": "What is 2+2?",
        "options": ["3", "4", "5", "6"],
        "answer": "B",
        "answer_index": 1,
        "category": "math",
        "question_id": 1,
        "src": "test",
        "cot_content": "",
    }
    dataset = ds_module.MMLUProDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(row)])})
    )
    return module.MMLUProZeroShotGenTask, dataset, row


def _openbookqa():
    module = importlib.import_module("sieval.tasks.openbookqa_kshot_gen")
    ds_module = importlib.import_module("sieval.datasets.openbookqa")
    row = {
        "id": "q1",
        "question_stem": "What is 2+2?",
        "choices": {"text": ["3", "4", "5", "6"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
    }
    dataset = ds_module.OpenBookQADataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(row)]),
                "test": HFDataset.from_list([dict(row)]),
            }
        )
    )
    return module.OpenBookQAFewShotGenTask, dataset, row


#: (factory, how the task's extractor spells a chosen letter). GPQA shuffles its
#: options behind a seeded RNG, so the gold letter is read off the prompt record
#: rather than assumed -- see `_letters`.
CASES = [
    (_gpqa, "Answer: {}"),
    (_mmlu, "Answer: {}"),
    (_mmlu_pro, "Answer: {}"),
    (_openbookqa, "{}"),
]
IDS = ["gpqa_diamond", "mmlu", "mmlu_pro", "openbookqa"]


async def _letters(task, raw, template):
    """(prompt record, a response that is right, one that is wrong)."""
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    # GPQA records the shuffled gold on the prompt; the others answer "B" by
    # construction of the fixtures above.
    gold = pre.get("reference") or "B"
    wrong = next(letter for letter in "ABCD" if letter != gold)
    return pre, template.format(gold), template.format(wrong)


@pytest.mark.parametrize("factory", [c[0] for c in CASES], ids=IDS)
def test_takes_no_sampling_knob(factory):
    # A `k` here would promise a pass@k the headline does not compute.
    task_cls, dataset, _ = factory()
    with pytest.raises(TypeError):
        task_cls(dataset, _StubChatModel(["x"]), k=2, n=2)


@pytest.mark.parametrize(("factory", "template"), CASES, ids=IDS)
@pytest.mark.anyio
async def test_every_rollout_is_extracted_and_graded(factory, template):
    # Four choices arrive (a model-level `n`), so four must land on disk.
    task_cls, dataset, raw = factory()
    task = task_cls(dataset, _StubChatModel(["x"]))
    pre, right, wrong = await _letters(task, raw, template)
    task = task_cls(dataset, _StubChatModel([right, wrong, right, wrong]))
    ctx = TaskContext(sample_id=0, raw_sample=raw, preprocess_result=pre)
    inf = await task.infer(pre, ctx)

    post = await task.postprocess(inf, ctx)
    assert len(post["rollouts"]) == 4, "a paid-for draw was discarded"

    ctx = TaskContext(
        sample_id=0, raw_sample=raw, preprocess_result=pre, postprocess_result=post
    )
    _, judgement = await task.feedback(post, ctx)
    assert len(judgement["rollouts"]) == 4
    assert [r["correct"] for r in judgement["rollouts"]] == [True, False, True, False]


@pytest.mark.parametrize(("factory", "template"), CASES, ids=IDS)
@pytest.mark.anyio
async def test_headline_scores_the_first_draw_only(factory, template):
    # Three of four correct, but the FIRST is wrong: the headline reads 0.
    task_cls, dataset, raw = factory()
    task = task_cls(dataset, _StubChatModel(["x"]))
    pre, right, wrong = await _letters(task, raw, template)
    task = task_cls(dataset, _StubChatModel([wrong, right, right, right]))
    ctx = TaskContext(sample_id=0, raw_sample=raw, preprocess_result=pre)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    ctx = TaskContext(
        sample_id=0, raw_sample=raw, preprocess_result=pre, postprocess_result=post
    )
    _, judgement = await task.feedback(post, ctx)
    assert [r["correct"] for r in judgement["rollouts"]] == [False, True, True, True]
    final = TaskContext(
        sample_id=0,
        raw_sample=raw,
        preprocess_result=pre,
        postprocess_result=post,
        feedback_result=judgement,
    )

    report = await task.report([final], [])
    assert report["score"] == pytest.approx(0.0)
    assert report[report["score_key"]] == pytest.approx(0.0)
    # No sampling metric appears: the extra draws are recorded, not scored.
    assert not {"pass@1", "avg@k", "pass@k", "maj@k", "n", "k"} & set(report)


@pytest.mark.parametrize("factory", [c[0] for c in CASES], ids=IDS)
@pytest.mark.anyio
async def test_report_declares_which_population_it_averages_over(factory):
    # All four exclude failed samples, where the DeepSeek-Math family counts
    # them wrong -- the reason the policy is recorded rather than unified.
    task_cls, dataset, _ = factory()
    task = task_cls(dataset, _StubChatModel(["x"]))
    report = await task.report([], [])
    assert report["denominator_policy"] == "judged"
