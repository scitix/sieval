"""Shared sampling contract for the four accuracy-headline math tasks.

GSM8K (chat and few-shot base), Hendrycks MATH and TheoremQA report an
``accuracy``-style headline rather than ``pass@1``, and each was single-draw by
construction: ``postprocess`` read ``inf.texts[0]``, ``feedback`` graded
``rollouts[0]``, and there was no ``k`` / ``n`` knob at all. RFC #74 calls that
wave 2; item E is what it needs first.

Two things are asserted here that no per-task file can, because they are
cross-task decisions:

1. **The whole draw is kept and graded.** A task that still caps at rollout 0
   passes every one of its own tests and silently pays for ``n`` generations to
   score one.
2. **The headline stays FIRST-ROLLOUT under n > 1.** These benchmarks publish a
   greedy single-draw number, so ``accuracy`` must keep meaning that; ``pass@1``
   (``c/n``, the better estimator of the same quantity) is reported beside it,
   never merged into it. At n=1 the two coincide, which is what makes adopting a
   budget non-breaking.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import importlib

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.gsm8k import GSM8KDataset
from sieval.datasets.hendrycks_math import HendrycksMathDataset
from sieval.tasks.gsm8k_0shot_gen import GSM8KZeroShotGenTask
from sieval.tasks.gsm8k_kshot_base_gen import GSM8KFewShotBaseGenTask
from sieval.tasks.hendrycks_math_kshot_base_gen import HendrycksMathFewShotBaseGenTask

#: Four rollouts: the FIRST is wrong, the other three agree on the gold. Chosen
#: so every metric this file cares about takes a different value -- accuracy 0,
#: pass@1 75, pass@k 100, maj@k 100 -- and a task that collapses any two of them
#: cannot pass.
FOUR_DRAWS_FIRST_WRONG = 4


class _StubChatModel(ChatModel):
    def __init__(self, texts):
        super().__init__(model="mock-chat", api_key="fake")
        self.last_kwargs: dict[str, object] = {}
        self._texts = list(texts)

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        n = int(kwargs.get("n", 1))
        return ModelOutput(model=self.meta(), texts=self._texts[:n])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


class _StubGenModel(GenModel):
    def __init__(self, texts):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}
        self._texts = list(texts)

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        n = int(kwargs.get("n", 1))
        return ModelOutput(model=self.meta(), texts=self._texts[:n])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


# --------------------------------------------------------------------------- #
# One adapter per task: how to build it, what a sample looks like, and what a
# response saying `answer` looks like in its own output dialect.
# --------------------------------------------------------------------------- #


def _gsm8k_dataset():
    row = {"question": "What is 40 + 2?", "answer": "Solution.\n#### 42"}
    return GSM8KDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(row)]),
                "test": HFDataset.from_list([dict(row)]),
            }
        )
    ), row


def _hendrycks_dataset():
    row = {
        "problem": "What is 8 + 8?",
        "level": "Level 1",
        "type": "Algebra",
        "solution": "We get $\\boxed{16}$.",
    }
    return HendrycksMathDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(row)]),
                "test": HFDataset.from_list([dict(row)]),
            }
        )
    ), row


def _theoremqa_dataset():
    module = importlib.import_module("sieval.datasets.theoremqa")
    row = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    return module.TheoremQADataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(row)])})
    ), row


def _theoremqa_task_cls():
    return importlib.import_module(
        "sieval.tasks.theoremqa_kshot_base_gen"
    ).TheoremQAKShotBaseGenTask


class _Case:
    def __init__(self, name, build, dataset, model_cls, gold, wrong, kwargs=None):
        self.name = name
        self._build = build
        self._dataset = dataset
        self._model_cls = model_cls
        self.gold = gold
        self.wrong = wrong
        self._kwargs = kwargs or {}

    def task(self, *, k=1, n=1, texts=None):
        dataset, raw = self._dataset()
        model = self._model_cls(texts if texts is not None else [self.gold])
        task = self._build(dataset, model, k=k, n=n, **self._kwargs)
        return task, model, raw


CASES = [
    _Case(
        "gsm8k_0shot_gen",
        lambda ds, m, **kw: GSM8KZeroShotGenTask(ds, m, **kw),
        _gsm8k_dataset,
        _StubChatModel,
        gold="The answer is $\\boxed{42}$.",
        wrong="The answer is $\\boxed{7}$.",
    ),
    _Case(
        "gsm8k_kshot_base_gen",
        lambda ds, m, **kw: GSM8KFewShotBaseGenTask(ds, m, **kw),
        _gsm8k_dataset,
        _StubGenModel,
        # lm-eval's strict match is the `#### ` delimiter, not prose.
        gold="Work it out.\n#### 42",
        wrong="Work it out.\n#### 7",
        kwargs={"n_shot": 0},
    ),
    _Case(
        "hendrycks_math_kshot_base_gen",
        lambda ds, m, **kw: HendrycksMathFewShotBaseGenTask(ds, m, **kw),
        _hendrycks_dataset,
        _StubGenModel,
        gold="So $\\boxed{16}$.",
        wrong="So $\\boxed{9}$.",
    ),
    _Case(
        "theoremqa_kshot_base_gen",
        lambda ds, m, **kw: _theoremqa_task_cls()(ds, m, **kw),
        _theoremqa_dataset,
        _StubGenModel,
        gold="The answer is 4",
        wrong="The answer is 9",
        kwargs={"n_shot": 0},
    ),
]
IDS = [c.name for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_k_greater_than_n_is_rejected(case):
    # Without the guard this constructs fine and pass@k comes out a confident
    # 0.0 (the `n < k` guard in pass_at_k) beside a real pass@1.
    with pytest.raises(ValueError, match=r"pass@2"):
        case.task(k=2, n=1)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_k_equal_to_n_is_accepted(case):
    assert case.task(k=4, n=4)[0] is not None


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.anyio
async def test_infer_forwards_the_sampling_budget(case):
    # `n` on the model is merged as `{**model_kwargs, **task_kwargs}`, so a task
    # that does not pass it lets a model-level `n` through to a postprocess that
    # keeps one draw -- paid for, discarded, and silent.
    task, model, raw = case.task(k=4, n=4)
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    await task.infer(pre, TaskContext(sample_id=0, raw_sample=raw))
    assert model.last_kwargs.get("n") == 4


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.anyio
async def test_the_whole_draw_is_extracted_and_graded(case):
    # The wave-2 defect exactly: `inf.texts[0]` in postprocess and `rollouts[0]`
    # in feedback. Either one alone leaves this at a length of 1.
    texts = [case.wrong] + [case.gold] * 3
    task, model, raw = case.task(k=4, n=4, texts=texts)
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    assert len(post["rollouts"]) == FOUR_DRAWS_FIRST_WRONG

    ctx = TaskContext(sample_id=0, raw_sample=raw, postprocess_result=post)
    _, judgement = await task.feedback(post, ctx)
    assert len(judgement["rollouts"]) == FOUR_DRAWS_FIRST_WRONG
    assert [r["correct"] for r in judgement["rollouts"]] == [False, True, True, True]


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.anyio
async def test_headline_stays_first_rollout_while_pass_at_1_averages(case):
    # The alignment decision, pinned: `accuracy` is what the paper published --
    # one greedy draw -- and `pass@1` is c/n over the whole draw. Merging them
    # would silently restate every published-number comparison.
    texts = [case.wrong] + [case.gold] * 3
    task, _, raw = case.task(k=4, n=4, texts=texts)
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    _, judgement = await task.feedback(
        post, TaskContext(sample_id=0, raw_sample=raw, postprocess_result=post)
    )
    final = TaskContext(
        sample_id=0,
        raw_sample=raw,
        postprocess_result=post,
        feedback_result=judgement,
    )

    report = await task.report([final], [])
    headline = report[report["score_key"]]
    assert headline == pytest.approx(0.0), "first rollout was wrong"
    assert report["score"] == headline
    assert report["pass@1"] == pytest.approx(75.0), "3 of 4 draws correct"
    assert report["avg@k"] == pytest.approx(75.0)
    assert report["pass@k"] == pytest.approx(100.0), "solved at least once"
    assert report["maj@k"] == pytest.approx(100.0), "the modal answer is correct"
    assert (report["n"], report["k"], report["n_short"]) == (4.0, 4.0, 0.0)


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.anyio
async def test_n_equals_one_reports_no_sampling_block(case):
    # The non-breaking property: at the default budget the report gains exactly
    # `score_key` and nothing else, so a stored row keeps every column it had.
    task, _, raw = case.task(k=1, n=1)
    report = await task.report([], [])
    assert report["score_key"] in report
    assert not {"avg@k", "pass@k", "maj@k", "n", "k", "n_short"} & set(report)
