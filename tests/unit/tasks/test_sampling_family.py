"""One sampling contract, over eight tasks whose headline is a single draw.

Wave 2 (GSM8K x2, Hendrycks MATH, TheoremQA) TAKES a ``k`` / ``n`` budget;
wave 3 (GPQA-Diamond, MMLU, MMLU-Pro, OpenBookQA) deliberately REFUSES one.
Either way two clauses hold, and neither is assertable per-task:

1. **No draw is discarded.** ``agenerate`` merges ``{**model_kwargs, **kwargs}``,
   so a model-level ``n`` reaches even a task passing none of its own.
2. **The headline scores the FIRST draw** — all eight publish a greedy
   single-draw number, with ``pass@1`` beside it where a budget exists.

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

DRAW = 4


class _StubMixin:
    """Configured texts, capped at whatever ``n`` the task asked for.

    A task without its own budget takes what the MODEL was configured for --
    the case clause 1 exists for.
    """

    _texts: list[str]
    last_kwargs: dict

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        n = int(kwargs.get("n", len(self._texts)))
        return ModelOutput(model=self.meta(), texts=self._texts[:n])  # type: ignore[attr-defined]

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])  # type: ignore[attr-defined]


class _StubChatModel(_StubMixin, ChatModel):
    def __init__(self, texts):
        ChatModel.__init__(self, model="mock-chat", api_key="fake")
        self.last_kwargs = {}
        self._texts = list(texts)


class _StubGenModel(_StubMixin, GenModel):
    def __init__(self, texts):
        GenModel.__init__(self, model="mock-gen", api_key="fake")
        self.last_kwargs = {}
        self._texts = list(texts)


# --------------------------------------------------------------------------- #
# Datasets. One row each, the smallest shape the loader accepts.
# --------------------------------------------------------------------------- #


def _hf(**splits):
    return HFDatasetDict({k: HFDataset.from_list([dict(v)]) for k, v in splits.items()})


def _gsm8k():
    module = importlib.import_module("sieval.datasets.gsm8k")
    row = {"question": "What is 40 + 2?", "answer": "Solution.\n#### 42"}
    return module.GSM8KDataset(_hf_dict=_hf(train=row, test=row)), row


def _hendrycks():
    module = importlib.import_module("sieval.datasets.hendrycks_math")
    row = {
        "problem": "What is 8 + 8?",
        "level": "Level 1",
        "type": "Algebra",
        "solution": "We get $\\boxed{16}$.",
    }
    return module.HendrycksMathDataset(_hf_dict=_hf(train=row, test=row)), row


def _theoremqa():
    module = importlib.import_module("sieval.datasets.theoremqa")
    row = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    return module.TheoremQADataset(_hf_dict=_hf(test=row)), row


def _gpqa():
    module = importlib.import_module("sieval.datasets.gpqa_diamond")
    row = {
        "Question": "What is 2+2?",
        "Correct Answer": "4",
        "Incorrect Answer 1": "3",
        "Incorrect Answer 2": "5",
        "Incorrect Answer 3": "6",
    }
    return module.GPQADiamondDataset(_hf_dict=_hf(test=row)), row


def _mmlu():
    module = importlib.import_module("sieval.datasets.mmlu")
    row = {
        "question": "What is 2+2?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
        "subject": "abstract_algebra",
    }
    return module.MMLUDataset(_hf_dict=_hf(test=row)), row


def _mmlu_pro():
    module = importlib.import_module("sieval.datasets.mmlu_pro")
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
    return module.MMLUProDataset(_hf_dict=_hf(test=row)), row


def _openbookqa():
    module = importlib.import_module("sieval.datasets.openbookqa")
    row = {
        "id": "q1",
        "question_stem": "What is 2+2?",
        "choices": {"text": ["3", "4", "5", "6"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
    }
    return module.OpenBookQADataset(_hf_dict=_hf(train=row, test=row)), row


def _task_cls(module: str, name: str):
    return getattr(importlib.import_module(module), name)


class _Case:
    """One task, plus how to say a right and a wrong answer to it."""

    def __init__(
        self,
        name: str,
        module: str,
        cls_name: str,
        dataset,
        model_cls,
        *,
        samples: bool,
        denominator: str,
        fixed: tuple[str, str] | None = None,
        letters: str | None = None,
        kwargs: dict | None = None,
    ):
        self.name = name
        self._module, self._cls_name = module, cls_name
        self._dataset, self._model_cls = dataset, model_cls
        #: Does the task take its own `k` / `n`? Wave 2 yes, wave 3 no.
        self.samples = samples
        self.denominator = denominator
        #: A fixed (right, wrong) pair, or a letter template. GPQA shuffles its
        #: options behind a seeded RNG, so its gold is read off the prompt.
        self._fixed, self._letters = fixed, letters
        self._kwargs = kwargs or {}

    def build(self, texts, *, k=None, n=None):
        dataset, raw = self._dataset()
        model = self._model_cls(texts)
        extra = dict(self._kwargs)
        if self.samples and k is not None:
            extra |= {"k": k, "n": n}
        return _task_cls(self._module, self._cls_name)(dataset, model, **extra), raw

    async def responses(self):
        """(prompt record, raw sample, a right response, a wrong one)."""
        task, raw = self.build(["x"])
        pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
        if self._fixed is not None:
            return pre, raw, *self._fixed
        assert self._letters is not None, "a case needs either `fixed` or `letters`"
        gold = pre.get("reference") or "B"
        wrong = next(letter for letter in "ABCD" if letter != gold)
        return pre, raw, self._letters.format(gold), self._letters.format(wrong)


CASES = [
    _Case(
        "gsm8k_0shot_gen",
        "sieval.tasks.gsm8k_0shot_gen",
        "GSM8KZeroShotGenTask",
        _gsm8k,
        _StubChatModel,
        samples=True,
        denominator="requested",
        fixed=("The answer is $\\boxed{42}$.", "The answer is $\\boxed{7}$."),
    ),
    _Case(
        "gsm8k_kshot_base_gen",
        "sieval.tasks.gsm8k_kshot_base_gen",
        "GSM8KFewShotBaseGenTask",
        _gsm8k,
        _StubGenModel,
        samples=True,
        denominator="judged",
        # lm-eval's strict match is the `#### ` delimiter, not prose.
        fixed=("Work it out.\n#### 42", "Work it out.\n#### 7"),
        kwargs={"n_shot": 0},
    ),
    _Case(
        "hendrycks_math_kshot_base_gen",
        "sieval.tasks.hendrycks_math_kshot_base_gen",
        "HendrycksMathFewShotBaseGenTask",
        _hendrycks,
        _StubGenModel,
        samples=True,
        denominator="requested",
        fixed=("So $\\boxed{16}$.", "So $\\boxed{9}$."),
    ),
    _Case(
        "theoremqa_kshot_base_gen",
        "sieval.tasks.theoremqa_kshot_base_gen",
        "TheoremQAKShotBaseGenTask",
        _theoremqa,
        _StubGenModel,
        samples=True,
        denominator="judged",
        fixed=("The answer is 4", "The answer is 9"),
        kwargs={"n_shot": 0},
    ),
    _Case(
        "gpqa_diamond_0shot_gen",
        "sieval.tasks.gpqa_diamond_0shot_gen",
        "GPQADiamondZeroShotGenTask",
        _gpqa,
        _StubChatModel,
        samples=False,
        denominator="judged",
        letters="Answer: {}",
    ),
    _Case(
        "mmlu_0shot_gen",
        "sieval.tasks.mmlu_0shot_gen",
        "MMLUZeroShotGenTask",
        _mmlu,
        _StubChatModel,
        samples=False,
        denominator="judged",
        letters="Answer: {}",
    ),
    _Case(
        "mmlu_pro_0shot_gen",
        "sieval.tasks.mmlu_pro_0shot_gen",
        "MMLUProZeroShotGenTask",
        _mmlu_pro,
        _StubChatModel,
        samples=False,
        denominator="judged",
        letters="Answer: {}",
    ),
    _Case(
        "openbookqa_kshot_gen",
        "sieval.tasks.openbookqa_kshot_gen",
        "OpenBookQAFewShotGenTask",
        _openbookqa,
        _StubChatModel,
        samples=False,
        denominator="judged",
        letters="{}",
    ),
]
IDS = [c.name for c in CASES]
SAMPLING = [c for c in CASES if c.samples]
SAMPLING_IDS = [c.name for c in SAMPLING]
SINGLE_DRAW = [c for c in CASES if not c.samples]
SINGLE_DRAW_IDS = [c.name for c in SINGLE_DRAW]


async def _score(case, texts, *, k=None, n=None):
    """Run one sample end to end and return (judgement, report)."""
    pre, raw, _, _ = await case.responses()
    task, _ = case.build(texts, k=k, n=n)
    ctx = TaskContext(sample_id=0, raw_sample=raw, preprocess_result=pre)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    ctx = TaskContext(
        sample_id=0, raw_sample=raw, preprocess_result=pre, postprocess_result=post
    )
    _, judgement = await task.feedback(post, ctx)
    final = TaskContext(
        sample_id=0,
        raw_sample=raw,
        preprocess_result=pre,
        postprocess_result=post,
        feedback_result=judgement,
    )
    return post, judgement, await task.report([final], [])


# --------------------------------------------------------------------------- #
# The budget: taken, or refused
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", SAMPLING, ids=SAMPLING_IDS)
def test_k_greater_than_n_is_rejected(case):
    # Without the guard this constructs fine and pass@k comes out 0.0.
    with pytest.raises(ValueError, match=r"pass@2"):
        case.build(["x"], k=2, n=1)


@pytest.mark.parametrize("case", SAMPLING, ids=SAMPLING_IDS)
def test_k_equal_to_n_is_accepted(case):
    assert case.build(["x"], k=DRAW, n=DRAW)[0] is not None


@pytest.mark.parametrize("case", SINGLE_DRAW, ids=SINGLE_DRAW_IDS)
def test_a_sampling_knob_is_refused(case):
    # A `k` here would promise a pass@k the headline does not compute.
    dataset, _ = case._dataset()
    with pytest.raises(TypeError):
        _task_cls(case._module, case._cls_name)(
            dataset, _StubChatModel(["x"]), k=2, n=2
        )


@pytest.mark.parametrize("case", SAMPLING, ids=SAMPLING_IDS)
@pytest.mark.anyio
async def test_infer_forwards_the_sampling_budget(case):
    # Or a model-level `n` gets through and its draws are discarded.
    pre, raw, _, _ = await case.responses()
    task, _ = case.build(["x"], k=DRAW, n=DRAW)
    await task.infer(pre, TaskContext(sample_id=0, raw_sample=raw))
    assert task.model.last_kwargs.get("n") == DRAW


# --------------------------------------------------------------------------- #
# Clause 1: no draw is discarded
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.anyio
async def test_every_rollout_is_extracted_and_graded(case):
    # `inf.texts[0]` or `rollouts[0]` alone leaves this at a length of 1.
    _, _, right, wrong = await case.responses()
    budget = {"k": DRAW, "n": DRAW} if case.samples else {}
    post, judgement, _ = await _score(case, [right, wrong, right, wrong], **budget)

    assert len(post["rollouts"]) == DRAW, "a paid-for draw was discarded"
    assert len(judgement["rollouts"]) == DRAW
    assert [r["correct"] for r in judgement["rollouts"]] == [True, False, True, False]


# --------------------------------------------------------------------------- #
# Clause 2: the headline is the first draw
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.anyio
async def test_headline_scores_the_first_draw_only(case):
    # Three of four correct, but the FIRST is wrong: the headline reads 0.
    _, _, right, wrong = await case.responses()
    budget = {"k": DRAW, "n": DRAW} if case.samples else {}
    _, judgement, report = await _score(case, [wrong, right, right, right], **budget)

    assert [r["correct"] for r in judgement["rollouts"]] == [False, True, True, True]
    assert report["score"] == pytest.approx(0.0)
    assert report[report["score_key"]] == pytest.approx(0.0)


@pytest.mark.parametrize("case", SAMPLING, ids=SAMPLING_IDS)
@pytest.mark.anyio
async def test_sampling_metrics_describe_the_whole_draw(case):
    # Same run, where the headline reads 0: `pass@1` is c/n over all four.
    _, _, right, wrong = await case.responses()
    _, _, report = await _score(case, [wrong, right, right, right], k=DRAW, n=DRAW)

    assert report["pass@1"] == pytest.approx(75.0)
    assert report["avg@n"] == pytest.approx(75.0)
    assert report["pass@k"] == pytest.approx(100.0)
    assert report["maj@k"] == pytest.approx(100.0)
    assert (report["n"], report["k"], report["n_short"]) == (4.0, 4.0, 0.0)


@pytest.mark.parametrize("case", SAMPLING, ids=SAMPLING_IDS)
@pytest.mark.anyio
async def test_no_sampling_block_at_the_default_budget(case):
    # At n=1 a stored row keeps every column it had.
    task, _ = case.build(["x"], k=1, n=1)
    report = await task.report([], [])
    assert report["score_key"] in report
    assert not {"avg@n", "pass@k", "maj@k", "n", "k", "n_short"} & set(report)


@pytest.mark.parametrize("case", SINGLE_DRAW, ids=SINGLE_DRAW_IDS)
@pytest.mark.anyio
async def test_no_sampling_metrics_without_a_budget(case):
    # Four rollouts graded, none scored into the headline.
    _, _, right, wrong = await case.responses()
    _, _, report = await _score(case, [wrong, right, right, right])
    assert not {"pass@1", "avg@n", "pass@k", "maj@k", "n", "k"} & set(report)


# --------------------------------------------------------------------------- #
# The population each headline is over
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.anyio
async def test_report_declares_which_population_it_averages_over(case):
    # Two values across these eight -- why it is recorded, not unified.
    task, _ = case.build(["x"], k=1, n=1)
    report = await task.report([], [])
    assert report["denominator_policy"] == case.denominator
