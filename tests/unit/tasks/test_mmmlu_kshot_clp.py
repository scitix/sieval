"""Unit tests for the full MMMLU k-shot base-model CLP task.

AI-Generated Code - GPT-5-Codex (OpenAI)
"""

from collections import Counter

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext, TaskStageOutput
from sieval.datasets.mmmlu import MMMLUDataset, MMMLUDatasetSample
from sieval.tasks.mmmlu_kshot_clp import (
    Feedback,
    MMMLUKShotClpTask,
    OfficialScores,
)

_FinalCtx = TaskContext[
    MMMLUDatasetSample, str, TaskStageOutput[OfficialScores], str, Feedback
]


class _TopLogprobGenModel(GenModel):
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
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(
            model=self.meta(),
            texts=[" C"],
            finish_reasons=["length"],
            logprobs_tokens=[" C"],
            logprobs=[-0.1],
            top_logprobs=[
                {" A": -3.0, " B": -2.0, " C": -0.1, " D": -4.0},
            ],
        )


def _sample(
    index: int,
    *,
    locale: str = "zh_cn",
    locale_display_name: str = "Simplified Chinese",
    subject: str = "abstract_algebra",
    category: str = "stem",
    answer: str = "C",
) -> MMMLUDatasetSample:
    return {
        "Question": f"题目 {locale} {subject} {index}",
        "A": f"选项A{index}",
        "B": f"选项B{index}",
        "C": f"选项C{index}",
        "D": f"选项D{index}",
        "Answer": answer,
        "Subject": subject,
        "Category": category,
        "Locale": locale,
        "LocaleDisplayName": locale_display_name,
    }


def _dataset(samples: list[MMMLUDatasetSample]) -> MMMLUDataset:
    return MMMLUDataset(
        _hf_dict=HFDatasetDict(
            {"test": HFDataset.from_list([dict(s) for s in samples])}
        )
    )


def _task(
    samples: list[MMMLUDatasetSample], *, k: int = 2, **kwargs
) -> MMMLUKShotClpTask:
    return MMMLUKShotClpTask(
        _dataset(samples),
        _TopLogprobGenModel(model="mock-gen", api_key="fake"),
        k=k,
        **kwargs,
    )


@pytest.mark.anyio
async def test_setup_reserves_fewshot_per_locale_subject_and_excludes_from_test():
    samples = [
        *[_sample(i, locale="zh_cn", subject="abstract_algebra") for i in range(3)],
        *[
            _sample(
                i,
                locale="de_de",
                locale_display_name="German",
                subject="business_ethics",
                category="other",
            )
            for i in range(3)
        ],
    ]
    task = _task(samples, k=2)

    await task.setup()

    test_set = task.dataset.test_set
    assert test_set is not None
    assert [row["Question"] for row in test_set] == [
        "题目 zh_cn abstract_algebra 2",
        "题目 de_de business_ethics 2",
    ]


@pytest.mark.anyio
async def test_preprocess_uses_same_locale_subject_fewshot_examples():
    samples = [
        *[_sample(i, locale="zh_cn", subject="abstract_algebra") for i in range(3)],
        *[
            _sample(
                i,
                locale="de_de",
                locale_display_name="German",
                subject="abstract_algebra",
            )
            for i in range(3)
        ],
    ]
    task = _task(samples, k=2)
    await task.setup()

    prompt = await task.preprocess(
        _sample(2, locale="zh_cn", subject="abstract_algebra"),
        TaskContext(sample_id=0, raw_sample=_sample(2)),
    )

    assert prompt.startswith(
        "The following are multiple choice questions (with answers) about "
        "abstract algebra (Simplified Chinese).\n\n"
    )
    assert prompt.count("Answer: C") == 2
    assert "German" not in prompt
    assert prompt.endswith("D. 选项D2\nAnswer:")


@pytest.mark.anyio
async def test_infer_postprocess_and_feedback_use_top_logprobs():
    raw = _sample(2)
    task = _task([_sample(0), _sample(1), raw], k=2)
    ctx = TaskContext(sample_id=2, raw_sample=raw)

    inferred = await task.infer("prompt", ctx)
    post = await task.postprocess(inferred, ctx)
    finalize, feedback = await task.feedback(
        post,
        TaskContext(sample_id=2, raw_sample=raw, infer_result=inferred),
    )

    assert isinstance(inferred, TaskStageOutput)
    assert post == "C"
    assert finalize is True
    assert feedback["correct"] is True
    assert feedback["locale"] == "zh_cn"
    assert feedback["category"] == "stem"


@pytest.mark.anyio
async def test_report_returns_weighted_overall_locale_category_and_subject_scores():
    task = _task([_sample(i) for i in range(3)], k=2)
    finals: list[_FinalCtx] = [
        TaskContext(
            sample_id=0,
            raw_sample=_sample(0, locale="zh_cn", subject="abstract_algebra"),
            feedback_result={
                "correct": True,
                "pred": "C",
                "answer": "C",
                "subject": "abstract_algebra",
                "category": "stem",
                "locale": "zh_cn",
                "prob_A": 0.0,
                "prob_B": 0.0,
                "prob_C": 1.0,
                "prob_D": 0.0,
            },
        ),
        TaskContext(
            sample_id=1,
            raw_sample=_sample(1, locale="zh_cn", subject="abstract_algebra"),
            feedback_result={
                "correct": False,
                "pred": "A",
                "answer": "C",
                "subject": "abstract_algebra",
                "category": "stem",
                "locale": "zh_cn",
                "prob_A": 1.0,
                "prob_B": 0.0,
                "prob_C": 0.0,
                "prob_D": 0.0,
            },
        ),
        TaskContext(
            sample_id=2,
            raw_sample=_sample(
                2,
                locale="de_de",
                locale_display_name="German",
                subject="business_ethics",
                category="other",
            ),
            feedback_result={
                "correct": True,
                "pred": "C",
                "answer": "C",
                "subject": "business_ethics",
                "category": "other",
                "locale": "de_de",
                "prob_A": 0.0,
                "prob_B": 0.0,
                "prob_C": 1.0,
                "prob_D": 0.0,
            },
        ),
    ]
    fails: list[_FinalCtx] = [
        TaskContext(
            sample_id=3,
            raw_sample=_sample(
                3,
                locale="de_de",
                locale_display_name="German",
                subject="business_ethics",
                category="other",
            ),
        )
    ]

    report = await task.report(finals, fails)

    # The de_de fail (sample 3) is an infra failure: excluded from the
    # denominator and reported separately, so de_de scores 100.0 (1/1 final)
    # rather than 50.0 (1/2 with the fail counted wrong), and overall is 2/3.
    assert report["score"] == pytest.approx(200 / 3)
    assert report["score_mmmlu"] == pytest.approx(200 / 3)
    assert report["score_locale_zh_cn"] == 50.0
    assert report["score_locale_de_de"] == 100.0
    assert report["score_locale_zh_cn_category_stem"] == 50.0
    assert report["score_locale_de_de_category_other"] == 100.0
    assert report["score_locale_zh_cn_subject_abstract_algebra"] == 50.0
    assert report["score_locale_de_de_subject_business_ethics"] == 100.0
    assert report["fails"] == 1.0
    assert isinstance(report["fails"], float)
    assert "pass@1" not in report


@pytest.mark.anyio
async def test_test_split_fewshot_requires_held_out_examples_per_locale_subject():
    task = _task([_sample(0), _sample(1)], k=2)

    with pytest.raises(ValueError, match="requires at least 3 test examples"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_samples_deterministically_by_locale_subject():
    samples = [
        *[_sample(i, locale="zh_cn", subject="abstract_algebra") for i in range(4)],
        *[_sample(i, locale="zh_cn", subject="business_ethics") for i in range(4)],
        *[
            _sample(
                i,
                locale="de_de",
                locale_display_name="German",
                subject="abstract_algebra",
            )
            for i in range(4)
        ],
        *[
            _sample(
                i,
                locale="de_de",
                locale_display_name="German",
                subject="business_ethics",
            )
            for i in range(4)
        ],
    ]
    task = _task(
        samples,
        k=0,
        sample_fraction=0.5,
        sample_seed=42,
        sample_by="locale_subject",
    )
    repeat = _task(
        samples,
        k=0,
        sample_fraction=0.5,
        sample_seed=42,
        sample_by="locale_subject",
    )

    await task.setup()
    await repeat.setup()

    test_set = task.dataset.test_set
    repeat_test_set = repeat.dataset.test_set
    assert test_set is not None
    assert repeat_test_set is not None
    assert len(test_set) == 8
    assert [row["Question"] for row in test_set] == [
        row["Question"] for row in repeat_test_set
    ]
    counts = Counter((row["Locale"], row["Subject"]) for row in test_set)
    assert counts == {
        ("de_de", "abstract_algebra"): 2,
        ("de_de", "business_ethics"): 2,
        ("zh_cn", "abstract_algebra"): 2,
        ("zh_cn", "business_ethics"): 2,
    }


@pytest.mark.anyio
async def test_setup_sampling_is_idempotent_under_repeated_calls():
    samples = [
        *[_sample(i, locale="zh_cn", subject="abstract_algebra") for i in range(4)],
        *[_sample(i, locale="zh_cn", subject="business_ethics") for i in range(4)],
    ]
    task = _task(
        samples,
        k=0,
        sample_fraction=0.5,
        sample_seed=42,
        sample_by="locale_subject",
    )

    await task.setup()
    first_set = task.dataset.test_set
    assert first_set is not None
    first = [row["Question"] for row in first_set]
    # A second setup() must not sample a sample (0.5 of the already-sampled set).
    await task.setup()
    second_set = task.dataset.test_set
    assert second_set is not None
    second = [row["Question"] for row in second_set]

    assert len(first) == 4
    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_fraction": 0}, "sample_fraction"),
        ({"sample_fraction": 1.5}, "sample_fraction"),
        ({"sample_seed": "42"}, "sample_seed"),
        ({"sample_by": "subject"}, "sample_by"),
        ({"k": -1}, "k must be >= 0"),
        ({"logprobs": 0}, "logprobs must be >= 1"),
    ],
)
def test_rejects_invalid_construction_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _task([_sample(0), _sample(1), _sample(2)], **kwargs)
