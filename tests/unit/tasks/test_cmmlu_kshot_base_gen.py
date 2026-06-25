"""
Unit tests for the CMMLU few-shot base-model task.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.cmmlu import CMMLUDataset, CMMLUDatasetSample
from sieval.tasks.cmmlu_kshot_base_gen import (
    CMMLU_CATEGORIES,
    CMMLU_SUBCATEGORIES,
    CMMLU_SUBJECT_DISPLAY_NAMES,
    CMMLUFewShotBaseGenTask,
)


class _DummyGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.logprob_prompts: list[str] = []

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
        _ = (max_tokens, logprobs, temperature, kwargs)
        self.logprob_prompts.append(prompt)
        assert echo is False
        return ModelOutput(
            model=self.meta(),
            texts=["B"],
            logprobs_tokens=["B"],
            logprobs=[-0.1],
            top_logprobs=[{"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0}],
        )


def _sample(
    question: str,
    answer: str = "B",
    subject: str = "anatomy",
) -> CMMLUDatasetSample:
    return {
        "question": question,
        "A": "选项甲",
        "B": "选项乙",
        "C": "选项丙",
        "D": "选项丁",
        "answer": answer,
        "subject": subject,
    }


def _dataset() -> CMMLUDataset:
    return CMMLUDataset(
        _hf_dict=HFDatasetDict(
            {
                "dev": HFDataset.from_list(
                    [
                        dict(_sample("示例一")),
                        dict(_sample("示例二")),
                        dict(_sample("逻辑示例", subject="logical")),
                    ]
                ),
                "test": HFDataset.from_list([dict(_sample("测试题"))]),
            }
        )
    )


def _dataset_without_dev() -> CMMLUDataset:
    return CMMLUDataset(
        _hf_dict=HFDatasetDict(
            {
                "test": HFDataset.from_list([dict(_sample("测试题"))]),
            }
        )
    )


@pytest.mark.anyio
async def test_k_controls_same_subject_few_shot_prompt():
    task = CMMLUFewShotBaseGenTask(_dataset(), _DummyGenModel(), k=1)
    await task.setup()

    prompt = task._build_prompt(_sample("测试题"))

    assert "示例一" in prompt
    assert "示例二" not in prompt
    assert "逻辑示例" not in prompt


@pytest.mark.anyio
async def test_zero_shot_omits_few_shot_examples():
    task = CMMLUFewShotBaseGenTask(_dataset(), _DummyGenModel(), k=0)
    await task.setup()

    prompt = task._build_prompt(_sample("测试题"))

    assert "示例一" not in prompt
    assert "测试题" in prompt


@pytest.mark.anyio
async def test_k_requires_few_shot_split():
    task = CMMLUFewShotBaseGenTask(_dataset_without_dev(), _DummyGenModel(), k=1)

    with pytest.raises(ValueError, match="requires a 'dev' split"):
        await task.setup()


@pytest.mark.anyio
async def test_few_shot_prompt_is_cached_per_subject():
    task = CMMLUFewShotBaseGenTask(_dataset(), _DummyGenModel(), k=2)
    await task.setup()

    first = task._build_prompt(_sample("测试题"))
    task._few_shot_by_subject["anatomy"].append(_sample("迟到示例"))
    second = task._build_prompt(_sample("第二题"))

    assert "示例一" in first
    assert "示例二" in first
    assert "迟到示例" not in second
    assert "第二题" in second


@pytest.mark.anyio
async def test_infer_postprocess_feedback_and_report():
    raw = _sample("测试题")
    model = _DummyGenModel()
    task = CMMLUFewShotBaseGenTask(_dataset(), model, k=0)
    ctx = TaskContext(sample_id=0, raw_sample=raw)

    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=raw))
    finalize, feedback = await task.feedback(
        post,
        TaskContext(sample_id=0, raw_sample=raw),
    )
    report = await task.report(
        [
            TaskContext(
                sample_id=0,
                raw_sample=raw,
                infer_result=inf,
                feedback_result=feedback,
            )
        ],
        [],
    )

    assert finalize is True
    assert post == "B"
    assert feedback["correct"] is True
    assert report["score"] == 100.0
    assert len(model.logprob_prompts) == 1


@pytest.mark.anyio
async def test_postprocess_raises_when_option_token_missing():
    # Only A/B/C present in top-k (D dropped) → must fail loudly, not guess.
    task = CMMLUFewShotBaseGenTask(_dataset(), _DummyGenModel(), k=0)
    inf = ModelOutput(
        model=_DummyGenModel().meta(),
        texts=["A"],
        top_logprobs=[{"A": -0.1, "B": -1.0, "C": -2.0}],
    )

    with pytest.raises(RuntimeError, match=r"missing option token.*'D'"):
        await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample("题")))


@pytest.mark.anyio
async def test_report_excludes_failures_from_subject_denominator():
    task = CMMLUFewShotBaseGenTask(_dataset(), _DummyGenModel(), k=0)
    correct_anatomy = _sample("解剖测试", answer="B", subject="anatomy")
    wrong_logical = _sample("逻辑测试", answer="A", subject="logical")
    failed_anatomy = _sample("失败测试", answer="A", subject="anatomy")

    report = await task.report(
        [
            TaskContext(
                sample_id=0,
                raw_sample=correct_anatomy,
                feedback_result={
                    "correct": True,
                    "pred": "B",
                    "answer": "B",
                },
            ),
            TaskContext(
                sample_id=1,
                raw_sample=wrong_logical,
                feedback_result={
                    "correct": False,
                    "pred": "B",
                    "answer": "A",
                },
            ),
        ],
        [
            TaskContext(sample_id=2, raw_sample=failed_anatomy).to_failed(
                None,
                "exception::RuntimeError",
                "boom",
            )
        ],
    )

    assert report["fails"] == 1.0
    assert report["overall"] == 50.0
    assert report["score"] == 50.0
    assert report["stem"] == 100.0
    assert report["humanities"] == 0.0


# ---------------------------------------------------------------------------
# Upstream-constant pinning — lock the prompt template and the
# subject/category taxonomy against silent edits. These mirror the official
# qwen2.py / categories.py at the pinned CMMLU SHA; a faithful-reproduction
# regression must fail loudly here rather than drift unnoticed.
# ---------------------------------------------------------------------------
def test_format_example_template_is_pinned():
    task = CMMLUFewShotBaseGenTask(_dataset(), _DummyGenModel(), k=0)
    sample = _sample("壁胸膜的分部不包括", answer="B")

    assert task._format_example(sample, include_answer=False) == (
        "题目：壁胸膜的分部不包括\nA. 选项甲\nB. 选项乙\nC. 选项丙\nD. 选项丁\n答案是："
    )
    assert task._format_example(sample, include_answer=True) == (
        "题目：壁胸膜的分部不包括\n"
        "A. 选项甲\nB. 选项乙\nC. 选项丙\nD. 选项丁\n"
        "答案是：B\n\n"
    )


@pytest.mark.anyio
async def test_few_shot_header_is_pinned():
    # Header template + the subject display-name lookup (anatomy → 解剖学).
    task = CMMLUFewShotBaseGenTask(_dataset(), _DummyGenModel(), k=1)
    await task.setup()

    prompt = task._build_few_shot_prompt("anatomy")

    assert prompt.startswith(
        "以下是关于解剖学的单项选择题，请直接给出正确答案的选项。\n\n"
    )


def test_category_partition_is_pinned():
    assert CMMLU_CATEGORIES == {
        "STEM": (
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
            "statistics",
        ),
        "Humanities": (
            "history",
            "philosophy",
            "law",
            "arts",
            "literature",
            "global",
        ),
        "Social Science": (
            "linguistics",
            "business",
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
            "education",
            "sociology",
        ),
        "Other": ("other",),
        "China specific": ("china specific",),
    }


def test_every_subject_is_classified_and_named():
    subjects = set(CMMLUDataset.SUBJECTS)

    assert len(CMMLUDataset.SUBJECTS) == 67
    assert set(CMMLU_SUBCATEGORIES) == subjects
    assert set(CMMLU_SUBJECT_DISPLAY_NAMES) == subjects


def test_every_subcategory_maps_to_a_category():
    # A subcategory used in CMMLU_SUBCATEGORIES but absent from every
    # CMMLU_CATEGORIES bucket would silently drop its subjects from all
    # category-level metrics — guard against that typo class.
    bucket_subcategories = {
        subcategory
        for subcategories in CMMLU_CATEGORIES.values()
        for subcategory in subcategories
    }
    used_subcategories = {
        subcategory
        for subcategories in CMMLU_SUBCATEGORIES.values()
        for subcategory in subcategories
    }

    assert used_subcategories <= bucket_subcategories
