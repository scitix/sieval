"""
CMMLU few-shot base-model generative task.

Mirrors the official CMMLU ``qwen2.py`` *base* path (``eval``, not
``eval_instruct``): non-CoT prompt, same-subject dev shots, and next-token
A/B/C/D choice scoring. Here scoring reads one next token from ``top_logprobs``
and argmaxes over A/B/C/D; the official ``eval`` argmaxes raw last-token logits
over the same token IDs. The two agree whenever all four option tokens are in
the requested top-k (softmax is monotonic) — the only divergence is an option
token outside top-k, scored ``-inf`` here. In the validated Qwen2.5-72B 5-shot
run with ``logprobs=100``, all 11,582 samples had finite A/B/C/D scores.

Target: 89.5 — Qwen2.5-72B *Base*, 5-shot, from the DeepSeek-V3 report's
base-model table; benchmarked against this task's ``overall`` (subject-level
macro-average). Treated as a cross-check only: DeepSeek reports a
"perplexity of each option" method with "length normalization", but its report
does not state whether the scored option is the letter or the answer text, and
its appendix template lists ``OPTIONS: A/B/C/D`` (single-letter, where length
normalization is a no-op) — so its exact mechanic is underspecified and not the
method this task reproduces. The reproduced method is the official CMMLU
``qwen2.py`` ``eval`` path above. Note the official CMMLU leaderboard's
"Qwen2.5-72B" (85.67) is the *Instruct* model, not this base-model target.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

from typing import Any, TypedDict, override

from sieval.core.datasets import Dataset
from sieval.core.models import Model, ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import CMMLUDatasetSample

CHOICES = ("A", "B", "C", "D")
DEFAULT_N_SHOT = 5
DEFAULT_LOGPROBS = 20

CMMLU_SUBJECT_DISPLAY_NAMES = {
    "agronomy": "农学",
    "anatomy": "解剖学",
    "ancient_chinese": "古汉语",
    "arts": "艺术学",
    "astronomy": "天文学",
    "business_ethics": "商业伦理",
    "chinese_civil_service_exam": "中国公务员考试",
    "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化",
    "chinese_foreign_policy": "中国外交政策",
    "chinese_history": "中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science": "大学精算学",
    "college_education": "大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics": "大学医学统计",
    "college_medicine": "大学医学",
    "computer_science": "计算机科学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese": "小学语文",
    "elementary_commonsense": "小学常识",
    "elementary_information_and_technology": "小学信息技术",
    "elementary_mathematics": "初等数学",
    "ethnology": "民族学",
    "food_science": "食品科学",
    "genetics": "遗传学",
    "global_facts": "全球事实",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "high_school_geography": "高中地理",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学",
    "high_school_politics": "高中政治",
    "human_sexuality": "人类性行为",
    "international_law": "国际法学",
    "journalism": "新闻学",
    "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础",
    "logical": "逻辑学",
    "machine_learning": "机器学习",
    "management": "管理学",
    "marketing": "市场营销",
    "marxist_theory": "马克思主义理论",
    "modern_chinese": "现代汉语",
    "nutrition": "营养学",
    "philosophy": "哲学",
    "professional_accounting": "专业会计",
    "professional_law": "专业法学",
    "professional_medicine": "专业医学",
    "professional_psychology": "专业心理学",
    "public_relations": "公共关系",
    "security_study": "安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history": "世界历史",
    "world_religions": "世界宗教",
}

CMMLU_SUBCATEGORIES = {
    "agronomy": ("other",),
    "anatomy": ("biology",),
    "ancient_chinese": ("linguistics", "china specific"),
    "arts": ("arts",),
    "astronomy": ("physics",),
    "business_ethics": ("business",),
    "chinese_civil_service_exam": ("politics", "china specific"),
    "chinese_driving_rule": ("other", "china specific"),
    "chinese_food_culture": ("culture", "china specific"),
    "chinese_foreign_policy": ("politics", "china specific"),
    "chinese_history": ("history", "china specific"),
    "chinese_literature": ("literature", "china specific"),
    "chinese_teacher_qualification": ("education", "china specific"),
    "clinical_knowledge": ("other",),
    "college_actuarial_science": ("math",),
    "college_education": ("education",),
    "college_engineering_hydrology": ("engineering",),
    "college_law": ("law",),
    "college_mathematics": ("math",),
    "college_medical_statistics": ("statistics",),
    "college_medicine": ("other",),
    "computer_science": ("computer science",),
    "computer_security": ("other",),
    "conceptual_physics": ("physics",),
    "construction_project_management": ("other", "china specific"),
    "economics": ("economics",),
    "education": ("education",),
    "electrical_engineering": ("engineering",),
    "elementary_chinese": ("linguistics", "china specific"),
    "elementary_commonsense": ("other", "china specific"),
    "elementary_information_and_technology": ("other",),
    "elementary_mathematics": ("math",),
    "ethnology": ("culture", "china specific"),
    "food_science": ("other",),
    "genetics": ("biology",),
    "global_facts": ("global",),
    "high_school_biology": ("biology",),
    "high_school_chemistry": ("chemistry",),
    "high_school_geography": ("geography",),
    "high_school_mathematics": ("math",),
    "high_school_physics": ("physics",),
    "high_school_politics": ("politics", "china specific"),
    "human_sexuality": ("other",),
    "international_law": ("law",),
    "journalism": ("sociology",),
    "jurisprudence": ("law",),
    "legal_and_moral_basis": ("other",),
    "logical": ("philosophy",),
    "machine_learning": ("computer science",),
    "management": ("business",),
    "marketing": ("business",),
    "marxist_theory": ("philosophy",),
    "modern_chinese": ("linguistics", "china specific"),
    "nutrition": ("other",),
    "philosophy": ("philosophy",),
    "professional_accounting": ("business",),
    "professional_law": ("law",),
    "professional_medicine": ("other",),
    "professional_psychology": ("psychology",),
    "public_relations": ("politics",),
    "security_study": ("politics",),
    "sociology": ("culture",),
    "sports_science": ("other",),
    "traditional_chinese_medicine": ("other", "china specific"),
    "virology": ("biology",),
    "world_history": ("history",),
    "world_religions": ("global",),
}

CMMLU_CATEGORIES = {
    "STEM": (
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
        "statistics",
    ),
    "Humanities": ("history", "philosophy", "law", "arts", "literature", "global"),
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

CMMLU_CATEGORY_SUBJECTS = {
    category: tuple(
        subject
        for subject, subcategories in CMMLU_SUBCATEGORIES.items()
        if any(subcategory in category_subcategories for subcategory in subcategories)
    )
    for category, category_subcategories in CMMLU_CATEGORIES.items()
}


class Feedback(TypedDict):
    correct: bool
    pred: str
    answer: str


def _choice_scores_from_top_logprobs(
    top_logprobs: list[dict[str, float]] | None,
) -> tuple[dict[str, float], bool]:
    scores = {label: float("-inf") for label in CHOICES}
    if not top_logprobs:
        return scores, False

    found = False
    for token, logprob in top_logprobs[0].items():
        label = token.strip()
        if label in scores:
            scores[label] = max(scores[label], logprob)
            found = True
    return scores, found


@sieval_task(
    name="cmmlu_kshot_base_gen",
    display_name="CMMLU (few-shot, base logprob)",
    description="CMMLU few-shot MCQ with same-subject dev examples and macro scoring.",
    eval_mode=EvalMode.PPL,
    n_shot=DEFAULT_N_SHOT,
    tags=("chinese", "multiple-choice", "base-model"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="cmmlu",
        url="https://github.com/haonan-li/CMMLU/blob/d6e7b716d8ac694f38969a6c0407437d1fded799/src/qwen2.py",
        notes=(
            "Mirrors the official qwen2.py base path (eval, not eval_instruct): "
            "non-CoT CMMLU prompt, same-subject dev shots, one-call next-token "
            "A/B/C/D scoring, and subject-level macro report. Runtime k is "
            "configurable. Uses API top_logprobs as an OpenAI-compatible "
            "substitute for the official raw-logits choice argmax (equivalent "
            "while all four option tokens are in top-k); the validated "
            "Qwen2.5-72B 5-shot run with logprobs=100 had no missing A/B/C/D "
            "top-k entries across 11,582 samples. Runner failures are reported "
            "separately and excluded from the score denominator. Target: 89.5 "
            "(Qwen2.5-72B Base, 5-shot, DeepSeek-V3 base-model table) as a "
            "cross-check only — DeepSeek's perplexity method is underspecified "
            "(letter vs. text unstated; appendix template lists single-letter "
            "OPTIONS) and is not the reproduced method. The official CMMLU "
            "leaderboard's 85.67 is the Instruct model, not this base target."
        ),
    ),
)
class CMMLUFewShotBaseGenTask(
    Task[
        CMMLUDatasetSample,
        str,
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset: Dataset[CMMLUDatasetSample],
        model: Model[Any],
        name: str | None = None,
        *,
        k: int = DEFAULT_N_SHOT,
        logprobs: int = DEFAULT_LOGPROBS,
        fewshot_split: str = "dev",
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if logprobs < 1:
            raise ValueError(f"logprobs must be >= 1, got {logprobs}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._logprobs = max(logprobs, len(CHOICES))
        self._fewshot_split = fewshot_split
        self._few_shot_by_subject: dict[str, list[CMMLUDatasetSample]] = {}
        self._few_shot_prompt_by_subject: dict[str, str] = {}

    @override
    async def setup(self) -> None:
        self._ensure_few_shot_pool()

    def _ensure_few_shot_pool(self) -> None:
        if self._few_shot_by_subject or self._k == 0:
            return
        split = self.dataset.dataset_dict.get(self._fewshot_split)
        if split is None:
            raise ValueError(
                "CMMLU few-shot base generative task requires a "
                f"{self._fewshot_split!r} split for few-shot examples."
            )
        for sample in split:
            subject = str(sample.get("subject") or "miscellaneous")
            self._few_shot_by_subject.setdefault(subject, []).append(sample)

    def _select_examples(self, subject: str) -> list[CMMLUDatasetSample]:
        if self._k == 0:
            return []
        self._ensure_few_shot_pool()
        return list(self._few_shot_by_subject.get(subject, []))[: self._k]

    def _format_example(
        self, sample: CMMLUDatasetSample, *, include_answer: bool = True
    ) -> str:
        prompt = f"题目：{sample['question']}"
        prompt += f"\nA. {sample['A']}"
        prompt += f"\nB. {sample['B']}"
        prompt += f"\nC. {sample['C']}"
        prompt += f"\nD. {sample['D']}"
        prompt += "\n答案是："
        if include_answer:
            prompt += f"{sample['answer']}\n\n"
        return prompt

    def _subject_display_name(self, subject: str) -> str:
        return CMMLU_SUBJECT_DISPLAY_NAMES.get(subject, subject or "通识")

    def _build_few_shot_prompt(self, subject: str) -> str:
        cached = self._few_shot_prompt_by_subject.get(subject)
        if cached is not None:
            return cached

        subject_zh = self._subject_display_name(subject)
        prompt = f"以下是关于{subject_zh}的单项选择题，请直接给出正确答案的选项。\n\n"
        for example in self._select_examples(subject):
            prompt += self._format_example(example, include_answer=True)
        self._few_shot_prompt_by_subject[subject] = prompt
        return prompt

    def _build_prompt(self, sample: CMMLUDatasetSample) -> str:
        subject = sample.get("subject") or "miscellaneous"
        question = self._format_example(sample, include_answer=False)
        return self._build_few_shot_prompt(subject) + question

    @override
    async def preprocess(self, raw, ctx):
        return self._build_prompt(raw)

    @override
    async def infer(self, pre, ctx):
        return await self.model.alogprobs(
            pre,
            max_tokens=1,
            logprobs=self._logprobs,
            echo=False,
        )

    @override
    async def postprocess(self, inf, ctx):
        scores, found = _choice_scores_from_top_logprobs(inf.top_logprobs)
        if not found:
            raise RuntimeError(
                "CMMLU top_logprobs did not contain any A/B/C/D option tokens."
            )
        return max(scores.items(), key=lambda item: item[1])[0]

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        if raw is None:
            return True, {"correct": False, "pred": post, "answer": ""}
        answer = raw["answer"]
        return True, {"correct": post == answer, "pred": post, "answer": answer}

    @override
    async def report(self, finals, fails):
        subject_corrects: dict[str, int] = {}
        subject_totals: dict[str, int] = {}

        for ctx in finals:
            raw = ctx.raw_sample
            if raw is None:
                continue
            subject = raw.get("subject") or "miscellaneous"
            subject_totals[subject] = subject_totals.get(subject, 0) + 1
            subject_corrects.setdefault(subject, 0)
            if ctx.feedback_result and ctx.feedback_result["correct"]:
                subject_corrects[subject] += 1

        if not subject_totals:
            return {"score": 0.0, "fails": float(len(fails)), "pass@1": 0.0}

        subject_acc = {
            subject: subject_corrects.get(subject, 0) * 100 / total
            for subject, total in subject_totals.items()
            if total
        }
        overall = sum(subject_acc.values()) / len(subject_acc)

        metrics: dict[str, float] = {
            "score": overall,
            "fails": float(len(fails)),
            "pass@1": overall,
            "overall": overall,
        }

        for category, subjects in CMMLU_CATEGORY_SUBJECTS.items():
            available_scores = [
                subject_acc[subject] for subject in subjects if subject in subject_acc
            ]
            key = category.lower().replace(" ", "_")
            metrics[key] = (
                sum(available_scores) / len(available_scores)
                if available_scores
                else 0.0
            )

        return metrics
