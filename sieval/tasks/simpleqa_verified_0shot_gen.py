"""SimpleQA Verified — 0-shot generative, LLM-autorater graded.

Faithful generative port of SimpleQA Verified (Google DeepMind, Haas et al.,
2025, arXiv:2509.07968): the model answers a short factuality question, and a
separate **LLM autorater** grades the free-form answer against the gold as
CORRECT / INCORRECT / NOT_ATTEMPTED. The headline metric is the F1 (harmonic
mean of the overall correct-rate and the accuracy-given-attempted).

The grader is supplied via the ``grader`` task arg (a model-config dict, or a
pre-built Model, on its own ``api_base``/``api_key``); the official autorater is
gpt-4.1-2025-04-14. Unlike sieval's deterministic-grader tasks, correctness
depends on a real grader model whose version sieval cannot pin the way it pins a
Hub revision, so for reproducibility pin the grader model and set
``temperature: 0``; each sample's grade and the grader model id are persisted in
the feedback record.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.simpleqa_verified import (
    GRADER_TEMPLATE,
    aggregate_metrics,
    parse_grade,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import SimpleQAVerifiedDatasetSample


class GradeFeedback(TypedDict):
    grade: str  # "CORRECT" | "INCORRECT" | "NOT_ATTEMPTED"
    gold: str
    predicted: str
    grader_model: str


@sieval_task(
    name="simpleqa_verified_0shot_gen",
    display_name="SimpleQA Verified (0-shot, generative)",
    description="Short-form factuality; free-form answer graded by an LLM autorater.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "factuality", "open-ended"),
    model_type="chat",
    status="stable",
    reference_impl=ReferenceImpl(
        source="simpleqa-verified",
        url="https://arxiv.org/abs/2509.07968",
        notes=(
            "Generative port of SimpleQA Verified (Google DeepMind, arXiv:"
            "2509.07968) — a 1,000-prompt curated subset of OpenAI SimpleQA. "
            "The autorater prompt is the paper's updated prompt (Appendix A); "
            "grade parsing (A/B/C -> CORRECT/INCORRECT/NOT_ATTEMPTED, default C) "
            "and the F1 aggregation mirror openai/simple-evals@5e623c2b "
            "simpleqa_eval.py. Headline metric = F1 (harmonic mean of overall-"
            "correct and correct-given-attempted; official Gemini 2.5 Pro = "
            "55.6). Grader is a REAL LLM (official autorater: gpt-4.1-2025-04-14) "
            "supplied via the `grader` task arg on its own api_base/api_key. "
            "REPRODUCIBILITY: unlike deterministic-grader tasks, scores depend "
            "on the grader endpoint's model version (not pinnable like a Hub "
            "revision) — pin the grader model + temperature=0; the per-sample "
            "grade and grader model id are persisted in the feedback record. "
            "VALIDATION: google/gemma-4-31B-it scored F1 9.95 (n=1000, grader "
            "openai/gpt-4.1 via OpenRouter), within the official 10.7±2.1 band."
        ),
    ),
)
class SimpleQAVerifiedZeroShotGenTask(
    Task[
        SimpleQAVerifiedDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        list[str],
        list[GradeFeedback],
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._build_grader(grader)

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (used by tests / advanced configs) or a
        model-config mapping (the YAML path, e.g.
        ``{model: gpt-4.1, api_base: ..., temperature: 0}``). Grading is
        mandatory — there is no deterministic fallback — so ``None`` raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "SimpleQA Verified requires an LLM grader. Pass `grader:` in the "
            "task args — a model-config dict such as "
            "{model: gpt-4.1, api_base: ..., api_key: ..., temperature: 0}."
        )

    @override
    async def preprocess(self, raw, ctx):
        return [{"role": "user", "content": raw["problem"]}]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        return list(inf.texts)

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        question = raw["problem"]
        gold = raw["answer"]
        grader_model = self._grader.meta()["model"]

        feedbacks: list[GradeFeedback] = []
        for predicted in post:
            prompt = GRADER_TEMPLATE.format(
                question=question,
                target=gold,
                predicted_answer=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            feedbacks.append(
                {
                    "grade": parse_grade(reply),
                    "gold": gold,
                    "predicted": predicted,
                    "grader_model": grader_model,
                }
            )
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        grades = [fb["grade"] for f in finals for fb in (f.feedback_result or [])]
        m = aggregate_metrics(grades)
        return {
            "score": m["f1"] * 100,
            "f1": m["f1"] * 100,
            "accuracy_given_attempted": m["accuracy_given_attempted"] * 100,
            "correct": m["is_correct"] * 100,
            "incorrect": m["is_incorrect"] * 100,
            "not_attempted": m["is_not_attempted"] * 100,
            "n_graded": len(grades),
            "fails": len(fails),
        }
