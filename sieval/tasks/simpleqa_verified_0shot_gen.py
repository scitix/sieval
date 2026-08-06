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
``temperature: 0``; each sample's parsed grade and the grader's full model
output (reply, reasoning, usage, finish reason, model id) are persisted under
the judgement record's ``extra`` — see :meth:`feedback` for why the raw output
is kept whole.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.simpleqa_verified import (
    GRADER_TEMPLATE,
    aggregate_metrics,
    parse_grade,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RolloutJudgement,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.utils.serialization import obj_to_dict
from sieval.datasets import SimpleQAVerifiedDatasetSample


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
            "parsed grade and the grader's full ModelOutput (grader_output: "
            "reply, reasoning, usage, finish_reasons, model id) are persisted "
            "under the judgement record's `extra` — the reply being the only "
            "evidence of a verdict a re-grade need not reproduce, and "
            "(parse_grade defaults to NOT_ATTEMPTED) the only way to tell format "
            "drift from a real abstention; finish_reasons separates a reasoning "
            "autorater that spent its budget thinking from an empty API "
            "response. "
            "VALIDATION: google/gemma-4-31B-it scored F1 9.95 (n=1000, grader "
            "openai/gpt-4.1 via OpenRouter), within the official 10.7±2.1 band. "
            "Official numbers (Gemini 2.5 Pro 55.6; the band above) are from the "
            "DeepMind SimpleQA Verified leaderboard: "
            "https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified"
        ),
    ),
)
class SimpleQAVerifiedZeroShotGenTask(
    Task[
        SimpleQAVerifiedDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
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
        return build_prompt_record(
            [{"role": "user", "content": raw["problem"]}],
            reference=raw["answer"],
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Free-form factuality answer: the response is the answer, no extraction.
        # A blank answer normalizes to None so `extracted` stays a real signal;
        # the grader still sees "" and will rate it NOT_ATTEMPTED.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Grade every rollout with the autorater, recording its full output.

        The grader is a model, so its output is persisted the way any model
        output is: ``extra["grader_output"]`` is the grader's ``ModelOutput``
        flattened to a plain dict — reply text, reasoning, usage, finish reason,
        request params, model id, all of it. Nothing is hand-picked, so no field
        is silently dropped and a future ``ModelOutput`` field is captured for
        free. ``extra["grade"]`` is the parsed verdict on top of that raw output
        — task logic, not model output, which is why it is a separate key.

        Keeping the reply matters because grading is not artifact-reproducible: a
        grader model version is not pinnable like a Hub revision, so the reply is
        the only durable evidence of a verdict a re-grade need not reproduce. And
        ``parse_grade`` defaults a non-matching reply to NOT_ATTEMPTED, which the
        F1 weights very differently from an incorrect answer — the reply is what
        separates format drift from a real abstention. A reasoning autorater can
        spend its whole budget thinking and return empty text; ``finish_reasons``
        (in the same output) is what separates that from an empty API response.

        The flattening uses ``add_type=False`` on purpose: it keeps the grader
        output a plain dict, so the judgement record stays uniformly plain-dict
        rather than nesting a typed ``@sieval_record`` object.
        """
        raw = ctx.raw_sample
        question = raw["problem"]
        gold = raw["answer"]

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            predicted = rollout.get("prediction") or ""
            prompt = GRADER_TEMPLATE.format(
                question=question,
                target=gold,
                predicted_answer=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            grade = parse_grade(reply)
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    # Three-way grade collapses to the headline bool; `grade` below
                    # keeps the full verdict, and report() aggregates from that.
                    grade == "CORRECT",
                    extra={
                        "grade": grade,
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )
        return True, build_judgement_record(gold, rollouts)

    @override
    async def report(self, finals, fails):
        graded = [
            r["extra"]["grade"]
            for f in finals
            for r in (f.feedback_result or {}).get("rollouts", [])
        ]
        # Pipeline failures (exhausted retries) never produced a gradeable
        # answer; count each failed sample's requested attempts as
        # NOT_ATTEMPTED so the F1 spans the full requested set — matching the
        # official full-set metric and the gen-task family (imo_answer_bench,
        # gsm8k), rather than only the successfully-graded subset.
        grades = graded + ["NOT_ATTEMPTED"] * (self._n * len(fails))
        m = aggregate_metrics(grades)
        return {
            "score": m["f1"] * 100,
            "f1": m["f1"] * 100,
            "accuracy_given_attempted": m["accuracy_given_attempted"] * 100,
            "correct": m["is_correct"] * 100,
            "incorrect": m["is_incorrect"] * 100,
            "not_attempted": m["is_not_attempted"] * 100,
            "n_graded": len(graded),
            "fails": len(fails),
        }
