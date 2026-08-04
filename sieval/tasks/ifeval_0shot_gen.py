from typing import override

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.datasets import IFEvalDatasetSample

# Grading is per prompt-level ("did the response follow *every* constraint") and
# instruction-level ("what fraction of its constraints did it follow"), each under a
# strict and a loose reading. `strict` is the headline: the task's `score` is the
# strict prompt-level accuracy.
_GRADES = ("strict", "loose")


@sieval_task(
    name="ifeval_0shot_gen",
    display_name="IFEval (0-shot, generative)",
    description=(
        "Instruction-Following Eval — 541 prompts with verifiable constraints."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="ifeval",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="google-research/instruction_following_eval",
        url="https://github.com/google-research/google-research/blob/f97f6adab57bd3065b24169bcfc559dc34d0db84/instruction_following_eval/evaluation_lib.py",
        notes="evaluation_lib + instructions_registry vendored from google-research.",
    ),
)
class IFEvalZeroShotGenTask(
    Task[
        IFEvalDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float],
    ]
):
    def __init__(self, dataset, model, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [{"role": "user", "content": raw["prompt"]}],
            # The "ground truth" is the constraint set the response must satisfy.
            reference=list(raw["instruction_id_list"]),
            extra={
                "key": raw["key"],
                "kwargs": self._clean_kwargs(raw["kwargs"]),
            },
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended task: the response *is* the answer, so no extraction step.
        # A blank response still normalizes to None, so `extracted` stays a real
        # signal -- pre-protocol, an empty string here was what tripped the
        # empty-postprocess anomaly rule, and that coverage must not be lost.
        text = inf.texts[0]  # n=1, only one choice
        return build_prediction_record([text if text.strip() else None])

    @override
    async def feedback(self, post, ctx):
        # Graded here rather than in report() so every sample's verdict is on disk
        # and inspectable. Both graders are pure per-sample -- their only use of the
        # response map is `prompt_to_response[inp.prompt]` -- so scoring one sample
        # at a time is equivalent to upstream's whole-set call.
        from sieval.community.instruction_following_eval.evaluation_lib import (
            InputExample,
            test_instruction_following_loose,
            test_instruction_following_strict,
        )

        graders = {
            "strict": test_instruction_following_strict,
            "loose": test_instruction_following_loose,
        }

        raw = ctx.raw_sample
        prompt = raw["prompt"]
        instruction_ids = list(raw["instruction_id_list"])
        response = post["rollouts"][0]["prediction"] or ""
        inp = InputExample(
            key=raw["key"],
            instruction_id_list=instruction_ids,
            prompt=prompt,
            kwargs=self._clean_kwargs(raw["kwargs"]),
        )

        # Both readings are co-equal published metrics, so both go in `metrics`;
        # strict is merely the one the headline points at. The per-instruction
        # lists stay in `extra` -- they are which constraints passed, and
        # report() pools those raw counts rather than averaging the rates here.
        metrics: dict[str, bool | float] = {}
        detail = {}
        for grade in _GRADES:
            out = graders[grade](inp, {prompt: response})
            followed = list(out.follow_instruction_list)
            metrics[f"{grade}_follow_all"] = out.follow_all_instructions
            metrics[f"{grade}_instruction_level"] = (
                sum(followed) / len(followed) if followed else 0.0
            )
            detail[grade] = {"follow_instruction_list": followed}

        # Derived, not recomputed, so the headline cannot disagree with the set.
        correct = bool(metrics["strict_follow_all"])
        score = float(metrics["strict_instruction_level"])
        return True, build_judgement_record(
            instruction_ids,
            [build_rollout_judgement(0, correct, score=score, metrics=metrics)],
            score=score,
            metrics=metrics,
            extra={"key": raw["key"], **detail},
        )

    @override
    async def report(self, finals, fails):
        results: dict[str, float] = {"fails": len(fails)}
        judgements = [f.feedback_result for f in finals]
        for grade in _GRADES:
            prompt_total = len(judgements)
            prompt_correct = sum(
                1 for j in judgements if j["metrics"][f"{grade}_follow_all"]
            )
            # Pooled from raw counts, not averaged from the per-sample rates in
            # `metrics` -- the two differ when samples carry different counts.
            followed = [
                j["extra"][grade]["follow_instruction_list"] for j in judgements
            ]
            instruction_total = sum(len(f) for f in followed)
            instruction_correct = sum(sum(f) for f in followed)

            prompt_level = prompt_correct / prompt_total if prompt_total else 0.0
            # `accuracy` and `prompt_level_accuracy` measure the same thing (upstream
            # computes them by two different routes); both are reported to keep the
            # published key set intact.
            results[f"{grade}_accuracy"] = prompt_level * 100
            results[f"{grade}_prompt_level_accuracy"] = prompt_level * 100
            results[f"{grade}_instruction_level_accuracy"] = (
                instruction_correct / instruction_total * 100
                if instruction_total
                else 0.0
            )
        # hard code score as strict prompt-level accuracy
        results["score"] = results["strict_prompt_level_accuracy"]
        return results

    def _clean_kwargs(self, kwargs):
        # avoid hf datasets underlying Arrow sparse struct problem
        return [{k: v for k, v in d.items() if v is not None} for d in kwargs]
