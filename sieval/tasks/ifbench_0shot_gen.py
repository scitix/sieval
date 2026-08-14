"""IFBench zero-shot generative task.

Deviations from the official AllenAI IFBench evaluation:
- Reasoning-chain stripping relies on the chat backend separating
  ``reasoning_content`` from ``content`` (``texts[0]`` is then the answer
  without the trace), instead of the upstream ``process_output="r1_style"``
  text parsing.
- The upstream ``stop=["</answer>"]`` sequence is not set: with backend-side
  reasoning separation the answer arrives in ``content`` without answer tags.

Official leaderboard hyperparameters (allenai/IFBench#5): temperature=0,
max_gen_toks=32768, stop=["</answer>"], process_output="r1_style", thinking
enabled. These do not reproduce the score on Qwen3 thinking mode — greedy
decoding (temperature=0) makes the reasoning trace loop and overrun the token
budget, yielding empty answers scored as failures; reproduction requires
Qwen3's recommended sampling (temperature=0.6, top_p=0.95, top_k=20). Set
decoding via the model config, not in this task.

Infra: scoring lazily fetches the NLTK corpora it needs (punkt, stopwords,
averaged_perceptron_tagger_eng) on first use if absent — an eval-time network
dependency. The Docker image pre-bakes them; offline runs must pre-stage them
(see SIEVAL_IFBENCH_NLTK_DATA in sieval.community.ifbench).

AI-Generated Code - GPT-5 (OpenAI)
"""

from typing import Any, override

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
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
)
from sieval.datasets import IFBenchDatasetSample

# Both readings are published IFBench metrics. `loose` is the headline (the task's
# `score` is loose prompt-level accuracy) -- the opposite of IFEval, where strict
# is the headline; the shape is otherwise identical.
_GRADES = ("strict", "loose")


@sieval_task(
    name="ifbench_0shot_gen",
    display_name="IFBench (0-shot, generative)",
    description=(
        "Precise instruction-following benchmark with verifiable OOD constraints."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="ifbench",
    model_type="chat",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="allenai/IFBench",
        url="https://github.com/allenai/IFBench/blob/1091c4c3de6c1f6ed12c012ed68f11ea450b0117/evaluation_lib.py",
        notes=(
            "evaluation_lib + instructions registry/checkers vendored from "
            "AllenAI IFBench. Headline score is prompt-level loose accuracy, "
            "the metric the IFBench paper reports. Comparison target: "
            "Qwen3-32B = 37.3 (AllenAI README leaderboard). The authors' "
            "hyperparameters in allenai/IFBench#5 use greedy temperature=0, "
            "which does not reproduce the score on this thinking model; "
            "reproduced with Qwen3's recommended sampling (temperature=0.6, "
            "top_p=0.95, top_k=20, max_tokens=38912). As that sampling is "
            "non-greedy the score is a stochastic band with 37.3 at the top "
            "edge, not a deterministic value. Four of the vendored checkers "
            "grade something other than what their own instruction says "
            "(format:line_indent, ratio:sentence_type, words:words_position, "
            "words:vowel). Kept as-is here, per the unqualified-name rule; "
            "ifbench_0shot_gen_fixed repairs them and carries the measured "
            "delta."
        ),
    ),
    # Not empirically validated as equivalent: the official temperature=0
    # protocol does not reproduce, and the 37.3 match is a stochastic sample
    # under substituted sampling. Faithful port, unverified reproduction.
    status="experimental",
)
class IFBenchZeroShotGenTask(
    Task[
        IFBenchDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    def __init__(self, dataset, model, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)

    def _instruction_dict(self) -> dict[str, type] | None:
        """Registry the checkers are looked up in; ``None`` is the vendored one.

        The single seam ``ifbench_0shot_gen_fixed`` needs. Everything else about
        how a sample is prompted, graded and pooled is shared, so overriding this
        cannot make the two tasks differ in any other way.
        """
        return None

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [{"role": "user", "content": raw["prompt"]}],
            # The "ground truth" is the constraint set the response must satisfy.
            reference=list(raw["instruction_id_list"]),
            extra={"key": raw["key"], "kwargs": self._clean_kwargs(raw["kwargs"])},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended task: the response *is* the answer, so no extraction step.
        # A blank response normalizes to None so `extracted` stays a real signal.
        text = inf.texts[0]
        return build_prediction_record([text if text.strip() else None])

    @override
    async def feedback(self, post, ctx):
        """Grade this sample's constraints, strict and loose.

        Graded here rather than in report() -- the same relocation #60 made for
        IFEval -- so every sample's verdict is on disk and inspectable instead of
        existing only inside the aggregate. Both graders are pure per-sample
        (their only use of the response map is `prompt_to_response[inp.prompt]`),
        so scoring one sample at a time is equivalent to upstream's whole-set call.
        """
        from sieval.community.ifbench.evaluation_lib import (
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
        response = post["rollouts"][0].get("prediction") or ""
        inp = InputExample(
            key=raw["key"],
            instruction_id_list=instruction_ids,
            prompt=prompt,
            kwargs=self._clean_kwargs(raw["kwargs"]),
        )

        # Both readings are co-equal published metrics, so both go in `metrics`;
        # loose is merely the one the headline points at. The per-instruction
        # lists stay in `extra` -- they are WHICH constraints passed, and report()
        # pools those raw counts rather than averaging the per-sample rates.
        metrics: dict[str, bool | float] = {}
        detail = {}
        instruction_dict = self._instruction_dict()
        for grade in _GRADES:
            out = graders[grade](
                inp, {prompt: response}, instruction_dict=instruction_dict
            )
            followed = list(out.follow_instruction_list)
            metrics[f"{grade}_follow_all"] = out.follow_all_instructions
            metrics[f"{grade}_instruction_level"] = (
                sum(followed) / len(followed) if followed else 0.0
            )
            detail[grade] = {"follow_instruction_list": followed}

        # Derived, not recomputed, so the headline cannot disagree with the set.
        correct = bool(metrics["loose_follow_all"])
        score = float(metrics["loose_instruction_level"])
        return True, build_judgement_record(
            instruction_ids,
            [build_rollout_judgement(0, correct, score=score, metrics=metrics)],
            score=score,
            metrics=metrics,
            extra={"key": raw["key"], **detail},
        )

    @override
    async def report(self, finals, fails):
        # Aggregation only -- grading moved to feedback(). Reading the persisted
        # verdicts also drops report()'s dependency on raw_sample being present.
        results: dict[str, float | str] = {"fails": len(fails)}
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

            results[f"{grade}_prompt_level_accuracy"] = (
                prompt_correct / prompt_total * 100 if prompt_total else 0.0
            )
            results[f"{grade}_instruction_level_accuracy"] = (
                instruction_correct / instruction_total * 100
                if instruction_total
                else 0.0
            )

        # IFBench reports prompt-level loose accuracy as the headline score.
        results["score"] = results["loose_prompt_level_accuracy"]
        # The loose/strict x prompt/instruction rates are co-equal, and IFBench's
        # headline is the *loose* one where IFEval's is strict — a difference two
        # sibling tasks cannot show in the value, only in this key. Denominator is
        # the judged set: `prompt_total` counts judgements, and a pipeline failure
        # is reported in `fails` rather than scored as a followed-nothing prompt.
        results[SCORE_KEY_FIELD] = "loose_prompt_level_accuracy"
        results[DENOMINATOR_FIELD] = DENOMINATOR_JUDGED
        return results

    def _clean_kwargs(self, kwargs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Avoid HF Datasets' Arrow sparse-struct representation for None fields.
        return [{k: v for k, v in d.items() if v is not None} for d in kwargs]
