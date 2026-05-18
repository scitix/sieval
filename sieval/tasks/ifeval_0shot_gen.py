from collections import defaultdict
from typing import override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.instruction_following_eval.evaluation_lib import (
    InputExample,
    OutputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import IFEvalDatasetSample


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
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        str,
        dict[str, float],
    ]
):
    def __init__(self, dataset, model, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)

    @override
    async def preprocess(self, raw, ctx):
        return [{"role": "user", "content": raw["prompt"]}]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    @override
    async def postprocess(self, inf, ctx):
        return inf.texts[0]  # # n=1, only one choice, and pass directly

    @override
    async def feedback(self, post, ctx):
        return True, post  # do nothing, pass directly

    @override
    async def report(self, finals, fails):
        inputs = [
            InputExample(
                key=f.raw_sample["key"],
                instruction_id_list=f.raw_sample["instruction_id_list"],
                prompt=f.raw_sample["prompt"],
                kwargs=self._clean_kwargs(f.raw_sample["kwargs"]),
            )
            for f in finals
        ]
        prompt_to_response = {f.raw_sample["prompt"]: f.feedback_result for f in finals}
        results = {"fails": len(fails)}
        for func, grade in [
            (test_instruction_following_strict, "strict"),
            (test_instruction_following_loose, "loose"),
        ]:
            outputs = [func(inp, prompt_to_response) for inp in inputs]
            follow_all_instructions = [o.follow_all_instructions for o in outputs]
            accuracy = sum(follow_all_instructions) / len(outputs)
            results[f"{grade}_accuracy"] = accuracy * 100

            report = self._get_report(outputs)
            results[f"{grade}_prompt_level_accuracy"] = (
                report.get("prompt-level", 0.0) * 100
            )
            results[f"{grade}_instruction_level_accuracy"] = (
                report.get("instruction-level", 0.0) * 100
            )
        # hard code score as strict prompt-level accuracy
        results["score"] = results["strict_prompt_level_accuracy"]
        return results

    def _clean_kwargs(self, kwargs):
        # avoid hf datasets underlying Arrow sparse struct problem
        return [{k: v for k, v in d.items() if v is not None} for d in kwargs]

    def _get_report(self, outputs: list[OutputExample]) -> dict[str, float]:
        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        tier0_total = defaultdict(int)
        tier0_correct = defaultdict(int)

        tier1_total = defaultdict(int)
        tier1_correct = defaultdict(int)

        for example in outputs:
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list

            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

            for instruction_id, followed_or_not in zip(
                instruction_id_list, follow_instruction_list, strict=True
            ):
                instruction_id = instruction_id.split(":")[0]
                tier0_total[instruction_id] += 1
                if followed_or_not:
                    tier0_correct[instruction_id] += 1

            for instruction_id, followed_or_not in zip(
                instruction_id_list, follow_instruction_list, strict=True
            ):
                tier1_total[instruction_id] += 1
                if followed_or_not:
                    tier1_correct[instruction_id] += 1

        # print(f"prompt-level: {prompt_correct / prompt_total}")
        # print(f"instruction-level: {instruction_correct / instruction_total}")
        # print()
        # for instruction_id in sorted(tier0_total.keys()):
        #     accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
        #     print(f"{instruction_id} {accuracy}")
        # print()
        # for instruction_id in sorted(tier1_total.keys()):
        #     accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
        #     print(f"{instruction_id} {accuracy}")
        return {
            "prompt-level": prompt_correct / prompt_total,
            "instruction-level": instruction_correct / instruction_total,
        }
