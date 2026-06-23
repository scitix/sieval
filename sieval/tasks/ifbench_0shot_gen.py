"""IFBench zero-shot generative task.

Deviations from the official AllenAI IFBench evaluation:
- Reasoning-chain stripping relies on the chat backend separating
  ``reasoning_content`` from ``content`` (``texts[0]`` is then the answer
  without the trace), instead of the upstream ``process_output="r1_style"``
  text parsing.
- The upstream ``stop=["</answer>"]`` sequence is not set: with backend-side
  reasoning separation the answer arrives in ``content`` without answer tags.

Official decoding values for reproduction (allenai/IFBench#5): temperature=0,
max_gen_toks=32768, stop=["</answer>"], process_output="r1_style", thinking
enabled. Set decoding via the model config, not in this task.

AI-Generated Code - GPT-5 (OpenAI)
"""

from typing import TYPE_CHECKING, Any, override

from openai.types.chat import ChatCompletionUserMessageParam

if TYPE_CHECKING:
    from sieval.community.ifbench.evaluation_lib import OutputExample

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import IFBenchDatasetSample


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
    reference_impl=ReferenceImpl(
        source="allenai/IFBench",
        url="https://github.com/allenai/IFBench/blob/1091c4c3de6c1f6ed12c012ed68f11ea450b0117/evaluation_lib.py",
        notes=(
            "evaluation_lib + instructions registry/checkers vendored from "
            "AllenAI IFBench. Headline score is prompt-level loose accuracy, "
            "the metric the IFBench paper reports. Comparison target: AllenAI "
            "leaderboard Qwen3-32B = 37.3 (allenai/IFBench#5)."
        ),
    ),
)
class IFBenchZeroShotGenTask(
    Task[
        IFBenchDatasetSample,
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
        return inf.texts[0]

    @override
    async def feedback(self, post, ctx):
        return True, post

    @override
    async def report(self, finals, fails):
        from sieval.community.ifbench.evaluation_lib import (
            InputExample,
            test_instruction_following_loose,
            test_instruction_following_strict,
        )

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
            report = self._get_report(outputs)
            results[f"{grade}_prompt_level_accuracy"] = (
                report.get("prompt-level", 0.0) * 100
            )
            results[f"{grade}_instruction_level_accuracy"] = (
                report.get("instruction-level", 0.0) * 100
            )

        # IFBench reports prompt-level loose accuracy as the headline score.
        results["score"] = results["loose_prompt_level_accuracy"]
        return results

    def _clean_kwargs(self, kwargs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Avoid HF Datasets' Arrow sparse-struct representation for None fields.
        return [{k: v for k, v in d.items() if v is not None} for d in kwargs]

    def _get_report(self, outputs: "list[OutputExample]") -> dict[str, float]:
        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        for example in outputs:
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list

            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

        if prompt_total == 0 or instruction_total == 0:
            return {"prompt-level": 0.0, "instruction-level": 0.0}

        return {
            "prompt-level": prompt_correct / prompt_total,
            "instruction-level": instruction_correct / instruction_total,
        }
