"""IMO-AnswerBench zero-shot generative task.

**Experimental / non-strict port** (``status="experimental"`` — not a frozen
leaderboard contract). IMO-Bench's upstream AnswerBench is an *agentic* harness:
the agent submits its answer via an ``answer`` tool call. This task reproduces it
in a *generative* setting (last-``\\boxed{}`` extraction) instead. Every deviation
from upstream is enumerated in ``reference_impl.notes`` below.

Dual-source lineage: the boxed prompt + last-``\\boxed{}`` extraction follow
eth-sri/matharena; answer equivalence is vendored verbatim from IMO-Bench's
``answer_verification.py`` (``community/imo_bench.py``), plus a documented gen-mode
normalizer (``verify_answer_gen``).

Infer prerequisites: olympiad reasoning traces are very long — set a large output
budget (``max_tokens`` ≈ 131072) and a generous client read-timeout (300s+). At
``max_tokens=65536`` ~22% of samples truncate mid-reasoning with no boxed answer
(scored wrong); the score is therefore budget-sensitive.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.imo_bench import verify_answer_gen
from sieval.community.matharena import build_prompt, extract_answer
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import IMOAnswerBenchDatasetSample

# IMO-Bench AnswerBench is an agentic harness whose only instruction is
# "Please reason step by step." and which reads the final answer via a tool call.
# For a non-agentic generative run we keep that reasoning instruction and add a
# boxed answer format so the short answer is parseable, then extract the last box.
IMO_ANSWER_BENCH_INSTRUCTION = (
    "Please reason step by step. Put your final answer within \\boxed{}."
)


class Feedback(TypedDict):
    correct: bool
    answer: str


@sieval_task(
    name="imo_answer_bench_0shot_gen",
    display_name="IMO-AnswerBench (0-shot, generative)",
    description=(
        "IMO-Bench AnswerBench (Google DeepMind) — 400 short-answer olympiad problems."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="IMO-Bench (Google DeepMind) + eth-sri/matharena",
        url="https://github.com/EnvCommons/IMO-Bench/blob/66b014f1b3799972ddfc32dbacea51b802586141/answer_verification.py",
        notes=(
            "NON-STRICT / EXPERIMENTAL port of IMO-Bench AnswerBench. Deviations "
            "from upstream:\n"
            "1. Harness type: upstream is agentic (answer submitted via an `answer` "
            "tool call); this is generative — last-\\boxed{} extraction.\n"
            "2. Prompt: upstream is the bare 'Please reason step by step.'; we append "
            "'Put your final answer within \\boxed{}.' plus a blank-line separator "
            "before the problem.\n"
            "3. Grading: verify_math_answer is vendored verbatim (math-verify + "
            "normalized-string fallback), but a NON-upstream normalizer "
            "verify_answer_gen (gen-mode formatting + multi-answer set matching) "
            "contributes ~11% of the score — raw verify_math_answer alone = "
            "260/400 = 65.0%, verify_answer_gen = 293/400 = 73.25% (DeepSeek-V4-Pro).\n"
            "4. Data source: HF mirror hf:Hwilner/imo-answerbench (functionally "
            "equivalent to upstream's OpenReward answerbench.csv).\n"
            "5. Dual lineage: prompt + last-\\boxed{} extraction are from "
            "eth-sri/matharena (community/matharena.py); the answer grader is "
            "IMO-Bench (community/imo_bench.py, @66b014f1).\n"
            "Known limitation: \\boxed{} extraction conflates format-compliance with "
            "math ability; a function-calling submission channel reproducing "
            "upstream's answer tool (and dropping verify_answer_gen) is the fidelity "
            "fix. Infer prereqs: large max_tokens (~131072) + generous client "
            "read-timeout (300s+); the score is budget-sensitive."
        ),
    ),
)
class IMOAnswerBenchZeroShotGenTask(
    Task[
        IMOAnswerBenchDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        list[str | None],
        list[Feedback],
        dict[str, float],
    ],
):
    def __init__(self, dataset, model, name: str | None = None, k: int = 1, n: int = 1):
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._n = n

    @override
    async def preprocess(self, raw, ctx):
        return [
            {
                "role": "user",
                "content": build_prompt(IMO_ANSWER_BENCH_INSTRUCTION, raw["question"]),
            },
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Last \boxed{}; non-strict -> fall back to last integer (matharena extractor).
        return [extract_answer(choice, strict_parsing=False) for choice in inf.texts]

    @override
    async def feedback(self, post, ctx):
        feedbacks: list[Feedback] = []
        ground_truth = ctx.raw_sample["answer"]
        for pred in post:
            if pred is None:
                feedbacks.append({"correct": False, "answer": ground_truth})
                continue
            try:
                # IMO-Bench equivalence: official math-verify grader + gen-mode
                # normalization / multi-answer set matching; gold first.
                correct = verify_answer_gen(ground_truth, pred)
            except Exception as e:
                logger.warning("Feedback failed for sample {}: {}", ctx.sample_id, e)
                correct = False
            feedbacks.append({"correct": correct, "answer": ground_truth})
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}

        pass_at_1_total = 0.0
        pass_at_k_total = 0.0
        for f in finals:
            feedbacks = f.feedback_result
            n_samples = len(feedbacks)
            correct_num = sum(1 for f in feedbacks if f["correct"])
            pass_at_1_total += self._pass_at_k(n_samples, correct_num, 1)
            if self._k > 1:
                pass_at_k_total += self._pass_at_k(n_samples, correct_num, self._k)

        pass_at_1 = pass_at_1_total * 100 / total
        metrics = {"score": pass_at_1, "fails": len(fails), "pass@1": pass_at_1}
        if self._k > 1:
            metrics[f"pass@{self._k}"] = pass_at_k_total * 100 / total
        return metrics

    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        if n < k:
            return 0.0
        if c == 0:
            return 0.0
        # Formula: 1 - product_{i=0}^{k-1} (n - c - i) / (n - i)
        # This calculates the probability that all k samples are wrong
        prob_all_wrong = 1.0
        for i in range(k):
            prob_all_wrong *= (n - c - i) / (n - i)
        return 1.0 - prob_all_wrong
