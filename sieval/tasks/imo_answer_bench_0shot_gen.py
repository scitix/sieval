"""IMO-AnswerBench zero-shot generative task.

Non-strict *generative* port of Google DeepMind's IMO-Bench AnswerBench (upstream
is agentic — the agent submits its answer via an ``answer`` tool call; here the
model answers generatively and we extract the last ``\\boxed{}``). The OFFICIAL
grader is an LLM autograder (AnswerAutoGrader, Gemini 2.5 Pro); as a deterministic,
reproducible SUBSTITUTE we vendor EnvCommons's ``verify_math_answer`` (math_verify)
fed by a parsing-layer ``normalize_answer`` (symmetric ``$``-wrapping like the HMMT
sibling) so it handles commutativity / factoring / set-equality. This is a
deliberate, strictly-more-conservative deviation from the official grader — every
divergence is enumerated in ``reference_impl.notes`` below.

Dual-source lineage: the boxed prompt + last-``\\boxed{}`` extraction follow
eth-sri/matharena (``community/matharena.py``); the answer grader is EnvCommons's
deterministic re-impl of IMO-Bench (``community/imo_bench.py``).

Infer prerequisites: olympiad reasoning traces are very long — set a large output
budget (``max_tokens`` ≈ 131072) and a generous client read-timeout (300s+). At
``max_tokens=65536`` ~22% of samples truncate mid-reasoning with no boxed answer
(scored wrong); the score is therefore budget-sensitive.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.imo_bench import normalize_answer, verify_math_answer
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
    status="stable",
    reference_impl=ReferenceImpl(
        source="IMO-Bench AnswerBench (Google DeepMind, arXiv 2511.01846) + eth-sri/matharena",  # noqa: E501
        url="https://github.com/google-deepmind/superhuman/tree/96fa6c4cc3a9bb7450ee7b6773b659d3a030dace/imobench",  # noqa: E501
        notes=(
            "Non-strict GENERATIVE port of IMO-Bench AnswerBench (Google DeepMind). "
            "Authoritative source: google-deepmind/superhuman (imobench/) + "
            "imobench.github.io + arXiv 2511.01846. Deviations from upstream:\n"
            "1. Harness: upstream is agentic (answer via an `answer` tool call); this "
            "is generative — last-\\boxed{} extraction (matharena extractor).\n"
            "2. Prompt: upstream is the bare 'Please reason step by step.'; we append "
            "'Put your final answer within \\boxed{}.' + a blank-line separator.\n"
            "3. Data: official answerbench_v2.csv (v2, released 2026-02-12, pinned by "
            "commit + sha256), which fixed ambiguous statements / incorrect answers; "
            "the old answerbench.csv (v1) is deprecated and NOT used. Two v2 rows "
            "carry known upstream spreadsheet artifacts, kept VERBATIM (faithful to "
            "the checksummed official CSV): imo-bench-algebra-036 (gold corrupted to "
            "the Category 'Algebra') and imo-bench-geometry-004 (gold Excel-"
            "autoformatted to the date serial '45752'); both grade wrong for ~any "
            "model (score impact <=0.5%).\n"
            "4. Grader: a DELIBERATE deviation from official. The official "
            "AnswerBench grader is an LLM autograder (AnswerAutoGrader, Gemini "
            "2.5 Pro; arXiv 2511.01846 §2.3/§5.1) and the paper rejects "
            "symbolic/SymPy matching as too narrow; the official imobench/ ships "
            "DATA ONLY (no grader code). With no runnable official grader, we "
            "vendor EnvCommons/IMO-Bench@66b014f1's deterministic "
            "verify_math_answer (its OpenReward re-impl; the 'No LLM graders / "
            "math_verify' wording is EnvCommons's README, not the paper). A "
            "deterministic grader fits sieval's reproducibility contract (an LLM "
            "grader would not) and is strictly MORE conservative — it cannot "
            "grade prose / infinite-set / functional-family answers, so sieval "
            "UNDER-counts vs the official autograder (see 'Out of scope' below). "
            "verify_math_answer / parse_answer are behaviorally identical to the "
            "pinned EnvCommons source (imports made lazy per sieval discipline), "
            "unchanged pin->HEAD.\n"
            "Grading path: a parsing-layer normalize_answer reconstructs the clean "
            "answer an agent would submit, then math_verify (symmetric $-wrap, HMMT-"
            "aligned) does all equivalence — commutativity / factoring / set-equality; "
            "no bespoke matching.\n"
            "Number: DeepSeek-V4-Pro pass@1 on v2 = 317/400 = 79.25% (real sieval "
            "eval through this task's code path, max_tokens=131072, 0 truncation, 0 "
            "unrecovered failures — 15 transient scitix upstream stream errors were "
            "recovered via --resume; the 2 corrupted-gold rows above count wrong). "
            "Historical: a v1 offline re-grade gave 76.75% — kept for reference only, "
            "NOT code-path-reproducible and computed against the deprecated v1 golds.\n"
            "Out of scope (needs upstream's agentic answer-tool channel or an LLM "
            "judge, not parsing): prose answers ('all odd primes'), infinite sets, "
            "quantified functional families — a few genuinely-correct such answers "
            "stay ungraded. \\boxed{} conflates format-compliance with math ability; "
            "the fidelity fix is a function-calling submission channel reproducing "
            "upstream's answer tool. Infer prereqs: large max_tokens (~131072) + "
            "generous client read-timeout (300s+); the score is budget-sensitive."
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
        # Extract the last \boxed{} (matharena extractor), then reconstruct the
        # clean answer an agent would submit (parsing layer) so math_verify can
        # judge equivalence. None => no boxed answer found.
        return [
            normalize_answer(extract_answer(choice, strict_parsing=False))
            for choice in inf.texts
        ]

    @override
    async def feedback(self, post, ctx):
        feedbacks: list[Feedback] = []
        ground_truth = ctx.raw_sample["answer"]
        gold = normalize_answer(ground_truth)
        for pred in post:
            if pred is None or gold is None:
                feedbacks.append({"correct": False, "answer": ground_truth})
                continue
            try:
                # Verbatim upstream grader (math_verify); symmetric $-wrapping like
                # the HMMT sibling so full expressions parse, gold first. math_verify
                # handles commutativity / factoring / set-equality — no bespoke logic.
                correct = verify_math_answer(f"${gold}$", f"${pred}$")
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
