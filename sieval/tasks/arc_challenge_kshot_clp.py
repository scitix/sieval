"""
ARC-Challenge few-shot base-model conditional-log-prob task (options format).

The "options" MCQ format (Borchmann, ARC "Challenge" Is Not That Challenging,
Findings of ACL 2025; arXiv:2412.17758): the candidate options are listed
``A/B/C/...`` in the prompt and the answer is the option
LETTER. Scoring reads the first output token's ``top_logprobs`` in ONE
inference and argmaxes over the option-letter log-probs — the ``clp`` protocol,
mirroring ``cmmlu_kshot_base_gen``. Scoring requires every option letter to be
present in the top-k and fails the sample otherwise, so partial coverage is
loud rather than a best-of-present guess (default ``logprobs=100``; SGLang
serves 100 by default, on vLLM start with ``--max-logprobs 100``).

This is the base-model options-format counterpart to the ``ppl`` separation task
(``arc_challenge_kshot_ppl``, which scores full option text). DeepSeek switched
from separation to options after V1, so the options-regime number is the target
here; the Qwen2.5 report uses separation (see the ppl sibling).

Comparison target: Qwen2.5-72B-Base ARC-Challenge 25-shot EM = 94.5, taken from
the DeepSeek-V3 report's Table 3 (the Qwen2.5-72B-Base column — DeepSeek-V3-Base's
own entry is 95.3). Not yet validated against a run, so ``status="experimental"``.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from typing import override

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.core.utils.ppl import choice_scores_from_top_logprobs
from sieval.datasets import ARCChallengeDatasetSample

from ._arc import (
    DEFAULT_CLP_LOGPROBS,
    DEFAULT_FEWSHOT_SEED,
    ARCFeedback,
    arc_report,
    build_arc_clp_fewshot_prefix,
    choice_label,
    choice_text,
    format_arc_clp_item,
    sample_arc_fewshot,
)

N_SHOT = 25


@sieval_task(
    name="arc_challenge_kshot_clp",
    display_name="ARC-Challenge (few-shot, conditional log-prob)",
    description="ARC-Challenge few-shot options-format next-token letter accuracy.",
    eval_mode=EvalMode.CLP,
    n_shot=N_SHOT,
    tags=("english", "science", "multiple-choice", "base-model"),
    model_type="gen",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/arc/arc_challenge.yaml"
        ),
        notes=(
            "Shares the ARC-Challenge split/dataset/revision with "
            "lm-evaluation-harness. Uses the 'options' MCQ format (arXiv "
            "2412.17758): options listed A/B/C/... in the prompt, answer is the "
            "option letter, scored by one-call next-token top_logprobs argmax "
            "(the clp protocol; mirrors cmmlu_kshot_base_gen). Requires all "
            "option letters in the top-k and fails the sample otherwise "
            "(default logprobs=100; SGLang serves 100, on vLLM use "
            "--max-logprobs 100). Comparison target: Qwen2.5-72B-Base "
            "ARC-Challenge 25-shot EM = 94.5, from the DeepSeek-V3 report's "
            "Table 3 (Qwen2.5-72B-Base column; DeepSeek switched separation->"
            "options after V1, and the ppl sibling reproduces the separation "
            "number, e.g. the Qwen2.5 report's ~72.4)."
        ),
    ),
)
class ARCChallengeFewShotClpTask(
    Task[
        ARCChallengeDatasetSample,
        str,
        ModelOutput,
        int,
        ARCFeedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        k: int = N_SHOT,
        logprobs: int = DEFAULT_CLP_LOGPROBS,
        fewshot_split: str = "train",
        fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if logprobs < 1:
            raise ValueError(f"logprobs must be >= 1, got {logprobs}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._logprobs = logprobs
        self._fewshot_split = fewshot_split
        self._fewshot_seed = fewshot_seed
        self._fewshot_prefix: str | None = None

    @override
    async def setup(self) -> None:
        # Built once here (setup runs before any preprocess) so the k-exemplar
        # prefix is not rejoined per sample.
        self._fewshot_prefix = self._build_fewshot_prefix()

    @override
    async def preprocess(self, raw, ctx):
        prefix = (
            self._fewshot_prefix
            if self._fewshot_prefix is not None
            else self._build_fewshot_prefix()
        )
        return prefix + format_arc_clp_item(raw["question"], raw["choices"])

    @override
    async def infer(self, pre, ctx):
        # One inference: the next-token distribution over the option letters.
        return await self.model.alogprobs(
            pre, max_tokens=1, logprobs=self._logprobs, echo=False
        )

    @override
    async def postprocess(self, inf, ctx):
        labels = [choice_label(i) for i in range(len(ctx.raw_sample["choices"]))]
        scores, all_present = choice_scores_from_top_logprobs(
            inf.top_logprobs, tuple(labels)
        )
        if not all_present:
            missing = [label for label in labels if scores[label] == float("-inf")]
            raise RuntimeError(
                f"ARC-Challenge top_logprobs missing option token(s) {missing}; "
                f"increase logprobs (got top-k of {self._logprobs}) or raise the "
                "server's max-logprobs so all option letters are returned."
            )
        best_label = max(scores.items(), key=lambda item: item[1])[0]
        return ord(best_label) - ord("A")

    @override
    async def feedback(self, post, ctx):
        answer = ctx.raw_sample["answer"]
        choices = ctx.raw_sample["choices"]
        return True, {
            "correct": post == answer,
            "answer": answer,
            "prediction": post,
            "answer_choice": choice_text(choices, answer),
            "prediction_choice": choice_text(choices, post),
        }

    @override
    async def report(self, finals, fails):
        return arc_report(finals, fails)

    def _build_fewshot_prefix(self) -> str:
        examples = sample_arc_fewshot(
            self.dataset, self._k, self._fewshot_split, self._fewshot_seed
        )
        return build_arc_clp_fewshot_prefix(examples)
