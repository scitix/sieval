"""
ARC-Easy few-shot base-model perplexity task (full text, unconditional norm).

Scores ARC-Easy in the SEPARATION regime (Qwen2.5-report style; the setup
DeepSeek used before switching to the options format after V1). Each answer
OPTION is scored as
a text continuation of ``"Question: {q}\nAnswer:"`` — no letters, no options
listed in the prompt — and the prediction is the option with the highest
UNCONDITIONALLY-NORMALIZED sequence log-likelihood (Brown et al. 2020):

    score_i = logP(option_i | few_shot + question + "Answer:")
              - logP(option_i | "Answer:")

``argmax`` over options; ``acc``/``score`` is exact-match vs the gold index.
This is the ``ppl`` protocol (one inference per candidate, full answer text) —
distinct from the single-letter ``clp`` method (CMMLU/MMLU-Base). Because the
per-option context is identical within a sample, its log-prob cancels in the
argmax, so summing the echoed INPUT tokens (excluding the trailing generated
token — see ``echoed_logprob``) is exact for EM.

Scored via ``SglangGenModel``'s echoed-input logprobs (``engine: sglang``). The
sglang server MUST be launched with ``--disable-radix-cache``: on a prefix-cache
hit sglang drops logprobs for cached positions, and the model fails loud rather
than score a truncated echoed sequence.

Comparison target — the SEPARATION regime (arXiv 2412.17758), NOT the
options/letter regime. No concrete ARC-Easy separation figure is cited as an
anchor here: the Qwen2.5 report's headline ARC number is ARC-Challenge (≈ 72.4,
see the ppl sibling), and neither it nor the DeepSeek reports give an ARC-Easy
*separation* score to compare against. The 98.4 options/letter figure belongs to
the ``clp`` sibling (``arc_easy_kshot_clp``), not this task. Not calibrated, so
``status="experimental"``.

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
from sieval.datasets import ARCEasyDatasetSample

from ._arc import (
    ARC_UNCOND_CONTEXT,
    DEFAULT_FEWSHOT_SEED,
    ARCFeedback,
    arc_report,
    build_arc_ppl_fewshot_prefix,
    choice_text,
    echoed_logprob,
    format_arc_ppl_context,
    sample_arc_fewshot,
)

N_SHOT = 25


@sieval_task(
    name="arc_easy_kshot_ppl",
    display_name="ARC-Easy (few-shot, perplexity)",
    description="ARC-Easy few-shot full-text unconditional-normalized accuracy.",
    eval_mode=EvalMode.PPL,
    n_shot=N_SHOT,
    tags=("english", "science", "multiple-choice", "base-model"),
    model_type="gen",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/arc/arc_easy.yaml"
        ),
        notes=(
            "Shares the ARC-Easy split/dataset/revision with "
            "lm-evaluation-harness, but scores in the SEPARATION regime "
            "(Qwen2.5-report style), not upstream acc/acc_norm: each option's "
            "full TEXT is scored as the "
            "continuation of 'Question: {q}\\nAnswer:' and normalized "
            "UNCONDITIONALLY (Brown et al. 2020) as logP(opt|context) - "
            "logP(opt|'Answer:'), argmax = prediction. This is the ppl "
            "protocol (one inference per option; full answer text), not the "
            "single-letter clp method. Requires the sglang server launched "
            "with --disable-radix-cache (SglangGenModel fails loud on a cache "
            "hit that truncates echoed logprobs). Comparison target — the "
            "SEPARATION regime (arXiv 2412.17758), Qwen2.5-report style. The "
            "98.4 options/letter figure is the clp sibling, not this task."
        ),
    ),
)
class ARCEasyFewShotPplTask(
    Task[
        ARCEasyDatasetSample,
        str,
        list[ModelOutput],
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
        fewshot_split: str = "train",
        fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
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
        return prefix + format_arc_ppl_context(raw["question"])

    @override
    async def infer(self, pre, ctx):
        # Per option: a conditional call (full context + option text) and an
        # unconditional call ("Answer:" + option text) for Brown-et-al.
        # normalization. echo=True returns the whole sequence's token logprobs.
        # logprobs=0: ppl reads only token_logprobs, not top_logprobs (matches
        # the hellaswag sibling and avoids requesting an unused top-k).
        outputs: list[ModelOutput] = []
        for choice in ctx.raw_sample["choices"]:
            outputs.append(
                await self.model.alogprobs(f"{pre} {choice}", echo=True, logprobs=0)
            )
            outputs.append(
                await self.model.alogprobs(
                    f"{ARC_UNCOND_CONTEXT} {choice}", echo=True, logprobs=0
                )
            )
        return outputs

    @override
    async def postprocess(self, inf, ctx):
        best_index = -1
        best_score: float | None = None
        for index in range(len(ctx.raw_sample["choices"])):
            score = echoed_logprob(inf[2 * index]) - echoed_logprob(inf[2 * index + 1])
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        return best_index

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
        return build_arc_ppl_fewshot_prefix(examples)
