"""
GSM1k 0-shot generative task — the chat-side half of the GSM8K/GSM1k pair.

Scale AI published GSM1k under one protocol only: 5-shot raw completion, ported
here as `gsm1k_kshot_base_gen`. That protocol needs a `gen` model, so on its own
it leaves GSM1k unrunnable for the chat endpoints this repo mostly verifies. This
task supplies the missing half by applying `gsm8k_0shot_gen`'s protocol — the
DeepSeek-Math zero-shot CoT path — to GSM1k, unchanged:

* Prompt (DeepSeek's `run_subset_parallel.py::markup_question`, language="en",
  task="cot"): the user turn is `{question}` followed by `"\\nPlease reason step
  by step, and put your final answer within \\boxed{}."`, with the chat template
  applied by the serving backend.
* Answer extraction: `extract_answer(reasoning, exhaust=False)` — DeepSeek's
  `extract_last_single_answer`: last `\\boxed{...}` if present, else the text
  after `"he answer is"`, else the last number, then `strip_string`.
* Scoring: `is_correct` — DeepSeek's `eval_last_single_answer` (numeric isclose
  with %-variants, then a sympy symbolic fallback). `score` is this accuracy.

Extraction and scoring live verbatim in `sieval.community.deepseek_math`, vendored
byte-faithfully from DeepSeek-Math at the pinned commit. Nothing about them is
GSM8K-specific: the gold is a bare integer either way. The *invocation* is the
sibling's too, and that part is not cosmetic: `is_correct` is offloaded to a
worker process under `GRADE_TIMEOUT` rather than called inline, because it
reaches `math_equal` with `timeout=False` and so runs `simplify` with no bound of
its own (criterion 2 in `core/utils/offload.py`). A grade-school gold is a bare
integer, but the *prediction* is arbitrary model output, and grading is
synchronous on the one event loop every runner in the session shares.

**This is a different measurement regime from upstream's, not a port of it — no
published GSM1k number corresponds to a 0-shot chat score.** What it buys is a
*prompt-exact pair*: run this task and `gsm8k_0shot_gen` against the same model
and the two differ only in which problem set the question came from — identical
prompt template, identical extractor, identical scorer, no few-shot exemplars to
vary. The GSM8K − GSM1k **diff** is the measurement, and it is the quantity GSM1k
exists to produce (the paper's Table 1 is a diff column first, an accuracy column
second). A single absolute number here aligns with nothing external.

The **diff** does have one external reference point, though. The paper's Table 2
"Alternative Prompt" is itself a second-regime column: for
Meta-Llama-3-8B-Instruct it reports 0.023 against Table 1's 0.062, i.e. a
non-standard prompt roughly halves the measured gap. Measured here on that model
(2026-08-11, greedy, `max_tokens=1024`, `fails=0`): GSM8K 71.57 → GSM1k 68.63,
diff **2.94** — the same halving, from a different non-standard regime. The two
protocols are not the same, so this is corroboration rather than alignment; what
it establishes is that the pairing lands inside the paper's own second-regime
range instead of somewhere unexplained.

Deviation from the sibling it mirrors: the gold needs no `answer.split("####")`
step, because GSM1k's `answer` field already *is* the bare final answer — see
`sieval/datasets/gsm1k.py`. Both tasks divide `report()` by
`len(finals) + len(fails)`, so a pipeline failure counts as wrong on both sides
of the diff.

`status="experimental"`: the extraction/scoring layer is the sibling's — the same
vendored functions, reached through the same offload — and the pairing has now
been run end-to-end (see the diff above). It stays experimental because, unlike
the sibling, it has no published column of its own to be validated *against* —
only a second-regime diff that corroborates rather than aligns.

Repro decoding (model-layer assets — set via `models:` / `infer_args`, not in
this code): greedy `temperature=0`, `top_p=1.0`, `max_tokens=1024`, stop = the
model's EOS only, matching `gsm8k_0shot_gen` so the pair stays comparable.

References:

* GSM1k paper (v4 — the revision Tables 1 and 2 are read from):
  <https://arxiv.org/abs/2405.00332v4>
* Protocol source: <https://github.com/deepseek-ai/DeepSeek-Math/tree/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import override

from loguru import logger

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
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    first_rollout_correct,
    health_metrics,
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import GSM1KDatasetSample

# Verbatim from run_subset_parallel.py::markup_question (language="en",
# task="cot"): f"{content}\nPlease reason step by step, and put your final
# answer within " + "\\boxed{}."
COT_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


@sieval_task(
    name="gsm1k_0shot_gen",
    display_name="GSM1k (0-shot, generative)",
    description="GSM1k 0-shot chat-model eval, prompt-paired with gsm8k_0shot_gen.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "math-word-problems", "open-ended"),
    deps_group="math",
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="deepseek-ai/DeepSeek-Math",
        url=(
            "https://github.com/deepseek-ai/DeepSeek-Math/tree/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation"
        ),
        notes=(
            "Protocol borrowed from the sibling `gsm8k_0shot_gen`, so the two "
            "form a prompt-exact pair: user turn = question + "
            '"Please reason step by step, and put your final answer within '
            '\\boxed{}.", chat template applied by the serving backend; '
            "extract_answer(exhaust=False) (= extract_last_single_answer) and "
            "is_correct/math_equal (= eval_last_single_answer) are vendored "
            "byte-for-byte in sieval.community.deepseek_math. Grading is "
            "offloaded to a worker process under GRADE_TIMEOUT, as for every "
            "sympy-backed grader here: `math_equal` is reached with "
            "timeout=False, so nothing else bounds `simplify`. Gold is GSM1k's "
            "`answer` verbatim (already the bare final answer, so no '####' "
            "split). Scale AI published GSM1k at 5-shot raw completion only "
            "(see gsm1k_kshot_base_gen) — no published number matches this "
            "0-shot chat protocol, so read the GSM8K - GSM1k diff, not the "
            "absolute score. Repeats: 1 rollout, greedy at temperature 0."
        ),
    ),
)
class GSM1KZeroShotGenTask(
    Task[
        GSM1KDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float | str],
    ]
):
    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {"role": "user", "content": raw["question"] + COT_INSTRUCTION},
            ],
            reference=raw["answer"],
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        from sieval.community.deepseek_math import extract_answer

        text = inf.texts[0] if inf.texts else ""
        # extract_answer returns "" when nothing was found; None is the protocol's
        # spelling of that, and feedback restores "" for the grader.
        return build_prediction_record([extract_answer(text, exhaust=False) or None])

    @override
    async def feedback(self, post, ctx):
        from sieval.community.deepseek_math import is_correct

        gold = ctx.raw_sample["answer"]
        # `or ""` gives the grader the same empty string a failed extraction
        # produced upstream, rather than a None it has no branch for.
        prediction = post["rollouts"][0].get("prediction") or ""
        # Offloaded like every other sympy-backed grader here (gsm8k_0shot_gen,
        # gsm_plus_0shot_gen, hendrycks_math): `is_correct` reaches `math_equal`
        # with its default `timeout=False`, so `simplify` runs with no bound of
        # its own — criterion 2 in `core/utils/offload.py`, which names these two
        # DeepSeek-Math graders explicitly. Inline it is unbounded CPU on the one
        # event loop every runner in the session shares: measured on this
        # grader, a 26-character prediction of nested `sqrt(...)` costs 46 s at
        # depth 11 and over 9 min at depth 12, and a bare integer gold cannot
        # short-circuit it.
        try:
            correct = await run_cpu_bound(
                is_correct,
                {"prediction": prediction, "answer": gold},
                timeout=GRADE_TIMEOUT,
            )
        except TimeoutError:
            # An ungradeable answer is a wrong answer, not a failed run — the
            # contract every sibling math grader keeps. Propagating would land
            # the sample in `fails`, which reads as infrastructure breakage; the
            # accuracy is identical either way, since `report` counts fails in
            # the denominator.
            logger.warning(
                "Grading sample {} exceeded {}s and was scored wrong; the "
                "prediction is likely a shape `simplify` cannot bound.",
                ctx.sample_id,
                GRADE_TIMEOUT,
            )
            correct = False
        return True, build_judgement_record(
            gold, [build_rollout_judgement(0, bool(correct))]
        )

    @override
    async def report(self, finals, fails):
        # Accuracy over the full requested set (finals + fails), matching
        # `gsm8k_0shot_gen` so both sides of the paired diff count a pipeline
        # failure as wrong rather than excluding it.
        total = len(finals) + len(fails)
        # First-rollout, because that is the axis a one-greedy-draw protocol
        # publishes — the shared helper the sibling uses, not a local re-count.
        accuracy = 100 * first_rollout_correct(finals) / total if total else 0.0
        metrics: dict[str, float | str] = {
            "score": accuracy,
            "fails": len(fails),
            "accuracy": accuracy,
            SCORE_KEY_FIELD: "accuracy",
            # REQUESTED on *both* halves of this pair (see `gsm8k_0shot_gen`), so
            # the diff cannot be a denominator artifact — the one thing a paired
            # benchmark must not leave ambiguous on disk.
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # On BOTH halves of the pair, for the same reason the extraction rules are
        # named rather than shared: without `n_unextracted` on each side, a
        # GSM8K - GSM1k gap cannot be told apart from a difference in how often
        # extraction failed, which is the confound this benchmark exists to avoid.
        metrics |= health_metrics(finals)
        return metrics
