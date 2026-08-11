"""
GSM-Plus 0-shot generative task, aligned with the upstream GSM-Plus evaluation.

Port of GSM-Plus's zero-shot CoT (chat/instruct) path (pinned commit
``3474129e``, ``scripts/openai_model_inference.py`` with ``--prompt_type cot``):

* Prompt (``prompt_template.py::cot_prompt_map_func``): a system turn carrying
  the ``#### [value]`` output contract, then a user turn
  ``"Question:\\n{question}\\nAnswer:\\nLet's think step by step."``. The serving
  backend applies the model's own chat template.
* Gold answer: ``extract_gold_ans`` fed ``solution``, **not** ``answer`` —
  upstream's ``get_gsmplus`` puts ``item["solution"]`` in its ``answers`` list.
* Answer extraction dispatches on the row's ``perturbation_type``
  (``extract_ans.py::test_answer``): seven perturbations use
  ``extract_pred_ans``, while ``critical thinking`` uses
  ``extract_pred_ans_none``, because that perturbation deletes a quantity the
  question needs, making the gold the string ``"None"`` ("unanswerable").
* Scoring: ``normalize_final_answer`` then ``check_sympy_equivalence``. Both,
  and the extractors, are vendored in ``sieval.community.gsm_plus``.
* ``score`` is accuracy over the whole set; ``report`` adds a per-perturbation
  breakdown and ``score_wo_critical_thinking`` (upstream's ``gsmplus_wo_ncr``),
  the pair the paper leads with.

**Fidelity, measured.** Replaying upstream's own stored predictions for both
models it published on this path reproduces its persisted ``gold`` and ``pred``
on 10552/10552 items each, and its verdict on 10527/10552 (99.76%; overall 61.43
vs published 61.19) for ``results/gpt-3.5-turbo.json``, 10537/10552 (99.86%;
85.72 vs 85.58) for ``results/gpt4.json`` — the latter after mapping its v0
labels, where ``necessary constraint removal`` is ``critical thinking``, the
``ncr`` of ``gsmplus_wo_ncr``.

All 40 verdict diffs go one way — upstream ``False``, this port ``True`` — and
every one is a genuinely equal pair (``3/1`` vs ``3``, ``7/20`` vs ``0.35``). The
cause is upstream's environment, not its logic: it pins ``sympy==1.12`` with no
``antlr4-python3-runtime``, so ``parse_latex`` raised into
``check_sympy_equivalence``'s bare ``except:``, making its published table a
string-equality one. sieval's ``[math]`` extra pins that runtime, so the same
vendored code reaches its symbolic branch. 31 of the 40 land on
``integer-decimal-fraction conversion``, the perturbation whose whole purpose is
rewriting integers as decimals and fractions. Matching the published digits would
mean deliberately breaking ``parse_latex``, so the port stays faithful.

**What ``status="stable"`` does not claim: a published-magnitude match.** Only
``gpt-3.5-turbo`` and ``gpt4`` — 2 of upstream's 28 published rows — came from
this 0-shot chat path; the other 26 came from ``general_model_inference.py``, an
n-shot raw-completion path, so a 0-shot score is no defect against an 8-shot row.
What transfers is the **shape**: ``critical thinking`` is the lowest of the eight
cells in 26 of 26 published rows carrying all of them, and two live runs
reproduce it (``gemma-3-27b-it`` full 10552: 84.11 overall / 63.08 that cell,
rank 1/8; ``gpt-oss-120b`` testmini: 80.83 / 47.67, rank 1/8; Spearman rho +0.976
vs the GPT-4 row for both). Live text was also scored differentially against
upstream's unmodified ``extract_ans.py``: 13752/13752 on gold, pred and verdict.

**Trap: the ``critical thinking`` cell is format-gated** — upstream's design,
ported deliberately. ``extract_pred_ans_none`` scores any response with no
``####`` marker as ``"None"``, i.e. *correct*, so an empty response, a
``\\boxed{}`` answer or a plain-prose number all score correct there, while the
same text is graded normally on the other seven perturbations. Measured share of
that cell's credit *earned* by a refusal phrase rather than *granted* by the
missing marker: 80.9% on the gemma run, but 1.4% on gpt-oss-120b. Read the cell
together with ``n_unextracted`` — a right-looking shape does not establish that
it measured refusal.

**Consequence: the repro decoding below is valid for a non-reasoning model
only.** A reasoning model spends ``max_tokens=512`` on hidden reasoning and
returns empty content, which the gate scores *correct* on ``critical thinking``
and wrong everywhere else — inverting the benchmark exactly where the paper's
headline sits. Paired control, same model and same 400 items with only
``max_tokens`` differing: that cell is 92.00 and the *easiest* of the eight at
512 (rank 8/8, rho +0.000 vs GPT-4), 64.00 and the hardest at 4096, while all
seven numeric cells rise. Most models tested returned empty content at 512, so
this is the common case rather than a corner.

Other deviations from upstream (documented, not silent):

* Only ``--prompt_type cot`` is ported; ``pot`` / ``complex`` / ``contrastive`` /
  ``ltm`` / ``cot_sc`` are separate prompting techniques (the paper's Table 5).
* Single rollout — upstream's ``cot_sc`` majority-votes 5 samples at temperature
  0.7 (``test_answer(mv=5)``); the ported path is ``mv == 1``.
* Report keys use the dataset's "distraction insertion" spelling; upstream's
  ``results/*.json`` dumps say "distractor insertion".
* Upstream's GSM8K-vs-GSM-Plus confusion matrix and decay rate
  (``report_metrics.py``) need a paired GSM8K run, so they belong to cross-task
  analysis; the ``seed_*`` columns they need are preserved on every sample.

Repro decoding (model-layer assets — set via ``models:`` / ``infer_args``, not in
this code): greedy ``temperature=0``, ``top_p=1.0``, ``max_tokens=512``, no stop
sequences (``extract_ans.py::invoke_openai`` defaults for non-``pot`` prompts).

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from collections import defaultdict
from typing import override

from loguru import logger

from sieval.community.gsm_plus import (
    CRITICAL_THINKING,
    extract_gold_ans,
    extract_prediction,
    is_equivalent,
)
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
    health_metrics,
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import GSMPlusDatasetSample

# Verbatim from prompt_template.py::cot_prompt_map_func, which returns
# (template, instruction) -> (user turn, system turn).
SYSTEM_INSTRUCTION = (
    "Your task is to solve a series of math word problems by providing the "
    "final answer. Use the format #### [value] to highlight your answer. For "
    "example, if the answer is 560, you should write #### 560. Make sure to "
    "carefully read and understand each problem before providing your answer."
)


def _user_turn(question: str) -> str:
    return f"Question:\n{question}\nAnswer:\nLet's think step by step."


def _metric_key(perturbation_type: str) -> str:
    """Slug a perturbation type into a report key.

    ``"integer-decimal-fraction conversion"`` ->
    ``"integer_decimal_fraction_conversion"``.
    """
    return perturbation_type.replace("-", "_").replace(" ", "_")


@sieval_task(
    name="gsm_plus_0shot_gen",
    display_name="GSM-Plus (0-shot, generative)",
    description="GSM-Plus 0-shot CoT eval, scored overall and per perturbation type.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "math-word-problems", "open-ended", "robustness"),
    deps_group="math",
    model_type="chat",
    status="stable",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="qtli/GSM-Plus",
        url=(
            "https://github.com/qtli/GSM-Plus/tree/3474129ec12fcd3e8ac08cb037aca1928efca98c/scripts"
        ),
        notes=(
            "Zero-shot CoT protocol (--prompt_type cot): system turn = the "
            '"#### [value]" instruction, user turn = "Question:\\n{question}\\n'
            "Answer:\\nLet's think step by step.\"; chat template applied by the "
            "serving backend. Gold from `solution` via extract_gold_ans; "
            "extraction dispatches on perturbation_type (extract_pred_ans_none "
            'for `critical thinking`, whose gold is the string "None"); '
            "normalize_final_answer + check_sympy_equivalence scoring, all "
            "vendored in sieval.community.gsm_plus. LICENSE: that file is "
            "CC-BY-SA-4.0, not the repo's Apache-2.0 -- upstream's code repo "
            "states none at the pinned commit, so the dataset's own CC-BY-SA-4.0 "
            "is applied to the scoring code accompanying it; share-alike "
            "attaches to that one file, not the rest of the tree (see its "
            "header). Grading is offloaded to a worker process under "
            "GRADE_TIMEOUT, as for every sympy-backed grader here. FIDELITY: "
            "replaying both published 0-shot rows reproduces upstream's gold and "
            "pred 10552/10552 each, and its verdict on 10527/10552 (99.76%, "
            "61.19 -> 61.43 overall) for gpt-3.5-turbo and 10537/10552 (99.86%, "
            "85.58 -> 85.72) for gpt4 (v0 labels mapped). All 40 diffs are "
            "genuinely-equal fraction/decimal pairs upstream scored wrong "
            "because it pins sympy without the ANTLR runtime parse_latex needs, "
            "so its bare `except:` fell back to string equality; the "
            "integer-decimal-fraction cell moves most (62.32 -> 63.84). "
            "MAGNITUDES ARE NOT COMPARABLE across models: only 2 of upstream's "
            "28 published rows came from this 0-shot chat path, the other 26 "
            "being n-shot raw completions via general_model_inference.py. The "
            "transferable anchor is the shape -- critical thinking is the lowest "
            "cell in 26 of 26 published rows, reproduced by two live runs "
            "(gemma-3-27b-it full 10552: 84.11 overall / 63.08 that cell; "
            "gpt-oss-120b testmini: 80.83 / 47.67; Spearman rho +0.976 vs the "
            "GPT-4 row for both). Live scoring differential vs upstream's "
            "unmodified extract_ans.py: 13752/13752 on gold, pred and verdict. "
            "NOTE the critical-thinking cell is format-gated -- "
            "extract_pred_ans_none scores any response with no '####' marker as "
            "correct, so an empty response scores correct there; 80.9% of that "
            "cell was refusal-earned on the gemma run but only 1.4% on "
            "gpt-oss-120b, so read it with n_unextracted. Single-rollout only "
            "(upstream's cot_sc majority-votes 5 samples at temperature 0.7). "
            "Upstream decoding: temperature 0, top_p 1, max_tokens 512, no stop "
            "-- valid for a NON-REASONING model only, since a reasoning model "
            "spends 512 on hidden reasoning and returns empty content, which the "
            "gate scores correct on critical thinking and wrong elsewhere "
            "(paired control, only max_tokens differing: that cell 92.00 at 512 "
            "vs 64.00 at 4096, rank 8/8 -> 1/8, all seven numeric cells rising)."
        ),
    ),
)
class GSMPlusZeroShotGenTask(
    Task[
        GSMPlusDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": _user_turn(raw["question"])},
            ],
            reference=extract_gold_ans(raw["solution"]),
            extra={"perturbation_type": raw["perturbation_type"]},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        text = inf.texts[0] if inf.texts else ""
        prediction = extract_prediction(text, ctx.raw_sample["perturbation_type"])
        # `""` is upstream's "nothing extracted", `None` the protocol's spelling
        # of it. The string `"None"` survives untouched — on a `critical thinking`
        # row that is a real answer ("unanswerable"), not a failed extraction.
        return build_prediction_record([prediction or None])

    @override
    async def feedback(self, post, ctx):
        gold = extract_gold_ans(ctx.raw_sample["solution"])
        perturbation_type = ctx.raw_sample["perturbation_type"]
        # `.get`, not `[]`: a None `prediction` is dropped on write, so indexing
        # works fresh and raises KeyError on resume — and that case is routine
        # here, being every `critical thinking` row whose response carried no
        # refusal phrase (487 of 10552 on a real gemma-3-27b-it run). `or ""`
        # restores what upstream's `test_answer` compares against.
        prediction = post["rollouts"][0].get("prediction") or ""
        # Offloaded like every other sympy-backed grader here: `parse_latex` +
        # `simplify` costs 4 ms typical / 328 ms worst case over upstream's own
        # 10552-item dump, and `simplify` is reached with no bound of its own
        # (criterion 2 in `core/utils/offload.py`). Inline that is ~42 s of CPU
        # on the one event loop every runner in the session shares.
        try:
            correct = await run_cpu_bound(
                is_equivalent, gold, prediction, timeout=GRADE_TIMEOUT
            )
        except TimeoutError:
            # An ungradeable answer is a wrong answer, not a failed run — the
            # contract every sibling math grader keeps. Propagating would land
            # the sample in `fails`, which reads as infrastructure breakage.
            logger.warning(
                "Grading sample {} exceeded {}s and was scored wrong; the "
                "prediction is likely a shape `simplify` cannot bound.",
                ctx.sample_id,
                GRADE_TIMEOUT,
            )
            correct = False
        return True, build_judgement_record(
            gold,
            [build_rollout_judgement(0, bool(correct))],
            extra={"perturbation_type": perturbation_type},
        )

    @override
    async def report(self, finals, fails):
        # Accuracy over the full requested set (finals + fails), matching the
        # gsm8k/math-0shot-gen family and upstream's own denominator — every item
        # in the prediction file — so a pipeline failure counts as wrong.
        per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
        correct_num = 0
        for ctx in finals:
            perturbation_type = ctx.feedback_result["extra"]["perturbation_type"]
            if ctx.feedback_result["rollouts"][0]["correct"]:
                correct_num += 1
                per_type[perturbation_type][0] += 1
            per_type[perturbation_type][1] += 1
        untyped_fails = 0
        for ctx in fails:
            # A fail scores 0 but still owes its perturbation a denominator slot,
            # and `raw_sample` is the only place its type survives — a fail never
            # reached feedback. One that died before the sample was attached has
            # no type, so it can only reach the type-free denominators.
            if ctx.raw_sample is not None:
                per_type[ctx.raw_sample["perturbation_type"]][1] += 1
            else:
                untyped_fails += 1

        total = len(finals) + len(fails)
        accuracy = 100 * correct_num / total if total else 0.0
        report: dict[str, float | str] = {
            "score": accuracy,
            "accuracy": accuracy,
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }

        wo_correct = sum(c for t, (c, _) in per_type.items() if t != CRITICAL_THINKING)
        # `+ untyped_fails` so this follows the DENOMINATOR_REQUESTED declared
        # above, as `score` does: a fail with no `raw_sample` cannot be shown to
        # be one of the `critical thinking` rows `gsmplus_wo_ncr` excludes, so
        # charging it scores it wrong rather than dropping it. Omitting it let
        # this co-headline read 100.0 while `score` read 50.0 on the same run.
        wo_total = (
            sum(n for t, (_, n) in per_type.items() if t != CRITICAL_THINKING)
            + untyped_fails
        )
        report["score_wo_critical_thinking"] = (
            100 * wo_correct / wo_total if wo_total else 0.0
        )
        for perturbation_type, (correct, seen) in sorted(per_type.items()):
            report[f"score_{_metric_key(perturbation_type)}"] = (
                100 * correct / seen if seen else 0.0
            )
        report["fails"] = len(fails)
        # `n_unextracted` is the format gate's own signal: without it the gate's
        # failure mode — a reasoning model returning empty content, scored
        # correct on `critical thinking` — is invisible in report.json.
        report.update(health_metrics(finals))
        return report
