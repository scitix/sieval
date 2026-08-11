"""
GSM-Plus 0-shot generative task, aligned with the upstream GSM-Plus evaluation.

Port of GSM-Plus's zero-shot CoT (chat/instruct) path (pinned commit
``3474129e``, ``scripts/openai_model_inference.py`` with ``--prompt_type cot``):

* Prompt (``prompt_template.py::cot_prompt_map_func``): a system turn carrying
  the ``#### [value]`` output contract, then a user turn
  ``"Question:\\n{question}\\nAnswer:\\nLet's think step by step."``. The serving
  backend applies the model's own chat template.
* Gold answer (``extract_ans.py::extract_gold_ans``, fed ``solution`` — upstream's
  ``get_gsmplus`` puts ``item["solution"]`` in its ``answers`` list, not
  ``item["answer"]``): the last ``####`` segment, first number, commas stripped.
* Answer extraction: dispatched on the row's ``perturbation_type``
  (``extract_ans.py::test_answer``). Seven perturbations use
  ``extract_pred_ans``; ``critical thinking`` uses ``extract_pred_ans_none``,
  because that perturbation deletes a quantity the question needs, making the
  gold answer the string ``"None"`` — "unanswerable" — rather than a number.
* Scoring: ``normalize_final_answer`` then ``check_sympy_equivalence``.
* ``score`` is accuracy over the whole set. ``report`` also breaks accuracy down
  per perturbation type and reports ``score_wo_critical_thinking``, upstream's
  ``gsmplus_wo_ncr`` — the paper leads with both, since ``critical thinking`` is
  the one cell where a refusal, not a number, is the right answer.

Extraction/scoring lives in ``sieval.community.gsm_plus``, ported from the pinned
commit's ``scripts/utils/extract_ans.py``.

Fidelity, measured — replaying upstream's own stored zero-shot-CoT predictions
(``results/gpt-3.5-turbo.json``, all 10552 items) through this pipeline
reproduces upstream's persisted ``gold`` and ``pred`` on **10552/10552** items,
and its ``result`` on **10527/10552 (99.76%)**.

All 25 verdict diffs go one way — upstream ``False``, this port ``True`` — and
every one is a pair that is genuinely equal (``3/1`` vs ``3``, ``7/20`` vs
``0.35``, ``2.45`` vs ``2450/1000``). The cause is upstream's environment, not
its logic: ``requirements.txt`` pins ``sympy==1.12`` and no
``antlr4-python3-runtime``, so ``parse_latex`` raises, and
``check_sympy_equivalence``'s bare ``except:`` degrades it to string equality.
Under sieval's ``[math]`` extra the ANTLR runtime *is* pinned, so the same
vendored code reaches its symbolic branch and returns the mathematically correct
verdict. Reproducing the published digits exactly would mean deliberately
breaking ``parse_latex`` — bespoke logic that diverges from upstream source — so
the port keeps the code faithful and accepts the documented gap.

That gap, this port vs. the paper's published GPT-3.5-Turbo CoT numbers:

* overall 61.43 vs 61.19; excluding critical thinking 63.45 vs 63.18
* unchanged: critical thinking 47.31, adding operation 48.45, distraction
  insertion 62.17, problem understanding 74.22
* higher: integer-decimal-fraction conversion 63.84 vs 62.32, reversing
  operation 55.42 vs 55.19, numerical substitution 69.60 vs 69.52, digit
  expansion 70.43 vs 70.36

20 of the 25 land on ``integer-decimal-fraction conversion``, which is what makes
the delta explainable rather than mysterious: that perturbation exists precisely
to rewrite integers as decimals and fractions, so it is where string equality and
symbolic equality disagree most.

The same reduction holds on a **second** published 0-shot model, which is what
makes it an explanation rather than a fitted excuse: replaying
``results/gpt4.json`` (10552 items, after mapping its v0 perturbation labels —
``necessary constraint removal`` is ``critical thinking``, the ``ncr`` of
``gsmplus_wo_ncr``) reproduces ``gold`` and ``pred`` on 10552/10552 and the
verdict on 10537/10552 (99.86%). All 15 diffs are again one-directional and
again genuinely-equal pairs, 11 of them on ``integer-decimal-fraction
conversion``; overall 85.72 vs published 85.58, and 6 of 8 cells land exactly.

**Why ``status="experimental"`` remains, and what it is no longer for.** It is
now there for **one** reason only: upstream's code repo states no license (see
``sieval.community.gsm_plus``'s header), so the redistribution question is open.
It is *not* for the protocol and no longer for the absence of a live run:

* A published-magnitude anchor is unreachable by construction, not by
  omission. Upstream has two inference paths, and only ``gpt-3.5-turbo`` and
  ``gpt4`` — 2 of its 28 published rows — came from the 0-shot chat path ported
  here; the other 26 came from ``general_model_inference.py``, an **n-shot raw
  completion** path (``gsm8k_nshot_prompt``, no chat template). A 0-shot score
  is not a defect against an 8-shot row.
* What does transfer is the **shape**: ``critical thinking`` is the lowest of the
  eight cells in **26 of 26** published rows carrying all of them. Two live runs
  reproduce it — ``google/gemma-3-27b-it`` over the full 10552 (overall 84.11,
  excl. critical thinking 87.11, ``critical thinking`` 63.08, rank 1/8) and
  ``openai/gpt-oss-120b`` over ``testmini`` (80.83 / 85.57 / 47.67, rank 1/8).
  Spearman rho against the GPT-4 row is +0.976 for both, and GPT-4 is the closest
  of all 26 published rows to each — one of the two rows this very protocol
  produced.
* Scoring was also differentially checked on live text, which no stored dump can
  cover: every rollout of those runs plus two controls (13,752 generations) was
  re-scored by upstream's unmodified ``extract_ans.py``
  (``test_answer(prompt_type="cot", mv=1)``), agreeing on ``gold``, ``pred`` and
  verdict on 13752/13752.

**The ``critical thinking`` cell is format-gated, and that is upstream's design,
ported deliberately.** ``extract_pred_ans_none`` scores any response with no
``####`` marker as ``"None"`` — i.e. correct — so an empty response, a
``\\boxed{}`` answer, or a plain-prose number all score correct on those rows,
while the same text is graded normally on the other seven perturbations. Measured
share of that cell's credit that was *earned* by a refusal phrase rather than
*granted* by the missing marker: 80.9% on the gemma run (673 earned / 159
granted), but only 1.4% on gpt-oss-120b, whose visible ``content`` is close to a
bare answer. A right-looking shape therefore does not establish that this cell
measured refusal — read it together with the earned/granted split.

**The repro decoding below is only valid for a non-reasoning model.** A reasoning
model spends ``max_tokens=512`` on hidden reasoning and returns empty content,
which the format gate then scores *correct* on ``critical thinking`` and wrong
everywhere else — inverting the benchmark exactly where the paper's headline
sits. Measured on a paired control (same model, same 400 items, only
``max_tokens`` differing): at 512 ``critical thinking`` is 92.00 and the *easiest*
of the eight cells (rank 8/8, Spearman rho +0.000 vs GPT-4); at 4096 it is 64.00
and the hardest (rank 1/8), while all seven numeric cells rise. Six of the ten
models reachable over this environment's OpenAI-protocol gateway return empty
content at 512, so this is the common case, not a corner.

Other deviations from upstream (documented, not silent):

* Only ``--prompt_type cot`` is ported. Upstream also ships ``pot``, ``complex``,
  ``contrastive``, ``ltm`` and ``cot_sc``; those are separate prompting
  techniques (the paper's Table 5), not this protocol.
* Single rollout. Upstream's ``cot_sc`` draws 5 samples at temperature 0.7 and
  majority-votes (``test_answer(mv=5)``); the ported path is ``mv == 1``.
* Upstream's ``results/*.json`` dumps spell the 7th perturbation "distractor
  insertion", while both the released dataset and upstream's own
  ``dataset/gsmplus_test.jsonl`` spell it "distraction insertion". A run sees the
  dataset spelling, so that is what the report keys use.
* Upstream also reports a GSM8K-vs-GSM-Plus confusion matrix and performance
  decay rate (``report_metrics.py::get_confusion_matrix``, ``pdr``). Both need a
  paired GSM8K run, so they belong to cross-task analysis, not this task's
  report; the ``seed_*`` columns needed to compute them are preserved on every
  sample.

Repro decoding (model-layer assets — set via ``models:`` / ``infer_args``, not in
this code): greedy ``temperature=0``, ``top_p=1.0``, ``max_tokens=512``, no stop
sequences (``extract_ans.py::invoke_openai`` defaults for non-``pot`` prompts).

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from collections import defaultdict
from typing import override

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
)
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
    status="experimental",
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
            "normalize_final_answer + check_sympy_equivalence scoring. All "
            "vendored in sieval.community.gsm_plus. UPSTREAM CODE STATES NO "
            "LICENSE (no LICENSE/NOTICE/SPDX at the pinned commit, GitHub API "
            "reports license: null); absence is not a grant, so the "
            "redistribution question is open -- see the header of "
            "sieval.community.gsm_plus. The dataset is separately CC-BY-SA-4.0 "
            "and is referenced, not redistributed. Replaying upstream's stored "
            "GPT-3.5-Turbo CoT predictions reproduces its gold/pred on "
            "10552/10552 items and its verdict on 10527/10552 (99.76%); the 25 "
            "diffs are all genuinely-equal fraction/decimal pairs that upstream "
            "scored wrong because it pins sympy without the ANTLR runtime "
            "parse_latex needs, so its bare `except:` fell back to string "
            "equality (published 61.19 overall -> 61.43 here; the "
            "integer-decimal-fraction cell moves most, 62.32 -> 63.84). The same "
            "reduction fits the second published 0-shot row: results/gpt4.json "
            "(v0 labels mapped) reproduces gold/pred 10552/10552 and verdict "
            "10537/10552 (99.86%), 85.58 -> 85.72, 6 of 8 cells exact. Only 2 of "
            "upstream's 28 published rows came from this 0-shot chat path (the "
            "other 26 are n-shot raw completions via general_model_inference.py), "
            "so magnitudes are not comparable across models; the transferable "
            "anchor is the shape -- critical thinking is the lowest cell in 26 of "
            "26 published rows, and two live runs reproduce it (gemma-3-27b-it "
            "full 10552: 84.11 overall / 63.08 critical thinking; gpt-oss-120b "
            "testmini: 80.83 / 47.67; Spearman rho +0.976 vs the GPT-4 row for "
            "both). Live scoring differential vs upstream's unmodified "
            "extract_ans.py: 13752/13752 on gold, pred and verdict. NOTE the "
            "critical-thinking cell is format-gated -- extract_pred_ans_none "
            "scores any response with no '####' marker as correct, so an empty "
            "response scores correct there; 80.9% of that cell was refusal-earned "
            "on the gemma run but only 1.4% on gpt-oss-120b. Single-rollout only: "
            "upstream's cot_sc majority-votes 5 samples at temperature 0.7 "
            "(test_answer mv=5). Upstream decoding for this path: temperature 0, "
            "top_p 1, max_tokens 512, no stop -- valid for a NON-REASONING model "
            "only: a reasoning model spends 512 on hidden reasoning and returns "
            "empty content, which the format gate scores correct on critical "
            "thinking and wrong elsewhere (paired control, only max_tokens "
            "differing: critical thinking 92.00 at 512 vs 64.00 at 4096, rank "
            "8/8 -> 1/8, all seven numeric cells rising)."
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
        # `""` is upstream's "nothing extracted"; `None` is the protocol's spelling
        # of that, and feedback restores `""` for the grader. The string `"None"`
        # survives untouched — on a `critical thinking` row it is a real answer
        # ("unanswerable"), not a failure to extract.
        return build_prediction_record([prediction or None])

    @override
    async def feedback(self, post, ctx):
        gold = extract_gold_ans(ctx.raw_sample["solution"])
        perturbation_type = ctx.raw_sample["perturbation_type"]
        # `.get` because `prediction` is NotRequired and omitted on write when it
        # was None, so indexing it works on the fresh path and raises KeyError on
        # the resumed one. That case is routine here, not exotic: it is every
        # `critical thinking` row whose response carried no refusal phrase — 487
        # of 10552 on a real gemma-3-27b-it run. `or ""` then restores exactly
        # what upstream's test_answer compares against.
        prediction = post["rollouts"][0].get("prediction") or ""
        correct = is_equivalent(gold, prediction)
        return True, build_judgement_record(
            gold,
            [build_rollout_judgement(0, correct)],
            extra={"perturbation_type": perturbation_type},
        )

    @override
    async def report(self, finals, fails):
        # Accuracy over the full requested set (finals + fails), matching the
        # gsm8k/math-0shot-gen family and upstream's own denominator (every item
        # in the prediction file): a pipeline failure counts as wrong, not as an
        # excluded sample.
        per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
        correct_num = 0
        for ctx in finals:
            perturbation_type = ctx.feedback_result["extra"]["perturbation_type"]
            if ctx.feedback_result["rollouts"][0]["correct"]:
                correct_num += 1
                per_type[perturbation_type][0] += 1
            per_type[perturbation_type][1] += 1
        for ctx in fails:
            # A failed sample scores 0 but still owes its perturbation a
            # denominator slot, and `raw_sample` is the only place its type
            # survives — a fail never reached feedback. Contexts that failed
            # before the sample was attached are counted in `total` only, so
            # per-type denominators can sum to less than it.
            if ctx.raw_sample is not None:
                per_type[ctx.raw_sample["perturbation_type"]][1] += 1

        total = len(finals) + len(fails)
        accuracy = 100 * correct_num / total if total else 0.0
        report: dict[str, float | str] = {
            "score": accuracy,
            "accuracy": accuracy,
            SCORE_KEY_FIELD: "accuracy",
            # `requested`, per the denominator comment above: a pipeline failure
            # is counted as wrong rather than excluded.
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }

        wo_correct = sum(c for t, (c, _) in per_type.items() if t != CRITICAL_THINKING)
        wo_total = sum(n for t, (_, n) in per_type.items() if t != CRITICAL_THINKING)
        report["score_wo_critical_thinking"] = (
            100 * wo_correct / wo_total if wo_total else 0.0
        )
        for perturbation_type, (correct, seen) in sorted(per_type.items()):
            report[f"score_{_metric_key(perturbation_type)}"] = (
                100 * correct / seen if seen else 0.0
            )
        report["fails"] = len(fails)
        return report
