"""AGIEval 0-shot generative task (chat models), all 21 subsets in one class.

AGIEval's zero-shot protocol is **two model calls**, and reproducing its numbers
means running both:

1. the model answers the exam question (upstream's ``convert_zero_shot`` prompt,
   which ends in a "the answer is" cue);
2. that reply is fed back with a per-language extraction cue ("Therefore, among A
   through E, the answer is"), and the short reply *that* produces is what gets
   parsed.

Stage 2 exists because stage 1's answer is prose: upstream's zero-shot parser is
"first A-F character in the reply", which on a chain of thought would return the
letter of the first option it happened to restate. Dropping stage 2 does not make
this a cheaper AGIEval — it makes it a different, worse one.

Per-sample routing is by ``subset`` (four prompt families, three answer parsers,
three comparison rules); ``report()`` gives per-subset accuracy plus the macro
averages upstream's leaderboard publishes. Which subsets run is a *dataset* knob
(``subsets=`` / ``group="math"``), not a task knob — see
:mod:`sieval.datasets.agieval`.

``extractor`` is required, because stage 2's model *is* part of the measurement:
pass a model-config dict (or a Model) to pin a fixed, cheap extractor as upstream
did, or the literal ``"self"`` to extract with the model under test — a different
measurement, not a cheaper one. See ``reference_impl.notes``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import cast, override

from sieval.community.agieval.dataset_loader import (
    MATH_SUBSETS,
    SUBSETS,
    second_stage_prompt,
    zero_shot_prompt,
)
from sieval.community.agieval.evaluation import (
    LEADERBOARD_EN_MCQ_SUBSETS,
    LEADERBOARD_ZH_MCQ_SUBSETS,
    evaluate_single_sample,
)
from sieval.community.agieval.post_process import post_process
from sieval.core.models import ChatModel, Model, ModelOutput
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
from sieval.datasets import AGIEvalDatasetSample

# Upstream sends this on BOTH calls: it lives in
# openai_api.query_azure_openai_chat, which run_prediction.py routes stage 1 and
# stage 2 through, so it is part of the measured prompt rather than serving
# config — and the published leaderboard numbers were produced with it.
_SYSTEM_TURN = "You are a helpful AI assistant."

# Macro averages over named subset groups, reported as `macro_<name>`. Distinct
# from the per-subset `score_<subset>` keys — and `macro_math` must stay distinct
# from `score_math`, the MATH subset's own accuracy.
_MACRO_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("en_mcq", LEADERBOARD_EN_MCQ_SUBSETS),
    ("zh_mcq", LEADERBOARD_ZH_MCQ_SUBSETS),
    ("math", MATH_SUBSETS),
)


# The one non-model value `extractor` accepts: run stage 2 on the model under
# test. Not a default — stage 2 defines the score, so it is chosen explicitly.
_EXTRACTOR_SELF = "self"


def _reject_extra_rollouts(output: ModelOutput, who: str) -> None:
    """Backstop for the ``n=1`` both calls pass: only rollout 0 can be scored.

    Unreachable by configuration — a stray ``n`` in the model args or in
    ``infer_args`` loses the merge to the call site. It fires only if a backend
    ignores ``n``, which is not a sample-level problem to score around.
    """
    if len(output.texts) > 1:
        raise ValueError(
            f"AGIEval requested one rollout, but {who} returned "
            f"{len(output.texts)} — the backend ignored `n=1`. Upstream samples "
            "each problem once, so there is no published multi-rollout AGIEval "
            "to compare against, and only rollout 0 reaches post_process. "
            "Scoring rollout 0 anyway would silently pick one of several "
            "samples for every problem in the run."
        )


@sieval_task(
    name="agieval_0shot_gen",
    display_name="AGIEval (0-shot, generative)",
    description="AGIEval v1.1 human exams — 21 subsets, two-stage 0-shot MCQ + cloze.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "chinese", "multiple-choice", "open-ended"),
    model_type="chat",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="AGIEval",
        url="https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/src/dataset_loader.py",
        notes=(
            "Port of AGIEval's own zero-shot pipeline (Microsoft, arXiv:2304.06364) "
            "on the pinned v1.1 data: prompts from src/dataset_loader.py, answer "
            "extraction from src/post_process.py, verdicts from src/evaluation.py, "
            "all vendored in community/agieval. TWO-STAGE: run_prediction.py's "
            "zero-shot path generates freely, then re-prompts with the reply plus "
            "generate_second_stage_input's cue (with_format_prompt=False) and "
            "parses THAT; post_process_and_evaluation.py scores the second-stage "
            "output, keeping the first only as an audit field. Both calls are "
            "returned from infer() so both land in profile.json. REPEATS: upstream "
            "samples each problem once (no n_repeats); its only retry re-queries "
            "replies that came back empty, which sieval covers with the model's "
            "max_retries. An empty first-stage reply still proceeds to stage 2 "
            "with an empty answer, as upstream's extract_answer does.\n"
            "DECODING: upstream's query_azure_openai_chat sends temperature=0 "
            "and stop=['<|im_end|>'] on both calls; both calls here request "
            "n=1 explicitly (upstream has no n knob at all). temperature "
            "belongs to the model config in sieval, which "
            "examples/agieval-math.yaml sets; the stop token is deliberately "
            "not reproduced, being ChatML's own end marker, which an "
            "OpenAI-protocol endpoint consumes as a turn delimiter rather than "
            "emitting as text — sending it cannot change a reply this port is "
            "able to read.\n"
            "DIVERGENCE (the one protocol-level choice): upstream runs stage 2 on "
            "gpt-35-turbo no matter which model produced stage 1; sieval takes "
            "the `extractor` task arg (a model-config dict, a Model, or the "
            'literal "self") and REQUIRES it, as the LLM-judge tasks require '
            "`grader`: stage 2's model is part of the measurement, not serving "
            "config, so there is no default two runs could safely be compared "
            'across. A pinned extractor matches upstream; "self" measures the '
            "model's own answer-extraction, which for a weak model is not the "
            "same number. Whichever you choose, report it next to the score.\n"
            "SCORING: set-of-letters for jec-qa-kd / jec-qa-ca / gaokao-physics, "
            "math equivalence for math / gaokao-mathcloze, exact string compare "
            "otherwise. The math grader is AGIEval's own vendored hendrycks/math "
            "is_equiv, NOT sieval.community.math (a trimmed variant of the same "
            "file whose dropped normalizations change verdicts).\n"
            "UPSTREAM QUIRK, kept verbatim: 7 of 351 gaokao-mathqa rows carry "
            "multi-letter gold labels ('AD', 'ACD', 'A B D', ...) even in v1.1, but "
            "the subset is not on upstream's multi_choice list, so it is scored by "
            "exact single-letter compare and those 7 rows are unwinnable (~2% of "
            "that subset). Fixing it would silently diverge from every published "
            "AGIEval number.\n"
            "SECOND UPSTREAM QUIRK, and it caps macro_math: 73 of 118 "
            "gaokao-mathcloze golds are wrapped in $...$, parse_math_answer returns "
            "boxed content verbatim, and is_equiv's _strip_string removes an escaped "
            "\\$ but not a bare one. A model emitting the gold's VALUE in \\boxed{} "
            "— what models actually do — therefore tops out at 45/118 = 38.1% on that "
            "subset, while one reproducing the $...$ byte for byte scores 113/118 = "
            "95.8%: the ceiling is a formatting property, not an absolute. Read a real "
            "model against 38.1. (Both measured through upstream's own code; the 5 "
            "residual losses are golds containing '=', which the parser cuts down to "
            "the right-hand side.) A perfect, naturally-formatting model cannot exceed "
            "about 96.9 on `score` — or 87.2 on `macro_math`, the key this quirk "
            "actually caps, gaokao-mathcloze being 1 of that group's 5 subsets.\n"
            "THE EXTRACTOR DEFINES THE SCORE — MEASURED, not asserted, which is "
            "why the arg is required rather than defaulted. On identical "
            "stage-1 replies (315 rows, "
            "15/subset) the macro moved 52.70–78.73 across five extractors: a "
            "26-point spread, wider than the published GPT-4o → GPT-3.5-Turbo gap of "
            "16.3. Most of it is NOT extractor capability but this parser: "
            "find_first_capital_letter returns the first character in {A..F} ANYWHERE "
            'in the reply, and both cues carry a capital A before the answer ("Among '
            'A through E, the answer is", "从A到D, 我们应选择"), so an extractor '
            "that opens by restating the cue is scored 'A' whatever it then says. "
            "Measured on a full 7,272-row run: 635 of 6,154 MCQ rows (10.32%) were "
            "scored wrong while the extraction had NAMED the gold letter — worth "
            "+9.90 on the 19-subset MCQ macro, +5.40 on macro_en_mcq and +13.18 on "
            "macro_zh_mcq, and +25.16 on gaokao-english alone (68.30 as scored, 93.46 "
            "if the restated cue is read past, against GPT-4's published 91.9). For "
            "jec-qa-kd / jec-qa-ca / gaokao-physics the mechanism differs: "
            "parse_qa_multiple_answer returns EVERY A–F character, so a verbose reply "
            "yields a set that cannot match a single-letter v1.1 gold (letters per "
            "reply 1.73 → 7.09, accuracy 68.9 → 11.1 as replies lengthen). PIN AN "
            "EXTRACTOR THAT IS BOTH TERSE AND TEMPERATURE-CONTROLLABLE, and report "
            "which one next to the score: a terse extractor echoes the cue ~0% of the "
            "time and loses almost nothing, and an extractor that rejects "
            "temperature=0 (the gpt-5 family does) is not reproducible — re-running "
            "extraction alone over the same stage-1 replies moved macro_en_mcq 1.14 "
            "points. Never compare two runs whose extractors differ.\n"
            "COMPARISON TARGETS (upstream README leaderboard, v1.1 zero-shot, "
            "MCQ-only for the en/zh rows): GPT-4o 62.3 all / 65.2 en / 63.3 zh; "
            "GPT-3.5-Turbo 46.0 / 54.1 / 45.0. Per-subset targets are in "
            "arXiv:2304.06364 Table 2 (v1.0, zero-shot, TD-003 / ChatGPT / GPT-4); "
            "its printed GPT-4 average of 56.4 is exactly the unweighted mean of its "
            "21 per-subset rows, which independently confirms this reduction. `score` "
            "is the macro over the subsets that ran (upstream's 'average for all "
            "datasets', denominator 21 when all are selected); macro_en_mcq / "
            "macro_zh_mcq mirror the two leaderboard groups — note gaokao-english is "
            "prompted in English but counted as Chinese there, matching upstream's "
            "driver. Asterisked leaderboard rows are v1.0 and not comparable to this "
            "v1.1 data.\n"
            "VALIDATION (2026-08-12, all 7,272 rows): google/gemma-4-31B-it answering "
            "at temperature 0 with gpt-5.4-mini pinned as extractor — score 68.25, "
            "macro_en_mcq 80.19, macro_zh_mcq 62.94, macro_math 70.76; 0 request "
            "errors, 12 stage-1 truncations, 0 empty stage-1 replies, 99 of 7,272 "
            "unparsed. No subset exceeded its replay ceiling. Shape vs Table 2 "
            "Spearman +0.49, rising to +0.66 once the cue-echo rows above are read "
            "past — the artifact, not the port, was distorting the per-subset "
            "ordering. Checked that a strong extractor transcribes rather than "
            "re-solves: on rows where stage 1 never named the gold letter it scores "
            "18.75%, below chance. This is an OPERATING POINT, not a reproduction: "
            "upstream's only zero-shot rows are GPT-4o and GPT-3.5-Turbo and neither "
            "is servable here, and the extractor cannot be gpt-35-turbo — so "
            'status stays "experimental".'
        ),
    ),
)
class AGIEvalZeroShotGenTask(
    Task[
        AGIEvalDatasetSample,
        PromptRecord,
        list[ModelOutput],
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key` / `denominator_policy`,
        # which name a column and a population rather than measuring one.
        dict[str, float | str],
    ]
):
    """AGIEval zero-shot: answer, extract, score — routed per subset."""

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        extractor: Mapping | Model | str | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._extractor = self._build_extractor(extractor, model)

    @staticmethod
    def _build_extractor(
        extractor: Mapping | Model | str | None, model: Model
    ) -> Model:
        """Resolve the ``extractor`` task arg into the model for stage 2.

        Stage 2 is where the score is decided — the macro moved 26 points across
        five extractors on identical stage-1 replies — so this has no safe
        default and ``None`` raises, the same way the LLM-judge tasks' ``grader``
        does. A Mapping is the YAML path, e.g.
        ``{model: gpt-3.5-turbo, api_base: ..., temperature: 0}``; the literal
        ``"self"`` opts into extracting with the model under test.
        """
        if isinstance(extractor, Model):
            return extractor
        # str first, and as an `elif` chain, so the Mapping branch narrows to a
        # real mapping: any other string is a typo, not a configuration, and
        # falls through to the error below rather than silently self-extracting.
        if isinstance(extractor, str):
            if extractor == _EXTRACTOR_SELF:
                return model
        elif isinstance(extractor, Mapping):
            return ChatModel(**extractor)
        raise ValueError(
            "AGIEval requires an `extractor` in the task args: stage 2's model "
            "decides the score (a 26-point macro spread across five extractors "
            "on identical stage-1 replies), so no default is comparable across "
            "runs. Pass a model-config dict such as {model: gpt-3.5-turbo, "
            "api_base: ..., temperature: 0} to match upstream, or the literal "
            f'"self" to extract with the model under test. Got {extractor!r}.'
        )

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {"role": "system", "content": _SYSTEM_TURN},
                {"role": "user", "content": zero_shot_prompt(raw["subset"], raw)},
            ],
            # Upstream's load_dataset_as_result_schema: the label when set, the
            # cloze answer otherwise.
            reference=raw["label"] if raw["label"] else raw["answer"],
            extra={"subset": raw["subset"]},
        )

    @override
    async def infer(self, pre, ctx):
        """Answer, then re-read the answer under the extraction cue.

        Returns both calls: the runner profiles a ``list[ModelOutput]`` stage
        value directly, so stage 2's spend needs no ``grader_output`` routing,
        and stage 1's reply — the actual reasoning — stays on disk as evidence.
        """
        subset = pre["extra"]["subset"]
        # `n=1` explicitly, as every task taking an `n` passes its own: call-time
        # kwargs win the `{**self._kwargs, **kwargs}` merge in the model layer,
        # so a stray `n` in the model args or in `infer_args` — meaningless here,
        # since upstream samples each problem once and only rollout 0 reaches
        # post_process — is pinned back to 1 instead of billing n x 2 calls per
        # problem. Checked after the call in case the backend ignored it.
        first = await self.model.agenerate(pre["prompt"], n=1)
        _reject_extra_rollouts(first, "the model under test")
        # An empty reply is carried into stage 2 as "", matching upstream's
        # extract_answer, rather than failing the sample.
        answer = first.texts[0] if first.texts else ""
        # The stage-1 QUESTION, selected by role: the prompt also carries the
        # system turn, and indexing [0] would splice that into stage 2's context.
        messages = cast(list[dict[str, str]], pre["prompt"])
        context = next(m["content"] for m in messages if m["role"] == "user")
        second = await self._extractor.agenerate(
            [
                {"role": "system", "content": _SYSTEM_TURN},
                {
                    "role": "user",
                    "content": second_stage_prompt(subset, context, answer),
                },
            ],
            n=1,
        )
        _reject_extra_rollouts(second, "the extractor")
        return [first, second]

    @override
    async def postprocess(self, inf, ctx):
        extraction = inf[-1]
        text = extraction.texts[0] if extraction.texts else ""
        return build_prediction_record([post_process(ctx.raw_sample["subset"], text)])

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        subset = raw["subset"]
        reference = raw["label"] if raw["label"] else raw["answer"]
        # `.get`, not `[...]`: `prediction` is NotRequired and post_process returns
        # None whenever extraction fails — routine on the cloze subsets — so the key
        # is absent from the record on disk and indexing it raises KeyError on
        # resume, never on a fresh run.
        prediction = post["rollouts"][0].get("prediction")
        correct = evaluate_single_sample(subset, prediction, reference)
        return True, build_judgement_record(
            reference,
            [build_rollout_judgement(0, correct)],
            extra={"subset": subset},
        )

    @override
    async def report(self, finals, fails):
        """Per-subset accuracy, macro over subsets, and the leaderboard macros.

        ``score`` is the unweighted mean of the per-subset accuracies — upstream's
        "average for all datasets" — over the subsets that ran, so a subset
        selection reports the macro of *that* selection. A ``macro_<group>`` key
        appears only when **every** member of the group was evaluated: these keys
        exist to be compared against published numbers, and a partial macro is
        not that number. Infra failures are reported in ``fails``, not scored 0.
        """
        by_subset: dict[str, list[bool]] = defaultdict(list)
        for ctx in finals:
            fb = ctx.feedback_result
            by_subset[fb["extra"]["subset"]].append(fb["rollouts"][0]["correct"])

        subset_acc = {
            subset: 100 * sum(correct) / len(correct)
            for subset, correct in by_subset.items()
        }
        overall = sum(subset_acc.values()) / len(subset_acc) if subset_acc else 0.0

        metrics: dict[str, float | str] = {
            "score": overall,
            "fails": float(len(fails)),
        }
        # Canonical subset order, so two runs' reports line up key for key.
        for subset in SUBSETS:
            if subset in subset_acc:
                metrics[f"score_{subset.replace('-', '_')}"] = subset_acc[subset]
        for group_name, group in _MACRO_GROUPS:
            if all(subset in subset_acc for subset in group):
                metrics[f"macro_{group_name}"] = sum(
                    subset_acc[subset] for subset in group
                ) / len(group)
        metrics[SCORE_KEY_FIELD] = "score"
        # Infra failures land in `fails` and are excluded from every per-subset
        # accuracy, so the headline is averaged over the samples that were judged.
        metrics[DENOMINATOR_FIELD] = DENOMINATOR_JUDGED
        return metrics
