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
this a cheaper AGIEval — it makes it a different, worse one. *Which* model runs
it is the required ``extractor`` arg: it moves the score by more than the
published GPT-4o → GPT-3.5-Turbo gap, so there is no default. See
``reference_impl.notes``.

Per-sample routing is by ``subset`` (four prompt families, three answer parsers,
three comparison rules); ``report()`` gives per-subset accuracy plus the macro
averages upstream's leaderboard publishes. Which subsets run is a *dataset* knob
(``subsets=`` / ``group="math"``), not a task knob — see
:mod:`sieval.datasets.agieval`.

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
    InputKind,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RequirementContext,
    Task,
    TaskModelRequirement,
    TaskRequirements,
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

# Upstream sends this on BOTH calls (openai_api.query_azure_openai_chat, which
# run_prediction.py routes both stages through), so it is part of the measured
# prompt: the published leaderboard numbers were produced with it.
_SYSTEM_TURN = "You are a helpful AI assistant."

# Reported as `macro_<name>`, distinct from the per-subset `score_<subset>` keys —
# `macro_math` must stay distinct from `score_math`, the MATH subset's own score.
_MACRO_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("en_mcq", LEADERBOARD_EN_MCQ_SUBSETS),
    ("zh_mcq", LEADERBOARD_ZH_MCQ_SUBSETS),
    ("math", MATH_SUBSETS),
)


# The one non-model value `extractor` accepts: run stage 2 on the model under test.
_EXTRACTOR_SELF = "self"


def _reject_extra_rollouts(output: ModelOutput, who: str) -> None:
    """Backstop for the ``n=1`` both calls pass: only rollout 0 can be scored.

    Unreachable by configuration — a stray ``n`` loses the kwargs merge to the
    call site — so it fires only if a backend ignored ``n``.
    """
    if len(output.texts) > 1:
        raise ValueError(
            f"AGIEval requested one rollout, but {who} returned "
            f"{len(output.texts)} — the backend ignored `n=1`. Upstream samples "
            "each problem once and only rollout 0 reaches post_process, so "
            "scoring it anyway would silently pick one of several samples for "
            "every problem in the run."
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
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="AGIEval",
        url="https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/src/dataset_loader.py",
        notes=(
            "Port of AGIEval's own zero-shot pipeline (Microsoft, arXiv:2304.06364) "
            "on the pinned v1.1 data: prompts from src/dataset_loader.py, answer "
            "extraction from src/post_process.py, verdicts from src/evaluation.py, "
            "all vendored in community/agieval.\n"
            "TWO-STAGE, as run_prediction.py's zero-shot path: generate freely, "
            "re-prompt with the reply plus generate_second_stage_input's cue "
            "(with_format_prompt=False), and parse THAT — post_process_and_"
            "evaluation.py scores the second-stage output and keeps the first only "
            "as an audit field. Both calls are returned from infer(), so both land "
            "in profile.json. REPEATS: upstream samples each problem once (no "
            "n_repeats); its only retry re-queries empty replies, which the model's "
            "max_retries covers. An empty stage-1 reply still proceeds, as "
            "upstream's extract_answer does.\n"
            "DECODING: upstream's query_azure_openai_chat sends temperature=0 and "
            "stop=['<|im_end|>'] on both calls. Here both request n=1 explicitly "
            "(upstream has no n knob at all) and temperature comes from the model "
            "config, as examples/agieval-math.yaml sets it. The stop token is "
            "deliberately not reproduced: an OpenAI-protocol endpoint consumes "
            "ChatML's own end marker as a turn delimiter rather than emitting it "
            "as text.\n"
            "DIVERGENCE (the one protocol-level choice): upstream runs stage 2 on "
            "gpt-35-turbo no matter which model produced stage 1; sieval requires "
            "an explicit `extractor` instead, as the LLM-judge tasks require "
            '`grader`. A pinned extractor matches upstream; "self" measures the '
            "model's own answer-extraction, which for a weak model is not the same "
            "number.\n"
            "SCORING: set-of-letters for jec-qa-kd / jec-qa-ca / gaokao-physics, "
            "math equivalence for math / gaokao-mathcloze, exact string compare "
            "otherwise. The math grader is AGIEval's own vendored hendrycks/math "
            "is_equiv, NOT sieval.community.math (a trimmed variant of the same "
            "file whose dropped normalizations change verdicts).\n"
            "TWO UPSTREAM QUIRKS, both kept verbatim — fixing either would "
            "silently diverge from every published AGIEval number. (1) 7 of 351 "
            "gaokao-mathqa rows carry multi-letter golds ('AD', 'A B D', ...) even "
            "in v1.1, but the subset is not on upstream's multi_choice list, so "
            "exact single-letter compare makes them unwinnable (~2% of that "
            "subset). (2) 73 of 118 gaokao-mathcloze golds are wrapped in $...$, "
            "parse_math_answer returns boxed content verbatim, and is_equiv's "
            "_strip_string removes an escaped \\$ but not a bare one — so a model "
            "emitting the gold's VALUE in \\boxed{}, which is what models do, tops "
            "out at 45/118 = 38.1% there, while one reproducing the $...$ byte for "
            "byte scores 113/118 = 95.8% (both measured through upstream's own "
            "code; the 5 residual losses are golds containing '='). Read a real "
            "model against 38.1 — the ceiling is a formatting property — and note "
            "that it caps even a naturally-formatting model at about 96.9 on "
            "`score` and 87.2 on `macro_math`, the key it really binds.\n"
            "THE EXTRACTOR DEFINES THE SCORE — measured, not asserted, which is why "
            "the arg has no default. On identical stage-1 replies (315 rows, "
            "15/subset) the macro moved 52.70–78.73 across five extractors: a "
            "26-point spread, wider than the published GPT-4o → GPT-3.5-Turbo gap "
            "of 16.3. Most of that is the parser, not extractor capability: "
            "find_first_capital_letter returns the first {A..F} character ANYWHERE "
            'in the reply, and both cues carry a capital A before the answer ("Among '
            'A through E, the answer is", "从A到D, 我们应选择"), so an extractor '
            "that opens by restating the cue is scored 'A' whatever it then says. On "
            "a full 7,272-row run it mis-scored 635 of 6,154 MCQ rows (10.32%) whose "
            "extraction had NAMED the gold letter — worth +9.90 on the 19-subset MCQ "
            "macro, +5.40 on macro_en_mcq, +13.18 on macro_zh_mcq, +25.16 on "
            "gaokao-english alone (68.30 as scored, 93.46 read past the cue, against "
            "GPT-4's published 91.9). On jec-qa-kd / jec-qa-ca / gaokao-physics the "
            "mechanism differs: parse_qa_multiple_answer returns EVERY A–F "
            "character, so a verbose reply yields a set no single-letter v1.1 gold "
            "can match (letters per reply 1.73 → 7.09, accuracy 68.9 → 11.1 as "
            "replies lengthen). So PIN AN EXTRACTOR THAT IS TERSE AND "
            "TEMPERATURE-CONTROLLABLE, and report which one next to the score: a "
            "terse one echoes the cue ~0% of the time, while one that rejects "
            "temperature=0 (the gpt-5 family does) is not reproducible — "
            "re-extracting over the same stage-1 replies moved macro_en_mcq 1.14 "
            "points. Never compare two runs whose extractors differ.\n"
            "COMPARISON TARGETS (upstream README leaderboard, v1.1 zero-shot, "
            "MCQ-only for the en/zh rows): GPT-4o 62.3 all / 65.2 en / 63.3 zh; "
            "GPT-3.5-Turbo 46.0 / 54.1 / 45.0; asterisked rows there are v1.0 and "
            "not comparable. Per-subset targets are in arXiv:2304.06364 Table 2 "
            "(v1.0, zero-shot, TD-003 / ChatGPT / GPT-4), whose printed GPT-4 "
            "average of 56.4 is exactly the unweighted mean of its 21 rows — which "
            "confirms `score` is upstream's 'average for all datasets' (denominator "
            "21 when all are selected). macro_en_mcq / macro_zh_mcq mirror the two "
            "leaderboard groups; gaokao-english is prompted in English but counted "
            "as Chinese there, matching upstream's driver.\n"
            "VALIDATION (2026-08-12, all 7,272 rows): google/gemma-4-31B-it "
            "answering at temperature 0 with gpt-5.4-mini pinned as extractor — "
            "score 68.25, macro_en_mcq 80.19, macro_zh_mcq 62.94, macro_math 70.76; "
            "0 request errors, 12 stage-1 truncations, 0 empty stage-1 replies, 99 "
            "unparsed, no subset above its replay ceiling. Shape vs Table 2 Spearman "
            "+0.49, rising to +0.66 once the cue-echo rows above are read past — the "
            "artifact, not the port, was distorting the per-subset ordering. The "
            "extractor transcribes rather than re-solves: on rows where stage 1 "
            "never named the gold letter it scores 18.75%, below chance. An "
            "OPERATING POINT, not a reproduction — upstream's only zero-shot rows "
            "are GPT-4o and GPT-3.5-Turbo, neither servable here, and the extractor "
            'cannot be gpt-35-turbo — so status stays "experimental".'
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

    @classmethod
    @override
    def model_requirements_for(
        cls, context: RequirementContext
    ) -> tuple[TaskModelRequirement, ...]:
        candidate = super().model_requirements_for(context)
        if context.task_args.get("extractor") == _EXTRACTOR_SELF:
            return candidate
        extractor = cls._bind_role_requirement(
            context,
            "extractor",
            TaskRequirements(input=InputKind.CHAT),
        )
        return candidate + extractor

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        extractor: Mapping | Model | str | None = None,
        models_by_role: Mapping[str, Model] | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._extractor = self._resolve_role_model(
            "extractor",
            extractor,
            models_by_role,
            build=lambda cfg: self._build_extractor(cfg, model),
        )

    @staticmethod
    def _build_extractor(
        extractor: Mapping | Model | str | None, model: Model
    ) -> Model:
        """Resolve the ``extractor`` task arg into the model for stage 2.

        No safe default, so ``None`` raises, the same way the LLM-judge tasks'
        ``grader`` does. A Mapping is the YAML path; the literal ``"self"`` opts
        into extracting with the model under test.
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
        # `n=1` explicitly: call-time kwargs win the `{**self._kwargs, **kwargs}`
        # merge in the model layer, so a stray `n` in the model args or in
        # `infer_args` — meaningless here — is pinned back to 1 instead of billing
        # n x 2 calls per problem. Checked after, in case the backend ignored it.
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
        # `.get`, not `[...]`: post_process returns None whenever extraction fails
        # — routine on the cloze subsets — so the key is absent from the record on
        # disk and indexing it would raise on resume, never on a fresh run.
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
        exist to be compared against published numbers, and a partial macro is not
        that number.
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
