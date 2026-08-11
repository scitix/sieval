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

Both stages default to the model under test. Pass ``extractor`` (a model-config
dict or a Model) to run stage 2 on a fixed, cheap model instead, which is what
upstream did — see ``reference_impl.notes``.

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
            "DIVERGENCE (the one protocol-level choice): upstream runs stage 2 on "
            "gpt-35-turbo no matter which model produced stage 1; sieval defaults "
            "stage 2 to the model under test and takes the `extractor` task arg "
            "(model-config dict or Model) to pin a separate one. Matching upstream "
            "exactly means passing an extractor; leaving it unset measures the "
            "model's own answer-extraction, which for a weak model is not the same "
            "number. Pin whichever you choose — the score depends on it.\n"
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
            "COMPARISON TARGETS (upstream README leaderboard, v1.1 zero-shot, "
            "MCQ-only for the en/zh rows): GPT-4o 62.3 all / 65.2 en / 63.3 zh; "
            "GPT-3.5-Turbo 46.0 / 54.1 / 45.0. `score` is the macro over the "
            "subsets that ran (upstream's 'average for all datasets', denominator "
            "21 when all are selected); macro_en_mcq / macro_zh_mcq mirror the two "
            "leaderboard groups — note gaokao-english is prompted in English but "
            "counted as Chinese there, matching upstream's driver. Asterisked "
            "leaderboard rows are v1.0 and not comparable to this v1.1 data. "
            "NOT YET VALIDATED against a run of our own, hence "
            'status="experimental".'
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
        dict[str, float],
    ]
):
    """AGIEval zero-shot: answer, extract, score — routed per subset."""

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        extractor: Mapping | Model | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._extractor = self._build_extractor(extractor, model)

    @staticmethod
    def _build_extractor(extractor: Mapping | Model | None, model: Model) -> Model:
        """Resolve the ``extractor`` task arg into the model for stage 2.

        ``None`` reuses the model under test (see the divergence note in
        ``reference_impl.notes``); a Mapping is the YAML path, e.g.
        ``{model: gpt-3.5-turbo, api_base: ..., temperature: 0}``.
        """
        if extractor is None:
            return model
        if isinstance(extractor, Model):
            return extractor
        if isinstance(extractor, Mapping):
            return ChatModel(**extractor)
        raise ValueError(
            "AGIEval `extractor` must be a model-config dict or a Model, got "
            f"{type(extractor).__name__}. Omit it to run answer extraction on "
            "the model under test."
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
        first = await self.model.agenerate(pre["prompt"])
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
            ]
        )
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
        prediction = post["rollouts"][0]["prediction"]
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

        metrics: dict[str, float] = {"score": overall, "fails": float(len(fails))}
        # Canonical subset order, so two runs' reports line up key for key.
        for subset in SUBSETS:
            if subset in subset_acc:
                metrics[f"score_{subset.replace('-', '_')}"] = subset_acc[subset]
        for group_name, group in _MACRO_GROUPS:
            if all(subset in subset_acc for subset in group):
                metrics[f"macro_{group_name}"] = sum(
                    subset_acc[subset] for subset in group
                ) / len(group)
        return metrics
