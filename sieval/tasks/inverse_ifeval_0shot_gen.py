"""Inverse IFEval — 0-shot generative, LLM-judge graded.

Port of Inverse IFEval (M-A-P; Zhang et al., 2025, arXiv:2509.04292): the model
answers a prompt whose instructions deliberately contradict the conventions it
absorbed during SFT (answer correctly, justify at length, comment your code,
format tidily), and an **LLM judge** scores the response 0 or 1 against the
sample's ``response_reference``. Headline metric is the pass rate x100.

Unusually, the judge prompt is DATA, not a task constant: every row ships its own
``judge_system_prompt`` (one of four, by instruction type) and
``judge_prompt_template`` (one of three), rendered verbatim — upstream publishes
no evaluation harness, so those shipped strings are the only reference
implementation there is. Two consequences:

* Five of the eight types use a template with no ``{prompt}`` slot, so their
  judge never sees the question, only the response against the reference
  requirements. Upstream's design, preserved.
* The judge is instructed in Chinese for English samples too; the templates are
  language-independent by construction.

The candidate gets no system prompt: the dataset supplies none, and inventing one
would change what the counter-intuitive instruction competes against.

Grader is a REAL LLM supplied via the ``grader`` task arg on its own
``api_base``/``api_key``. A judge endpoint's model version is not pinnable like a
Hub revision, so pin the grader model and set ``temperature: 0``; each rollout
persists ``answer_score``, ``grader_parsed`` and the judge's whole ``ModelOutput``
(``extra.grader_output``). The paper's per-type "adaptive judge matrix" names none
of its models, so a single grader is used and any comparison must cite it.

Decoding is model-layer, via ``models:`` / ``infer_args`` — never here. The paper
states no temperature, token budget or repeat count; ``reference_impl.notes``
records what its tables imply.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.inverse_ifeval import (
    breakdown_metrics,
    parse_answer_score,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    EvalMode,
    InputKind,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RequirementContext,
    RolloutJudgement,
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
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
    sampling_report,
)
from sieval.core.utils.serialization import obj_to_dict
from sieval.datasets import InverseIFEvalDataset, InverseIFEvalDatasetSample


@sieval_task(
    name="inverse_ifeval_0shot_gen",
    display_name="Inverse IFEval (0-shot, generative)",
    description=(
        "Counter-intuitive instruction following, graded 0/1 by an LLM judge."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("chinese", "english", "open-ended"),
    model_type="chat",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="m-a-p/Inverse_IFEval",
        url="https://huggingface.co/datasets/m-a-p/Inverse_IFEval/blob/35f1da157640526e62b7685b682d748fa55ccfd0/README.md",
        notes=(
            "Upstream publishes the dataset ONLY — no evaluation harness, so "
            "there is no grader to vendor. The judge system prompt and prompt "
            "template are per-sample columns, rendered verbatim; the 0/1 rubric "
            'and the `【JSON】` / {"answer_score": N} reply contract are stated '
            "inside those shipped prompts. "
            "METRIC (recovered from Tables 2/3, 22 models x 2 languages, which "
            "state neither aggregation nor scale): Overall is a POOLED mean of "
            "per-sample 0/1 verdicts x100, reconstructed from each row's eight "
            "per-type cells to a mean absolute error of 0.002 (max 0.01 = the "
            "tables' own rounding) over all 44 rows; an unweighted macro-average "
            "misses by 1.4 mean / 4.4 max. Reported as `pass@1` over the "
            "requested denominator, which IS that pooled mean. "
            "COLUMN LABELS: the paper's `CC` and `CCF` are SWAPPED relative to "
            "the expansion of its own type list — every cell is a multiple of "
            "1/(6n), fixing `CC` to n=41 (Counter-Conventional Formatting) and "
            "`CCF` to n=99 (Code without Comments), the only reading the pooled "
            "reconstruction lands under. See "
            "sieval.community.inverse_ifeval.PAPER_COLUMNS before comparing a "
            "per-type number. "
            "REPEAT PROTOCOL: that same quantization implies SIX rollouts per "
            "sample; the paper states no repeat count. Defaults to n=1 — pass "
            "`n: 6` to match, and budget for it: grading is one judge call per "
            "rollout, serial within a sample, so n=6 costs six judge round-trips "
            "per sample. "
            "JUDGE: the paper's per-type 'adaptive judge matrix' names none of "
            "its models and states no decoding params for the evaluated models, "
            "so a single grader is used and published numbers are NOT "
            "reproducible as published — cite it with any comparison. "
            "Reference points (English / Chinese "
            "Overall): o3-high 75.66 / 76.52, GPT-5-high 73.72 / 76.02, "
            "Qwen3-32B 47.04 / 49.28, Qwen3-30B-A3B-Instruct 30.43 / 31.42. "
            "MEASURED on this port: Qwen3-30B-A3B-Instruct-2507 (temp 0.7 / "
            "top_p 0.8 / top_k 20, n=1, all 1,012) scores 33.00 / 33.60 graded "
            "by gpt-oss-120b at temperature 0, against a published 30.43 / 31.42 "
            "with a binomial SE of +-2.1 per language. Re-grading those SAME "
            "responses with Qwen3-30B-A3B-thinking gives 42.29 / 39.92 at 84.3% "
            "per-sample agreement: the grader alone moves the headline ~7.8 "
            "points, three times the residual against the published figure. That "
            "is what `experimental` records."
        ),
    ),
    # Faithful to everything upstream published, but the published numbers cannot
    # be reproduced: the judge matrix, the candidate decoding params and the
    # repeat count are all withheld, and the metric had to be recovered from the
    # result tables rather than read from a spec.
    status="experimental",
)
class InverseIFEvalZeroShotGenTask(
    Task[
        InverseIFEvalDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float | str | None],
    ]
):
    @classmethod
    @override
    def model_requirements_for(
        cls, context: RequirementContext
    ) -> tuple[TaskModelRequirement, ...]:
        candidate = super().model_requirements_for(context)
        grader = cls._bind_role_requirement(
            context,
            "grader",
            TaskRequirements(input=InputKind.CHAT),
        )
        return candidate + grader

    def __init__(
        self,
        dataset: InverseIFEvalDataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
        k: int = 1,
        models_by_role: Mapping[str, Model] | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(f"k must be <= n, got k={k}, n={n}")
        self._n = n
        self._k = k
        self._grader = self._resolve_role_model(
            "grader",
            grader,
            models_by_role,
            build=self._build_grader,
        )

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (tests / advanced configs) or a model-config
        mapping (the YAML path). Grading is mandatory — the rubric is prose in a
        ``response_reference``, so there is no deterministic fallback — so
        ``None`` raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "Inverse IFEval requires an LLM judge. Pass `grader:` in the task "
            "args — a model-config dict such as "
            "{model: gpt-4.1, api_base: ..., api_key: ..., temperature: 0}."
        )

    @override
    async def preprocess(self, raw, ctx):
        # No system prompt: the dataset ships none, and the counter-intuitive
        # instruction lives in the user turn.
        return build_prompt_record(
            [{"role": "user", "content": raw["prompt"]}],
            # The "gold" is the prose requirement the judge grades against.
            reference=raw["response_reference"],
            extra={
                "instruction_types": raw["instruction_types"],
                "language": raw["language"],
            },
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended: the response IS the answer, so no extraction step. A blank
        # response normalizes to None so `extracted` stays a real signal; the
        # judge still sees "" and the shipped rubric scores it 0.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Grade every rollout with the sample's own judge prompt.

        The template is rendered with ``str.format``; all three shipped templates
        contain nothing but their own placeholders, and braces inside the
        substituted values (abundant — a third of the code samples carry ``{``)
        are not re-interpreted, so no escaping is needed.

        Only the judge's ``texts[0]`` is parsed, never the prompt: every shipped
        ``judge_system_prompt`` ends with a worked example scoring 1, so parsing
        one would read as PASS and inflate the score invisibly. An unparseable
        reply is recorded (``grader_parsed=False``) and counted incorrect rather
        than dropped — the rubric has no third state, and the persisted reply is
        what separates judge format drift from a real 0.

        The grader's whole ``ModelOutput`` is persisted flattened rather than
        hand-picked: a grader model version is not pinnable like a Hub revision,
        which makes the reply the only durable evidence of a verdict.
        ``add_type=False`` keeps it a plain dict, so the judgement record does not
        nest a typed record object.
        """
        raw = ctx.raw_sample
        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            response = rollout.get("prediction") or ""
            judge_prompt = raw["judge_prompt_template"].format(
                prompt=raw["prompt"],
                response_reference=raw["response_reference"],
                response=response,
            )
            out = await self._grader.agenerate(
                [
                    {"role": "system", "content": raw["judge_system_prompt"]},
                    {"role": "user", "content": judge_prompt},
                ]
            )
            reply = out.texts[0] if out.texts else ""
            score, matched = parse_answer_score(reply)
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    score == 1,
                    extra={
                        "answer_score": score,
                        # The raw token behind an off-rubric verdict (e.g. "100"),
                        # so a rubric violation is distinguishable from silence.
                        "answer_score_raw": matched,
                        "grader_parsed": score is not None,
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )
        # Language and type travel on the judgement so `report` reads them off the
        # persisted record instead of depending on `raw_sample` being present.
        return True, build_judgement_record(
            raw["response_reference"],
            rollouts,
            extra={
                "instruction_types": raw["instruction_types"],
                "language": raw["language"],
            },
        )

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        # Headline through the shared block: at n=1 its `pass@1` over the
        # requested denominator IS this benchmark's pooled mean (see
        # sieval.community.inverse_ifeval), and failures count as incorrect the
        # way the paper scores an unanswered sample. `votes=False`: a majority
        # vote over free-form prose has no definition here.
        #
        # The other six LLM-judged tasks report `health_metrics` alone, since
        # RFC #74 defers `pass@k` / `maj@k` for the judged family. This one opts
        # IN, on that RFC's own three grounds rather than against them: the
        # verdict is a single binary 0/1, so "pass" needs no defining (unlike
        # simpleqa's three-way grade); no clustering is involved (`votes=False`);
        # and the grader cost it weighs is one this protocol asks for anyway,
        # since the paper's cell quantization implies n=6. Withholding the block
        # would hide the spread of a draw already paid for.
        rolled = sampling_report(
            finals, n=self._n, k=self._k, denominator=total, votes=False
        )
        pass_at_1 = rolled["pass@1"]

        graded: list[tuple[str, str, bool]] = []
        n_grader_unparsed = 0
        for ctx in finals:
            judgement = ctx.feedback_result or {}
            extra = judgement.get("extra") or {}
            language = extra.get("language", "unknown")
            instruction_type = extra.get("instruction_types", "unknown")
            for verdict in judgement.get("rollouts") or []:
                graded.append((language, instruction_type, bool(verdict["correct"])))
                if not (verdict.get("extra") or {}).get("grader_parsed", True):
                    n_grader_unparsed += 1

        metrics: dict[str, float | str | None] = {
            "score": pass_at_1,
            "pass@1": pass_at_1,
            "fails": len(fails),
            "n_graded": len(graded),
            # A count over graded attempts, not a rate over the requested set:
            # these rollouts ARE scored (0), so this is judge health, not a
            # missing denominator.
            "n_grader_unparsed": n_grader_unparsed,
            # Which subset was evaluated, so a single-language run is
            # distinguishable from a full one in the report alone. It names the
            # dataset's `language` ARG, not the rows that survived: a run
            # narrowed by `operations:` filtering instead still reads "both", and
            # it is the per-language breakdowns that disambiguate that one.
            "language": self._language_subset(),
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        if self._n > 1:
            # At n=1 the rest only restates `pass@1`.
            metrics.update(rolled)
        return metrics | health_metrics(finals) | breakdown_metrics(graded)

    def _language_subset(self) -> str:
        """The dataset's language selection, echoed for the report."""
        # `getattr`: a task can be handed a dataset built from a materialized
        # dict, which bypasses `load` and carries the class default.
        return getattr(self.dataset, "language", None) or "both"
