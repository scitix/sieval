"""AA-LCR — 0-shot generative, LLM-equality-checker graded.

Faithful generative port of Artificial Analysis Long Context Reasoning (AA-LCR):
the model answers a hard reasoning question with a set of real-world documents
(~100k tokens) loaded into the same prompt, and a separate **LLM equality
checker** grades the free-form answer against the official answer as binary
CORRECT / INCORRECT. The headline metric is accuracy over the 100 questions.

The upstream repo publishes no eval code — only the dataset card. The prompt and
grader templates are the card's own snippets (see ``sieval.community.aa_lcr``).
The grader is supplied via the ``grader`` task arg (a model-config dict, or a
pre-built Model, on its own ``api_base``/``api_key``); the official checker is
Qwen3 235B A22B 2507 Non-reasoning. Unlike sieval's deterministic-grader tasks,
correctness depends on a real grader model whose version sieval cannot pin the
way it pins a Hub revision, so for reproducibility pin the grader model and set
``temperature: 0``; each sample's grade and the checker's whole ``ModelOutput``
(``extra.grader_output``: reply, reasoning, usage, finish reasons, model id) are
persisted on the judgement record — see :meth:`feedback` for why the raw output
is kept whole.

Deviations / by-design behavior worth knowing:

* An empty/whitespace candidate answer (e.g. the model exhausted its token
  budget mid-reasoning) is graded INCORRECT **without** invoking the grader —
  the checker returns CORRECT for an empty candidate, which would otherwise
  inflate accuracy. ``extra.grader_output`` is ABSENT on this path because no
  checker call was made — absence being unambiguous where #51's empty
  ``grader_reply`` string was not; the matching prediction rollout's
  ``extracted: false`` identifies it independently.
  Pipeline failures are likewise counted INCORRECT (full-set metric).

Serving requirement — inputs run 71k–114k tokens, so ``max_tokens`` must leave
input + output inside the context window (e.g. 16384 against gpt-oss's 131072).
Do **not** enable sglang's ``allow_auto_truncate``: on inputs this large it
silently drops document tokens with no error, invalidating a benchmark whose
premise is the full document set — a hard error on overflow is the safe
behavior. At high reasoning effort the longest inputs have little output room
left; attempts that spend it all on reasoning land as empty answers (graded
INCORRECT per above), and the ``gen`` truncation detection rule flags the rest.

Reproduction decoding: ``n`` (repeats) is a **task arg** — set it in
``tasks.<name>.args.n`` (AA-LCR uses ``n=3``). ``infer`` forwards it as a
call-time kwarg to ``agenerate``, and call-time wins over model config
(``{**self._kwargs, **kwargs}``), so setting ``n`` on the model is silently
overridden by the task default (``_n=1``). Temperature stays model-layer (set
via ``models:`` / ``infer_args``): AA runs reasoning models at
``temperature=0.6``. pass@1 is aggregated across the ``n`` attempts.
Comparison target is the Artificial Analysis public leaderboard
(https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning);
scoring protocol at https://artificialanalysis.ai/methodology/intelligence-benchmarking.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.aa_lcr import (
    GRADER_TEMPLATE,
    aggregate_metrics,
    build_prompt,
    parse_grade,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RolloutJudgement,
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
from sieval.core.utils.serialization import obj_to_dict
from sieval.datasets import AALCRDatasetSample


@sieval_task(
    name="aa_lcr_0shot_gen",
    display_name="AA-LCR (0-shot, generative)",
    description="Long-context multi-document reasoning; LLM-graded free-form answers.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "long-context", "reasoning", "open-ended"),
    model_type="chat",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="aa-lcr",
        url="https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/blob/bdae010bbce259820c0e34c1d7cce210d966fb75/README.md",
        notes=(
            "Generative port of Artificial Analysis Long Context Reasoning "
            "(AA-LCR): 100 hard reasoning questions over 234 documents across 30 "
            "document sets (avg ~99k input tokens/set). No upstream eval code — "
            "the input prompt and the binary CORRECT/INCORRECT equality-checker "
            "prompt are the dataset card's own snippets (reproduced verbatim in "
            "sieval.community.aa_lcr, HF revision "
            "bdae010bbce259820c0e34c1d7cce210d966fb75). Headline metric = "
            "accuracy over the full set (pipeline failures counted INCORRECT). "
            "Grader is a REAL LLM (official checker: Qwen3 235B A22B 2507 "
            "Non-reasoning) supplied via the `grader` task arg on its own "
            "api_base/api_key. REPRODUCIBILITY: unlike deterministic-grader "
            "tasks, scores depend on the grader endpoint's model version (not "
            "pinnable like a Hub revision) — pin the grader model + "
            "temperature=0; the per-sample grade, grader model id, and the "
            "checker's full ModelOutput (extra.grader_output) are persisted — "
            "the reply "
            "being the only evidence of a verdict a re-grade need not reproduce, "
            "and (parse_grade sends anything unreadable to INCORRECT) the only "
            "way to tell format drift from a real negative. Documents are "
            "prompted in data_source_filenames order (loader-guaranteed), per "
            "the card. DEVIATION (the port's only score-affecting one): "
            "empty/whitespace candidates are graded INCORRECT without invoking "
            "the checker — the checker returns CORRECT on an empty candidate, "
            "which would inflate accuracy (grader_output absent there, no call "
            "made). REPEATS: AA-LCR uses n=3 (pass@1 aggregated across "
            "attempts); `n` is a task arg (tasks.<name>.args.n), NOT a model "
            "arg — infer forwards it call-time and call-time wins, so setting "
            "`n` on the model is overridden by the task default n=1. "
            "VALIDATION: reproduced gpt-oss-120b / gpt-oss-20b (reasoning=high) "
            "within ~2-3 pts of the AA public leaderboard (official 50.7 / "
            "30.7), grader Qwen3-235B-A22B-Instruct-2507 at temperature 0, n=3."
        ),
    ),
)
class AALCRZeroShotGenTask(
    Task[
        AALCRDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._build_grader(grader)

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (used by tests / advanced configs) or a
        model-config mapping (the YAML path, e.g.
        ``{model: Qwen3-235B-A22B-2507, api_base: ..., temperature: 0}``).
        Grading is mandatory — there is no deterministic fallback — so ``None``
        raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "AA-LCR requires an LLM grader. Pass `grader:` in the task args — a "
            "model-config dict such as {model: Qwen3-235B-A22B-2507, "
            "api_base: ..., api_key: ..., temperature: 0}."
        )

    @override
    async def preprocess(self, raw, ctx):
        content = build_prompt(raw["documents"], raw["question"])
        return build_prompt_record(
            [{"role": "user", "content": content}],
            reference=raw["answer"],
            extra={"question_id": raw["question_id"]},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended answer: the response *is* the answer. Normalizing a blank to
        # None makes `extracted` a real signal AND is exactly the empty-candidate
        # condition feedback() short-circuits on -- one notion of "no answer",
        # spelled once.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        question = raw["question"]
        gold = raw["answer"]

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            predicted = rollout.get("prediction")
            # Defensive, aa_lcr-specific by design (simpleqa_verified/browsecomp
            # omit it): the checker returns CORRECT on an empty candidate, which
            # would inflate accuracy. Not observed in the 4x300 validation runs
            # (0 empty, 0 length-truncated), but tight output budgets on these
            # ~100k-token prompts can yield empty answers.
            if predicted is None:
                # No checker call, so there is no grader output to record -- and
                # its ABSENCE is now the durable signal, rather than #51's empty
                # `grader_reply` string, which was confusable with a grader that
                # returned nothing. `extracted: false` on the matching prediction
                # rollout identifies this branch independently.
                rollouts.append(
                    build_rollout_judgement(
                        rollout["index"], False, extra={"grade": "INCORRECT"}
                    )
                )
                continue
            prompt = GRADER_TEMPLATE.format(
                question=question,
                official_answer=gold,
                candidate_answer=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            grade = parse_grade(reply)
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    grade == "CORRECT",
                    extra={
                        "grade": grade,
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )
        return True, build_judgement_record(
            gold, rollouts, extra={"question_id": raw["question_id"]}
        )

    @override
    async def report(self, finals, fails):
        graded = [
            r["extra"]["grade"]
            for f in finals
            for r in (f.feedback_result or {}).get("rollouts", [])
        ]
        # Pipeline failures (exhausted retries) never produced a gradeable
        # answer; count each failed sample's requested attempts as INCORRECT so
        # the accuracy spans the full requested set — matching the official
        # full-set metric and the gen-task family, rather than only the
        # successfully-graded subset.
        grades = graded + ["INCORRECT"] * (self._n * len(fails))
        m = aggregate_metrics(grades)
        return {
            "score": m["accuracy"] * 100,
            "accuracy": m["accuracy"] * 100,
            "correct": m["is_correct"] * 100,
            "incorrect": m["is_incorrect"] * 100,
            "n_graded": len(graded),
            "fails": len(fails),
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            # The module docstring leans on `extracted: false` to tell an empty
            # response apart from one the grader failed; without the count in the
            # report that distinction lives only in the shards. Deliberately only
            # `health_metrics` and not the rest of the sampling block: RFC #74
            # defers `pass@k` / `maj@k` for the LLM-judged family, while this one
            # measures extraction rather than the draw and is outside that gate.
        } | health_metrics(finals)
