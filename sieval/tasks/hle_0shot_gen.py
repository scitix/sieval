"""Humanity's Last Exam (HLE) — 0-shot generative, LLM-judge graded.

Generative port of HLE (Center for AI Safety; Phan et al., 2025). The model
answers a closed-ended, multi-domain academic question under the HLE system
prompt (an ``Explanation / Answer / Confidence`` format), and a separate **LLM
judge** decides whether the free-form answer matches the gold. Headline metric
is accuracy; the judge also extracts the model's confidence, from which a
calibration error is computed (alongside a 95% Wald confidence interval).

Subset: a dataset-level choice — ``HLEDataset`` drops image questions unless
``datasets.hle.args.text_only: false`` (a sieval addition; upstream always
grades the full set), and ``report()`` echoes it as ``subset``. Because it lives
on the dataset, this task stays isomorphic to upstream
``run_model_predictions.py``: it grades whatever it is handed, attaching an
``image_url`` content block for image questions exactly as upstream
``format_message`` does.

Deviations from upstream (``hle_eval`` @ 26dca2e; see ``sieval.community.hle``):

* The upstream o1-only ``system``→``user`` role swap is dropped; the system
  prompt is always sent as ``system`` (correct for general instruct models).
  o1-family candidates that require the swap are therefore out of scope.
* The judge is reached through ``ChatModel`` (text), not upstream's
  ``beta.chat.completions.parse`` structured output; its ``correct``/``confidence``
  fields are parsed from the reply (see ``sieval.community.hle.parse_judge``).
  Upstream's server-enforced schema makes a malformed reply near-impossible; the
  text path widens that failure surface, so the reply itself is persisted.
* Calibration error is guarded below the bin size for slices/tests (docs there).

Decoding params are model-layer, set via ``models:`` / ``infer_args`` — never by
this task. Upstream HLE defaults to ``temperature=0`` and advises
``max_completion_tokens>=8192`` for reasoning models; specific reproductions
override these (e.g. a technical report may evaluate at ``temperature=1.0``,
``top_p=0.95`` with a large token budget).

Grader is a REAL LLM supplied via the ``grader`` task arg on its own
``api_base``/``api_key``. Correctness depends on the judge endpoint's model
version (not pinnable like a Hub revision) — pin the grader model for
reproducibility; each sample's ``correct``, ``confidence``, ``judge_parsed`` and
the judge's whole ``ModelOutput`` (``extra.grader_output``: reply, reasoning,
usage, finish reasons, model id) are persisted on the judgement record — see
:meth:`feedback` for why the raw output is kept whole rather than hand-picked.
The judge's decoding is likewise model-layer (set via
the ``grader`` config); upstream runs it at ``max_completion_tokens=4096``.

Target: report against technical-report HLE numbers (e.g. the GLM series
evaluates the text-only subset with a strong LLM judge, such as GPT-5.2); the
grading protocol (judge model, subset) is report-specific, so cite it alongside
any comparison.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.hle import (
    JUDGE_PROMPT,
    SYSTEM_PROMPT,
    aggregate_metrics,
    parse_judge,
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
from sieval.datasets import HLEDataset, HLEDatasetSample


@sieval_task(
    name="hle_0shot_gen",
    display_name="Humanity's Last Exam (0-shot, generative)",
    description="Multi-domain frontier academic QA graded by an LLM judge.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "reasoning", "academic"),
    model_type="chat",
    deps_group="hle",
    status="stable",
    reference_impl=ReferenceImpl(
        source="hle",
        url="https://github.com/centerforaisafety/hle/tree/26dca2e253b405105b4c3d8c2f5af06f86f90c66/hle_eval",
        notes=(
            "Generative port of Humanity's Last Exam (Center for AI Safety). "
            "SYSTEM_PROMPT and JUDGE_PROMPT are vendored byte-for-byte; the judge "
            "runs through sieval's ChatModel (text) rather than upstream's "
            "beta.chat.completions.parse structured output, and its correct/"
            "confidence fields are parsed from the reply. Metrics mirror upstream "
            "dump_metrics: accuracy, a 95% Wald confidence interval, and "
            "calibration error (calib_err, p=2, beta=100). Subset selection is a "
            "sieval addition living on HLEDataset (datasets.hle.args.text_only, "
            "default true — the full text+image set needs a vision-capable "
            "candidate + judge); text-only numbers are the asterisked variant in "
            "reports, the full set the unmarked headline, and report() echoes "
            "which subset was graded. "
            "Grader is a REAL LLM (upstream default o3-mini-2025-01-31) "
            "supplied via the `grader` task arg on its own api_base/api_key. "
            "REPRODUCIBILITY: scores depend on the judge endpoint's model version "
            "(not pinnable like a Hub revision) — pin the grader model; the "
            "per-sample correct/confidence and the judge's full ModelOutput "
            "(extra.grader_output: reply, reasoning, usage, finish_reasons, model "
            "id) are persisted — the reply being the only evidence of a verdict a "
            "re-run need not reproduce, and what separates format drift from a "
            "matcher gap behind the judge_unparsed count; finish_reasons "
            "separates a reasoning judge that spent its whole budget thinking "
            "from an empty API response. "
            "VALIDATION: gpt-oss-20b scored 12.14 / 3.61 (reasoning=high / low, "
            "judge GPT-5.2, text-only, no tools) vs the gpt-oss model card "
            "(arXiv:2508.10925) 10.9 / 4.2 — within <3pp."
        ),
    ),
)
class HLEZeroShotGenTask(
    Task[
        HLEDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float | str | None],
    ]
):
    def __init__(
        self,
        dataset: HLEDataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        # Which subset was loaded is the dataset's decision; read it back so
        # `report()` can record it.
        self._text_only = dataset.text_only
        self._grader = self._build_grader(grader)

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (tests / advanced configs) or a model-config
        mapping (the YAML path, e.g. ``{model: gpt-5.2, api_base: ...}``).
        Grading is mandatory — there is no deterministic fallback — so ``None``
        raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "HLE requires an LLM judge. Pass `grader:` in the task args — a "
            "model-config dict such as "
            "{model: gpt-5.2, api_base: ..., api_key: ..., reasoning_effort: medium}."
        )

    @override
    async def preprocess(self, raw, ctx):
        content: list[dict] = [{"type": "text", "text": raw["question"]}]
        if raw["image"]:  # "" when not multi-modal
            content.append({"type": "image_url", "image_url": {"url": raw["image"]}})
        # Annotated because the user turn nests a content-block list, which makes
        # the literal infer a union; a record's `prompt` is JSONValue -- whatever
        # shape the model kind takes.
        messages: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return build_prompt_record(
            messages,
            reference=raw["answer"],
            extra={"has_image": bool(raw["image"])},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended answer: the response *is* the answer, so no extraction. A
        # blank response normalizes to None so `extracted` stays a real signal;
        # the judge still receives "" and rates it.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Judge every rollout, recording the judge's full output.

        ``extra["grader_output"]`` is the judge's whole ``ModelOutput`` flattened
        to a plain dict -- reply text, reasoning, usage, finish reasons, model id.
        This closes the limit #51 documented for this task: the old flat
        ``grader_reply`` was response content only, so a reasoning judge that
        spent its entire budget thinking recorded an empty reply indistinguishable
        from an empty API response. ``finish_reasons``, in the same output, is
        what separates them -- and HLE is normally run with a reasoning judge, so
        this was the most reachable case of the three.

        ``confidence`` and ``judge_parsed`` are task logic derived from that raw
        output, which is why they are separate keys. They live in ``extra`` rather
        than ``metrics``: neither measures whether the answer was right.
        ``confidence`` is the judge's self-reported number and is the raw material
        report() needs to pool a calibration error, which a per-sample metric
        could not reconstruct.
        """
        raw = ctx.raw_sample
        question = raw["question"]
        gold = raw["answer"]

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            predicted = rollout.get("prediction") or ""
            prompt = JUDGE_PROMPT.format(
                question=question,
                correct_answer=gold,
                response=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            correct, confidence, parsed = parse_judge(reply)
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    correct,
                    extra={
                        "confidence": confidence,
                        "judge_parsed": parsed,
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )
        return True, build_judgement_record(gold, rollouts)

    @override
    async def report(self, finals, fails):
        # Only parsed judge replies feed the grading/calibration arrays. An
        # unparseable reply is kept in `n` (counted incorrect) but dropped here,
        # mirroring upstream's None on judge failure — it must not contribute a
        # spurious max-confidence point that would inflate calibration_error.
        correct: list[bool] = []
        confidence: list[int] = []
        judge_unparsed = 0
        for f in finals:
            for fb in (f.feedback_result or {}).get("rollouts", []):
                if fb["extra"]["judge_parsed"]:
                    correct.append(fb["correct"])
                    confidence.append(fb["extra"]["confidence"])
                else:
                    judge_unparsed += 1
        # Denominator spans the full requested set; pipeline failures (candidate
        # produced no gradeable answer) count as incorrect — matching upstream
        # (n = total questions) and the *_gen family, not just graded attempts.
        n = (len(finals) + len(fails)) * self._n
        m = aggregate_metrics(correct, confidence, n)
        # `judge_unparsed` is a count over the graded attempts in `finals`, not a
        # rate over `n` (which also spans `fails`). `subset` records which set
        # was evaluated so a text-only run is distinguishable from a full-set one
        # in the report alone. Length-capped attempts are NOT counted here — the
        # `gen`-scoped truncation detection rule already reports them per sample
        # (it also covers max_tokens / content_filter, which a finish_reasons
        # tally in this method would miss).
        return {
            "score": m["accuracy"],
            "accuracy": m["accuracy"],
            "confidence_interval": m["confidence_interval"],
            "calibration_error": m["calibration_error"],
            "n": n,
            "n_graded": len(correct),
            "fails": len(fails),
            "judge_unparsed": judge_unparsed,
            "subset": "text_only" if self._text_only else "full",
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            # `judge_unparsed` counts the GRADER failing to answer; `n_unextracted`
            # counts the candidate producing nothing to grade. Both grade
            # incorrect, and without the second one they are indistinguishable in
            # the report. Deliberately only `health_metrics` and not the rest of
            # the sampling block: RFC #74 defers `pass@k` / `maj@k` for the
            # LLM-judged family, while this one measures extraction rather than
            # the draw and is outside that gate.
        } | health_metrics(finals)
