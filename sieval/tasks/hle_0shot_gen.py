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
reproducibility; each sample's ``correct``, ``confidence``, ``judge_parsed``,
grader model id and the judge's reply (``grader_reply``) are persisted in the
feedback record — see ``JudgeFeedback.grader_reply`` for what that reply does and
does not let you diagnose. The judge's decoding is likewise model-layer (set via
the ``grader`` config); upstream runs it at ``max_completion_tokens=4096``.

Target: report against technical-report HLE numbers (e.g. the GLM series
evaluates the text-only subset with a strong LLM judge, such as GPT-5.2); the
grading protocol (judge model, subset) is report-specific, so cite it alongside
any comparison.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import TypedDict, override

from openai.types.chat import ChatCompletionMessageParam

from sieval.community.hle import (
    JUDGE_PROMPT,
    SYSTEM_PROMPT,
    aggregate_metrics,
    parse_judge,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import HLEDataset, HLEDatasetSample


class JudgeFeedback(TypedDict):
    correct: bool
    confidence: int
    judge_parsed: bool
    gold: str
    predicted: str
    grader_model: str
    # The judge's reply verbatim, on every attempt — the text every field above
    # comes from. When `judge_parsed` is False it is the only evidence of *why*
    # (format drift, an error body and a matcher gap are identical in the
    # `judge_unparsed` count alone); on the parsed path it is what makes a
    # wrong-but-parsed verdict — the kind that moves the score — auditable.
    #
    # Scope: `ModelOutput.texts`, response content only. A judge that spends its
    # whole budget on reasoning returns empty content, so an empty reply is
    # indistinguishable from an empty API response; the `finish_reasons` and
    # `reasoning_texts` that would separate them are not captured. Reachable
    # here — HLE is normally run with a reasoning judge.
    grader_reply: str


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
            "per-sample correct/confidence, grader model id, and the judge's "
            "reply (grader_reply) are persisted — the reply being the only "
            "evidence of a verdict a re-run need not reproduce, and what "
            "separates format drift from a matcher gap behind the judge_unparsed "
            "count. It is response content only, so a judge that spends its whole "
            "budget on reasoning still records an empty reply. "
            "VALIDATION: gpt-oss-20b scored 12.14 / 3.61 (reasoning=high / low, "
            "judge GPT-5.2, text-only, no tools) vs the gpt-oss model card "
            "(arXiv:2508.10925) 10.9 / 4.2 — within <3pp."
        ),
    ),
)
class HLEZeroShotGenTask(
    Task[
        HLEDatasetSample,
        list[ChatCompletionMessageParam],
        ModelOutput,
        list[str],
        list[JudgeFeedback],
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
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        return list(inf.texts)

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        question = raw["question"]
        gold = raw["answer"]
        grader_model = self._grader.meta()["model"]

        feedbacks: list[JudgeFeedback] = []
        for predicted in post:
            prompt = JUDGE_PROMPT.format(
                question=question,
                correct_answer=gold,
                response=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            correct, confidence, parsed = parse_judge(reply)
            feedbacks.append(
                {
                    "correct": correct,
                    "confidence": confidence,
                    "judge_parsed": parsed,
                    "gold": gold,
                    "predicted": predicted,
                    "grader_model": grader_model,
                    "grader_reply": reply,
                }
            )
        return True, feedbacks

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
            for fb in f.feedback_result or []:
                if fb["judge_parsed"]:
                    correct.append(fb["correct"])
                    confidence.append(fb["confidence"])
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
        }
