"""
Shared implementation for the PlatinumBench ``math``-parsing subsets.

Five subsets (singleop, singleq, multiarith, svamp, gsm8k) declare
``platinum_parsing_strategy == "math"`` at the pinned revision and are scored
identically upstream, so they share one base class and differ only by the
``subset`` they bind to. Each leaf module is a decorated 3-line subclass, which
is what gives the leaderboard five columns instead of one aggregate.

Alignment with MadryLab/platinum-benchmarks at ``8fd2f82``:

* Prompt: taken verbatim from the row. Upstream builds no prompt of its own —
  ``get_prompt`` picks between the data's ``platinum_prompt`` (CoT) and
  ``platinum_prompt_no_cot`` columns and sends the string as a single user turn.
* Extraction: ``get_parse_fn("math")``, vendored byte-faithfully in
  ``sieval.community.platinum_bench``.
* Scoring: ``check_prediction``, also vendored. For these five subsets that
  resolves to ``float(platinum_target[0]) == float(prediction)`` — **exact**
  float equality, not a tolerance, which is why the parse function strips a
  trailing ``.0`` before comparison.

Deviations, all deliberate:

* ``prompt_variant`` (task arg, default ``"cot"``) replaces upstream's
  ``ModelEngineFactory.reasoning_models`` name list, which decides CoT vs no-CoT
  — and, for the o1 snapshots, an extra prompt rewrite — by hardcoded model
  name. A name list cannot classify an arbitrary served endpoint, and silently
  switching prompts on a model string is exactly the kind of implicit behaviour
  that makes two runs incomparable. All three of upstream's prompts are
  reachable, explicitly: ``cot``, ``no_cot`` (reasoning models), and
  ``no_cot_o1`` (the o1-2024-12-17 / o1-preview rewrite).
* A parse failure becomes ``prediction=None`` and ``correct=False``. Upstream
  reaches the same verdict, but by exception: ``parse_fn_math`` raises
  ``AttributeError`` on digit-free output, ``run_benchmark.py`` catches it with a
  bare ``except`` and stores the string ``'parsing error'``. Its
  ``prediction != 'Parsing error'`` guard in ``check_prediction`` never matches
  that lowercase value, so upstream then raises ``ValueError`` inside
  ``float()`` and swallows that too. Same score, without replicating the bug.
* Accuracy is over the full requested set (``finals + fails``), so a pipeline
  failure counts as wrong. Upstream has no failure bucket — every row it fetched
  gets a verdict — so this keeps the denominators equal.

Upstream reports **error counts** per subset and a plain (non-size-weighted)
mean across subsets, so ``report()`` emits ``errors`` alongside ``accuracy``;
the cross-subset mean is the leaderboard's job, not this task's.

Repro decoding (model-layer assets — set via ``models:`` / ``infer_args``, not
in this code): ``temperature=0.5``, ``top_p`` unset, one sample per question, no
seed (``models.py::ModelInferenceEngine``); the o1 / o3 snapshots instead run at
``temperature=1``, which their API required. Notably *not* greedy — upstream
sampled at 0.5, so a single run is not bit-reproducible upstream either.

Infer prerequisites: set ``max_tokens=6000`` (upstream's
``ModelInferenceEngine`` default) — the score is budget-sensitive, because a
short budget truncates CoT before the ``Answer:`` line and converts correct
reasoning into parse failures, which this benchmark counts as model errors. The
task deliberately forwards **no** decoding param of its own: ``agenerate`` merges
``{**model_kwargs, **task_kwargs}``, so a task-side value would silently outrank
whatever the caller configured, and the one knob a user most needs to turn here
would be the one they cannot. (``n`` is forwarded, but it is the sampling budget
rather than a decoding param.) Upstream sized 6000 for non-thinking models; on a
thinking model raise it, since a live run of all five subsets against Qwen3-32B
spent the whole 6000 inside the reasoning channel on one singleop question and
returned an empty answer that scores as an error. Read ``errors`` together with
``anomalies.json`` — ``truncated_output`` is what separates "ran out of budget"
from "got it wrong".

Validation — every math cell of the paper's Table 3 (24 models x 5 subsets =
120 error counts) is reproduced exactly by this task. Sampling at 0.5 with no
seed makes a fresh run unreproducible by construction, so the check instead
replays upstream's own published inferences
(``madrylab/platinum-bench-paper-cache``, one pickle per subset, keyed by
``(prompt, temperature, 0, model)``) through the real pipeline: this dataset's
rejected-row filter, these leaves, and ``preprocess`` → ``report``. Every one of
the 968 prompts per model hits a cache entry, which is also what proves the
prompt this task sends is upstream's byte for byte. Two things that check pins
down and a fresh run could not: the counts are exact rather than
within-noise (a 0-19-error metric over 100-274 questions has a binomial
sigma near 2), and it runs against
``madrylab/platinum-bench-paper-version`` — the repo the paper's numbers were
computed on. The revision pinned for actual use is the current
``madrylab/platinum-bench``, whose later cleaning pass rejects 15 more math rows
(968 → 953: multiarith 171→170, svamp 273→265, gsm8k 274→268), so its scores are
close to but not identical with the paper's, and upstream publishes no table for
it. Upstream defaults to the current repo too, with ``--paper-version`` opt-in.

What that replay deliberately does not cover is the model layer, which it stubs
out. So it was also run live, end-to-end: all five subsets, 953 questions,
Qwen3-32B over an OpenAI-protocol endpoint, ``temperature=0.5``. Zero pipeline
failures; 6 errors, of which 5 are ordinary wrong answers with a clean
``Answer:`` line and the sixth is the budget truncation described above. Nothing
about a live run can confirm a *published* number — no model upstream evaluated
is still served anywhere — so the two checks answer different questions: the
replay pins scoring fidelity, the live run pins that the configured
``max_tokens`` and ``temperature`` reach the wire and that real completions
survive the round trip.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import ClassVar, Literal, override

from loguru import logger

from sieval.community.platinum_bench import check_prediction, get_parse_fn
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    SCORE_KEY_FIELD,
    aggregate,
    count_short,
    rollout_metrics,
)
from sieval.datasets import PlatinumBenchDatasetSample

PLATINUM_UPSTREAM_URL = "https://github.com/MadryLab/platinum-benchmarks/blob/8fd2f82e63c49ea1cca4266f4dded82b7ddbcb55/src/utils.py"

# Shared tail of every leaf's `reference_impl.notes`. `sieval task show` is the
# only place these facts surface, so they are repeated per task rather than left
# to a module docstring no CLI reads.
PLATINUM_REFERENCE_NOTES = (
    "Prompt comes from the row, not from this code: upstream get_prompt() picks "
    "between the data's platinum_prompt (CoT) and platinum_prompt_no_cot columns "
    "and sends it as one user turn. Extraction (get_parse_fn('math')) and scoring "
    "(check_prediction, i.e. exact float equality on platinum_target[0]) are "
    "vendored byte-faithfully in sieval.community.platinum_bench. Rows with "
    "cleaning_status='rejected' are dropped, matching the published leaderboard. "
    "Deviations: prompt_variant (task arg, default cot) replaces upstream's "
    "hardcoded reasoning-model name list, which also decided an o1-only prompt "
    "rewrite; all three upstream prompts are reachable explicitly as cot / "
    "no_cot / no_cot_o1. A parse failure is recorded as prediction=None/"
    "correct=False instead of upstream's swallowed AttributeError. "
    "Repro decoding (model-layer assets — set via models: / infer_args, not in "
    "this code): temperature=0.5, one sample per question, no seed; the o1/o3 "
    "snapshots ran at temperature=1. Not greedy, so upstream runs are not "
    "bit-reproducible either. Infer prereqs: set max_tokens=6000 (upstream's "
    "default) — the score is budget-sensitive, since a smaller budget truncates "
    "CoT before the 'Answer:' line and turns correct reasoning into parse "
    "errors. The task forwards no budget of its own on purpose: agenerate "
    "merges {**model_kwargs, **task_kwargs}, so a task-side value would "
    "silently outrank the one you configured. That budget was sized for "
    "non-thinking models: on a thinking model raise it, since a live Qwen3-32B "
    "run spent all 6000 in the reasoning channel on one question and scored the "
    "empty answer as an error. anomalies.json flags exactly that case as "
    "truncated_output, which is what separates it from a wrong answer. "
    "Validated: all 120 math error counts of the paper's Table 3 (24 models x 5 "
    "subsets) reproduce exactly, by replaying upstream's own published "
    "inferences (madrylab/platinum-bench-paper-cache) through this pipeline "
    "against madrylab/platinum-bench-paper-version. Reproduce a published row "
    "with prompt_variant=cot at temperature 0.5, except: no_cot for DeepSeek-R1 "
    "and Gemini Thinking; no_cot at temperature 1 for o1-mini and o3-mini; "
    "no_cot_o1 at temperature 1 for o1-2024-12-17 and o1-preview. The pinned "
    "revision is the current madrylab/platinum-bench, which rejects 15 more math "
    "rows than the paper version (968 -> 953), so its scores are near but not "
    "equal to the paper's and upstream publishes no table for it. The replay stubs "
    "the model layer, so it was also run live end-to-end (all 5 subsets, 953 "
    "questions, Qwen3-32B over an OpenAI-protocol endpoint): 0 pipeline "
    "failures, 6 errors, 5 of them genuine wrong answers. Qwen3-32B has no "
    "published row, so alignment also ran two models that do, each over the "
    "paper version's 968 questions at temperature 0.5, one sample. Upstream "
    "served both of its open-weight rows through DeepInfra (its model factory "
    "routes Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct there), so those "
    "runs pin that provider with allow_fallbacks=false: an aggregator otherwise "
    "routes across providers that quantize differently, and quantization "
    "changes completions. Qwen2.5-72B-Instruct scored 0/0/0/4/4 = 8 against "
    "Table 3's 0/0/0/4/7 = 11; Llama-3.3-70B-Instruct scored 1/0/0/6/9 = 16 "
    "against 0/0/0/7/7 = 14 there and 0/0/0/8/7 = 15 on a bf16 endpoint, which "
    "brackets the serving-precision effect. Each lands within 0.7 of sigma_D = "
    "sqrt(paper + live), the spread of a difference of two unseeded draws — the "
    "published count is a single sample too, so exact equality is not "
    "achievable and repeats would not sharpen the comparison. Three of the five "
    "cells publish 0 errors, so agreeing there is close to automatic; svamp and "
    "gsm8k carry the signal. What separates sampling noise from a scoring "
    "difference is which questions were missed: 9 of 15, 9 of 16 and 7 of 8 "
    "live misses are misses in upstream's own completions, and the rest move in "
    "both directions (live-only and paper-only both non-zero), which is the "
    "signature of resampling rather than of a one-sided scoring change. All "
    "three runs had 0 pipeline failures, and every miss finished on a stop "
    "token with a parseable integer, so none is a truncation or parse artifact. "
    "The replay is what pins scoring fidelity; these runs pin the live path."
)

# `parsing_strategy` value shared by all five subsets this base serves. Asserted
# per sample rather than assumed: it is a data column, so a future revision can
# change it, and a silently-wrong parser would look like a model regression.
MATH_PARSING_STRATEGY = "math"

# Resolved once: `get_parse_fn` rebuilds its whole strategy table on every call
# and depends on nothing but the strategy name, which `preprocess` pins per
# sample anyway.
PARSE_MATH_ANSWER = get_parse_fn(MATH_PARSING_STRATEGY)

TPromptVariant = Literal["cot", "no_cot", "no_cot_o1"]

_PROMPT_VARIANTS: tuple[TPromptVariant, ...] = ("cot", "no_cot", "no_cot_o1")

# Upstream's o1-family edit. The no-CoT column reads "Then, provide the final
# answer ..." — a leftover conjunction from the CoT wording, since nothing
# precedes it once the reasoning sentence is gone. Upstream rewrites it for the
# o1 snapshots only, so reproducing their o1 rows needs the edited string and
# reproducing o1-mini / o3-mini needs the unedited one.
_O1_PROMPT_EDIT = ("Then, provide", "Provide")


class PlatinumMathGenTask(
    Task[
        PlatinumBenchDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    """Base for one PlatinumBench math subset; leaves set :attr:`subset`."""

    #: The `madrylab/platinum-bench` config this task scores. The single source
    #: of truth for it: the dataset must be narrowed to this subset, which
    #: `setup()` enforces.
    subset: ClassVar[str]

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        prompt_variant: TPromptVariant = "cot",
        k: int = 1,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        self._k = k
        self._n = n
        if prompt_variant not in _PROMPT_VARIANTS:
            raise ValueError(
                f"prompt_variant must be one of {list(_PROMPT_VARIANTS)}, "
                f"got {prompt_variant!r}."
            )
        self._prompt_variant = prompt_variant

    def _prompt_text(self, raw: PlatinumBenchDatasetSample) -> str:
        """The prompt column this run reads — upstream's ``get_prompt`` branches."""
        if self._prompt_variant == "cot":
            return raw["platinum_prompt"]
        text = raw["platinum_prompt_no_cot"]
        if self._prompt_variant == "no_cot_o1":
            return text.replace(*_O1_PROMPT_EDIT)
        return text

    @override
    async def setup(self) -> None:
        # The loader merges all 14 configs into one split, so the caller narrows
        # it to a single subset. Forget that, or narrow to a sibling subset, and
        # the run would score real answers against the wrong questions — a
        # silently plausible result. Fail here instead, before any tokens are
        # spent: `setup()` runs before the runner counts samples.
        #
        # A missing test split, a dataset of some other class, and an empty one
        # all collapse to `[]`, so one comparison covers every way the wiring
        # can be wrong.
        test_set = self.dataset.test_set
        if test_set is None or "subset" not in test_set.column_names:
            present: list[str] = []
        else:
            present = sorted(test_set.unique("subset"))
        if present != [self.subset]:
            raise ValueError(
                f"{type(self).__name__} scores the '{self.subset}' subset but its "
                f"dataset carries {present!r}. Narrow it to one subset with "
                f"`operations: [{{filter: {{by: subset, value: {self.subset}}}}}]`."
            )

    @override
    async def preprocess(self, raw, ctx):
        strategy = raw["platinum_parsing_strategy"]
        if strategy != MATH_PARSING_STRATEGY:
            raise ValueError(
                f"{type(self).__name__} parses answers with the "
                f"'{MATH_PARSING_STRATEGY}' strategy, but this row declares "
                f"{strategy!r}. The pinned revision's parsing strategy changed."
            )
        # The row carries the fully-rendered prompt; upstream sends it as-is as a
        # single user turn and lets the backend apply the chat template.
        return build_prompt_record(
            [{"role": "user", "content": self._prompt_text(raw)}],
            reference=list(raw["platinum_target"]),
        )

    @override
    async def infer(self, pre, ctx):
        # No decoding params: `agenerate` merges `{**model_kwargs, **task_kwargs}`,
        # so anything passed here would silently outrank the caller's config.
        # `n` is the exception — the sampling budget `k` was validated against,
        # so it has to reach the model (sieval/tasks/CLAUDE.md, "n_shot vs k").
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # One prediction per rollout: `texts[0]` would discard the rest of a
        # draw that was already generated and paid for.
        return build_prediction_record(
            [self._parse_one(text) for text in inf.texts] or [None]
        )

    @staticmethod
    def _parse_one(text: str) -> str | None:
        try:
            return PARSE_MATH_ANSWER(text)
        except AttributeError:
            # No digit anywhere in the output: upstream's `re.search(...).group()`
            # on None. `None` is the protocol's spelling of "could not extract".
            return None

    @override
    async def feedback(self, post, ctx):
        reference = list(ctx.raw_sample["platinum_target"])
        judgements = []
        for index, rollout in enumerate(post["rollouts"]):
            # `.get()`, not `[]`: a None prediction is ABSENT once the record
            # round-trips through disk, which is what the resume path reads.
            prediction = rollout.get("prediction")
            if prediction is None:
                correct = False
            else:
                try:
                    correct = check_prediction(
                        prediction,
                        reference,
                        # Unused upstream, but this is the same string it passes.
                        self._prompt_text(ctx.raw_sample),
                        self.subset,
                    )
                except (TypeError, ValueError):
                    # The parse regex can yield a non-float string (e.g. "1.2.3"),
                    # which upstream feeds straight into float() and swallows via its
                    # bare except. Unparseable-as-float is wrong, not an error.
                    correct = False
            judgements.append(build_rollout_judgement(index, correct))
        return True, build_judgement_record(
            reference,
            judgements or [build_rollout_judgement(0, False)],
            extra={"subset": self.subset},
        )

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            return {
                "score": 0.0,
                "fails": len(fails),
                "accuracy": 0.0,
                "errors": 0,
                SCORE_KEY_FIELD: "accuracy",
            }
        # `accuracy` and `errors` keep their first-rollout definition so they stay
        # comparable with upstream's per-dataset tables; the sampling metrics below are
        # additive and never touch them.
        correct_num = sum(
            1 for ctx in finals if ctx.feedback_result["rollouts"][0]["correct"]
        )
        accuracy = 100 * correct_num / total
        metrics: dict[str, float | str] = {
            "score": accuracy,
            "fails": len(fails),
            "accuracy": accuracy,
            # Upstream's headline unit: how many of this subset's questions the
            # model got wrong. Directly comparable to its per-dataset tables.
            "errors": total - correct_num,
            SCORE_KEY_FIELD: "accuracy",
        }
        if self._n <= 1:
            return metrics

        # Averaged over `total`, the denominator `accuracy` already uses, so a
        # failed sample counts as wrong in both. Declared rather than unified
        # across tasks, which would change stored numbers (RFC #74 F).
        per_problem = []
        observed = []
        for ctx in finals:
            rollouts = ctx.feedback_result["rollouts"]
            observed.append(len(rollouts))
            answers = [
                r.get("prediction")
                for r in (ctx.postprocess_result or {}).get("rollouts", [])
            ]
            per_problem.append(
                rollout_metrics(
                    [bool(r["correct"]) for r in rollouts],
                    answers if len(answers) == len(rollouts) else None,
                    k=self._k,
                )
            )
        metrics.update(aggregate(per_problem, total))
        metrics["n"] = float(self._n)
        metrics["k"] = float(self._k)
        short = count_short(observed, self._n)
        metrics["n_short"] = float(short)
        if short:
            logger.warning(
                "{}/{} sample(s) came back with fewer than the requested n={} "
                "rollout(s); they contribute 0 to pass@k and bias every sampling "
                "metric downward.",
                short,
                len(finals),
                self._n,
            )
        return metrics
