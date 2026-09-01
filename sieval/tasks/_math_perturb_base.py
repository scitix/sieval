"""
Shared 0-shot generative protocol for MATH-P-Simple and MATH-P-Hard.

The two testsets are graded identically and reported identically — they differ
only in which 279 rows they carry — so everything but the sample type lives
here, and each leaf module is metadata plus one generic argument.

**Prompt.** Upstream ships an evaluation package and no inference code, and the
paper (arXiv:2502.06453 §3) specifies only "zero-shot chain-of-thought (CoT)"
with no tool use. So the prompt is a sieval choice, and the one taken is
DeepSeek-Math's zero-shot CoT user turn — ``{problem}`` followed by
``"\\nPlease reason step by step, and put your final answer within \\boxed{}."``
— because MATH-Perturb's grader *is* DeepSeek-Math's grader (upstream's
``evaluation/README.md`` says so), and that instruction is the one its extractor
was written against. The chat template is applied by the serving backend, as
everywhere else in this tree. A different prompt is a different measurement, so
this is stated rather than left to be inferred from the code.

**Grading** is upstream's ``answer_check(problem, response, answer, "perturb")``
verbatim, vendored in ``sieval.community.math_perturb``: the gold is wrapped in
``\\boxed{}`` and re-extracted (so a multi-valued label splits the same way a
prediction does), the response is extracted with ``extract_math_answer``, and
the two lists are compared by ``eval_math`` at ``prec=MAX_ABS_TOL`` (1e-7).
Offloaded to a worker under ``GRADE_TIMEOUT`` like every sympy-backed grader
here.

**Fidelity, measured.** Executing upstream's own functions on shared inputs — all
558 rows of the pinned data, and 9043 (row, synthetic response) cases spanning
every control-flow branch of its extractor — this port reproduces upstream's
extracted gold on 558/558, its extracted prediction on 9043/9043, and its verdict
on 9043/9043.

*Scope of that corpus, stated because it is narrower than it reads.* Every case
in it is ASCII, so it exercises no entry of ``_fix_unicode``'s replacement table
— the one place where a *character* rather than a code path is the thing being
ported. A damaged entry there is invisible to this measurement, and to ``ruff``,
which skips ``sieval/community``, and to a unified diff, which renders a folded
key identically to the original. So that table is pinned codepoint-by-codepoint
in ``tests/unit/community/test_math_perturb.py`` instead of being inferred from
verdict parity.

*The execution-safety divergence costs nothing here.* Re-running that corpus with
``parse_latex`` disabled in each module's own globals — which forces every
symbolic comparison down the ``parse_expr`` path the guards sit on — upstream and
this port agree on 9043/9043 verdicts. Same result the ``deepseek_math`` port
measured on GSM8K and MATH.

**Trap: upstream's ``backend="lark"`` is weaker than sympy's default, and that is
kept.** Upstream calls ``parse_latex(s, backend="lark")`` explicitly in its own
source (its comment: the backend "does not require
antlr4-python3-runtime==4.11"), so this is a code choice, not the broken-pinned-
dependency case ``gsm_plus`` documents, and the unqualified task tracks it.
Measured on the 571 gold atoms the two files extract to, ``parse_latex`` fails on
40 (7.0%) under lark and 32 (5.6%) under ANTLR; the sets are not nested — 13 fail
only under lark, 5 only under ANTLR. **The lark backend cannot parse ``\\pi`` at
all** (sympy 1.14): all 5 gold atoms containing it are in the lark-only group,
along with comma-grouped integers (``270,000``), ``\\emptyset``, and interval /
set unions. What fails to parse falls through to ``parse_expr``, which fails too,
so those comparisons are decided by string equality — including for a correct
answer spelled differently. The 5 ANTLR-only failures are all
``\\begin{pmatrix}`` matrices, which ``math_equal`` handles in a branch of its own
before it ever parses. Verdict cost over the 9043-case corpus: 40 differ, all
lark-``False``/ANTLR-``True``; 33 are an artifact of the corpus appending ``\\%``
to a non-numeric gold, and the 5 realistic ones are all unicode-normalized
answers, i.e. exactly the o1/o3-mini case ``_fix_unicode`` was added for.

**No ``maj@k`` / ``self_consistency``.** A MATH-Perturb answer is a *set* of
atoms, and upstream supplies equivalence only pairwise (``math_equal``), never as
a canonical key — so clustering rollouts would need a canonicalization upstream
does not define. A vote built on string equality over set reprs would be a
plausible-looking wrong number, so the block is asked for without votes.

**Not ported: the paper's third column, ``Original``.** Table 1 scores the 279
*unperturbed* MATH problems alongside these two, and the drop against it is the
paper's headline. Upstream publishes no mapping from ``problem_id`` back to the
seed MATH row and ships no ``Original`` file, so reconstructing it means matching
279 edited statements against MATH's 12500 — inference, not a port. The
``dataset_type="original"`` arm of ``answer_check`` is carried anyway, so the
grader is ready if that mapping is ever established.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import importlib.util
from collections import defaultdict
from collections.abc import Sequence
from typing import cast, override

from loguru import logger

from sieval.community.math_perturb import (
    MAX_ABS_TOL,
    eval_math,
    extract_ground_truth_answer,
    extract_predicted_answer,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    JudgementRecord,
    NonRetriableSampleError,
    PredictionRecord,
    PromptRecord,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    ProblemGrouping,
    health_metrics,
    merge_metrics,
    metric_interval,
    rollout_metrics,
    rollout_view,
    sampling_report,
    ungated_intervals,
)
from sieval.core.types import JSONValue
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound

#: Upstream's evaluation package, pinned. One constant so the two leaves cite one
#: commit. The entry point ``answer_check`` lives one level up, in the same
#: commit's ``evaluate.py``; this points at the four-file package it composes,
#: which is the bulk of what is ported.
MATH_PERTURB_UPSTREAM_URL = (
    "https://github.com/Kaffaljidhmah2/MATH-Perturb/tree/"
    "df4840f680fce405c9449008564574961c7f4df1/evaluation"
)

#: Everything both leaves say. Two copies is how they would come to disagree.
MATH_PERTURB_REFERENCE_NOTES = (
    "PROMPT IS A SIEVAL CHOICE: upstream ships an evaluation package and NO "
    "inference code, and the paper specifies only zero-shot CoT with no tool "
    "use, so no upstream prompt exists to port. Used here: DeepSeek-Math's "
    "zero-shot CoT user turn, problem + '\\nPlease reason step by step, and put "
    "your final answer within \\boxed{}.', chat template applied by the serving "
    "backend -- chosen because MATH-Perturb's grader IS DeepSeek-Math's grader "
    "(upstream's evaluation/README.md says so) and that instruction is what its "
    "extractor was written against. A different prompt is a different "
    "measurement. GRADING: upstream's answer_check(problem, response, answer, "
    "'perturb') verbatim, vendored in sieval.community.math_perturb -- gold "
    "wrapped in \\boxed{} and re-extracted so a multi-valued label splits like a "
    "prediction does, response extracted by extract_math_answer, both lists "
    "compared by eval_math at prec=MAX_ABS_TOL (1e-7, upstream's tightening of "
    "DeepSeek-Math's 1e-3). Offloaded to a worker under GRADE_TIMEOUT like every "
    "sympy-backed grader here. FIDELITY, MEASURED by executing upstream's own "
    "functions on shared inputs -- all 558 rows plus 9043 (row, synthetic "
    "response) cases spanning every control-flow branch of its extractor: "
    "extracted gold 558/558, extracted prediction 9043/9043, verdict 9043/9043. "
    "That corpus is ASCII, so it exercises NO entry of _fix_unicode's "
    "replacement table -- the one place a CHARACTER rather than a code path is "
    "what is being ported, and where a key folded to an ASCII lookalike is a "
    "live identity no-op that a diff renders identically and ruff never reads "
    "(it skips sieval/community). That table is pinned codepoint-by-codepoint in "
    "tests/unit/community/test_math_perturb.py instead. ONE DIVERGENCE, "
    "taken for execution safety rather than as a repair: upstream inherits "
    "DeepSeek-Math's hole unchanged -- a prediction reaches a bare parse_expr "
    "whose namespace carries __builtins__, and an unparseable one is returned as "
    "raw text to N, which sympifies it with sympy's own default namespace -- so "
    "a boxed __import__('os').system(...) runs while the sample still grades "
    "wrong. Here the parse is guarded (sieval.community._sympy_guards) and an "
    "unparseable answer refuses the comparison. MEASURED COST: ZERO -- rerunning "
    "the same 9043 cases with parse_latex disabled in each module's own globals, "
    "which forces every comparison down the guarded path, upstream and this port "
    "agree 9043/9043. TRAP -- upstream's backend='lark' is weaker than sympy's "
    "default ANTLR backend, and is KEPT because upstream's own source names it "
    "(a code choice, not gsm_plus's broken-pinned-dependency case). Over the 571 "
    "gold atoms these two files extract to, parse_latex fails on 40 (7.0%) under "
    "lark vs 32 (5.6%) under ANTLR, and the sets are not nested: 13 lark-only, 5 "
    "ANTLR-only. lark cannot parse \\pi AT ALL on sympy 1.14 -- all 5 \\pi atoms "
    "are lark-only failures, with comma-grouped integers (270,000), \\emptyset "
    "and interval/set unions; those comparisons fall through to parse_expr, fail "
    "there too, and are decided by string equality, so a correct answer spelled "
    "differently scores wrong. The 5 ANTLR-only failures are all "
    "\\begin{pmatrix} matrices, which math_equal handles in a branch of its own "
    "before parsing. Verdict cost over the 9043 cases: 40 differ, every one "
    "lark-False/ANTLR-True; 33 are a corpus artifact (\\% appended to a "
    "non-numeric gold) and the 5 realistic ones are all unicode-normalized "
    "answers, the o1/o3-mini case upstream added _fix_unicode for. NO maj@k / "
    "self_consistency: an answer here is a SET of atoms and upstream supplies "
    "equivalence only pairwise (math_equal), never as a canonical vote key, so "
    "the vote block is not computed rather than clustered on set reprs. REPEATS: "
    "upstream's leaderboard is one sample per problem and this task defaults to "
    "n=1; the paper's inference-time-scaling section draws N=64 (N=8 for "
    "o1-mini) for pass@k and self-consistency -- set n as a task arg "
    "(tasks.<name>.args.n), since the model's n is silently overridden "
    "call-time, and k>n is rejected at construction. DECODING: upstream "
    "publishes no single spec, only per-model recommendations on its project "
    "page (DeepSeek-R1 series temperature 0.6 / top_p 0.95 / 64k max length; "
    "QwQ-32B 32k / 0.6 / top_k 40 / 0.95; Claude-3.7-Sonnet extended thinking "
    "budget 56000, max 64000). Set them via models: / infer_args, not here. "
    "NOT PORTED: the paper's third column, Original -- the 279 UNPERTURBED MATH "
    "problems, against which both drops are measured. Upstream ships no Original "
    "file and no mapping from problem_id back to the seed MATH row, so building "
    "it means matching 279 edited statements against MATH's 12500, which is "
    "inference rather than a port; answer_check's dataset_type='original' arm is "
    "carried anyway so the grader is ready if that mapping is established. "
    "REFERENCE NUMBERS: cite the paper's Table 1 (arXiv:2502.06453v1, zero-shot "
    "CoT, All column). The project page keeps a larger leaderboard that is "
    "periodically re-run -- its numbers have MOVED from the paper's (o1-mini "
    "MATH-P-Hard 78.49 in Table 1, 79.69 there) and its CSV column order is "
    "Hard/Simple/Original, the REVERSE of the paper's, so a row read positionally "
    "off the wrong source is off by a whole benchmark. Only the CLOSED-model "
    "rows were re-run, though: all four open-weight rows on that CSV "
    "(Llama-3.1-8B-Instruct, Gemma-2-9b-it, Deepseek-math-7b-rl, "
    "Qwen2.5-Math-7B-Instruct) are digit-identical to Table 1, so the anchors "
    "usable without an API key are the same in both places. LICENSE: upstream is "
    "Apache-2.0, but its README restricts the data to academic research and asks "
    "in bold that it never be used as training data. "
    "ALIGNMENT, MEASURED (2026-08-31; sglang, one H100, greedy temperature=0, "
    "max_tokens=3072, n=1, chat template applied by the server): the anchor is "
    "deepseek-math-7b-rl, chosen because this prompt IS that model's own "
    "zero-shot CoT turn -- the one published row where the prompt is not a "
    "sieval choice. Three runs of one config reproduce it: MATH-P-Simple 33.33 / "
    "32.97 / 33.69, mean 33.33 against Table 1's 33.33; MATH-P-Hard 14.70 / "
    "13.98 / 13.98, mean 14.22 against 13.62, i.e. one problem below the "
    "observed minimum. The published pair is inside the measurement, which is "
    "the claim -- NOT a digit match, and a single run must not be quoted as one: "
    "a fourth run differing ONLY in client concurrency (8 rather than 32) lands "
    "30.82 / 12.54, so the batching regime is worth ~2.5 points on this model. "
    "Qwen2.5-Math-7B-Instruct does NOT reproduce: four runs give Simple 68.10 / "
    "66.67 / 67.03 / 66.67 and Hard 39.07 / 39.78 / 39.78 / 39.78 against "
    "published 51.61 / 27.24, a residual of ~15 points that is not measurement "
    "(same-config range 1.43 Simple, 0.71 Hard) and not the prompt: four prompt "
    "arms (this one, a generic system turn suppressing Qwen-Math's built-in "
    "system prompt, no boxed-answer instruction at all, and \"Let's think step "
    'by step") span 66.67-70.25 Simple and 38.35-39.78 Hard, and driving the '
    "bare text through the OpenAI completions endpoint with NO chat template "
    "gives 70.25 / "
    "40.50 -- higher, not lower. Its verdicts were also re-derived with "
    "upstream's own answer_check (below), so the residual is on the generation "
    "side of a row this port cannot reconstruct. Read the two rows accordingly: "
    "the open-weight anchors here are reproducible for the model whose native "
    "prompt this is, and indicative only for the others. "
    "max_tokens IS THE DOMINANT FREE PARAMETER, and upstream publishes none: "
    "same model, same prompt, only the budget moving, Simple runs 31.90 (512) / "
    "62.01 (1024) / 66.67 (2048) / 66.67 (3072) and Hard 14.34 / 31.18 / 40.14 / "
    "39.78, i.e. ~34 points of range on Simple against ~3 for the whole prompt "
    "sweep. At 512 the model is simply cut off (190 of 279 Simple responses hit "
    "the cap). The published 51.61 / 27.24 falls between the 512 and 1024 arms, "
    "which is the most economical account of that row -- a smaller generation "
    "budget, not a different port. Fix max_tokens before comparing anything here "
    "to Table 1; it is an infer_args knob, deliberately not a task constant. "
    "GRADER DIFFERENTIAL ON LIVE TEXT: 1116 real responses from those two models "
    "graded by upstream's unmodified answer_check agree with this port on gold, "
    "prediction and verdict 1116/1116, and every published report cell "
    "(pass@1, both seed splits, all seven subject cells, both n_problems) "
    "re-derives exactly from the per-sample judgements on disk. "
    "TRAP -- A MISSING lark WOULD DEGRADE SCORING SILENTLY, so the task refuses "
    "to build without it. parse_latex(backend='lark') raises ImportError when "
    "the package is absent, symbolic_equal's bare `except` swallows it "
    "(upstream's control flow, kept), and every symbolic comparison falls "
    "through to string equality. Measured on the same 1116 responses: 62.01 vs "
    "68.10 and 34.05 vs 39.07 (Qwen2.5-Math-7B-Instruct, Simple / Hard), 28.32 "
    "vs 33.33 and 13.62 vs 14.70 (deepseek-math-7b-rl) -- 5-6 points on three of "
    "four cells, always understating, with fails=0 and no exception raised "
    "anywhere. NO call-site handler can catch that, because the swallow is below "
    "it; and `lark` being in the `math` extra does not settle it either -- an "
    "environment that installed that extra BEFORE it was added there satisfies "
    "the group and still lacks the package. So __init__ checks for it and raises "
    "ImportError, before any inference budget is spent. "
    "SYMPY VERSION: upstream pins sympy==1.13.2 and sieval runs 1.14. Over the "
    "same 1116 responses that moves 3-4 verdicts per model, every one of them "
    "1.13.2-False / 1.14-True, and every one a complex number written with `i` "
    "(plus one interval union) -- so the newer sympy only accepts answers that "
    "are correct, worth at most +0.7 points. An environment difference, like "
    "gsm_plus's missing ANTLR runtime, not a code one."
)

# Verbatim from DeepSeek-Math's run_subset_parallel.py::markup_question
# (language="en", task="cot") -- see the module docstring on why this prompt.
COT_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

#: ``dataset_type`` argument every row here is graded under. Upstream's other
#: value, ``"original"``, belongs to the unported third column.
PERTURB = "perturb"

#: The paper's ``train`` / ``test`` columns: which MATH split the SEED problem
#: came from, carried per row as ``original_split``. Named ``seed`` rather than
#: ``original`` so nothing reads as the unported ``Original`` *set*.
SEED_SPLITS = ("train", "test")

#: MATH's seven subjects, carried per row as ``type``. Both testsets hold all
#: seven at the pinned commit, in identical counts (79 Algebra / 48 Intermediate
#: Algebra / 38 Counting & Probability / 36 Number Theory / 35 Prealgebra / 22
#: Precalculus / 21 Geometry). Named here for the same reason as
#: :data:`SEED_SPLITS`: so the breakdown publishes the same seven columns on
#: every run rather than only the ones a given draw happened to contain.
MATH_SUBJECTS = (
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
)


def require_lark_backend() -> None:
    """Refuse to build the task when the grader's LaTeX backend is missing.

    ``symbolic_equal`` reaches ``parse_latex(s, backend="lark")``, which raises
    ``ImportError`` when ``lark`` is absent — and upstream's bare ``except``,
    kept here, turns that raise into a ``False`` verdict. The failure is
    therefore invisible from inside grading: nothing propagates, ``fails`` stays
    0, and every symbolic comparison falls through to string equality. Measured
    on 1116 stored responses, that understates by 5–6 points on three of four
    cells — a plausible number, not an error.

    No call-site handler can see it, and ``deps_group="math"`` does not settle
    it either: an environment that installed that group *before* ``lark`` was
    added to it satisfies the group and still lacks the package. So it is
    checked once, here, where it costs a single ``find_spec`` and fails before
    any inference budget is spent rather than after a full run has produced a
    wrong score.
    """
    if importlib.util.find_spec("lark") is None:
        raise ImportError(
            "MATH-Perturb's grader needs the `lark` LaTeX backend, which is not "
            "installed. Without it every symbolic comparison falls back to "
            "string equality and this task under-scores by several points "
            "instead of failing. Install the `math` dependency group "
            "(`pdm install -G math`) — note that an environment which installed "
            "that group before `lark` was added to it satisfies the group and "
            "still lacks the package."
        )


def seed_score_key(seed_split: str) -> str:
    """Report key for one seed-split cell."""
    return f"score_seed_{seed_split}"


def seed_count_key(seed_split: str) -> str:
    """Population key one seed-split cell's interval is clustered on."""
    return f"n_problems_seed_{seed_split}"


def type_score_key(subject: str) -> str:
    """Report key for one MATH subject cell.

    ``"Counting & Probability"`` -> ``"score_type_counting_and_probability"``.
    """
    slug = subject.lower().replace("&", "and").replace("-", "_")
    return "score_type_" + "_".join(slug.split())


def gold_atoms(problem: str, answer: str) -> list[str]:
    """Upstream's extracted gold for one row, or raise if there is none.

    ``extract_ground_truth_answer`` wraps the label in ``\\boxed{}`` and runs the
    extractor over it, so a usable label always yields at least one non-empty
    atom. An empty list, or a list of nothing but empty atoms, means the row
    carries no ground truth — and a value-reference task must fail such a sample
    rather than record a verdict reached without one, or grade every unanswerable
    response correct against ``[""]``. Non-retriable: the miss is in the row, and
    no later attempt recovers it. The pinned data has no such row; this is the
    guard for a source that changes underneath.
    """
    atoms: list[str] = extract_ground_truth_answer(problem, answer, PERTURB)
    if not any(atom for atom in atoms):
        raise NonRetriableSampleError(
            f"MATH-Perturb row carries no extractable ground truth: answer="
            f"{answer!r} extracted to {atoms!r}."
        )
    return atoms


def _as_json(atoms: Sequence[str]) -> JSONValue:
    """Answer atoms as a record value.

    ``JSONValue``'s ``list`` arm is invariant, so a ``list[str]`` is not a
    ``list[JSONValue]`` even though every element of it is one. This is that
    widening and nothing else — the list is rebuilt, never reinterpreted.
    """
    return cast("JSONValue", list(atoms))


def grade_extracted(gold: Sequence[str], prediction: Sequence[str]) -> bool:
    """Upstream's verdict for one already-extracted (gold, prediction) pair.

    ``answer_check`` is exactly ``extract_ground_truth_answer`` +
    ``extract_predicted_answer`` + ``eval_math`` at ``prec=MAX_ABS_TOL``, and it
    re-extracts the prediction from the raw response text. That re-extraction is
    the same call ``postprocess`` already made, so composing the three steps
    scores the identical pair — while depending only on the postprocess record,
    which is what makes grading survive a resume that carries no ``infer_result``.

    A fresh dict per call because ``eval_math`` mutates the one it is given.

    Module-level so it can be handed to a worker process by name.
    """
    return bool(
        eval_math(
            {"answer": list(gold), "prediction": list(prediction)}, prec=MAX_ABS_TOL
        )
    )


def _per_problem_pass_at_1(final, n_requested: int) -> float:
    """One judged sample's ``pass@1``, computed the way the headline block does.

    Read through ``rollout_metrics`` rather than as ``mean(correct)`` so a cell
    and the headline cannot drift apart if that definition ever moves.
    """
    correct, _ = rollout_view(final)
    return rollout_metrics(correct, None, k=1, n_requested=n_requested)["pass@1"]


def _restrict(
    grouping: ProblemGrouping | None, positions: Sequence[int]
) -> ProblemGrouping | None:
    """*grouping* narrowed to *positions*, so a repeated split still collapses.

    Without this a cell over a repeated split would read each copy of a problem
    as an independent question and report an interval that is too narrow.
    """
    if grouping is None:
        return None
    keys = [grouping.keys[position] for position in positions]
    return ProblemGrouping(keys, len(set(keys)))


class MathPerturbZeroShotGenTask[TSample](
    Task[
        TSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one; `list[float]` carries an interval, and
        # `dict[str, str]` the `ci95_units` map naming each interval's unit.
        dict[str, float | str | list[float] | dict[str, str]],
    ]
):
    """Base for one MATH-Perturb testset; leaves supply the sample type."""

    def __init__(self, dataset, model, name: str | None = None, k: int = 1, n: int = 1):
        super().__init__(dataset=dataset, model=model, name=name)
        # Environment precondition first: it is the failure that would otherwise
        # be found only by reading the score.
        require_lark_backend()
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}. "
                "Raise the task arg `n` (tasks.<name>.args.n) to at least k — "
                "setting `n` on the model is silently overridden call-time."
            )
        self._k = k
        self._n = n

    @override
    async def preprocess(self, raw, ctx):
        # The extracted gold, not the raw label: it is the side `eval_math`
        # actually compares, and `raw_sample` is never serialized, so without it
        # a prompt row on disk carries no ground truth at all.
        return build_prompt_record(
            [{"role": "user", "content": raw["problem"] + COT_INSTRUCTION}],
            reference=_as_json(gold_atoms(raw["problem"], raw["answer"])),
            extra={
                "problem_id": raw["problem_id"],
                "original_split": raw["original_split"],
                "type": raw["type"],
            },
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        problem = ctx.raw_sample["problem"]
        predictions: list[JSONValue | None] = []
        for text in inf.texts:
            atoms = extract_predicted_answer(problem, text)
            # `None`, not `[]` or `[""]`: `build_prediction_record` derives
            # `extracted` from `prediction is not None`, so either would record a
            # failed extraction as a successful one and hide it from
            # `n_unextracted`. `any`, matching `gold_atoms` -- and the empty
            # ATOM, not just the empty list, is the reachable half: upstream's
            # extractor returns `[""]` for `\boxed{}`, `\boxed{ }`,
            # `\boxed{\text{}}`, `\boxed{\,}` and for a reply that stops at "The
            # answer is ", which is the shape a response truncated at
            # `max_tokens` takes -- the very case `n_unextracted` is read to
            # detect. Verdict-neutral either way (`math_equal` refuses an empty
            # prediction, and `gold_atoms` guarantees a non-empty gold), so this
            # moves the health count and no score.
            #
            # `any`, not `all`: a reply that boxes twice and leaves one blank
            # (`\boxed{42} and \boxed{}`) extracts to `["42", ""]` -- a real
            # prediction, which upstream's all-must-match rule then scores wrong.
            # A wrong answer is not a missing one, and recording it as
            # unextracted would blame the parser for the model's answer.
            predictions.append(_as_json(atoms) if any(atoms) else None)
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        gold = gold_atoms(raw["problem"], raw["answer"])
        rollouts = []
        for rollout in post["rollouts"]:
            index = rollout["index"]
            # `.get`, not `[]`: a None prediction is dropped on write, so
            # indexing raises KeyError on resume rather than reading "nothing
            # extracted" (`.claude/rules/records.md`).
            prediction = rollout.get("prediction")
            if prediction is None:
                rollouts.append(build_rollout_judgement(index, False))
                continue
            try:
                correct = await run_cpu_bound(
                    grade_extracted, gold, prediction, timeout=GRADE_TIMEOUT
                )
            except TimeoutError:
                # A grade that could not be computed IN TIME is a wrong answer,
                # not a failed run -- the prediction is a shape `simplify` cannot
                # bound, which is the model's problem. The contract every sibling
                # math grader keeps, and `report` counts fails in the denominator
                # so the accuracy is the same either way.
                #
                # Every OTHER exception propagates, and the sample lands in
                # `fails` as `exception::<class>`. A grader that is broken rather
                # than slow -- a dead worker, an OOM-killed child -- must not be
                # indistinguishable from a model that answered wrongly:
                # swallowed, it produced a low score on a run whose `fails` was 0
                # and whose only trace was a log line. Propagating costs nothing
                # -- raising here goes straight to FAILED, no re-inference, and
                # DENOMINATOR_REQUESTED already charges a fail as wrong -- so
                # `fails` plus that reason is the grader-error count, and no
                # metric is owed for it.
                #
                # What this handler CANNOT reach: anything the vendored grader
                # swallows below it. A missing `lark` raises ImportError inside
                # `parse_latex`, and `symbolic_equal`'s bare `except` -- upstream
                # control flow, kept -- turns it into a False verdict that never
                # becomes an exception here. That failure is caught at
                # construction instead (see `__init__`), because no call-site
                # handler can see it.
                logger.warning(
                    "Grading sample {} rollout {} exceeded {}s and was scored "
                    "wrong; the prediction is likely a shape `simplify` cannot "
                    "bound.",
                    ctx.sample_id,
                    index,
                    GRADE_TIMEOUT,
                )
                correct = False
            rollouts.append(build_rollout_judgement(index, bool(correct)))
        return True, build_judgement_record(
            _as_json(gold),
            rollouts,
            extra={
                "original_split": raw["original_split"],
                "type": raw["type"],
            },
        )

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        grouping = self.problem_groups(finals)
        rolled = sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=total,
            # See the module docstring: a set-valued answer has no canonical
            # vote key here, so the vote columns are not computed at all.
            votes=False,
            score_key="pass@1",
            grouping=grouping,
        )
        # Read back out of the shared block, so `score` cannot drift from it.
        pass_at_1 = rolled["pass@1"]
        report: dict[str, float | str | list[float] | dict[str, str]] = {
            "score": pass_at_1,
            "pass@1": pass_at_1,
            "fails": len(fails),
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # Every interval-bearing fragment is collected and folded ONCE at the
        # end, with `merge_metrics`, because each carries its own `ci95_units`
        # and a plain merge keeps only the last one's declarations -- silently,
        # since the intervals themselves all survive.
        intervals: list[dict[str, float | list[float] | dict[str, str]]] = [
            # Outside the n>1 gate, because the metrics they bracket are:
            # `pass@1` is published at every budget, and so is the headline
            # copied from it.
            ungated_intervals(rolled, metrics=("score", "pass@1"))
        ]
        if self._n > 1:
            # At n=1 the rest only restates `pass@1`. A superset of the fragment
            # above on the keys they share, and identical on those keys.
            intervals.append(rolled)

        # Per-problem pass@1, plus where each problem sits in `finals`, so a cell
        # can be restricted positionally rather than re-derived.
        seed_values: dict[str, list[float]] = defaultdict(list)
        seed_positions: dict[str, list[int]] = defaultdict(list)
        seed_denominators: dict[str, int] = defaultdict(int)
        type_correct: dict[str, float] = defaultdict(float)
        type_denominators: dict[str, int] = defaultdict(int)
        for position, ctx in enumerate(finals):
            extra = ctx.feedback_result["extra"]
            value = _per_problem_pass_at_1(ctx, self._n)
            seed_split = extra["original_split"]
            seed_values[seed_split].append(value)
            seed_positions[seed_split].append(position)
            seed_denominators[seed_split] += 1
            type_correct[extra["type"]] += value
            type_denominators[extra["type"]] += 1
        for ctx in fails:
            # A fail scores 0 but still owes its cells a denominator slot, per the
            # DENOMINATOR_REQUESTED declared above -- and `raw_sample` is the only
            # place those labels survive, since a fail never reached feedback. One
            # that died before the sample was attached carries neither, so it can
            # only be charged to the headline.
            if ctx.raw_sample is not None:
                seed_denominators[ctx.raw_sample["original_split"]] += 1
                type_denominators[ctx.raw_sample["type"]] += 1

        # Both seed cells are always published, at their declared denominator, so
        # a run whose split happens to hold no train-seeded rows reports 0.0 over
        # 0 rather than growing and losing a column between runs.
        for seed_split in SEED_SPLITS:
            values = seed_values[seed_split]
            denominator = seed_denominators[seed_split]
            report[seed_score_key(seed_split)] = (
                100 * sum(values) / denominator if denominator else 0.0
            )
            cell_grouping = _restrict(grouping, seed_positions[seed_split])
            # Published unconditionally, beside the rate it belongs to: a rate
            # over 20 problems and one over 200 are the same number and a
            # different claim, and the empty path needs the count most. The
            # expression is the one `metric_interval` would write for the same
            # arguments, so when an interval does exist and overwrites this in
            # the fold, it overwrites it with an identical value.
            #
            # Unrepeated -- every config in this tree -- each sample is its own
            # problem and this IS the declared denominator. On a repeated split
            # the two nouns come apart in one direction: the denominator charges
            # a wholly-failed problem as wrong (DENOMINATOR_REQUESTED above),
            # while a grouping counts the problems the run OBSERVED, since a
            # failed sample's copy number does not survive on `raw_sample`.
            # Nothing published is wrong -- `n_problems` cancels out of the
            # estimate entirely and a smaller one only widens the interval.
            report[seed_count_key(seed_split)] = float(
                denominator if cell_grouping is None else cell_grouping.n_problems
            )
            intervals.append(
                metric_interval(
                    seed_score_key(seed_split),
                    values,
                    denominator=denominator,
                    group_keys=None if cell_grouping is None else cell_grouping.keys,
                    n_problems=(
                        None if cell_grouping is None else cell_grouping.n_problems
                    ),
                    unit=seed_count_key(seed_split),
                )
            )
        # Subject cells get no interval: seven populations of 21-79 problems are a
        # breakdown, not seven headlines, and one count per cell would be seven
        # more keys nothing ranks on. They are still published for all seven
        # subjects rather than only the observed ones -- same reason the seed
        # cells above are: a column that appears only when the draw happens to
        # contain it is one a consumer cannot key on, and a `limit` or a `filter`
        # is enough to drop one. The UNION, not `MATH_SUBJECTS` alone, so a
        # source that grows an eighth subject reports it rather than dropping it
        # silently -- the failure a fixed list would introduce here.
        for subject in sorted(set(MATH_SUBJECTS) | set(type_denominators)):
            denominator = type_denominators[subject]
            report[type_score_key(subject)] = (
                100 * type_correct[subject] / denominator if denominator else 0.0
            )
        # Extraction health is a fact about the parser, not about the draw, and
        # n=1 is where a stopped extractor hides longest -- so it is ungated.
        report.update(health_metrics(finals))
        # `report` holds no interval keys of its own, so the folded block can be
        # merged onto it plainly; the two seed counts it wrote are overwritten
        # with the identical values `metric_interval` computed for them.
        return report | merge_metrics(*intervals)
