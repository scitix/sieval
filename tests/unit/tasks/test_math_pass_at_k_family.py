"""Shared contract for the pass@k math tasks.

These thirteen task modules are clones of one another in ``__init__`` and
``report``. Asserting the contract once, over all of them, is what stops a fix
landing in one file and silently drifting in the other twelve — the failure mode
that produced the ``k > n`` and report-key-set bugs in the first place. The
estimators themselves now live in :mod:`sieval.core.tasks.metrics` and are tested
there; what is asserted here is that every member routes through them and reports
the same keys.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sys

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ChatModel, ModelOutput, Request, Response
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import interval_declaration_problems
from sieval.datasets.aime_2024 import AIME2024Dataset
from sieval.datasets.aime_2025 import AIME2025Dataset
from sieval.datasets.aime_2026 import AIME2026Dataset
from sieval.datasets.apex_2025 import Apex2025Dataset
from sieval.datasets.apex_shortlist_2025 import ApexShortlist2025Dataset
from sieval.datasets.brumo_2025 import BRUMO2025Dataset
from sieval.datasets.cmimc_2025 import CMIMC2025Dataset
from sieval.datasets.hmmt_feb_2025 import HMMTFeb2025Dataset
from sieval.datasets.hmmt_feb_2026 import HMMTFeb2026Dataset
from sieval.datasets.hmmt_nov_2025 import HMMTNov2025Dataset
from sieval.datasets.imo_answer_bench import IMOAnswerBenchDataset
from sieval.datasets.math_500 import MATH500Dataset
from sieval.datasets.smt_2025 import SMT2025Dataset
from sieval.tasks.aime_2024_0shot_gen import AIME2024ZeroShotGenTask
from sieval.tasks.aime_2025_0shot_gen import AIME2025ZeroShotGenTask
from sieval.tasks.aime_2026_0shot_gen import AIME2026ZeroShotGenTask
from sieval.tasks.apex_2025_0shot_gen import Apex2025ZeroShotGenTask
from sieval.tasks.apex_shortlist_2025_0shot_gen import ApexShortlist2025ZeroShotGenTask
from sieval.tasks.brumo_2025_0shot_gen import BRUMO2025ZeroShotGenTask
from sieval.tasks.cmimc_2025_0shot_gen import CMIMC2025ZeroShotGenTask
from sieval.tasks.hmmt_feb_2025_0shot_gen import HMMTFeb2025ZeroShotGenTask
from sieval.tasks.hmmt_feb_2026_0shot_gen import HMMTFeb2026ZeroShotGenTask
from sieval.tasks.hmmt_nov_2025_0shot_gen import HMMTNov2025ZeroShotGenTask
from sieval.tasks.imo_answer_bench_0shot_gen import IMOAnswerBenchZeroShotGenTask
from sieval.tasks.math_500_0shot_gen import MATH500ZeroShotGenTask
from sieval.tasks.smt_2025_0shot_gen import SMT2025ZeroShotGenTask
from tests.conftest import HandlerTransport

PROBLEM = "What is 6 times 7?"
ANSWER = "42"

# (task_cls, dataset_cls, sample field holding the problem statement).
# The field is whatever the loader exposes: every MathArena/HF source ships
# `problem`, so only opencompass/AIME2025 (which ships `question`) differs.
FAMILY = [
    (AIME2024ZeroShotGenTask, AIME2024Dataset, "problem"),
    (AIME2025ZeroShotGenTask, AIME2025Dataset, "question"),
    (AIME2026ZeroShotGenTask, AIME2026Dataset, "problem"),
    (HMMTFeb2025ZeroShotGenTask, HMMTFeb2025Dataset, "problem"),
    (HMMTFeb2026ZeroShotGenTask, HMMTFeb2026Dataset, "problem"),
    (HMMTNov2025ZeroShotGenTask, HMMTNov2025Dataset, "problem"),
    (MATH500ZeroShotGenTask, MATH500Dataset, "problem"),
    (IMOAnswerBenchZeroShotGenTask, IMOAnswerBenchDataset, "problem"),
    (BRUMO2025ZeroShotGenTask, BRUMO2025Dataset, "problem"),
    (SMT2025ZeroShotGenTask, SMT2025Dataset, "problem"),
    (CMIMC2025ZeroShotGenTask, CMIMC2025Dataset, "problem"),
    (Apex2025ZeroShotGenTask, Apex2025Dataset, "problem"),
    (ApexShortlist2025ZeroShotGenTask, ApexShortlist2025Dataset, "problem"),
]
IDS = [t.__name__ for t, _, _ in FAMILY]


def _feedback(k: int):
    """A `k`-rollout all-correct `JudgementRecord`.

    The whole family now reads one shape, so this takes no task class: the fork
    that built a legacy list-of-dicts for unmigrated members is gone.
    """
    return build_judgement_record(
        ANSWER, [build_rollout_judgement(i, True) for i in range(k)]
    )


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=(rf"\boxed{{{ANSWER}}}",) * req.sampling.n)


def _sample(field: str) -> dict[str, str]:
    return {field: PROBLEM, "answer": ANSWER}


def _build(task_cls, dataset_cls, field, *, k: int = 1, n: int = 1):
    rows = HFDataset.from_list([_sample(field)])
    dataset = dataset_cls(_hf_dict=HFDatasetDict({"train": rows, "test": rows}))
    return task_cls(dataset, _StubChatModel(), k=k, n=n)


def _grader_module(task_cls):
    """The module whose `run_cpu_bound` name the task actually calls.

    Each member does `from sieval.core.utils.offload import run_cpu_bound`, so the
    binding that matters is the one in the TASK's namespace. Patching
    `offload.run_cpu_bound` instead resolves fine and intercepts nothing.

    Resolved from where `feedback` is DEFINED, not from the task class: a member
    that inherits `feedback` from a shared base grades through the base module's
    binding, and `task_cls.__module__` would name the leaf — which has no
    `run_cpu_bound` to patch. `math_perturb_{simple,hard}` are that shape today.
    """
    return sys.modules[task_cls.feedback.__module__]


class _Raiser:
    """An async `run_cpu_bound` stand-in that always raises *exc*.

    Counts its calls, so a test cannot pass because the patch was never reached —
    which is what a rename of the grading call site would otherwise look like. A
    class rather than a closure with a function attribute, which `ty` rejects.
    """

    def __init__(self, exc: type[BaseException]):
        self._exc = exc
        self.calls = 0

    async def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise self._exc("grader stub")


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
def test_k_greater_than_n_is_rejected(task_cls, dataset_cls, field):
    # Without the guard this constructs fine and every pass@4 comes out 0.0 —
    # a confidently wrong headline number rather than an error.
    with pytest.raises(ValueError, match=r"pass@4"):
        _build(task_cls, dataset_cls, field, k=4, n=1)


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
def test_k_equal_to_n_is_accepted(task_cls, dataset_cls, field):
    # The guard must reject only k > n, not the legitimate k == n.
    assert _build(task_cls, dataset_cls, field, k=4, n=4) is not None


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.anyio
async def test_report_key_set_is_identical_when_empty(task_cls, dataset_cls, field, k):
    task = _build(task_cls, dataset_cls, field, k=k, n=k)
    raw = _sample(field)
    populated = await task.report(
        [
            TaskContext(
                sample_id=0,
                raw_sample=raw,
                feedback_result=_feedback(k),
                # A CLEAN run: predictions present, so `maj@k` is computable and
                # the populated key set is the widest one this budget produces.
                postprocess_result=build_prediction_record([ANSWER] * k),
            )
        ],
        [],
    )
    empty = await task.report([], [])

    # An empty run previously dropped `pass@1`/`pass@k`, so a consumer reading
    # those keys hit a KeyError only on fully-failed runs.
    assert set(empty) == set(populated)
    assert "pass@1" in empty
    if k > 1:
        # The budget lives in the `n`/`k` fields, never in the key, so the
        # leaderboard column keeps its identity when the budget changes.
        assert {"avg@n", "pass@k", "maj@k", "n", "k", "n_short"} <= set(empty)
        assert f"pass@{k}" not in empty
    else:
        # n == 1: nothing was drawn, so there is no draw to describe.
        assert not {"avg@n", "pass@k", "maj@k", "n", "k", "n_short"} & set(empty)

    # One problem, so nothing here has dispersion to bracket and neither report
    # carries an interval at all. That is its own contract -- a report with no
    # intervals declares no units -- and it is all this case can pin: the
    # per-metric declarations need two problems to exist, which is what
    # `test_report_carries_an_interval_around_the_headline` supplies.
    for shape, report in (("populated", populated), ("empty", empty)):
        assert interval_declaration_problems(report) == [], shape
        assert not [key for key in report if key.endswith("_ci95")], shape


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.anyio
async def test_report_carries_an_interval_around_the_headline(
    task_cls, dataset_cls, field, k
):
    # Two problems split evenly is the smallest case with genuine spread --
    # `wilson_interval` needs >= 2 problems and 0 < p < 1 to emit anything.
    #
    # Both budgets, because `k > 1` is where the whole sampling block folds in on
    # top of the always-published pair, and that is the fold whose declarations
    # can go missing.
    task = _build(task_cls, dataset_cls, field, k=k, n=k)
    raw = _sample(field)
    report = await task.report(
        [
            TaskContext(
                sample_id=0,
                raw_sample=raw,
                feedback_result=_feedback(k),
                postprocess_result=build_prediction_record([ANSWER] * k),
            ),
            TaskContext(
                sample_id=1,
                raw_sample=raw,
                feedback_result=build_judgement_record(
                    ANSWER, [build_rollout_judgement(i, False) for i in range(k)]
                ),
                postprocess_result=build_prediction_record(["0"] * k),
            ),
        ],
        [],
    )
    lo, hi = report["score_ci95"]
    assert lo < report["score"] < hi
    assert report["n_problems"] == 2
    # Every interval the block published, not just the headline's: a per-metric
    # key is built from a metric NAME, so `check_preflight.py` cannot enumerate
    # them and the runner reaches them only AFTER a finished run has written its
    # `report.json`. At k > 1 that is five more intervals than the pair
    # `ungated_intervals` checks on its way through.
    assert interval_declaration_problems(report) == []


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.anyio
async def test_maj_at_k_clusters_equivalent_latex_into_one_vote(
    task_cls, dataset_cls, field
):
    # Votes are clustered on the canonicalizer this family already applies to
    # its golds, so `\dfrac{1}{2}` and `\frac{1}{2}` are one answer. Without it
    # they split 2-2 with `1/3`, and a tie is not a majority -- maj@k would read
    # 0.0 for a model that gave the same answer three times out of four.
    task = _build(task_cls, dataset_cls, field, k=4, n=4)
    raw = _sample(field)
    populated = await task.report(
        [
            TaskContext(
                sample_id=0,
                raw_sample=raw,
                feedback_result=build_judgement_record(
                    r"\frac{1}{2}",
                    [build_rollout_judgement(i, i != 3) for i in range(4)],
                ),
                postprocess_result=build_prediction_record(
                    [
                        r"\frac{1}{2}",
                        r"\dfrac{1}{2}",
                        r"\left(\frac{1}{2}\right)",
                        "1/3",
                    ]
                ),
            )
        ],
        [],
    )
    assert populated["pass@1"] == pytest.approx(75.0)
    assert populated["maj@k"] == pytest.approx(100.0)


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.anyio
async def test_report_survives_an_answer_that_breaks_the_canonicalizer(
    task_cls, dataset_cls, field
):
    # `strip_string` indexes into what it is repairing, so a bare trailing
    # `\frac` or `\sqrt` raises IndexError -- fine on curated golds, but these
    # are raw model answers arriving at the end of a scored run. An unclustered
    # vote costs one cluster; an exception costs the whole report.
    task = _build(task_cls, dataset_cls, field, k=2, n=2)
    raw = _sample(field)
    report = await task.report(
        [
            TaskContext(
                sample_id=0,
                raw_sample=raw,
                feedback_result=_feedback(2),
                postprocess_result=build_prediction_record(["\\frac", "\\sqrt"]),
            )
        ],
        [],
    )
    assert report["pass@1"] == pytest.approx(100.0)


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.anyio
async def test_preprocess_reads_the_field_its_loader_exposes(
    task_cls, dataset_cls, field
):
    # Guards the loader/task field-name contract: a task still reading the old
    # `question` key against a `problem` loader would KeyError here.
    task = _build(task_cls, dataset_cls, field)
    raw = _sample(field)

    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert PROBLEM in str(pre["prompt"])
    # The whole family is on the protocol now, so the gold reaching disk from
    # preprocess is a family-wide contract rather than a per-task detail.
    assert pre["reference"] == ANSWER


# The matharena-sourced subset of FAMILY: only these mirror upstream's grader.py,
# so only these owe it the list_answer rule. AIME 2024/2025, MATH-500 and
# IMO-AnswerBench answer to different references and are deliberately excluded.
MATHARENA_FAMILY = [
    (AIME2026ZeroShotGenTask, AIME2026Dataset, "problem"),
    (HMMTFeb2025ZeroShotGenTask, HMMTFeb2025Dataset, "problem"),
    (HMMTFeb2026ZeroShotGenTask, HMMTFeb2026Dataset, "problem"),
    (HMMTNov2025ZeroShotGenTask, HMMTNov2025Dataset, "problem"),
    (BRUMO2025ZeroShotGenTask, BRUMO2025Dataset, "problem"),
    (SMT2025ZeroShotGenTask, SMT2025Dataset, "problem"),
    (CMIMC2025ZeroShotGenTask, CMIMC2025Dataset, "problem"),
    (Apex2025ZeroShotGenTask, Apex2025Dataset, "problem"),
    (ApexShortlist2025ZeroShotGenTask, ApexShortlist2025Dataset, "problem"),
]
MATHARENA_IDS = [t.__name__ for t, _, _ in MATHARENA_FAMILY]

TWO_BOXES = "Working it out.\nThe roots are \\boxed{2} and \\boxed{3}."


@pytest.mark.parametrize(
    ("task_cls", "dataset_cls", "field"), MATHARENA_FAMILY, ids=MATHARENA_IDS
)
@pytest.mark.anyio
async def test_postprocess_derives_list_answer_from_the_gold(
    task_cls, dataset_cls, field
):
    # grader.py keys list_answer off a comma in the gold; every matharena port must
    # do the same, or a piecewise-boxed multi-part answer scores 0 here and 1
    # upstream (brumo p23, smt p32, hmmt_feb_2025 p10).
    task = _build(task_cls, dataset_cls, field)
    inf = ModelOutput(model=task.model.meta(), texts=[TWO_BOXES])

    listed = await task.postprocess(
        inf, TaskContext(sample_id=0, raw_sample={field: PROBLEM, "answer": "2,3"})
    )
    assert listed["rollouts"][0]["prediction"] == "2,3"

    scalar = await task.postprocess(
        inf, TaskContext(sample_id=0, raw_sample={field: PROBLEM, "answer": "3"})
    )
    assert scalar["rollouts"][0]["prediction"] == "3"


@pytest.mark.parametrize(
    ("task_cls", "dataset_cls", "field"), MATHARENA_FAMILY, ids=MATHARENA_IDS
)
@pytest.mark.anyio
async def test_postprocess_survives_a_missing_raw_sample(task_cls, dataset_cls, field):
    # raw_sample is Optional and the runner backfills it, but postprocess must
    # degrade to upstream's default rather than raise if it is ever absent —
    # the resume path is exactly where #70's KeyError hid.
    task = _build(task_cls, dataset_cls, field)
    inf = ModelOutput(model=task.model.meta(), texts=[TWO_BOXES])
    post = await task.postprocess(inf, TaskContext(sample_id=0))
    assert post["rollouts"][0]["prediction"] == "3"


# --- grading failure: only a TIMEOUT is a wrong answer -----------------------


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.anyio
async def test_a_grader_timeout_is_a_wrong_answer(
    task_cls, dataset_cls, field, monkeypatch
):
    """The half that must stay swallowed — upstream's contract for a slow grade.

    A prediction `simplify` cannot bound is the model's problem, not the run's,
    and `report` counts fails in the denominator either way, so propagating this
    one would only trade a truthful number for a scarier-looking one.
    """
    task = _build(task_cls, dataset_cls, field)
    stub = _Raiser(TimeoutError)
    monkeypatch.setattr(_grader_module(task_cls), "run_cpu_bound", stub)
    raw = _sample(field)
    post = build_prediction_record(["0"])
    _, judgement = await task.feedback(
        post, TaskContext(sample_id=0, raw_sample=raw, postprocess_result=post)
    )
    assert judgement["rollouts"][0]["correct"] is False
    assert stub.calls > 0, "the grading call site moved; this test intercepted nothing"


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.parametrize("exc", [ValueError, AttributeError, ImportError, OSError])
@pytest.mark.anyio
async def test_a_broken_grader_propagates_instead_of_scoring_zero(
    task_cls, dataset_cls, field, exc, monkeypatch
):
    """A grader that is BROKEN rather than slow must not read as a wrong answer.

    Swallowed, every one of these scored the sample 0 and left `fails` at 0, so a
    dead worker or an optional dependency missing from the environment produced a
    low score on a run that looked clean — the shape a missing LaTeX backend
    already takes on this family (worth 5-6 points on MATH-Perturb, 1.24 pp on
    MATH). Propagated, the runner writes `exception::<class>` on the sample and
    `fails` becomes the signal; nothing new has to be counted.
    """
    task = _build(task_cls, dataset_cls, field)
    stub = _Raiser(exc)
    monkeypatch.setattr(_grader_module(task_cls), "run_cpu_bound", stub)
    raw = _sample(field)
    post = build_prediction_record(["0"])
    with pytest.raises(exc):
        await task.feedback(
            post, TaskContext(sample_id=0, raw_sample=raw, postprocess_result=post)
        )
    assert stub.calls > 0, "the grading call site moved; this test intercepted nothing"


@pytest.mark.parametrize(("task_cls", "dataset_cls", "field"), FAMILY, ids=IDS)
@pytest.mark.anyio
async def test_moving_a_sample_into_fails_does_not_move_the_headline(
    task_cls, dataset_cls, field
):
    """Why narrowing is score-neutral, asserted rather than argued.

    Every member declares `DENOMINATOR_REQUESTED`, so `finals + fails` is the
    denominator and a fail is already charged as wrong. The two readings of one
    ungradeable sample — a judged-wrong final, or a fail — must therefore give
    the same headline, or this change would be a scoring change wearing a
    robustness label.
    """
    task = _build(task_cls, dataset_cls, field)
    raw = _sample(field)

    def _judged(sample_id: int, *, correct: bool) -> TaskContext:
        return TaskContext(
            sample_id=sample_id,
            raw_sample=raw,
            feedback_result=build_judgement_record(
                ANSWER, [build_rollout_judgement(0, correct)]
            ),
            postprocess_result=build_prediction_record([ANSWER if correct else "0"]),
        )

    # Four problems, and the SURVIVORS must stay mixed: `wilson_interval` needs
    # >= 2 problems and 0 < p < 1, so a fixture whose remaining finals are all
    # correct would drop `score_ci95` / `n_problems` and hide the comparison
    # behind a KeyError rather than making it.
    survivors = [_judged(0, correct=True), _judged(1, correct=True)]
    survivors.append(_judged(2, correct=False))
    # What the runner builds for a stage that raised: no judgement, no prediction.
    ungradeable_as_fail = TaskContext(sample_id=3, raw_sample=raw)

    before = await task.report([*survivors, _judged(3, correct=False)], [])
    after = await task.report(survivors, [ungradeable_as_fail])

    assert before["score"] == after["score"] == 50.0
    assert before["fails"] == 0
    assert after["fails"] == 1
    # Everything but `fails` and the intervals is IDENTICAL -- including
    # `n_problems`, which is the REQUESTED population (4 either way) and not a
    # count of what came back. Asserted as a whole-dict comparison rather than
    # key by key, so a member that publishes an extra column cannot drift
    # unnoticed through a test that only checks the keys this one thought of.
    moves = {"fails", "score_ci95", "pass@1_ci95"}
    assert set(before) == set(after)
    assert {k: v for k, v in before.items() if k not in moves} == {
        k: v for k, v in after.items() if k not in moves
    }
    # The intervals DO move, and not always outward: they are estimated over the
    # units that came back while being scaled to the requested denominator, so
    # dropping one shifts the spread in whichever direction that unit's value
    # pulled it. Here the survivors are less dispersed than the full set, so the
    # bound tightens. Pre-existing `_clustered_interval` semantics, reached more
    # often now -- stated rather than asserted in a direction.
    assert before["score_ci95"] != after["score_ci95"]
    assert after["score_ci95"][0] < after["score"] < after["score_ci95"][1]
    assert interval_declaration_problems(after) == []
