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

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ChatModel, ModelOutput
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
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

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[rf"\boxed{{{ANSWER}}}"])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(field: str) -> dict[str, str]:
    return {field: PROBLEM, "answer": ANSWER}


def _build(task_cls, dataset_cls, field, *, k: int = 1, n: int = 1):
    rows = HFDataset.from_list([_sample(field)])
    dataset = dataset_cls(_hf_dict=HFDatasetDict({"train": rows, "test": rows}))
    return task_cls(dataset, _StubChatModel(), k=k, n=n)


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
