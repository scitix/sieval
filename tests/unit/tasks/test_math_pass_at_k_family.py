"""Shared contract for the pass@k math tasks.

These eight task modules are clones of one another in ``__init__``, ``report``
and ``_pass_at_k``. Asserting the contract once, over all of them, is what stops
a fix landing in one file and silently drifting in the other seven — the failure
mode that produced the ``k > n`` and report-key-set bugs in the first place.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ChatModel, ModelOutput
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_rollout_judgement,
)
from sieval.datasets.aime_2024 import AIME2024Dataset
from sieval.datasets.aime_2025 import AIME2025Dataset
from sieval.datasets.aime_2026 import AIME2026Dataset
from sieval.datasets.hmmt_feb_2025 import HMMTFeb2025Dataset
from sieval.datasets.hmmt_feb_2026 import HMMTFeb2026Dataset
from sieval.datasets.hmmt_nov_2025 import HMMTNov2025Dataset
from sieval.datasets.imo_answer_bench import IMOAnswerBenchDataset
from sieval.datasets.math_500 import MATH500Dataset
from sieval.tasks.aime_2024_0shot_gen import AIME2024ZeroShotGenTask
from sieval.tasks.aime_2025_0shot_gen import AIME2025ZeroShotGenTask
from sieval.tasks.aime_2026_0shot_gen import AIME2026ZeroShotGenTask
from sieval.tasks.hmmt_feb_2025_0shot_gen import HMMTFeb2025ZeroShotGenTask
from sieval.tasks.hmmt_feb_2026_0shot_gen import HMMTFeb2026ZeroShotGenTask
from sieval.tasks.hmmt_nov_2025_0shot_gen import HMMTNov2025ZeroShotGenTask
from sieval.tasks.imo_answer_bench_0shot_gen import IMOAnswerBenchZeroShotGenTask
from sieval.tasks.math_500_0shot_gen import MATH500ZeroShotGenTask

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
]
IDS = [t.__name__ for t, _, _ in FAMILY]

# Members already migrated to the stage-output protocol; the rest still return the
# legacy list-of-dicts feedback. Delete this set (and `_feedback`'s branch) once the
# whole family has migrated.
PROTOCOL_TASKS = {AIME2026ZeroShotGenTask, HMMTFeb2026ZeroShotGenTask}


def _feedback(task_cls, k: int):
    """A `k`-rollout all-correct feedback value in whichever shape *task_cls* reads."""
    if task_cls in PROTOCOL_TASKS:
        return build_judgement_record(
            ANSWER, [build_rollout_judgement(i, True) for i in range(k)]
        )
    return [{"correct": True, "answer": ANSWER} for _ in range(k)]


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
    feedback = _feedback(task_cls, k)
    populated = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)], []
    )
    empty = await task.report([], [])

    # An empty run previously dropped `pass@1`/`pass@k`, so a consumer reading
    # those keys hit a KeyError only on fully-failed runs.
    assert set(empty) == set(populated)
    assert "pass@1" in empty
    if k > 1:  # for k == 1 the pass@k key *is* pass@1
        assert f"pass@{k}" in empty


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

    assert PROBLEM in str(pre)
