"""Shared contract for the pass@k code tasks.

HumanEval x2, MBPP, LiveCodeBench x2: same ``k`` / ``n`` pair as the math
family, same metrics minus ``maj@k`` (two correct programs are not one answer).

Four of the five arrived without the ``k > n`` guard, where ``pass@k`` came out
a confident 0.0 beside a real ``pass@1``. Asserting it once over all five is
what stops that recurring in the next task added.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import build_judgement_record, build_rollout_judgement
from sieval.datasets.human_eval import HumanEvalDataset
from sieval.datasets.livecodebench_code_generation import LiveCodeBenchDataset
from sieval.datasets.mbpp import MBPPDataset
from sieval.tasks.human_eval_0shot_base_gen import HumanEvalZeroShotBaseGenTask
from sieval.tasks.human_eval_0shot_gen import HumanEvalZeroShotGenTask
from sieval.tasks.livecodebench_code_generation_0shot_gen import (
    LiveCodeBenchCodeGenerationZeroShotGenTask,
)
from sieval.tasks.livecodebench_code_generation_kshot_base_gen import (
    LiveCodeBenchCodeGenerationFewShotBaseGenTask,
)
from sieval.tasks.mbpp_kshot_base_gen import MBPPFewShotBaseGenTask


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=["pass"])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


class _StubGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=["pass"])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _human_eval() -> HumanEvalDataset:
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def f():\n",
        "canonical_solution": "    return 1\n",
        "test": "def check(f):\n    assert f() == 1\n",
        "entry_point": "f",
    }
    return HumanEvalDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([row])})
    )


def _mbpp() -> MBPPDataset:
    row = {
        "task_id": 11,
        "text": "Write a function to return 1.",
        "code": "def one():\n    return 1",
        "test_list": ["assert one() == 1"],
        "test_setup_code": "",
        "challenge_test_list": [],
    }
    return MBPPDataset(
        _hf_dict=HFDatasetDict(
            {
                "prompt": HFDataset.from_list([dict(row)]),
                "test": HFDataset.from_list([dict(row)]),
            }
        )
    )


def _livecodebench() -> LiveCodeBenchDataset:
    row = {
        "question_content": "Double the input.",
        "starter_code": "",
        "public_test_cases": json.dumps(
            [{"input": "1\n", "output": "2", "testtype": "stdin"}]
        ),
        "private_test_cases": json.dumps(
            [{"input": "2\n", "output": "4", "testtype": "stdin"}]
        ),
        "metadata": json.dumps({}),
    }
    return LiveCodeBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([row])})
    )


# (task class, dataset factory, model factory). Every member takes `k` and `n`.
FAMILY = [
    (HumanEvalZeroShotGenTask, _human_eval, _StubChatModel),
    (HumanEvalZeroShotBaseGenTask, _human_eval, _StubGenModel),
    (MBPPFewShotBaseGenTask, _mbpp, _StubGenModel),
    (LiveCodeBenchCodeGenerationZeroShotGenTask, _livecodebench, _StubChatModel),
    (LiveCodeBenchCodeGenerationFewShotBaseGenTask, _livecodebench, _StubGenModel),
]
IDS = [t.__name__ for t, _, _ in FAMILY]


def _build(task_cls, dataset_factory, model_factory, **kwargs):
    return task_cls(dataset_factory(), model_factory(), **kwargs)


@pytest.mark.parametrize(("task_cls", "dataset", "model"), FAMILY, ids=IDS)
def test_k_greater_than_n_is_rejected(task_cls, dataset, model):
    # Without the guard this constructs fine and pass@k comes out 0.0.
    with pytest.raises(ValueError, match=r"(?i)pass@2|k must be <= n"):
        _build(task_cls, dataset, model, k=2, n=1)


@pytest.mark.parametrize(("task_cls", "dataset", "model"), FAMILY, ids=IDS)
def test_k_equal_to_n_is_accepted(task_cls, dataset, model):
    # The guard must reject only k > n, not the legitimate k == n.
    assert _build(task_cls, dataset, model, k=4, n=4) is not None


@pytest.mark.parametrize(("task_cls", "dataset", "model"), FAMILY, ids=IDS)
@pytest.mark.anyio
async def test_report_omits_maj_at_k_for_programs(task_cls, dataset, model):
    # Absent rather than 0.0, which would read as "the majority was wrong".
    task = _build(task_cls, dataset, model, k=4, n=4)
    try:
        report = await task.report([], [])
    finally:
        await task.shutdown()

    assert {"pass@1", "avg@n", "pass@k", "n", "k", "n_short"} <= set(report)
    assert "maj@k" not in report
    assert report["score_key"] == "pass@1"


@pytest.mark.parametrize(("task_cls", "dataset", "model"), FAMILY, ids=IDS)
@pytest.mark.anyio
async def test_pass_at_k_column_carries_a_literal_k(task_cls, dataset, model):
    task = _build(task_cls, dataset, model, k=2, n=2)
    finals = [
        _final(
            build_judgement_record(
                None,
                [
                    build_rollout_judgement(0, True, extra={"msg": "passed"}),
                    build_rollout_judgement(1, False, extra={"msg": "failed"}),
                ],
            )
        )
    ]
    try:
        report = await task.report(finals, [])
    finally:
        await task.shutdown()

    # n=2, c=1. The column is `pass@k`, not `pass@2`.
    assert report["pass@1"] == pytest.approx(50.0)
    assert report["pass@k"] == pytest.approx(100.0)
    assert "pass@2" not in report
    assert (report["n"], report["k"]) == (2.0, 2.0)


def _final(judgement):
    from sieval.core.tasks import TaskContext, build_prediction_record

    ctx = TaskContext(sample_id=0, raw_sample={})
    ctx = ctx.to_preprocessed({"prompt": "p"})
    ctx = ctx.to_inferred("inf")
    ctx = ctx.to_postprocessed(build_prediction_record(["a", "b"]))
    return ctx.to_feedback(judgement).to_final()
