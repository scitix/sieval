"""Unit tests for the LiveCodeBench code-generation 0-shot chat task.

Focused on the execution budget the task puts on the wire. Official LiveCodeBench
budgets each test case, not the suite: ``lcb_runner`` re-arms ``signal.alarm(timeout)``
inside the case loop of ``grade_call_based`` / ``grade_stdio``, with
``codegen_metrics(..., timeout=6)`` supplying the default, and ``check_correctness``
joining the worker at ``(timeout + 1) * n + 5`` as a backstop. ``timeout_per_case``
opts into that rule; absent it the task must send exactly what it always sent, which
is what keeps existing result dirs comparable.

The evaluator half of the same feature is covered by
``tests/unit/vendor/code_evaluator/test_exec_py_test.py``.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.livecodebench_code_generation import LiveCodeBenchDataset
from sieval.tasks.livecodebench_code_generation_0shot_gen import (
    LiveCodeBenchCodeGenerationZeroShotGenTask,
)

# One public + two private -> three cases, so the case-count scaling is visible.
_PUBLIC = [{"input": "1\n", "output": "2", "testtype": "stdin"}]
_PRIVATE = [
    {"input": "2\n", "output": "4", "testtype": "stdin"},
    {"input": "3\n", "output": "6", "testtype": "stdin"},
]
_N_CASES = len(_PUBLIC) + len(_PRIVATE)


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=["print(1)"])

    async def _alogprobs_impl(
        self,
        prompt,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


class _Response:
    """Enough of an httpx response for `feedback` -- a pass on every case."""

    @staticmethod
    def raise_for_status() -> None:
        return None

    @staticmethod
    def json() -> dict:
        return {
            "status": True,
            "msg": "",
            "data": {"n_cases": _N_CASES, "n_passed": _N_CASES},
        }


class _CapturingEvaluator:
    """Stands in for the code-eval service, recording what was asked of it."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.deadlines: list[float] = []

    async def post(self, url, *, json, timeout):
        _ = url
        self.bodies.append(json)
        self.deadlines.append(timeout)
        return _Response()

    async def aclose(self) -> None:
        return None


def _raw() -> dict:
    return {
        "question_content": "Double the input.",
        "starter_code": "",
        "public_test_cases": json.dumps(_PUBLIC),
        "private_test_cases": json.dumps(_PRIVATE),
        # No func_name -> stdio mode, the common LiveCodeBench shape.
        "metadata": json.dumps({}),
    }


async def _post_one(**kwargs) -> _CapturingEvaluator:
    """Run `feedback` for a single rollout and hand back what the evaluator saw."""
    dataset = LiveCodeBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_raw()])})
    )
    task = LiveCodeBenchCodeGenerationZeroShotGenTask(
        dataset, _StubChatModel(), **kwargs
    )
    await task._http_client.aclose()  # the real client is never used
    evaluator = _CapturingEvaluator()
    task._http_client = evaluator
    try:
        await task.feedback(
            build_prediction_record(["print(int(input()) * 2)"]),
            TaskContext(sample_id=0, raw_sample=_raw()),
        )
    finally:
        await task.shutdown()
    return evaluator


# --------------------------------------------------------------------------- #
# Execution budget on the wire
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_without_per_case_the_request_is_what_it_always_was():
    # The compatibility promise: absent `timeout_per_case` nothing about the
    # request changes, so no existing result dir moves.
    evaluator = await _post_one(timeout=30.0)

    (body,) = evaluator.bodies
    assert body["timeout"] == 30.0 + _N_CASES * 2.0 == 36.0
    # Not merely None -- the key must be absent, so an evaluator that predates the
    # field sees a byte-identical body.
    assert "timeout_per_case" not in body
    assert evaluator.deadlines == [36.0 + 2]


@pytest.mark.anyio
async def test_per_case_sends_the_field_and_upstreams_backstop_wall():
    evaluator = await _post_one(timeout=30.0, timeout_per_case=6.0)

    (body,) = evaluator.bodies
    assert body["timeout_per_case"] == 6.0
    # check_correctness joins its worker at (timeout + 1) * n + 5; `timeout`, the
    # whole-suite base, is deliberately NOT part of it any more.
    assert body["timeout"] == (6.0 + 1.0) * _N_CASES + 5.0 == 26.0
    assert evaluator.deadlines == [26.0 + 2]


@pytest.mark.anyio
async def test_http_deadline_stays_outside_the_suite_wall():
    # The client must not give up before the wall it just asked the server to
    # enforce, or a timing-out submission surfaces as a transport error instead.
    for kwargs in ({}, {"timeout_per_case": 6.0}):
        evaluator = await _post_one(timeout=30.0, **kwargs)
        (body,) = evaluator.bodies
        assert evaluator.deadlines[0] > body["timeout"]


# --------------------------------------------------------------------------- #
# report()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "msg",
    [
        "failed: subprocess timeout: 8.0s",  # whole-suite wall
        "failed: case timeout: 6.0s",  # per-case budget
        "failed: compile timeout: 6.0s",  # per-case budget, during compilation
    ],
)
@pytest.mark.anyio
async def test_every_timeout_message_reaches_the_timeouts_counter(msg):
    # `timeouts` is a substring check over the evaluator's free-text `msg`. The
    # per-case patch introduced two new wordings; both have to keep landing in it.
    dataset = LiveCodeBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_raw()])})
    )
    task = LiveCodeBenchCodeGenerationZeroShotGenTask(dataset, _StubChatModel())
    try:
        judgement = build_judgement_record(
            None, [build_rollout_judgement(0, False, extra={"msg": msg})]
        )
        report = await task.report(
            [TaskContext(sample_id=0, raw_sample=_raw(), feedback_result=judgement)],
            [],
        )
    finally:
        await task.shutdown()

    assert report["timeouts"] == 1
    assert report["score"] == 0.0
