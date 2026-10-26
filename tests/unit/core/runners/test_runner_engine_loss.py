"""A ``TaskRunner`` whose engine is lost fails what still needs it as
``engine_died``, and a resume retries those samples.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import pytest

from sieval.core.runners import EngineGuard, EngineLoss, TaskRunner
from sieval.core.tasks.consts import ENGINE_DIED_REASON, TaskAction, TaskStage
from sieval.core.tasks.loader import TaskLoader
from tests.conftest import MockChatModel, MockDataset, make_config

from .test_runner import DEFAULT_ANSWERS, MockTask


class LosingModel(MockChatModel):
    """Answers *answered* prompts, then marks its engine lost and fails."""

    def __init__(self, loss: EngineLoss, answered: int) -> None:
        super().__init__(answers=DEFAULT_ANSWERS)
        self._loss = loss
        self._left = answered

    async def _stub_arun(self, req):
        if self._left <= 0:
            self._loss.mark_lost("engine", "engine lost: exited with code 137")
            raise ConnectionError("Connection refused")
        self._left -= 1
        return await super()._stub_arun(req)


def _runner(tmp_path, model, loss: EngineLoss | None, **config) -> TaskRunner:
    task = MockTask(dataset=MockDataset(), model=model, name="t")
    guard = (
        EngineGuard(loss, "t", {TaskAction.INFER: frozenset({"engine"})})
        if loss is not None
        else None
    )
    return TaskRunner(
        task,
        make_config(tmp_path, concurrency_limit=1, max_retries=1, **config),
        engine_guard=guard,
    )


async def _failed(tmp_path) -> dict:
    task = MockTask(dataset=MockDataset(), model=MockChatModel(), name="t")
    contexts = await TaskLoader(
        task=task, root_dir=tmp_path / "results"
    ).load_initial_state()
    return {sid: c for sid, c in contexts.items() if c.stage == TaskStage.FAILED}


@pytest.mark.anyio
async def test_samples_after_the_loss_fail_as_engine_died(tmp_path):
    loss = EngineLoss()
    report = await _runner(tmp_path, LosingModel(loss, answered=1), loss).arun()

    assert report["total"] == 3
    assert report["sieval"]["failed"] == 2
    failed = await _failed(tmp_path)
    assert len(failed) == 2
    assert {c.error_reason for c in failed.values()} == {ENGINE_DIED_REASON}
    assert {c.error_action for c in failed.values()} == {TaskAction.INFER}
    assert loss.failed == {"t": 2}


@pytest.mark.anyio
async def test_a_resume_retries_them_without_spending_a_retry(tmp_path):
    loss = EngineLoss()
    await _runner(tmp_path, LosingModel(loss, answered=1), loss).arun()

    for _ in range(3):  # more resumes than max_retries
        loss = EngineLoss()
        await _runner(tmp_path, LosingModel(loss, answered=0), loss, resume=True).arun()
    report = await _runner(
        tmp_path, MockChatModel(answers=DEFAULT_ANSWERS), None, resume=True
    ).arun()

    assert (report["accuracy"], report["sieval"]["failed"]) == (1.0, 0)


@pytest.mark.anyio
async def test_without_a_guard_the_failure_is_the_request_s_own(tmp_path):
    loss = EngineLoss()
    await _runner(tmp_path, LosingModel(loss, answered=1), None).arun()

    reasons = {c.error_reason for c in (await _failed(tmp_path)).values()}
    assert reasons == {"exception::ConnectionError"}
