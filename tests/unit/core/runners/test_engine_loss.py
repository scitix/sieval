"""Tests for sieval.core.runners.engine_loss.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import anyio
import pytest

from sieval.core.runners.engine_loss import EngineDiedError, EngineGuard, EngineLoss
from sieval.core.tasks.consts import TaskAction

E = frozenset({"engine"})


class TestEngineLoss:
    @pytest.mark.anyio
    async def test_a_stage_needing_no_engine_runs_after_a_loss(self):
        loss = EngineLoss()
        loss.mark_lost("engine", "gone")
        ran = False
        with loss.guard(frozenset()):
            ran = True
        assert ran
        assert loss.failed == {}

    @pytest.mark.anyio
    async def test_a_stage_on_a_lost_engine_never_starts(self):
        loss = EngineLoss()
        loss.mark_lost("engine", "engine lost: exited 137")
        with pytest.raises(EngineDiedError, match="exited 137"), loss.guard(E, "t"):
            pytest.fail("ran")
        assert loss.failed == {"t": 1}

    @pytest.mark.anyio
    async def test_another_engine_s_loss_leaves_a_stage_alone(self):
        loss = EngineLoss()
        loss.mark_lost("other", "gone")
        with loss.guard(E, "t"):
            pass
        assert loss.failed == {}

    @pytest.mark.anyio
    async def test_a_stage_in_flight_is_cancelled(self):
        loss = EngineLoss()
        outcome: list[BaseException] = []

        async def stage():
            try:
                with loss.guard(E, "t"):
                    await anyio.sleep_forever()
            except EngineDiedError as exc:
                outcome.append(exc)

        async with anyio.create_task_group() as tg:
            tg.start_soon(stage)
            await anyio.sleep(0.01)
            loss.mark_lost("engine", "first")
            loss.mark_lost("engine", "second")

        assert [str(e) for e in outcome] == ["first"]
        assert loss.failed == {"t": 1}

    @pytest.mark.anyio
    async def test_an_error_after_the_loss_reads_as_the_loss(self):
        loss = EngineLoss()
        with (
            pytest.raises(EngineDiedError, match="gone") as caught,
            loss.guard(E, "t"),
        ):
            loss.mark_lost("engine", "gone")
            raise ConnectionError("refused")
        assert isinstance(caught.value.__cause__, ConnectionError)

    @pytest.mark.anyio
    async def test_an_error_with_the_engine_up_is_the_stage_s_own(self):
        loss = EngineLoss()
        with pytest.raises(ConnectionError), loss.guard(E, "t"):
            raise ConnectionError("refused")
        assert loss.failed == {}

    @pytest.mark.anyio
    async def test_an_outer_cancel_is_not_a_loss(self):
        loss = EngineLoss()
        with anyio.move_on_after(0.01) as outer:
            with loss.guard(E, "t"):
                await anyio.sleep_forever()
        assert outer.cancelled_caught
        assert loss.failed == {}


class TestEngineGuard:
    @pytest.mark.anyio
    async def test_guards_only_the_stages_that_call_the_engine(self):
        loss = EngineLoss()
        loss.mark_lost("engine", "gone")
        guard = EngineGuard(loss, "t", {TaskAction.INFER: E})
        with guard.guard(TaskAction.POSTPROCESS):
            pass
        with pytest.raises(EngineDiedError), guard.guard(TaskAction.INFER):
            pass
        assert loss.failed == {"t": 1}
