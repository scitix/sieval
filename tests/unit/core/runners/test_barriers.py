"""Session-wide stage barriers (``sieval.core.runners.barriers``).

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import anyio
import pytest

from sieval.core.runners.barriers import (
    BarrierError,
    StageBarriers,
    check_barrier_stage,
    finished,
)
from sieval.core.tasks.consts import TaskAction, TaskStage
from sieval.core.tasks.context import TaskContext


class _Member:
    def __init__(self, name: str, unfinished: dict[str, int]) -> None:
        self.barrier_name = name
        self.left = dict(unfinished)

    def unfinished(self, stage: str) -> int:
        return self.left.get(stage, 0)


def _ctx(stage: TaskStage, iteration: int = 0) -> TaskContext:
    return TaskContext(sample_id=0, stage=stage, iteration=iteration)


class TestCheckBarrierStage:
    def test_names_every_stage_identity_but_data(self) -> None:
        for stage in ("preprocess", "inference", "postprocess", "feedback", "report"):
            assert check_barrier_stage(stage, "x") == stage

    @pytest.mark.parametrize("stage", ["data", "infer", "", None, 3])
    def test_refuses_anything_else_naming_where(self, stage: object) -> None:
        with pytest.raises(ValueError, match="runner_config.stop_after must be"):
            check_barrier_stage(stage, "runner_config.stop_after")


class TestFinished:
    def test_a_sample_finishes_a_stage_once_it_reaches_the_record_after_it(
        self,
    ) -> None:
        assert not finished(_ctx(TaskStage.PREPROCESSED), "inference")
        assert finished(_ctx(TaskStage.INFERRED), "inference")
        assert finished(_ctx(TaskStage.POSTPROCESSED), "inference")
        assert not finished(_ctx(TaskStage.POSTPROCESSED), "feedback")

    def test_a_terminal_sample_has_finished_every_stage(self) -> None:
        for terminal in (TaskStage.FINAL, TaskStage.FAILED):
            for stage in ("preprocess", "inference", "feedback", "report"):
                assert finished(_ctx(terminal), stage)

    def test_a_later_iteration_has_finished_the_first(self) -> None:
        assert finished(_ctx(TaskStage.INITIAL, iteration=1), "postprocess")


class TestStageBarriers:
    @pytest.mark.anyio
    async def test_waits_for_every_member_then_runs_the_hook_once(self) -> None:
        seen: list[str] = []

        async def hook(stage: str) -> None:
            seen.append(stage)

        barriers = StageBarriers(["inference"], on_barrier=hook, members=2)
        a = _Member("a", {"inference": 1})
        b = _Member("b", {"inference": 0})
        await barriers.join(a)
        await barriers.join(b)
        released = anyio.Event()

        async def wait() -> None:
            stopped = await barriers.before("postprocess")
            assert stopped is False
            released.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(wait)
            await anyio.sleep(0.01)
            assert not released.is_set()
            a.left["inference"] = 0
            await barriers.update()
            await barriers.update()
        assert released.is_set()
        assert seen == ["inference"]
        assert barriers.crossed == ["inference"]

    @pytest.mark.anyio
    async def test_a_member_not_yet_joined_holds_the_barrier(self) -> None:
        barriers = StageBarriers(["inference"], members=2)
        await barriers.join(_Member("a", {}))
        assert barriers.crossed == []
        await barriers.leave(_Member("b", {}))
        assert barriers.crossed == ["inference"]

    @pytest.mark.anyio
    async def test_a_stage_before_the_first_barrier_starts_at_once(self) -> None:
        barriers = StageBarriers(["postprocess"], members=1)
        await barriers.join(_Member("a", {"postprocess": 3}))
        with anyio.fail_after(1):
            assert await barriers.before_action(TaskAction.INFER) is False

    @pytest.mark.anyio
    async def test_stop_after_ends_the_run_at_its_barrier_and_skips_its_hook(
        self,
    ) -> None:
        seen: list[str] = []

        async def hook(stage: str) -> None:
            seen.append(stage)

        barriers = StageBarriers(
            ["preprocess"], stop_after="inference", on_barrier=hook, members=1
        )
        assert barriers.stages == ("preprocess", "inference")
        await barriers.join(_Member("a", {}))
        assert barriers.stops_before("postprocess")
        assert not barriers.stops_before("inference")
        assert await barriers.before("postprocess") is True
        # The hook still runs at the stop: an engine with `until` it stops.
        assert seen == ["preprocess", "inference"]

    @pytest.mark.anyio
    async def test_a_hook_that_raises_stops_every_waiter(self) -> None:
        async def hook(_: str) -> None:
            raise RuntimeError("engine failed to start")

        barriers = StageBarriers(["inference"], on_barrier=hook, members=1)
        with pytest.raises(RuntimeError, match="engine failed"):
            await barriers.join(_Member("a", {}))
        with pytest.raises(BarrierError, match="inference"):
            await barriers.before("postprocess")

    def test_no_barrier_is_falsy(self) -> None:
        assert not StageBarriers()
        assert StageBarriers(stop_after="report")
