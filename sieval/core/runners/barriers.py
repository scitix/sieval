"""Session-wide stage barriers: all samples finish a stage before any starts the next.

Opt-in; without a barrier samples stream as before. An optional async hook runs
once per barrier before any sample crosses it, and ``stop_after`` ends the run
at its barrier. Scheduling only: in no identity, never refuses a resume.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Protocol

import anyio

from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.context import TaskContext

if TYPE_CHECKING:
    from sieval.core.tasks.consts import TaskAction

#: The stage identity names, in pipeline order.
BARRIER_STAGES: tuple[str, ...] = (
    "preprocess",
    "inference",
    "postprocess",
    "feedback",
    "report",
)

#: The record stage that finishes each barrier stage. ``feedback`` is finished
#: only by a terminal sample; ``report`` has no per-sample state.
_FINISHED_AT: Mapping[str, TaskStage] = {
    "preprocess": TaskStage.PREPROCESSED,
    "inference": TaskStage.INFERRED,
    "postprocess": TaskStage.POSTPROCESSED,
}
_RANK: Mapping[TaskStage, int] = {
    TaskStage.INITIAL: 0,
    TaskStage.PREPROCESSED: 1,
    TaskStage.INFERRED: 2,
    TaskStage.POSTPROCESSED: 3,
    TaskStage.FEEDBACK: 4,
}

#: The stage each runner action starts. ``report`` is started by finalizing.
ACTION_STAGE: Mapping[str, str] = {
    "preprocess": "preprocess",
    "infer": "inference",
    "postprocess": "postprocess",
    "feedback": "feedback",
}

#: Barriers an iterating task cannot pass: it sends samples from ``feedback``
#: back to ``inference``.
ITERATION_BARRIERS: frozenset[str] = frozenset({"inference", "postprocess"})

BarrierHook = Callable[[str], Awaitable[None]]


class BarrierError(RuntimeError):
    """A barrier's hook raised; every runner waiting on it stops with this."""


def check_barrier_stage(stage: object, where: str) -> str:
    """*stage* when it names a barrier stage; raises :class:`ValueError` naming
    *where* otherwise."""
    if not isinstance(stage, str) or stage not in BARRIER_STAGES:
        raise ValueError(
            f"{where} must be one of {', '.join(BARRIER_STAGES)}; got {stage!r}"
        )
    return stage


def stage_index(stage: str) -> int:
    return BARRIER_STAGES.index(stage)


def finished(ctx: TaskContext, stage: str) -> bool:
    """Whether *ctx* has finished *stage* or failed. Monotone: a later
    iteration has finished every stage of an earlier one."""
    if ctx.is_terminal():
        return True
    target = _FINISHED_AT.get(stage)
    if target is None:
        return False
    if ctx.iteration > 0:
        return True
    return _RANK.get(ctx.stage, -1) >= _RANK[target]


class BarrierMember(Protocol):
    """What a barrier reads of a runner."""

    @property
    def barrier_name(self) -> str: ...

    def unfinished(self, stage: str) -> int: ...


class StageBarriers:
    """The barriers of one session, shared by every runner in it.

    *on_barrier* runs once per barrier except ``report``, which nothing
    follows. A barrier waits for all *members* runners to join or leave.
    """

    def __init__(
        self,
        barriers: Iterable[str] = (),
        *,
        stop_after: str | None = None,
        on_barrier: BarrierHook | None = None,
        members: int = 0,
    ) -> None:
        stages = {check_barrier_stage(s, "barrier") for s in barriers}
        if stop_after is not None:
            stages.add(check_barrier_stage(stop_after, "stop_after"))
        self.stages: tuple[str, ...] = tuple(sorted(stages, key=stage_index))
        self.stop_after = stop_after
        self._hook = on_barrier
        self._expected = members
        self._joined: dict[int, BarrierMember] = {}
        self._left: set[int] = set()
        self._released = {stage: anyio.Event() for stage in self.stages}
        self._failure: BaseException | None = None
        self._lock = anyio.Lock()
        #: Stages whose hook ran, in order; for tests and the run record.
        self.crossed: list[str] = []

    def __bool__(self) -> bool:
        return bool(self.stages)

    def set_members(self, count: int) -> None:
        self._expected = count

    def stops_before(self, stage: str) -> bool:
        """Whether the run ends before *stage* starts."""
        return self.stop_after is not None and stage_index(stage) > stage_index(
            self.stop_after
        )

    async def join(self, member: BarrierMember) -> None:
        """Count *member*'s samples from now on."""
        self._joined[id(member)] = member
        await self.update()

    async def leave(self, member: BarrierMember) -> None:
        """*member* has nothing left to run: it holds no barrier back."""
        self._left.add(id(member))
        self._joined.pop(id(member), None)
        await self.update()

    async def update(self) -> None:
        """Release every barrier whose stage every member has finished, in order."""
        for stage in self.stages:
            if self._released[stage].is_set():
                continue
            if stage == "report" or not self._all_finished(stage):
                return
            await self._release(stage)

    def _all_finished(self, stage: str) -> bool:
        if len(self._joined) + len(self._left) < self._expected:
            return False
        return all(m.unfinished(stage) == 0 for m in self._joined.values())

    async def _release(self, stage: str) -> None:
        async with self._lock:
            event = self._released[stage]
            if event.is_set():
                return
            try:
                # Once every member has left, no sample crosses: no hook.
                if (
                    self._hook is not None
                    and self._joined
                    and not self.stops_before(stage)
                ):
                    await self._hook(stage)
                self.crossed.append(stage)
            except BaseException as exc:
                self._failure = exc
                raise
            finally:
                # Set on failure too, so no runner waits forever on a barrier
                # that will never open; each one raises instead.
                event.set()

    async def before(self, stage: str) -> bool:
        """Wait until *stage* may start; ``True`` when the run stops first.

        Waits on every barrier on an earlier stage, in order. Raises
        :class:`BarrierError` when a barrier's hook raised.
        """
        index = stage_index(stage)
        for barrier in self.stages:
            if stage_index(barrier) >= index:
                break
            await self._released[barrier].wait()
            if self._failure is not None:
                raise BarrierError(
                    f"The hook at the barrier on {barrier!r} failed: {self._failure!r}"
                ) from self._failure
            if barrier == self.stop_after:
                return True
        return False

    async def before_action(self, action: "TaskAction") -> bool:
        return await self.before(ACTION_STAGE[action.value])
