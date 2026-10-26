"""Failing fast once an engine that serves a run is lost.

The CLI watches each engine it launched and calls :meth:`EngineLoss.mark_lost`
when one exits or stops answering. Each runner holds an :class:`EngineGuard`
naming the engines each stage calls. A stage that needs a lost engine fails as
:class:`EngineDiedError` before it sends anything, and one in flight is
cancelled.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field

import anyio

from sieval.core.models.overflow import ContextOverflowError
from sieval.core.tasks.consts import NonRetriableSampleError
from sieval.core.tasks.context import TaskAction

# How often a failed stage looks for a loss while it settles.
_SETTLE_POLL = 0.05


class EngineDiedError(Exception):
    """A stage needed an engine that is no longer serving.

    *in_flight* says whether the stage had started when the engine was lost:
    such a sample may be what brought it down.
    """

    def __init__(self, message: str, *, in_flight: bool) -> None:
        super().__init__(message)
        self.in_flight = in_flight


class EngineLoss:
    """The engines lost during one run, shared by every runner of it.

    A stage that raises while its engines still look alive waits up to
    *settle* seconds for a loss before its error counts as its own: a
    connection drops before the watcher sees the process exit.
    """

    def __init__(self, settle: float = 0.0) -> None:
        self._settle = settle
        self._lost: dict[str, str] = {}
        self._in_flight: set[tuple[frozenset[str], anyio.CancelScope]] = set()
        #: Stages failed as :class:`EngineDiedError`, by task.
        self.failed: dict[str, int] = {}

    def mark_lost(self, engine: str, message: str) -> None:
        """Record *engine* as lost (the first message is kept) and cancel every
        stage in flight on it."""
        self._lost.setdefault(engine, message)
        for engines, scope in list(self._in_flight):
            if engine in engines:
                scope.cancel()

    def lost(self, engines: frozenset[str]) -> str | None:
        """The message of the first of *engines* that was lost, if any was."""
        for engine in sorted(engines):
            if engine in self._lost:
                return self._lost[engine]
        return None

    @contextlib.asynccontextmanager
    async def guard(
        self, engines: frozenset[str], task: str = ""
    ) -> AsyncIterator[None]:
        """Run the block unless one of *engines* is lost, and cancel it if one is.

        Raises :class:`EngineDiedError` instead of starting, when cancelled, and
        in place of an error the block raises once an engine is lost, except a
        :class:`NonRetriableSampleError` or :class:`ContextOverflowError`, which
        the sample's own input decided. Each one counts in :attr:`failed`.
        """
        if not engines:
            yield
            return
        try:
            async with self._guard(engines):
                yield
        except EngineDiedError:
            self.failed[task] = self.failed.get(task, 0) + 1
            raise

    @contextlib.asynccontextmanager
    async def _guard(self, engines: frozenset[str]) -> AsyncIterator[None]:
        message = self.lost(engines)
        if message is not None:
            raise EngineDiedError(message, in_flight=False)
        try:
            with anyio.CancelScope() as scope:
                entry = (engines, scope)
                self._in_flight.add(entry)
                try:
                    yield
                finally:
                    self._in_flight.discard(entry)
        except (NonRetriableSampleError, ContextOverflowError):
            raise
        except Exception as exc:
            message = self.lost(engines) or await self._settled(engines)
            if message is None:
                raise
            raise EngineDiedError(message, in_flight=True) from exc
        if scope.cancelled_caught:
            raise EngineDiedError(self.lost(engines) or "engine lost", in_flight=True)

    async def _settled(self, engines: frozenset[str]) -> str | None:
        with anyio.move_on_after(self._settle):
            while (message := self.lost(engines)) is None:
                await anyio.sleep(_SETTLE_POLL)
            return message
        return None


@dataclass(frozen=True)
class EngineGuard:
    """One task's view of an :class:`EngineLoss`: the engines each stage calls."""

    loss: EngineLoss
    task: str
    needs: Mapping[TaskAction, frozenset[str]] = field(default_factory=dict)

    def guard(self, action: TaskAction) -> contextlib.AbstractAsyncContextManager[None]:
        return self.loss.guard(self.needs.get(action, frozenset()), self.task)
