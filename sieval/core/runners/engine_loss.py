"""Failing fast once an engine that serves a run is lost.

The CLI watches each engine it launched and calls :meth:`EngineLoss.mark_lost`
when one exits or stops answering. Each runner holds an :class:`EngineGuard`
naming, per stage, the engines that stage calls. From then on a stage that
needs the lost engine fails before it sends anything, and one in flight is
cancelled, both as :class:`EngineDiedError`. The runner records that as
:data:`~sieval.core.tasks.consts.ENGINE_DIED_REASON`, which a resume retries.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import anyio

from sieval.core.tasks.context import TaskAction


class EngineDiedError(Exception):
    """A stage needed an engine that is no longer serving."""


class EngineLoss:
    """The engines lost during one run, shared by every runner of it."""

    def __init__(self) -> None:
        self._lost: dict[str, str] = {}
        self._in_flight: set[tuple[frozenset[str], anyio.CancelScope]] = set()
        #: Stages failed as :class:`EngineDiedError`, by task.
        self.failed: dict[str, int] = {}

    def mark_lost(self, engine: str, message: str) -> None:
        """Record *engine* as lost and cancel every stage in flight on it.

        The first message for an engine is kept.
        """
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

    @contextlib.contextmanager
    def guard(self, engines: frozenset[str], task: str = "") -> Iterator[None]:
        """Run the block unless one of *engines* is lost; cancel it if one is.

        Raises :class:`EngineDiedError` instead of starting, when cancelled, and
        in place of any exception the block raises once an engine is lost: a
        request that failed against a dead engine says nothing about the sample.
        Each one raised counts in :attr:`failed` under *task*.
        """
        if not engines:
            yield
            return
        try:
            with self._guard(engines):
                yield
        except EngineDiedError:
            self.failed[task] = self.failed.get(task, 0) + 1
            raise

    @contextlib.contextmanager
    def _guard(self, engines: frozenset[str]) -> Iterator[None]:
        message = self.lost(engines)
        if message is not None:
            raise EngineDiedError(message)
        with anyio.CancelScope() as scope:
            entry = (engines, scope)
            self._in_flight.add(entry)
            try:
                yield
            except Exception as exc:
                message = self.lost(engines)
                if message is None:
                    raise
                raise EngineDiedError(message) from exc
            finally:
                self._in_flight.discard(entry)
        if scope.cancelled_caught:
            raise EngineDiedError(self.lost(engines) or "engine lost")


@dataclass(frozen=True)
class EngineGuard:
    """One task's view of an :class:`EngineLoss`: the engines each stage calls."""

    loss: EngineLoss
    task: str
    needs: Mapping[TaskAction, frozenset[str]] = field(default_factory=dict)

    def guard(self, action: TaskAction) -> contextlib.AbstractContextManager[None]:
        return self.loss.guard(self.needs.get(action, frozenset()), self.task)
