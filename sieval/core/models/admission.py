"""Admitting requests to an engine by the KV cache they may hold.

One ``n: 16`` request at 28672 tokens holds what 16 ``n: 1`` ones do, so a
request count does not bound an engine's KV cache. Past it, the engine
retracts running requests and queues the rest. :class:`TokenBudget` admits by
tokens instead, before the request is sent, so no client timeout runs while
it waits. :class:`AdaptiveTokenBudget` also holds requests back while the
engine reports it is congested. :class:`PendingTokenBudget` stands for either
until the engine it gates is up and has said what it holds.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

import anyio
from loguru import logger


def _at_least(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class TokenBudget:
    """A first-come-first-served gate over *capacity* KV tokens.

    A request reserves ``n * min(context, max_tokens + headroom)`` tokens
    (:meth:`weight`), a full *context* per sequence when ``max_tokens`` is
    unset. It is admitted when it fits beside what is held, or alone when it
    is larger than the whole capacity. Admission keeps arrival order, so small
    requests never starve a large one.
    """

    def __init__(self, capacity: int, *, context: int, headroom: int = 0) -> None:
        self._capacity = _at_least("capacity", capacity)
        self._context = _at_least("context", context)
        self._headroom = _at_least("headroom", headroom, minimum=0)
        self._held = 0
        self._waiters: deque[tuple[int, anyio.Event]] = deque()

    # ``borrowed_tokens`` / ``total_tokens``: read by a request record
    # (``request_log.LimiterLoad``).
    @property
    def borrowed_tokens(self) -> int:
        return self._held

    @property
    def total_tokens(self) -> int:
        return self._capacity

    def weight(self, n: int, max_tokens: int | None) -> int:
        """Tokens a request of *n* sequences up to *max_tokens* reserves."""
        _at_least("n", n)
        if max_tokens is None:
            return n * self._context
        return n * min(self._context, max_tokens + self._headroom)

    def reserve(self, tokens: int) -> "_Reservation":
        """An async context manager holding *tokens* while inside."""
        return _Reservation(self, _at_least("tokens", tokens))

    def _fits(self, tokens: int) -> bool:
        return self._held == 0 or self._held + tokens <= self._capacity

    def _admit(self) -> None:
        while self._waiters and self._fits(self._waiters[0][0]):
            tokens, event = self._waiters.popleft()
            self._held += tokens
            event.set()

    async def _acquire(self, tokens: int) -> None:
        if not self._waiters and self._fits(tokens):
            self._held += tokens
            return
        entry = (tokens, anyio.Event())
        self._waiters.append(entry)
        try:
            await self._wait(entry[1])
        except BaseException:
            if entry[1].is_set():  # admitted as it was cancelled
                self._release(tokens)
            else:
                self._waiters.remove(entry)
                self._admit()
            raise

    async def _wait(self, admitted: anyio.Event) -> None:
        await admitted.wait()

    def _release(self, tokens: int) -> None:
        self._held -= tokens
        self._admit()


class _Reservation:
    def __init__(self, budget: TokenBudget, tokens: int) -> None:
        self._budget = budget
        self._tokens = tokens

    async def __aenter__(self) -> Self:
        await self._budget._acquire(self._tokens)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._budget._release(self._tokens)


@dataclass(frozen=True)
class EngineLoad:
    """Requests the engine holds waiting, and the share of its fullest KV
    pool in use."""

    waiting: int
    token_usage: float


class LoadProbe(Protocol):
    """Reads the engine's load; ``None`` when it does not answer."""

    async def __call__(self) -> EngineLoad | None: ...

    async def aclose(self) -> None: ...


class AdaptiveTokenBudget(TokenBudget):
    """A :class:`TokenBudget` that also waits while the engine is congested.

    While requests wait, one of them reads *probe* every *interval* seconds.
    The engine is congested while it reports waiting requests or a KV pool at
    least *busy* full; then nothing more is admitted beside what is held. A
    probe that fails or does not answer counts as not congested.
    """

    def __init__(
        self,
        capacity: int,
        *,
        context: int,
        headroom: int = 0,
        probe: LoadProbe,
        interval: float = 1.0,
        busy: float = 0.95,
    ) -> None:
        super().__init__(capacity, context=context, headroom=headroom)
        if not interval > 0:
            raise ValueError("interval must be > 0")
        if not 0 < busy <= 1:
            raise ValueError("busy must be in (0, 1]")
        self._probe = probe
        self._interval = interval
        self._busy = busy
        self._congested = False
        self._polling = False
        self._last_poll = -math.inf
        #: Times the engine turned congested at a read.
        self.held_back = 0

    @property
    def congested(self) -> bool:
        return self._congested

    def _fits(self, tokens: int) -> bool:
        return not (self._held and self._congested) and super()._fits(tokens)

    async def _wait(self, admitted: anyio.Event) -> None:
        # Every waiter may read, so reads go on when the last reader is
        # admitted; ``_last_poll`` keeps it to one read per interval.
        while not admitted.is_set():
            with anyio.move_on_after(self._interval):
                await admitted.wait()
            due = anyio.current_time() - self._last_poll >= self._interval
            if not admitted.is_set() and due and not self._polling:
                await self._observe()

    async def _observe(self) -> None:
        self._polling = True
        try:
            load = await self._probe()
        except Exception:  # a failed read must not fail a request
            load = None
        finally:
            self._polling = False
            self._last_poll = anyio.current_time()
        congested = load is not None and (
            load.waiting > 0 or load.token_usage >= self._busy
        )
        if congested and not self._congested:
            self.held_back += 1
        self._congested = congested
        self._admit()

    async def aclose(self) -> None:
        """Close the probe; log how often the engine held requests back."""
        if self.held_back:
            logger.info(
                "KV admission: the engine turned congested {} time(s)", self.held_back
            )
        await self._probe.aclose()


class PendingTokenBudget:
    """The KV admission of an engine not up yet, resolved at its first request.

    Planning binds a root to the deployment its launch will realize, before
    the engine exists, so it knows nothing of its pool; a root started at a
    barrier stays bound that way for the run. *resolve* reads the engine once,
    at the first request through it, which is after the engine is up, and
    returns the :class:`TokenBudget` to use, or ``None`` for no admission.
    Planning sends no request, so it never resolves.
    """

    def __init__(self, resolve: Callable[[], Awaitable[TokenBudget | None]]) -> None:
        self._resolve = resolve
        self._lock = anyio.Lock()
        self._done = False
        self._budget: TokenBudget | None = None

    @property
    def resolved(self) -> TokenBudget | None:
        """The resolved budget; ``None`` before resolution or for none."""
        return self._budget

    async def resolve(self) -> TokenBudget | None:
        if not self._done:
            async with self._lock:
                if not self._done:
                    self._budget = await self._resolve()
                    self._done = True
        return self._budget

    async def aclose(self) -> None:
        if isinstance(self._budget, AdaptiveTokenBudget):
            await self._budget.aclose()
