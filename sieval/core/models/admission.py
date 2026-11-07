"""Admitting requests to an engine by the KV cache they may hold.

One ``n: 16`` request at 28672 tokens holds what 16 ``n: 1`` ones do, so a
request count does not bound an engine's KV cache. Past it, the engine
retracts running requests and queues the rest. :class:`TokenBudget` admits by
tokens instead, before the request is sent, so no client timeout runs while
it waits.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections import deque
from types import TracebackType
from typing import Self

import anyio


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
            await entry[1].wait()
        except BaseException:
            if entry[1].is_set():  # admitted as it was cancelled
                self._release(tokens)
            else:
                self._waiters.remove(entry)
                self._admit()
            raise

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
