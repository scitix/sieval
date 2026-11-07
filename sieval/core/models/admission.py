"""Admitting requests to an engine by the KV cache they may hold.

A launched engine holds a fixed number of KV tokens. A request asking for ``n``
sequences of up to ``max_tokens`` each may hold ``n`` times that, so counting
requests says little: one ``n: 16`` request at 28672 tokens holds what 16
``n: 1`` ones do. Sent past that room, the engine queues and retracts what it
already admitted, and the queue only grows.

:class:`TokenBudget` admits a request while the tokens it may hold fit in the
engine's KV room, first come first served. The wait is before the request is
sent, so no client timeout runs during it.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections import deque
from types import TracebackType
from typing import Self

import anyio


class TokenBudget:
    """A FIFO admission gate over *capacity* tokens of an engine's KV cache.

    Each request reserves ``n * min(context, max_tokens + headroom)`` tokens
    (:meth:`weight`); ``max_tokens`` unset reserves a full *context* per
    sequence. *headroom* is the room a prompt is given beside the output.

    A request is admitted while the reservations held plus its own fit in the
    capacity. One larger than the whole capacity is admitted alone, when
    nothing else is held. Admission is in arrival order, so a large request is
    not starved by a stream of small ones.

    ``capacity`` may be changed while requests wait (:meth:`set_capacity`);
    waiters are re-examined at once.
    """

    def __init__(self, capacity: int, *, context: int, headroom: int = 0) -> None:
        for name, value in (("capacity", capacity), ("context", context)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"TokenBudget {name} must be a positive integer")
        if isinstance(headroom, bool) or not isinstance(headroom, int) or headroom < 0:
            raise ValueError("TokenBudget headroom must be a non-negative integer")
        self._capacity = capacity
        self._context = context
        self._headroom = headroom
        self._held = 0
        self._waiters: deque[tuple[int, anyio.Event]] = deque()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def context(self) -> int:
        return self._context

    @property
    def held(self) -> int:
        """Tokens reserved by admitted requests."""
        return self._held

    @property
    def waiting(self) -> int:
        """Requests waiting for admission."""
        return len(self._waiters)

    # The two attributes a request record reads off a limiter
    # (``request_log.limiter_snapshot``).
    @property
    def borrowed_tokens(self) -> int:
        return self._held

    @property
    def total_tokens(self) -> int:
        return self._capacity

    def weight(self, n: int, max_tokens: int | None) -> int:
        """Tokens one request of *n* sequences up to *max_tokens* reserves."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        per_sequence = (
            self._context
            if max_tokens is None
            else min(self._context, max_tokens + self._headroom)
        )
        return n * per_sequence

    def set_capacity(self, capacity: int) -> None:
        """Change the capacity; waiters that now fit are admitted."""
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("TokenBudget capacity must be a positive integer")
        self._capacity = capacity
        self._admit()

    def _fits(self, tokens: int) -> bool:
        return self._held == 0 or self._held + tokens <= self._capacity

    def _admit(self) -> None:
        while self._waiters and self._fits(self._waiters[0][0]):
            tokens, event = self._waiters.popleft()
            self._held += tokens
            event.set()

    def reserve(self, tokens: int) -> "_Reservation":
        """An async context manager holding *tokens* while inside."""
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError("a reservation must be a positive number of tokens")
        return _Reservation(self, tokens)

    async def _acquire(self, tokens: int) -> None:
        if not self._waiters and self._fits(tokens):
            self._held += tokens
            return
        event = anyio.Event()
        entry = (tokens, event)
        self._waiters.append(entry)
        try:
            await self._wait(event)
        except BaseException:
            if event.is_set():
                # Admitted just as it was cancelled: give the room back.
                self._release(tokens)
            else:
                self._waiters.remove(entry)
                self._admit()
            raise

    async def _wait(self, event: anyio.Event) -> None:
        """Wait until *event* is set; a subclass may look at the engine meanwhile."""
        await event.wait()

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
