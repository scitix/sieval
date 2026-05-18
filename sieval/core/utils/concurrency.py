from contextlib import AbstractAsyncContextManager, suppress
from typing import Self


class CompositeLimiter:
    """
    Acquires multiple limiters sequentially in the order provided.

    IMPORTANT: To avoid deadlocks, all callers must pass limiters in the same
    order (global before local, shared before stage-specific). The canonical
    order is enforced by `get_limiter_for()` which always returns
    CompositeLimiter(global_limiter, stage_limiter).
    """

    def __init__(self, *limiters: AbstractAsyncContextManager | None):
        self._limiters = [limiter for limiter in limiters if limiter is not None]

    async def __aenter__(self) -> Self:
        acquired: list[AbstractAsyncContextManager] = []
        try:
            for limiter in self._limiters:
                await limiter.__aenter__()
                acquired.append(limiter)
        except BaseException:
            # Roll back already-acquired limiters to avoid token leaks.
            for limiter in reversed(acquired):
                with suppress(Exception):
                    await limiter.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for limiter in reversed(self._limiters):
            await limiter.__aexit__(exc_type, exc_val, exc_tb)
