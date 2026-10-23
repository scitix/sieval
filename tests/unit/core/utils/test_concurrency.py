"""
Unit tests for sieval/core/utils/concurrency.py.

Covers: CompositeLimiter (sequential acquire/release, None filtering).

AI-Generated Code - GPT-5.3-Codex (OpenAI)
"""

import anyio
import pytest

from sieval.core.utils.concurrency import CompositeLimiter


class _FailOnEnterLimiter:
    async def __aenter__(self):
        raise RuntimeError("enter failure")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TestCompositeLimiter:
    @pytest.mark.anyio
    async def test_single_limiter_acquired(self):
        limiter = anyio.CapacityLimiter(2)
        composite = CompositeLimiter(limiter)
        async with composite:
            assert limiter.borrowed_tokens == 1
        assert limiter.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_two_limiters_both_acquired(self):
        l1 = anyio.CapacityLimiter(5)
        l2 = anyio.CapacityLimiter(3)
        composite = CompositeLimiter(l1, l2)
        async with composite:
            assert l1.borrowed_tokens == 1
            assert l2.borrowed_tokens == 1
        assert l1.borrowed_tokens == 0
        assert l2.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_none_limiters_ignored(self):
        l1 = anyio.CapacityLimiter(5)
        composite = CompositeLimiter(None, l1, None)
        async with composite:
            assert l1.borrowed_tokens == 1
        assert l1.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_empty_composite_no_error(self):
        composite = CompositeLimiter()
        async with composite:
            pass  # should not raise

    @pytest.mark.anyio
    async def test_all_none_no_error(self):
        composite = CompositeLimiter(None, None)
        async with composite:
            pass

    @pytest.mark.anyio
    async def test_release_on_exception(self):
        """Limiters must be released even if body raises."""
        l1 = anyio.CapacityLimiter(5)
        l2 = anyio.CapacityLimiter(5)
        composite = CompositeLimiter(l1, l2)
        with pytest.raises(ValueError):
            async with composite:
                raise ValueError("boom")
        assert l1.borrowed_tokens == 0
        assert l2.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_enter_failure_rolls_back_acquired_limiters(self):
        l1 = anyio.CapacityLimiter(1)
        failing = _FailOnEnterLimiter()
        composite = CompositeLimiter(l1, failing)

        with pytest.raises(RuntimeError, match="enter failure"):
            async with composite:
                pass

        assert l1.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_returns_self_on_aenter(self):
        composite = CompositeLimiter()
        result = await composite.__aenter__()
        assert result is composite
        await composite.__aexit__(None, None, None)


class TestCapacityLimiterHandoff:
    """The stage and model limits rest on ``anyio.CapacityLimiter`` holding its
    bound, which anyio before 4.15 did not: a release woke the next waiter
    without handing it the token, so a task scheduled before that waiter ran
    found the queue empty, took the token, and the limit held one more task
    than it allowed. That is how ``concurrency_limits={"infer": 4}`` was seen
    to run 5 inferences at once."""

    @pytest.mark.anyio
    async def test_a_release_hands_the_token_to_the_waiter(self):
        limiter = anyio.CapacityLimiter(1)
        holding = peak = 0

        async def hold():
            nonlocal holding, peak
            async with limiter:
                holding += 1
                peak = max(peak, holding)
                await anyio.sleep(0.01)
                holding -= 1

        async with anyio.create_task_group() as tg:
            await limiter.acquire()
            tg.start_soon(hold)
            await anyio.sleep(0.01)  # the first task now waits in the queue
            # One step, no checkpoint between: a newcomer is scheduled ahead of
            # the waiter's wake-up, then the token is released.
            tg.start_soon(hold)
            limiter.release()

        assert peak == 1
