"""KV admission: requests wait for room by the tokens they may hold.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from typing import Any
from unittest.mock import patch

import anyio
import httpx
import pytest

from sieval.core.models import SglangGenModel
from sieval.core.models.admission import TokenBudget


class TestWeight:
    def test_a_request_holds_n_sequences_of_its_output_and_headroom(self) -> None:
        budget = TokenBudget(1000, context=32768, headroom=4096)
        assert budget.weight(8, 28672) == 8 * 32768
        assert budget.weight(1, 1024) == 1024 + 4096
        assert budget.weight(4, None) == 4 * 32768

    @pytest.mark.parametrize("n", [0, -1, True])
    def test_a_bad_n_is_refused(self, n: Any) -> None:
        with pytest.raises(ValueError, match="n must be"):
            TokenBudget(10, context=10).weight(n, 1)

    @pytest.mark.parametrize(
        "kwargs",
        [{"capacity": 0, "context": 1}, {"capacity": 1, "context": 0}],
    )
    def test_bad_sizes_are_refused(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            TokenBudget(**kwargs)


class TestAdmission:
    @pytest.mark.anyio
    async def test_requests_wait_until_their_tokens_fit(self) -> None:
        budget = TokenBudget(100, context=100)
        order: list[str] = []
        release = anyio.Event()

        async def hold(name: str, tokens: int) -> None:
            async with budget.reserve(tokens):
                order.append(name)
                await release.wait()

        async with anyio.create_task_group() as tg:
            tg.start_soon(hold, "a", 60)
            await anyio.wait_all_tasks_blocked()
            tg.start_soon(hold, "b", 60)
            await anyio.wait_all_tasks_blocked()
            assert order == ["a"]
            assert budget.waiting == 1
            assert budget.held == 60
            release.set()
        assert order == ["a", "b"]
        assert budget.held == 0

    @pytest.mark.anyio
    async def test_admission_is_first_come_first_served(self) -> None:
        budget = TokenBudget(100, context=100)
        order: list[str] = []
        gates = {name: anyio.Event() for name in "abc"}

        async def hold(name: str, tokens: int) -> None:
            async with budget.reserve(tokens):
                order.append(name)
                await gates[name].wait()

        async with anyio.create_task_group() as tg:
            tg.start_soon(hold, "a", 90)
            await anyio.wait_all_tasks_blocked()
            tg.start_soon(hold, "b", 80)  # does not fit beside a
            await anyio.wait_all_tasks_blocked()
            tg.start_soon(hold, "c", 5)  # would fit, but b is first
            await anyio.wait_all_tasks_blocked()
            assert order == ["a"]
            gates["a"].set()
            await anyio.wait_all_tasks_blocked()
            assert order == ["a", "b", "c"]
            gates["b"].set()
            gates["c"].set()

    @pytest.mark.anyio
    async def test_a_request_larger_than_the_room_runs_alone(self) -> None:
        budget = TokenBudget(10, context=10)
        async with budget.reserve(50):
            assert budget.held == 50

    @pytest.mark.anyio
    async def test_a_cancelled_waiter_leaves_the_queue(self) -> None:
        budget = TokenBudget(10, context=10)
        async with budget.reserve(10):
            with anyio.move_on_after(0.01):
                async with budget.reserve(5):
                    pytest.fail("admitted past the room")
            assert budget.waiting == 0
        assert budget.held == 0

    @pytest.mark.anyio
    async def test_more_capacity_admits_waiters(self) -> None:
        budget = TokenBudget(10, context=10)
        admitted = anyio.Event()

        async def wait() -> None:
            async with budget.reserve(10):
                admitted.set()

        async with budget.reserve(10), anyio.create_task_group() as tg:
            tg.start_soon(wait)
            await anyio.wait_all_tasks_blocked()
            assert not admitted.is_set()
            budget.set_capacity(20)
            await admitted.wait()


class TestModelAdmission:
    @pytest.mark.anyio
    async def test_a_model_reserves_n_times_its_max_tokens(self) -> None:
        budget = TokenBudget(10_000, context=32768, headroom=0)
        seen: list[int] = []

        def engine(_: httpx.Request) -> httpx.Response:
            seen.append(budget.held)
            return httpx.Response(
                200,
                json=[
                    {
                        "text": "o",
                        "meta_info": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "cached_tokens": 0,
                            "finish_reason": {"type": "stop"},
                        },
                    }
                ]
                * 4,
            )

        with patch(
            "sieval.core.models.connection_factory.httpx.AsyncHTTPTransport",
            return_value=httpx.MockTransport(engine),
        ):
            model = SglangGenModel(model="m", api_base="http://e", api_key="k")
        model.pool._token_budget = budget

        await model.agenerate("hi", n=4, max_tokens=100)

        assert seen == [400]
        assert budget.held == 0
