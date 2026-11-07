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
        with pytest.raises(ValueError, match="n must be an integer >= 1"):
            TokenBudget(10, context=10).weight(n, 1)

    @pytest.mark.parametrize(
        ("kwargs", "name"),
        [
            ({"capacity": 0, "context": 1}, "capacity"),
            ({"capacity": 1, "context": 0}, "context"),
            ({"capacity": 1, "context": 1, "headroom": -1}, "headroom"),
        ],
    )
    def test_bad_sizes_are_refused(self, kwargs: dict[str, int], name: str) -> None:
        with pytest.raises(ValueError, match=f"^{name} must be"):
            TokenBudget(**kwargs)

    def test_a_bad_reservation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="^tokens must be"):
            TokenBudget(10, context=10).reserve(0)


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
            assert budget.borrowed_tokens == 60
            release.set()
        assert order == ["a", "b"]
        assert budget.borrowed_tokens == 0

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
            assert budget.borrowed_tokens == 50

    @pytest.mark.anyio
    async def test_a_cancelled_waiter_leaves_the_queue(self) -> None:
        budget = TokenBudget(10, context=10)
        async with budget.reserve(10):
            with anyio.move_on_after(0.01):
                async with budget.reserve(5):
                    pytest.fail("admitted past the room")
        # The cancelled waiter is gone: the next request is admitted at once.
        with anyio.fail_after(1):
            async with budget.reserve(10):
                assert budget.borrowed_tokens == 10
        assert budget.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_a_cancelled_head_lets_the_next_waiter_in(self) -> None:
        budget = TokenBudget(10, context=10)
        admitted = anyio.Event()

        async def big(scope: anyio.CancelScope) -> None:
            with scope:
                async with budget.reserve(10):
                    pytest.fail("admitted past the room")

        async def small() -> None:
            async with budget.reserve(2):
                admitted.set()

        head = anyio.CancelScope()
        async with budget.reserve(5), anyio.create_task_group() as tg:
            tg.start_soon(big, head)
            await anyio.wait_all_tasks_blocked()
            tg.start_soon(small)  # fits, but waits behind big
            await anyio.wait_all_tasks_blocked()
            assert not admitted.is_set()
            head.cancel()
            with anyio.fail_after(1):
                await admitted.wait()

    @pytest.mark.anyio
    async def test_a_waiter_cancelled_as_it_is_admitted_gives_the_room_back(
        self,
    ) -> None:
        budget = TokenBudget(10, context=10)
        scope = anyio.CancelScope()

        async def wait() -> None:
            with scope:
                async with budget.reserve(10):
                    pytest.fail("ran after its cancellation")

        held = budget.reserve(10)
        await held.__aenter__()
        async with anyio.create_task_group() as tg:
            tg.start_soon(wait)
            await anyio.wait_all_tasks_blocked()
            scope.cancel()
            await held.__aexit__(None, None, None)  # admits the cancelled waiter
        assert (budget.borrowed_tokens, budget.total_tokens) == (0, 10)


class TestModelAdmission:
    @pytest.mark.anyio
    async def test_a_model_reserves_n_times_its_max_tokens(self) -> None:
        budget = TokenBudget(10_000, context=32768, headroom=0)
        seen: list[int] = []

        def engine(_: httpx.Request) -> httpx.Response:
            seen.append(budget.borrowed_tokens)
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
        assert budget.borrowed_tokens == 0
