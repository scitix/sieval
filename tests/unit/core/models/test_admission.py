"""KV admission: requests wait for room by the tokens they may hold.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from typing import Any
from unittest.mock import patch

import anyio
import httpx
import pytest

from sieval.core.models import SglangGenModel
from sieval.core.models.admission import (
    AdaptiveTokenBudget,
    EngineLoad,
    PendingTokenBudget,
    TokenBudget,
)


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

    @pytest.mark.anyio
    async def test_a_pending_budget_is_read_at_the_first_request_once(self) -> None:
        budget = TokenBudget(10_000, context=32768, headroom=0)
        reads: list[int] = []
        seen: list[int] = []

        async def resolve() -> TokenBudget:
            reads.append(1)
            return budget

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
                ],
            )

        with patch(
            "sieval.core.models.connection_factory.httpx.AsyncHTTPTransport",
            return_value=httpx.MockTransport(engine),
        ):
            model = SglangGenModel(model="m", api_base="http://e", api_key="k")
        pending = PendingTokenBudget(resolve)
        model.pool._token_budget = pending
        # Bound, not read: planning sends no request.
        assert model.pool.token_budget is None
        assert reads == []

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(lambda: model.agenerate("hi", max_tokens=100))

        assert reads == [1]
        # Every request reserved from the budget it resolved.
        assert len(seen) == 3 and all(0 < held <= 300 for held in seen)
        assert model.pool.token_budget is budget
        assert budget.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_a_pending_budget_that_fails_fails_the_request(self) -> None:
        async def resolve() -> TokenBudget:
            raise RuntimeError("no KV pool")

        pending = PendingTokenBudget(resolve)
        with pytest.raises(RuntimeError, match="no KV pool"):
            await pending.resolve()

    @pytest.mark.anyio
    async def test_a_pending_budget_closes_the_adaptive_one_it_resolved(
        self,
    ) -> None:
        probe = _Probe(None)

        async def resolve() -> TokenBudget:
            return _adaptive(probe)

        pending = PendingTokenBudget(resolve)
        await pending.aclose()  # never resolved: nothing to close
        assert probe.closed is False
        await pending.resolve()
        await pending.aclose()
        assert probe.closed is True


class _Probe:
    """Answers *loads* in turn, then the last one forever."""

    def __init__(self, *loads: EngineLoad | None) -> None:
        self.loads = list(loads)
        self.calls = 0
        self.closed = False

    async def __call__(self) -> EngineLoad | None:
        self.calls += 1
        return self.loads.pop(0) if len(self.loads) > 1 else self.loads[0]

    async def aclose(self) -> None:
        self.closed = True


_BUSY = EngineLoad(waiting=2, token_usage=0.5)
_FREE = EngineLoad(waiting=0, token_usage=0.4)


def _adaptive(probe: _Probe, capacity: int = 100) -> AdaptiveTokenBudget:
    return AdaptiveTokenBudget(capacity, context=100, probe=probe, interval=0.01)


class TestAdaptive:
    @pytest.mark.parametrize(
        "busy_load",
        [
            EngineLoad(waiting=3, token_usage=0.5),
            EngineLoad(waiting=0, token_usage=0.95),
        ],
        ids=["queueing", "kv-full"],
    )
    @pytest.mark.anyio
    async def test_a_request_that_fits_waits_while_the_engine_is_congested(
        self, busy_load: EngineLoad
    ) -> None:
        probe = _Probe(busy_load, busy_load, busy_load, _FREE)
        budget = _adaptive(probe)
        budget._congested = True  # as the last read left it
        admitted = anyio.Event()

        async def second() -> None:
            async with budget.reserve(10):
                admitted.set()

        async with budget.reserve(10), anyio.create_task_group() as tg:
            tg.start_soon(second)
            await anyio.wait_all_tasks_blocked()
            # 20 of 100 tokens would fit; congestion holds it back.
            assert budget.borrowed_tokens == 10 and not admitted.is_set()
            with anyio.fail_after(2):
                await admitted.wait()
        assert probe.calls == 4
        assert not budget.congested

    @pytest.mark.anyio
    async def test_reading_survives_the_reader_being_admitted(self) -> None:
        # Regression: only the first waiter read the engine. Admitted while
        # congested (nothing else held), it stopped reading, and the stale
        # flag let the rest through one at a time.
        probe = _Probe(_BUSY)
        budget = _adaptive(probe)
        budget._congested = True
        admitted: list[str] = []
        release_a = anyio.Event()

        async def req(name: str, gate: anyio.Event | None = None) -> None:
            async with budget.reserve(10):
                admitted.append(name)
                if gate is not None:
                    await gate.wait()
                else:
                    await anyio.sleep(10)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(req, "a", release_a)
                await anyio.wait_all_tasks_blocked()
                for name in "bcd":
                    tg.start_soon(req, name)
                    await anyio.wait_all_tasks_blocked()
                assert admitted == ["a"]
                release_a.set()  # held drops to 0: "b" is admitted alone
                while admitted != ["a", "b"]:
                    await anyio.sleep(0.005)
                probe.loads = [_FREE]  # the engine drains
                while len(admitted) < 4:
                    await anyio.sleep(0.005)
                tg.cancel_scope.cancel()
        assert not budget.congested

    @pytest.mark.anyio
    async def test_one_read_per_interval_however_many_wait(self) -> None:
        probe = _Probe(_BUSY)
        budget = AdaptiveTokenBudget(100, context=100, probe=probe, interval=0.05)
        budget._congested = True
        async with budget.reserve(10), anyio.create_task_group() as tg:
            for _ in range(20):
                tg.start_soon(lambda: budget.reserve(10).__aenter__())
            await anyio.sleep(0.26)
            tg.cancel_scope.cancel()
        assert 3 <= probe.calls <= 6

    @pytest.mark.anyio
    async def test_congestion_seen_at_a_read_counts_once(self) -> None:
        probe = _Probe(_BUSY, _BUSY, _BUSY, _FREE)
        budget = _adaptive(probe)
        async with budget.reserve(100), anyio.create_task_group() as tg:
            tg.start_soon(lambda: budget.reserve(1).__aenter__())
            while probe.calls < 3:
                await anyio.sleep(0.005)
            tg.cancel_scope.cancel()
        assert budget.held_back == 1

    @pytest.mark.anyio
    async def test_nothing_held_back_while_nothing_is_admitted(self) -> None:
        budget = _adaptive(_Probe(_BUSY))
        budget._congested = True
        async with budget.reserve(10):
            assert budget.borrowed_tokens == 10

    @pytest.mark.parametrize(
        "failure", [None, RuntimeError("down")], ids=["no-answer", "raises"]
    )
    @pytest.mark.anyio
    async def test_an_engine_that_does_not_answer_leaves_the_budget(
        self, failure: Exception | None
    ) -> None:
        probe = _Probe(None) if failure is None else _Raising(failure)
        budget = _adaptive(probe)
        budget._congested = True
        done = anyio.Event()

        async def second() -> None:
            async with budget.reserve(30):
                done.set()

        async with budget.reserve(60), anyio.create_task_group() as tg:
            tg.start_soon(second)
            with anyio.fail_after(2):
                await done.wait()
        assert not budget.congested

    @pytest.mark.anyio
    async def test_aclose_closes_the_probe(self) -> None:
        probe = _Probe(None)
        await _adaptive(probe).aclose()
        assert probe.closed

    @pytest.mark.parametrize(
        ("interval", "busy", "match"),
        [(0, 0.9, "interval"), (1, 0, "busy"), (1, 1.5, "busy")],
    )
    def test_bad_settings_are_refused(
        self, interval: float, busy: float, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            AdaptiveTokenBudget(
                1, context=1, probe=_Probe(None), interval=interval, busy=busy
            )


class _Raising(_Probe):
    def __init__(self, exc: Exception) -> None:
        super().__init__(None)
        self.exc = exc

    async def __call__(self) -> EngineLoad | None:
        raise self.exc
