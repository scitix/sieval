"""``concurrency_limit: auto``: KV admission, and the throughput controller.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import anyio
import httpx
import pytest

from sieval.core.models import SglangGenModel
from sieval.core.models.admission import (
    AdaptiveLimit,
    AdaptiveTokenBudget,
    AutoConcurrency,
    CombinedGate,
    Decision,
    EngineLoad,
    EngineSample,
    PendingGate,
    ThroughputController,
    TokenBudget,
    Window,
    is_overload_failure,
)
from sieval.core.models.ir import UsageStats


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
        model.pool._gate = budget

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
        pending = PendingGate(resolve)
        model.pool._gate = pending
        # Bound, not read: planning sends no request.
        assert model.pool.gate is None
        assert reads == []

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(lambda: model.agenerate("hi", max_tokens=100))

        assert reads == [1]
        # Every request reserved from the budget it resolved.
        assert len(seen) == 3 and all(0 < held <= 300 for held in seen)
        assert model.pool.gate is budget
        assert budget.borrowed_tokens == 0

    @pytest.mark.anyio
    async def test_a_pending_budget_that_fails_fails_every_request_reading_once(
        self,
    ) -> None:
        reads: list[int] = []

        async def resolve() -> TokenBudget:
            reads.append(1)
            await anyio.sleep(0)
            raise RuntimeError("no KV pool")

        pending = PendingGate(resolve)
        failed: list[str] = []

        async def request() -> None:
            try:
                await pending.resolve()
            except RuntimeError as exc:
                failed.append(str(exc))

        async with anyio.create_task_group() as tg:
            for _ in range(5):
                tg.start_soon(request)
        await request()

        assert failed == ["no KV pool"] * 6
        assert reads == [1]

    @pytest.mark.anyio
    async def test_a_pending_budget_closes_the_adaptive_one_it_resolved(
        self,
    ) -> None:
        probe = _Probe(None)

        async def resolve() -> TokenBudget:
            return _adaptive(probe)

        pending = PendingGate(resolve)
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


# ---------------------------------------------------------------------------
# concurrency_limit: auto, signal throughput
# ---------------------------------------------------------------------------


def _climb(
    curve: Callable[[int], float],
    windows: int = 120,
    *,
    start: int = 32,
    maximum: int = 1024,
    events: Callable[[int, int], Window | None] | None = None,
    **kwargs: Any,
) -> tuple[ThroughputController, list[int], list[Decision]]:
    """Feed a controller *windows* 5 s windows of the throughput *curve* gives
    at its limit; *events* may replace a window by its index and limit."""
    c = ThroughputController(start=start, minimum=8, maximum=maximum, **kwargs)
    limits: list[int] = []
    decisions: list[Decision] = []
    for i in range(windows):
        w = events(i, c.limit) if events is not None else None
        if w is None:
            w = Window(seconds=5.0, tokens=int(curve(c.limit) * 5))
        decisions.append(c.observe(w))
        limits.append(c.limit)
    return c, limits, decisions


def _rising(limit: int) -> float:
    return 100.0 * limit


def _plateau(limit: int) -> float:
    """Decode-bound: throughput stops growing at 64 in flight."""
    return 100.0 * min(limit, 64)


def _moe(limit: int) -> float:
    """MoE: throughput peaks at 96 in flight, then falls."""
    return 100.0 * limit if limit <= 96 else 9600.0 - 50.0 * (limit - 96)


class TestThroughputController:
    def test_rising_throughput_climbs_to_the_ceiling(self) -> None:
        c, limits, _ = _climb(_rising, maximum=300)
        # Slow start doubles: 32 -> 64 -> 128 -> 256, then the ceiling.
        assert limits[:12] == [32] * 2 + [64] * 3 + [128] * 3 + [256] * 3 + [300]
        assert c.limit == 300
        assert all(limit <= 300 for limit in limits)

    def test_a_plateau_settles_at_its_knee(self) -> None:
        c, limits, decisions = _climb(_plateau)
        # Up to 128 finds no gain, back to 64; probes either side lose.
        assert max(limits) == 128
        assert limits[-40:].count(64) >= 30
        assert set(limits[20:]) <= {52, 64, 80}
        reasons = {d.reason for d in decisions if d.changed}
        assert {"start", "rising", "plateau", "probe down", "falling"} <= reasons

    def test_moe_throughput_that_falls_past_its_peak_stays_near_it(self) -> None:
        _, limits, decisions = _climb(_moe)
        assert 256 in limits  # slow start overshoots once
        tail = limits[-40:]
        assert all(80 <= limit <= 128 for limit in tail)
        assert any(d.reason == "falling" for d in decisions)
        # Best measured throughput near the peak.
        best = max(_moe(limit) for limit in tail)
        assert best >= 0.95 * 9600

    def test_queueing_is_not_a_signal(self) -> None:
        # Windows say nothing about the engine's queue: a rising curve climbs
        # however many requests the engine holds waiting.
        c, _, _ = _climb(_rising, windows=12)
        assert c.limit > 128

    def test_retractions_halve_the_limit_and_cap_the_climb(self) -> None:
        def events(i: int, limit: int) -> Window | None:
            if i == 12:
                return Window(seconds=5.0, tokens=int(_rising(limit) * 5), retracted=3)
            return None

        c, limits, decisions = _climb(_rising, windows=24, events=events)
        assert (limits[11], limits[12]) == (512, 256)
        assert decisions[12].reason == "retracted"
        # Climbs again by quarters, but stays below where it retracted.
        assert max(limits[12:]) < 512
        assert c.limit > 256

    def test_retraction_backoff_can_be_turned_off(self) -> None:
        def events(i: int, limit: int) -> Window | None:
            if i == 6:
                return Window(seconds=5.0, tokens=int(_rising(limit) * 5), retracted=3)
            return None

        _, limits, decisions = _climb(
            _rising, windows=12, events=events, retraction_backoff=False
        )
        assert all(d.reason != "retracted" for d in decisions)
        assert limits == sorted(limits)

    @pytest.mark.parametrize("failures", [1, 5])
    def test_timeouts_and_5xx_halve_the_limit(self, failures: int) -> None:
        def events(i: int, _limit: int) -> Window | None:
            if i == 8:
                return Window(seconds=5.0, tokens=1, failures=failures)
            return None

        _, limits, decisions = _climb(_rising, windows=9, events=events)
        assert decisions[8].reason == "failed"
        assert limits[8] == limits[7] // 2

    def test_backoff_stops_at_the_floor(self) -> None:
        c = ThroughputController(start=10, minimum=8, maximum=64)
        d = c.observe(Window(seconds=5.0, tokens=1, failures=1))
        assert (d.limit, d.reason) == (8, "failed")
        d = c.observe(Window(seconds=5.0, tokens=1, failures=1))
        assert d.limit == 8 and not d.changed

    @pytest.mark.parametrize(
        "window",
        [
            Window(seconds=5.0, tokens=None),
            Window(seconds=0.0, tokens=100),
            Window(seconds=5.0, tokens=100, saturated=False),
            Window(seconds=5.0, tokens=100, over=True),
        ],
        ids=["no-reading", "no-time", "below-limit", "draining"],
    )
    def test_a_window_that_cannot_be_read_changes_nothing(self, window: Window) -> None:
        c = ThroughputController(start=32, minimum=8, maximum=1024)
        for _ in range(10):
            d = c.observe(window)
            assert d.limit == 32 and not d.changed

    def test_a_probe_failure_between_reads_restarts_the_measurement(self) -> None:
        c = ThroughputController(start=32, minimum=8, maximum=1024)
        good = Window(seconds=5.0, tokens=16_000)
        assert c.observe(good).reason == "settling"
        assert c.observe(good).reason == "measuring"
        assert c.observe(Window(seconds=5.0, tokens=None)).reason == "no reading"
        # The one sample before the gap is dropped: two more are needed.
        assert c.observe(good).reason == "measuring"
        assert c.observe(good).limit == 64

    def test_a_full_kv_pool_stops_the_climb(self) -> None:
        def events(_i: int, limit: int) -> Window | None:
            full = limit >= 128
            return Window(seconds=5.0, tokens=int(_rising(limit) * 5), kv_full=full)

        c, limits, decisions = _climb(_rising, windows=40, events=events)
        assert max(limits) == 128
        assert any(d.reason == "kv full" for d in decisions)

    def test_the_kv_ceiling_can_be_turned_off(self) -> None:
        def events(_i: int, limit: int) -> Window | None:

            return Window(seconds=5.0, tokens=int(_rising(limit) * 5), kv_full=True)

        _, limits, _ = _climb(_rising, windows=40, events=events, kv_ceiling=False)
        assert max(limits) > 128

    def test_it_is_deterministic(self) -> None:
        assert _climb(_moe)[1] == _climb(_moe)[1]

    def test_a_lower_engine_maximum_lowers_the_limit(self) -> None:
        c = ThroughputController(start=32, minimum=8, maximum=1024)
        d = c.set_maximum(20)
        assert (d.limit, d.reason, d.previous) == (20, "engine max", 32)
        assert c.set_maximum(4).limit == 8  # never below the floor

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"minimum": 0, "maximum": 1}, "min"),
            ({"minimum": 4, "maximum": 2}, "max must be >= min"),
            ({"minimum": 1, "maximum": 2, "gain": 0}, "gain"),
        ],
    )
    def test_bad_settings_are_refused(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ThroughputController(start=1, **kwargs)


class TestAutoConcurrencyOptions:
    def test_auto_alone_is_the_recommended_defaults(self) -> None:
        options = AutoConcurrency.from_config("auto")
        assert options == AutoConcurrency()
        assert options is not None
        assert (options.signal, options.kv_ceiling, options.retraction_backoff) == (
            "throughput",
            True,
            True,
        )
        assert not options.needs_kv_pool

    def test_options_are_read_under_auto(self) -> None:
        options = AutoConcurrency.from_config(
            {"auto": {"signal": "both", "kv_ceiling": False, "max": 128}}
        )
        assert options == AutoConcurrency("both", kv_ceiling=False, max=128)
        assert options is not None and options.needs_kv_pool

    @pytest.mark.parametrize("value", [None, 16, "fast", {"max": 3}])
    def test_anything_else_is_not_auto(self, value: object) -> None:
        assert AutoConcurrency.from_config(value) is None

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            ({"auto": {"signal": "queue"}}, "signal must be one of"),
            ({"auto": {"start": 0}}, "start must be"),
            ({"auto": {"kv_ceiling": "yes"}}, "kv_ceiling must be true or false"),
            ({"auto": {"interval": 0}}, "interval must be"),
            ({"auto": {"interval": True}}, "interval must be"),
            ({"auto": {"bogus": 1}}, "unknown option"),
            ({"auto": [1]}, "mapping of options"),
            ({"auto": {}, "x": 1}, "'auto' alone"),
        ],
    )
    def test_bad_options_are_refused(self, value: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            AutoConcurrency.from_config(value)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Counters:
    """A throughput probe: each read advances *clock* by 5 s and the token
    counter by what *curve* gives at the gate's limit."""

    def __init__(
        self,
        clock: _Clock,
        curve: Callable[[int], float],
        *,
        fail: Callable[[int], bool] = lambda _read: False,
        sample: Callable[[int, int], EngineSample] | None = None,
    ) -> None:
        self.clock = clock
        self.curve = curve
        self.fail = fail
        self.sample = sample
        self.gate: AdaptiveLimit | None = None
        self.reads = 0
        self.tokens = 0
        self.closed = False

    async def __call__(self) -> EngineSample | None:
        self.reads += 1
        assert self.gate is not None
        self.tokens += int(self.curve(self.gate.limit) * 5)
        if self.fail(self.reads):
            raise RuntimeError("engine down")
        if self.sample is not None:
            return self.sample(self.reads, self.tokens)
        return EngineSample(tokens=self.tokens, retracted=0, token_usage=0.3)

    async def aclose(self) -> None:
        self.closed = True


async def _saturate(gate: AdaptiveLimit, probe: _Counters, reads: int) -> None:
    """Keep the gate full with waiters, a 5 s window per probe read, until
    *probe* was read *reads* times."""

    async def hold() -> None:
        async with gate.admit(1, None):
            await anyio.sleep(3600)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            for _ in range(gate.limit + 4):
                tg.start_soon(hold)
            while probe.reads < reads:
                before = probe.reads
                probe.clock.now += 5.0  # the next window is due
                while probe.reads == before:
                    await anyio.sleep(0.001)
                # More waiters than any limit: the gate stays saturated.
                for _ in range(gate.limit + 4 - len(gate._waiters) - gate._held):
                    tg.start_soon(hold)
            tg.cancel_scope.cancel()


def _gate(
    curve: Callable[[int], float], **kwargs: Any
) -> tuple[AdaptiveLimit, _Counters]:
    clock = _Clock()
    options = AutoConcurrency(start=8, min=2, max=64, interval=0.001)
    probe = _Counters(clock, curve, **kwargs)
    gate = AdaptiveLimit(options, probe=probe, name="subject", clock=clock)
    probe.gate = gate
    return gate, probe


class TestAdaptiveLimit:
    def test_a_request_counts_its_sequences(self) -> None:
        gate, _ = _gate(_rising)
        assert gate.limit == 8 and gate.total_tokens == 8
        res = gate.admit(8, 1024)
        assert res._units == 8

    @pytest.mark.anyio
    async def test_the_limit_follows_throughput_and_logs_each_change(self) -> None:
        gate, probe = _gate(_plateau)
        lines: list[str] = []
        with patch(
            "sieval.core.models.admission.logger.info",
            side_effect=lambda msg, *a: lines.append(msg.format(*a)),
        ):
            await _saturate(gate, probe, reads=12)
        assert gate.limit > 8
        assert gate.changes[0] == (16, "start")
        assert lines[0] == "auto subject: limit 8 -> 16 (start, 800 tok/s)"
        assert len(lines) == len(gate.changes)
        assert gate.record_extra["source"] == "engine"

    @pytest.mark.anyio
    async def test_a_failing_probe_fails_no_request_and_moves_nothing(self) -> None:
        gate, probe = _gate(_rising, fail=lambda _: True)
        await _saturate(gate, probe, reads=10)
        assert gate.limit == 8 and gate.changes == []
        assert gate.decision.reason in {"start", "no reading"}

    @pytest.mark.anyio
    async def test_an_intermittent_probe_still_climbs(self) -> None:
        gate, probe = _gate(_rising, fail=lambda read: read % 4 == 0)
        await _saturate(gate, probe, reads=24)
        assert gate.limit > 8

    @pytest.mark.anyio
    async def test_engine_retractions_back_off(self) -> None:
        def sample(read: int, tokens: int) -> EngineSample:
            return EngineSample(tokens=tokens, retracted=0 if read < 8 else read)

        gate, probe = _gate(_rising, sample=sample)
        await _saturate(gate, probe, reads=9)
        assert "retracted" in {reason for _, reason in gate.changes}

    @pytest.mark.anyio
    async def test_without_engine_counters_it_counts_answered_usage(self) -> None:
        def sample(_read: int, _tokens: int) -> EngineSample:
            return EngineSample(token_usage=0.2)

        gate, probe = _gate(_rising, sample=sample)
        gate.completed(UsageStats(input_tokens=100, output_tokens=50, cached_tokens=40))
        gate.completed(None)
        assert gate._client_tokens == 110
        await _saturate(gate, probe, reads=3)
        assert gate.record_extra["source"] == "client"

    @pytest.mark.anyio
    async def test_the_engine_bound_caps_the_limit(self) -> None:
        def sample(_read: int, tokens: int) -> EngineSample:
            return EngineSample(tokens=tokens, max_running=2)

        clock = _Clock()
        probe = _Counters(clock, _rising, sample=sample)
        gate = AdaptiveLimit(
            AutoConcurrency(start=16, min=2, interval=0.001),
            probe=probe,
            clock=clock,
        )
        probe.gate = gate
        await _saturate(gate, probe, reads=2)
        assert gate.limit == 4
        assert gate.changes[0] == (4, "engine max")

    def test_overload_failures_are_counted(self) -> None:
        gate, _ = _gate(_rising)
        gate.failed(httpx.ReadTimeout("slow"))
        gate.failed(_Status(503))
        gate.failed(_Status(400))
        gate.failed(ValueError("bad"))
        assert gate._failures == 2

    @pytest.mark.anyio
    async def test_aclose_closes_the_probe(self) -> None:
        gate, probe = _gate(_rising)
        await gate.aclose()
        assert probe.closed


class _Status(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status_code = status


class TestOverloadFailure:
    def test_a_timeout_behind_an_sdk_error_counts(self) -> None:
        try:
            try:
                raise httpx.ReadTimeout("slow")
            except httpx.ReadTimeout as exc:
                raise RuntimeError("sdk") from exc
        except RuntimeError as wrapped:
            assert is_overload_failure(wrapped)

    def test_a_status_on_the_response_counts(self) -> None:
        exc = Exception()
        exc.response = httpx.Response(502)  # type: ignore[attr-defined]
        assert is_overload_failure(exc)
        assert not is_overload_failure(Exception())


class TestCombinedGate:
    @pytest.mark.anyio
    async def test_both_bounds_hold_and_release_together(self) -> None:
        gate, probe = _gate(_rising)
        budget = TokenBudget(1000, context=100)
        both = CombinedGate(gate, budget)
        async with both.admit(4, 50):
            assert (gate.borrowed_tokens, budget.borrowed_tokens) == (4, 200)
        assert (gate.borrowed_tokens, budget.borrowed_tokens) == (0, 0)
        assert set(both.record_limiters()) == {"auto", "kv_tokens"}
        both.completed(UsageStats(input_tokens=1, output_tokens=1))
        both.failed(_Status(500))
        assert gate._client_tokens == 2 and gate._failures == 1
        await both.aclose()
        assert probe.closed


class TestPoolReportsToTheGate:
    @pytest.mark.anyio
    async def test_a_failed_request_is_reported(self) -> None:
        from sieval.core.models import ConnectionIdentity
        from sieval.core.models.deployment import ConnectionPool

        gate, _ = _gate(_rising)
        identity = ConnectionIdentity(
            endpoint="http://e/v1",
            connection_family="openai_sdk",
            credential_scope="s",
            retry_policy="r",
            quota_scope="q",
        )
        pool = ConnectionPool(_Conn(), identity, gate=gate)
        with pytest.raises(_Status):
            async with pool.acquire(size=(2, None)):
                assert gate.borrowed_tokens == 2
                raise _Status(504)
        assert gate._failures == 1 and gate.borrowed_tokens == 0
        # No size: the gate is not entered.
        async with pool.acquire():
            assert gate.borrowed_tokens == 0


class _Conn:
    async def aclose(self) -> None:
        pass
