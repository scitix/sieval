"""Admitting requests to an engine, for ``concurrency_limit: auto``.

``signal: throughput`` (the default) is :class:`AdaptiveLimit`: a bound on
the sequences in flight, where one ``n: 8`` request counts 8, that
:class:`ThroughputController` moves by the token throughput it measures. It
raises the bound while throughput rises, goes back when throughput stops
rising or falls, and halves it when the engine retracts requests or requests
time out or fail with a 5xx. Queueing in the engine is not a signal: a
saturated engine queues.

``signal: kv`` is :class:`AdaptiveTokenBudget`: a :class:`TokenBudget` that
admits by the KV cache a request may hold, ``n * min(context, max_tokens +
headroom)``, and holds requests back while the engine reports queued
requests or a full pool. ``signal: both`` is :class:`CombinedGate`: the
adaptive bound under a plain :class:`TokenBudget`. :class:`PendingGate`
defers a gate to the first request, for an engine not up when its root was
bound.

Every gate waits before the request is sent, so no client timeout runs while
it waits. None changes what a request asks: throughput only.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import math
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

import anyio
from loguru import logger

from sieval.core.types import JSONValue

from .ir import UsageStats

if TYPE_CHECKING:
    # ``request_log`` imports the bridge, which imports the pool, which
    # imports this module: read it only where it is used.
    from .request_log import LimiterLoad


def _at_least(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class AdmissionGate(Protocol):
    """What a connection pool admits a request through."""

    def admit(self, n: int, max_tokens: int | None) -> AbstractAsyncContextManager[Any]:
        """Hold a request of *n* sequences up to *max_tokens* while inside."""
        ...

    def completed(self, usage: UsageStats | None) -> None:
        """A request this gate admitted answered with *usage*."""
        ...

    def failed(self, exc: BaseException) -> None:
        """A request this gate admitted raised *exc*."""
        ...

    def record_limiters(self) -> Mapping[str, "LimiterLoad"]:
        """What a request record shows under ``in_flight``, by key."""
        ...

    async def aclose(self) -> None: ...


class _FifoGate:
    """A first-come-first-served gate over *capacity* units.

    A request is admitted when it fits beside what is held, or alone when it
    is larger than the whole capacity. Admission keeps arrival order, so
    small requests never starve a large one.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = _at_least("capacity", capacity)
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

    def reserve(self, units: int) -> "_Reservation":
        """An async context manager holding *units* while inside."""
        return _Reservation(self, _at_least("tokens", units))

    def _fits(self, units: int) -> bool:
        return self._held == 0 or self._held + units <= self._capacity

    def _admit(self) -> None:
        while self._waiters and self._fits(self._waiters[0][0]):
            units, event = self._waiters.popleft()
            self._held += units
            event.set()

    async def _acquire(self, units: int) -> None:
        if not self._waiters and self._fits(units):
            self._held += units
            return
        entry = (units, anyio.Event())
        self._waiters.append(entry)
        try:
            await self._wait(entry[1])
        except BaseException:
            if entry[1].is_set():  # admitted as it was cancelled
                self._release(units)
            else:
                self._waiters.remove(entry)
                self._admit()
            raise

    async def _wait(self, admitted: anyio.Event) -> None:
        await admitted.wait()

    def _release(self, units: int) -> None:
        self._held -= units
        self._admit()

    def completed(self, usage: UsageStats | None) -> None:
        """A request this gate admitted answered with *usage*."""

    def failed(self, exc: BaseException) -> None:
        """A request this gate admitted raised *exc*."""

    async def aclose(self) -> None:
        """Release what the gate holds open; a fixed gate holds nothing."""


class _Reservation:
    def __init__(self, gate: _FifoGate, units: int) -> None:
        self._gate = gate
        self._units = units

    async def __aenter__(self) -> Self:
        await self._gate._acquire(self._units)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._gate._release(self._units)


class TokenBudget(_FifoGate):
    """A first-come-first-served gate over *capacity* KV tokens.

    A request reserves ``n * min(context, max_tokens + headroom)`` tokens
    (:meth:`weight`), a full *context* per sequence when ``max_tokens`` is
    unset.
    """

    def __init__(self, capacity: int, *, context: int, headroom: int = 0) -> None:
        super().__init__(capacity)
        self._context = _at_least("context", context)
        self._headroom = _at_least("headroom", headroom, minimum=0)

    def weight(self, n: int, max_tokens: int | None) -> int:
        """Tokens a request of *n* sequences up to *max_tokens* reserves."""
        _at_least("n", n)
        if max_tokens is None:
            return n * self._context
        return n * min(self._context, max_tokens + self._headroom)

    def admit(self, n: int, max_tokens: int | None) -> _Reservation:
        return self.reserve(self.weight(n, max_tokens))

    def record_limiters(self) -> Mapping[str, "LimiterLoad"]:
        return {"kv_tokens": self}


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

    def _fits(self, units: int) -> bool:
        return not (self._held and self._congested) and super()._fits(units)

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


# ---------------------------------------------------------------------------
# Throughput: the controller, its options, and the gate that applies it
# ---------------------------------------------------------------------------

Signal = Literal["throughput", "kv", "both"]
SIGNALS: tuple[Signal, ...] = ("throughput", "kv", "both")


@dataclass(frozen=True)
class AutoConcurrency:
    """``concurrency_limit: auto`` and its options; ``auto`` alone is the
    default of each.

    *signal* picks the gate: ``throughput`` (:class:`AdaptiveLimit`), ``kv``
    (:class:`AdaptiveTokenBudget`, the gate before 0.15.7) or ``both``. The
    rest tune the throughput controller: *start*, *min* and *max* bound the
    sequences in flight (*max* ``None``: twice the requests the engine runs
    at once, else :data:`DEFAULT_MAX`); *kv_ceiling* stops raising the bound
    while the engine's KV pool is full; *retraction_backoff* halves it when
    the engine retracts requests. *interval* is the seconds per window.
    """

    signal: Signal = "throughput"
    start: int = 32
    min: int = 8
    max: int | None = None
    kv_ceiling: bool = True
    retraction_backoff: bool = True
    interval: float = 5.0

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError(f"signal must be one of {', '.join(SIGNALS)}")
        for name in ("start", "min"):
            _at_least(name, getattr(self, name))
        if self.max is not None:
            _at_least("max", self.max)
            if self.max < self.min:
                raise ValueError("max must be >= min")
        for name in ("kv_ceiling", "retraction_backoff"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        if (
            isinstance(self.interval, bool)
            or not isinstance(self.interval, int | float)
            or not self.interval > 0
        ):
            raise ValueError("interval must be a number of seconds > 0")

    @property
    def needs_kv_pool(self) -> bool:
        """Whether the gate reserves KV tokens, so needs the engine's pool."""
        return self.signal != "throughput"

    @classmethod
    def from_config(cls, value: object) -> "AutoConcurrency | None":
        """The options a ``concurrency_limit`` *value* sets: ``auto``, or
        ``{auto: {option: value, ...}}``; ``None`` for anything else (a
        number, unset), which the caller checks."""
        if value == "auto":
            return cls()
        if not isinstance(value, Mapping) or "auto" not in value:
            return None
        if len(value) != 1:
            raise ValueError(
                "concurrency_limit takes 'auto' alone or {auto: {options}}, "
                f"got keys {', '.join(sorted(map(str, value)))}"
            )
        options = cast(Mapping[str, object], value).get("auto")
        if options is None:
            return cls()
        if not isinstance(options, Mapping):
            raise ValueError("concurrency_limit.auto must be a mapping of options")
        known = sorted(cls.__dataclass_fields__)
        unknown = sorted(str(k) for k in options if k not in known)
        if unknown:
            raise ValueError(
                f"concurrency_limit.auto has unknown option(s) {', '.join(unknown)}; "
                f"expected any of {', '.join(known)}"
            )
        try:
            return cls(**cast(dict[str, Any], dict(options)))
        except ValueError as exc:
            raise ValueError(f"concurrency_limit.auto: {exc}") from None


#: The ceiling when neither ``max`` nor the engine says one.
DEFAULT_MAX = 1024


@dataclass(frozen=True)
class Window:
    """What one window saw, as :class:`ThroughputController` reads it.

    *tokens* the engine processed (``None``: no reading), its retractions and
    the requests that timed out or failed with a 5xx. *saturated*: requests
    waited at the limit for the whole window. *over*: more was in flight
    than the limit, after a cut, so the window measures the old limit.
    *kv_full*: the engine's fullest KV pool was full at the window's end.
    """

    seconds: float
    tokens: int | None
    retracted: int = 0
    failures: int = 0
    saturated: bool = True
    over: bool = False
    kv_full: bool = False


@dataclass(frozen=True)
class Decision:
    """The controller's answer to a window: the *limit* after it, why, and
    the throughput (tok/s) it measured when it measured one."""

    limit: int
    reason: str
    throughput: float | None = None
    previous: int | None = None

    @property
    def changed(self) -> bool:
        return self.previous is not None and self.previous != self.limit


class ThroughputController:
    """A hill climb over the sequences in flight, on measured throughput.

    After a change it lets *settle* windows pass, then averages *measure*
    windows into a throughput at that limit and compares it with the
    reference, the best limit so far:

    - first measurement: double (slow start);
    - up and *gain* better: keep going up (doubling until the first stop,
      then a quarter of the limit at a time);
    - up and not *gain* better (plateau) or worse (MoE past its peak): go
      back to the reference and hold;
    - down and no more than *gain* worse: keep the smaller limit, which holds
      as much, and hold; worse: go back and hold.

    A hold lasts *hold* measurements, refreshing the reference, then probes
    the other way from the last probe. Retractions (with
    *retraction_backoff*) and failures halve the limit and cap later climbs
    below where they happened until the next hold ends. With *kv_ceiling* a
    full KV pool stops a climb. A window with no reading, one not saturated
    (demand is below the limit: nothing to learn) or one still draining a
    cut changes nothing. Pure and deterministic: windows in, decisions out.
    """

    def __init__(
        self,
        *,
        start: int,
        minimum: int,
        maximum: int,
        kv_ceiling: bool = True,
        retraction_backoff: bool = True,
        gain: float = 0.05,
        settle: int = 1,
        measure: int = 2,
        hold: int = 6,
    ) -> None:
        _at_least("min", minimum)
        if maximum < minimum:
            raise ValueError("max must be >= min")
        if not 0 < gain < 1:
            raise ValueError("gain must be in (0, 1)")
        self.minimum = minimum
        self.maximum = maximum
        self.limit = min(max(_at_least("start", start), minimum), maximum)
        self._kv_ceiling = kv_ceiling
        self._retraction_backoff = retraction_backoff
        self._gain = gain
        self._settle = _at_least("settle", settle, minimum=0)
        self._measure = _at_least("measure", measure)
        self._hold = _at_least("hold", hold)
        self._reference: tuple[int, float] | None = None
        self._samples: list[float] = []
        self._since_change = 0
        self._slow_start = True
        self._holding = 0
        self._next_probe: Literal["up", "down"] = "up"
        self._cap: int | None = None
        self.last = Decision(self.limit, "start")

    def observe(self, window: Window) -> Decision:
        """The decision for *window*; also :attr:`last`."""
        self.last = self._observe(window)
        return self.last

    def set_maximum(self, maximum: int) -> Decision:
        """Bound later limits by *maximum* (at least :attr:`minimum`),
        lowering the limit to it when above."""
        self.maximum = max(self.minimum, maximum)
        if self.limit > self.maximum:
            self.last = self._move(self.maximum, "engine max", None)
        return self.last

    def _observe(self, w: Window) -> Decision:
        if w.over:
            return self._stay("draining")
        if w.failures:
            return self._backoff("failed")
        if w.retracted and self._retraction_backoff:
            return self._backoff("retracted")
        if w.tokens is None or not w.seconds > 0:
            self._samples.clear()
            return self._stay("no reading")
        if not w.saturated:
            self._samples.clear()
            return self._stay("below limit")
        if self._since_change < self._settle:
            self._since_change += 1
            return self._stay("settling")
        self._samples.append(w.tokens / w.seconds)
        if len(self._samples) < self._measure:
            return self._stay("measuring")
        rate = sum(self._samples) / len(self._samples)
        self._samples.clear()
        return self._judge(rate, w.kv_full)

    def _judge(self, rate: float, kv_full: bool) -> Decision:
        ref = self._reference
        if self._holding:
            self._reference = (self.limit, rate)
            self._holding -= 1
            if self._holding:
                return self._stay("holding", rate)
            self._cap = None
            if self._next_probe == "down":
                self._next_probe = "up"
                return self._move(self._down(), "probe down", rate)
            self._next_probe = "down"
            return self._up(rate, kv_full, "probe up")
        if ref is None or ref[0] == self.limit:
            self._reference = (self.limit, rate)
            return self._up(rate, kv_full, "start" if ref is None else "probe up")
        ref_limit, ref_rate = ref
        if self.limit > ref_limit:
            if rate >= ref_rate * (1 + self._gain):
                self._reference = (self.limit, rate)
                return self._up(rate, kv_full, "rising")
            reason = "falling" if rate < ref_rate * (1 - self._gain) else "plateau"
            return self._back_to(ref, reason, rate)
        if rate >= ref_rate * (1 - self._gain):
            # As much at fewer: keep it, and hold, so noise cannot walk the
            # limit down a step per measurement.
            self._reference = (self.limit, rate)
            return self._hold_here("same lower", rate)
        return self._back_to(ref, "falling", rate)

    def _up(self, rate: float, kv_full: bool, reason: str) -> Decision:
        if kv_full and self._kv_ceiling:
            return self._hold_here("kv full", rate)
        step = self.limit if self._slow_start else max(1, self.limit // 4)
        target = min(self.limit + step, self.maximum)
        if self._cap is not None:
            target = min(target, self._cap - 1)
        if target <= self.limit:
            return self._hold_here("at max", rate)
        return self._move(target, reason, rate)

    def _down(self) -> int:
        return max(self.minimum, self.limit - max(1, self.limit // 5))

    def _back_to(self, ref: tuple[int, float], reason: str, rate: float) -> Decision:
        self._slow_start = False
        self._reference = ref
        self._holding = self._hold
        return self._move(ref[0], reason, rate)

    def _hold_here(self, reason: str, rate: float | None) -> Decision:
        self._slow_start = False
        self._holding = self._hold
        return self._stay(reason, rate)

    def _backoff(self, reason: str) -> Decision:
        self._slow_start = False
        self._cap = self.limit
        self._reference = None
        self._holding = 0
        self._samples.clear()
        return self._move(max(self.minimum, self.limit // 2), reason, None)

    def _move(self, limit: int, reason: str, rate: float | None) -> Decision:
        previous = self.limit
        if limit == previous:
            # A move that cannot move (at min): hold there.
            self._holding = self._holding or self._hold
            return self._stay(reason, rate)
        self.limit = limit
        self._since_change = 0
        self._samples.clear()
        return Decision(limit, reason, rate, previous)

    def _stay(self, reason: str, rate: float | None = None) -> Decision:
        return Decision(self.limit, reason, rate, self.limit)


@dataclass(frozen=True)
class EngineSample:
    """Cumulative counters and gauges one engine read gave; ``None`` for
    what the engine did not report.

    *tokens*: prompt tokens computed (cache hits excluded) plus tokens
    generated. *retracted*: requests the engine retracted. *token_usage*:
    the fullest KV pool's share in use. *max_running*: the requests the
    engine runs at once, over its data-parallel ranks.
    """

    tokens: int | None = None
    retracted: int | None = None
    token_usage: float | None = None
    max_running: int | None = None


class ThroughputProbe(Protocol):
    """Reads an engine's counters; ``None`` when it does not answer."""

    async def __call__(self) -> EngineSample | None: ...

    async def aclose(self) -> None: ...


class AdaptiveLimit(_FifoGate):
    """A first-come-first-served bound on the sequences in flight, moved by
    a :class:`ThroughputController` (the ``signal: throughput`` gate).

    While requests wait at the bound, one of them reads *probe* every
    ``options.interval`` seconds and hands the window to the controller.
    Tokens come from the engine's counters; when the engine reports none,
    from the usage of the requests that answered here. A probe that fails
    or does not answer is a window with no reading: nothing changes and no
    request fails. *clock* is injectable for tests. *name* labels the run
    log line printed when the bound changes.
    """

    #: Its key under a request record's ``in_flight``.
    record_key = "auto"

    def __init__(
        self,
        options: AutoConcurrency,
        *,
        probe: ThroughputProbe,
        name: str = "",
        clock: Callable[[], float] | None = None,
    ) -> None:
        maximum = options.max if options.max is not None else DEFAULT_MAX
        self._controller = ThroughputController(
            start=min(options.start, maximum),
            minimum=min(options.min, maximum),
            maximum=maximum,
            kv_ceiling=options.kv_ceiling,
            retraction_backoff=options.retraction_backoff,
        )
        super().__init__(self._controller.limit)
        self._options = options
        self._probe = probe
        self._name = name
        self._clock = clock or anyio.current_time
        self._polling = False
        self._window_start: float | None = None
        self._last: EngineSample | None = None
        self._client_tokens = 0
        self._client_at_start = 0
        self._failures = 0
        self._unblocked = False
        self._peak = 0
        self._max_known = options.max is not None
        #: Where window tokens come from, fixed at the first answered read:
        #: one source per run, so windows compare.
        self._source: Literal["engine", "client"] | None = None
        #: Every limit change, as (limit, reason), for the closing summary.
        self.changes: list[tuple[int, str]] = []

    @property
    def limit(self) -> int:
        return self._capacity

    @property
    def decision(self) -> Decision:
        return self._controller.last

    @property
    def record_extra(self) -> dict[str, JSONValue]:
        """What a request record adds beside ``borrowed`` / ``total``."""
        out: dict[str, JSONValue] = {"decision": self.decision.reason}
        if self._source is not None:
            out["source"] = self._source
        return out

    def admit(self, n: int, max_tokens: int | None) -> _Reservation:
        return self.reserve(_at_least("n", n))

    def record_limiters(self) -> Mapping[str, "LimiterLoad"]:
        return {self.record_key: self}

    def completed(self, usage: UsageStats | None) -> None:
        if usage is not None:
            computed = usage.input_tokens - (usage.cached_tokens or 0)
            self._client_tokens += max(0, computed) + usage.output_tokens

    def failed(self, exc: BaseException) -> None:
        if is_overload_failure(exc):
            self._failures += 1

    def _acquire_held(self) -> None:
        self._peak = max(self._peak, self._held)

    def _admit(self) -> None:
        super()._admit()
        self._acquire_held()

    async def _acquire(self, units: int) -> None:
        await super()._acquire(units)
        self._acquire_held()

    def _release(self, units: int) -> None:
        super()._release(units)
        if not self._waiters and self._held < self._capacity:
            # A slot is free and nobody wants it: demand is below the limit.
            self._unblocked = True

    async def _wait(self, admitted: anyio.Event) -> None:
        interval = self._options.interval
        while not admitted.is_set():
            with anyio.move_on_after(interval):
                await admitted.wait()
            if admitted.is_set() or self._polling:
                continue
            start = self._window_start
            if start is None or self._clock() - start >= interval:
                await self._tick()

    async def _tick(self) -> None:
        self._polling = True
        try:
            sample = await self._probe()
        except Exception:  # a failed read must not fail a request
            sample = None
        finally:
            self._polling = False
        now = self._clock()
        if self._window_start is None:
            self._start_window(now, sample)
            return
        if sample is not None and not self._max_known and sample.max_running:
            # The engine's own bound, from its KV pool: a coarse ceiling.
            self._max_known = True
            self._apply(self._controller.set_maximum(2 * sample.max_running))
        window = self._window(now, sample)

        decision = self._controller.observe(window)
        self._start_window(now, sample)
        self._apply(decision)

    def _window(self, now: float, sample: EngineSample | None) -> Window:
        assert self._window_start is not None
        last = self._last
        if self._source is None and sample is not None:
            self._source = "engine" if sample.tokens is not None else "client"
        tokens: int | None = None
        if self._source == "client":
            tokens = self._client_tokens - self._client_at_start
        elif sample is not None and last is not None:
            tokens = _delta(sample.tokens, last.tokens)
        retracted = 0
        if sample is not None and last is not None:
            retracted = _delta(sample.retracted, last.retracted) or 0
        usage = sample.token_usage if sample is not None else None
        return Window(
            seconds=now - self._window_start,
            tokens=tokens,
            retracted=retracted,
            failures=self._failures,
            saturated=not self._unblocked,
            over=self._peak > self._capacity,
            kv_full=usage is not None and usage >= _KV_FULL,
        )

    def _start_window(self, now: float, sample: EngineSample | None) -> None:
        self._window_start = now
        # No read leaves no base: the next window would span two.
        self._last = sample
        self._client_at_start = self._client_tokens
        self._failures = 0
        self._unblocked = False
        self._peak = self._held

    def _apply(self, decision: Decision) -> None:
        if not decision.changed:
            return
        self._capacity = decision.limit
        self.changes.append((decision.limit, decision.reason))
        rate = (
            f", {decision.throughput:,.0f} tok/s"
            if decision.throughput is not None
            else ""
        )
        logger.info(
            "auto {}: limit {} -> {} ({}{})",
            self._name or "concurrency",
            decision.previous,
            decision.limit,
            decision.reason,
            rate,
        )
        self._admit()

    async def aclose(self) -> None:
        """Close the probe; log where the limit ended."""
        if self.changes:
            logger.info(
                "auto {}: limit ended at {} after {} change(s)",
                self._name or "concurrency",
                self.limit,
                len(self.changes),
            )
        await self._probe.aclose()


#: The KV pool share a ``kv_ceiling`` reads as full.
_KV_FULL = 0.95


def _delta(now: int | None, before: int | None) -> int | None:
    """A counter's growth; ``None`` unread, or reset (an engine restart)."""
    if now is None or before is None or now < before:
        return None
    return now - before


def is_overload_failure(exc: BaseException) -> bool:
    """Whether *exc* says the engine is overloaded: a timeout, or a 5xx."""
    from .request_log import timeout_phase

    if timeout_phase(exc) is not None:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return isinstance(status, int) and status >= 500


class CombinedGate:
    """``signal: both``: :class:`AdaptiveLimit` first, then the KV
    reservation of *budget*."""

    def __init__(self, limit: AdaptiveLimit, budget: TokenBudget) -> None:
        self.limit = limit
        self.budget = budget

    @asynccontextmanager
    async def admit(self, n: int, max_tokens: int | None) -> AsyncIterator[None]:
        async with (
            self.limit.admit(n, max_tokens),
            self.budget.admit(n, max_tokens),
        ):
            yield

    def completed(self, usage: UsageStats | None) -> None:
        self.limit.completed(usage)

    def failed(self, exc: BaseException) -> None:
        self.limit.failed(exc)

    def record_limiters(self) -> Mapping[str, "LimiterLoad"]:
        return {**self.limit.record_limiters(), **self.budget.record_limiters()}

    async def aclose(self) -> None:
        await self.limit.aclose()
        await self.budget.aclose()


class PendingGate:
    """Admission for an engine not up when the root was bound.

    *resolve* runs once, at the first request (the engine is up by then), and
    gives the gate, or ``None`` for no admission. A failed read is kept and
    raised to every later request, so a broken engine is not read per request.
    """

    def __init__(self, resolve: Callable[[], Awaitable[AdmissionGate | None]]) -> None:
        self._resolve = resolve
        self._lock = anyio.Lock()
        self._done = False
        self._gate: AdmissionGate | None = None
        self._error: Exception | None = None

    @property
    def resolved(self) -> AdmissionGate | None:
        """The gate once resolved; ``None`` before that or for none."""
        return self._gate

    async def resolve(self) -> AdmissionGate | None:
        if not self._done:
            async with self._lock:
                if not self._done:
                    try:
                        self._gate = await self._resolve()
                    except Exception as exc:
                        self._error = exc
                    self._done = True
        if self._error is not None:
            # Reset: re-raising one instance would grow its traceback per request.
            raise self._error.with_traceback(None)
        return self._gate

    async def aclose(self) -> None:
        if self._gate is not None:
            await self._gate.aclose()
