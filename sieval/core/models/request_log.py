"""One record per model request attempt, observed below every retry loop.

Every model client is a :class:`RecordingAsyncClient`, sitting under whichever
loop retries -- the OpenAI SDK's, or its own copy of it for ``async_http_json``
-- so each HTTP attempt is seen. Two context variables say whose request it is:

* :func:`stage_scope` -- set by the runner around one stage of one sample;
* :func:`observe_call` -- set by ``Model.arun`` around one model call.

With either unset nothing is recorded. A record holds no credential, header or
body, and feeds no identity, digest or ``model_calls[]``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextvars
import email.utils
import inspect
import math
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import anyio
import httpx
from openai import DefaultAsyncHttpxClient

from sieval.core.types import JSONValue

from .ir import UsageStats

#: Headers a provider may name its request id under, in lookup order.
REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-amzn-requestid")

_TIMEOUT_PHASE_BY_TYPE: tuple[tuple[type[httpx.TimeoutException], str], ...] = (
    (httpx.ConnectTimeout, "connect"),
    (httpx.ReadTimeout, "read"),
    (httpx.WriteTimeout, "write"),
    (httpx.PoolTimeout, "pool"),
)


def timeout_phase(exc: BaseException) -> str | None:
    """The ``httpx`` phase that timed out behind *exc*, if one did.

    Follows ``__cause__`` only: the OpenAI SDK raises ``APITimeoutError``
    *from* the ``httpx`` error; an unrelated ``__context__`` is not a cause.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for timeout_type, phase in _TIMEOUT_PHASE_BY_TYPE:
            if isinstance(current, timeout_type):
                return phase
        current = current.__cause__
    return None


#: The OpenAI SDK's limits, for every model client. httpx's default of 100
#: silently queued requests past it, whatever ``concurrency_limit`` said.
DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000, max_keepalive_connections=100
)

#: The OpenAI SDK's backoff: 0.5s doubling, capped at 8s; ``retry-after``
#: obeyed within (0, 60] seconds.
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
MAX_RETRY_AFTER = 60.0


def should_retry(response: httpx.Response) -> bool:
    """The OpenAI SDK's rule: ``x-should-retry`` if sent, else 408/409/429/5xx."""

    header = response.headers.get("x-should-retry")
    if header == "true":
        return True
    if header == "false":
        return False
    status = response.status_code
    return status in (408, 409, 429) or status >= 500


def retry_delay(
    max_retries: int,
    remaining: int,
    headers: httpx.Headers | None,
) -> float:
    """Seconds before the next try, on the OpenAI SDK's schedule and jitter."""

    retry_after = _retry_after_s(headers) if headers is not None else None
    if retry_after is not None and 0 < retry_after <= MAX_RETRY_AFTER:
        return retry_after
    taken = min(max_retries - remaining, 1000)
    delay = min(INITIAL_RETRY_DELAY * pow(2.0, taken), MAX_RETRY_DELAY)
    return max(delay * (1 - 0.25 * random.random()), 0.0)


def _retry_after_s(headers: httpx.Headers) -> float | None:
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return float(value) / scale
        except ValueError:
            pass
    when = email.utils.parsedate_tz(headers.get("retry-after") or "")
    if when is None:
        return None
    return float(email.utils.mktime_tz(when) - time.time())


class RequestSink(Protocol):
    """Where a run's attempt records go."""

    def role_of(self, model_meta: Mapping[str, object]) -> str:
        """``subject``, a judge role, or ``unknown``."""
        ...

    def record(self, line: dict[str, JSONValue]) -> None:
        """Accept one attempt record; must not block."""
        ...


def limiter_snapshot(limiter: anyio.CapacityLimiter) -> dict[str, JSONValue]:
    """``{"borrowed", "total"}``; an unbounded total is ``None``."""

    total = limiter.total_tokens
    return {
        "borrowed": limiter.borrowed_tokens,
        "total": None if math.isinf(total) else int(total),
    }


@dataclass
class StageScope:
    """One stage of one sample, as the runner entered it."""

    sink: RequestSink
    task: str
    run: str | None
    stage: str
    sample_id: str | int
    iteration: int
    repeat_index: int | None
    #: Limiters the stage holds, by name.
    limiters: Mapping[str, anyio.CapacityLimiter]
    #: The run's ``time.monotonic()`` start, the base of ``offset_s``.
    run_start: float
    _calls: int = 0

    def next_call(self) -> int:
        self._calls += 1
        return self._calls


@dataclass
class _Attempt:
    number: int
    started_at: float
    started: float
    in_flight: dict[str, JSONValue]
    ended: float | None = None
    status: int | None = None
    error: str | None = None
    outcome: str | None = None
    request_id: str | None = None
    retry_after_header_s: float | None = None
    #: When the request headers went out (httpcore's trace).
    sent: float | None = None
    on_end: Callable[[], None] | None = field(default=None, repr=False)

    def ok(self) -> bool:
        return self.outcome == "2xx" and self.error is None

    def end(self) -> None:
        """Stamp the end once, and leave the connection's in-flight count."""

        if self.ended is not None:
            return
        self.ended = time.monotonic()
        if self.on_end is not None:
            self.on_end()


@dataclass
class CallScope:
    """One model call and its attempts, recorded when it ends."""

    stage: StageScope
    call: int
    role: str
    phase: str
    model: dict[str, JSONValue]
    n: int
    limiters: Mapping[str, anyio.CapacityLimiter]
    entered: float
    queue_s: float | None = None
    usage: UsageStats | None = None
    attempts: list[_Attempt] = field(default_factory=list)

    def admitted(self) -> None:
        """Mark the call through every limiter; the wait is ``queue_s``."""

        self.queue_s = time.monotonic() - self.entered

    def begin_attempt(self, connection_in_flight: int) -> _Attempt:
        in_flight: dict[str, JSONValue] = {
            name: limiter_snapshot(limiter)
            for name, limiter in (*self.stage.limiters.items(), *self.limiters.items())
        }
        in_flight["connection"] = {"borrowed": connection_in_flight, "total": None}
        attempt = _Attempt(
            number=len(self.attempts) + 1,
            started_at=time.time(),
            started=time.monotonic(),
            in_flight=in_flight,
        )
        self.attempts.append(attempt)
        return attempt

    def lines(self) -> Iterator[dict[str, JSONValue]]:
        stage = self.stage
        usage = (
            {k: v for k, v in asdict(self.usage).items() if v is not None}
            if self.usage is not None
            else None
        )
        last = len(self.attempts)
        for index, attempt in enumerate(self.attempts):
            ended = attempt.ended if attempt.ended is not None else attempt.started
            following = self.attempts[index + 1] if index + 1 < last else None
            yield {
                "run": stage.run,
                "task": stage.task,
                "phase": self.phase,
                "role": self.role,
                "model": self.model,
                "stage": stage.stage,
                "sample_id": stage.sample_id,
                "iteration": stage.iteration,
                "repeat_index": stage.repeat_index,
                "call": self.call,
                "n": self.n,
                "attempt": attempt.number,
                "attempts": last,
                "started_at": _iso(attempt.started_at),
                "ended_at": _iso(attempt.started_at + (ended - attempt.started)),
                "offset_s": round(attempt.started - stage.run_start, 6),
                "latency_s": round(ended - attempt.started, 6),
                # Part of `latency_s` before the headers went out; `None` if never.
                "connect_s": (
                    round(attempt.sent - attempt.started, 6)
                    if attempt.sent is not None
                    else None
                ),
                # The call's limiter wait, on its first attempt only.
                "queue_s": (
                    round(self.queue_s, 6)
                    if index == 0 and self.queue_s is not None
                    else None
                ),
                "status": attempt.status,
                "outcome": attempt.outcome,
                "error": attempt.error,
                "request_id": attempt.request_id,
                "retry_after_s": (
                    round(following.started - ended, 6)
                    if following is not None
                    else None
                ),
                "retry_after_header_s": attempt.retry_after_header_s,
                "usage": usage if index == last - 1 and attempt.ok() else None,
                "in_flight": attempt.in_flight,
            }


_STAGE: contextvars.ContextVar[StageScope | None] = contextvars.ContextVar(
    "sieval_request_stage", default=None
)
_CALL: contextvars.ContextVar[CallScope | None] = contextvars.ContextVar(
    "sieval_request_call", default=None
)


@contextmanager
def stage_scope(scope: StageScope | None) -> Iterator[None]:
    """Attribute every model call made inside to *scope* (``None``: record none)."""

    token = _STAGE.set(scope)
    try:
        yield
    finally:
        _STAGE.reset(token)


@contextmanager
def observe_call(
    *,
    model_meta: Callable[[], Mapping[str, object]],
    model: Mapping[str, JSONValue],
    n: int,
    limiters: Mapping[str, anyio.CapacityLimiter],
) -> Iterator[CallScope | None]:
    """Open one call in the current stage; ``None`` outside a run.

    On exit, raised or not, the call's attempts go to the stage's sink.
    """

    stage = _STAGE.get()
    if stage is None:
        yield None
        return
    role = stage.sink.role_of(model_meta())
    call = CallScope(
        stage=stage,
        call=stage.next_call(),
        role=role,
        # A judge called from `infer` is still a judge.
        phase=stage.stage if role in ("subject", "unknown") else "judge",
        model=dict(model),
        n=n,
        limiters=limiters,
        entered=time.monotonic(),
    )
    token = _CALL.set(call)
    try:
        yield call
    finally:
        _CALL.reset(token)
        # A streamed body never closed ends with its call.
        for attempt in call.attempts:
            attempt.end()
        for line in call.lines():
            stage.sink.record(line)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def request_id_of(response: httpx.Response) -> str | None:
    for name in REQUEST_ID_HEADERS:
        value = response.headers.get(name)
        if value:
            return value[:200]
    return None


def retry_after_header_s(response: httpx.Response) -> float | None:
    """The delay the server asked for, in seconds."""

    return _retry_after_s(response.headers)


def status_outcome(status: int) -> str:
    if 200 <= status < 300:
        return "2xx"
    if status == 429:
        return "429"
    if 400 <= status < 500:
        return "4xx"
    if status >= 500:
        return "5xx"
    return "other"


def error_outcome(exc: BaseException) -> tuple[str, str]:
    """``(error, outcome)`` for an attempt that raised; class names only."""

    if isinstance(exc, anyio.get_cancelled_exc_class()):
        return "cancelled", "cancelled"
    phase = timeout_phase(exc)
    if phase is not None:
        return f"timeout::{phase}", f"timeout_{phase}"
    error = f"exception::{type(exc).__name__}"
    if isinstance(exc, httpx.TransportError):
        return error, "connection_error"
    return error, "other"


def _finish_error(attempt: _Attempt, exc: BaseException) -> None:
    if attempt.ended is not None:
        return
    attempt.error, outcome = error_outcome(exc)
    # A body that broke after its status arrived keeps the status.
    if attempt.outcome is None or attempt.outcome == "2xx":
        attempt.outcome = outcome
    attempt.end()


def _finish_response(attempt: _Attempt, response: httpx.Response) -> None:
    attempt.status = response.status_code
    attempt.outcome = status_outcome(response.status_code)
    attempt.request_id = request_id_of(response)
    if attempt.outcome != "2xx":
        attempt.retry_after_header_s = retry_after_header_s(response)


class _AttemptStream(httpx.AsyncByteStream):
    """A streamed body that ends its attempt when read out or closed."""

    def __init__(self, inner: httpx.AsyncByteStream, attempt: _Attempt) -> None:
        self._inner = inner
        self._attempt = attempt

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                yield chunk
        except BaseException as exc:
            _finish_error(self._attempt, exc)
            raise
        self._attempt.end()

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            self._attempt.end()


def _sent_trace(
    attempt: _Attempt, inner: Callable[..., object] | None
) -> Callable[[str, dict], Awaitable[None]]:
    """An httpcore ``trace`` stamping when the headers go out, then *inner*."""

    async def trace(name: str, info: dict) -> None:
        if attempt.sent is None and name.endswith("send_request_headers.started"):
            attempt.sent = time.monotonic()
        if inner is not None:
            result = inner(name, info)
            if inspect.isawaitable(result):
                await result

    return trace


class RecordingAsyncClient(httpx.AsyncClient):
    """``httpx.AsyncClient`` that records each try inside an open call.

    *max_retries* retries as the OpenAI SDK does (:func:`should_retry`,
    :func:`retry_delay`); the transport must not retry on its own.
    """

    def __init__(self, *, max_retries: int = 0, **kwargs: object) -> None:
        kwargs.setdefault("limits", DEFAULT_CONNECTION_LIMITS)
        kwargs.setdefault("follow_redirects", True)
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        if isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self._max_retries = max_retries
        self._in_flight = 0

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        **kwargs: object,
    ) -> httpx.Response:
        retries_taken = 0
        while True:
            remaining = self._max_retries - retries_taken
            try:
                response = await self._send_once(request, stream=stream, **kwargs)
            except httpx.TransportError:
                if remaining <= 0:
                    raise
                await anyio.sleep(retry_delay(self._max_retries, remaining, None))
                retries_taken += 1
                continue
            if remaining > 0 and response.is_error and should_retry(response):
                await response.aclose()
                await anyio.sleep(
                    retry_delay(self._max_retries, remaining, response.headers)
                )
                retries_taken += 1
                continue
            return response

    async def _send_once(
        self,
        request: httpx.Request,
        *,
        stream: bool,
        **kwargs: object,
    ) -> httpx.Response:
        call = _CALL.get()
        if call is None:
            return await super().send(request, stream=stream, **kwargs)  # ty: ignore[invalid-argument-type]
        self._in_flight += 1
        attempt = call.begin_attempt(self._in_flight)
        attempt.on_end = self._leave
        # Restored after the try: a retry of this request must not chain onto
        # this attempt's trace and stamp it again.
        extensions = request.extensions
        request.extensions = {
            **extensions,
            "trace": _sent_trace(attempt, extensions.get("trace")),
        }
        try:
            response = await super().send(request, stream=stream, **kwargs)  # ty: ignore[invalid-argument-type]
        except BaseException as exc:
            _finish_error(attempt, exc)
            raise
        finally:
            request.extensions = extensions
        _finish_response(attempt, response)
        # An error status ends here: a retry loop closes its body only after
        # the backoff, which would bill the wait as latency.
        if stream and attempt.outcome == "2xx":
            response.stream = _AttemptStream(
                response.stream,  # ty: ignore[invalid-argument-type]
                attempt,
            )
        else:
            attempt.end()
        return response

    def _leave(self) -> None:
        self._in_flight -= 1


class RecordingOpenAIClient(RecordingAsyncClient, DefaultAsyncHttpxClient):
    """The OpenAI SDK's default client, recorded; the SDK retries above it."""
