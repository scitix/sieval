"""One record per model request attempt, observed below every retry loop.

Every client sieval builds to reach a model is a :class:`RecordingAsyncClient`.
It sits *under* whichever loop retries the request -- the OpenAI SDK's, or its
own copy of that loop for ``async_http_json`` -- so each HTTP attempt is seen
on its own, SDK retries included, and ``max_retries`` means the same in both.
The client cannot see which sample or phase sent a request; two context
variables carry that:

* :func:`stage_scope` -- set by the runner around one stage of one sample;
* :func:`observe_call` -- set by ``Model.arun`` / ``SglangGenModel.arun``
  around one model call, after the request is built.

With either one unset nothing is recorded, so a call made outside a run (a
health check, a test) costs one ``ContextVar.get``.

A record carries no credential, header, or body: the request id, a status, an
exception class, and timings. Nothing here feeds an identity, a digest, or
``model_calls[]``.

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
    *from* the ``httpx`` error, while an unrelated error raised inside an
    ``except`` block merely has it as ``__context__``.
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


#: Connection limits for every model client sieval builds -- the OpenAI SDK's.
#: httpx's own default of 100 queued an ``async_http_json`` run's requests
#: past that silently, whatever its ``concurrency_limit``.
DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000, max_keepalive_connections=100
)

#: Retry backoff, as the OpenAI SDK has it: 0.5s doubling, capped at 8s.
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
#: A server's ``retry-after`` is obeyed only within (0, 60] seconds.
MAX_RETRY_AFTER = 60.0


def should_retry(response: httpx.Response) -> bool:
    """Whether an error *response* is retried: the OpenAI SDK's rule.

    ``x-should-retry`` decides when the server sends it; otherwise 408, 409,
    429 and every 5xx are retried.
    """

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
    """Seconds to wait before the next try: the OpenAI SDK's schedule.

    A ``retry-after-ms`` / ``retry-after`` within (0, 60] seconds is obeyed;
    otherwise ``0.5 * 2**n`` capped at 8, times a jitter in (0.75, 1].
    """

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
        """``subject``, a judge role, or ``unknown`` for the model of *model_meta*."""
        ...

    def record(self, line: dict[str, JSONValue]) -> None:
        """Accept one attempt record; must not block or await."""
        ...


type LimiterLike = object


def limiter_snapshot(limiter: LimiterLike | None) -> dict[str, JSONValue] | None:
    """``{"borrowed", "total"}`` of an ``anyio.CapacityLimiter``, else ``None``.

    An unbounded total is ``None``: JSON has no infinity.
    """

    borrowed = getattr(limiter, "borrowed_tokens", None)
    total = getattr(limiter, "total_tokens", None)
    if not isinstance(borrowed, int) or not isinstance(total, int | float):
        return None
    return {
        "borrowed": borrowed,
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
    #: Limiters the stage holds, by name (``global``, ``task``, ``stage`` ...).
    limiters: Mapping[str, LimiterLike]
    #: Offset base for ``offset_s``: the run's ``time.monotonic()`` start.
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
    #: When the request went out on a connection (httpcore's trace).
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
    """One model call: every attempt it made, written out when it ends."""

    stage: StageScope
    call: int
    role: str
    phase: str
    model: dict[str, JSONValue]
    n: int
    limiters: Mapping[str, LimiterLike]
    entered: float
    queue_s: float | None = None
    usage: UsageStats | None = None
    attempts: list[_Attempt] = field(default_factory=list)

    def admitted(self) -> None:
        """Mark the call through every limiter; the wait is ``queue_s``."""

        self.queue_s = time.monotonic() - self.entered

    def begin_attempt(self, connection_in_flight: int) -> _Attempt:
        in_flight: dict[str, JSONValue] = {}
        for name, limiter in (*self.stage.limiters.items(), *self.limiters.items()):
            snapshot = limiter_snapshot(limiter)
            if snapshot is not None:
                in_flight[name] = snapshot
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
                # Of `latency_s`, the wait for a pooled connection (and any
                # connect) before the request went out; `None` if it never did.
                "connect_s": (
                    round(attempt.sent - attempt.started, 6)
                    if attempt.sent is not None
                    else None
                ),
                # The limiter wait is the call's, before its first attempt.
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
    limiters: Mapping[str, LimiterLike],
) -> Iterator[CallScope | None]:
    """Open one call inside the current stage; yield ``None`` outside a run.

    *model_meta* is called only when a stage is open: it resolves the role.
    On exit the call's attempts go to the stage's sink -- also when it raised.
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
        # A streamed body nobody read to the end or closed ends with its call.
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
    """The delay the server asked for, in seconds, read as the retry loops do."""

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
    """A streamed body that ends its attempt when the body is closed."""

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
        # Read to the end: the caller may close it only after the call returns.
        self._attempt.end()

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            self._attempt.end()


def _sent_trace(
    attempt: _Attempt, inner: Callable[..., object] | None
) -> Callable[[str, dict], Awaitable[None]]:
    """An httpcore ``trace`` extension stamping when the headers go out."""

    async def trace(name: str, info: dict) -> None:
        if attempt.sent is None and name.endswith("send_request_headers.started"):
            attempt.sent = time.monotonic()
        if inner is not None:
            result = inner(name, info)
            if inspect.isawaitable(result):
                await result

    return trace


class RecordingAsyncClient(httpx.AsyncClient):
    """``httpx.AsyncClient`` that records each ``send`` inside an open call.

    *max_retries* retries a request the way the OpenAI SDK does (see
    :func:`should_retry` and :func:`retry_delay`), so ``max_retries`` means
    the same in every connection family. Each try is one recorded attempt.
    Give the transport no retries of its own.

    ``RecordingOpenAIClient`` passes ``0``: the SDK runs that loop above it.
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
        request.extensions = {
            **request.extensions,
            "trace": _sent_trace(attempt, request.extensions.get("trace")),
        }
        try:
            response = await super().send(request, stream=stream, **kwargs)  # ty: ignore[invalid-argument-type]
        except BaseException as exc:
            _finish_error(attempt, exc)
            raise
        _finish_response(attempt, response)
        # An error status ends the attempt: a retry loop may close its body
        # only after sleeping, which would bill the wait as latency.
        if stream and attempt.outcome == "2xx":
            # Ends when the caller closes the body, as every streamed read does.
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
    """The OpenAI SDK's own default client, recorded.

    Passed as ``AsyncOpenAI(http_client=...)``; the SDK's retry loop and
    backoff are untouched and run above it, so this client retries nothing.
    """
