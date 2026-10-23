"""Tests for per-attempt request records (``sieval.core.models.request_log``).

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import math
import time
import types
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import anyio
import httpx
import openai
import pytest

from sieval.core.models import ChatModel, connection_factory
from sieval.core.models import request_log as rl
from sieval.core.models.connection_factory import (
    CONNECTION_FACTORY_REGISTRY,
    AsyncHTTPJSONConnection,
    ConnectionRequest,
)
from sieval.core.models.request_log import (
    RecordingAsyncClient,
    RecordingOpenAIClient,
    StageScope,
    observe_call,
    stage_scope,
    timeout_phase,
)
from sieval.core.types import JSONValue

SECRET = "sk-SENTINEL-NEVER-RECORDED"


class _Sink:
    def __init__(self, roles: Mapping[str, str] | None = None) -> None:
        self.lines: list[dict[str, JSONValue]] = []
        self._roles = dict(roles or {})

    def role_of(self, model_meta: Mapping[str, object]) -> str:
        return self._roles.get(str(model_meta.get("model")), "subject")

    def record(self, line: dict[str, JSONValue]) -> None:
        self.lines.append(line)


def _scope(
    sink: _Sink, stage: str = "infer", **limiters: anyio.CapacityLimiter
) -> StageScope:
    return StageScope(
        sink=sink,
        task="t",
        run="run-1",
        stage=stage,
        sample_id=7,
        iteration=0,
        repeat_index=2,
        limiters=limiters,
        run_start=0.0,
    )


def _chat_body(text: str = "hi") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _sse_chat(text: str = "hi") -> httpx.Response:
    chunk = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
    return httpx.Response(
        200,
        content=body,
        headers={"content-type": "text/event-stream", "x-request-id": "req-ok"},
    )


def _sdk(handler: Callable[[httpx.Request], httpx.Response], retries: int):
    return openai.AsyncOpenAI(
        base_url="https://gw.example/v1",
        api_key=SECRET,
        max_retries=retries,
        http_client=RecordingOpenAIClient(transport=httpx.MockTransport(handler)),
    )


def _replies(*responses: httpx.Response | Exception):
    queue = list(responses)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler, seen


async def _call(client: openai.AsyncOpenAI, sink: _Sink) -> Any:
    with (
        stage_scope(_scope(sink)),
        observe_call(
            model_meta=lambda: {"model": "m"},
            model={"id": "m"},
            n=1,
            limiters={},
        ) as call,
    ):
        assert call is not None
        call.admitted()
        return await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "q"}]
        )


class TestSdkRetriesAreRecordedPerAttempt:
    @pytest.mark.anyio
    async def test_each_sdk_retry_is_its_own_line(self) -> None:
        handler, seen = _replies(
            httpx.Response(
                429,
                json={"error": {"message": "slow down"}},
                headers={"retry-after-ms": "1", "x-request-id": "req-429"},
            ),
            httpx.Response(503, json={"error": {"message": "busy"}}),
            httpx.Response(200, json=_chat_body(), headers={"x-request-id": "req-ok"}),
        )
        sink = _Sink()
        async with _sdk(handler, retries=2) as client:
            await _call(client, sink)

        assert len(seen) == 3
        assert [line["attempt"] for line in sink.lines] == [1, 2, 3]
        assert {line["attempts"] for line in sink.lines} == {3}
        assert [line["status"] for line in sink.lines] == [429, 503, 200]
        assert [line["outcome"] for line in sink.lines] == ["429", "5xx", "2xx"]
        assert [line["request_id"] for line in sink.lines] == [
            "req-429",
            None,
            "req-ok",
        ]
        assert sink.lines[0]["retry_after_header_s"] == 0.001
        # The wait before each retry, measured; none after the last attempt.
        waits = [line["retry_after_s"] for line in sink.lines]
        assert isinstance(waits[0], float) and waits[0] >= 0.001
        assert isinstance(waits[1], float) and waits[1] > 0
        assert waits[2] is None
        # The limiter wait is the call's, on its first attempt only.
        assert sink.lines[0]["queue_s"] is not None
        assert sink.lines[1]["queue_s"] is None

    @pytest.mark.anyio
    async def test_the_sdk_retry_count_is_unchanged(self) -> None:
        """The SDK's loop runs above the recording client, so it keeps its count."""

        def always_429() -> tuple[Callable, list[httpx.Request]]:
            return _replies(
                *[
                    httpx.Response(
                        429, json={"error": {}}, headers={"retry-after-ms": "1"}
                    )
                    for _ in range(4)
                ]
            )

        recorded_handler, recorded = always_429()
        sink = _Sink()
        async with _sdk(recorded_handler, retries=3) as client:
            with pytest.raises(openai.RateLimitError):
                await _call(client, sink)

        plain_handler, plain = always_429()
        async with openai.AsyncOpenAI(
            base_url="https://gw.example/v1",
            api_key=SECRET,
            max_retries=3,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(plain_handler)),
        ) as client:
            with pytest.raises(openai.RateLimitError):
                await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "q"}]
                )

        assert len(recorded) == len(plain) == 4
        # A failed call is recorded too: every attempt, the last one included.
        assert [line["outcome"] for line in sink.lines] == ["429"] * 4
        # Each retry is the SDK's own, which numbers it.
        assert [r.headers["x-stainless-retry-count"] for r in recorded] == [
            "0",
            "1",
            "2",
            "3",
        ]

    @pytest.mark.anyio
    async def test_a_timeout_is_recorded_by_phase(self) -> None:
        request = httpx.Request("POST", "https://gw.example/v1/chat/completions")
        handler, _ = _replies(httpx.ReadTimeout("slow", request=request))
        sink = _Sink()
        async with _sdk(handler, retries=0) as client:
            with pytest.raises(openai.APITimeoutError):
                await _call(client, sink)

        [line] = sink.lines
        assert line["status"] is None
        assert line["outcome"] == "timeout_read"
        assert line["error"] == "timeout::read"


class TestStreamedAttempts:
    @staticmethod
    async def _stream(
        handler: Callable[[httpx.Request], httpx.Response],
        body: Callable[[httpx.Response], Any],
    ) -> list[dict[str, JSONValue]]:
        client = RecordingAsyncClient(transport=httpx.MockTransport(handler))
        sink = _Sink()
        with (
            stage_scope(_scope(sink)),
            observe_call(model_meta=lambda: {"model": "m"}, model={}, n=1, limiters={}),
        ):
            async with client.stream("POST", "https://gw.example/x") as response:
                await body(response)
        await client.aclose()
        return sink.lines

    @pytest.mark.anyio
    async def test_a_streamed_attempt_lasts_until_its_body_is_read(self) -> None:
        class _Slow(httpx.AsyncByteStream):
            async def __aiter__(self):
                await anyio.sleep(0.05)
                yield b"data: [DONE]\n\n"

        async def read(response: httpx.Response) -> None:
            await response.aread()

        [line] = await self._stream(
            lambda _: httpx.Response(
                200, stream=_Slow(), headers={"x-request-id": "req-ok"}
            ),
            read,
        )
        assert line["latency_s"] >= 0.05  # ty: ignore[unsupported-operator]
        assert line["request_id"] == "req-ok"

    @pytest.mark.anyio
    async def test_an_error_status_ends_before_a_retry_sleeps(self) -> None:
        """A retry loop closes a 429's body only after its wait."""

        async def sleep_then_close(_: httpx.Response) -> None:
            await anyio.sleep(0.2)

        [line] = await self._stream(lambda _: httpx.Response(429), sleep_then_close)
        assert line["outcome"] == "429"
        assert line["latency_s"] < 0.2  # ty: ignore[unsupported-operator]

    @pytest.mark.anyio
    async def test_a_body_that_breaks_keeps_its_status(self) -> None:
        class _Broken(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: {}\n\n"
                raise httpx.ReadError("reset")

        async def read(response: httpx.Response) -> None:
            with pytest.raises(httpx.ReadError):
                await response.aread()

        [line] = await self._stream(
            lambda _: httpx.Response(200, stream=_Broken()), read
        )
        assert line["status"] == 200
        assert line["outcome"] == "connection_error"
        assert line["error"] == "exception::ReadError"

    @pytest.mark.anyio
    async def test_the_connection_count_is_released_by_every_end(self) -> None:
        client = RecordingAsyncClient(
            transport=httpx.MockTransport(lambda _: _sse_chat())
        )
        sink = _Sink()
        with stage_scope(_scope(sink)):
            with observe_call(
                model_meta=lambda: {"model": "m"}, model={}, n=1, limiters={}
            ):
                # Never read nor closed: the call's exit ends it.
                await client.send(
                    client.build_request("POST", "https://gw.example/x"), stream=True
                )
                assert client._in_flight == 1
            with observe_call(
                model_meta=lambda: {"model": "m"}, model={}, n=1, limiters={}
            ):
                await client.post("https://gw.example/x")
        await client.aclose()
        assert client._in_flight == 0
        assert [line["in_flight"] for line in sink.lines] == [
            {"connection": {"borrowed": 1, "total": None}},
            {"connection": {"borrowed": 1, "total": None}},
        ]


class TestConnectionWait:
    @pytest.mark.anyio
    async def test_a_full_pool_is_billed_as_connect_not_hidden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One pooled connection, two concurrent calls: the second waits for it.

        The wait is inside `latency_s` -- the request was not out -- and
        `connect_s` says how much of it.
        """
        # A fake clock for the log only: a wall-clock sleep under-runs on CI.
        now = [100.0]
        monkeypatch.setattr(
            rl, "time", types.SimpleNamespace(monotonic=lambda: now[0], time=time.time)
        )
        release = anyio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            del request
            await release.wait()
            return httpx.Response(200, json={})

        sink = _Sink()
        client = RecordingAsyncClient(transport=_PoolOfOne(handler))
        with stage_scope(_scope(sink)):
            async with anyio.create_task_group() as tg:
                for _ in range(2):
                    tg.start_soon(self._one, client)
                await anyio.wait_all_tasks_blocked()
                now[0] += 0.5
                release.set()
        await client.aclose()
        assert sorted(line["connect_s"] for line in sink.lines) == [0.0, 0.5]

    @staticmethod
    async def _one(client: RecordingAsyncClient) -> None:
        with observe_call(
            model_meta=lambda: {"model": "m"}, model={}, n=1, limiters={}
        ):
            await client.post("https://gw.example/x")

    @pytest.mark.anyio
    async def test_a_callers_own_trace_still_runs(self) -> None:
        names: list[str] = []
        client = RecordingAsyncClient(
            transport=_PoolOfOne(lambda _: httpx.Response(200))
        )
        sink = _Sink()
        with (
            stage_scope(_scope(sink)),
            observe_call(model_meta=lambda: {"model": "m"}, model={}, n=1, limiters={}),
        ):
            await client.post(
                "https://gw.example/x",
                extensions={"trace": lambda name, _info: names.append(name)},
            )
        await client.aclose()
        assert "http11.send_request_headers.started" in names
        assert sink.lines[0]["connect_s"] is not None

    @pytest.mark.anyio
    async def test_a_retried_connect_failure_is_not_stamped_by_the_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Our own retry resends the same `Request`: its trace must not chain
        onto the failed attempt, which never sent its headers."""

        refused = [True]

        async def handler(request: httpx.Request) -> httpx.Response:
            if refused.pop() if refused else False:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json={})

        monkeypatch.setattr(rl, "retry_delay", lambda *_: 0.05)
        sink = _Sink()
        client = RecordingAsyncClient(max_retries=1, transport=_Refusing(handler))
        with (
            stage_scope(_scope(sink)),
            observe_call(model_meta=lambda: {"model": "m"}, model={}, n=1, limiters={}),
        ):
            await client.post("https://gw.example/x")
        await client.aclose()
        failed, ok = sink.lines
        assert failed["outcome"] == "connection_error"
        assert failed["connect_s"] is None
        connect, latency = ok["connect_s"], ok["latency_s"]
        assert isinstance(connect, float) and isinstance(latency, float)
        assert connect <= latency


class _Refusing(httpx.AsyncBaseTransport):
    """*handler* may raise before the headers go out; otherwise they do."""

    def __init__(self, handler: Callable) -> None:
        self._handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        result = await self._handler(request)
        trace = request.extensions.get("trace")
        if trace is not None:
            await trace("http11.send_request_headers.started", {})
        return result


class _PoolOfOne(httpx.AsyncBaseTransport):
    """One connection at a time, emitting httpcore's request-headers trace."""

    def __init__(self, handler: Callable) -> None:
        self._handler = handler
        self._lock = anyio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self._lock:
            trace = request.extensions.get("trace")
            if trace is not None:
                result = trace("http11.send_request_headers.started", {})
                if hasattr(result, "__await__"):
                    await result
            result = self._handler(request)
            if hasattr(result, "__await__"):
                result = await result
            return result


class TestNothingIsRecordedOutsideARun:
    @pytest.mark.anyio
    async def test_a_call_without_a_stage_records_nothing(self) -> None:
        handler, seen = _replies(httpx.Response(200, json=_chat_body()))
        async with _sdk(handler, retries=0) as client:
            with observe_call(
                model_meta=lambda: pytest.fail("no stage, no role lookup"),
                model={"id": "m"},
                n=1,
                limiters={},
            ) as call:
                assert call is None
                await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "q"}]
                )
        assert len(seen) == 1

    @pytest.mark.anyio
    async def test_a_stage_without_a_call_records_nothing(self) -> None:
        handler, _ = _replies(httpx.Response(200, json=_chat_body()))
        sink = _Sink()
        async with _sdk(handler, retries=0) as client:
            with stage_scope(_scope(sink)):
                await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "q"}]
                )
        assert sink.lines == []


class TestUnifiedRetries:
    """``async_http_json`` retries the way the OpenAI SDK does, attempt for attempt.

    Each case runs one scripted reply sequence through the real SDK and through
    :class:`RecordingAsyncClient`, with sleeps captured and jitter pinned, and
    requires the same number of requests, the same waits and the same outcome.
    """

    @pytest.fixture
    def sleeps(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(round(delay, 6))

        monkeypatch.setattr(rl.anyio, "sleep", fake_sleep)
        monkeypatch.setattr(openai._base_client.anyio, "sleep", fake_sleep)
        # Jitter pinned to its midpoint on both sides.
        monkeypatch.setattr(rl.random, "random", lambda: 0.5)
        monkeypatch.setattr(openai._base_client, "random", lambda: 0.5)
        yield delays

    @staticmethod
    def _script() -> list[list[httpx.Response | type[httpx.TransportError]]]:
        def r(status: int, **headers: str) -> httpx.Response:
            return httpx.Response(status, json={"error": {}}, headers=headers)

        ok = httpx.Response(200, json=_chat_body())
        return [
            [r(429), r(503), r(500), ok],
            [r(429, **{"retry-after": "2"}), ok],
            [r(429, **{"retry-after-ms": "1500"}), ok],
            [r(429, **{"retry-after": "120"}), ok],  # past 60s: backoff instead
            [r(503, **{"x-should-retry": "false"}), ok],
            [r(400, **{"x-should-retry": "true"}), ok],
            [r(400), ok],
            [r(404), ok],
            [r(408), r(409), ok],
            [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, ok],
            [r(502)] * 6,
        ]

    @staticmethod
    def _handler(script: list) -> tuple[Callable, list[httpx.Request]]:
        queue = list(script)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            item = queue.pop(0)
            if isinstance(item, type):
                raise item("scripted", request=request)
            return httpx.Response(
                item.status_code, headers=item.headers, content=item.content
            )

        return handler, seen

    async def _via_sdk(self, script: list, sleeps: list[float]) -> tuple:
        handler, seen = self._handler(script)
        sleeps.clear()
        async with openai.AsyncOpenAI(
            base_url="https://gw.example/v1",
            api_key="k",
            max_retries=3,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            try:
                await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "q"}]
                )
                status: object = 200
            except openai.APIStatusError as exc:
                status = exc.status_code
            except openai.APIConnectionError as exc:
                status = type(exc.__cause__).__name__
        return len(seen), list(sleeps), status

    async def _via_recording(self, script: list, sleeps: list[float]) -> tuple:
        handler, seen = self._handler(script)
        sleeps.clear()
        client = RecordingAsyncClient(
            max_retries=3, transport=httpx.MockTransport(handler)
        )
        try:
            response = await client.post("https://gw.example/v1/x", json={})
            status: object = response.status_code
        except httpx.TransportError as exc:
            status = type(exc).__name__
        finally:
            await client.aclose()
        return len(seen), list(sleeps), status

    @pytest.mark.anyio
    @pytest.mark.parametrize("case", range(11))
    async def test_matches_the_openai_sdk(self, case: int, sleeps: list[float]) -> None:
        script = self._script()[case]
        sdk = await self._via_sdk(script, sleeps)
        ours = await self._via_recording(script, sleeps)
        assert ours == sdk

    @pytest.mark.anyio
    async def test_every_try_is_one_recorded_attempt(self, sleeps: list[float]) -> None:
        handler, _ = self._handler(
            [
                httpx.Response(429, headers={"retry-after-ms": "10"}),
                httpx.ConnectError,
                httpx.Response(200, json={}),
            ]
        )
        client = RecordingAsyncClient(
            max_retries=3, transport=httpx.MockTransport(handler)
        )
        sink = _Sink()
        with (
            stage_scope(_scope(sink)),
            observe_call(model_meta=lambda: {"model": "m"}, model={}, n=1, limiters={}),
        ):
            await client.post("https://gw.example/x", json={})
        await client.aclose()
        assert [line["outcome"] for line in sink.lines] == [
            "429",
            "connection_error",
            "2xx",
        ]
        assert sink.lines[0]["retry_after_header_s"] == 0.01
        assert sleeps == [0.01, 0.875]

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            RecordingAsyncClient(max_retries=-1)

    @pytest.mark.anyio
    async def test_both_families_get_the_same_client_defaults(self) -> None:
        http_json = CONNECTION_FACTORY_REGISTRY.create(
            "async_http_json",
            ConnectionRequest("https://gw.example/v1", None, max_retries=4),
        )
        sdk = CONNECTION_FACTORY_REGISTRY.create(
            "openai_sdk",
            ConnectionRequest("https://gw.example/v1", SECRET, max_retries=4),
        )
        assert isinstance(http_json, AsyncHTTPJSONConnection)
        assert isinstance(sdk, openai.AsyncOpenAI)
        ours, theirs = http_json.client, sdk._client
        assert isinstance(ours, RecordingAsyncClient)
        assert isinstance(theirs, RecordingOpenAIClient)
        # The SDK retries above its client; the httpx family in it.
        assert (ours._max_retries, theirs._max_retries, sdk.max_retries) == (4, 0, 4)
        assert ours._transport._pool._retries == 0  # ty: ignore[unresolved-attribute]
        for client in (ours, theirs):
            pool = client._transport._pool  # ty: ignore[unresolved-attribute]
            assert pool._max_connections == 1000
            assert pool._max_keepalive_connections == 100
            assert client.follow_redirects is True
            assert client.timeout == connection_factory.DEFAULT_REQUEST_TIMEOUT
        await http_json.aclose()
        await sdk.close()

    def test_the_policy_names_the_new_meaning(self) -> None:
        assert (
            CONNECTION_FACTORY_REGISTRY.retry_policy("async_http_json", 3)
            == "sieval-http:max-retries=3"
        )
        with pytest.raises(ValueError, match="does not belong"):
            CONNECTION_FACTORY_REGISTRY.validate_retry_policy(
                "async_http_json", "httpx-transport:max-connect-retries=3"
            )


class TestModelArun:
    @pytest.fixture
    def wire(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[httpx.Request]]:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _sse_chat()

        def client(**kwargs: Any) -> RecordingOpenAIClient:
            return RecordingOpenAIClient(
                transport=httpx.MockTransport(handler), **kwargs
            )

        monkeypatch.setattr(
            "sieval.core.models._legacy_binding.RecordingOpenAIClient", client
        )
        yield seen

    @pytest.mark.anyio
    async def test_a_model_call_is_attributed_to_its_stage_and_sample(
        self, wire: list[httpx.Request]
    ) -> None:
        stage_limiter = anyio.CapacityLimiter(3)
        model = ChatModel(
            model="cand", api_base="https://gw.example/v1", api_key=SECRET
        )
        sink = _Sink()
        async with stage_limiter:
            with stage_scope(_scope(sink, task_stage=stage_limiter)):
                await model.agenerate("q")

        [line] = sink.lines
        assert line["phase"] == "infer"
        assert line["role"] == "subject"
        assert line["model"] == {
            "id": "cand",
            "dialect": "openai_chat",
            "deployment": None,
        }
        assert (line["sample_id"], line["iteration"], line["repeat_index"]) == (7, 0, 2)
        assert line["run"] == "run-1"
        assert line["request_id"] == "req-ok"
        assert line["in_flight"] == {
            "task_stage": {"borrowed": 1, "total": 3},
            "connection": {"borrowed": 1, "total": None},
        }
        assert isinstance(line["latency_s"], float)
        # The usage the response reported, on the attempt that returned it.
        assert line["usage"] == {
            "input_tokens": 3,
            "output_tokens": 1,
            "total_tokens": 4,
        }
        assert SECRET not in json.dumps(sink.lines)

    @pytest.mark.anyio
    async def test_a_judge_is_a_judge_whatever_stage_calls_it(
        self, wire: list[httpx.Request]
    ) -> None:
        judge = ChatModel(model="judge", api_base="https://gw.example/v1", api_key="k")
        sink = _Sink(roles={"judge": "grader"})
        with stage_scope(_scope(sink, stage="infer")):
            await judge.agenerate("q")
        [line] = sink.lines
        assert (line["phase"], line["role"], line["stage"]) == (
            "judge",
            "grader",
            "infer",
        )

    @pytest.mark.anyio
    async def test_a_model_local_cap_is_recorded_with_the_shared_quota(
        self, wire: list[httpx.Request]
    ) -> None:
        parent = ChatModel(
            model="m",
            api_base="https://gw.example/v1",
            api_key="k",
            concurrency_limit=8,
        )
        capped = parent.with_args(concurrency_limit=2)
        sink = _Sink()
        with stage_scope(_scope(sink)):
            await capped.agenerate("q")
        [line] = sink.lines
        in_flight = line["in_flight"]
        assert isinstance(in_flight, dict)
        assert in_flight["model_shared"] == {"borrowed": 1, "total": 8}
        assert in_flight["model_local"] == {"borrowed": 1, "total": 2}


class TestHelpers:
    def test_limiter_snapshot_reads_an_unbounded_total_as_none(self) -> None:
        limiter = anyio.CapacityLimiter(math.inf)
        assert rl.limiter_snapshot(limiter) == {"borrowed": 0, "total": None}

    def test_retry_after_header(self) -> None:
        def response(**headers: str) -> httpx.Response:
            return httpx.Response(429, headers=headers)

        assert rl.retry_after_header_s(response(**{"retry-after-ms": "250"})) == 0.25
        assert rl.retry_after_header_s(response(**{"retry-after": "3"})) == 3.0
        # An HTTP date is read relative to now, as the SDK reads it.
        past = rl.retry_after_header_s(
            response(**{"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
        )
        assert past is not None and past < 0
        assert rl.retry_after_header_s(response(**{"retry-after": "soon"})) is None
        assert rl.retry_after_header_s(response()) is None

    @pytest.mark.parametrize(
        ("status", "outcome"),
        [(200, "2xx"), (429, "429"), (404, "4xx"), (502, "5xx"), (302, "other")],
    )
    def test_status_outcome(self, status: int, outcome: str) -> None:
        assert rl.status_outcome(status) == outcome

    def test_request_id_header_order(self) -> None:
        response = httpx.Response(200, headers={"request-id": "b", "x-request-id": "a"})
        assert rl.request_id_of(response) == "a"
        assert rl.request_id_of(httpx.Response(200, headers={"request-id": "b"})) == "b"
        assert rl.request_id_of(httpx.Response(200)) is None

    @pytest.mark.anyio
    async def test_error_outcome(self) -> None:
        request = httpx.Request("GET", "https://x")
        assert rl.error_outcome(httpx.ConnectError("x", request=request)) == (
            "exception::ConnectError",
            "connection_error",
        )
        assert rl.error_outcome(ValueError("secret body")) == (
            "exception::ValueError",
            "other",
        )
        assert rl.error_outcome(anyio.get_cancelled_exc_class()()) == (
            "cancelled",
            "cancelled",
        )


class TestTimeoutPhase:
    _REQUEST = httpx.Request("POST", "http://127.0.0.1:30000/v1/chat/completions")

    @pytest.mark.parametrize(
        ("error_type", "phase"),
        [
            (httpx.ConnectTimeout, "connect"),
            (httpx.ReadTimeout, "read"),
            (httpx.WriteTimeout, "write"),
            (httpx.PoolTimeout, "pool"),
        ],
    )
    def test_reads_the_phase_through_the_sdk_wrapper(
        self, error_type: type[httpx.TimeoutException], phase: str
    ) -> None:
        try:
            try:
                raise error_type("timed out", request=self._REQUEST)
            except httpx.TimeoutException as err:
                raise openai.APITimeoutError(request=self._REQUEST) from err
        except openai.APITimeoutError as wrapped:
            assert timeout_phase(wrapped) == phase

    def test_an_error_without_a_timeout_cause_has_no_phase(self) -> None:
        assert timeout_phase(RuntimeError("boom")) is None
        assert timeout_phase(openai.APITimeoutError(request=self._REQUEST)) is None

    def test_a_timeout_only_in_the_context_has_no_phase(self) -> None:
        """An unrelated error raised while handling a timeout is not one."""
        try:
            try:
                raise httpx.ReadTimeout("timed out", request=self._REQUEST)
            except httpx.TimeoutException:
                raise ValueError("grader failed")  # noqa: B904
        except ValueError as unrelated:
            assert timeout_phase(unrelated) is None

    def test_a_cyclic_cause_chain_terminates(self) -> None:
        first = RuntimeError("a")
        second = RuntimeError("b")
        first.__cause__ = second
        second.__cause__ = first

        assert timeout_phase(first) is None
