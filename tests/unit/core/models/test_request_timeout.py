"""A call's ``timeout`` bounds the client, for every model client.

``timeout`` is a request argument a config sets (a model's ``args`` or a task's
``infer_args``); unset, each client keeps its declared default
(``client_timeout`` over ``DEFAULT_REQUEST_TIMEOUT``). Until 0.15.5 it reached
the wire body and never the client, so a board asking for 5400 s lost its
queued samples at the client's 600 s read timeout.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import anyio
import httpx
import openai
import pytest

from sieval.core.models import (
    DEFAULT_REQUEST_TIMEOUT,
    ChatModel,
    SglangGenModel,
)
from sieval.core.models.dialect import OutputContractError
from sieval.core.models.request_log import (
    RecordingAsyncClient,
    RecordingOpenAIClient,
    request_timeout_scope,
)

_BASE = "http://engine.test"


class _Server:
    """Answer with *payload* and keep every request it saw."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.payload)


def _sglang(server: _Server, **kwargs: Any) -> SglangGenModel:
    with patch(
        "sieval.core.models.connection_factory.httpx.AsyncHTTPTransport",
        return_value=httpx.MockTransport(server),
    ):
        return SglangGenModel(model="m", api_base=_BASE, api_key="local", **kwargs)


def _sglang_reply() -> dict[str, Any]:
    return {
        "text": "out",
        "meta_info": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "finish_reason": {"type": "stop"},
        },
    }


def _chat_reply() -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "out"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chat(server: _Server, **kwargs: Any) -> ChatModel:
    model = ChatModel(model="m", api_base=f"{_BASE}/v1", api_key="k", **kwargs)
    client = model.pool.connection
    assert isinstance(client, openai.AsyncOpenAI)
    # Keep the SDK's own client and swap only its transport, so the SDK's
    # per-request timeout handling is the one under test.
    http = client._client
    assert isinstance(http, RecordingOpenAIClient)
    http._transport = httpx.MockTransport(server)
    return model


def _phases(request: httpx.Request) -> dict[str, float | None]:
    return dict(request.extensions["timeout"])


class TestRequestTimeoutScope:
    @pytest.mark.anyio
    async def test_unset_keeps_the_clients_phases(self) -> None:
        server = _Server({})
        client = RecordingAsyncClient(
            transport=httpx.MockTransport(server),
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        async with client:
            with request_timeout_scope(None):
                await client.post("http://x/y")

        assert _phases(server.requests[0]) == {
            "connect": 30.0,
            "read": 600.0,
            "write": 600.0,
            "pool": 600.0,
        }

    @pytest.mark.anyio
    async def test_set_replaces_read_write_and_pool_and_keeps_connect(self) -> None:
        server = _Server({})
        client = RecordingAsyncClient(
            transport=httpx.MockTransport(server),
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        async with client:
            with request_timeout_scope(5400.0):
                await client.post("http://x/y")
            await client.post("http://x/y")

        assert _phases(server.requests[0]) == {
            "connect": 30.0,
            "read": 5400.0,
            "write": 5400.0,
            "pool": 5400.0,
        }
        # The scope ended: the next request is the client's again.
        assert _phases(server.requests[1])["read"] == 600.0

    @pytest.mark.anyio
    async def test_every_retry_keeps_the_requests_timeout(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(503 if len(seen) == 1 else 200, json={})

        client = RecordingAsyncClient(
            max_retries=1, transport=httpx.MockTransport(handler)
        )
        async with client:
            with (
                patch("sieval.core.models.request_log.anyio.sleep"),
                request_timeout_scope(1800.0),
            ):
                await client.post("http://x/y")

        assert [_phases(r)["read"] for r in seen] == [1800.0, 1800.0]


class TestModelTimeout:
    """``timeout`` set (model default, call argument) or unset, both families."""

    @pytest.mark.anyio
    async def test_sglang_unset_keeps_the_declared_default(self) -> None:
        server = _Server(_sglang_reply())
        model = _sglang(server)

        await model.agenerate("hi")

        assert _phases(server.requests[0]) == DEFAULT_REQUEST_TIMEOUT.as_dict()

    @pytest.mark.anyio
    async def test_sglang_a_model_timeout_bounds_the_client_and_is_not_sent(
        self,
    ) -> None:
        server = _Server(_sglang_reply())
        model = _sglang(server).with_args(timeout=5400, max_tokens=8)

        await model.agenerate("hi")

        request = server.requests[0]
        assert _phases(request)["read"] == 5400.0
        assert _phases(request)["pool"] == 5400.0
        assert _phases(request)["connect"] == DEFAULT_REQUEST_TIMEOUT.connect
        body = json.loads(request.content)
        assert "timeout" not in body
        assert "timeout" not in body.get("sampling_params", {})

    @pytest.mark.anyio
    async def test_sglang_a_call_timeout_beats_the_models(self) -> None:
        server = _Server(_sglang_reply())
        model = _sglang(server).with_args(timeout=5400)

        await model.agenerate("hi", timeout=120)

        assert _phases(server.requests[0])["read"] == 120.0

    @pytest.mark.anyio
    async def test_sglang_logprobs_honour_the_timeout(self) -> None:
        reply = _sglang_reply()
        reply["meta_info"]["input_token_logprobs"] = [[None, 1, None], [-0.5, 2, None]]
        reply["meta_info"]["output_token_logprobs"] = [[-0.1, 3, None]]
        server = _Server(reply)
        model = _sglang(server).with_args(timeout=900)

        # The reply carries no token text, so the call fails after it was sent;
        # what it was sent with is what is under test.
        with pytest.raises(OutputContractError, match="no token text"):
            await model.alogprobs("hi")

        assert _phases(server.requests[0])["read"] == 900.0

    @pytest.mark.anyio
    async def test_chat_unset_keeps_the_declared_default(self) -> None:
        server = _Server(_chat_reply())
        model = _chat(server)

        await model.agenerate("hi", stream=False)

        assert _phases(server.requests[0])["read"] == DEFAULT_REQUEST_TIMEOUT.read

    @pytest.mark.anyio
    async def test_chat_a_model_timeout_bounds_the_client_and_is_not_sent(
        self,
    ) -> None:
        server = _Server(_chat_reply())
        model = _chat(server).with_args(timeout=5400)

        await model.agenerate("hi", stream=False)

        request = server.requests[0]
        assert _phases(request)["read"] == 5400.0
        assert "timeout" not in json.loads(request.content)

    @pytest.mark.anyio
    async def test_a_client_timeout_stays_the_default_under_a_request_timeout(
        self,
    ) -> None:
        server = _Server(_sglang_reply())
        model = _sglang(server, client_timeout={"connect": 45, "read": 1200})

        await model.agenerate("hi")
        await model.with_args(timeout=3000).agenerate("hi")

        assert _phases(server.requests[0])["read"] == 1200.0
        assert _phases(server.requests[1])["read"] == 3000.0
        # connect is the client's either way.
        assert _phases(server.requests[1])["connect"] == 45.0

    @pytest.mark.parametrize("value", [0, -1, True, "600"])
    @pytest.mark.anyio
    async def test_a_bad_timeout_is_refused_before_anything_is_sent(
        self, value: object
    ) -> None:
        server = _Server(_sglang_reply())
        model = _sglang(server)

        with pytest.raises(ValueError, match="timeout must be a positive number"):
            await model.agenerate("hi", timeout=value)
        assert server.requests == []


@pytest.fixture
def slow_engine() -> Iterator[Callable[[float], str]]:
    """An HTTP server that answers /generate after *delay* seconds."""

    servers: list[socket.socket] = []
    stop = threading.Event()

    def start(delay: float) -> str:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        sock.settimeout(0.1)
        servers.append(sock)
        body = json.dumps(_sglang_reply()).encode()

        def serve() -> None:
            while not stop.is_set():
                try:
                    conn, _ = sock.accept()
                except OSError:
                    continue
                with conn:
                    conn.settimeout(5)
                    data = b""
                    while b"\r\n\r\n" not in data:
                        data += conn.recv(65536)
                    head, _, rest = data.partition(b"\r\n\r\n")
                    length = next(
                        int(line.split(b":")[1])
                        for line in head.split(b"\r\n")
                        if line.lower().startswith(b"content-length")
                    )
                    while len(rest) < length:
                        rest += conn.recv(65536)
                    time.sleep(delay)
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\n".encode()
                        + b"Connection: close\r\n\r\n"
                        + body
                    )

        threading.Thread(target=serve, daemon=True).start()
        return f"http://127.0.0.1:{sock.getsockname()[1]}"

    yield start
    stop.set()
    for sock in servers:
        sock.close()


class TestSlowEngine:
    """Over a real socket: the client waits as long as the config says."""

    @pytest.mark.anyio
    async def test_a_reply_slower_than_the_default_succeeds_within_timeout(
        self, slow_engine: Callable[[float], str]
    ) -> None:
        base = slow_engine(0.6)
        model = SglangGenModel(
            model="m",
            api_base=base,
            api_key="local",
            max_retries=0,
            client_timeout={"read": 0.2},
        ).with_args(timeout=3)

        with anyio.fail_after(10):
            output = await model.agenerate("hi")

        assert output.texts == ["out"]

    @pytest.mark.anyio
    async def test_without_timeout_the_clients_read_timeout_still_fires(
        self, slow_engine: Callable[[float], str]
    ) -> None:
        base = slow_engine(0.6)
        model = SglangGenModel(
            model="m",
            api_base=base,
            api_key="local",
            max_retries=0,
            client_timeout={"read": 0.2},
        )

        with anyio.fail_after(10), pytest.raises(httpx.ReadTimeout):
            await model.agenerate("hi")
