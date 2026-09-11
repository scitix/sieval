"""Shared machinery for math tasks that may run Python while solving.

The loop, the protocol adapters and the sandbox client live here; a leaf binds a
benchmark's dataset and reuses that benchmark's own prompt, extractor and grader
unchanged. Keeping the grader identical to the no-tool sibling is what makes the
pair a measurable difference rather than two unrelated numbers.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
from dataclasses import dataclass, field

import httpx

from sieval.core.models import ModelOutput
from sieval.core.utils.serialization import sieval_record

#: Mirrors the service's own cap. Kept here as well so a client pointed at an
#: older service still bounds what reaches a shard record and the model's context.
MAX_STREAM_CHARS = 8192

DEFAULT_CODE_RUN_API = "http://localhost:11451/code-runs"


@sieval_record
@dataclass
class ToolCall:
    """One code execution: what was sent, and everything that came back."""

    index: int
    code: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated: bool
    wall_s: float
    #: Reserved for a future stateful backend; always None while execution is
    #: one fresh process per call. Absent rather than null once serialized.
    session_id: str | None = None
    #: Text the model generated after its closing fence, which was discarded
    #: rather than sent. Empty in the ordinary case. Recorded because a model
    #: that writes its own fake tool output is scoring on a hallucination, and
    #: the rate has to be measurable rather than assumed.
    discarded_tail: str = ""


@sieval_record
@dataclass
class RolloutTrajectory:
    """One rollout's whole solve: every model call and every execution."""

    index: int
    outputs: list[ModelOutput] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: "answered" | "budget_exhausted" | "no_tool_use" | "request_failed"
    stop_reason: str = "answered"


@sieval_record
@dataclass
class MathToolTrajectory:
    """Every rollout of one sample, plus what produced them."""

    rollouts: list[RolloutTrajectory]
    protocol: str
    sandbox: dict


class SandboxClient:
    """One code-evaluator `/code-runs` endpoint, as a source of `ToolCall`s."""

    def __init__(
        self,
        *,
        api: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float,
        memory_limit: int = 1024,
        max_connections: int = 8,
    ) -> None:
        self._api = (
            api if api else os.getenv("SIEVAL_CODE_RUN_API", DEFAULT_CODE_RUN_API)
        )
        self._client = http_client or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections)
        )
        self._timeout = timeout
        self._memory_limit = memory_limit
        self.service_version: str | None = None

    async def run(self, code: str) -> ToolCall:
        """Execute *code*; never raise.

        A sandbox that is down, slow or malformed produces a ToolCall carrying
        the reason, which the loop feeds back to the model and the record keeps.
        Raising instead would fail the sample -- and under DENOMINATOR_REQUESTED
        a failed sample is charged as wrong, so one flaky request would read as
        a model that could not do arithmetic.
        """
        try:
            resp = await self._client.post(
                self._api,
                json={
                    "uuid": f"{id(self):x}-{len(code)}",
                    "lang": "python",
                    "code": code,
                    "timeout": self._timeout,
                    "memory_limit": self._memory_limit,
                },
                # Above the service's own wall, so a server-side timeout comes
                # back as a RESULT (timed_out=True) rather than as a client
                # abort that cannot tell a slow snippet from a dead service.
                timeout=self._timeout + 15.0,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            return ToolCall(
                index=0,
                code=code,
                stdout="",
                stderr=f"sandbox unreachable: [{type(exc).__name__}] {exc}",
                exit_code=None,
                timed_out=False,
                truncated=False,
                wall_s=0.0,
            )
        data = body.get("data") or {}
        self.service_version = data.get("service_version") or self.service_version
        return ToolCall(
            index=0,
            code=code,
            stdout=data.get("stdout", "")[:MAX_STREAM_CHARS],
            stderr=(data.get("stderr") or body.get("msg", ""))[:MAX_STREAM_CHARS],
            exit_code=data.get("exit_code"),
            timed_out=bool(data.get("timed_out", False)),
            truncated=bool(data.get("truncated", False)),
            wall_s=float(data.get("wall_s", 0.0)),
            session_id=data.get("session_id"),
        )
