"""Shared machinery for math tasks that may run Python while solving.

The loop, the protocol adapters and the sandbox client live here; a leaf binds a
benchmark's dataset and reuses that benchmark's own prompt, extractor and grader
unchanged. Keeping the grader identical to the no-tool sibling is what makes the
pair a measurable difference rather than two unrelated numbers.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
import re
from dataclasses import dataclass, field

import httpx

from sieval.core.models import ModelOutput
from sieval.core.utils.serialization import sieval_record

#: Mirrors the service's own cap. Kept here as well so a client pointed at an
#: older service still bounds what reaches a shard record and the model's context.
MAX_STREAM_CHARS = 8192

DEFAULT_CODE_RUN_API = "http://localhost:11451/code-runs"


def _get[T](data: dict, key: str, default: T) -> T:
    """Read *key* from *data*, filling in for both "absent" and "explicit null".

    ``dict.get(key, default)`` substitutes *default* only when the key is
    missing; a service that sends the key with an explicit ``null`` -- valid
    JSON regardless of what this route's schema currently promises -- passes
    ``None`` through unchanged, and a caller that expected *default*'s type
    breaks on it instead. Every optional-but-typed field pulled out of a
    response goes through this one function, so a field added later inherits
    the guard rather than needing its own copy of this reasoning.
    """
    value = data.get(key, default)
    return default if value is None else value


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

        A sandbox that is down, slow, malformed, or has drifted from this
        client's expected response shape produces a ToolCall carrying the
        reason, which the loop feeds back to the model and the record keeps.
        Raising instead would fail the sample -- and under DENOMINATOR_REQUESTED
        a failed sample is charged as wrong, so one flaky request or one
        unexpected null would read as a model that could not do arithmetic.
        Mapping the response into a ToolCall is inside the same guarded region
        as sending the request, rather than a narrower guard that trusts the
        body once it parses as JSON: a response shaped unexpectedly -- not a
        dict at all, or a field null where today's contract has it required --
        is exactly as unpredictable as a response that never arrives, and both
        need the same one fallback.
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
            data = body.get("data") or {}
            self.service_version = _get(data, "service_version", self.service_version)
            return ToolCall(
                index=0,
                code=code,
                stdout=_get(data, "stdout", "")[:MAX_STREAM_CHARS],
                stderr=(_get(data, "stderr", "") or _get(body, "msg", ""))[
                    :MAX_STREAM_CHARS
                ],
                exit_code=data.get("exit_code"),
                timed_out=bool(_get(data, "timed_out", False)),
                truncated=bool(_get(data, "truncated", False)),
                wall_s=float(_get(data, "wall_s", 0.0)),
                session_id=data.get("session_id"),
            )
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


#: The protocol, stated to the model. PINNED: this string is part of what the
#: task measures, so rewording it moves every score and makes a stored delta
#: incomparable to a fresh one. A test asserts its exact bytes.
TOOL_SYSTEM_PROMPT = (
    "You may run Python to help you compute. To do so, write a single fenced "
    "block:\n"
    "```python\n"
    "# your code; print() what you need to see\n"
    "```\n"
    "Stop immediately after the closing fence. The program's output will be "
    "given to you in the next message, and you may then run more code or give "
    "your final answer. Code you do not print produces no output. Each block "
    "runs in a fresh interpreter, so repeat any definitions you still need."
)

#: Stop sequence. Ending generation at the closing fence is what keeps the model
#: from inventing its own tool output; the adapter still truncates, because not
#: every backend honours `stop`.
FENCE_STOP = "```\n"

_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


class TextToolAdapter:
    """The fenced-code-block protocol: any chat model can speak it."""

    protocol = "text"

    def system_prompt(self) -> str:
        return TOOL_SYSTEM_PROMPT

    def extract_call(self, output: ModelOutput) -> tuple[str, str] | None:
        """``(code, discarded_tail)``, or None when the model did not call.

        None covers both "answered" and "produced nothing usable" -- an
        unterminated fence included, because executing a program the model was
        cut off mid-way through would grade a syntax error it never wrote.
        """
        if not output.texts:
            return None
        text = output.texts[0]
        match = _FENCE.search(text)
        if match is None:
            return None
        code = match.group(1)
        if not code.strip():
            return None
        return code, text[match.end() :]

    def render(self, call: ToolCall) -> list[dict]:
        """The assistant turn that made the call, and the result turn.

        The assistant turn is rebuilt from the CODE rather than replayed from the
        reply, so the invented tail is not smuggled back into context by the very
        message that reports the real output.
        """
        if call.timed_out:
            result = f"The code timed out and was stopped.\n{call.stderr}".strip()
        elif call.exit_code != 0:
            result = (call.stderr or "The code failed with no output.").strip()
        elif not call.stdout.strip():
            result = "The code ran and printed nothing."
        else:
            result = call.stdout.rstrip()
        if call.truncated:
            result += "\n[output truncated]"
        return [
            {"role": "assistant", "content": f"```python\n{call.code}```"},
            {"role": "user", "content": f"Output:\n```\n{result}\n```"},
        ]
