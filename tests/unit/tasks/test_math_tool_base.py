import os

import httpx
import pytest

from sieval.core.models import ChatModel, ModelOutput, Request, Response
from sieval.core.tasks import TaskStageOutput
from sieval.core.utils.serialization import dict_to_obj, obj_to_dict
from sieval.tasks._math_tool_base import (
    MathToolTrajectory,
    RolloutTrajectory,
    SandboxClient,
    TextToolAdapter,
    ToolCall,
)
from tests.conftest import HandlerTransport


def _transport(payload: dict, status: int = 200) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_run_maps_a_successful_reply():
    client = SandboxClient(
        api="http://x/code-runs",
        http_client=_transport(
            {
                "status": True,
                "msg": "",
                "data": {
                    "stdout": "42\n",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                    "truncated": False,
                    "wall_s": 0.01,
                    "session_id": None,
                    "service_version": "code-runs/1",
                },
            }
        ),
        timeout=5.0,
    )
    call = await client.run("print(42)")
    assert call.stdout == "42\n"
    assert call.exit_code == 0
    assert call.timed_out is False
    assert call.session_id is None
    assert client.service_version == "code-runs/1"


@pytest.mark.anyio
async def test_a_transport_failure_becomes_a_recorded_call_not_a_raise():
    # The sandbox being unreachable must not fail the sample: the model gets a
    # message it can react to, and the trajectory records what happened. A raise
    # here would fail every sample in the run for one flaky request, and the
    # runner would charge them all as wrong.
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = SandboxClient(
        api="http://x/code-runs",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
        timeout=5.0,
    )
    call = await client.run("print(1)")
    assert call.exit_code is None
    assert "ConnectError" in call.stderr
    assert call.stdout == ""


@pytest.mark.anyio
async def test_a_non_dict_response_body_becomes_a_recorded_call_not_a_raise():
    # A body that parses as JSON but is not a dict at all -- a list, a bare
    # string, top-level null -- offers no `.get`, so mapping the response has
    # to sit inside the same guarded region as the request itself, rather than
    # a narrower guard that trusts anything which came back as valid JSON.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = SandboxClient(
        api="http://x/code-runs",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timeout=5.0,
    )
    call = await client.run("print(1)")
    assert call.exit_code is None
    assert "AttributeError" in call.stderr
    assert call.stdout == ""


@pytest.mark.anyio
async def test_stdout_explicit_null_is_mapped_to_empty_not_a_raise():
    # `.get(key, default)` only substitutes the default when the key is
    # ABSENT; a service that sends `"stdout": null` passes None through
    # unchanged, and slicing None used to raise. The rest of the payload is
    # still legitimate, so it must still come through rather than being
    # discarded along with the one unexpected field.
    client = SandboxClient(
        api="http://x/code-runs",
        http_client=_transport(
            {
                "status": True,
                "msg": "",
                "data": {
                    "stdout": None,
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                    "truncated": False,
                    "wall_s": 0.02,
                    "session_id": None,
                    "service_version": "code-runs/1",
                },
            }
        ),
        timeout=5.0,
    )
    call = await client.run("print(1)")
    assert call.stdout == ""
    assert call.exit_code == 0
    assert call.wall_s == 0.02


@pytest.mark.anyio
async def test_wall_s_explicit_null_is_mapped_to_zero_not_a_raise():
    # Same defect, different field: `float(None)` raises before this fix.
    client = SandboxClient(
        api="http://x/code-runs",
        http_client=_transport(
            {
                "status": True,
                "msg": "",
                "data": {
                    "stdout": "1\n",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                    "truncated": False,
                    "wall_s": None,
                    "session_id": None,
                    "service_version": "code-runs/1",
                },
            }
        ),
        timeout=5.0,
    )
    call = await client.run("print(1)")
    assert call.wall_s == 0.0
    assert call.stdout == "1\n"
    assert call.exit_code == 0


def test_trajectory_round_trips_and_omits_a_null_session_id():
    trajectory = MathToolTrajectory(
        rollouts=[
            RolloutTrajectory(
                index=0,
                outputs=[],
                tool_calls=[
                    ToolCall(
                        index=0,
                        code="print(1)",
                        stdout="1\n",
                        stderr="",
                        exit_code=0,
                        timed_out=False,
                        truncated=False,
                        wall_s=0.01,
                    )
                ],
                stop_reason="answered",
            )
        ],
        protocol="text",
        sandbox={"service_version": "code-runs/1"},
    )
    flat = obj_to_dict(trajectory, add_type=True)
    # `obj_to_dict` drops None values, so a null session_id is ABSENT on disk
    # rather than stored as null. Asserted because a reader checking
    # `["session_id"]` would KeyError only on records written in this mode.
    assert "session_id" not in flat["rollouts"][0]["tool_calls"][0]
    back = dict_to_obj(flat, {})
    assert back.rollouts[0].tool_calls[0].stdout == "1\n"
    assert back.rollouts[0].tool_calls[0].session_id is None
    assert back.protocol == "text"


def test_a_sandbox_outage_call_round_trips_as_a_tool_call_not_a_dict():
    # An outage-shaped call has exit_code=None, which obj_to_dict drops from the
    # persisted record the same way it drops a null session_id. Unlike
    # session_id, exit_code IS read back as an attribute (postprocess reads
    # call.exit_code, call.timed_out, call.discarded_tail), so if ToolCall
    # cannot be reconstructed from a payload missing that key, dict_to_obj's
    # `suppress(Exception)` around `target(**payload)` swallows the TypeError
    # and hands back a raw dict -- and the first attribute access on it raises
    # AttributeError on a resume, long after the run that produced the record.
    trajectory = MathToolTrajectory(
        rollouts=[
            RolloutTrajectory(
                index=0,
                outputs=[],
                tool_calls=[
                    ToolCall(
                        index=0,
                        code="print(1)",
                        stdout="",
                        stderr="sandbox unreachable: [ConnectError] refused",
                        exit_code=None,
                        timed_out=False,
                        truncated=False,
                        wall_s=0.0,
                        object="sandbox",
                        reason="unreachable",
                    )
                ],
                stop_reason="sandbox_unreachable",
            )
        ],
        protocol="text",
        sandbox={"service_version": None, "fully_served": False},
    )
    flat = obj_to_dict(trajectory, add_type=True)
    back = dict_to_obj(flat, {})
    call = back.rollouts[0].tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.object == "sandbox"
    assert call.reason == "unreachable"
    # These three are exactly what MathToolTask.postprocess reads off a call.
    assert call.exit_code is None
    assert call.timed_out is False
    assert call.discarded_tail == ""


@pytest.mark.anyio
async def test_live_service_honours_the_contract():
    """Skipped unless a code-evaluator is reachable.

    The mock tests pin the client against this plan's description of the
    service. Only this one pins the description against the service, which is
    where a contract drifts: a field renamed on the server is invisible to every
    mocked test in the file.

    The variable is the FLEET's, not this task's: `SIEVAL_CODE_EVAL_API` names
    the `/evaluations` route every code-executing task already reads, and the
    `/code-runs` route this task needs is derived from it. Setting a
    task-specific variable instead would leave a deployment that only exports
    the established name pointed at `localhost`.
    """
    api = os.getenv("SIEVAL_CODE_EVAL_API")
    if not api:
        pytest.skip("SIEVAL_CODE_EVAL_API is not set")
    client = SandboxClient(api=api, timeout=10.0)
    call = await client.run("print(6 * 7)")
    assert call.stdout.strip() == "42"
    assert call.exit_code == 0
    assert client.service_version, "service_version is required by the contract"
    # The route was derived, not copied: a live service answering `/code-runs`
    # proves the join, since `api` itself points at `/evaluations`.
    assert call.object == "code"
    assert client.fully_served is True


@pytest.mark.anyio
async def test_the_code_runs_url_is_derived_from_the_fleet_variable(monkeypatch):
    # The one assertion a mocked transport can still make about the URL: what it
    # was actually POSTed to. Asserting on a private attribute would pass even if
    # `run()` posted somewhere else entirely.
    monkeypatch.setenv("SIEVAL_CODE_EVAL_API", "http://host:9/api/evaluations")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"status": True, "data": {"exit_code": 0}})

    client = SandboxClient(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timeout=5.0,
    )
    await client.run("print(1)")
    # The prefix survives, and only the last segment was swapped.
    assert seen == ["http://host:9/api/code-runs"]


@pytest.mark.anyio
async def test_an_explicit_api_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("SIEVAL_CODE_EVAL_API", "http://ignored/evaluations")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"status": True, "data": {"exit_code": 0}})

    client = SandboxClient(
        api="http://explicit/code-runs",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timeout=5.0,
    )
    await client.run("print(1)")
    assert seen == ["http://explicit/code-runs"]


@pytest.mark.anyio
async def test_an_injected_client_is_not_closed_by_the_sandbox():
    # The task calls `aclose()` on shutdown. A test's client outlives that call
    # and is reused by the next assertion, so closing one the sandbox did not
    # create would turn a passing test into an unrelated `RuntimeError`.
    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, json={"status": True, "data": {"exit_code": 0}}
            )
        )
    )
    client = SandboxClient(http_client=injected, timeout=5.0)
    await client.aclose()
    assert not injected.is_closed


@pytest.mark.anyio
async def test_a_sandbox_marks_itself_unserved_after_any_failure():
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = SandboxClient(
        api="http://x/code-runs",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
        timeout=5.0,
    )
    assert client.fully_served is True
    call = await client.run("print(1)")
    assert call.object == "sandbox"
    assert call.reason == "unreachable"
    # A version alone cannot say this: the flag is what makes a partly-served
    # run distinguishable from a fully-served one after the fact.
    assert client.fully_served is False


@pytest.mark.anyio
async def test_a_client_side_timeout_is_recorded_as_its_own_reason():
    # `TimeoutException` and `ConnectError` are siblings under `TransportError`,
    # so the ordering of the two handlers is what decides which reason lands --
    # a service that is up but wedged is a different operational fact from one
    # that refused the connection, and the record keeps them apart.
    def slow(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no reply")

    client = SandboxClient(
        api="http://x/code-runs",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(slow)),
        timeout=5.0,
    )
    call = await client.run("print(1)")
    assert call.object == "sandbox"
    assert call.reason == "timeout"
    assert call.timed_out is False, (
        "timed_out describes the SERVICE stopping the program; a client-side "
        "abort is a different fact and must not be conflated with it"
    )
    assert client.fully_served is False


def _out(text: str) -> ModelOutput:
    return ModelOutput(
        model={"model": "m", "api_base": None, "default_params": {}}, texts=[text]
    )


def test_extracts_a_fenced_block():
    result = TextToolAdapter().extract_call(
        _out("Let me compute.\n```python\nprint(6*7)\n```")
    )
    assert result is not None
    code, tail = result
    assert code == "print(6*7)\n"
    assert tail == ""


def test_no_fence_means_the_model_answered():
    assert TextToolAdapter().extract_call(_out("The answer is 42.")) is None


def test_text_after_the_closing_fence_is_discarded_and_recorded():
    # The model invented its own tool output. Everything after the closing fence
    # is dropped, and the tail is returned so the rate can be reported.
    result = TextToolAdapter().extract_call(
        _out("```python\nprint(6*7)\n```\nOutput: 42\nSo \\boxed{42}.")
    )
    assert result is not None
    code, tail = result
    assert code == "print(6*7)\n"
    assert "Output: 42" in tail


def test_an_unterminated_fence_is_not_a_call():
    # Ran out of tokens mid-code. Executing a truncated program would grade a
    # syntax error the model did not commit.
    assert TextToolAdapter().extract_call(_out("```python\nprint(6*7)")) is None


def test_empty_output_is_not_a_call():
    assert (
        TextToolAdapter().extract_call(
            ModelOutput(
                model={"model": "m", "api_base": None, "default_params": {}}, texts=[]
            )
        )
        is None
    )


def test_render_feeds_stdout_back_as_a_user_turn():
    messages = TextToolAdapter().render(
        ToolCall(
            index=0,
            code="print(1)",
            stdout="1\n",
            stderr="",
            exit_code=0,
            timed_out=False,
            truncated=False,
            wall_s=0.0,
        )
    )
    assert messages[0]["role"] == "assistant"
    assert "print(1)" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "1" in messages[1]["content"]


def test_render_reports_an_error_to_the_model():
    messages = TextToolAdapter().render(
        ToolCall(
            index=0,
            code="1/0",
            stdout="",
            stderr="ZeroDivisionError: x",
            exit_code=1,
            timed_out=False,
            truncated=False,
            wall_s=0.0,
        )
    )
    assert "ZeroDivisionError" in messages[1]["content"]


def test_render_reports_a_timeout_to_the_model():
    messages = TextToolAdapter().render(
        ToolCall(
            index=0,
            code="while True: pass",
            stdout="",
            stderr="killed after 10s",
            exit_code=None,
            timed_out=True,
            truncated=False,
            wall_s=10.0,
        )
    )
    assert "timed out" in messages[1]["content"]
    assert "killed after 10s" in messages[1]["content"]


def test_render_tells_the_model_nothing_was_printed():
    # A run that exits cleanly but prints nothing must read differently from a
    # genuine success -- otherwise the model reasons from an output that was
    # never produced, which is the same silent-failure shape this adapter
    # exists to prevent on the other side of the call.
    messages = TextToolAdapter().render(
        ToolCall(
            index=0,
            code="x = 1",
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
            truncated=False,
            wall_s=0.01,
        )
    )
    assert "printed nothing" in messages[1]["content"]


def test_the_protocol_prompt_is_pinned():
    # Not a tautology check: this is the one string that silently changes every
    # score. A full-literal equality check -- not a hash, not substrings -- is
    # what makes a reword show up as an actual two-sided diff, forcing whoever
    # edits it to confront that stored tool-vs-no-tool deltas are now invalid.
    from sieval.tasks._math_tool_base import TOOL_SYSTEM_PROMPT

    expected = (
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
    assert expected == TOOL_SYSTEM_PROMPT


class _ScriptedModel(ChatModel):
    """Replies from a list, one per request.

    A real `ChatModel` over a stub transport, NOT a duck type: `Task.__init__`
    runs `_validate_model_requirements`, which reads `dialect_id` and the runtime
    plan off the model and rejects anything that merely looks like one. This is
    the shape `tests/unit/tasks/test_math_pass_at_k_family.py` already uses for
    every member of this family.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.n_requests = 0
        # Kept on the model so tests can reach the `Request`s the transport
        # recorded, rather than only observing the loop's own outputs.
        self.transport = HandlerTransport(self._stub_arun, "openai_chat")
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return self.transport

    async def _stub_arun(self, req: Request) -> Response:
        self.n_requests += 1
        text = self._replies.pop(0) if self._replies else r"\boxed{0}"
        return Response(texts=(text,))


class _DyingModel(_ScriptedModel):
    """Answers the first request; every later one fails."""

    async def _stub_arun(self, req: Request) -> Response:
        if self.n_requests:
            self.n_requests += 1
            raise RuntimeError("gateway died")
        return await super()._stub_arun(req)


class _DyingAfterModel(_ScriptedModel):
    """Answers *n* requests, then fails every later one.

    Needed to put a DEAD SANDBOX and a DEAD MODEL in the same rollout, in that
    order: a dead sandbox alone ends in an answer, so the only way to make the
    two reasons compete is to let the tool call fail first and the request fail
    second -- which is also the only order in which the loser is decided by the
    override rule rather than by which branch ran.
    """

    def __init__(self, replies, survive: int):
        self._survive = survive
        super().__init__(replies)

    async def _stub_arun(self, req: Request) -> Response:
        if self.n_requests >= self._survive:
            self.n_requests += 1
            raise RuntimeError("gateway died")
        return await super()._stub_arun(req)


class _ScriptedSandbox:
    """Stands in for `SandboxClient`; the loop only ever calls `.run`."""

    service_version = "code-runs/1"
    fully_served = True

    def __init__(self, stdouts):
        self._stdouts = list(stdouts)
        self.codes = []

    async def run(self, code):
        self.codes.append(code)
        stdout = self._stdouts.pop(0) if self._stdouts else ""
        return ToolCall(
            index=0,
            code=code,
            stdout=stdout,
            stderr="",
            exit_code=0,
            timed_out=False,
            truncated=False,
            wall_s=0.0,
        )


class _DeadSandbox:
    """Every call comes back as one that never reached the service.

    `object="sandbox"` is the whole distinction under test: `exit_code` is None
    here exactly as it is for a program that ran and crashed, so anything reading
    only the exit code cannot tell an outage from the model's own bug.
    """

    service_version: str | None = None
    fully_served = False

    def __init__(self, serve_first: bool = False, stdout: str = "42\n"):
        # `serve_first` models the mixed case: one healthy call, then the service
        # dies. That is the shape whose damage a version string cannot express --
        # `service_version` is stamped by the successful call and then describes a
        # run that was only partly served.
        self._serve_first = serve_first
        self._stdout = stdout
        self.service_version = "code-runs/1" if serve_first else None
        self.codes: list[str] = []
        self.calls = 0

    async def run(self, code):
        self.calls += 1
        self.codes.append(code)
        if self._serve_first and self.calls == 1:
            return ToolCall(
                index=0,
                code=code,
                stdout=self._stdout,
                stderr="",
                exit_code=0,
                timed_out=False,
                truncated=False,
                wall_s=0.0,
            )
        return ToolCall(
            index=0,
            code=code,
            stdout="",
            stderr="sandbox unreachable: [ConnectError] refused",
            exit_code=None,
            timed_out=False,
            truncated=False,
            wall_s=0.0,
            object="sandbox",
            reason="unreachable",
        )


@pytest.mark.anyio
async def test_loop_runs_code_then_answers():
    from sieval.tasks._math_tool_base import run_tool_loop

    model = _ScriptedModel(["```python\nprint(42)\n```", "So \\boxed{42}."])
    sandbox = _ScriptedSandbox(["42\n"])
    traj = await run_tool_loop(
        model=model,
        sandbox=sandbox,
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=4,
    )
    assert traj.stop_reason == "answered"
    assert len(traj.outputs) == 2
    assert [c.code for c in traj.tool_calls] == ["print(42)\n"]
    assert sandbox.codes == ["print(42)\n"]
    # The stop sequence is the only thing that stops the model generating past
    # its own closing fence, inventing an output, and reasoning from it -- the
    # one failure here that produces a plausible wrong answer instead of an
    # error. Dropping `stop=` from the loop's `agenerate` call leaves every
    # other assertion in this file green, so it is asserted against the Request
    # the transport actually received. The literal is spelled out rather than
    # read off `FENCE_STOP`: comparing the constant to itself cannot fail, and
    # the wire is what the model obeys.
    assert [req.sampling.stop for req in model.transport.requests] == [
        ("```\n",),
        ("```\n",),
    ]


@pytest.mark.anyio
async def test_loop_stops_at_the_budget():
    from sieval.tasks._math_tool_base import run_tool_loop

    model = _ScriptedModel(["```python\nprint(1)\n```"] * 10)
    traj = await run_tool_loop(
        model=model,
        sandbox=_ScriptedSandbox([""] * 10),
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=2,
    )
    assert traj.stop_reason == "budget_exhausted"
    assert len(traj.tool_calls) == 2
    # One model call per tool call, plus the final one that had no budget left.
    assert len(traj.outputs) == 3


@pytest.mark.anyio
async def test_a_model_that_never_calls_is_not_an_error():
    from sieval.tasks._math_tool_base import run_tool_loop

    traj = await run_tool_loop(
        model=_ScriptedModel(["\\boxed{7}"]),
        sandbox=_ScriptedSandbox([]),
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=4,
    )
    assert traj.stop_reason == "no_tool_use"
    assert traj.tool_calls == []
    assert len(traj.outputs) == 1


@pytest.mark.anyio
async def test_a_failed_request_mid_loop_keeps_what_was_answered():
    from sieval.tasks._math_tool_base import run_tool_loop

    traj = await run_tool_loop(
        model=_DyingModel(["```python\nprint(1)\n```"]),
        sandbox=_ScriptedSandbox(["1\n"]),
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=4,
    )
    assert traj.stop_reason == "request_failed"
    assert len(traj.outputs) == 1


@pytest.mark.anyio
async def test_a_total_sandbox_outage_is_recorded_and_stops_the_rollout():
    """The failure this whole finding is about: the service is simply not there.

    Before this, all of it was invisible: `exit_code is None` meant the counters
    read as healthy, `stop_reason` said "answered", and `render()` told the model
    its own code had failed. A rollout whose every execution attempt bounced is
    not evidence about the model, so the reason has to be readable off the
    rollout -- and it must OVERRIDE "answered", because the model did go on to
    answer, from its own prior knowledge, without the affordance it was being
    scored on.
    """
    from sieval.tasks._math_tool_base import run_tool_loop

    sandbox = _DeadSandbox()
    traj = await run_tool_loop(
        model=_ScriptedModel(["```python\nprint(6 * 7)\n```", r"\boxed{42}"]),
        sandbox=sandbox,
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=4,
    )
    assert traj.stop_reason == "sandbox_unreachable"
    assert len(traj.tool_calls) == 1, "the loop still feeds the failure back once"
    assert traj.tool_calls[0].object == "sandbox"
    assert traj.tool_calls[0].reason == "unreachable"
    # It answered anyway -- which is exactly why the stop reason has to be the
    # sandbox's and not the model's outcome.
    assert len(traj.outputs) == 2
    assert sandbox.fully_served is False


@pytest.mark.anyio
async def test_the_model_is_not_blamed_for_an_outage():
    # The rendered turn is what the model reasons from. "The code failed with no
    # output" would send it off rewriting a program that was never executed --
    # and would make an infrastructure fault look like a model behaviour in the
    # transcript a human later reads.
    adapter = TextToolAdapter()
    dead = _DeadSandbox()
    call = await dead.run("print(1)")
    assert call.exit_code is None, "the exit code alone cannot carry this"
    result = adapter.render(call)[1]["content"]
    assert "NOT run" in result
    assert "infrastructure" in result
    assert "failed with no output" not in result


@pytest.mark.anyio
async def test_a_partly_served_run_still_reports_the_outage():
    """One healthy call then an outage -- the case a version string lies about.

    `service_version` is stamped by the call that succeeded, so the sample reads
    as though the service was there for it. `fully_served` is what keeps the two
    apart, and `stop_reason` stays the sandbox's even though a real execution did
    happen earlier in the rollout.
    """
    from sieval.tasks._math_tool_base import run_tool_loop

    sandbox = _DeadSandbox(serve_first=True)
    traj = await run_tool_loop(
        model=_ScriptedModel(
            [
                "```python\nprint(6 * 7)\n```",
                "```python\nprint(7 * 7)\n```",
                r"\boxed{42}",
            ]
        ),
        sandbox=sandbox,
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=4,
    )
    assert sandbox.service_version == "code-runs/1", "the first call stamped this"
    assert sandbox.fully_served is False
    assert traj.stop_reason == "sandbox_unreachable"
    # The first call really ran, and is kept as such: dropping it would erase the
    # one execution the run did get.
    assert [call.object for call in traj.tool_calls] == ["code", "sandbox"]


@pytest.mark.anyio
async def test_a_dying_model_endpoint_still_reports_the_sandbox_when_it_failed():
    # Both reasons are true here, and the sandbox's wins: the one tool call
    # reached the service, the call after it did not, and then the model endpoint
    # died. The rollout is not a model result either way, so filing it as
    # `request_failed` would put a run the outage invalidated into the shard as a
    # model-side outcome. The override is unconditional rather than
    # order-dependent.
    from sieval.tasks._math_tool_base import run_tool_loop

    traj = await run_tool_loop(
        model=_DyingAfterModel(["```python\nprint(1)\n```"], survive=2),
        sandbox=_DeadSandbox(),
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=4,
    )
    assert traj.tool_calls[0].object == "sandbox"
    assert traj.stop_reason == "sandbox_unreachable"


@pytest.mark.anyio
async def test_a_model_request_failure_with_a_healthy_sandbox_is_kept():
    # The converse, which is what keeps the override from swallowing everything:
    # the one tool call the rollout made reached the service and really ran, so
    # there is no outage to report and the model's own failure stands.
    from sieval.tasks._math_tool_base import run_tool_loop

    traj = await run_tool_loop(
        model=_DyingAfterModel(["```python\nprint(1)\n```"], survive=1),
        sandbox=_DeadSandbox(serve_first=True),
        adapter=TextToolAdapter(),
        messages=[{"role": "user", "content": "q"}],
        index=0,
        max_tool_calls=4,
    )
    assert len(traj.tool_calls) == 1
    assert traj.tool_calls[0].object == "code"
    assert traj.stop_reason == "request_failed"


@pytest.fixture
def aime_tool_task():
    """A 2-rollout AIME tool task over a scripted model and sandbox.

    Constructed through a real leaf rather than a bare `MathToolTask` subclass
    defined here: `infer` is what boxes the stage meta, and the boxing has to be
    exercised on the same path a registered task takes -- a test-local subclass
    would pin its own copy of the class shape and stop covering the leaf.

    The model is `_ScriptedModel`, a real `ChatModel` over a stub transport,
    because `Task.__init__` validates the model's dialect and runtime plan.

    Imports are inside the fixture so this module does not drag the dataset and
    the leaf into every test in the file, and so the two stub classes above can
    be imported FROM here by the leaf's own test module without a cycle.
    """
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict as HFDatasetDict

    from sieval.datasets.aime_2025 import AIME2025Dataset
    from sieval.tasks.aime_2025_0shot_gen_tool import AIME2025ZeroShotGenToolTask

    rows = HFDataset.from_list([{"question": "What is 6 times 7?", "answer": "42"}])
    task = AIME2025ZeroShotGenToolTask(
        AIME2025Dataset(_hf_dict=HFDatasetDict({"train": rows, "test": rows})),
        # Two rollouts. Exactly ONE reply runs code and only ONE fenced block
        # exists in the script, so the TOTAL is fixed at three requests -- the
        # code reply plus two answers -- even though WHICH rollout draws the
        # fenced block is not: `asyncio.gather` interleaves the two loops, so the
        # first loop to call `agenerate` pops the code reply and the other starts
        # with an answer. Every reply leads to the same boxed value, so the
        # outcome is identical either way. Asserting a per-rollout split here
        # would be a flake; the per-rollout count is pinned deterministically by
        # `test_loop_runs_code_then_answers`, which drives a single loop and reads
        # the transport's own request list.
        _ScriptedModel(["```python\nprint(42)\n```", r"\boxed{42}", r"\boxed{42}"]),
        n=2,
        max_tool_calls=4,
    )
    task._sandbox = _ScriptedSandbox(["42\n"])
    return task


@pytest.mark.anyio
async def test_infer_reports_every_model_call_in_its_stage_meta(aime_tool_task):
    # Three model calls across two rollouts must all appear in `model_calls`.
    # Asserting the COUNT, not that a helper was called: a later refactor that
    # moves the boxing would keep a call-shape assertion green while the token
    # spend silently vanished from profile.json.
    #
    # The count is also the only assertion available here: rollouts run
    # concurrently under `asyncio.gather`, so which rollout pops which scripted
    # reply is not deterministic. The fixture is built so every reply leads to
    # the same answer, and a claim that a SPECIFIC rollout ran the code would be
    # a flake rather than a check. The per-rollout request count is pinned
    # deterministically by `test_loop_runs_code_then_answers` above, which drives
    # one rollout and reads the transport's requests directly.
    boxed = await aime_tool_task.infer(
        {"prompt": [{"role": "user", "content": "q"}]},
        aime_tool_task.make_context(0),
    )
    assert isinstance(boxed, TaskStageOutput)
    assert len(boxed.meta["model_calls"]) == 3
