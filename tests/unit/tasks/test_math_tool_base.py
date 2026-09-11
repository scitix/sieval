import os

import httpx
import pytest

from sieval.core.models import ModelOutput
from sieval.core.utils.serialization import dict_to_obj, obj_to_dict
from sieval.tasks._math_tool_base import (
    MathToolTrajectory,
    RolloutTrajectory,
    SandboxClient,
    TextToolAdapter,
    ToolCall,
)


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


@pytest.mark.anyio
async def test_live_service_honours_the_contract():
    """Skipped unless a code-evaluator is reachable.

    The mock tests pin the client against this plan's description of the
    service. Only this one pins the description against the service, which is
    where a contract drifts: a field renamed on the server is invisible to every
    mocked test in the file.
    """
    api = os.getenv("SIEVAL_CODE_RUN_API")
    if not api:
        pytest.skip("SIEVAL_CODE_RUN_API is not set")
    client = SandboxClient(api=api, timeout=10.0)
    call = await client.run("print(6 * 7)")
    assert call.stdout.strip() == "42"
    assert call.exit_code == 0
    assert client.service_version, "service_version is required by the contract"


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


def test_the_protocol_prompt_is_pinned():
    # Not a tautology check: this is the one string that silently changes every
    # score. Editing it must be a deliberate act that fails this test first, and
    # any change invalidates stored deltas.
    from sieval.tasks._math_tool_base import TOOL_SYSTEM_PROMPT

    assert TOOL_SYSTEM_PROMPT.startswith("You may run Python to help you compute.")
    assert "Stop immediately after the closing fence." in TOOL_SYSTEM_PROMPT
    assert "fresh interpreter" in TOOL_SYSTEM_PROMPT
    assert len(TOOL_SYSTEM_PROMPT) == len(TOOL_SYSTEM_PROMPT.strip())
