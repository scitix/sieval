import os

import httpx
import pytest

from sieval.core.utils.serialization import dict_to_obj, obj_to_dict
from sieval.tasks._math_tool_base import (
    MathToolTrajectory,
    RolloutTrajectory,
    SandboxClient,
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
