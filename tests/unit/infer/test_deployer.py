"""
Unit tests for LocalDeployer.

Tests focus on unit-level behavior — subprocess launching is mocked to
avoid spawning real processes.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from sieval.core.models import ServingFacts
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferCondition, InferHandle, InferPhase
from sieval.infer.deployer import (
    DeployError,
    DeployTimeoutError,
    LocalDeployer,
)
from sieval.infer.topology.models import (
    DeploymentPlan,
    DeviceGroup,
    ParallelTopology,
    RoleAssignment,
    WellKnownRole,
)


def _make_handle(
    pid: str = "12345",
    role: str = "full",
    health_url: str = "http://localhost:8000/health",
    endpoint: str = "http://localhost:8000/v1",
    log_file: str = "/tmp/test.log",
    backend: str = "sglang",
) -> InferHandle:
    return InferHandle(
        backend=backend,
        handle_id=pid,
        endpoint=endpoint,
        metadata={
            "cmd": ["test", "serve"],
            "log_file": log_file,
            "role": role,
            "health_url": health_url,
        },
    )


def _make_command(
    role: str = "full",
    health_url: str = "http://localhost:8000/health",
    backend: str = "sglang",
) -> BackendCommand:
    return BackendCommand(
        cli_args=["echo", "test"],
        backend=backend,
        role=role,
        health_url=health_url,
    )


# ---------- _launch_one ----------


class TestLaunchOne:
    @pytest.mark.anyio
    async def test_missing_executable_names_the_binary(self):
        """A missing engine binary must be reported as such.

        Popen would raise a bare FileNotFoundError naming neither the role nor
        the command, and indistinguishable from a missing working_dir — while
        the operator's real question is whether the engine is installed.
        """
        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["sglang-not-installed", "serve", "--port", "30000"],
            backend="sglang",
            role="full",
            health_url="http://localhost:30000/health",
        )
        with pytest.raises(DeployError) as excinfo:
            await deployer._launch_one(cmd)

        message = str(excinfo.value)
        assert "sglang-not-installed" in message
        assert "PATH" in message
        assert "full" in message

    @pytest.mark.anyio
    async def test_engine_on_a_cmd_env_path_is_not_rejected(self, tmp_path: Path):
        """An engine reachable only via cmd.env's PATH must still launch.

        Popen resolves a bare argv[0] against the *child's* PATH, so a plan
        whose ``env:`` points PATH at the prefix holding the engine works.
        Probing this process's PATH instead would reject it before launch and
        report the engine as absent when it is installed.
        """
        engine = tmp_path / "sglang"
        engine.write_text("#!/bin/sh\nsleep 60\n")
        engine.chmod(0o755)

        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["sglang", "serve", "--port", "30000"],
            backend="sglang",
            role="full",
            env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
            health_url="http://localhost:30000/health",
        )

        with (
            patch("sieval.infer.deployer.subprocess.Popen") as mock_popen,
            patch("sieval.infer.deployer._LOG_DIR", Path("/tmp")),
        ):
            mock_popen.return_value = MagicMock(pid=4242)
            handle = await deployer._launch_one(cmd)

        assert handle.handle_id == "4242"

    @pytest.mark.anyio
    async def test_present_but_non_executable_is_not_called_missing(
        self, tmp_path: Path
    ):
        """A non-executable engine must not be reported as absent.

        Popen raises PermissionError there, not FileNotFoundError, and telling
        the operator it is "not found on PATH" sends them to install a binary
        that is already sitting in front of them.
        """
        engine = tmp_path / "sglang"
        engine.write_text("#!/bin/sh\nsleep 60\n")
        engine.chmod(0o644)

        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["sglang", "serve"],
            backend="sglang",
            role="full",
            env={"PATH": str(tmp_path)},
            health_url="http://localhost:30000/health",
        )

        with pytest.raises(DeployError) as excinfo:
            await deployer._launch_one(cmd)

        message = str(excinfo.value)
        assert "not executable" in message
        assert "not found on PATH" not in message

    @pytest.mark.anyio
    async def test_relative_executable_resolves_against_working_dir(
        self, tmp_path: Path
    ):
        """A relative argv[0] is resolved against working_dir, as Popen does.

        Popen(cwd=...) resolves a name carrying a directory relative to that
        cwd, so probing it relative to the deployer's own cwd would reject a
        launch that works.
        """
        (tmp_path / "bin").mkdir()
        engine = tmp_path / "bin" / "sglang"
        engine.write_text("#!/bin/sh\nsleep 60\n")
        engine.chmod(0o755)

        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["./bin/sglang", "serve"],
            backend="sglang",
            role="full",
            working_dir=str(tmp_path),
            health_url="http://localhost:30000/health",
        )

        with (
            patch("sieval.infer.deployer.subprocess.Popen") as mock_popen,
            patch("sieval.infer.deployer._LOG_DIR", Path("/tmp")),
        ):
            mock_popen.return_value = MagicMock(pid=777)
            handle = await deployer._launch_one(cmd)

        assert handle.handle_id == "777"

    @pytest.mark.anyio
    async def test_creates_handle_with_pid(self):
        """_launch_one should spawn a subprocess and return InferHandle."""
        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["sleep", "60"],
            backend="sglang",
            role="full",
            health_url="http://localhost:8000/health",
        )

        with (
            patch("sieval.infer.deployer.subprocess.Popen") as mock_popen,
            patch("sieval.infer.deployer._LOG_DIR", Path("/tmp")),
        ):
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            handle = await deployer._launch_one(cmd)

        assert handle.handle_id == "12345"
        assert handle.metadata["role"] == "full"
        assert handle.backend == "sglang"  # engine name, not role

    @pytest.mark.anyio
    async def test_backend_field_uses_engine_name(self):
        """handle.backend should be the engine name, not the role name."""
        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["echo", "test"],
            backend="vllm",
            role="prefill",
            health_url="http://localhost:8000/health",
        )

        with (
            patch("sieval.infer.deployer.subprocess.Popen") as mock_popen,
            patch("sieval.infer.deployer._LOG_DIR", Path("/tmp")),
        ):
            mock_process = MagicMock()
            mock_process.pid = 99999
            mock_popen.return_value = mock_process
            handle = await deployer._launch_one(cmd)

        assert handle.backend == "vllm"  # NOT "prefill"
        assert handle.metadata["role"] == "prefill"

    @pytest.mark.anyio
    async def test_endpoint_derived_from_health_url(self):
        """Endpoint should be derived from health_url → /v1 path."""
        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["echo", "test"],
            role="full",
            health_url="http://10.0.1.5:30000/health",
        )

        with (
            patch("sieval.infer.deployer.subprocess.Popen") as mock_popen,
            patch("sieval.infer.deployer._LOG_DIR", Path("/tmp")),
        ):
            mock_process = MagicMock()
            mock_process.pid = 99999
            mock_popen.return_value = mock_process
            handle = await deployer._launch_one(cmd)

        assert handle.endpoint == "http://10.0.1.5:30000/v1"


# ---------- status ----------


class TestStatus:
    @pytest.mark.anyio
    async def test_pid_not_alive_returns_stopped(self):
        deployer = LocalDeployer()
        handle = _make_handle()

        with patch("sieval.infer.deployer.pid_alive", return_value=False):
            phase, conditions = await deployer.status(handle)
        assert phase == InferPhase.STOPPED
        assert conditions["ready"].status is False
        assert conditions["ready"].reason == "process_exited"

    @pytest.mark.anyio
    async def test_invalid_pid_returns_failed(self):
        deployer = LocalDeployer()
        handle = _make_handle(pid="not-a-number")
        phase, conditions = await deployer.status(handle)
        assert phase == InferPhase.FAILED
        assert conditions["ready"].status is False

    @pytest.mark.anyio
    async def test_pid_alive_http_200_returns_ready(self):
        deployer = LocalDeployer()
        handle = _make_handle()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with (
            patch("sieval.infer.deployer.pid_alive", return_value=True),
            patch("sieval.infer.deployer.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            phase, conditions = await deployer.status(handle)
        assert phase == InferPhase.RUNNING
        assert conditions["ready"].status is True
        assert conditions["ready"].reason == ""

    @pytest.mark.anyio
    async def test_pid_alive_http_error_returns_not_ready(self):
        deployer = LocalDeployer()
        handle = _make_handle()

        import httpx

        with (
            patch("sieval.infer.deployer.pid_alive", return_value=True),
            patch("sieval.infer.deployer.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            phase, conditions = await deployer.status(handle)
        assert phase == InferPhase.RUNNING
        assert conditions["ready"].status is False
        assert conditions["ready"].reason == "connection_refused"

    @pytest.mark.anyio
    async def test_no_health_url_returns_running(self):
        deployer = LocalDeployer()
        handle = _make_handle(health_url="", endpoint="")

        with patch("sieval.infer.deployer.pid_alive", return_value=True):
            phase, conditions = await deployer.status(handle)
        assert phase == InferPhase.RUNNING
        assert conditions["ready"].status is False
        assert conditions["ready"].reason == "no_health_url"


# ---------- delete ----------


class TestDelete:
    @pytest.mark.anyio
    async def test_delete_sends_sigterm(self):
        deployer = LocalDeployer()
        handle = _make_handle()

        with (
            patch("sieval.infer.deployer.pid_alive", side_effect=[True, False]),
            patch("sieval.infer.deployer.kill_process_group") as mock_kill,
        ):
            await deployer.delete(handle)

        import signal

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    @pytest.mark.anyio
    async def test_delete_already_dead(self):
        deployer = LocalDeployer()
        handle = _make_handle()

        with (
            patch("sieval.infer.deployer.pid_alive", return_value=False),
            patch("sieval.infer.deployer.kill_process_group") as mock_kill,
        ):
            await deployer.delete(handle)

        mock_kill.assert_not_called()

    @pytest.mark.anyio
    async def test_delete_invalid_pid(self):
        deployer = LocalDeployer()
        handle = _make_handle(pid="invalid")
        # Should not raise
        await deployer.delete(handle)


# ---------- build_deployment ----------


class TestBuildDeployment:
    def test_single_full_role(self):
        deployer = LocalDeployer()
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role=WellKnownRole.FULL,
                    devices=DeviceGroup(count=8),
                    topology=ParallelTopology(tp=4, dp=2),
                ),
            ),
        )
        handles = [_make_handle(role="full", endpoint="http://localhost:30000/v1")]

        facts = ServingFacts(
            engine_version="0.4.6",
            tokenizer_available=True,
            prefix_cache_enabled=False,
            max_top_logprobs=20,
        )
        deployment = deployer.build_deployment(
            plan,
            handles,
            deployment_id="managed-1",
            metrics_url="http://localhost:30000/metrics",
            facts=facts,
        )

        assert deployment.deployment_id == "managed-1"
        assert deployment.api_base == "http://localhost:30000/v1"
        assert deployment.endpoints == {"full": "http://localhost:30000/v1"}
        assert deployment.engine.engine_id == "sglang"
        assert deployment.engine_source == "deployment"
        assert deployment.plan is not None
        assert deployment.plan.engine_id == "sglang"
        assert deployment.plan.fingerprint.startswith("sha256:")
        assert deployment.topology is not None
        assert deployment.topology.is_disaggregated is False
        assert deployment.topology.roles == ("full",)
        assert deployment.topology.total_gpus == 8
        assert deployment.metrics_url == "http://localhost:30000/metrics"
        assert deployment.facts is facts
        assert deployment.fingerprint.startswith("sha256:")

    def test_pd_disaggregated(self):
        deployer = LocalDeployer()
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role="prefill",
                    devices=DeviceGroup(count=4),
                    topology=ParallelTopology(tp=4),
                ),
                RoleAssignment(
                    role="decode",
                    devices=DeviceGroup(count=4),
                    topology=ParallelTopology(tp=2, dp=2),
                ),
            ),
        )
        handles = [
            _make_handle(role="prefill", endpoint="http://localhost:30000/v1"),
            _make_handle(role="decode", endpoint="http://localhost:30001/v1"),
        ]

        deployment = deployer.build_deployment(plan, handles)
        assert deployment.deployment_id is not None
        assert deployment.deployment_id.startswith("local:")
        assert deployment.topology is not None
        assert deployment.topology.is_disaggregated is True
        assert deployment.topology.roles == ("prefill", "decode")
        assert deployment.topology.total_gpus == 8
        assert len(deployment.endpoints) == 2
        assert deployment.facts == ServingFacts()

    def test_no_endpoint_handles(self):
        deployer = LocalDeployer()
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=4),
                    topology=ParallelTopology(tp=4),
                ),
            ),
        )
        handles = [_make_handle(role="full", endpoint="")]

        deployment = deployer.build_deployment(plan, handles)
        assert deployment.api_base == ""
        assert deployment.endpoints == {}

    def test_local_identity_is_independent_of_handle_order(self):
        deployer = LocalDeployer()
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role="prefill",
                    devices=DeviceGroup(count=1),
                    topology=ParallelTopology(),
                ),
                RoleAssignment(
                    role="decode",
                    devices=DeviceGroup(count=1),
                    topology=ParallelTopology(),
                ),
            ),
        )
        handles = [
            _make_handle(
                pid="101",
                role="prefill",
                endpoint="http://localhost:30000/v1",
            ),
            _make_handle(
                pid="102",
                role="decode",
                endpoint="http://localhost:30001/v1",
            ),
        ]

        first = deployer.build_deployment(plan, handles)
        second = deployer.build_deployment(plan, list(reversed(handles)))
        assert first.deployment_id == second.deployment_id
        assert first.api_base == second.api_base == "http://localhost:30001/v1"
        assert first.fingerprint == second.fingerprint

    def test_conflicting_endpoints_for_one_role_fail_loudly(self):
        deployer = LocalDeployer()
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=1),
                    topology=ParallelTopology(),
                ),
            ),
        )
        handles = [
            _make_handle(
                pid="101",
                role="full",
                endpoint="http://localhost:8000/v1",
                backend="vllm",
            ),
            _make_handle(
                pid="102",
                role="full",
                endpoint="http://localhost:8001/v1",
                backend="vllm",
            ),
        ]

        with pytest.raises(ValueError, match="different endpoints.*'full'"):
            deployer.build_deployment(plan, handles)


# ---------- logs ----------


class TestLogs:
    @pytest.mark.anyio
    async def test_logs_tail(self, tmp_path: Path):
        deployer = LocalDeployer()
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        handle = _make_handle(log_file=str(log_file))

        lines = []
        async for line in deployer.logs(handle, tail=3):
            lines.append(line)

        assert len(lines) == 3
        assert lines[0] == "line3"
        assert lines[-1] == "line5"

    @pytest.mark.anyio
    async def test_logs_no_file(self):
        deployer = LocalDeployer()
        handle = _make_handle(log_file="/nonexistent/path.log")

        lines = []
        async for line in deployer.logs(handle):
            lines.append(line)

        assert lines == []

    @pytest.mark.anyio
    async def test_logs_no_log_path(self):
        deployer = LocalDeployer()
        handle = InferHandle(
            backend="full",
            handle_id="12345",
            endpoint="http://localhost:8000/v1",
            metadata={},
        )

        lines = []
        async for line in deployer.logs(handle):
            lines.append(line)

        assert lines == []

    @pytest.mark.anyio
    async def test_logs_large_file_seeks(self, tmp_path: Path):
        """Large files: seek near end and drop partial first line."""
        deployer = LocalDeployer()
        log_file = tmp_path / "big.log"
        # Write a file large enough that chunk_size < file_size (triggers seek)
        # tail=3, chunk_size=3*256=768. Write >768 bytes.
        many_lines = [f"log line {i}: " + "x" * 200 for i in range(10)]
        log_file.write_text("\n".join(many_lines) + "\n")

        handle = _make_handle(log_file=str(log_file))
        lines = []
        async for line in deployer.logs(handle, tail=3):
            lines.append(line)

        assert len(lines) == 3
        assert "log line 9" in lines[-1]


# ---------- deploy (integration-level, mocked) ----------


class TestDeploy:
    @pytest.mark.anyio
    async def test_deploy_detach(self):
        """Detach mode should return immediately after launch."""
        deployer = LocalDeployer()
        cmd = _make_command()

        with patch.object(
            deployer,
            "_launch_one",
            new_callable=AsyncMock,
        ) as mock_launch:
            mock_launch.return_value = _make_handle()
            handles = await deployer.deploy([cmd], detach=True)

        assert len(handles) == 1
        assert handles[0].handle_id == "12345"

    @pytest.mark.anyio
    async def test_deploy_cleanup_on_failure(self):
        """If launch fails, already-launched handles should be cleaned up."""
        deployer = LocalDeployer()

        call_count = 0

        async def mock_launch(cmd):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("launch failed")
            return _make_handle(pid=str(10000 + call_count))

        with (
            patch.object(
                deployer,
                "_launch_one",
                side_effect=mock_launch,
            ),
            patch.object(
                deployer,
                "delete",
                new_callable=AsyncMock,
            ) as mock_delete,
            pytest.raises(RuntimeError, match="launch failed"),
        ):
            await deployer.deploy(
                [_make_command(), _make_command()],
                detach=True,
            )

        # First handle should have been cleaned up
        assert mock_delete.call_count == 1

    @pytest.mark.anyio
    async def test_deploy_polls_until_ready(self):
        """Non-detach deploy should poll until all handles are ready."""
        deployer = LocalDeployer()
        cmd = _make_command()

        poll_count = 0

        async def mock_status(handle):  # noqa: ARG001
            nonlocal poll_count
            poll_count += 1
            if poll_count < 3:
                return (
                    InferPhase.RUNNING,
                    {
                        "ready": InferCondition(
                            status=False, reason="health_check_failed"
                        )
                    },
                )
            return (InferPhase.RUNNING, {"ready": InferCondition(status=True)})

        with (
            patch.object(
                deployer,
                "_launch_one",
                new_callable=AsyncMock,
                return_value=_make_handle(),
            ),
            patch.object(deployer, "status", side_effect=mock_status),
        ):
            handles = await deployer.deploy([cmd], detach=False, poll_interval=0.01)

        assert len(handles) == 1
        assert poll_count >= 3

    @pytest.mark.anyio
    async def test_deploy_timeout_raises(self):
        """Deploy should raise DeployTimeoutError when timeout is exceeded."""
        deployer = LocalDeployer()
        cmd = _make_command()

        with (
            patch.object(
                deployer,
                "_launch_one",
                new_callable=AsyncMock,
                return_value=_make_handle(),
            ),
            patch.object(
                deployer,
                "status",
                new_callable=AsyncMock,
                return_value=(
                    InferPhase.RUNNING,
                    {
                        "ready": InferCondition(
                            status=False, reason="health_check_failed"
                        )
                    },
                ),
            ),
            patch.object(
                deployer,
                "delete",
                new_callable=AsyncMock,
            ),
            pytest.raises(DeployTimeoutError, match="Not all processes ready"),
        ):
            await deployer.deploy([cmd], detach=False, timeout=0.05, poll_interval=0.01)

    @pytest.mark.anyio
    async def test_deploy_timeout_quotes_engine_log(self, tmp_path: Path):
        """A timeout must quote the engine log, like a crash already does.

        The engine's log is written outside the run directory and the process
        is killed on the way out, so a timeout that does not quote it usually
        leaves no evidence of why the server never came up.
        """
        log_file = tmp_path / "full.log"
        log_file.write_text("Loading weights...\nCUDA out of memory\n")
        deployer = LocalDeployer()
        cmd = _make_command()

        with (
            patch.object(
                deployer,
                "_launch_one",
                new_callable=AsyncMock,
                return_value=_make_handle(log_file=str(log_file)),
            ),
            patch.object(
                deployer,
                "status",
                new_callable=AsyncMock,
                return_value=(
                    InferPhase.RUNNING,
                    {
                        "ready": InferCondition(
                            status=False, reason="connection_refused"
                        )
                    },
                ),
            ),
            patch.object(deployer, "delete", new_callable=AsyncMock),
            pytest.raises(DeployTimeoutError) as excinfo,
        ):
            await deployer.deploy([cmd], detach=False, timeout=0.05, poll_interval=0.01)

        message = str(excinfo.value)
        assert "CUDA out of memory" in message
        assert str(log_file) in message
        # The reason distinguishes "still loading" from "up but unhealthy".
        assert "connection_refused" in message
        # The launch command makes a wrong or absent engine visible here too.
        assert "test serve" in message

    @pytest.mark.anyio
    async def test_deploy_timeout_names_log_path_when_empty(self, tmp_path: Path):
        """An engine that logged nothing still gets its path named."""
        log_file = tmp_path / "silent.log"
        log_file.write_text("")
        deployer = LocalDeployer()
        cmd = _make_command()

        with (
            patch.object(
                deployer,
                "_launch_one",
                new_callable=AsyncMock,
                return_value=_make_handle(log_file=str(log_file)),
            ),
            patch.object(
                deployer,
                "status",
                new_callable=AsyncMock,
                return_value=(
                    InferPhase.RUNNING,
                    {
                        "ready": InferCondition(
                            status=False, reason="health_check_failed"
                        )
                    },
                ),
            ),
            patch.object(deployer, "delete", new_callable=AsyncMock),
            pytest.raises(DeployTimeoutError) as excinfo,
        ):
            await deployer.deploy([cmd], detach=False, timeout=0.05, poll_interval=0.01)

        message = str(excinfo.value)
        assert str(log_file) in message
        assert "empty" in message

    @pytest.mark.anyio
    async def test_deploy_process_died_raises(self):
        """Deploy should raise DeployError if a process dies during polling."""
        deployer = LocalDeployer()
        cmd = _make_command()

        with (
            patch.object(
                deployer,
                "_launch_one",
                new_callable=AsyncMock,
                return_value=_make_handle(),
            ),
            patch.object(
                deployer,
                "status",
                new_callable=AsyncMock,
                return_value=(
                    InferPhase.FAILED,
                    {"ready": InferCondition(status=False, reason="deploy_error")},
                ),
            ),
            patch.object(
                deployer,
                "_read_tail",
                new_callable=AsyncMock,
                return_value=["ERROR: out of memory"],
            ),
            patch.object(
                deployer,
                "delete",
                new_callable=AsyncMock,
            ),
            pytest.raises(DeployError, match="failed") as excinfo,
        ):
            await deployer.deploy([cmd], detach=False, poll_interval=0.01)

        # A crash quotes the same block a timeout does: tail, log path and the
        # launch command. The log is written outside the run directory, so the
        # path is no more discoverable after a crash than after a hang.
        message = str(excinfo.value)
        assert "ERROR: out of memory" in message
        assert "/tmp/test.log" in message
        assert "test serve" in message

    @pytest.mark.anyio
    async def test_deploy_progress_callback(self):
        """Progress callback should be called during polling."""
        deployer = LocalDeployer()
        cmd = _make_command()
        progress_calls: list[tuple[float, str]] = []

        poll_count = 0

        async def mock_status(handle):  # noqa: ARG001
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 2:
                return (InferPhase.RUNNING, {"ready": InferCondition(status=True)})
            return (
                InferPhase.RUNNING,
                {"ready": InferCondition(status=False, reason="health_check_failed")},
            )

        def on_progress(elapsed: float, status_str: str):
            progress_calls.append((elapsed, status_str))

        with (
            patch.object(
                deployer,
                "_launch_one",
                new_callable=AsyncMock,
                return_value=_make_handle(),
            ),
            patch.object(deployer, "status", side_effect=mock_status),
        ):
            await deployer.deploy(
                [cmd],
                detach=False,
                poll_interval=0.01,
                on_progress=on_progress,
            )

        assert len(progress_calls) >= 1
        # Each callback gets (elapsed_seconds, summary_string)
        assert "full=" in progress_calls[0][1]
        # The not-ready reason rides along, so a run's own log says whether
        # the engine is still loading or already answering unhealthily.
        assert "health_check_failed" in progress_calls[0][1]


# ---------- _read_tail ----------


class TestReadTail:
    @pytest.mark.anyio
    async def test_read_tail_with_content(self, tmp_path: Path):
        deployer = LocalDeployer()
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3\n  \nline5\n")

        handle = _make_handle(log_file=str(log_file))
        lines = await deployer._read_tail(handle, n=3)
        # Should skip blank lines and return last 3 non-empty
        assert len(lines) == 3
        assert lines[-1] == "line5"

    @pytest.mark.anyio
    async def test_read_tail_large_file_seeks(self, tmp_path: Path):
        """For large files, _read_tail should seek near the end, not read all."""
        deployer = LocalDeployer()
        log_file = tmp_path / "big.log"
        # Write a file large enough that the seek-based read kicks in
        # n=5 with 256 bytes/line = 1280 byte chunk. Write >1280 bytes.
        many_lines = [f"log line {i}: " + "x" * 200 for i in range(20)]
        log_file.write_text("\n".join(many_lines) + "\n")

        handle = _make_handle(log_file=str(log_file))
        lines = await deployer._read_tail(handle, n=5)
        assert len(lines) == 5
        # Should contain the last lines
        assert "log line 19" in lines[-1]

    @pytest.mark.anyio
    async def test_read_tail_no_file(self):
        deployer = LocalDeployer()
        handle = _make_handle(log_file="/nonexistent/path.log")
        lines = await deployer._read_tail(handle)
        assert lines == []

    @pytest.mark.anyio
    async def test_read_tail_no_log_path(self):
        deployer = LocalDeployer()
        handle = InferHandle(
            backend="sglang", handle_id="1", endpoint=None, metadata={}
        )
        lines = await deployer._read_tail(handle)
        assert lines == []


# ---------- status edge cases ----------


class TestStatusEdgeCases:
    @pytest.mark.anyio
    async def test_status_fallback_to_endpoint(self):
        deployer = LocalDeployer()
        handle = InferHandle(
            backend="sglang",
            handle_id="12345",
            endpoint="http://localhost:30000/v1",
            metadata={"role": "full"},
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with (
            patch("sieval.infer.deployer.pid_alive", return_value=True),
            patch("sieval.infer.deployer.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            phase, conditions = await deployer.status(handle)

        assert phase == InferPhase.RUNNING
        assert conditions["ready"].status is True
        mock_client.get.assert_called_once()
        call_url = mock_client.get.call_args[0][0]
        assert "/health" in call_url

    @pytest.mark.anyio
    async def test_status_http_non_200_returns_not_ready(self):
        deployer = LocalDeployer()
        handle = _make_handle()

        mock_response = MagicMock()
        mock_response.status_code = 503

        with (
            patch("sieval.infer.deployer.pid_alive", return_value=True),
            patch("sieval.infer.deployer.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            phase, conditions = await deployer.status(handle)
        assert phase == InferPhase.RUNNING
        assert conditions["ready"].status is False
        assert conditions["ready"].reason == "health_check_failed"

    @pytest.mark.anyio
    async def test_status_transport_error_returns_not_ready(self):
        """Other transport errors (ReadError, etc.) should be caught gracefully."""
        import httpx

        deployer = LocalDeployer()
        handle = _make_handle()

        with (
            patch("sieval.infer.deployer.pid_alive", return_value=True),
            patch("sieval.infer.deployer.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ReadError("connection reset")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            phase, conditions = await deployer.status(handle)
        assert phase == InferPhase.RUNNING
        assert conditions["ready"].status is False
        assert conditions["ready"].reason == "ReadError"


# ---------- delete edge cases ----------


class TestDeleteEdgeCases:
    @pytest.mark.anyio
    async def test_delete_sigkill_after_timeout(self):
        """If process doesn't exit after SIGTERM, SIGKILL should be sent."""
        import signal

        deployer = LocalDeployer()
        handle = _make_handle()

        with (
            patch("sieval.infer.deployer.pid_alive", return_value=True),
            patch("sieval.infer.deployer.kill_process_group") as mock_kill,
            patch("sieval.infer.deployer._GRACEFUL_SHUTDOWN_TIMEOUT", 0.05),
        ):
            await deployer.delete(handle)

        calls = mock_kill.call_args_list
        assert len(calls) >= 2
        assert calls[0][0][1] == signal.SIGTERM
        assert calls[-1][0][1] == signal.SIGKILL


# ---------- launch with env ----------


class TestLaunchWithEnv:
    @pytest.mark.anyio
    async def test_launch_with_custom_env(self):
        """BackendCommand.env should be merged with os.environ."""
        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["echo", "test"],
            backend="sglang",
            role="full",
            health_url="http://localhost:8000/health",
            env={"CUSTOM_VAR": "value"},
        )

        with (
            patch("sieval.infer.deployer.subprocess.Popen") as mock_popen,
            patch("sieval.infer.deployer._LOG_DIR", Path("/tmp")),
        ):
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            await deployer._launch_one(cmd)

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["env"]["CUSTOM_VAR"] == "value"


# ---------- deploy cleanup warning ----------


class TestDeployCleanupWarning:
    @pytest.mark.anyio
    async def test_cleanup_failure_logged_not_raised(self):
        """If cleanup fails during error handling, it should be logged not raised."""
        deployer = LocalDeployer()

        call_count = 0

        async def mock_launch(cmd):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("launch failed")
            return _make_handle(pid=str(10000 + call_count))

        async def mock_delete(handle):  # noqa: ARG001
            raise OSError("cleanup failed")

        with (
            patch.object(deployer, "_launch_one", side_effect=mock_launch),
            patch.object(deployer, "delete", side_effect=mock_delete),
            pytest.raises(RuntimeError, match="launch failed"),
        ):
            await deployer.deploy(
                [_make_command(), _make_command()],
                detach=True,
            )


# ---------- logs follow mode ----------


class TestLogsFollow:
    @pytest.mark.anyio
    async def test_logs_follow_reads_new_content(self, tmp_path: Path):
        """Follow mode should yield new lines written after initial read."""
        deployer = LocalDeployer()
        log_file = tmp_path / "follow.log"
        log_file.write_text("line1\nline2\n")

        handle = _make_handle(log_file=str(log_file))

        lines: list[str] = []
        line_count = 0

        async def collect_with_timeout():
            nonlocal line_count
            async for line in deployer.logs(handle, tail=10, follow=True):
                lines.append(line)
                line_count += 1
                if line_count >= 4:
                    break

        async def write_new_lines():
            await anyio.sleep(0.2)
            with open(log_file, "a") as f:
                f.write("line3\nline4\n")

        async with anyio.create_task_group() as tg:
            tg.start_soon(write_new_lines)
            with anyio.fail_after(5):
                await collect_with_timeout()

        assert "line1" in lines
        assert "line2" in lines
        assert "line3" in lines or "line4" in lines
