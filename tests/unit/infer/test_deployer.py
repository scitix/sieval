"""
Unit tests for LocalDeployer.

Tests focus on unit-level behavior — subprocess launching is mocked to
avoid spawning real processes.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from sieval.core.models import ServingFacts
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferCondition, InferHandle, InferPhase
from sieval.infer.deployer import (
    DEFAULT_READY_TIMEOUT,
    ENGINE_HEALTH_INTERVAL,
    ENGINE_UNHEALTHY_AFTER,
    DeployError,
    DeployTimeoutError,
    EngineExit,
    LocalDeployer,
    _logged_progress,
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


# ---------- DEFAULT_READY_TIMEOUT ----------


class TestDefaultReadyTimeout:
    def test_leaves_room_for_a_cold_start(self):
        """264s cold was measured for a 30B MoE, so 300s was not a margin."""
        assert DEFAULT_READY_TIMEOUT >= 600.0


# ---------- _launch_one ----------


class TestLaunchOne:
    @pytest.mark.anyio
    async def test_missing_executable_names_the_binary(self):
        """Popen's bare FileNotFoundError names neither the role nor the cmd."""
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
    async def test_log_file_per_role_is_honored(self, tmp_path: Path):
        """A caller-chosen log path is created and recorded in the handle."""
        log_file = tmp_path / "results" / "logs" / "m" / "full.log"
        cmd = BackendCommand(cli_args=["sh", "-c", "exit 0"], role="full")
        with patch("sieval.infer.deployer.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=4242)
            handle = await LocalDeployer()._launch_one(cmd, log_file=log_file)
        assert log_file.exists()
        assert handle.metadata["log_file"] == str(log_file)

    @pytest.mark.anyio
    async def test_default_log_names_differ_within_one_second(self, tmp_path: Path):
        """Two launches in the same second no longer share one log file."""
        cmd = BackendCommand(cli_args=["sh", "-c", "exit 0"], role="full")
        with (
            patch("sieval.infer.deployer.subprocess.Popen") as mock_popen,
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", tmp_path),
        ):
            mock_popen.return_value = MagicMock(pid=4242)
            first = await LocalDeployer()._launch_one(cmd)
            second = await LocalDeployer()._launch_one(cmd)
        assert first.metadata["log_file"] != second.metadata["log_file"]
        assert re.fullmatch(
            r"full-\d{8}T\d{6}Z-[0-9a-f]{8}\.log",
            Path(str(first.metadata["log_file"])).name,
        )

    @pytest.mark.anyio
    async def test_rejected_launch_creates_no_log_dir_and_names_no_log(
        self, tmp_path: Path
    ):
        """A rejected launch must create no log dir and name no log path."""
        log_dir = tmp_path / "logs"
        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=["sglang-not-installed", "serve"],
            backend="sglang",
            role="full",
            health_url="http://localhost:30000/health",
        )

        with (
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", log_dir),
            patch("sieval.infer.deployer.logger") as mock_logger,
            pytest.raises(DeployError),
        ):
            await deployer._launch_one(cmd)

        assert not log_dir.exists()
        logged = " ".join(str(call) for call in mock_logger.info.call_args_list)
        assert "Logging to" not in logged

    @pytest.mark.anyio
    async def test_directory_is_not_reported_as_absent(self, tmp_path: Path):
        """which() rejects a directory under F_OK, so it must not read absent."""
        engine_dir = tmp_path / "sglang"
        engine_dir.mkdir()

        deployer = LocalDeployer()
        cmd = BackendCommand(
            cli_args=[str(engine_dir), "serve"],
            backend="sglang",
            role="full",
            health_url="http://localhost:30000/health",
        )

        with pytest.raises(DeployError) as excinfo:
            await deployer._launch_one(cmd)

        message = str(excinfo.value)
        assert "is a directory" in message
        assert "does not exist" not in message

    @pytest.mark.anyio
    async def test_engine_on_a_cmd_env_path_is_not_rejected(self, tmp_path: Path):
        """Popen resolves argv[0] against the child's PATH, so cmd.env wins."""
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
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", Path("/tmp")),
        ):
            mock_popen.return_value = MagicMock(pid=4242)
            handle = await deployer._launch_one(cmd)

        assert handle.handle_id == "4242"

    @pytest.mark.anyio
    async def test_present_but_non_executable_is_not_called_missing(
        self, tmp_path: Path
    ):
        """Popen raises PermissionError here, so reporting it absent misleads."""
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
        """Popen(cwd=...) resolves a relative argv[0] there, not against ours."""
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
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", Path("/tmp")),
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
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", Path("/tmp")),
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
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", Path("/tmp")),
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
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", Path("/tmp")),
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
    async def test_logs_collapses_progress_bar_redraws(self, tmp_path: Path):
        """`sieval infer logs` reads the same log, so it must split the same."""
        log_file = tmp_path / "engine.log"
        log_file.write_text(
            "real line 1\nreal line 2\n"
            + "".join(f"\rLoading shards: {pct}%" for pct in range(101))
        )

        deployer = LocalDeployer()
        handle = _make_handle(log_file=str(log_file))
        lines = [line async for line in deployer.logs(handle, tail=3)]

        assert "real line 1" in lines
        assert sum("Loading shards" in line for line in lines) == 1

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
        # Write a file large enough that chunk_size < file_size (triggers seek).
        many_lines = [f"log line {i}: " + "x" * 200 for i in range(10)]
        log_file.write_text("\n".join(many_lines) + "\n")

        handle = _make_handle(log_file=str(log_file))
        lines = []
        # The read window is sized in bytes, so it must be narrowed here or the
        # whole file fits and neither the seek nor the partial-drop path runs.
        with patch("sieval.infer.deployer._LOG_TAIL_WINDOW_BYTES", 768):
            async for line in deployer.logs(handle, tail=3):
                lines.append(line)

        assert len(lines) == 3
        assert "log line 9" in lines[-1]
        assert not any("log line 0" in line for line in lines)


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

        async def mock_launch(cmd, *, log_file=None):  # noqa: ARG001
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
        """The process is killed on the way out; unquoted, its log is lost."""
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

        # A crash quotes the same block as a timeout: tail, log path, command.
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
        # The not-ready reason rides along, not just the phase.
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
        # Write a file large enough that the seek-based read kicks in.
        many_lines = [f"log line {i}: " + "x" * 200 for i in range(20)]
        log_file.write_text("\n".join(many_lines) + "\n")

        handle = _make_handle(log_file=str(log_file))
        # The read window is sized in bytes, so it must be narrowed here or the
        # whole file fits and the seek path never runs.
        with patch("sieval.infer.deployer._LOG_TAIL_WINDOW_BYTES", 1280):
            lines = await deployer._read_tail(handle, n=5)
        assert len(lines) == 5
        # Should contain the last lines
        assert "log line 19" in lines[-1]
        assert not any("log line 0" in line for line in lines)

    @pytest.mark.anyio
    async def test_read_tail_keeps_the_error_under_progress_bar_redraws(
        self, tmp_path: Path
    ):
        """A redrawn progress bar must not push the traceback out of the tail.

        Counting each carriage-return frame as a line spends the whole window
        on one bar and buries the evidence the tail exists to carry.
        """
        log_file = tmp_path / "engine.log"
        log_file.write_text(
            "[08:45:50] server args: dp_size=4\n"
            "[08:46:02] Scheduler hit an exception: Traceback "
            "(most recent call last):\n"
            "ImportError: /usr/lib/sgl_kernel.so: undefined symbol\n"
            + "".join(
                f"\rLoading safetensors checkpoint shards: {pct:3d}% "
                f"Completed | {pct}/100 [00:{pct:02d}<00:00, 1.9it/s]"
                for pct in range(101)
            )
        )

        deployer = LocalDeployer()
        handle = _make_handle(log_file=str(log_file))
        lines = await deployer._read_tail(handle)

        assert any("ImportError" in line for line in lines)
        # The bar collapses to its final frame rather than 101 of them.
        assert sum("Loading safetensors" in line for line in lines) == 1

    @pytest.mark.anyio
    async def test_read_tail_collapses_a_redrawn_line_to_its_last_frame(
        self, tmp_path: Path
    ):
        """One carriage-return-redrawn line is one line: its final frame."""
        log_file = tmp_path / "bar.log"
        log_file.write_text("before\nstep 1\rstep 2\rstep 3\nafter\n")

        deployer = LocalDeployer()
        handle = _make_handle(log_file=str(log_file))
        lines = await deployer._read_tail(handle)

        assert lines == ["before", "step 3", "after"]

    @pytest.mark.anyio
    async def test_read_tail_handles_crlf_line_endings(self, tmp_path: Path):
        """The last frame of ``"line1\\r"`` is ``""`` — CRLF must not vanish."""
        log_file = tmp_path / "crlf.log"
        log_file.write_bytes(b"line1\r\nline2\r\nline3\r\n")

        deployer = LocalDeployer()
        handle = _make_handle(log_file=str(log_file))
        lines = await deployer._read_tail(handle)

        assert lines == ["line1", "line2", "line3"]

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
            patch("sieval.infer.deployer.ENGINE_LOG_DIR", Path("/tmp")),
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

        async def mock_launch(cmd, *, log_file=None):  # noqa: ARG001
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


# ---------- watch ----------


class TestWatch:
    """`watch` returns once a launched engine is lost, and only then."""

    @staticmethod
    def _launched(
        deployer: LocalDeployer, poll: list[int | None], pid: str = "12345"
    ) -> InferHandle:
        process = MagicMock()
        process.pid = int(pid)
        process.poll.side_effect = poll + [poll[-1]] * 1000
        deployer._processes[pid] = process
        return _make_handle(pid=pid)

    @pytest.mark.anyio
    async def test_an_exit_returns_its_code_and_log(self, tmp_path):
        log = tmp_path / "full.log"
        log.write_text("loading\ntorch.OutOfMemoryError: CUDA out of memory\n")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None, None, -9])
        handle.metadata["log_file"] = str(log)

        with (
            patch("sieval.infer.deployer.ENGINE_POLL_INTERVAL", 0.001),
            patch("sieval.infer.deployer.kill_orphaned_group") as orphans,
        ):
            exit = await deployer.watch([handle])

        assert (exit.role, exit.pid, exit.exit_code) == ("full", "12345", -9)
        assert "CUDA out of memory" in exit.log
        assert exit.summary() == "process 12345 (full) was killed by SIGKILL"
        # Workers left behind would hold the GPUs.
        orphans.assert_called_once_with(12345)

    @pytest.mark.anyio
    async def test_a_failing_health_check_counts_only_once_it_lasts(self):
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        probes = iter(
            [
                InferCondition(status=False, reason="health_check_timeout"),
                InferCondition(status=True),  # recovered: the clock restarts
                *[InferCondition(status=False, reason="health_check_timeout")] * 99,
            ]
        )
        seen: list[float] = []

        async def probe(_handle):
            seen.append(anyio.current_time())
            return next(probes)

        with (
            patch("sieval.infer.deployer.ENGINE_POLL_INTERVAL", 0.001),
            patch("sieval.infer.deployer.ENGINE_HEALTH_INTERVAL", 0.01),
            patch("sieval.infer.deployer.ENGINE_UNHEALTHY_AFTER", 0.05),
            patch.object(deployer, "_probe_liveness", side_effect=probe),
        ):
            exit = await deployer.watch([handle])

        assert exit.exit_code is None
        assert exit.reason == "health_check_timeout"
        assert exit.unhealthy_for is not None and exit.unhealthy_for >= 0.05
        # Counted from the third probe, after the recovery.
        assert len(seen) >= 3
        assert "failed its health check" in exit.summary()

    @staticmethod
    async def _watch_with(
        deployer: LocalDeployer,
        handle: InferHandle,
        probe,
        *,
        within: float = 5,
    ) -> EngineExit | None:
        """Watch *handle* with *probe* as its liveness probe, 10 ms windows and
        a 50 ms loss threshold; ``None`` when it is still alive after *within*."""
        with (
            patch("sieval.infer.deployer.ENGINE_POLL_INTERVAL", 0.001),
            patch("sieval.infer.deployer.ENGINE_HEALTH_INTERVAL", 0.01),
            patch("sieval.infer.deployer.ENGINE_UNHEALTHY_AFTER", 0.05),
            patch.object(deployer, "_probe_liveness", side_effect=probe),
            anyio.move_on_after(within),
        ):
            return await deployer.watch([handle])
        return None

    @pytest.mark.anyio
    async def test_a_busy_engine_starving_its_health_check_stays_alive(
        self, tmp_path: Path
    ):
        # The aries loss: /health timed out for minutes while SGLang decoded.
        log = tmp_path / "full.log"
        log.write_text("")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle.metadata["log_file"] = str(log)

        async def probe(_handle):
            with log.open("a") as fh:
                fh.write(
                    "[2026-11-05 03:12:09 DP0 TP0] Decode batch, #running-req: 24, "
                    "#token: 41210, token usage: 0.11, cuda graph: True, "
                    "gen throughput (token/s): 2401.37, #queue-req: 0\n"
                )
            return InferCondition(status=False, reason="health_check_timeout")

        assert await self._watch_with(deployer, handle, probe, within=0.3) is None

    @pytest.mark.anyio
    async def test_prefill_alone_counts_as_progress(self, tmp_path: Path):
        log = tmp_path / "full.log"
        log.write_text("")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle.metadata["log_file"] = str(log)

        async def probe(_handle):
            with log.open("a") as fh:
                fh.write(
                    "Prefill batch, #new-seq: 1, #new-token: 8192, "
                    "#cached-token: 0, input throughput (token/s): 31002.11\n"
                )
            return InferCondition(status=False, reason="health_check_timeout")

        assert await self._watch_with(deployer, handle, probe, within=0.3) is None

    @pytest.mark.anyio
    async def test_a_passing_ping_keeps_a_silent_engine_alive(self, tmp_path: Path):
        # Idle: SGLang logs no batch, so only the probe speaks for it.
        log = tmp_path / "full.log"
        log.write_text("")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle.metadata["log_file"] = str(log)

        async def probe(_handle):
            return InferCondition(status=True)

        assert await self._watch_with(deployer, handle, probe, within=0.3) is None

    @pytest.mark.anyio
    async def test_no_probe_and_no_progress_is_lost(self, tmp_path: Path):
        log = tmp_path / "full.log"
        # Progress before the watch started does not count.
        log.write_text("Decode batch, #running-req: 3, gen throughput: 90.0\n")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle.metadata["log_file"] = str(log)

        async def probe(_handle):
            with log.open("a") as fh:
                fh.write("[2026-11-05 03:12:09] INFO: 127.0.0.1 GET /metrics 200\n")
            return InferCondition(status=False, reason="health_check_timeout")

        reads: list[tuple[bool, int]] = []

        async def recorded(path: Path, offset: int, backend: str):
            read = await _logged_progress(path, offset, backend)
            reads.append(read)
            return read

        with patch("sieval.infer.deployer._logged_progress", side_effect=recorded):
            exit = await self._watch_with(deployer, handle, probe)

        assert exit is not None and exit.exit_code is None
        # Not even the first window saw the line logged before the watch.
        assert reads and not any(progressed for progressed, _ in reads)
        assert exit.reason == "health_check_timeout"
        assert exit.unhealthy_for is not None and exit.unhealthy_for >= 0.05
        assert "logged no progress" in exit.summary()

    @pytest.mark.anyio
    async def test_progress_restarts_the_clock(self, tmp_path: Path):
        log = tmp_path / "full.log"
        log.write_text("")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle.metadata["log_file"] = str(log)
        calls: list[float] = []

        async def probe(_handle):
            calls.append(anyio.current_time())
            if len(calls) == 3:
                with log.open("a") as fh:
                    fh.write("Decode batch, #running-req: 1\n")
            return InferCondition(status=False, reason="health_check_timeout")

        exit = await self._watch_with(deployer, handle, probe)

        assert exit is not None
        # Counted from the probe after the one that saw progress.
        assert exit.unhealthy_for is not None
        assert anyio.current_time() - calls[2] >= 0.05

    @pytest.mark.anyio
    async def test_nothing_for_180s_is_lost(self, tmp_path: Path):
        # The defaults, scaled 1000x down.
        assert (ENGINE_HEALTH_INTERVAL, ENGINE_UNHEALTHY_AFTER) == (10.0, 180.0)
        log = tmp_path / "full.log"
        log.write_text("")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle.metadata["log_file"] = str(log)
        probes: list[float] = []

        async def probe(_handle):
            probes.append(anyio.current_time())
            return InferCondition(status=False, reason="health_check_timeout")

        with (
            patch("sieval.infer.deployer.ENGINE_POLL_INTERVAL", 0.001),
            patch("sieval.infer.deployer.ENGINE_HEALTH_INTERVAL", 0.01),
            patch("sieval.infer.deployer.ENGINE_UNHEALTHY_AFTER", 0.18),
            patch.object(deployer, "_probe_liveness", side_effect=probe),
            anyio.fail_after(10),
        ):
            exit = await deployer.watch([handle])

        assert exit.exit_code is None
        assert exit.unhealthy_for is not None and exit.unhealthy_for >= 0.18
        # Counted from the first failed probe, not from the watch's start.
        assert anyio.current_time() - probes[0] >= 0.18

    @pytest.mark.anyio
    async def test_vllm_idle_throughput_lines_are_not_progress(self, tmp_path: Path):
        log = tmp_path / "full.log"
        log.write_text("")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle = InferHandle(
            backend="vllm",
            handle_id=handle.handle_id,
            endpoint=handle.endpoint,
            metadata={**handle.metadata, "log_file": str(log)},
        )

        async def probe(_handle):
            with log.open("a") as fh:
                fh.write(
                    "INFO 11-05 03:12:09 [loggers.py:257] Engine 000: "
                    "Avg prompt throughput: 0.0 tokens/s, "
                    "Avg generation throughput: 0.0 tokens/s, "
                    "Running: 0 reqs, Waiting: 0 reqs\n"
                )
            return InferCondition(status=False, reason="health_check_failed")

        exit = await self._watch_with(deployer, handle, probe)
        assert exit is not None and exit.exit_code is None

    @pytest.mark.anyio
    async def test_vllm_throughput_is_progress(self, tmp_path: Path):
        log = tmp_path / "full.log"
        log.write_text("")
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        handle = InferHandle(
            backend="vllm",
            handle_id=handle.handle_id,
            endpoint=handle.endpoint,
            metadata={**handle.metadata, "log_file": str(log)},
        )

        async def probe(_handle):
            with log.open("a") as fh:
                fh.write(
                    "Avg prompt throughput: 0.0 tokens/s, "
                    "Avg generation throughput: 1843.2 tokens/s, Running: 31 reqs\n"
                )
            return InferCondition(status=False, reason="health_check_timeout")

        assert await self._watch_with(deployer, handle, probe, within=0.3) is None

    @pytest.mark.anyio
    async def test_a_rotated_log_is_read_from_its_start(self, tmp_path: Path):
        log = tmp_path / "full.log"
        log.write_text("x" * 4096)
        progressed, offset = await _logged_progress(log, 4096, "sglang")
        assert (progressed, offset) == (False, 4096)
        log.write_text("Decode batch, #running-req: 2\n")
        progressed, _ = await _logged_progress(log, offset, "sglang")
        assert progressed

    @pytest.mark.anyio
    async def test_a_missing_log_is_no_progress(self, tmp_path: Path):
        assert await _logged_progress(tmp_path / "absent.log", 7, "sglang") == (
            False,
            7,
        )

    @pytest.mark.anyio
    async def test_only_the_last_window_of_new_bytes_is_read(self, tmp_path: Path):
        log = tmp_path / "full.log"
        line = "Decode batch, #running-req: 2\n"
        with patch("sieval.infer.deployer._LOG_TAIL_WINDOW_BYTES", 1024):
            # Progress more than a window before the end is not read.
            log.write_text(line + "x" * 1024)
            size = log.stat().st_size
            assert await _logged_progress(log, 0, "sglang") == (False, size)
            # Inside the window it is.
            log.write_text("x" * 1023 + "\n" + line)
            size = log.stat().st_size
            assert await _logged_progress(log, 0, "sglang") == (True, size)

    @pytest.mark.anyio
    async def test_an_engine_without_a_progress_signal_shows_none(self, tmp_path: Path):
        log = tmp_path / "full.log"
        log.write_text("Decode batch, #running-req: 2\n")
        assert await _logged_progress(log, 0, "full") == (False, 0)

    @pytest.mark.anyio
    async def test_liveness_pings_sglang_and_falls_back_to_health(self):
        deployer = LocalDeployer()
        handle = _make_handle(health_url="http://10.0.0.5:30000/health")
        urls: list[str] = []

        async def get(url: str) -> InferCondition:
            urls.append(url)
            return InferCondition(status=url.endswith("/ping"))

        with patch.object(deployer, "_probe", side_effect=get):
            alive = await deployer._probe_liveness(handle)
        assert alive is not None and alive.status

        assert urls == ["http://10.0.0.5:30000/ping"]

        urls.clear()

        async def no_ping(url: str) -> InferCondition:
            urls.append(url)
            ok = url.endswith("/health")
            return InferCondition(status=ok, reason="" if ok else "health_check_failed")

        with patch.object(deployer, "_probe", side_effect=no_ping):
            alive = await deployer._probe_liveness(handle)
        assert alive is not None and alive.status
        # An SGLang without /ping is probed as before.
        assert urls == [
            "http://10.0.0.5:30000/ping",
            "http://10.0.0.5:30000/health",
        ]

    @pytest.mark.anyio
    async def test_liveness_of_vllm_is_its_health_check(self):
        deployer = LocalDeployer()
        handle = _make_handle(backend="vllm")
        urls: list[str] = []

        async def get(url: str) -> InferCondition:
            urls.append(url)
            return InferCondition(status=True)

        with patch.object(deployer, "_probe", side_effect=get):
            alive = await deployer._probe_liveness(handle)
        assert alive is not None and alive.status
        assert urls == ["http://localhost:8000/health"]

    @pytest.mark.anyio
    async def test_liveness_without_a_health_url_is_none(self):
        handle = _make_handle(health_url="", endpoint="")
        assert await LocalDeployer()._probe_liveness(handle) is None

    @pytest.mark.anyio
    async def test_a_hanging_probe_does_not_delay_an_exit(self):
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None] * 20 + [137])

        async def probe(_handle):
            await anyio.sleep_forever()

        with (
            patch("sieval.infer.deployer.ENGINE_POLL_INTERVAL", 0.001),
            patch("sieval.infer.deployer.ENGINE_HEALTH_INTERVAL", 0.001),
            patch("sieval.infer.deployer.kill_orphaned_group"),
            patch.object(deployer, "_probe_liveness", side_effect=probe),
            anyio.fail_after(5),
        ):
            engine_exit = await deployer.watch([handle])

        assert engine_exit.exit_code == 137

    @pytest.mark.anyio
    async def test_an_engine_it_did_not_launch_is_never_lost(self):
        deployer = LocalDeployer()
        with anyio.move_on_after(0.05) as scope:
            await deployer.watch([_make_handle()])
        assert scope.cancelled_caught

    @pytest.mark.anyio
    async def test_delete_stops_watching(self):
        deployer = LocalDeployer()
        handle = self._launched(deployer, [None])
        with patch("sieval.infer.deployer.pid_alive", return_value=False):
            await deployer.delete(handle)
        assert deployer._processes == {}

    def test_summary_of_an_exit_code(self):
        exit = EngineExit(role="full", pid="7", exit_code=137, log="")
        assert exit.summary() == "process 7 (full) exited with code 137"
        assert "signal 99" in EngineExit("full", "7", -99, "").summary()

    @pytest.mark.anyio
    async def test_launch_holds_the_process(self, tmp_path):
        deployer = LocalDeployer()
        cmd = BackendCommand(cli_args=["true"], backend="sglang")
        handle = await deployer._launch_one(cmd, log_file=tmp_path / "x.log")
        process = deployer._processes[handle.handle_id]
        assert process.wait(timeout=10) == 0


class TestKvFacts:
    """What a ready SGLang engine says it holds."""

    @staticmethod
    async def _facts(payload: object, status: int = 200, backend: str = "sglang"):
        from unittest.mock import patch as _patch

        import httpx as _httpx

        from sieval.infer.deployer import kv_facts

        seen: list[str] = []

        def handler(request: _httpx.Request) -> _httpx.Response:
            seen.append(str(request.url))
            return _httpx.Response(status, json=payload)

        real = _httpx.AsyncClient

        def client(*_args, **_kwargs):
            return real(transport=_httpx.MockTransport(handler))

        with _patch("sieval.infer.deployer.httpx.AsyncClient", client):
            facts = await kv_facts(backend, "http://127.0.0.1:30000/v1")
        return facts, seen

    @pytest.mark.anyio
    async def test_reads_the_kv_pool_context_and_replicas(self) -> None:
        facts, seen = await self._facts(
            {"max_total_num_tokens": 2266113, "context_length": 32768, "dp_size": 2}
        )
        assert seen == ["http://127.0.0.1:30000/get_server_info"]
        assert (facts.kv_capacity_tokens, facts.context_tokens, facts.replicas) == (
            2266113,
            32768,
            2,
        )

    @pytest.mark.anyio
    async def test_dp_attention_ranks_each_hold_a_pool(self) -> None:
        # SGLang runs one scheduler, and sizes one pool, per DP rank either way.
        facts, _ = await self._facts(
            {
                "max_total_num_tokens": 100,
                "context_length": 10,
                "dp_size": 4,
                "enable_dp_attention": True,
            }
        )
        assert (facts.kv_capacity_tokens, facts.replicas) == (100, 4)

    @pytest.mark.anyio
    async def test_an_unpinned_window_is_the_longest_input(self) -> None:
        facts, _ = await self._facts(
            {
                "max_total_num_tokens": 100,
                "context_length": None,
                "max_req_input_len": 40955,
            }
        )
        assert (facts.kv_capacity_tokens, facts.context_tokens) == (100, 40955)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("payload", "status"),
        [
            ({}, 200),
            ({"max_total_num_tokens": 5}, 200),  # no window: no admission
            ({"context_length": 5}, 200),
            ({"max_total_num_tokens": 5, "context_length": 5}, 500),
            ([], 200),
        ],
    )
    async def test_no_answer_gives_no_facts(self, payload: object, status: int) -> None:
        facts, _ = await self._facts(payload, status)
        assert facts == ServingFacts()

    @pytest.mark.anyio
    async def test_other_backends_are_not_asked(self) -> None:
        facts, seen = await self._facts({}, backend="vllm")
        assert seen == [] and facts.kv_capacity_tokens is None


class TestSglangLoadProbe:
    @staticmethod
    async def _probe(routes: dict[str, object]):
        from unittest.mock import patch as _patch

        import httpx as _httpx

        from sieval.infer.deployer import SglangLoadProbe

        seen: list[str] = []

        def handler(request: _httpx.Request) -> _httpx.Response:
            seen.append(str(request.url))
            body = routes.get(request.url.path)
            if body is None:
                return _httpx.Response(404)
            return _httpx.Response(200, json=body)

        real = _httpx.AsyncClient

        def client(*_args, **kwargs):
            return real(transport=_httpx.MockTransport(handler), **kwargs)

        with _patch("sieval.infer.deployer.httpx.AsyncClient", client):
            probe = SglangLoadProbe("http://127.0.0.1:30000/v1")
            try:
                return await probe(), seen
            finally:
                await probe.aclose()

    @pytest.mark.anyio
    async def test_sums_waiting_and_takes_the_fullest_rank(self) -> None:
        load, seen = await self._probe(
            {
                "/v1/loads": {
                    "loads": [
                        {"num_waiting_reqs": 2, "token_usage": 0.4},
                        {"num_waiting_reqs": 1, "token_usage": 0.97},
                    ]
                }
            }
        )
        assert load is not None
        assert (load.waiting, load.token_usage) == (3, 0.97)
        assert seen == ["http://127.0.0.1:30000/v1/loads?include=core"]

    @pytest.mark.anyio
    async def test_the_deprecated_get_load_is_not_read(self) -> None:
        # Its num_tokens is used + pending prefill, not the pool: a 1%-used
        # pool would read as 100% full.
        load, seen = await self._probe(
            {
                "/get_load": [
                    {"num_waiting_reqs": 0, "num_tokens": 1000, "num_pending_tokens": 0}
                ]
            }
        )
        assert load is None
        assert [url.split("?")[0] for url in seen] == [
            "http://127.0.0.1:30000/v1/loads"
        ]

    @pytest.mark.parametrize(
        "body",
        [{}, {"loads": []}, {"loads": ["x"]}, ["not", "a", "mapping"]],
        ids=["no-loads", "empty", "no-ranks", "list"],
    )
    @pytest.mark.anyio
    async def test_an_unusable_answer_is_none(self, body: object) -> None:
        load, _ = await self._probe({"/v1/loads": body})
        assert load is None

    @pytest.mark.anyio
    async def test_no_answer_is_none(self) -> None:
        load, _ = await self._probe({})
        assert load is None
