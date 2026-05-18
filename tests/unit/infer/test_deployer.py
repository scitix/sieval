"""
Unit tests for LocalDeployer.

Tests focus on unit-level behavior — subprocess launching is mocked to
avoid spawning real processes.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

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
) -> InferHandle:
    return InferHandle(
        backend=role,
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


# ---------- build_capabilities ----------


class TestBuildCapabilities:
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

        caps = deployer.build_capabilities(plan, handles)
        assert caps.api_base == "http://localhost:30000/v1"
        assert caps.is_disaggregated is False
        assert caps.roles == ("full",)
        assert caps.total_gpus == 8
        assert caps.endpoints == {"full": "http://localhost:30000/v1"}

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

        caps = deployer.build_capabilities(plan, handles)
        assert caps.is_disaggregated is True
        assert caps.roles == ("prefill", "decode")
        assert caps.total_gpus == 8
        assert len(caps.endpoints) == 2

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

        caps = deployer.build_capabilities(plan, handles)
        assert caps.api_base == ""
        assert caps.endpoints == {}


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
            pytest.raises(DeployError, match="failed"),
        ):
            await deployer.deploy([cmd], detach=False, poll_interval=0.01)

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
