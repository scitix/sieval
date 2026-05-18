"""
Deployer: execute BackendCommands and manage inference process lifecycle.

Unifies VLLMBackend/SglangBackend create/status/delete/logs (identical
implementations) and launcher.py polling into a single LocalDeployer.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import os
import platform
import signal
import subprocess
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import anyio
import httpx
from anyio.to_thread import run_sync
from loguru import logger

from sieval.infer.backends.process import kill_process_group, pid_alive
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferCondition, InferEnv, InferHandle, InferPhase
from sieval.infer.topology.models import DeploymentCapabilities, DeploymentPlan

_HEALTH_CHECK_TIMEOUT = 2.0
_GRACEFUL_SHUTDOWN_TIMEOUT = 10.0
_LOG_DIR = Path.home() / ".sieval" / "logs"
_FAILURE_LOG_TAIL = 30

ProgressCallback = Callable[[float, str], None]


class DeployError(Exception):
    """Base exception for deployment errors."""


class DeployTimeoutError(DeployError):
    """Raised when deployment doesn't become ready within timeout."""


class LocalDeployer:
    """Local subprocess deployer.

    Replaces: VLLMBackend.create/status/delete/logs +
              SglangBackend.create/status/delete/logs +
              launcher.launch.
    """

    async def deploy(
        self,
        commands: list[BackendCommand],
        *,
        detach: bool = False,
        timeout: float = 300.0,
        poll_interval: float = 5.0,
        on_progress: ProgressCallback | None = None,
    ) -> list[InferHandle]:
        """Deploy all commands and wait for readiness.

        Args:
            commands: Backend commands to execute.
            detach: Return immediately after launching (don't wait for ready).
            timeout: Max seconds to wait for all processes to be ready.
            poll_interval: Seconds between health checks.
            on_progress: Optional callback(elapsed_seconds, status_string).

        Returns:
            List of InferHandles (one per command).
        """
        handles: list[InferHandle] = []

        try:
            for cmd in commands:
                handle = await self._launch_one(cmd)
                handles.append(handle)

            if detach:
                return handles

            # Poll all handles until ready
            await self._poll_all_until_ready(
                handles,
                timeout=timeout,
                poll_interval=poll_interval,
                on_progress=on_progress,
            )
            return handles

        except Exception:
            # Cleanup on any failure (graceful: SIGTERM → wait → SIGKILL)
            for h in handles:
                try:
                    await self.delete(h)
                except Exception as exc:
                    logger.warning("Cleanup failed for {}: {}", h.handle_id, exc)
            raise
        except BaseException:
            # BaseException minus Exception = KeyboardInterrupt, CancelledError,
            # SystemExit — skip graceful shutdown, force-kill immediately.
            for h in handles:
                try:
                    pid = int(h.handle_id)
                    kill_process_group(pid, signal.SIGKILL)
                except (ValueError, ProcessLookupError, OSError):
                    pass
            raise

    async def _launch_one(self, cmd: BackendCommand) -> InferHandle:
        """Launch a single subprocess from a BackendCommand."""
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        log_file = _LOG_DIR / f"{cmd.role}-{ts}.log"

        logger.info("Launching [{}]: {}", cmd.role, " ".join(cmd.cli_args))
        logger.info("Logging to {}", log_file)

        env = None
        if cmd.env:
            env = {**os.environ, **cmd.env}

        with open(log_file, "a") as fh:
            process = subprocess.Popen(
                cmd.cli_args,
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=fh,
                start_new_session=True,
                env=env,
                cwd=cmd.working_dir,
            )

        # Derive endpoint from health_url
        endpoint = None
        if cmd.health_url:
            parsed = urlparse(cmd.health_url)
            endpoint = urlunparse((parsed.scheme, parsed.netloc, "/v1", "", "", ""))

        return InferHandle(
            backend=cmd.backend or cmd.role,  # engine name; fallback to role
            handle_id=str(process.pid),
            endpoint=endpoint,
            metadata={
                "cmd": cmd.cli_args,
                "log_file": str(log_file),
                "role": cmd.role,
                "health_url": cmd.health_url or "",
            },
        )

    async def status(
        self, handle: InferHandle
    ) -> tuple[InferPhase, dict[str, InferCondition]]:
        """Check process status via PID + HTTP health probe.

        Returns (phase, conditions) where conditions currently contains
        only the ``ready`` key.
        """
        try:
            pid = int(handle.handle_id)
        except ValueError:
            return InferPhase.FAILED, {
                "ready": InferCondition(status=False, reason="invalid_pid")
            }

        if not pid_alive(pid):
            return InferPhase.STOPPED, {
                "ready": InferCondition(status=False, reason="process_exited")
            }

        health_url = handle.metadata.get("health_url", "")
        if not health_url:
            if handle.endpoint:
                parsed = urlparse(handle.endpoint)
                health_url = urlunparse(
                    (parsed.scheme, parsed.netloc, "/health", "", "", "")
                )
            else:
                return InferPhase.RUNNING, {
                    "ready": InferCondition(status=False, reason="no_health_url")
                }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(str(health_url), timeout=_HEALTH_CHECK_TIMEOUT)
                if resp.status_code == 200:
                    return InferPhase.RUNNING, {"ready": InferCondition(status=True)}
                return InferPhase.RUNNING, {
                    "ready": InferCondition(status=False, reason="health_check_failed")
                }
        except httpx.ConnectError:
            return InferPhase.RUNNING, {
                "ready": InferCondition(status=False, reason="connection_refused")
            }
        except httpx.TimeoutException:
            return InferPhase.RUNNING, {
                "ready": InferCondition(status=False, reason="health_check_timeout")
            }
        except httpx.TransportError as exc:
            return InferPhase.RUNNING, {
                "ready": InferCondition(status=False, reason=type(exc).__name__)
            }

    async def delete(self, handle: InferHandle) -> None:
        """Stop process gracefully (SIGTERM), force-kill if needed (SIGKILL)."""
        try:
            pid = int(handle.handle_id)
        except ValueError:
            return

        if not pid_alive(pid):
            return

        logger.info("Sending SIGTERM to process group {}", pid)
        kill_process_group(pid, signal.SIGTERM)

        deadline = anyio.current_time() + _GRACEFUL_SHUTDOWN_TIMEOUT
        while anyio.current_time() < deadline:
            if not pid_alive(pid):
                logger.info("Process {} exited after SIGTERM", pid)
                return
            await anyio.sleep(0.5)

        logger.warning("Process {} did not exit, sending SIGKILL", pid)
        kill_process_group(pid, signal.SIGKILL)

        # SIGKILL is async — wait briefly for the kernel to reap the process
        deadline = anyio.current_time() + 2.0
        while anyio.current_time() < deadline:
            if not pid_alive(pid):
                logger.info("Process {} exited after SIGKILL", pid)
                return
            await anyio.sleep(0.2)

        # Final check — pid_alive already attempts waitpid reap internally
        if pid_alive(pid):
            logger.error("Process {} still alive after SIGKILL + timeout", pid)

    async def logs(
        self,
        handle: InferHandle,
        *,
        tail: int = 50,
        follow: bool = False,
    ) -> AsyncIterator[str]:
        """Stream log lines from the process log file."""
        log_path_str = handle.metadata.get("log_file")
        if not log_path_str:
            logger.warning("No log_file in handle metadata for {}", handle.handle_id)
            return

        path = Path(str(log_path_str))
        if not path.exists():
            logger.warning("Log file {} does not exist", path)
            return

        # Read only the tail portion instead of the entire file.
        # 256 bytes/line is a generous estimate for log lines.
        chunk_size = tail * 256
        async with await anyio.open_file(path, "rb") as f:
            file_size = await f.seek(0, 2)  # seek to end
            read_start = max(0, file_size - chunk_size)
            await f.seek(read_start)
            raw = await f.read()

        text = raw.decode(errors="replace")
        lines = text.splitlines()
        # If we didn't read from the start, the first line may be partial — drop it.
        if read_start > 0 and lines:
            lines = lines[1:]
        for line in lines[-tail:]:
            yield line

        if not follow:
            return

        offset = file_size
        while True:
            await anyio.sleep(1.0)
            try:
                async with await anyio.open_file(path, "rb") as f:
                    await f.seek(offset)
                    new_raw = await f.read()
            except FileNotFoundError:
                return
            if new_raw:
                offset += len(new_raw)
                for line in new_raw.decode(errors="replace").splitlines():
                    yield line

    async def _poll_all_until_ready(
        self,
        handles: list[InferHandle],
        *,
        timeout: float,
        poll_interval: float,
        on_progress: ProgressCallback | None,
    ) -> None:
        """Poll all handles until all ready, or any FAILED/timeout."""
        start = anyio.current_time()
        deadline = start + timeout

        while True:
            all_ready = True
            summary_parts: list[str] = []

            for handle in handles:
                phase, conditions = await self.status(handle)
                role = handle.metadata.get("role", "?")
                ready = conditions.get("ready")
                ready_str = "ready" if ready and ready.status else phase.value
                summary_parts.append(f"{role}={ready_str}")

                if phase in (InferPhase.FAILED, InferPhase.STOPPED):
                    tail = await self._read_tail(handle)
                    msg = f"Process {handle.handle_id} ({role}) {phase.value}"
                    if tail:
                        msg += f"\n--- last {len(tail)} lines ---\n" + "\n".join(tail)
                    raise DeployError(msg)

                if not (ready and ready.status):
                    all_ready = False

            elapsed = anyio.current_time() - start
            if on_progress:
                on_progress(elapsed, ", ".join(summary_parts))

            if all_ready:
                return

            if anyio.current_time() >= deadline:
                summary = ", ".join(summary_parts)
                msg = f"Not all processes ready within {timeout}s ({summary})"
                raise DeployTimeoutError(msg)

            await anyio.sleep(poll_interval)

    async def _read_tail(
        self, handle: InferHandle, n: int = _FAILURE_LOG_TAIL
    ) -> list[str]:
        """Read the last N non-empty lines from the handle's log file."""
        log_path_str = handle.metadata.get("log_file")
        if not log_path_str:
            return []
        path = Path(str(log_path_str))
        if not path.exists():
            return []
        try:
            # Estimate: 256 bytes/line is generous for log lines
            chunk_size = n * 256
            async with await anyio.open_file(path, "rb") as f:
                file_size = await f.seek(0, 2)
                read_start = max(0, file_size - chunk_size)
                await f.seek(read_start)
                raw = await f.read()
            lines = [
                line
                for line in raw.decode(errors="replace").splitlines()
                if line.strip()
            ]
            # Drop first line if partial (didn't read from start)
            if read_start > 0 and lines:
                lines = lines[1:]
            return lines[-n:]
        except Exception:
            return []

    def build_capabilities(
        self,
        plan: DeploymentPlan,
        handles: list[InferHandle],
    ) -> DeploymentCapabilities:
        """Construct DeploymentCapabilities after all handles are ready."""
        endpoints: dict[str, str] = {}
        api_base = ""

        for handle in handles:
            role = str(handle.metadata.get("role", "full"))
            if handle.endpoint:
                endpoints[role] = handle.endpoint
                if not api_base:
                    api_base = handle.endpoint

        return DeploymentCapabilities(
            api_base=api_base,
            is_disaggregated=plan.is_disaggregated,
            roles=tuple(a.role for a in plan.assignments),
            total_gpus=plan.total_gpus,
            endpoints=endpoints,
        )


# ---------------------------------------------------------------------------
# Environment collection (migrated from backends/backend.py)
# ---------------------------------------------------------------------------


def _detect_framework(backend: str) -> str:
    """Best-effort framework version detection for the given backend."""
    if backend == "sglang":
        try:
            import sglang  # type: ignore[unresolved-import]

            version = getattr(sglang, "__version__", "")
            if not version:
                from sglang.version import (  # type: ignore[unresolved-import]
                    __version__ as v,
                )

                version = v
            return f"sglang=={version}" if version else "sglang"
        except Exception:  # noqa: BLE001
            return "sglang"
    if backend == "vllm":
        try:
            from vllm.collect_env import get_env_info  # type: ignore[unresolved-import]

            info = get_env_info()
            version = getattr(info, "vllm_version", "")
            return f"vllm=={version}" if version else "vllm"
        except Exception:  # noqa: BLE001
            return "vllm"
    return "unknown"


def _collect_basic_env_sync(backend: str = "") -> InferEnv:
    """Synchronous helper for nvidia-smi probing.

    Isolated so it can be offloaded to a worker thread via
    ``anyio.to_thread.run_sync``, keeping the event loop unblocked.
    """
    framework = _detect_framework(backend) if backend else "unknown"
    cuda_version = ""
    driver_version = ""
    gpu_model = ""
    gpu_count = 0

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            gpu_count = len(lines)
            first_line = lines[0].split(", ")
            if len(first_line) >= 2:
                gpu_model = first_line[0].strip()
                driver_version = first_line[1].strip()

        # Extract CUDA version from stderr of the same call
        # nvidia-smi --query-gpu prints the header table to stderr
        # Fall back to a plain nvidia-smi call only if needed
        header_output = result.stderr if result.returncode == 0 else ""
        if "CUDA Version" not in header_output:
            header_result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if header_result.returncode == 0:
                header_output = header_result.stdout

        for line in header_output.split("\n"):
            if "CUDA Version" in line:
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    cuda_version = parts[1].strip().rstrip("|").strip()
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("nvidia-smi not available — returning partial env info")

    return InferEnv(
        framework=framework,
        cuda_version=cuda_version,
        driver_version=driver_version,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        python_version=platform.python_version(),
    )


async def collect_basic_env(backend: str = "") -> InferEnv:
    """Collect basic environment info, optionally detecting framework version.

    Args:
        backend: Backend name (e.g. ``"sglang"``, ``"vllm"``).  When provided
            the function attempts to import the corresponding package and read
            its version.

    Runs nvidia-smi in a worker thread to avoid blocking the event loop.
    """
    return await run_sync(lambda: _collect_basic_env_sync(backend))
