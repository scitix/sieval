"""
Inference service lifecycle: handle I/O, deployment, and status.

Manages the on-disk handle files, deployer interactions, and
status probing for inference services.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import anyio
import typer
from loguru import logger

from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferCondition, InferEnv, InferHandle, InferPhase
from sieval.infer.deployer import LocalDeployer, collect_basic_env

HANDLE_DIR = Path.home() / ".sieval" / "handles"
default_deployer = LocalDeployer()
# Keys stored in handle JSON but not part of InferHandle fields.
HANDLE_EXTRA_KEYS = frozenset(("phase", "conditions", "pid", "env", "status"))


def display_status(phase: InferPhase, conditions: dict[str, InferCondition]) -> str:
    """Derive a human-readable STATUS string from phase + conditions.

    Follows k8s ``kubectl get pod`` convention: phase for terminal states,
    ready condition for running states.
    """
    ready = conditions.get("ready")
    if phase == InferPhase.PENDING:
        return "Pending"
    if phase == InferPhase.STOPPING:
        return "Stopping"
    if phase in (InferPhase.STOPPED, InferPhase.FAILED):
        label = phase.value.capitalize()
        if ready and ready.reason:
            return f"{label} ({ready.reason})"
        return label
    # phase == RUNNING
    if ready and ready.status:
        return "Ready"
    reason = ready.reason if ready else "unknown"
    if reason == "no_health_url":
        return "Running (no health check)"
    return f"NotReady ({reason})"


def parse_phase(data: dict[str, object]) -> InferPhase:
    """Extract phase from handle data, with backward compat for legacy files."""
    raw = data.get("phase")
    if isinstance(raw, str) and raw:
        try:
            return InferPhase(raw)
        except ValueError:
            pass
    # Legacy: "status": "starting" → PENDING, anything else → RUNNING
    if data.get("status") == "starting":
        return InferPhase.PENDING
    return InferPhase.RUNNING


async def probe_and_sync(
    handle: InferHandle,
    handle_path: Path,
) -> tuple[InferPhase, dict[str, InferCondition]]:
    """Probe deployer status and sync changes back to the handle file.

    Writes back only when the phase actually changed, keeping handle
    files up-to-date regardless of which CLI command triggered the probe.
    """
    phase, conditions = await default_deployer.status(handle)

    apath = anyio.Path(handle_path)
    data = json.loads(await apath.read_text())

    # Don't let a deployer probe overwrite STOPPING with a non-terminal
    # phase — the stop command set STOPPING intentionally and the deployer
    # may still report RUNNING until the process actually exits.
    file_phase = data.get("phase")
    effective_phase = phase
    if file_phase == InferPhase.STOPPING.value and phase not in (
        InferPhase.STOPPED,
        InferPhase.FAILED,
    ):
        effective_phase = InferPhase.STOPPING

    new_conds = {
        k: {"status": c.status, "reason": c.reason} for k, c in conditions.items()
    }
    if (
        data.get("phase") != effective_phase.value
        or data.get("conditions") != new_conds
    ):
        data["phase"] = effective_phase.value
        data["conditions"] = new_conds
        tmp = handle_path.with_suffix(".tmp")
        try:
            await anyio.Path(tmp).write_text(json.dumps(data, indent=2))
            os.replace(str(tmp), str(apath))
        except OSError as e:
            with contextlib.suppress(OSError):
                await anyio.Path(tmp).unlink(missing_ok=True)
            logger.warning("Failed to sync handle file (returning stale state): {}", e)

    return phase, conditions


def claim_handle(model_name: str, *, backend: str = "") -> Path:
    """Atomically create a pending handle file for *model_name*.

    Uses ``O_CREAT | O_EXCL`` so that only one process can claim the
    name.  If the file already exists the caller must inspect its
    contents to decide whether to abort or clean up.

    Raises ``FileExistsError`` if another process already holds the claim.
    Returns the handle path on success.
    """
    HANDLE_DIR.mkdir(parents=True, exist_ok=True)
    handle_path = HANDLE_DIR / f"{model_name}.json"
    data: dict[str, object] = {
        "phase": "pending",
        "conditions": {"ready": {"status": False, "reason": "deploying"}},
        "pid": os.getpid(),
        "backend": backend,
    }
    fd = os.open(str(handle_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, json.dumps(data, indent=2).encode())
    finally:
        os.close(fd)
    return handle_path


def save_handle(
    model_name: str,
    handle: InferHandle,
    env: InferEnv | None = None,
) -> Path:
    """Persist a handle (and optional env snapshot) to disk.

    Writes to a temporary file first, then atomically replaces the
    target via ``os.replace()`` to prevent partial-read races.
    """
    HANDLE_DIR.mkdir(parents=True, exist_ok=True)
    handle_path = HANDLE_DIR / f"{model_name}.json"
    data: dict[str, object] = {
        "phase": "running",
        "conditions": {"ready": {"status": False, "reason": "initializing"}},
        "backend": handle.backend,
        "endpoint": handle.endpoint,
        "handle_id": handle.handle_id,
        "metadata": handle.metadata,
    }
    if env is not None:
        data["env"] = asdict(env)
    # Atomic write: tmp in same directory → os.replace() is atomic on
    # the same filesystem (POSIX guarantee).
    fd, tmp_path = tempfile.mkstemp(dir=HANDLE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, handle_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return handle_path


async def launch_model(
    model_name: str,
    commands: list[BackendCommand],
    backend: str,
    deployer: LocalDeployer | None = None,
    *,
    on_progress: Callable[[float, str], None] | None = None,
    detach: bool = False,
    timeout: float = 300.0,
    already_claimed: bool = False,
) -> tuple[list[InferHandle], InferEnv | None]:
    """Claim handle, deploy, collect env, and persist running state.

    Shared lifecycle used by both ``sieval infer start`` and ``sieval run``.

    When *already_claimed* is True, skips the ``claim_handle`` call
    (caller already owns the handle file, e.g. after stale-handle
    recovery in ``infer start``).

    Raises ``FileExistsError`` if the model name is already claimed
    (only when *already_claimed* is False).
    Cleans up the handle file on deploy failure.

    Returns (handles, env).
    """
    _dep = deployer or default_deployer
    handle_path = HANDLE_DIR / f"{model_name}.json"

    if not already_claimed:
        claim_handle(model_name, backend=backend)
    try:
        handles = await _dep.deploy(
            commands,
            detach=detach,
            timeout=timeout,
            on_progress=on_progress,
        )

        env: InferEnv | None = None
        try:
            env = await collect_basic_env(backend)
        except Exception as exc:
            logger.debug("collect_env failed for {}: {}", model_name, exc)

        if handles:
            save_handle(model_name, handles[0], env=env)
    except BaseException:
        handle_path.unlink(missing_ok=True)
        raise

    return handles, env


async def cleanup_model(
    model_name: str,
    handles: list[InferHandle],
    deployer: LocalDeployer | None = None,
) -> None:
    """Stop services and remove handle file for a model.

    Best-effort: logs warnings on failure but never raises.
    """
    _dep = deployer or default_deployer
    for handle in handles:
        try:
            await _dep.delete(handle)
            logger.debug(
                "Stopped {} (PID={})",
                handle.metadata.get("role", "?"),
                handle.handle_id,
            )
        except BaseException as e:
            logger.warning("Failed to stop {}: {}", handle.handle_id, e)

    handle_path = HANDLE_DIR / f"{model_name}.json"
    handle_path.unlink(missing_ok=True)


def load_handle(name: str) -> InferHandle:
    """Load a persisted handle by model name."""
    handle_path = HANDLE_DIR / f"{name}.json"
    if not handle_path.exists():
        raise typer.BadParameter(f"No handle found for {name!r}")
    data = json.loads(handle_path.read_text())

    phase = parse_phase(data)
    if phase == InferPhase.PENDING:
        raise typer.BadParameter(f"Model {name!r} is still starting up — not ready yet")

    # Strip fields not part of InferHandle
    for key in HANDLE_EXTRA_KEYS:
        data.pop(key, None)
    return InferHandle(**data)


def load_env(name: str) -> InferEnv | None:
    """Load the persisted InferEnv snapshot for a model."""
    handle_path = HANDLE_DIR / f"{name}.json"
    if not handle_path.exists():
        return None
    data = json.loads(handle_path.read_text())
    env_data = data.get("env")
    if env_data is None:
        return None
    return InferEnv(**env_data)
