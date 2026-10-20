"""
Inference service lifecycle: handle I/O, deployment, and status.

Handles live under ``~/.sieval/handles``, one directory per owner:
``sieval infer start`` writes ``<model>.json`` at the top level, and
``sieval run`` writes ``run-<pid>/<model>.json`` so that the two never collide
on a model name. Every change to a handle file goes through ``transition()``,
which holds one file lock over the whole directory tree.

Engine logs from ``sieval infer start`` live under ``~/.sieval/logs`` as
``<role>-<deployment_id>.log``; the last ``keep_logs`` per role survive each
launch. ``sieval run`` writes them beside its results when it has a
``result_dir``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import fcntl
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import typer
from anyio.to_thread import run_sync
from loguru import logger

from sieval.infer.backends.process import pid_alive
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import (
    InferCondition,
    InferEnv,
    InferHandle,
    InferPhase,
    MetadataValue,
)
from sieval.infer.deployer import (
    DEFAULT_READY_TIMEOUT,
    ENGINE_LOG_DIR,
    LocalDeployer,
    collect_basic_env,
    new_deployment_id,
)

HANDLE_DIR = Path.home() / ".sieval" / "handles"
default_deployer = LocalDeployer()

# Prefix of the per-process handle directory `sieval run` owns.
RUN_DIR_PREFIX = "run-"
_LOCK_NAME = ".lock"

DEFAULT_KEEP_LOGS = 3
KEEP_LOGS_ENV = "SIEVAL_KEEP_LOGS"

# Engine log names under ENGINE_LOG_DIR: `<role>-<deployment_id>.log`, plus the
# `<role>-<UTC second>.log` form earlier versions wrote, so pruning reaches both.
_LOG_SUFFIX_RE = r"(?:\d{8}T\d{6}Z-[0-9a-f]{8}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"

# Fields an older handle file used, mapped to their current names.
_LEGACY_FIELDS = {"pid": "owner_pid", "handle_id": "engine_pid"}


class HandleHeldError(FileExistsError):
    """A live process holds the handle that a claim asked for."""

    def __init__(self, path: Path, data: dict[str, object], pid: int) -> None:
        super().__init__(f"{path} is held by PID {pid}")
        self.path = path
        self.data = data
        self.pid = pid
        self.phase = parse_phase(data)


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


def parse_phase(data: Mapping[str, object]) -> InferPhase:
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


# ---------------------------------------------------------------------------
# Handle file format
# ---------------------------------------------------------------------------


def normalize_handle_data(data: dict[str, object]) -> dict[str, object]:
    """Return *data* with legacy field names (`pid`, `handle_id`) renamed."""
    out = dict(data)
    for old, new in _LEGACY_FIELDS.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    return out


def handle_from_data(data: Mapping[str, object]) -> InferHandle:
    """Build the deployer's ``InferHandle`` from (normalized) handle data."""
    metadata = data.get("metadata")
    endpoint = data.get("endpoint")
    return InferHandle(
        backend=str(data.get("backend", "")),
        endpoint=str(endpoint) if endpoint else None,
        handle_id=str(data.get("engine_pid", "")),
        metadata=(
            dict(cast(dict[str, MetadataValue], metadata))
            if isinstance(metadata, dict)
            else {}
        ),
    )


def log_file_of(data: Mapping[str, object] | None) -> Path | None:
    """The engine log a handle names, if any."""
    if not data:
        return None
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    log_file = cast(dict[str, object], metadata).get("log_file")
    return Path(str(log_file)) if log_file else None


def _as_pid(raw: object) -> int | None:
    try:
        return int(raw) if raw is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # Not a numeric PID (e.g. a container ID): liveness is unknowable.
        return None


def holder_pid(data: Mapping[str, object]) -> int | None:
    """The PID whose liveness decides whether a handle is held.

    A pending handle has no engine yet, so its owner holds it; afterwards the
    engine process does.
    """
    if parse_phase(data) == InferPhase.PENDING:
        return _as_pid(data.get("owner_pid"))
    return _as_pid(data.get("engine_pid"))


def _holder_alive(data: Mapping[str, object]) -> bool:
    pid = holder_pid(data)
    return pid is not None and pid_alive(pid)


# ---------------------------------------------------------------------------
# transition(): the one locked writer
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _handles_lock() -> Iterator[None]:
    """Hold an exclusive lock over the whole handle tree."""
    HANDLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(HANDLE_DIR / _LOCK_NAME, "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, data: object) -> None:
    """Write JSON to a unique temp file beside *path*, then replace *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _read_handle_file(path: Path) -> dict[str, object] | None:
    """Read and normalize a handle file; ``None`` when absent or unreadable."""
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return {"corrupt": True}
    if not isinstance(raw, dict):
        return {"corrupt": True}
    return normalize_handle_data(raw)


# Returned by an update function to leave the file untouched.
UNCHANGED: dict[str, object] = {}


def transition(
    path: Path,
    update: Callable[[dict[str, object] | None], dict[str, object] | None],
) -> dict[str, object] | None:
    """Apply one phase change to the handle at *path* under the handle lock.

    *update* receives the current normalized data (``None`` when the file does
    not exist; ``{"corrupt": True}`` when it cannot be parsed) and returns the
    new data to write, ``None`` to delete the file, or ``UNCHANGED`` to leave it
    alone. Exceptions from *update* propagate with the file untouched.

    Returns what the file holds afterwards.
    """
    with _handles_lock():
        current = _read_handle_file(path)
        new = update(current)
        if new is UNCHANGED:
            return current
        if new is None:
            path.unlink(missing_ok=True)
            return None
        write_json_atomic(path, new)
        return new


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def handle_path_for(name: str) -> Path:
    """Path of the handle *name* refers to.

    *name* is a model name for ``sieval infer start`` handles, or
    ``run-<pid>/<model>`` for a handle ``sieval run`` owns.
    """
    path = (HANDLE_DIR / f"{name}.json").resolve()
    root = HANDLE_DIR.resolve()
    if root not in path.parents or len(path.relative_to(root).parts) > 2:
        raise typer.BadParameter(f"Invalid handle name {name!r}")
    return path


def run_handle_dir(pid: int | None = None) -> Path:
    """Handle directory a ``sieval run`` process owns."""
    return HANDLE_DIR / f"{RUN_DIR_PREFIX}{os.getpid() if pid is None else pid}"


def iter_handle_files() -> Iterator[tuple[str, Path]]:
    """Yield ``(name, path)`` for every handle, top-level first."""
    if not HANDLE_DIR.is_dir():
        return
    for path in sorted(HANDLE_DIR.glob("*.json")):
        yield path.stem, path
    for run_dir in sorted(_run_dirs()):
        for path in sorted(run_dir.glob("*.json")):
            yield f"{run_dir.name}/{path.stem}", path


def _run_dirs() -> list[Path]:
    if not HANDLE_DIR.is_dir():
        return []
    return [
        p
        for p in HANDLE_DIR.iterdir()
        if p.is_dir()
        and p.name.startswith(RUN_DIR_PREFIX)
        and _as_pid(p.name.removeprefix(RUN_DIR_PREFIX)) is not None
    ]


def remove_run_handle_dir(run_dir: Path) -> None:
    """Remove a ``run-<pid>`` directory and everything left in it."""
    if not run_dir.exists():
        return
    with _handles_lock():
        shutil.rmtree(run_dir, ignore_errors=True)


def sweep_stale_run_dirs() -> list[Path]:
    """Remove handles left by ``sieval run`` processes that no longer exist.

    A handle whose engine is still alive is kept, so an orphaned engine stays
    visible in ``sieval infer list`` and reachable by ``sieval infer stop``.
    The directory goes once it holds no such handle. Returns removed dirs.
    """
    removed: list[Path] = []
    for run_dir in _run_dirs():
        owner = _as_pid(run_dir.name.removeprefix(RUN_DIR_PREFIX))
        if owner is None or pid_alive(owner):
            continue
        with _handles_lock():
            kept = False
            for path in run_dir.glob("*.json"):
                data = _read_handle_file(path)
                if (
                    data is not None
                    and parse_phase(data) != InferPhase.PENDING
                    and _holder_alive(data)
                ):
                    kept = True
                    logger.warning(
                        "Engine PID {} from finished sieval run {} is still alive",
                        data.get("engine_pid"),
                        owner,
                    )
                    continue
                path.unlink(missing_ok=True)
            if not kept:
                shutil.rmtree(run_dir, ignore_errors=True)
                removed.append(run_dir)
    return removed


# ---------------------------------------------------------------------------
# Phase changes
# ---------------------------------------------------------------------------


def claim_handle(
    handle_path: Path,
    *,
    backend: str = "",
    replace_stale: bool = False,
) -> Path:
    """Create a pending handle at *handle_path* owned by this process.

    Raises ``HandleHeldError`` when a live process holds the handle. Any other
    existing handle (dead holder, unknown holder, corrupt file) raises
    ``FileExistsError`` unless *replace_stale*, in which case it is replaced
    within the same locked transition.
    """

    def _claim(current: dict[str, object] | None) -> dict[str, object]:
        if current is not None:
            if not current.get("corrupt"):
                pid = holder_pid(current)
                if pid is not None and pid_alive(pid):
                    raise HandleHeldError(handle_path, current, pid)
            if not replace_stale:
                raise FileExistsError(f"{handle_path} already exists")
            logger.info("Replacing stale handle {}", handle_path)
        return {
            "phase": InferPhase.PENDING.value,
            "conditions": {"ready": {"status": False, "reason": "deploying"}},
            "owner_pid": os.getpid(),
            "backend": backend,
        }

    transition(handle_path, _claim)
    return handle_path


def save_handle(
    handle_path: Path,
    handle: InferHandle,
    env: InferEnv | None = None,
    *,
    deployment_id: str | None = None,
    keep_logs: int | None = None,
) -> Path:
    """Record a launched engine: the handle moves from pending to running.

    *keep_logs* is recorded when the engine logs to ``ENGINE_LOG_DIR``, so a
    later ``infer stop`` knows whether to delete the log.
    """
    engine_pid: object = handle.handle_id
    if handle.handle_id.isdigit():
        engine_pid = int(handle.handle_id)

    def _save(current: dict[str, object] | None) -> dict[str, object]:
        base = current if current and not current.get("corrupt") else {}
        data: dict[str, object] = {
            "phase": InferPhase.RUNNING.value,
            "conditions": {"ready": {"status": False, "reason": "initializing"}},
            "owner_pid": base.get("owner_pid", os.getpid()),
            "backend": handle.backend,
            "endpoint": handle.endpoint,
            "engine_pid": engine_pid,
            "metadata": handle.metadata,
        }
        if deployment_id is not None:
            data["deployment_id"] = deployment_id
        if keep_logs is not None:
            data["keep_logs"] = keep_logs
        if env is not None:
            data["env"] = asdict(env)
        return data

    transition(handle_path, _save)
    return handle_path


def mark_stopping(handle_path: Path) -> None:
    """Move a handle to STOPPING, if it still exists."""

    def _stop(current: dict[str, object] | None) -> dict[str, object] | None:
        if current is None or current.get("corrupt"):
            return UNCHANGED
        return {**current, "phase": InferPhase.STOPPING.value}

    transition(handle_path, _stop)


def remove_handle(handle_path: Path) -> dict[str, object] | None:
    """Delete a handle; return the data it held."""
    removed: dict[str, object] | None = None

    def _remove(current: dict[str, object] | None) -> None:
        nonlocal removed
        removed = current
        return None

    transition(handle_path, _remove)
    return removed


def remove_if_dead(handle_path: Path) -> dict[str, object] | None:
    """Delete a handle whose holder is dead, re-checked under the lock.

    The re-check keeps a handle that another process reclaimed between the
    caller's read and this call. Returns the deleted data, or ``None`` when
    the handle was kept or absent.
    """
    removed: dict[str, object] | None = None

    def _check(current: dict[str, object] | None) -> dict[str, object] | None:
        nonlocal removed
        if current is None or _holder_alive(current):
            return UNCHANGED
        removed = current
        return None

    transition(handle_path, _check)
    return removed


async def probe_and_sync(
    handle: InferHandle,
    handle_path: Path,
) -> tuple[InferPhase, dict[str, InferCondition]]:
    """Probe deployer status and sync changes back to the handle file.

    The probe runs outside the lock; the write-back is one ``transition()``.
    A handle deleted in the meantime stays deleted.
    """
    phase, conditions = await default_deployer.status(handle)
    new_conds = {
        k: {"status": c.status, "reason": c.reason} for k, c in conditions.items()
    }

    def _sync(current: dict[str, object] | None) -> dict[str, object] | None:
        if current is None or current.get("corrupt"):
            return UNCHANGED
        # Don't let a probe overwrite STOPPING with a non-terminal phase: the
        # stop command set it, and the engine may still answer until it exits.
        effective = phase
        if current.get("phase") == InferPhase.STOPPING.value and phase not in (
            InferPhase.STOPPED,
            InferPhase.FAILED,
        ):
            effective = InferPhase.STOPPING
        if (
            current.get("phase") == effective.value
            and current.get("conditions") == new_conds
        ):
            return UNCHANGED
        return {**current, "phase": effective.value, "conditions": new_conds}

    try:
        await run_sync(transition, handle_path, _sync)
    except OSError as e:
        logger.warning("Failed to sync handle file (returning stale state): {}", e)

    return phase, conditions


# ---------------------------------------------------------------------------
# Engine logs
# ---------------------------------------------------------------------------


def resolve_keep_logs(flag: int | None) -> int:
    """Number of engine logs to keep per role: flag, then env, then default."""
    if flag is not None:
        return flag
    raw = os.environ.get(KEEP_LOGS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_KEEP_LOGS
    try:
        return int(raw)
    except ValueError:
        raise typer.BadParameter(
            f"{KEEP_LOGS_ENV} must be an integer, got {raw!r}",
            param_hint="--keep-logs",
        ) from None


def _live_log_files() -> set[Path]:
    """Log files named by any existing handle; pruning never removes these."""
    live: set[Path] = set()
    for _, path in iter_handle_files():
        log_file = log_file_of(_read_handle_file(path))
        if log_file is not None:
            live.add(log_file)
    return live


def prune_engine_logs(
    roles: list[str],
    keep: int,
    *,
    log_dir: Path | None = None,
) -> list[Path]:
    """Delete old engine logs so a launch leaves at most *keep* per role.

    Runs before the launch, so it keeps ``keep - 1`` existing logs per role. A
    log a handle still names is never deleted, though it counts toward
    *keep*. A negative *keep* keeps every log. Returns the deleted paths.
    """
    if keep < 0:
        return []
    root = ENGINE_LOG_DIR if log_dir is None else log_dir
    if not root.is_dir():
        return []
    live = {p.resolve() for p in _live_log_files()}
    deleted: list[Path] = []
    for role in sorted(set(roles)):
        pattern = re.compile(rf"{re.escape(role)}-{_LOG_SUFFIX_RE}\.log")
        logs = [p for p in root.iterdir() if pattern.fullmatch(p.name)]
        logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in logs[max(keep - 1, 0) :]:
            if path.resolve() in live:
                continue
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
                deleted.append(path)
    return deleted


def delete_logs_if_unkept(data: Mapping[str, object] | None) -> None:
    """Delete the engine log of a stopped handle that asked to keep none."""
    if not data or data.get("keep_logs") != 0:
        return
    log_file = log_file_of(data)
    if log_file is not None:
        log_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Launch and cleanup
# ---------------------------------------------------------------------------


async def launch_model(
    model_name: str,
    commands: list[BackendCommand],
    backend: str,
    deployer: LocalDeployer | None = None,
    *,
    handle_path: Path | None = None,
    log_dir: Path | None = None,
    keep_logs: int = DEFAULT_KEEP_LOGS,
    on_progress: Callable[[float, str], None] | None = None,
    detach: bool = False,
    timeout: float = DEFAULT_READY_TIMEOUT,
    already_claimed: bool = False,
    replace_stale: bool = False,
) -> tuple[list[InferHandle], InferEnv | None]:
    """Claim handle, deploy, collect env, and persist running state.

    Shared lifecycle used by both ``sieval infer start`` and ``sieval run``.

    *handle_path* defaults to ``HANDLE_DIR/<model_name>.json``. With *log_dir*,
    each role logs to ``<log_dir>/<role>.log``; without it, to
    ``ENGINE_LOG_DIR/<role>-<deployment_id>.log``, pruned to *keep_logs* per
    role first.

    When *already_claimed* is True, skips the ``claim_handle`` call (the
    caller already owns the handle file). Otherwise raises
    ``FileExistsError`` if the handle exists, unless *replace_stale* and its
    holder is dead. Removes the handle on deploy failure.

    Returns (handles, env).
    """
    _dep = deployer or default_deployer
    path = handle_path or HANDLE_DIR / f"{model_name}.json"
    deployment_id = new_deployment_id()

    roles = [cmd.role for cmd in commands]
    if log_dir is None:
        prune_engine_logs(roles, keep_logs)
        log_files = {
            role: ENGINE_LOG_DIR / f"{role}-{deployment_id}.log" for role in roles
        }
    else:
        log_files = {role: log_dir / f"{role}.log" for role in roles}

    if not already_claimed:
        claim_handle(path, backend=backend, replace_stale=replace_stale)
    try:
        handles = await _dep.deploy(
            commands,
            detach=detach,
            timeout=timeout,
            on_progress=on_progress,
            log_files=log_files,
        )

        env: InferEnv | None = None
        try:
            env = await collect_basic_env(backend)
        except Exception as exc:
            logger.debug("collect_env failed for {}: {}", model_name, exc)

        if handles:
            save_handle(
                path,
                handles[0],
                env=env,
                deployment_id=deployment_id,
                keep_logs=keep_logs if log_dir is None else None,
            )
    except BaseException:
        with contextlib.suppress(OSError):
            remove_handle(path)
        raise

    return handles, env


async def cleanup_model(
    model_name: str,
    handles: list[InferHandle],
    deployer: LocalDeployer | None = None,
    *,
    handle_path: Path | None = None,
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

    path = handle_path or HANDLE_DIR / f"{model_name}.json"
    try:
        data = remove_handle(path)
        delete_logs_if_unkept(data)
    except OSError as e:
        logger.warning("Failed to remove handle {}: {}", path, e)


def _load_data(name: str) -> dict[str, object] | None:
    data = _read_handle_file(handle_path_for(name))
    if data is not None and data.get("corrupt"):
        raise typer.BadParameter(f"Handle for {name!r} is unreadable")
    return data


def load_handle(name: str) -> InferHandle:
    """Load a persisted handle by name (``<model>`` or ``run-<pid>/<model>``)."""
    data = _load_data(name)
    if data is None:
        raise typer.BadParameter(f"No handle found for {name!r}")
    if parse_phase(data) == InferPhase.PENDING:
        raise typer.BadParameter(f"Model {name!r} is still starting up — not ready yet")
    return handle_from_data(data)


def load_env(name: str) -> InferEnv | None:
    """Load the persisted InferEnv snapshot for a model."""
    data = _load_data(name)
    if data is None:
        return None
    env_data = data.get("env")
    if not isinstance(env_data, dict):
        return None
    return InferEnv(**cast(dict[str, Any], env_data))
