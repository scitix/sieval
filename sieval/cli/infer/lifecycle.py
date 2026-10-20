"""
Inference service lifecycle: handle I/O, deployment, and status.

Handles live in ``~/.sieval/handles``: ``<model>.json`` for ``infer start``,
``run-<pid>/<model>.json`` for ``sieval run``. Every handle write goes through
``transition()`` under one file lock. Engine logs live in
``~/.sieval/logs/<model>/``, pruned per role at each launch.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import enum
import errno
import fcntl
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

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

# `<model>/<role>-<deployment_id>.log`, and the flat `<role>-<UTC second>.log`
# earlier versions wrote.
_DEPLOYMENT_ID_RE = r"\d{8}T\d{6}Z-[0-9a-f]{8}"
_LEGACY_LOG_SUFFIX_RE = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"

# Fields an older handle file used, mapped to their current names.
_LEGACY_FIELDS = {"pid": "owner_pid", "handle_id": "engine_pid"}


class HandleHeldError(FileExistsError):
    """A live process holds the handle that a claim asked for."""

    def __init__(
        self,
        path: Path,
        data: dict[str, object],
        pid: int,
        message: str | None = None,
    ) -> None:
        super().__init__(message or f"{path} is held by PID {pid}")
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


def log_files_of(data: Mapping[str, object] | None) -> set[Path]:
    """Every engine log a handle names: ``log_files`` (set at claim) and
    ``metadata.log_file`` (set once the engine runs; older handles have only
    this)."""
    if not data:
        return set()
    found: set[Path] = set()
    listed = data.get("log_files")
    if isinstance(listed, list):
        found.update(Path(str(p)) for p in cast(list[object], listed) if p)
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        log_file = cast(dict[str, object], metadata).get("log_file")
        if log_file:
            found.add(Path(str(log_file)))
    return found


def _as_pid(raw: object) -> int | None:
    if not isinstance(raw, int | str):
        return None
    try:
        return int(raw)
    except ValueError:
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


class _Unchanged(enum.Enum):
    UNCHANGED = enum.auto()


# Returned by an update function to leave the file untouched.
UNCHANGED = _Unchanged.UNCHANGED

type HandleUpdate = Callable[
    [dict[str, object] | None], dict[str, object] | None | _Unchanged
]


def transition(path: Path, update: HandleUpdate) -> dict[str, object] | None:
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
    """Remove a ``run-<pid>`` directory if empty.

    A handle still here came from an earlier process with this PID whose engine
    outlived it; it stays for ``infer list`` / ``infer stop``.
    """
    if not run_dir.exists():
        return
    with _handles_lock():
        for tmp in run_dir.glob(".*.tmp"):
            tmp.unlink(missing_ok=True)
        try:
            run_dir.rmdir()
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT):
                raise
            if e.errno != errno.ENOENT:
                logger.warning(
                    "Keeping {}: it holds handles from an earlier process", run_dir
                )


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
        if "log_files" in base:
            data["log_files"] = base["log_files"]
        if deployment_id is not None:
            data["deployment_id"] = deployment_id
        if keep_logs is not None:
            data["keep_logs"] = keep_logs
        if env is not None:
            data["env"] = asdict(env)
        return data

    transition(handle_path, _save)
    return handle_path


def record_log_files(handle_path: Path, log_files: list[Path]) -> None:
    """Name a pending launch's logs in its handle, so pruning spares them."""

    def _record(current: dict[str, object] | None) -> dict[str, object] | _Unchanged:
        if current is None or current.get("corrupt"):
            return UNCHANGED
        return {**current, "log_files": [str(p) for p in log_files]}

    transition(handle_path, _record)


def mark_stopping(handle_path: Path) -> None:
    """Move a handle to STOPPING, if it still exists."""

    def _stop(current: dict[str, object] | None) -> dict[str, object] | _Unchanged:
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

    def _check(current: dict[str, object] | None) -> None | _Unchanged:
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

    def _sync(current: dict[str, object] | None) -> dict[str, object] | _Unchanged:
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
    """Engine logs to keep per model and role: flag, then env, then default."""
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
        live.update(log_files_of(_read_handle_file(path)))
    return live


def _prune_group(logs: list[Path], keep: int, live: set[Path]) -> list[Path]:
    """Delete all but the newest *keep* of *logs*, sparing live ones."""
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    deleted: list[Path] = []
    for path in logs[keep:]:
        if path.resolve() in live:
            continue
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
            deleted.append(path)
    return deleted


def prune_engine_logs(
    model_name: str,
    roles: list[str],
    keep: int,
    *,
    log_dir: Path | None = None,
) -> list[Path]:
    """Prune *model_name*'s logs to ``keep - 1`` per role before a launch.

    Flat logs from earlier versions are pruned to *keep* per role as their own
    group. Logs a handle names are never deleted. Negative *keep* keeps all.
    Returns the deleted paths.
    """
    if keep < 0:
        return []
    root = ENGINE_LOG_DIR if log_dir is None else log_dir
    if not root.is_dir():
        return []
    live = {p.resolve() for p in _live_log_files()}
    model_dir = root / model_name
    deleted: list[Path] = []
    for role in sorted(set(roles)):
        escaped = re.escape(role)
        if model_dir.is_dir():
            current = re.compile(rf"{escaped}-{_DEPLOYMENT_ID_RE}\.log")
            logs = [p for p in model_dir.iterdir() if current.fullmatch(p.name)]
            deleted += _prune_group(logs, max(keep - 1, 0), live)
        legacy = re.compile(rf"{escaped}-{_LEGACY_LOG_SUFFIX_RE}\.log")
        flat = [p for p in root.iterdir() if legacy.fullmatch(p.name)]
        deleted += _prune_group(flat, keep, live)
    return deleted


def delete_logs_if_unkept(data: Mapping[str, object] | None) -> None:
    """Delete the engine logs of a stopped handle that asked to keep none."""
    if not data or data.get("keep_logs") != 0:
        return
    for log_file in log_files_of(data):
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
    keep_logs: int | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    detach: bool = False,
    timeout: float = DEFAULT_READY_TIMEOUT,
    already_claimed: bool = False,
    replace_stale: bool = False,
) -> tuple[list[InferHandle], InferEnv | None]:
    """Claim handle, deploy, collect env, and persist running state.

    Shared lifecycle used by both ``sieval infer start`` and ``sieval run``.

    *handle_path* defaults to ``HANDLE_DIR/<model_name>.json``. With *log_dir*,
    each role logs to ``<log_dir>/<role>.log`` and *keep_logs* is ignored;
    otherwise logs go to ``ENGINE_LOG_DIR/<model_name>/``, pruned to
    *keep_logs* (default ``DEFAULT_KEEP_LOGS``) per role.

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
    keep: int | None = None
    if log_dir is None:
        keep = DEFAULT_KEEP_LOGS if keep_logs is None else keep_logs
        model_log_dir = ENGINE_LOG_DIR / model_name
        log_files = {
            role: model_log_dir / f"{role}-{deployment_id}.log" for role in roles
        }
    else:
        log_files = {role: log_dir / f"{role}.log" for role in roles}

    if not already_claimed:
        claim_handle(path, backend=backend, replace_stale=replace_stale)
    try:
        record_log_files(path, list(log_files.values()))
        if keep is not None:
            prune_engine_logs(model_name, roles, keep)
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
                keep_logs=keep,
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
    return InferEnv(**cast(dict, env_data))
