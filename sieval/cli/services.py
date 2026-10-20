"""Start the vendored services the configured tasks need, for one SiEval process.

:func:`vendored_services_for` starts each service a task declares in
``Task.services`` whose endpoint variable is unset, once, shared by every task,
and stops it when the block exits. A preset variable is used as given. Services
run through the engine lifecycle: a handle under ``run-<pid>/``, a log at
``<result_dir>/logs/<name>/server.log``, their own process group.

code-eval runs ``<python> -m uvicorn app.server:app`` on this host, with no
isolation beyond the evaluator's own guards. ``SIEVAL_CODE_EVAL_DIR`` names the
copy (default: the checkout's ``vendor/code-evaluator``) and
``SIEVAL_CODE_EVAL_PYTHON`` the interpreter (default: SiEval's own).

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import anyio
from loguru import logger

from sieval.cli.infer.lifecycle import (
    cleanup_model,
    launch_model,
    remove_run_handle_dir,
    resolve_keep_logs,
    run_handle_dir,
)
from sieval.core.tasks.services import (
    CODE_EVAL,
    SERVICE_ENDPOINT_ENV,
    VENDORED_SERVICES_ENV,
    ServiceRuntime,
    check_service_names,
)
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferHandle
from sieval.infer.deployer import DeployError

CODE_EVAL_DIR_ENV = "SIEVAL_CODE_EVAL_DIR"
CODE_EVAL_PYTHON_ENV = "SIEVAL_CODE_EVAL_PYTHON"

#: ``<result_dir>/logs/`` holds what SiEval starts, so no task may be named this.
RESERVED_LOG_DIR = "logs"

HANDLE_PREFIX = "service-"
SERVICE_ROLE = "server"

READY_TIMEOUT = 120.0
_POLL_INTERVAL = 0.5
# The free port is released before the service binds it, so another process can
# take it in between; that start fails and is retried on a new port.
_START_ATTEMPTS = 3

# What the server imports; tests, docs and Dockerfiles do not change a verdict.
_CODE_EVAL_SOURCES = ("app", "quotebench")
_VENDORED_COMMIT_RE = re.compile(r"Vendored at commit:\s*`([0-9a-f]{7,40})`")
# Needed by the server itself, and by graded programs (requirements/livecodebench.txt).
_CODE_EVAL_SERVER_MODULES = ("fastapi", "uvicorn", "loguru", "psutil")
_CODE_EVAL_PROGRAM_MODULES = ("numpy", "pandas", "sortedcontainers")
# Recorded in the grader identity: graded programs import these from the
# service's interpreter (SciCode the last three).
_CODE_EVAL_RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "sortedcontainers",
    "scipy",
    "sympy",
    "matplotlib",
)

# Names this process has started and not yet stopped. The endpoint variables are
# process-wide, so a second concurrent run would share a service the first one
# stops; it is refused instead.
_running: set[str] = set()


@dataclass(frozen=True)
class VendoredService:
    """One vendored service, ready to launch."""

    name: str
    command: list[str]
    working_dir: Path
    health_path: str
    endpoint_path: str
    commit: str | None
    digest: str
    runtime: ServiceRuntime | None = None


def _checkout_vendor_dir() -> Path:
    import sieval

    return Path(sieval.__file__).resolve().parent.parent / "vendor"


def code_eval_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The code-evaluator copy to run; ``FileNotFoundError`` when there is none."""
    env = os.environ if environ is None else environ
    configured = env.get(CODE_EVAL_DIR_ENV, "").strip()
    path = (
        Path(configured).expanduser()
        if configured
        else _checkout_vendor_dir() / "code-evaluator"
    )
    if not (path / "app" / "server.py").is_file():
        where = f"{CODE_EVAL_DIR_ENV}={configured}" if configured else str(path)
        raise FileNotFoundError(
            f"A task needs the code-eval service and "
            f"{SERVICE_ENDPOINT_ENV[CODE_EVAL]} is unset, but no code-evaluator "
            f"is at {where} (an installed wheel carries none). Set "
            f"{CODE_EVAL_DIR_ENV} to a vendor/code-evaluator directory, or "
            f"{SERVICE_ENDPOINT_ENV[CODE_EVAL]} to a running service."
        )
    return path


def sources_digest(root: Path, subdirs: Iterable[str]) -> str:
    """``sha256:`` over every ``.py`` file under *subdirs*, by relative path."""
    files: dict[str, str] = {}
    for sub in subdirs:
        for path in sorted((root / sub).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def vendored_commit(root: Path) -> str | None:
    """The upstream commit ``VENDORED.md`` names, ``None`` when it names none."""
    try:
        match = _VENDORED_COMMIT_RE.search((root / "VENDORED.md").read_text())
    except OSError:
        return None
    return match.group(1) if match else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _own_runtime(packages: Iterable[str]) -> ServiceRuntime:
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "packages": {name: _package_version(name) for name in packages},
    }


_RUNTIME_PROBE = """\
import importlib.metadata as m, json, platform, sys
def v(n):
    try:
        return m.version(n)
    except m.PackageNotFoundError:
        return None
print(json.dumps({"python": platform.python_version(), "platform": sys.platform,
                  "packages": {n: v(n) for n in sys.argv[1:]}}))
"""


def interpreter_runtime(python: str, packages: Iterable[str]) -> ServiceRuntime:
    """Ask *python* for its version and the versions of *packages*."""
    names = list(packages)
    if python == sys.executable:
        return _own_runtime(names)
    try:
        out = subprocess.run(
            [python, "-c", _RUNTIME_PROBE, *names],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{CODE_EVAL_PYTHON_ENV}={python} cannot run: {exc}"
        ) from exc


def code_eval_service(
    port: int, environ: Mapping[str, str] | None = None
) -> VendoredService:
    """Describe the vendored code-evaluator on *port*.

    Under SiEval's own interpreter the ``code-eval`` group must be installed;
    another interpreter is checked only by whether it starts.
    """
    env = os.environ if environ is None else environ
    root = code_eval_dir(env)
    python = env.get(CODE_EVAL_PYTHON_ENV, "").strip()
    if not python:
        python = sys.executable
        missing = [
            m
            for m in (*_CODE_EVAL_SERVER_MODULES, *_CODE_EVAL_PROGRAM_MODULES)
            if importlib.util.find_spec(m) is None
        ]
        if missing:
            raise ModuleNotFoundError(
                f"The code-eval service needs {', '.join(missing)} in "
                f"{python}. Install the group (`pip install 'sieval[code-eval]'`), "
                f"set {CODE_EVAL_PYTHON_ENV} to an interpreter that has them, or "
                f"set {SERVICE_ENDPOINT_ENV[CODE_EVAL]} to a running service."
            )
    return VendoredService(
        name=CODE_EVAL,
        command=[
            python,
            "-m",
            "uvicorn",
            "app.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        working_dir=root,
        health_path="/health",
        endpoint_path="/evaluations",
        commit=vendored_commit(root),
        digest=sources_digest(root, _CODE_EVAL_SOURCES),
        runtime=interpreter_runtime(python, _CODE_EVAL_RUNTIME_PACKAGES),
    )


_BUILDERS = {CODE_EVAL: code_eval_service}


def services_to_start(
    required: Iterable[str], environ: Mapping[str, str] | None = None
) -> list[str]:
    """The services in *required* whose endpoint variable is unset."""
    env = os.environ if environ is None else environ
    names = sorted(set(required))
    check_service_names(names, "The configured tasks")
    return [n for n in names if not env.get(SERVICE_ENDPOINT_ENV[n], "").strip()]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def _env_overlay(values: Mapping[str, str]) -> Iterator[None]:
    """Set *values* in ``os.environ`` for the block, then restore it."""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


async def _start(
    name: str,
    *,
    handle_path: Path,
    log_dir: Path | None,
    keep: int | None,
) -> tuple[VendoredService, str, list[InferHandle]]:
    """Launch *name* on a free port; return it, its base URL and its handles."""
    for attempt in range(1, _START_ATTEMPTS + 1):
        port = _free_port()
        service = _BUILDERS[name](port)
        base = f"http://127.0.0.1:{port}"
        command = BackendCommand(
            cli_args=service.command,
            backend=name,
            host="127.0.0.1",
            role=SERVICE_ROLE,
            working_dir=str(service.working_dir),
            health_url=base + service.health_path,
        )
        try:
            handles, _ = await launch_model(
                name,
                [command],
                backend=name,
                handle_path=handle_path,
                log_dir=log_dir,
                keep_logs=keep,
                timeout=READY_TIMEOUT,
                poll_interval=_POLL_INTERVAL,
                collect_env=False,
                replace_stale=True,
            )
        except DeployError:
            if attempt == _START_ATTEMPTS:
                raise
            logger.warning("{} did not start on port {}; retrying", name, port)
            continue
        return service, base, handles
    raise AssertionError("unreachable")


@contextlib.asynccontextmanager
async def vendored_services_for(
    required: Iterable[str],
    *,
    result_dir: Callable[[], str | Path | None],
) -> AsyncIterator[dict[str, str]]:
    """Start each service in *required* whose endpoint is unset; yield endpoints.

    *result_dir* is read only when a service starts; its ``logs/<name>/`` then
    holds the log. Yields ``{name: endpoint}`` of what was started, with the
    endpoint variables and :data:`VENDORED_SERVICES_ENV` set for the block.
    Everything started is stopped on exit, shielded from cancellation.
    """
    names = services_to_start(required)
    busy = sorted(set(required) & _running)
    if busy:
        raise RuntimeError(
            f"{', '.join(busy)} is already started by another run in this "
            "process, which stops it when that run ends. Run one session per "
            f"process, or set {SERVICE_ENDPOINT_ENV[busy[0]]} to a service "
            "you manage."
        )
    if not names:
        yield {}
        return

    log_root = result_dir()
    handle_dir = run_handle_dir()
    keep = None if log_root else resolve_keep_logs(None)
    launched: list[tuple[str, list[InferHandle]]] = []
    endpoints: dict[str, str] = {}
    provenance: dict[str, dict] = {}
    try:
        for name in names:
            logger.warning(
                "Starting vendored service {} on this host. Graded programs run "
                "without isolation; set {} to a containerized service to isolate "
                "them.",
                name,
                SERVICE_ENDPOINT_ENV[name],
            )
            service, base, handles = await _start(
                name,
                handle_path=handle_dir / f"{HANDLE_PREFIX}{name}.json",
                log_dir=Path(log_root) / RESERVED_LOG_DIR / name if log_root else None,
                keep=keep,
            )
            launched.append((name, handles))
            _running.add(name)
            endpoints[name] = base + service.endpoint_path
            provenance[name] = {
                "commit": service.commit,
                "digest": service.digest,
                "runtime": service.runtime,
                "endpoint": endpoints[name],
            }
            logger.info("{} ready at {}", name, endpoints[name])

        overlay = {SERVICE_ENDPOINT_ENV[n]: url for n, url in endpoints.items()}
        overlay[VENDORED_SERVICES_ENV] = json.dumps(provenance, sort_keys=True)
        with _env_overlay(overlay):
            yield endpoints
    finally:
        with anyio.CancelScope(shield=True):
            for name, handles in reversed(launched):
                await cleanup_model(
                    name,
                    handles,
                    handle_path=handle_dir / f"{HANDLE_PREFIX}{name}.json",
                )
                _running.discard(name)
            # `sieval run` removes this directory itself once its engines stop.
            if not any(handle_dir.glob("*.json")):
                remove_run_handle_dir(handle_dir)
