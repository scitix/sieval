"""Tests for the vendored services SiEval starts for the tasks that declare them.

The service under test is a stub HTTP server on the standard library, started
through the same lifecycle as the real code-evaluator; the real one is launched
in ``tests/integration/test_vendored_code_eval.py``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import anyio
import httpx
import orjson
import pytest

from sieval.cli import services
from sieval.cli.services import (
    CODE_EVAL_DIR_ENV,
    CODE_EVAL_PYTHON_ENV,
    VendoredService,
    check_code_eval,
    check_vendored_services,
    code_eval_dir,
    code_eval_service,
    sources_digest,
    vendored_commit,
    vendored_services_for,
)
from sieval.cli.session import arun_session
from sieval.core.tasks.services import (
    CODE_EVAL,
    SERVICE_ENDPOINT_ENV,
    VENDORED_SERVICES_ENV,
    ServiceRuntime,
)
from sieval.infer.backends.process import pid_alive
from tests.unit.core.runners.test_runner import MockTask

_ENDPOINT_ENV = SERVICE_ENDPOINT_ENV[CODE_EVAL]
_RUNTIME: ServiceRuntime = {
    "python": "3.12.0",
    "platform": "linux",
    "packages": {"numpy": "2.2.0"},
}

_STUB = """\
import http.server, os, sys

with open(sys.argv[2], "a") as fh:
    fh.write(f"{os.getpid()}\\n")


class H(http.server.BaseHTTPRequestHandler):
    def _ok(self, body: bytes) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._ok(b'{"status": true}')

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._ok(b'{"status": true, "msg": "stub"}')

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace code-eval's launch with a stub server; return its start record.

    Each start appends the server's PID to the returned file.
    """
    script = tmp_path / "stub_server.py"
    script.write_text(_STUB)
    starts = tmp_path / "starts.txt"
    starts.touch()

    def _build(port: int) -> VendoredService:
        return VendoredService(
            name=CODE_EVAL,
            command=[sys.executable, str(script), str(port), str(starts)],
            working_dir=tmp_path,
            health_path="/health",
            endpoint_path="/evaluations",
            commit="e4802268",
            digest="sha256:stub",
            runtime=_RUNTIME,
        )

    monkeypatch.setitem(services._BUILDERS, CODE_EVAL, _build)
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    monkeypatch.setattr(
        "sieval.cli.infer.lifecycle.ENGINE_LOG_DIR", tmp_path / "engine-logs"
    )
    monkeypatch.delenv(_ENDPOINT_ENV, raising=False)
    monkeypatch.delenv(VENDORED_SERVICES_ENV, raising=False)
    return starts


def _pids(starts: Path) -> list[int]:
    return [int(line) for line in starts.read_text().split()]


# ===================================================================
# The context manager
# ===================================================================
class TestVendoredServicesFor:
    @pytest.mark.anyio
    async def test_starts_once_sets_the_endpoint_and_stops(
        self, stub: Path, tmp_path: Path
    ) -> None:
        result_dir = tmp_path / "out"
        async with vendored_services_for(
            [CODE_EVAL, CODE_EVAL], result_dir=lambda: result_dir
        ) as endpoints:
            url = endpoints[CODE_EVAL]
            assert os.environ[_ENDPOINT_ENV] == url
            recorded = json.loads(os.environ[VENDORED_SERVICES_ENV])[CODE_EVAL]
            assert recorded == {
                "commit": "e4802268",
                "digest": "sha256:stub",
                "runtime": _RUNTIME,
                "endpoint": url,
            }
            async with httpx.AsyncClient() as client:
                assert (await client.post(url, json={})).status_code == 200
            [pid] = _pids(stub)
            assert pid_alive(pid)
            assert list((tmp_path / "handles").glob("run-*/service-code-eval.json"))

        assert not pid_alive(pid)
        assert _ENDPOINT_ENV not in os.environ
        assert VENDORED_SERVICES_ENV not in os.environ
        assert not list((tmp_path / "handles").glob("run-*"))
        assert (result_dir / "logs" / "services" / CODE_EVAL / "server.log").exists()

    @pytest.mark.anyio
    async def test_a_preset_endpoint_starts_nothing(
        self, stub: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENDPOINT_ENV, "http://sandbox:11451/evaluations")
        async with vendored_services_for(
            [CODE_EVAL], result_dir=lambda: None
        ) as endpoints:
            assert endpoints == {}
            assert os.environ[_ENDPOINT_ENV] == "http://sandbox:11451/evaluations"
            assert VENDORED_SERVICES_ENV not in os.environ
        assert _pids(stub) == []

    @pytest.mark.anyio
    async def test_no_declared_service_starts_nothing(self, stub: Path) -> None:
        async with vendored_services_for([], result_dir=lambda: None) as endpoints:
            assert endpoints == {}
        assert _pids(stub) == []

    @pytest.mark.anyio
    async def test_an_error_in_the_block_stops_the_service(self, stub: Path) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with vendored_services_for([CODE_EVAL], result_dir=lambda: None):
                raise RuntimeError("boom")
        [pid] = _pids(stub)
        assert not pid_alive(pid)
        assert _ENDPOINT_ENV not in os.environ

    @pytest.mark.anyio
    async def test_cancellation_stops_the_service(self, stub: Path) -> None:
        """An interrupt cancels the run; cleanup still completes, shielded."""
        with anyio.CancelScope() as scope:
            async with vendored_services_for([CODE_EVAL], result_dir=lambda: None):
                scope.cancel()
                await anyio.sleep(30)
        assert scope.cancelled_caught
        [pid] = _pids(stub)
        assert not pid_alive(pid)

    @pytest.mark.anyio
    async def test_keyboard_interrupt_stops_the_service(self, stub: Path) -> None:
        with pytest.raises(KeyboardInterrupt):
            async with vendored_services_for([CODE_EVAL], result_dir=lambda: None):
                raise KeyboardInterrupt
        [pid] = _pids(stub)
        assert not pid_alive(pid)

    @pytest.mark.anyio
    async def test_an_unknown_service_is_refused(self, stub: Path) -> None:
        with pytest.raises(ValueError, match="unknown service"):
            async with vendored_services_for(["sandbox"], result_dir=lambda: None):
                pass

    @pytest.mark.anyio
    async def test_a_second_run_in_the_same_process_is_refused(
        self, stub: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The outer run would stop the service under the inner one."""
        async with vendored_services_for([CODE_EVAL], result_dir=lambda: None):
            with pytest.raises(RuntimeError, match="one session per process"):
                async with vendored_services_for([CODE_EVAL], result_dir=lambda: None):
                    pass
        assert len(_pids(stub)) == 1
        async with vendored_services_for([CODE_EVAL], result_dir=lambda: None):
            pass  # released once the first run stopped it
        assert len(_pids(stub)) == 2

    @pytest.mark.anyio
    async def test_a_port_taken_before_the_service_binds_it_is_retried(
        self, stub: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            busy = taken.getsockname()[1]
            ports = iter([busy, services._free_port()])
            monkeypatch.setattr(services, "_free_port", lambda: next(ports))
            monkeypatch.setattr(services, "READY_TIMEOUT", 10.0)
            async with vendored_services_for(
                [CODE_EVAL], result_dir=lambda: None
            ) as endpoints:
                assert f":{busy}/" not in endpoints[CODE_EVAL]
        assert len(_pids(stub)) == 2  # the first died on the taken port

    @pytest.mark.anyio
    async def test_without_a_result_dir_logs_go_under_the_service_name(
        self, stub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SIEVAL_KEEP_LOGS", raising=False)
        async with vendored_services_for([CODE_EVAL], result_dir=lambda: None):
            pass
        [log] = (tmp_path / "engine-logs" / CODE_EVAL).glob("server-*.log")
        assert log.exists()


# ===================================================================
# Locating and describing the code-evaluator
# ===================================================================
class TestCodeEvalService:
    def test_default_dir_is_the_checkouts_vendored_copy(self) -> None:
        root = code_eval_dir({})
        assert (root / "app" / "server.py").is_file()
        assert root.name == "code-evaluator"

    def test_a_configured_dir_without_the_app_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=CODE_EVAL_DIR_ENV):
            code_eval_dir({CODE_EVAL_DIR_ENV: str(tmp_path)})

    def test_command_runs_uvicorn_in_the_evaluator_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(services, "interpreter_runtime", lambda *_: _RUNTIME)
        service = code_eval_service(5000, {CODE_EVAL_PYTHON_ENV: "/opt/ce/bin/python"})
        assert service.command == [
            "/opt/ce/bin/python",
            "-m",
            "uvicorn",
            "app.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
        assert (service.working_dir / "app" / "server.py").is_file()
        assert service.commit == vendored_commit(service.working_dir)
        assert service.digest.startswith("sha256:")

    def test_own_interpreter_without_the_group_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(services.importlib.util, "find_spec", lambda _name: None)
        with pytest.raises(ModuleNotFoundError, match="sieval\\[code-eval\\]"):
            code_eval_service(5000, {})

    def test_a_module_graded_programs_import_is_checked_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = services.importlib.util.find_spec
        monkeypatch.setattr(
            services.importlib.util,
            "find_spec",
            lambda name: None if name == "sortedcontainers" else real(name),
        )
        with pytest.raises(ModuleNotFoundError, match="needs sortedcontainers in"):
            code_eval_service(5000, {})

    def test_runtime_records_the_interpreter_and_program_packages(self) -> None:
        runtime = code_eval_service(5000, {}).runtime
        assert runtime is not None
        assert runtime["python"] == platform.python_version()
        assert runtime["packages"]["numpy"] == importlib.metadata.version("numpy")
        assert set(runtime["packages"]) >= {"numpy", "pandas", "scipy"}

    def test_another_interpreter_is_asked_for_its_runtime(self) -> None:
        # The probe path, not the in-process shortcut: same answer either way.
        own = services._own_runtime(["numpy", "no-such-package"])
        # Same interpreter under another spelling, so the subprocess probe runs.
        python = str(Path(sys.executable).parent / "." / Path(sys.executable).name)
        other = services.interpreter_runtime(python, ["numpy", "no-such-package"])
        assert other == own
        assert other["packages"]["no-such-package"] is None

    def test_an_interpreter_that_cannot_run_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match=CODE_EVAL_PYTHON_ENV):
            services.interpreter_runtime(str(tmp_path / "python"), ["numpy"])

    def test_vendored_commit_reads_vendored_md(self, tmp_path: Path) -> None:
        (tmp_path / "VENDORED.md").write_text("- Vendored at commit: `abcdef1`\n")
        assert vendored_commit(tmp_path) == "abcdef1"
        (tmp_path / "VENDORED.md").write_text("no commit here\n")
        assert vendored_commit(tmp_path) is None
        assert vendored_commit(tmp_path / "missing") is None

    def test_digest_covers_python_sources_only(self, tmp_path: Path) -> None:
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "server.py").write_text("x = 1\n")
        (tmp_path / "app" / "README.md").write_text("docs")
        first = sources_digest(tmp_path, ["app"])
        (tmp_path / "app" / "README.md").write_text("other docs")
        (tmp_path / "app" / "__pycache__").mkdir()
        (tmp_path / "app" / "__pycache__" / "server.py").write_text("cached")
        assert sources_digest(tmp_path, ["app"]) == first
        (tmp_path / "app" / "server.py").write_text("x = 2\n")
        assert sources_digest(tmp_path, ["app"]) != first


class TestCheckBeforeLaunch:
    """`sieval run` checks a service can start before it starts any engine."""

    def test_a_bad_dir_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=CODE_EVAL_DIR_ENV):
            check_code_eval({CODE_EVAL_DIR_ENV: str(tmp_path)})

    def test_an_interpreter_that_cannot_run_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match=CODE_EVAL_PYTHON_ENV):
            check_code_eval({CODE_EVAL_PYTHON_ENV: str(tmp_path / "no-python")})

    def test_a_working_setup_passes(self) -> None:
        check_code_eval({})

    def test_a_service_with_its_endpoint_set_is_not_checked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(CODE_EVAL_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(_ENDPOINT_ENV, "http://evaluator/evaluations")
        check_vendored_services({CODE_EVAL})
        monkeypatch.delenv(_ENDPOINT_ENV)
        with pytest.raises(FileNotFoundError):
            check_vendored_services({CODE_EVAL})


# ===================================================================
# Through a whole session
# ===================================================================
class CodeEvalMockTask(MockTask):
    """Posts each answer to the code-eval endpoint, like the code tasks do."""

    services = frozenset({CODE_EVAL})

    def __init__(self, dataset, model, name=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._api = os.environ[_ENDPOINT_ENV]

    async def feedback(self, post, ctx):
        async with httpx.AsyncClient() as client:
            resp = await client.post(self._api, json={"code": post})
        assert resp.status_code == 200
        return await super().feedback(post, ctx)


_CONFIG = """\
result_dir: "{result_dir}"
models:
  mock_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"
datasets:
  ds:
    class: tests.conftest.MockDataset
tasks:
{tasks}
"""

_TASK = """\
  {name}:
    class: {cls}
    dataset: ds
    model: mock_model
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
"""

_CODE_TASK = "tests.unit.cli.test_services.CodeEvalMockTask"
_PLAIN_TASK = "tests.unit.core.runners.test_runner.MockTask"


def _write_config(tmp_path: Path, tasks: dict[str, str]) -> tuple[Path, Path]:
    result_dir = tmp_path / "out"
    body = "".join(_TASK.format(name=n, cls=c) for n, c in tasks.items())
    path = tmp_path / "cfg.yaml"
    path.write_text(_CONFIG.format(result_dir=result_dir, tasks=body))
    return path, result_dir


def _grader(result_dir: Path, task: str) -> dict:
    """The ``feedback`` identity, which records the services a task calls."""
    meta = orjson.loads((result_dir / task / "meta.json").read_bytes())
    return meta["identity"]["feedback"]


class TestSession:
    @pytest.mark.anyio
    async def test_two_tasks_share_one_service_and_record_it(
        self, stub: Path, tmp_path: Path
    ) -> None:
        config, result_dir = _write_config(
            tmp_path, {"a": _CODE_TASK, "b": _CODE_TASK, "plain": _PLAIN_TASK}
        )
        reports = await arun_session(config)

        assert set(reports) == {"a", "b", "plain"}
        [pid] = _pids(stub)
        assert not pid_alive(pid)
        assert _ENDPOINT_ENV not in os.environ
        assert (result_dir / "logs" / "services" / CODE_EVAL / "server.log").exists()
        for task in ("a", "b"):
            assert _grader(result_dir, task)["services"] == [
                {
                    "name": CODE_EVAL,
                    "source": "vendored",
                    "commit": "e4802268",
                    "digest": "sha256:stub",
                    "runtime": _RUNTIME,
                }
            ]
        assert "services" not in _grader(result_dir, "plain")

    @pytest.mark.anyio
    async def test_a_preset_endpoint_is_used_and_recorded_as_external(
        self, stub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The sandbox is "external": run the stub by hand and point at it.
        port = services._free_port()
        proc = subprocess.Popen(
            [sys.executable, str(tmp_path / "stub_server.py"), str(port), str(stub)]
        )
        try:
            external = f"http://127.0.0.1:{port}/evaluations"
            async with httpx.AsyncClient() as client:
                for _ in range(100):
                    with contextlib.suppress(httpx.TransportError):
                        await client.get(f"http://127.0.0.1:{port}/health")
                        break
                    await anyio.sleep(0.05)
            monkeypatch.setenv(_ENDPOINT_ENV, external)
            config, result_dir = _write_config(tmp_path, {"a": _CODE_TASK})
            await arun_session(config)
        finally:
            proc.terminate()
            proc.wait()
        assert len(_pids(stub)) == 1  # only the one started by hand
        grader = _grader(result_dir, "a")
        assert grader["services"] == [{"name": CODE_EVAL, "source": "external"}]
        assert external not in json.dumps(grader)

    @pytest.mark.anyio
    async def test_a_task_named_logs_is_refused(
        self, stub: Path, tmp_path: Path
    ) -> None:
        config, _ = _write_config(tmp_path, {"logs": _PLAIN_TASK})
        with pytest.raises(ValueError, match="'logs' is reserved"):
            await arun_session(config)
        assert _pids(stub) == []
