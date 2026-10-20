"""The vendored code-evaluator, started by SiEval and graded against for real.

Needs the ``code-eval`` dependency group, which CI installs; skipped without it.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
from pathlib import Path

import httpx
import pytest

from sieval.cli.services import vendored_services_for
from sieval.core.tasks.services import (
    CODE_EVAL,
    SERVICE_ENDPOINT_ENV,
    VENDORED_SERVICES_ENV,
)
from sieval.infer.backends.process import pid_alive

pytest.importorskip("fastapi", reason="needs the code-eval dependency group")
pytest.importorskip("uvicorn", reason="needs the code-eval dependency group")

_ENDPOINT_ENV = SERVICE_ENDPOINT_ENV[CODE_EVAL]


@pytest.mark.anyio
async def test_sieval_starts_grades_with_and_stops_the_vendored_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    monkeypatch.delenv(_ENDPOINT_ENV, raising=False)
    monkeypatch.delenv(VENDORED_SERVICES_ENV, raising=False)
    monkeypatch.delenv("SIEVAL_CODE_EVAL_DIR", raising=False)
    monkeypatch.delenv("SIEVAL_CODE_EVAL_PYTHON", raising=False)

    async with vendored_services_for(
        [CODE_EVAL], result_dir=lambda: tmp_path
    ) as endpoints:
        url = endpoints[CODE_EVAL]
        assert os.environ[_ENDPOINT_ENV] == url
        [handle] = (tmp_path / "handles").glob("run-*/service-code-eval.json")
        pid = int(handle.read_text().split('"engine_pid": ')[1].split(",")[0])
        async with httpx.AsyncClient(timeout=30) as client:
            passed = await client.post(
                url,
                json={
                    "uuid": "pass",
                    "source": "human-eval",
                    "code": "assert sum([1, 2]) == 3",
                    "timeout": 5.0,
                },
            )
            failed = await client.post(
                url,
                json={
                    "uuid": "fail",
                    "source": "human-eval",
                    "code": "assert sum([1, 2]) == 4",
                    "timeout": 5.0,
                },
            )
        assert passed.json()["status"] is True
        assert failed.json()["status"] is False

    assert not pid_alive(pid)
    assert _ENDPOINT_ENV not in os.environ
    assert (tmp_path / "logs" / CODE_EVAL / "server.log").stat().st_size > 0
