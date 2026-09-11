"""Route-level tests for ``POST /code-runs``, through the HTTP layer.

Covers what only the FastAPI route adds on top of ``exec_py_run.execute_run``:
the response shape (field set, ``service_version``, the always-null
``session_id``), the truncation flag on an oversized stream, and the refusal
of any non-python ``lang`` before anything executes. Execution semantics --
stdout capture, traceback-on-raise, partial stdout surviving a later raise,
timeout reporting, statelessness -- are tested directly against
``execute_run`` in ``test_exec_py_run.py`` instead, one layer down.

``fastapi`` and ``httpx`` are already pinned in ``requirements.txt`` for the
service itself, so ``TestClient`` adds no new dependency here.

Run from ``vendor/code-evaluator``:

    python -m pytest tests/ -q

AI-Generated Code - Claude Sonnet 5 (Anthropic)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402

client = TestClient(app)


def _post(code: str, **kw):
    body = {"uuid": "t1", "lang": "python", "code": code}
    body.update(kw)
    return client.post("/code-runs", json=body).json()


def test_response_shape_on_a_clean_run():
    out = _post("print(6 * 7)")
    assert out["status"] is True
    assert out["msg"] == ""
    data = out["data"]
    assert data["stdout"].strip() == "42"
    assert data["stderr"] == ""
    assert data["exit_code"] == 0
    assert data["timed_out"] is False
    assert data["truncated"] is False
    assert isinstance(data["wall_s"], float)
    # Reserved and stateless in this version -- see README's Code-run section.
    assert data["session_id"] is None
    assert data["service_version"] == "code-runs/1"


def test_stdout_is_capped_and_flagged():
    out = _post("print('x' * 20000)")
    assert len(out["data"]["stdout"]) <= 8192
    assert out["data"]["truncated"] is True


def test_non_python_lang_is_refused_before_executing():
    out = _post("print(1)", lang="javascript")
    assert out["status"] is False
    assert "python" in out["msg"]
    assert "javascript" in out["msg"]
    # Refused before anything ran: no partial result to report, unlike a
    # snippet that ran and raised.
    assert out["data"] is None
