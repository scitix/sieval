"""Result directories written by released SiEval resume under the next release.

The fixtures are ``human_eval_0shot_gen`` task directories that SiEval 0.10.0,
0.10.2 and 0.11.0 wrote for five HumanEval problems against a fake model.
``human_eval-5.jsonl`` holds those five problems, so the data identity they
recorded matches. All three recorded identity schema 1; a resume reads it and
writes schema 4, rendering the five recorded prompts once to check that
``preprocess`` is unchanged.

The grading tests point ``SIEVAL_CODE_EVAL_API`` at a fake code evaluator and
declare it the vendored service each fixture recorded, so the service identity
matches and only what a test changes differs. The evaluator passes the one
correct answer, as the real one did, and counts what it is asked.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import shutil
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openai._base_client
import orjson
import pytest
import yaml
from typer.testing import CliRunner

import sieval
import sieval.cli.session
import sieval.core.runners.runner
import sieval.core.utils.meta
from sieval.cli.main import app
from sieval.core.runners import task_resume_plan
from sieval.core.runners.runner import gate_resume_version
from sieval.core.tasks.run_meta import (
    GRADING_STAGES,
    IDENTITY_SCHEMA,
    RESULT_SCHEMA_VERSION,
    is_code_identity,
)
from sieval.core.tasks.services import VENDORED_SERVICES_ENV
from sieval.tasks import HumanEvalZeroShotGenTask

FIXTURES = Path(__file__).parent / "fixtures"
TASK = "human_eval_0shot_gen"
RELEASES = ("0.10.0", "0.10.2", "0.11.0")
NEXT = "0.12.0"

#: The only fixture answer that passes its tests.
_PASSING = "return any(abs(a - b) < threshold"


def _copy(release: str, tmp_path: Path) -> Path:
    root = tmp_path / "result"
    shutil.copytree(FIXTURES / f"result-dir-{release}" / TASK, root / TASK)
    return root


def _meta(root: Path) -> dict:
    return orjson.loads((root / TASK / "meta.json").read_bytes())


@pytest.mark.parametrize("release", RELEASES)
@pytest.mark.parametrize("current", [NEXT, f"{NEXT}+gabc1234", "0.12.1.dev3"])
def test_the_gate_admits_a_released_directory(release, current, tmp_path):
    root = _copy(release, tmp_path)
    recorded = _meta(root).get("result_schema_version")
    assert recorded in (None, RESULT_SCHEMA_VERSION)
    gate_resume_version(root / TASK, current)  # no raise


class _Evaluator:
    """A code evaluator on a free port that counts the programs it is sent."""

    def __init__(self) -> None:
        self.programs: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002 - the stdlib's name
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.programs.append(body["code"])
                passed = _PASSING in body["code"]
                reply = {
                    "status": passed,
                    "msg": "" if passed else "failed: [AssertionError] ",
                    "data": {"n_cases": 1, "n_passed": int(passed)},
                }
                payload = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}/evaluations"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def next_release(monkeypatch, tmp_path):
    pytest.importorskip("fastapi", reason="needs the code-eval dependency group")
    for module in (sieval, sieval.cli.session, sieval.core.runners.runner):
        monkeypatch.setattr(module, "__version__", NEXT)
    monkeypatch.setattr(sieval.core.utils.meta, "__version__", NEXT)
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    for name in (
        "SIEVAL_CODE_EVAL_API",
        VENDORED_SERVICES_ENV,
        "SIEVAL_CODE_EVAL_DIR",
        "SIEVAL_CODE_EVAL_PYTHON",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def evaluator(monkeypatch):
    """Serve code-eval as the vendored copy each fixture recorded."""
    served = _Evaluator()

    def declare(recorded_service: dict) -> _Evaluator:
        provenance = {
            "code-eval": {
                "commit": recorded_service["commit"],
                "digest": recorded_service["digest"],
                "runtime": recorded_service["runtime"],
                "endpoint": served.url,
            }
        }
        monkeypatch.setenv("SIEVAL_CODE_EVAL_API", served.url)
        monkeypatch.setenv(VENDORED_SERVICES_ENV, json.dumps(provenance))
        return served

    yield declare
    served.close()


def _config(tmp_path: Path, args: dict | None = None) -> Path:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    shutil.copy(FIXTURES / "human_eval-5.jsonl", data / "test.jsonl")
    task: dict = {
        "class": "HumanEvalZeroShotGenTask",
        "dataset": "human_eval",
        "infer_args": {"max_tokens": 64},
        "model": "subject",
    }
    if args:
        task["args"] = args
    config = {
        # As the recorded run's effective_config.yaml, but for the dataset path
        # and key, which it redacted. Nothing listens at api_base: the
        # recorded inference is reused.
        "models": {
            "subject": {
                "name": "fake",
                "api_base": "http://127.0.0.1:18777/v1",
                "api_key": "unused",
                "args": {"concurrency_limit": 4, "temperature": 0.0},
            }
        },
        "datasets": {
            "human_eval": {
                "class": "HumanEvalDataset",
                "path": str(data),
                "operations": [{"slice": {"num": 5}}],
            }
        },
        "tasks": {TASK: task},
    }
    path = tmp_path / "sieval.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _resume(config: Path, root: Path) -> dict:
    result = CliRunner().invoke(
        app,
        ["eval", str(config), "--result-dir", str(root), "--resume", "-o", "json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
    return orjson.loads((root / TASK / "report.json").read_bytes())


def _recorded_service(release: str) -> dict:
    meta = orjson.loads(
        (FIXTURES / f"result-dir-{release}" / TASK / "meta.json").read_bytes()
    )
    [service] = meta["identity"]["grader"]["services"]
    return service


@pytest.mark.usefixtures("next_release")
@pytest.mark.parametrize("release", RELEASES)
class TestResumeAcrossReleases:
    def test_an_unchanged_task_keeps_every_verdict_and_upgrades_the_identity(
        self, release, tmp_path, evaluator
    ):
        served = evaluator(_recorded_service(release))
        root = _copy(release, tmp_path)
        recorded = _meta(root)
        assert recorded["identity"]["schema"] == 1

        report = _resume(_config(tmp_path), root)

        assert served.programs == []  # no sample was graded again
        meta = _meta(root)
        assert meta["version"] == release
        assert meta["result_schema_version"] == RESULT_SCHEMA_VERSION
        first, resumed = meta["runs"]
        assert first["sieval"] == release
        assert resumed["sieval"] == NEXT
        assert resumed["start"] == "resume"
        assert resumed["reran"] == []
        assert resumed["resume"]["kept"] == 5
        assert resumed["resume"]["repostprocessed"] == 0
        assert resumed["resume"]["restarted"] == 0
        identity = meta["identity"]
        assert identity["schema"] == IDENTITY_SCHEMA == 4
        assert "grader" not in identity and "prompt" not in identity
        old = recorded["identity"]
        assert identity["data"] == old["data"]
        # The rendered prompts became the code identity a fresh run writes.
        assert is_code_identity(identity["preprocess"])
        assert identity["preprocess"] != old["prompt"]
        # Matched by the digest schema 1 wrote; rewritten in schema 4's, which
        # leaves serving out, with the requests the recorded run sent.
        assert identity["inference"]["digest"] != old["inference"]["digest"]
        assert identity["inference"]["requests"] == old["inference"]["requests"]
        assert "serving" in resumed
        assert all(identity[stage] is not None for stage in GRADING_STAGES)
        assert identity["feedback"]["services"] == old["grader"]["services"]
        assert report["score"] == 20.0
        assert report["sieval"]["final"] == 5
        # The reused report gains the count a 0.10.x report never wrote.
        assert report["sieval"]["context_overflow"] == 0

    def test_the_upgraded_block_is_what_a_fresh_run_writes(
        self, release, tmp_path, evaluator
    ):
        """A schema-2 block upgraded in place equals the one computed for a
        fresh run of the same task, stage by stage."""
        evaluator(_recorded_service(release))
        root = _copy(release, tmp_path)
        _resume(_config(tmp_path), root)
        upgraded = _meta(root)["identity"]

        task = HumanEvalZeroShotGenTask(dataset=_dataset(), model=_model(), name=TASK)
        from sieval.core.runners.runner import TaskRunner, TaskRunnerConfig

        runner = TaskRunner(
            task,
            TaskRunnerConfig(result_dir=str(tmp_path / "fresh"), show_progress=False),
        )
        fresh = runner._grading_identities()
        assert fresh is not None
        for stage in GRADING_STAGES:
            assert upgraded[stage] == fresh[stage], stage
        assert upgraded["preprocess"] == runner._preprocess_identity()

    def test_a_changed_prompt_restarts_every_sample(
        self, release, tmp_path, evaluator, monkeypatch
    ):
        """The recorded prompts no longer render: preprocess changed, so the
        upgrade restarts every sample instead of keeping its inference."""
        evaluator(_recorded_service(release))
        root = _copy(release, tmp_path)
        _patch_preprocess(tmp_path, monkeypatch)
        # The refused connections retry; skip the SDK's backoff, not the retries.
        monkeypatch.setattr(
            openai._base_client.BaseClient, "_calculate_retry_timeout", lambda *_: 0
        )
        result = CliRunner().invoke(
            app,
            [
                "eval",
                str(_config(tmp_path)),
                "--result-dir",
                str(root),
                "--resume",
                "-o",
                "json",
            ],
        )
        # Nothing serves the model, so the restarted samples fail at infer.
        assert result.exception is None, result.output
        resumed = _meta(root)["runs"][-1]
        assert resumed["resume"]["restarted"] == 5
        assert resumed["resume"]["kept"] == 0

    def test_a_changed_grader_regrades(self, release, tmp_path, evaluator):
        served = evaluator(_recorded_service(release))
        root = _copy(release, tmp_path)

        report = _resume(_config(tmp_path, args={"timeout": 4.0}), root)

        # Every sample regraded on its recorded inference; none inferred again.
        assert len(served.programs) == 5
        resumed = _meta(root)["runs"][-1]
        assert resumed["reran"] == ["postprocess", "feedback", "report"]
        assert resumed["resume"]["repostprocessed"] == 5
        identity = _meta(root)["identity"]
        assert identity["schema"] == IDENTITY_SCHEMA
        assert identity["feedback"]["args"]["timeout"] == 4.0
        # `timeout` is a grading argument, never read by preprocess.
        assert "timeout" not in identity["preprocess"]["args"]
        assert resumed["resume"]["restarted"] == 0
        assert report["score"] == 20.0

    def test_a_changed_report_rereports_only(
        self, release, tmp_path, evaluator, monkeypatch
    ):
        """After the upgrade, a change to `report` alone recomputes
        `report.json` from the records: no code-eval call, no model call."""
        served = evaluator(_recorded_service(release))
        root = _copy(release, tmp_path)
        config = _config(tmp_path)
        _resume(config, root)

        _patch_report(tmp_path, monkeypatch)
        report = _resume(config, root)

        assert served.programs == []
        assert report["patched"] is True
        assert report["score"] == 20.0
        resumed = _meta(root)["runs"][-1]
        assert resumed["reran"] == ["report"]
        assert resumed["resume"]["kept"] == 5
        assert resumed["resume"]["rereported"] == 5
        assert resumed["resume"]["regraded"] == 0
        assert resumed["resume"]["repostprocessed"] == 0
        assert "infer" not in (resumed.get("requests") or {})
        identity = _meta(root)["identity"]
        assert any("report_patch" in key for key in identity["report"]["modules"])
        assert not any("report_patch" in key for key in identity["feedback"]["modules"])

    def test_the_dry_run_plan_names_what_a_resume_does(self, release, tmp_path):
        recorded = orjson.loads(
            (FIXTURES / f"result-dir-{release}" / TASK / "meta.json").read_bytes()
        )
        task = HumanEvalZeroShotGenTask(dataset=_dataset(), model=_model(), name=TASK)
        plan = task_resume_plan(
            task,
            recorded,
            runner_keys={"deterministic": False, "max_iterations": 1, "max_retries": 3},
        )
        assert plan["identity_schema"] == 1
        # No vendored service is declared here, so the grader differs.
        assert plan["legacy_grader"] == "regrade"
        assert plan["rerun_from"] == "postprocess"
        assert plan["stages"]["postprocess"] == "differs"


def _patch_report(tmp_path: Path, monkeypatch) -> None:
    _patch(
        tmp_path,
        monkeypatch,
        "report",
        """
        async def report(self, finals, fails):
            return {**(await _original(self, finals, fails)), "patched": True}
        """,
    )


def _patch_preprocess(tmp_path: Path, monkeypatch) -> None:
    _patch(
        tmp_path,
        monkeypatch,
        "preprocess",
        """
        async def preprocess(self, raw, ctx):
            record = await _original(self, raw, ctx)
            record["prompt"][0]["content"] += "\\n"
            return record
        """,
    )


def _patch(tmp_path: Path, monkeypatch, stage: str, body: str) -> None:
    """Replace *stage* from a package on disk, as a plugin patch does.

    A function defined in this test module would make the stage's identity
    walk this module and everything it imports, the CLI included.
    """
    package = f"{stage}_patch"
    root = tmp_path / "patches"
    (root / package).mkdir(parents=True)
    (root / package / "__init__.py").write_text(
        "from sieval.tasks import HumanEvalZeroShotGenTask\n\n"
        f"_original = HumanEvalZeroShotGenTask.{stage}\n\n" + textwrap.dedent(body)
    )
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delitem(sys.modules, package, raising=False)
    import importlib

    patch = importlib.import_module(package)
    monkeypatch.setattr(HumanEvalZeroShotGenTask, stage, getattr(patch, stage))


def _dataset():
    """The five fixture problems; the grading identities never read them."""
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict as HFDatasetDict

    from sieval.datasets import HumanEvalDataset

    rows = [json.loads(line) for line in (FIXTURES / "human_eval-5.jsonl").open()]
    return HumanEvalDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list(rows)}))


def _model():
    from tests.conftest import MockChatModel

    return MockChatModel()
