"""Result directories written by released SiEval resume under the next release.

The fixtures are ``human_eval_0shot_gen`` task directories that SiEval 0.10.0
and 0.10.2 wrote for five HumanEval problems against a fake model, before
``result_schema_version`` existed. ``human_eval-5.jsonl`` holds those five
problems, so the data identity they recorded matches.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import shutil
from pathlib import Path

import orjson
import pytest
import yaml
from typer.testing import CliRunner

import sieval
import sieval.cli.session
import sieval.core.runners.runner
import sieval.core.utils.meta
from sieval.cli.main import app
from sieval.core.runners.runner import gate_resume_version
from sieval.core.tasks.run_meta import RESULT_SCHEMA_VERSION

FIXTURES = Path(__file__).parent / "fixtures"
TASK = "human_eval_0shot_gen"
RELEASES = ("0.10.0", "0.10.2")
NEXT = "0.11.0"


def _copy(release: str, tmp_path: Path) -> Path:
    root = tmp_path / "result"
    shutil.copytree(FIXTURES / f"result-dir-{release}" / TASK, root / TASK)
    return root


@pytest.mark.parametrize("release", RELEASES)
@pytest.mark.parametrize("current", [NEXT, f"{NEXT}+gabc1234", "0.11.1.dev3"])
def test_the_gate_admits_a_released_directory(release, current, tmp_path):
    root = _copy(release, tmp_path)
    assert "result_schema_version" not in orjson.loads(
        (root / TASK / "meta.json").read_bytes()
    )
    gate_resume_version(root / TASK, current)  # no raise


@pytest.mark.parametrize("release", RELEASES)
def test_eval_resumes_a_released_directory_under_the_next_release(
    release, tmp_path, monkeypatch
):
    pytest.importorskip("fastapi", reason="needs the code-eval dependency group")
    pytest.importorskip("uvicorn", reason="needs the code-eval dependency group")
    for module in (sieval, sieval.cli.session, sieval.core.runners.runner):
        monkeypatch.setattr(module, "__version__", NEXT)
    monkeypatch.setattr(sieval.core.utils.meta, "__version__", NEXT)
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    for name in (
        "SIEVAL_CODE_EVAL_API",
        "SIEVAL_VENDORED_SERVICES",
        "SIEVAL_CODE_EVAL_DIR",
        "SIEVAL_CODE_EVAL_PYTHON",
    ):
        monkeypatch.delenv(name, raising=False)

    root = _copy(release, tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(FIXTURES / "human_eval-5.jsonl", data / "test.jsonl")
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
        "tasks": {
            TASK: {
                "class": "HumanEvalZeroShotGenTask",
                "dataset": "human_eval",
                "infer_args": {"max_tokens": 64},
                "model": "subject",
            }
        },
    }
    config_path = tmp_path / "sieval.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(config_path),
            "--result-dir",
            str(root),
            "--resume",
            "-o",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
    meta = orjson.loads((root / TASK / "meta.json").read_bytes())
    assert meta["version"] == release
    assert meta["result_schema_version"] == RESULT_SCHEMA_VERSION
    first, resumed = meta["runs"]
    assert first["sieval"] == release
    assert resumed["sieval"] == NEXT
    assert resumed["start"] == "resume"
    # Inference is reused, never rerun; nothing listens at api_base.
    assert "inference" not in resumed["reran"]
    report = orjson.loads((root / TASK / "report.json").read_bytes())
    assert report["score"] == 20.0
    assert report["sieval"]["final"] == 5
    # The reused report gains the count a 0.10.x report never wrote.
    assert report["sieval"]["context_overflow"] == 0
