"""Pattern: ``sieval run --dry-run`` is ``--mode light`` by default, and
``--mode full`` loads datasets and starts services as the run does. Both report
the same plan.

The config is a fixture board: two local parquet datasets loaded through
``load_dataset``, one sliced by an operation, a task on each (one declaring the
vendored code-eval service, stubbed) and an ``api_base`` model, so no engine is
launched. Each dry-run gets a fresh ``HF_DATASETS_CACHE``, as a Harmony
config-check job does.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from datasets import DatasetDict, config, load_dataset
from typer.testing import CliRunner

from sieval.cli import app
from sieval.core.datasets import Dataset
from sieval.core.tasks import Task
from sieval.core.tasks.services import CODE_EVAL
from tests.unit.cli.test_services import _pids, install_code_eval_stub

runner = CliRunner()


class ParquetDataset(Dataset):
    """A staged parquet repo, read as a registered loader reads one."""

    def load(self, name_or_path: str, **kwargs) -> DatasetDict:
        files = {
            split: [
                str(p) for p in sorted(Path(name_or_path).glob(f"{split}-*.parquet"))
            ]
            for split in ("test", "dev")
        }
        loaded = load_dataset("parquet", data_files=files, **kwargs)
        assert isinstance(loaded, DatasetDict)
        # Per-row work, as C-Eval's subject tag: one more cache file on disk.
        return DatasetDict(
            {k: v.map(lambda r: {"n": len(r["q"])}) for k, v in loaded.items()}
        )


class AskTask(Task):
    model_type = "chat"
    services = frozenset()

    async def preprocess(self, raw, ctx):
        return raw["q"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0]

    async def feedback(self, post, ctx):
        return True, {}

    async def report(self, finals, fails):
        return {"total": len(finals) + len(fails)}


class CodeAskTask(AskTask):
    """Declares code-eval, as the code tasks do; never calls it here."""

    services = frozenset({CODE_EVAL})


def _stage(root: Path, name: str, n: int) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    rows = [{"q": f"{name}-{i}", "a": i} for i in range(n)]
    half = n // 2
    pq.write_table(pa.Table.from_pylist(rows[:half]), repo / "test-0.parquet")
    pq.write_table(pa.Table.from_pylist(rows[half:]), repo / "test-1.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:2]), repo / "dev-0.parquet")
    return repo


def _config(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    here = "tests.integration.test_dry_run_datasets"
    cfg = {
        "result_dir": str(tmp_path / "results"),
        "models": {
            "api": {
                "name": "org/model",
                "api_base": "http://127.0.0.1:9/v1",
                "api_key": "k",
            }
        },
        "datasets": {
            "small": {
                "class": f"{here}.ParquetDataset",
                "path": str(_stage(data, "small", 40)),
                "operations": [{"slice": {"num": 3}}],
            },
            "whole": {
                "class": f"{here}.ParquetDataset",
                "path": str(_stage(data, "whole", 25)),
            },
        },
        "tasks": {
            name: {
                "class": f"{here}.{cls}",
                "dataset": name,
                "model": "api",
                "runner_config": {"show_progress": False},
            }
            for name, cls in (("small", "AskTask"), ("whole", "CodeAskTask"))
        },
    }
    path = tmp_path / "board.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _dry_run(
    config_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch, *flags: str
) -> dict[str, object]:
    monkeypatch.setattr(config, "HF_DATASETS_CACHE", cache)
    result = runner.invoke(
        app, ["run", str(config_path), "--dry-run", *flags, "-o", "json"]
    )
    assert result.exit_code == 0, result.output
    return cast(dict, json.loads(result.stdout)["data"])


def _arrow_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.arrow")) if root.exists() else []


@pytest.fixture(autouse=True)
def code_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """code-eval launched as a stub server; each start appends its PID."""
    return install_code_eval_stub(tmp_path, monkeypatch)


_BASE = ["present", "rows", "count", "schema"]


def _plain_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typer's usage errors unboxed and wide. Under GitHub Actions typer forces
    a colored, boxed error that splits the message."""
    import typer.rich_utils as rich_utils

    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", None, raising=False)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", None, raising=False)
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", 200, raising=False)


class TestDryRunDatasets:
    def test_light_and_full_report_the_same_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code_eval: Path
    ) -> None:
        config_path = _config(tmp_path)
        light = _dry_run(config_path, tmp_path / "light", monkeypatch)
        assert _pids(code_eval) == []
        full = _dry_run(config_path, tmp_path / "full", monkeypatch, "--mode", "full")
        assert len(_pids(code_eval)) == 1

        assert (light.pop("mode"), full.pop("mode")) == ("light", "full")
        assert light.pop("datasets") == {
            name: {
                "read": "memory",
                "verified": _BASE,
                "unverified": ["arrow_cache_write"],
            }
            for name in ("small", "whole")
        }
        assert full.pop("datasets") == {
            name: {
                "read": "cache",
                "verified": [*_BASE, "arrow_cache_write"],
                "unverified": [],
            }
            for name in ("small", "whole")
        }
        assert light.pop("services") == {
            CODE_EVAL: {
                "started": False,
                "verified": ["resolves", "command", "interpreter", "port"],
                "unverified": ["starts", "health", "task_setup"],
                "setup_unreached": [],
            }
        }
        assert full.pop("services") == {
            CODE_EVAL: {
                "started": True,
                "verified": [
                    "resolves",
                    "command",
                    "interpreter",
                    "port",
                    "starts",
                    "health",
                    "task_setup",
                ],
                "unverified": [],
                "setup_unreached": [],
            }
        }
        # Every other key, `tasks.<t>.samples` and the stage identities
        # included, as Harmony parses them.
        assert light == full
        tasks = cast(dict, light["tasks"])
        assert {name: tasks[name]["samples"] for name in tasks} == {
            "small": 3,
            "whole": 25,
        }

    def test_light_writes_no_arrow_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = _config(tmp_path)
        _dry_run(config_path, tmp_path / "light", monkeypatch)
        assert _arrow_files(tmp_path / "light") == []

    def test_full_loads_through_the_arrow_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = _config(tmp_path)
        _dry_run(config_path, tmp_path / "full", monkeypatch, "--mode", "full")
        # Each split's prepared file, then each `map` over it.
        assert len(_arrow_files(tmp_path / "full")) >= 8

    def test_a_missing_dataset_fails_alike(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = _config(tmp_path)
        cfg = yaml.safe_load(config_path.read_text())
        cfg["datasets"]["whole"]["path"] = str(tmp_path / "absent")
        config_path.write_text(yaml.safe_dump(cfg))

        details = {}
        for mode, flags in (("light", ()), ("full", ("--mode", "full"))):
            monkeypatch.setattr(config, "HF_DATASETS_CACHE", tmp_path / mode)
            result = runner.invoke(
                app, ["run", str(config_path), "--dry-run", *flags, "-o", "json"]
            )
            assert result.exit_code == 1
            data = json.loads(result.stdout)["data"]
            (check,) = [c for c in data["checks"] if c["name"] == "resume_plan"]
            details[mode] = check["detail"]
            assert data["mode"] == mode
            assert (data["datasets"], data["services"]) == (None, None)
        assert details["light"] == details["full"]

    def test_an_unknown_mode_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _plain_errors(monkeypatch)
        result = runner.invoke(
            app, ["run", str(_config(tmp_path)), "--dry-run", "--mode", "auto"]
        )
        assert result.exit_code == 2
        assert "known: light, full" in result.output
