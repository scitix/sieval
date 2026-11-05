"""Pattern: ``sieval run --dry-run`` reads datasets light by default, and
``--full-datasets`` loads them as the run does. Both report the same plan.

The config is a fixture board: two local parquet datasets loaded through
``load_dataset``, one sliced by an operation, a task on each, and an
``api_base`` model, so nothing is launched. Each dry-run gets a fresh
``HF_DATASETS_CACHE``, as a Harmony config-check job does.

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
                "class": f"{here}.AskTask",
                "dataset": name,
                "model": "api",
                "runner_config": {"show_progress": False},
            }
            for name in ("small", "whole")
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


class TestDryRunDatasets:
    def test_light_and_full_report_the_same_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = _config(tmp_path)
        light = _dry_run(config_path, tmp_path / "light", monkeypatch)
        full = _dry_run(config_path, tmp_path / "full", monkeypatch, "--full-datasets")

        assert light.pop("datasets") == {
            "mode": "light",
            "read": {
                name: {"read": "memory", "unverified": ["arrow_cache_write"]}
                for name in ("small", "whole")
            },
        }
        assert full.pop("datasets") == {"mode": "full"}
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
        _dry_run(config_path, tmp_path / "full", monkeypatch, "--full-datasets")
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
        for mode, flags in (("light", ()), ("full", ("--full-datasets",))):
            monkeypatch.setattr(config, "HF_DATASETS_CACHE", tmp_path / mode)
            result = runner.invoke(
                app, ["run", str(config_path), "--dry-run", *flags, "-o", "json"]
            )
            assert result.exit_code == 1
            data = json.loads(result.stdout)["data"]
            (check,) = [c for c in data["checks"] if c["name"] == "resume_plan"]
            details[mode] = check["detail"]
            assert data["datasets"]["mode"] == mode
        assert details["light"] == details["full"]

    def test_full_datasets_needs_dry_run(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", str(_config(tmp_path)), "--full-datasets"])
        assert result.exit_code == 2
        assert "--dry-run only" in result.output
