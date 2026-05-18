"""Tests for scripts/sync_meta_index.py.

Verifies default-mode writes, `--check` pass, and `--check` fail behavior.
The script imports from ``sieval.core.tasks.meta`` and ``sieval.core.datasets.meta``
and walks the ``sieval.tasks`` / ``sieval.datasets`` packages to populate the
registries; tests monkeypatch ``import_all_tasks`` and ``import_all_datasets``
to no-ops and pre-populate the registries with controlled dummy entries.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import TypedDict

import pytest

from sieval.core.datasets import Dataset
from sieval.core.datasets.meta import (
    DATASET_REGISTRY,
    SAMPLE_TO_DATASET,
    Category,
    Level1Category,
    sieval_dataset,
)
from sieval.core.tasks import Task
from sieval.core.tasks.meta import TASK_REGISTRY, EvalMode, sieval_task

# Stub sample types — each must be a distinct TypedDict so the dataset
# registry does not raise a duplicate-sample-type error across tests.


class _SampleTask(TypedDict):
    text: str


class _SampleAnother(TypedDict):
    text: str


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "sync_meta_index.py"


def _load_script():
    """Import scripts/sync_meta_index.py as a module (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(
        "_sync_meta_index_test_module", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_registry(monkeypatch):
    """Clear both registries for the test, restore after.

    Also stubs ``import_all_tasks`` and ``import_all_datasets`` so the real
    packages are not walked.
    """
    task_snapshot = dict(TASK_REGISTRY)
    dataset_snapshot = dict(DATASET_REGISTRY)
    sample_snapshot = dict(SAMPLE_TO_DATASET)

    TASK_REGISTRY.clear()
    DATASET_REGISTRY.clear()
    SAMPLE_TO_DATASET.clear()

    script = _load_script()
    monkeypatch.setattr(script, "import_all_tasks", lambda: None)
    monkeypatch.setattr(script, "import_all_datasets", lambda: None)

    @sieval_dataset(
        name="sample_dataset",
        display_name="Sample Dataset",
        description="Sample dataset for testing.",
        source="hf:x/y",
        categories=(Category(Level1Category.MATHEMATICS),),
    )
    class _SampleDataset(Dataset[_SampleTask]):
        def load(self, name_or_path, **kwargs):
            raise NotImplementedError

    @sieval_task(
        name="sample_task",
        display_name="Sample",
        description="sample",
        eval_mode=EvalMode.GEN,
    )
    class _Sample(Task[_SampleTask, None, None, None, None, dict[str, float]]):
        pass

    yield script

    TASK_REGISTRY.clear()
    TASK_REGISTRY.update(task_snapshot)
    DATASET_REGISTRY.clear()
    DATASET_REGISTRY.update(dataset_snapshot)
    SAMPLE_TO_DATASET.clear()
    SAMPLE_TO_DATASET.update(sample_snapshot)


def test_default_mode_writes_schema_and_sorted_tasks(
    tmp_path, monkeypatch, isolated_registry, capsys
):
    """Default mode writes schema_version, datasets, and tasks sorted by name."""
    script = isolated_registry

    # Register a second dataset + task so we can observe sort order.
    @sieval_dataset(
        name="another_dataset",
        display_name="Another Dataset",
        description="Another dataset for testing.",
        source="hf:a/b",
        categories=(Category(Level1Category.MATHEMATICS),),
    )
    class _AnotherDataset(Dataset[_SampleAnother]):
        def load(self, name_or_path, **kwargs):
            raise NotImplementedError

    @sieval_task(
        name="another_task",
        display_name="Another",
        description="another",
        eval_mode=EvalMode.GEN,
    )
    class _Another(Task[_SampleAnother, None, None, None, None, dict[str, float]]):
        pass

    out_path = tmp_path / "index.json"
    # Patch ROOT so the ``relative_to(ROOT)`` print path succeeds.
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "INDEX_PATH", out_path)
    monkeypatch.setattr(sys, "argv", ["sync_meta_index.py"])

    assert script.main() == 0

    data = json.loads(out_path.read_text())
    assert data["schema_version"] == 1

    # Both sections present and sorted by name.
    dataset_names = [entry["name"] for entry in data["datasets"]]
    assert dataset_names == ["another_dataset", "sample_dataset"]

    task_names = [entry["name"] for entry in data["tasks"]]
    assert task_names == ["another_task", "sample_task"]

    captured = capsys.readouterr()
    assert "2 dataset" in captured.out
    assert "2 task" in captured.out


def test_check_mode_passes_when_in_sync(tmp_path, monkeypatch, isolated_registry):
    """--check exits 0 when committed index matches rendered payload."""
    script = isolated_registry
    out_path = tmp_path / "index.json"
    monkeypatch.setattr(script, "INDEX_PATH", out_path)

    # Seed the file with the rendered payload.
    out_path.write_text(script.render_payload(), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["sync_meta_index.py", "--check"])
    assert script.main() == 0


def test_check_mode_fails_when_stale(tmp_path, monkeypatch, isolated_registry):
    """--check raises SystemExit with a remediation message when stale."""
    script = isolated_registry
    out_path = tmp_path / "index.json"
    monkeypatch.setattr(script, "INDEX_PATH", out_path)

    out_path.write_text(
        '{"schema_version": 1, "datasets": [], "tasks": []}\n', encoding="utf-8"
    )

    monkeypatch.setattr(sys, "argv", ["sync_meta_index.py", "--check"])
    with pytest.raises(SystemExit) as exc_info:
        script.main()

    message = str(exc_info.value)
    assert "out of date" in message.lower()
    assert "sync_meta_index.py" in message


def test_check_mode_fails_when_file_missing(tmp_path, monkeypatch, isolated_registry):
    """--check flags a missing file as stale (empty current != rendered)."""
    script = isolated_registry
    out_path = tmp_path / "index.json"
    assert not out_path.exists()
    monkeypatch.setattr(script, "INDEX_PATH", out_path)

    monkeypatch.setattr(sys, "argv", ["sync_meta_index.py", "--check"])
    with pytest.raises(SystemExit):
        script.main()


def test_payload_has_datasets_and_tasks_with_fk():
    """Payload includes both sections; tasks have dataset FK, no removed fields."""
    from scripts.sync_meta_index import render_payload

    payload = json.loads(render_payload())
    assert payload["schema_version"] == 1
    assert "datasets" in payload
    assert "tasks" in payload
    # Both sorted by name
    assert payload["datasets"] == sorted(payload["datasets"], key=lambda d: d["name"])
    assert payload["tasks"] == sorted(payload["tasks"], key=lambda d: d["name"])
    # Pilot smoke
    dataset_names = {d["name"] for d in payload["datasets"]}
    assert "aime_2024" in dataset_names
    task_names = {t["name"] for t in payload["tasks"]}
    assert "aime_2024_0shot_gen" in task_names
    # FK resolves
    aime_task = next(t for t in payload["tasks"] if t["name"] == "aime_2024_0shot_gen")
    assert aime_task["dataset"] == "aime_2024"
    # Old fields gone
    assert "dataset_source" not in aime_task
    assert "categories" not in aime_task
