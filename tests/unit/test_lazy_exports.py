"""
Unit tests for sieval/core/lazy_exports.py.

Covers: Lazy package imports, stub export synchronization.

AI-Generated Code - GPT-5.3-Codex (OpenAI)
"""

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import ModuleIsolation


def _drop_package_modules(package_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def _preserve_registries():
    """Snapshot/restore the task + dataset registries *and* both packages'
    module caches around every test — see `ModuleIsolation` for why the two
    must move together.

    Tests here re-import sieval.tasks / sieval.datasets submodules, which
    re-invokes @sieval_task / @sieval_dataset and trips the duplicate-name
    guards; clearing the registries first lets the re-import proceed. The
    packages themselves are in scope (not just their submodules) because these
    tests replace the package objects to exercise the lazy `__getattr__`.
    Tasks pull datasets via `from sieval.datasets import ...`, so a task-side
    re-import only works if the dataset side is purged too.
    """
    try:
        from sieval.core.datasets.meta import (
            DATASET_REGISTRY,
            SAMPLE_TO_DATASET,
        )
        from sieval.core.tasks.meta import _TASK_CLASSES, TASK_REGISTRY
    except ImportError:
        yield
        return
    task_snapshot = dict(TASK_REGISTRY)
    task_classes_snapshot = dict(_TASK_CLASSES)
    dataset_snapshot = dict(DATASET_REGISTRY)
    sample_map_snapshot = dict(SAMPLE_TO_DATASET)
    modules = ModuleIsolation(
        ("sieval.tasks", "sieval.tasks.", "sieval.datasets", "sieval.datasets.")
    )
    modules.snapshot()

    TASK_REGISTRY.clear()
    _TASK_CLASSES.clear()
    DATASET_REGISTRY.clear()
    SAMPLE_TO_DATASET.clear()
    modules.evict()
    try:
        yield
    finally:
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(task_snapshot)
        _TASK_CLASSES.clear()
        _TASK_CLASSES.update(task_classes_snapshot)
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(dataset_snapshot)
        SAMPLE_TO_DATASET.clear()
        SAMPLE_TO_DATASET.update(sample_map_snapshot)
        modules.restore()


def _read_stub_all(stub_path: Path) -> list[str]:
    module_ast = ast.parse(
        stub_path.read_text(encoding="utf-8"),
        filename=str(stub_path),
    )
    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return list(ast.literal_eval(node.value))
    raise AssertionError(f"{stub_path} does not define __all__")


@pytest.mark.parametrize(
    ("package_name", "sample_export"),
    [
        ("sieval.tasks", "AIME2024ZeroShotGenTask"),
        ("sieval.datasets", "AIME2024Dataset"),
    ],
)
def test_package_import_is_lazy(package_name: str, sample_export: str) -> None:
    _drop_package_modules(package_name)

    package = importlib.import_module(package_name)
    assert sample_export in package.__all__
    # Verify laziness: no submodules should be loaded yet
    assert not any(name.startswith(f"{package_name}.") for name in sys.modules)
    # Verify the export is actually accessible (triggers lazy import)
    obj = getattr(package, sample_export)
    assert obj is not None, f"{sample_export} resolved to None"
    assert isinstance(obj, type), f"{sample_export} should be a class, got {type(obj)}"


@pytest.mark.parametrize(
    ("package_name", "sample_export", "sample_module"),
    [
        (
            "sieval.tasks",
            "AIME2024ZeroShotGenTask",
            "aime_2024_0shot_gen",
        ),
        ("sieval.datasets", "AIME2024Dataset", "aime_2024"),
    ],
)
def test_access_export_triggers_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
    sample_export: str,
    sample_module: str,
) -> None:
    _drop_package_modules(package_name)
    package = importlib.import_module(package_name)

    dummy_module = ModuleType(f"{package_name}.{sample_module}")
    dummy_value = type(sample_export, (), {})
    setattr(dummy_module, sample_export, dummy_value)

    calls: list[str] = []

    def fake_import(module_name: str) -> ModuleType:
        calls.append(module_name)
        return dummy_module

    monkeypatch.setattr(package.importlib, "import_module", fake_import)

    resolved = getattr(package, sample_export)
    assert resolved is dummy_value
    assert calls == [f"{package_name}.{sample_module}"]
    assert package.__dict__[sample_export] is dummy_value


@pytest.mark.parametrize("package_name", ["sieval.tasks", "sieval.datasets"])
def test_stub_exports_match_runtime_exports(package_name: str) -> None:
    package = importlib.import_module(package_name)
    package_file = package.__file__
    assert package_file is not None
    stub_exports = _read_stub_all(Path(package_file).with_suffix(".pyi"))
    assert stub_exports == package.__all__
