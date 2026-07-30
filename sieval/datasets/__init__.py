"""
Lazy-loading registry for dataset classes.

AI-Generated Code - GPT-5.3-Codex (OpenAI)
"""

import ast
import importlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_EXPORT_SUFFIXES = ("Dataset", "DatasetSample", "CSVSample")


def _iter_module_paths_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix == ".py"
        and path.name != "__init__.py"
        and not path.name.startswith("_")
    )


def _iter_module_paths() -> list[Path]:
    return _iter_module_paths_in(_PACKAGE_DIR)


def _iter_subpackage_dirs() -> list[Path]:
    """Return sorted subdirectories that contain ``__init__.py``."""
    return sorted(
        path
        for path in _PACKAGE_DIR.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "__init__.py").exists()
    )


def _is_export_name(name: str) -> bool:
    return not name.startswith("_") and name.endswith(_DATASET_EXPORT_SUFFIXES)


def _is_typeddict_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "TypedDict"
    if isinstance(func, ast.Attribute):
        return func.attr == "TypedDict"
    return False


def _scan_dataset_exports(module_path: Path) -> list[str]:
    """Return public ``_DATASET_EXPORT_SUFFIXES``-suffixed classes and ``TypedDict``
    assignments defined in *module_path* (AST only)."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))

    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_export_name(node.name):
            names.append(node.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_export_name(node.targets[0].id)
            and _is_typeddict_call(node.value)
        ):
            names.append(node.targets[0].id)
    return names


def _discover_dataset_exports() -> dict[str, str]:
    """Map each export name to the module ``__getattr__`` should import it from.

    ``scripts/sync_package_stubs.py`` reimplements this scan deliberately — the stub
    generator must not import the package it generates stubs for — so the two have to
    agree for the generated ``.pyi`` to match runtime resolution.
    """
    export_to_module: dict[str, str] = {}

    def _register(export_name: str, module_name: str) -> None:
        previous_module = export_to_module.get(export_name)
        if previous_module and previous_module != module_name:
            raise RuntimeError(
                f"Duplicate dataset export '{export_name}' found in "
                f"'{previous_module}' and '{module_name}'."
            )
        export_to_module[export_name] = module_name

    # 1) Flat .py modules
    for module_path in _iter_module_paths():
        for name in _scan_dataset_exports(module_path):
            _register(name, module_path.stem)

    # 2) Subpackage .py modules — "subpkg.module_stem", matching sieval/tasks/
    # __init__.py. Resolving to the defining module keeps the registry independent of
    # what the subpackage's __init__ re-exports, and keeps _register's guard effective
    # within a subpackage (the bare subpackage name made same-named exports collide
    # silently).
    for subpkg_dir in _iter_subpackage_dirs():
        for module_path in _iter_module_paths_in(subpkg_dir):
            for name in _scan_dataset_exports(module_path):
                _register(name, f"{subpkg_dir.name}.{module_path.stem}")

    return export_to_module


def _build_export_map() -> Mapping[str, str]:
    return MappingProxyType(_discover_dataset_exports())


_EXPORT_TO_MODULE: Mapping[str, str] = _build_export_map()
__all__ = sorted(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> type[object]:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.{module_name}")
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module.__name__!r} does not define {name!r}"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
