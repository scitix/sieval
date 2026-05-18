"""
Lazy-loading registry for task classes.

AI-Generated Code - GPT-5.3-Codex (OpenAI)
"""

import ast
import importlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

_PACKAGE_DIR = Path(__file__).resolve().parent
_TASK_EXPORT_SUFFIX = "Task"


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
    """Return sorted subdirectories of the package that contain ``__init__.py``."""
    return sorted(
        path
        for path in _PACKAGE_DIR.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "__init__.py").exists()
    )


def _scan_task_classes(module_path: Path) -> list[str]:
    """Return public ``*Task`` class names defined in *module_path* (AST only)."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and not node.name.startswith("_")
        and node.name.endswith(_TASK_EXPORT_SUFFIX)
    ]


def _discover_task_exports() -> dict[str, str]:
    export_to_module: dict[str, str] = {}

    def _register(export_name: str, module_name: str) -> None:
        previous_module = export_to_module.get(export_name)
        if previous_module and previous_module != module_name:
            raise RuntimeError(
                f"Duplicate task export '{export_name}' found in "
                f"'{previous_module}' and '{module_name}'."
            )
        export_to_module[export_name] = module_name

    # 1) Flat .py modules
    for module_path in _iter_module_paths():
        for name in _scan_task_classes(module_path):
            _register(name, module_path.stem)

    # 2) Subpackage .py modules — mapped as "subpkg.module_stem"
    for subpkg_dir in _iter_subpackage_dirs():
        subpkg_name = subpkg_dir.name
        for module_path in _iter_module_paths_in(subpkg_dir):
            for name in _scan_task_classes(module_path):
                qualified = f"{subpkg_name}.{module_path.stem}"
                _register(name, qualified)

    return export_to_module


def _build_export_map() -> Mapping[str, str]:
    return MappingProxyType(_discover_task_exports())


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
