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


def _iter_module_paths() -> list[Path]:
    return sorted(
        path
        for path in _PACKAGE_DIR.iterdir()
        if path.suffix == ".py"
        and path.name != "__init__.py"
        and not path.name.startswith("_")
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


def _discover_dataset_exports() -> dict[str, str]:
    export_to_module: dict[str, str] = {}
    for module_path in _iter_module_paths():
        module_name = module_path.stem
        module_ast = ast.parse(
            module_path.read_text(encoding="utf-8"),
            filename=str(module_path),
        )

        for node in module_ast.body:
            export_name: str | None = None

            if isinstance(node, ast.ClassDef) and _is_export_name(node.name):
                export_name = node.name
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _is_export_name(node.targets[0].id)
                and _is_typeddict_call(node.value)
            ):
                export_name = node.targets[0].id

            if export_name is None:
                continue

            previous_module = export_to_module.get(export_name)
            if previous_module and previous_module != module_name:
                raise RuntimeError(
                    f"Duplicate dataset export '{export_name}' found in "
                    f"'{previous_module}' and '{module_name}'."
                )
            export_to_module[export_name] = module_name

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
