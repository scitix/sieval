"""AI-Generated Code - GPT-5.3-Codex (OpenAI)"""

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "sieval" / "tasks"
DATASETS_DIR = ROOT / "sieval" / "datasets"


def _iter_module_paths(package_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in package_dir.iterdir()
        if path.suffix == ".py"
        and path.name != "__init__.py"
        and not path.name.startswith("_")
    )


def _is_typeddict_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "TypedDict"
    if isinstance(func, ast.Attribute):
        return func.attr == "TypedDict"
    return False


def _iter_subpackage_dirs(package_dir: Path) -> list[Path]:
    """Return sorted subdirectories of *package_dir* that contain ``__init__.py``."""
    return sorted(
        path
        for path in package_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "__init__.py").exists()
    )


def _register_export(
    export_to_module: dict[str, str],
    export_name: str,
    module_name: str,
    kind: str,
) -> None:
    previous_module = export_to_module.get(export_name)
    if previous_module and previous_module != module_name:
        raise RuntimeError(
            f"Duplicate {kind} export '{export_name}' found in "
            f"'{previous_module}' and '{module_name}'."
        )
    export_to_module[export_name] = module_name


def _discover_task_classes(
    module_paths: list[Path],
) -> dict[str, str]:
    """Scan *module_paths* for public ``*Task`` class definitions.

    Returns ``{ClassName: module_stem}`` mapping.
    """
    export_to_module: dict[str, str] = {}
    for module_path in module_paths:
        module_name = module_path.stem
        module_ast = ast.parse(
            module_path.read_text(encoding="utf-8"),
            filename=str(module_path),
        )
        for node in module_ast.body:
            if not isinstance(node, ast.ClassDef):
                continue
            export_name = node.name
            if export_name.startswith("_") or not export_name.endswith("Task"):
                continue
            _register_export(export_to_module, export_name, module_name, "task")
    return export_to_module


def discover_tasks(package_dir: Path) -> dict[str, str]:
    """Discover all task exports from flat modules and subpackages."""
    export_to_module: dict[str, str] = {}

    # 1) Flat .py modules
    for name, mod in _discover_task_classes(_iter_module_paths(package_dir)).items():
        _register_export(export_to_module, name, mod, "task")

    # 2) Subpackage task modules — scan .py files inside each subpackage
    for subpkg_dir in _iter_subpackage_dirs(package_dir):
        subpkg_name = subpkg_dir.name
        for name, _mod in _discover_task_classes(
            _iter_module_paths(subpkg_dir)
        ).items():
            _register_export(export_to_module, name, subpkg_name, "task")

    return export_to_module


def discover_subpackage_tasks(subpkg_dir: Path) -> dict[str, str]:
    """Discover task exports within a single subpackage directory.

    Returns ``{ClassName: module_stem}`` mapping (module-level, not
    prefixed with the subpackage name).
    """
    return _discover_task_classes(_iter_module_paths(subpkg_dir))


def discover_datasets(package_dir: Path) -> dict[str, str]:
    export_to_module: dict[str, str] = {}
    suffixes = ("Dataset", "DatasetSample", "CSVSample")

    for module_path in _iter_module_paths(package_dir):
        module_name = module_path.stem
        module_ast = ast.parse(
            module_path.read_text(encoding="utf-8"),
            filename=str(module_path),
        )

        for node in module_ast.body:
            export_name: str | None = None

            if (
                isinstance(node, ast.ClassDef)
                and not node.name.startswith("_")
                and node.name.endswith(suffixes)
            ):
                export_name = node.name
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and not node.targets[0].id.startswith("_")
                and node.targets[0].id.endswith(suffixes)
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


def _isort_key(name: str) -> list:
    """Natural-sort key matching ruff's isort member ordering."""
    parts = re.split(r"(\d+)", name.casefold())
    return [int(p) if p.isdigit() else p for p in parts]


def render_stub(export_to_module: dict[str, str]) -> str:
    lines: list[str] = [
        "# This file is auto-generated by scripts/sync_package_stubs.py",
        "# Do not edit manually.",
        "",
    ]

    names_by_module: dict[str, list[str]] = {}
    for export_name, module_name in export_to_module.items():
        names_by_module.setdefault(module_name, []).append(export_name)

    for module_name in sorted(names_by_module):
        lines.append(f"from .{module_name} import (")
        for export_name in sorted(names_by_module[module_name], key=_isort_key):
            lines.append(f"    {export_name},")
        lines.append(")")

    lines.extend(
        [
            "",
            "__all__ = [",
            *[f'    "{name}",' for name in sorted(export_to_module)],
            "]",
            "",
        ]
    )
    return "\n".join(lines)


def sync_stub(stub_path: Path, rendered: str, check: bool) -> bool:
    current = stub_path.read_text(encoding="utf-8") if stub_path.exists() else ""

    if current == rendered:
        return True

    if check:
        return False

    stub_path.write_text(rendered, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate __init__.pyi stubs for lazy-export packages."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether generated stubs are up-to-date without writing files.",
    )
    args = parser.parse_args()

    ok = True

    # Top-level packages
    ok &= sync_stub(
        TASKS_DIR / "__init__.pyi",
        render_stub(discover_tasks(TASKS_DIR)),
        check=args.check,
    )
    ok &= sync_stub(
        DATASETS_DIR / "__init__.pyi",
        render_stub(discover_datasets(DATASETS_DIR)),
        check=args.check,
    )

    # Task subpackages
    for subpkg_dir in _iter_subpackage_dirs(TASKS_DIR):
        ok &= sync_stub(
            subpkg_dir / "__init__.pyi",
            render_stub(discover_subpackage_tasks(subpkg_dir)),
            check=args.check,
        )

    if ok:
        return 0

    raise SystemExit(
        "One or more __init__.pyi stubs are out of date. "
        "Run `python scripts/sync_package_stubs.py` to regenerate."
    )


if __name__ == "__main__":
    raise SystemExit(main())
